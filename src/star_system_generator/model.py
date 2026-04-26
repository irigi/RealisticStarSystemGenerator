"""Baseline hierarchical conditional autoregressive generator."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import DEFAULT_MAX_PLANETS
from .torch_utils import require_torch

torch = require_torch()
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


@dataclass(frozen=True)
class Batch:
    star: torch.Tensor
    host: torch.Tensor
    count: torch.Tensor
    planets: torch.Tensor
    mask: torch.Tensor


class HierarchicalSystemGenerator(nn.Module):
    """Hurdle + count + ordered sequence decoder.

    Planet emissions are diagonal Gaussian in transformed variables. This is
    intentionally simple; it establishes a likelihood-trained baseline before
    adding mixture-density or flow heads.
    """

    def __init__(
        self,
        *,
        star_dim: int = 2,
        planet_dim: int = 4,
        hidden_dim: int = 128,
        max_planets: int = DEFAULT_MAX_PLANETS,
    ) -> None:
        super().__init__()
        self.planet_dim = planet_dim
        self.hidden_dim = hidden_dim
        self.max_planets = max_planets

        self.star_encoder = nn.Sequential(
            nn.Linear(star_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.host_head = nn.Linear(hidden_dim, 1)
        # Count classes are 1..max_planets for hosted systems. Zero planets are
        # handled only by the host hurdle, avoiding duplicated zero mass.
        self.count_head = nn.Linear(hidden_dim, max_planets)

        self.start_token = nn.Parameter(torch.zeros(planet_dim))
        self.planet_input = nn.Linear(planet_dim, hidden_dim)
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, planet_dim)
        self.log_std_head = nn.Linear(hidden_dim, planet_dim)

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        star_context = self.star_encoder(batch.star)
        host_logits = self.host_head(star_context).squeeze(-1)
        count_logits = self.count_head(star_context)

        teacher = self._teacher_forcing_inputs(batch.planets)
        decoder_input = self.planet_input(teacher)
        hidden0 = star_context.unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, hidden0)
        means = self.mean_head(decoded)
        log_stds = self.log_std_head(decoded).clamp(-6.0, 3.0)

        return {
            "host_logits": host_logits,
            "count_logits": count_logits,
            "means": means,
            "log_stds": log_stds,
        }

    def loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = self.forward(batch)
        host_loss = F.binary_cross_entropy_with_logits(
            outputs["host_logits"],
            batch.host.float(),
        )

        hosted = batch.host.bool()
        if hosted.any():
            count_target = (batch.count[hosted] - 1).clamp_min(0)
            count_loss = F.cross_entropy(outputs["count_logits"][hosted], count_target)
        else:
            count_loss = outputs["count_logits"].sum() * 0.0

        planet_loss = gaussian_nll(
            batch.planets,
            outputs["means"],
            outputs["log_stds"],
            batch.mask,
        )
        total = host_loss + count_loss + planet_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "host_loss": float(host_loss.detach().cpu()),
            "count_loss": float(count_loss.detach().cpu()),
            "planet_loss": float(planet_loss.detach().cpu()),
        }
        return total, metrics

    @torch.no_grad()
    def sample(self, star: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample systems for a batch of star features.

        Returns ``host``, ``count``, and padded planet feature tensors.
        """

        self.eval()
        device = star.device
        batch_size = star.shape[0]
        star_context = self.star_encoder(star)
        host_probs = torch.sigmoid(self.host_head(star_context).squeeze(-1))
        host = torch.bernoulli(host_probs).long()

        count_probs = torch.softmax(self.count_head(star_context), dim=-1)
        count = torch.multinomial(count_probs, num_samples=1).squeeze(-1) + 1
        count = torch.where(host == 1, count, torch.zeros_like(count))

        planets = torch.zeros(batch_size, self.max_planets, self.planet_dim, device=device)
        previous = self.start_token.expand(batch_size, -1)
        hidden = star_context.unsqueeze(0)
        for index in range(self.max_planets):
            decoder_input = self.planet_input(previous).unsqueeze(1)
            decoded, hidden = self.decoder(decoder_input, hidden)
            mean = self.mean_head(decoded[:, 0, :])
            log_std = self.log_std_head(decoded[:, 0, :]).clamp(-6.0, 3.0)
            sample = mean + torch.randn_like(mean) * torch.exp(log_std)
            active = index < count
            planets[active, index, :] = sample[active]
            previous = sample
        return host, count, planets

    def _teacher_forcing_inputs(self, planets: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = planets.shape
        start = self.start_token.view(1, 1, -1).expand(batch_size, 1, -1)
        if steps == 1:
            return start
        return torch.cat([start, planets[:, :-1, :]], dim=1)


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    variance_term = ((target - mean) / torch.exp(log_std)).pow(2)
    nll = 0.5 * variance_term + log_std + 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793, device=target.device))
    masked = nll.sum(dim=-1) * mask.float()
    denominator = mask.float().sum().clamp_min(1.0)
    return masked.sum() / denominator
