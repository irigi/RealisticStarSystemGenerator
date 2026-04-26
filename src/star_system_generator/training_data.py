"""Dataset and batching helpers for PyTorch training."""

from __future__ import annotations

from typing import Iterable, Sequence

from .catalog import EncodedSystem, encode_system, read_physical_catalog
from .constants import DEFAULT_MAX_PLANETS
from .model import Batch
from .torch_utils import require_torch

torch = require_torch()
from torch.utils.data import Dataset, IterableDataset, get_worker_info  # noqa: E402


class EncodedSystemDataset(Dataset):
    def __init__(self, systems: Sequence[EncodedSystem]) -> None:
        self.systems = list(systems)

    def __len__(self) -> int:
        return len(self.systems)

    def __getitem__(self, index: int) -> EncodedSystem:
        return self.systems[index]


class CatalogIterableDataset(IterableDataset):
    """Stream encoded systems from whole catalog realizations.

    This avoids keeping tens of full catalog realizations in memory during
    normal training. For multi-worker loading, catalog ids are sharded by
    worker id.
    """

    def __init__(
        self,
        *,
        catalog_root: str,
        catalog_ids: Sequence[int],
        max_planets: int,
        max_systems_per_catalog: int | None = None,
        shuffle_catalogs: bool = False,
    ) -> None:
        self.catalog_root = catalog_root
        self.catalog_ids = list(catalog_ids)
        self.max_planets = max_planets
        self.max_systems_per_catalog = max_systems_per_catalog
        self.shuffle_catalogs = shuffle_catalogs

    def __iter__(self) -> Iterable[EncodedSystem]:
        ids = list(self.catalog_ids)
        worker = get_worker_info()
        if worker is not None:
            ids = ids[worker.id :: worker.num_workers]
        if self.shuffle_catalogs:
            generator = torch.Generator()
            generator.manual_seed(torch.initial_seed())
            order = torch.randperm(len(ids), generator=generator).tolist()
            ids = [ids[index] for index in order]

        for catalog_id in ids:
            _, systems = read_physical_catalog(
                self.catalog_root,
                catalog_id,
                max_systems=self.max_systems_per_catalog,
            )
            for system in systems:
                yield encode_system(system, max_planets=self.max_planets)


class Collator:
    def __init__(self, *, max_planets: int = DEFAULT_MAX_PLANETS, planet_dim: int = 4) -> None:
        self.max_planets = max_planets
        self.planet_dim = planet_dim

    def __call__(self, systems: Sequence[EncodedSystem]) -> Batch:
        batch_size = len(systems)
        star = torch.zeros(batch_size, 2, dtype=torch.float32)
        host = torch.zeros(batch_size, dtype=torch.long)
        count = torch.zeros(batch_size, dtype=torch.long)
        planets = torch.zeros(batch_size, self.max_planets, self.planet_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, self.max_planets, dtype=torch.bool)

        for row, system in enumerate(systems):
            star[row] = torch.tensor(system.star_features, dtype=torch.float32)
            host[row] = system.host_flag
            n = min(system.multiplicity, self.max_planets)
            count[row] = n
            if n:
                planets[row, :n] = torch.tensor(system.planet_features[:n], dtype=torch.float32)
                mask[row, :n] = True

        return Batch(star=star, host=host, count=count, planets=planets, mask=mask)
