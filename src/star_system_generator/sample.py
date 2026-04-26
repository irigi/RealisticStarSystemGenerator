"""Sample generated systems from a trained baseline model."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .catalog import read_physical_catalog
from .constants import DEFAULT_CATALOG_ROOT, SOLAR_MASSES_TO_EARTH_MASSES, SOLAR_RADII_TO_EARTH_RADII
from .model import HierarchicalSystemGenerator
from .torch_utils import require_torch

torch = require_torch()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog-root", default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--catalog-id", type=int, default=86)
    parser.add_argument("--num-stars", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    model = HierarchicalSystemGenerator(
        hidden_dim=config["hidden_dim"],
        max_planets=config["max_planets"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])

    _, systems = read_physical_catalog(args.catalog_root, args.catalog_id, max_systems=args.num_stars)
    star_features = torch.tensor(
        [[math.log(system.star_mass_solar), math.log(system.star_radius_solar)] for system in systems],
        dtype=torch.float32,
        device=args.device,
    )
    host, count, planets = model.sample(star_features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_id",
                "star_id",
                "generated_planet_index",
                "period_days",
                "radius_earth",
                "mass_earth",
                "ecc",
            ],
        )
        writer.writeheader()
        for row, system in enumerate(systems):
            if int(host[row].item()) == 0:
                continue
            log_period = None
            for planet_index in range(int(count[row].item())):
                values = planets[row, planet_index].detach().cpu().tolist()
                if planet_index == 0:
                    log_period = values[0]
                else:
                    log_period = (log_period or 0.0) + max(values[0], 1.0e-8)
                writer.writerow(
                    {
                        "target_id": system.target_id,
                        "star_id": system.star_id,
                        "generated_planet_index": planet_index + 1,
                        "period_days": math.exp(log_period),
                        "radius_earth": math.exp(values[1]),
                        "mass_earth": math.exp(values[2]),
                        "ecc": min(max(values[3], 0.0), 0.999999),
                    }
                )

    print(f"Wrote generated planet rows to {output_path}")
    print(f"unit_constants: R_sun_to_R_earth={SOLAR_RADII_TO_EARTH_RADII}, M_sun_to_M_earth={SOLAR_MASSES_TO_EARTH_MASSES}")


if __name__ == "__main__":
    main()
