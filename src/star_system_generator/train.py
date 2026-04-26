"""Train the baseline hierarchical system generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean

from .catalog import load_encoded_systems, parse_catalog_ids
from .constants import DEFAULT_CATALOG_ROOT, DEFAULT_MAX_PLANETS
from .model import HierarchicalSystemGenerator
from .torch_utils import require_torch
from .training_data import CatalogIterableDataset, Collator, EncodedSystemDataset

torch = require_torch()
from torch.utils.data import DataLoader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--train-catalogs", default="1-70")
    parser.add_argument("--val-catalogs", default="71-85")
    parser.add_argument("--output-dir", default="runs/baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-planets", type=int, default=DEFAULT_MAX_PLANETS)
    parser.add_argument("--max-systems-per-catalog", type=int, default=None)
    parser.add_argument("--preload", action="store_true", help="Preload datasets into memory; useful only for small smoke tests.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ids = parse_catalog_ids(args.train_catalogs)
    val_ids = parse_catalog_ids(args.val_catalogs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collator = Collator(max_planets=args.max_planets)
    if args.preload:
        print(f"Preloading train catalogs: {train_ids}")
        train_dataset = EncodedSystemDataset(
            load_encoded_systems(
                args.catalog_root,
                train_ids,
                max_planets=args.max_planets,
                max_systems_per_catalog=args.max_systems_per_catalog,
            )
        )
        print(f"Preloading val catalogs: {val_ids}")
        val_dataset = EncodedSystemDataset(
            load_encoded_systems(
                args.catalog_root,
                val_ids,
                max_planets=args.max_planets,
                max_systems_per_catalog=args.max_systems_per_catalog,
            )
        )
        train_shuffle = True
    else:
        print(f"Streaming train catalogs: {train_ids}")
        train_dataset = CatalogIterableDataset(
            catalog_root=args.catalog_root,
            catalog_ids=train_ids,
            max_planets=args.max_planets,
            max_systems_per_catalog=args.max_systems_per_catalog,
            shuffle_catalogs=True,
        )
        print(f"Streaming val catalogs: {val_ids}")
        val_dataset = CatalogIterableDataset(
            catalog_root=args.catalog_root,
            catalog_ids=val_ids,
            max_planets=args.max_planets,
            max_systems_per_catalog=args.max_systems_per_catalog,
            shuffle_catalogs=False,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    model = HierarchicalSystemGenerator(
        hidden_dim=args.hidden_dim,
        max_planets=args.max_planets,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    config = vars(args).copy()
    config["train_catalog_ids"] = train_ids
    config["val_catalog_ids"] = val_ids
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=args.device)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=args.device)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_host={val_metrics['host_loss']:.4f} "
            f"val_count={val_metrics['count_loss']:.4f} "
            f"val_planet={val_metrics['planet_loss']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                },
                output_dir / "model.pt",
            )


def run_epoch(model, loader, *, optimizer, device: str) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_rows: list[dict[str, float]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(batch)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _, metrics = model.loss(batch)
        metric_rows.append(metrics)
    return {key: mean(row[key] for row in metric_rows) for key in metric_rows[0]}


def move_batch(batch, device: str):
    return type(batch)(
        star=batch.star.to(device),
        host=batch.host.to(device),
        count=batch.count.to(device),
        planets=batch.planets.to(device),
        mask=batch.mask.to(device),
    )


if __name__ == "__main__":
    main()
