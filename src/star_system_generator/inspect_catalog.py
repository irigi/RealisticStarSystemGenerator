"""CLI for inspecting one zipped SysSim physical catalog realization."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .catalog import list_catalog_ids, read_physical_catalog, summarize_systems
from .constants import DEFAULT_CATALOG_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--catalog-id", type=int, default=1)
    parser.add_argument("--max-systems", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.catalog_root)
    ids = list_catalog_ids(root)
    print(f"catalog_root={root}")
    print(f"available_catalogs={len(ids)} first={ids[:5]} last={ids[-5:]}")

    metadata, systems = read_physical_catalog(root, args.catalog_id, max_systems=args.max_systems)
    print(f"catalog_id={args.catalog_id}")
    print("metadata_sample:")
    for key in sorted(metadata)[:12]:
        print(f"  {key}: {metadata[key]}")

    summary = summarize_systems(systems)
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    multiplicities = Counter(system.num_planets for system in systems)
    print("multiplicity_counts:")
    for count in sorted(multiplicities):
        print(f"  {count}: {multiplicities[count]}")


if __name__ == "__main__":
    main()

