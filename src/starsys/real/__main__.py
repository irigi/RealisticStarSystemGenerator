"""``python -m starsys.real NAME``: the stellar fit and conditioned planetary systems of a real star."""

from __future__ import annotations

import argparse

from . import condition as cd
from . import star as st
from .catalog import find


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="star name (HD, HIP, GJ, TIC, Gaia DR3, Bayer, Flamsteed or proper name)")
    ap.add_argument("-n", type=int, default=100, help="number of systems")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-gaia", action="store_true", help="do not query Gaia DR3 online")
    ap.add_argument("--controversial", action="store_true", help="include planets flagged controversial")
    ap.add_argument("--star-only", action="store_true", help="only fit the star(s)")
    ap.add_argument("--show", type=int, default=0, help="also print this many of the systems in full")
    ap.add_argument("--json", help="write the systems to this JSON file")
    ap.add_argument("--tables", help="write the systems as tables into this directory")
    ap.add_argument("--format", choices=("csv", "parquet"), default="csv", help="table format (default csv)")
    args = ap.parse_args(argv)

    record = find(args.name, use_gaia=not args.no_gaia)
    if args.star_only:
        print(st.fit(record, n=max(args.n, 500), seed=args.seed).summary())
        return
    c = cd.condition(record, n=args.n, seed=args.seed, use_gaia=not args.no_gaia,
                     include_controversial=args.controversial)
    print(c.summary())
    if args.show or args.json or args.tables:
        from .. import describe as ds
        from .. import export as ex
        from ..generate import Systems
        bundle = Systems.from_conditioned(c)
        for i in range(min(args.show, len(bundle))):
            print()
            print(ds.system_text(bundle, i))
        if args.json:
            print(f"wrote {len(bundle)} systems to {ex.write_json(bundle, args.json)}")
        if args.tables:
            paths = ex.write_tables(bundle, args.tables, args.format)
            print(f"wrote {', '.join(p.name for p in paths)} to {args.tables}")


if __name__ == "__main__":
    main()
