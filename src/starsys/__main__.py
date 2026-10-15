"""``python -m starsys``: random star systems as readable summaries, JSON or tables.

Examples::

    python -m starsys -n 5 --seed 1                      # five systems, printed
    python -m starsys -n 2000 --spectral G --habitable --show 3
    python -m starsys --id S1-42                          # one system by its stable ID
    python -m starsys -n 1000 --seed 2 --json out/systems.json --tables out/tables --format parquet
"""

from __future__ import annotations

import argparse

import numpy as np

from . import describe as ds
from . import export as ex
from .generate import generate


def select(bundle, spectral: str | None = None, habitable: bool = False, min_planets: int = 0,
           single: bool = False) -> np.ndarray:
    """Indices of the systems whose primary has spectral class ``spectral`` (letters, e.g. "GK"),
    that have a rocky planet in the habitable zone (``habitable``), at least ``min_planets``
    planets, or a single star (``single``)."""
    c, p = bundle.components, bundle.planets
    n = len(bundle)
    ok = np.ones(n, dtype=bool)
    primary = np.searchsorted(np.asarray(c["system"]), np.arange(n))
    if spectral:
        sp = np.array([str(t)[:1] for t in np.asarray(c["spectral_type"])[primary]])
        kind = np.asarray(c["kind"])[primary]
        ok &= np.isin(sp, list(spectral.upper())) & (kind == "star")
    if single:
        ok &= np.asarray(bundle.systems["n_components"]) == 1
    n_pl = np.bincount(np.asarray(p["system"]), minlength=n)
    ok &= n_pl >= min_planets
    if habitable and len(p["a_au"]):
        _, climate = ds.surface_temperature(p)
        hab = (climate == ds.CLIMATE_TEMPERATE) | (climate == ds.CLIMATE_MARGINAL)
        ok &= np.bincount(np.asarray(p["system"])[hab], minlength=n) > 0
    return np.flatnonzero(ok)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m starsys", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", nargs="+", help="print the systems with these stable IDs (e.g. S1-42, see starsys.ids)")
    ap.add_argument("-n", type=int, default=5, help="systems to generate (default 5)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spectral", help="keep systems whose primary is of these classes, e.g. G or FGK")
    ap.add_argument("--habitable", action="store_true", help="keep systems with a rocky habitable-zone planet")
    ap.add_argument("--single", action="store_true", help="keep single-star systems")
    ap.add_argument("--min-planets", type=int, default=0)
    ap.add_argument("--show", type=int, default=None, help="print at most this many systems (default: all kept, "
                                                         "or none when writing files)")
    ap.add_argument("--no-moons", action="store_true", help="omit moons from the printed summaries")
    ap.add_argument("--json", help="write the kept systems to this JSON file")
    ap.add_argument("--tables", help="write all generated systems as tables into this directory")
    ap.add_argument("--format", choices=("csv", "parquet"), default="csv", help="table format (default csv)")
    ap.add_argument("--bare-fraction", type=float, default=None,
                    help="setting temperate_bare_fraction (0-1): temperate small planets born rocky")
    args = ap.parse_args(argv)

    kwargs = {} if args.bare_fraction is None else {"temperate_bare_fraction": args.bare_fraction}
    if args.id:
        from . import ids
        for sid in args.id:
            print(ds.system_text(ids.system(sid, planet_kwargs=kwargs), 0, moons=not args.no_moons))
            print()
        return
    bundle = generate(args.n, seed=args.seed, planet_kwargs=kwargs)
    keep = select(bundle, args.spectral, args.habitable, args.min_planets, args.single)
    writing = bool(args.json or args.tables)
    show = args.show if args.show is not None else (0 if writing else len(keep))
    for i in keep[:show]:
        print(ds.system_text(bundle, int(i), moons=not args.no_moons))
        print()
    if not len(keep):
        print(f"none of the {args.n} systems passes the selection; generate more with -n")
    if args.json:
        print(f"wrote {len(keep)} systems to {ex.write_json(bundle, args.json, keep)}")
    if args.tables:
        paths = ex.write_tables(bundle, args.tables, args.format)
        print(f"wrote {', '.join(p.name for p in paths)} ({len(bundle)} systems) to {args.tables}")


if __name__ == "__main__":
    main()
