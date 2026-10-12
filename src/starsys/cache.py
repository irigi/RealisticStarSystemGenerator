"""Convert SysSim CSV catalogs into compact parquet files under ``data/syssim/<model>/``.

Outputs per model:

- ``planets.parquet``: every physical planet, with ``catalog_id``; masses/radii in Earth units.
- ``stars.parquet``: every simulated target incl. zero-planet stars (if published).
- ``params.parquet``: one row of model parameters per catalog (from the CSV headers).

Usage: ``python -m starsys.cache hmc h20 hfr2020_linear_fswp``
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import DATA_DIR
from .syssim_catalogs import (
    MODELS,
    available_catalog_ids,
    read_physical_catalog,
    read_physical_stars,
)


def model_cache_dir(name: str):
    return DATA_DIR / "syssim" / name


def _load_one(args):
    name, cid = args
    model = MODELS[name]
    params, planets = read_physical_catalog(model, cid)
    stars = read_physical_stars(model, cid)
    planets.insert(0, "catalog_id", cid)
    if stars is not None:
        stars.insert(0, "catalog_id", cid)
    return cid, params, planets, stars


def _flatten_params(params: dict) -> dict:
    return {k: (repr(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()}


def build(name: str, workers: int = 8) -> None:
    model = MODELS[name]
    out = model_cache_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    ids = available_catalog_ids(model)
    writers: dict[str, pq.ParquetWriter] = {}
    rows = []
    try:
        with ProcessPoolExecutor(workers) as pool:
            for cid, params, planets, stars in pool.map(_load_one, [(name, i) for i in ids]):
                rows.append({"catalog_id": cid, **_flatten_params(params)})
                for key, df in (("planets", planets), ("stars", stars)):
                    if df is None:
                        continue
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if key not in writers:
                        writers[key] = pq.ParquetWriter(out / f"{key}.parquet", table.schema, compression="zstd")
                    writers[key].write_table(table)
                print(f"{name}: catalog {cid} ({len(planets)} planets)", flush=True)
    finally:
        for w in writers.values():
            w.close()
    pd.DataFrame(rows).sort_values("catalog_id").to_parquet(out / "params.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=sorted(MODELS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name in args.models:
        build(name, args.workers)


if __name__ == "__main__":
    main()
