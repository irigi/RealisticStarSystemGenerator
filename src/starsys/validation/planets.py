"""Validate the planet-attachment glue (:mod:`starsys.planets`) on Kepler-like stars.

Default check: the Kepler stars of each HM-C catalog are passed through
:func:`starsys.planets.attach_planets` as single main-sequence stars, with colour
derived from Teff (instead of the catalog's measured bp-rp) and the catalog's header
parameters. The resulting planets are compared with the published catalog by
two-sample KS statistics. The noise floor is KS between two runs of the bare port on
the catalog colours (as in :mod:`starsys.validation.hmc`). Ratios near 1 mean the glue
and the colour mapping reproduce the published model.

``--colour`` prints the Teff-binned offset between Kepler de-reddened bp-rp and the
Mamajek dwarf sequence (the table in :mod:`starsys.stars.spectral`).

Usage: ``python -m starsys.validation.planets --catalogs 1-10``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..inner.hmc import HMCParams
from ..planets import attach_planets
from ..stars.spectral import bp_rp_from_teff
from ..syssim_catalogs import MODELS, read_physical_catalog, read_physical_stars
from .hmc import _parse_ids, _run_port, _stats, load_hmc_stellar_table


def colour_offsets(stellar: pd.DataFrame, width: float = 200.0) -> pd.DataFrame:
    resid = stellar["color"] - bp_rp_from_teff(stellar["teff"], kepler_scale=False)
    bins = np.arange(3900.0, 7100.0 + width, width)
    cut = pd.cut(stellar["teff"], bins)
    out = resid.groupby(cut, observed=True).agg(["size", "median"])
    out["sigma_mad"] = resid.groupby(cut, observed=True).apply(lambda r: 1.4826 * (r - r.median()).abs().median())
    return out


def _run_glue(params: HMCParams, stars: pd.DataFrame, seed: int):
    n = len(stars)
    components = {
        "system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, "star", dtype=object),
        "phase": np.zeros(n), "mass": stars["mass"].to_numpy(), "radius": stars["radius"].to_numpy(),
        "teff": stars["teff"].to_numpy(), "age_gyr": np.full(n, params.system_age),
    }
    systems = {"age_gyr": np.full(n, params.system_age), "n_components": np.ones(n, int)}
    pl = attach_planets(systems, components, seed=seed, params=params)
    df = pd.DataFrame({k: v for k, v in pl.items() if k not in ("system", "role", "model")})
    df["target"] = df.pop("component")
    return np.bincount(df["target"], minlength=n), df


def validate_catalog(cid: int, stellar: pd.DataFrame) -> pd.DataFrame:
    model = MODELS["hmc"]
    header, pub = read_physical_catalog(model, cid)
    targets = read_physical_stars(model, cid)
    params = HMCParams.from_header(header)
    stars = targets[["target_id", "star_id"]].merge(stellar, on="star_id", how="left")
    pub = pub.rename(columns={"target_id": "target"})
    pub["target"] -= 1
    pub_counts = targets["num_planets"].to_numpy()

    g_counts, g = _run_glue(params, stars, seed=30_000 + cid)
    a_counts, a = _run_port(params, stars, seed=10_000 + cid, criterion="hill")
    b_counts, b = _run_port(params, stars, seed=20_000 + cid, criterion="hill")
    s_pub, s_g, s_a, s_b = (_stats(c, d) for c, d in
                            ((pub_counts, pub), (g_counts, g), (a_counts, a), (b_counts, b)))
    rows = [{"catalog": cid, "statistic": "f_hosts", "published": (pub_counts > 0).mean(),
             "glue": (g_counts > 0).mean()}]
    for key in s_pub:
        rows.append({
            "catalog": cid, "statistic": key,
            "published": np.median(s_pub[key]), "glue": np.median(s_g[key]),
            "KS_glue_vs_pub": ks_2samp(s_g[key], s_pub[key]).statistic,
            "KS_port_vs_port": ks_2samp(s_a[key], s_b[key]).statistic,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogs", default="1-5")
    parser.add_argument("--colour", action="store_true", help="print the Kepler colour-offset table and exit")
    args = parser.parse_args()
    stellar = load_hmc_stellar_table()
    pd.set_option("display.width", 200)
    if args.colour:
        print(colour_offsets(stellar).round(3).to_string())
        return
    res = pd.concat([validate_catalog(c, stellar) for c in _parse_ids(args.catalogs)])
    summary = res.groupby("statistic", sort=False).agg(
        published=("published", "mean"), glue=("glue", "mean"),
        KS_glue_vs_pub=("KS_glue_vs_pub", "mean"), KS_port_vs_port=("KS_port_vs_port", "mean"))
    summary["ratio"] = summary["KS_glue_vs_pub"] / summary["KS_port_vs_port"]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
