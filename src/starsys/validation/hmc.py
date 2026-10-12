"""Validate the HM-C port against the published HM-C physical catalogs.

For each catalog: rerun the port with that catalog's header parameters on the
exact same simulated stars, then compare distributions with two-sample KS
statistics. The noise floor is the KS statistic between two independent runs of
the port (same parameters, same stars). A faithful port gives
KS(port, published) ~ KS(port, port).

Usage: ``python -m starsys.validation.hmc --catalogs 1-10``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..inner.hmc import HMCParams, generate_catalog
from ..syssim_catalogs import MODELS, read_physical_catalog, read_physical_stars, read_stellar_table


def _parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def _stats(counts: np.ndarray, planets: pd.DataFrame) -> dict[str, np.ndarray]:
    """Samples whose distributions are compared."""
    pl = planets.sort_values(["target", "period"])
    same = pl["target"].to_numpy()[1:] == pl["target"].to_numpy()[:-1]
    per = pl["period"].to_numpy()
    rad = pl["planet_radius"].to_numpy()
    multi = pl.groupby("target")["period"].transform("size").to_numpy() > 1
    return {
        "n_planets|host": counts[counts > 0].astype(float),
        "log10 period": np.log10(per),
        "radius": rad,
        "log10 mass": np.log10(pl["planet_mass"].to_numpy()),
        "period ratio": (per[1:] / per[:-1])[same],
        "radius ratio": (rad[1:] / rad[:-1])[same],
        "ecc (multis)": pl["ecc"].to_numpy()[multi],
        "ecc (singles)": pl["ecc"].to_numpy()[~multi],
        "incl_invariable (multis)": pl["incl_invariable"].to_numpy()[multi],
    }


def _run_port(params: HMCParams, stars: pd.DataFrame, seed: int, criterion: str):
    counts, pl = generate_catalog(params, stars["mass"], stars["radius"], stars["teff"], stars["color"],
                                  seed=seed, amd_pair_criterion=criterion)
    df = pd.DataFrame(pl)
    df["target"] = df.pop("star_index")
    return counts, df


def validate_catalog(cid: int, stellar: pd.DataFrame, criterion: str = "hill") -> pd.DataFrame:
    model = MODELS["hmc"]
    header, pub = read_physical_catalog(model, cid)
    targets = read_physical_stars(model, cid)
    params = HMCParams.from_header(header)
    stars = targets[["target_id", "star_id", "num_planets"]].merge(stellar, on="star_id", how="left")
    if not np.allclose(stars["mass"], targets["star_mass"], atol=1e-3):
        raise AssertionError("stellar table does not match the catalog's star masses")

    pub = pub.rename(columns={"target_id": "target"})
    pub["target"] -= 1  # 0-based to match the port
    pub_counts = targets["num_planets"].to_numpy()

    a_counts, a = _run_port(params, stars, seed=10_000 + cid, criterion=criterion)
    b_counts, b = _run_port(params, stars, seed=20_000 + cid, criterion=criterion)
    s_pub, s_a, s_b = _stats(pub_counts, pub), _stats(a_counts, a), _stats(b_counts, b)

    rows = [{"catalog": cid, "statistic": "f_hosts", "published": (pub_counts > 0).mean(),
             "port": (a_counts > 0).mean(), "port2": (b_counts > 0).mean()}]
    for key in s_pub:
        rows.append({
            "catalog": cid, "statistic": key,
            "published": np.median(s_pub[key]), "port": np.median(s_a[key]), "port2": np.median(s_b[key]),
            "KS_port_vs_pub": ks_2samp(s_a[key], s_pub[key]).statistic,
            "KS_port_vs_port": ks_2samp(s_a[key], s_b[key]).statistic,
        })
    return pd.DataFrame(rows)


def load_hmc_stellar_table() -> pd.DataFrame:
    stellar = read_stellar_table(MODELS["hmc"].stellar_table)
    stellar["color"] = stellar["bp_rp"] - stellar["e_bp_min_rp_interp"]
    return stellar[["star_id", "mass", "radius", "teff", "color"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogs", default="1-5")
    parser.add_argument("--criterion", default="hill", choices=["hill", "mmr"])
    args = parser.parse_args()
    stellar = load_hmc_stellar_table()
    res = pd.concat([validate_catalog(c, stellar, args.criterion) for c in _parse_ids(args.catalogs)])
    pd.set_option("display.width", 200)
    summary = res.groupby("statistic", sort=False).agg(
        published=("published", "mean"), port=("port", "mean"),
        KS_port_vs_pub=("KS_port_vs_pub", "mean"), KS_port_vs_port=("KS_port_vs_port", "mean"))
    summary["ratio"] = summary["KS_port_vs_pub"] / summary["KS_port_vs_port"]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
