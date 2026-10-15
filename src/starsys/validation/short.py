"""Planets inside 3 d: checks of :mod:`starsys.inner.short`.

Model planets come from the stellar population through :func:`starsys.planets.attach_planets`
(all models), for a mock magnitude-limited Kepler sample of primaries (binaries included;
:func:`starsys.binaries.mock_kepler`) with the Teff distribution reweighted to HM-C's Kepler
targets. Rates are intrinsic.

- **Hsu+2019** (same Kepler/Gaia FGK sample as HM-C): planets per star in their period bins
  0.5-1, 1-2, 2-4 and 4-8 d, by radius group. Below 3 d the short-period component is calibrated
  to these numbers; 2-4 and 4-8 d test the junction with HM-C.
- **Sanchis-Ojeda+2014:** USPs (their P = 4.2-24 h, R = 0.84-4 Earth radii) around 0.15 +- 0.05%
  of F, 0.51 +- 0.07% of G, 0.83 +- 0.18% of K and 1.10 +- 0.40% of M dwarfs; 5.5 +- 0.5 per
  thousand overall. The model starts at 0.5 d, so it is expected to sit somewhat below (their
  occurrence rises with period inside 1 d).

Usage: ``python -m starsys.validation.short [--systems 600000]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import binaries as bn
from .. import planets
from ..inner import short as sh
from ..stars.population import draw_systems
from .hmc import load_hmc_stellar_table

R_GROUPS = {"0.5-1 Re": (0.5, 1.0), "1-2 Re": (1.0, 2.0), "2-4 Re": (2.0, 4.0), "4-16 Re": (4.0, 16.0)}
P_BINS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))
HSU_4_8 = np.array([9.2e-2, 3.18e-2, 2.40e-2, 1.91e-2, 1.40e-2, 6.9e-3, 1.03e-2, 1.10e-2, 3.2e-3, 1.54e-3, 6.1e-4,
                    1.17e-3, 1.65e-3])
SO2014 = {"F (6100-7100 K)": (6100, 7100, "0.15 +- 0.05"), "G (5100-6100 K)": (5100, 6100, "0.51 +- 0.07"),
          "K (4100-5100 K)": (4100, 5100, "0.83 +- 0.18"), "M (3100-4100 K)": (3100, 4100, "1.10 +- 0.40")}


def kepler_weights(systems, comp):
    """Mock-Kepler primaries and weights, Teff reweighted to HM-C's Kepler targets."""
    prim, w, _ = bn.mock_kepler(systems, comp, planets.DEFAULT_BINARIES)
    kep = load_hmc_stellar_table()["teff"].to_numpy()
    teff = np.asarray(comp["teff"])[prim]
    edges = np.arange(np.floor(kep.min() / 100) * 100, kep.max() + 100, 100.0)
    h = np.histogram(kep, edges)[0] / len(kep)
    b = np.digitize(teff, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    prim, w, b = prim[ok], w[ok], b[ok]
    wb = np.bincount(b, weights=w, minlength=len(h))
    return prim, w * h[b] / np.maximum(wb[b], 1e-30)


def _hsu(p_bin, r_lo, r_hi):
    edges = sh.HSU2019_R_EDGES
    rates = sh.HSU2019_RATES[p_bin] if p_bin < 3 else HSU_4_8
    sel = (edges[:-1] >= r_lo) & (edges[1:] <= r_hi)
    return rates[sel].sum()


def hsu_comparison(systems, comp, pl) -> pd.DataFrame:
    prim, w = kepler_weights(systems, comp)
    wt = np.zeros(len(comp["mass"]))
    wt[prim] = w
    s = (pl["host_kind"] == "S") & np.isin(pl["model"], ("hmc", "hmc_m", "short_period"))
    pw = wt[pl["component"][s]]
    rows = []
    for j, (lo, hi) in enumerate(P_BINS):
        for name, (r0, r1) in R_GROUPS.items():
            x = (pl["period"][s] >= lo) & (pl["period"][s] < hi) & (pl["planet_radius"][s] >= r0) & (
                pl["planet_radius"][s] < r1)
            short = x & (pl["model"][s] == "short_period")
            rows.append({"P (d)": f"{lo:g}-{hi:g}", "R": name, "model": pw[x].sum() / w.sum(),
                         "of which short_period": pw[short].sum() / w.sum(), "Hsu+2019": _hsu(j, r0, r1)})
    out = pd.DataFrame(rows).set_index(["P (d)", "R"])
    out["ratio"] = out["model"] / out["Hsu+2019"].replace(0.0, np.nan)
    return out


def usp_by_type(comp, pl) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    ms = (comp["kind"] == "star") & np.isin(comp["phase"], (-1, 0))
    x = (pl["host_kind"] == "S") & (pl["period"] < 1.0) & (pl["planet_radius"] >= 0.84) & (pl["planet_radius"] < 4.0)
    has = np.bincount(pl["component"][x], minlength=len(teff)) > 0
    rows = []
    for name, (lo, hi, obs) in SO2014.items():
        sel = ms & (teff >= lo) & (teff < hi)
        rows.append({"hosts": name, "stars": int(sel.sum()), "model %": 100 * has[sel].mean(),
                     "Sanchis-Ojeda+2014 %": obs})
    # Kepler's M dwarfs are mostly early M (Dressing & Charbonneau 2015: median 0.5 Msun).
    sel = ms & (teff >= 3100) & (teff < 4100) & (np.asarray(comp["mass"]) >= 0.35)
    rows.append({"hosts": "M, early (>= 0.35 Msun, Kepler-like)", "stars": int(sel.sum()),
                 "model %": 100 * has[sel].mean(), "Sanchis-Ojeda+2014 %": SO2014["M (3100-4100 K)"][2]})
    return pd.DataFrame(rows).set_index("hosts")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    systems, comp = draw_systems(args.systems, seed=args.seed)
    pl = planets.attach_planets(systems, comp, seed=args.seed + 1, giants=None, small=None)
    print("== Planets per star, mock Kepler sample vs Hsu+2019 ==")
    print(hsu_comparison(systems, comp, pl))
    print("\n== USPs (P < 1 d, 0.84-4 Re): % of main-sequence stars ==")
    print(usp_by_type(comp, pl))


if __name__ == "__main__":
    main()
