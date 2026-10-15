"""Derived planet properties: checks of :mod:`starsys.derived` and of the Kozai-Lidov step.

- **Formulae against the NASA Exoplanet Archive** (composite parameters): insolation from the
  archive's stellar luminosity and semi-major axis against its ``pl_insol``, and the zero-albedo
  equilibrium temperature against ``pl_eqt`` (papers mostly quote A = 0 and full redistribution).
- **Known planets:** the Solar System, Proxima Cen b and TRAPPIST-1 e, f, g: insolation,
  T_eq, habitable-zone code and spin state.
- **Population** (all models): habitable-zone planets per main-sequence star by type and size,
  the fraction of HZ planets that are tidally locked, and the Kozai-Lidov census (share of
  planets cycling, their eccentricities, planets removed).
- **Kozai-Lidov vs observed eccentricities:** giant planets (M sin i >= 0.3 MJup, 0.1-5 au)
  around stars with and without a stellar companion in the NASA archive, against the model's
  Sun-like hosts with and without a companion within 2000 au.

Usage: ``python -m starsys.validation.derived [--systems 200000]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import derived as dv
from .. import planets
from ..paths import CATALOGS_DIR
from ..stars.population import draw_systems

ARCHIVE = CATALOGS_DIR / "observational" / "nasa_pscomppars.csv"
HZ_NAMES = {dv.HZ_TOO_HOT: "too hot", dv.HZ_OPT_INNER: "optimistic inner", dv.HZ_CONSERVATIVE: "conservative",
            dv.HZ_OPT_OUTER: "optimistic outer", dv.HZ_TOO_COLD: "too cold", dv.HZ_UNDEFINED: "undefined"}
SPIN_NAMES = {dv.SPIN_FREE: "free", dv.SPIN_SYNC: "synchronous", dv.SPIN_32: "3:2"}


def archive_check() -> pd.DataFrame:
    t = pd.read_csv(ARCHIVE, comment="#", usecols=["pl_name", "pl_orbsmax", "pl_orbeccen", "pl_insol", "pl_eqt",
                                                     "st_lum", "st_teff"])
    rows = []
    s = 10 ** t["st_lum"] / t["pl_orbsmax"] ** 2
    ok = np.isfinite(s) & np.isfinite(t["pl_insol"]) & (t["pl_insol"] > 0)
    r = (s / t["pl_insol"])[ok]
    rows.append({"quantity": "insolation L/a^2 vs pl_insol", "planets": int(ok.sum()), "median ratio": r.median(),
                 "16th pct": r.quantile(0.16), "84th pct": r.quantile(0.84)})
    teq = dv.T_EQ_EARTH_FLUX * s**0.25
    ok = np.isfinite(teq) & np.isfinite(t["pl_eqt"]) & (t["pl_eqt"] > 0)
    r = (teq / t["pl_eqt"])[ok]
    rows.append({"quantity": "T_eq (A = 0) vs pl_eqt", "planets": int(ok.sum()), "median ratio": r.median(),
                 "16th pct": r.quantile(0.16), "84th pct": r.quantile(0.84)})
    return pd.DataFrame(rows).set_index("quantity")


def _one_star(lum, teff, mass, age, a, m, r, e, names):
    n = len(a)
    comp = {"system": np.zeros(1, int), "role": np.zeros(1, int), "luminosity": np.array([lum]),
            "teff": np.array([teff]), "mass": np.array([mass]), "age_gyr": np.array([age]),
            "a_au": np.array([np.nan]), "ecc": np.array([np.nan])}
    pl = {"system": np.zeros(n, int), "component": np.zeros(n, int), "host_kind": np.full(n, "S", dtype=object),
          "host_mass": np.full(n, mass), "a_au": np.array(a, float), "ecc": np.array(e, float),
          "planet_mass": np.array(m, float), "planet_radius": np.array(r, float), "lost_envelope": np.zeros(n, bool)}
    out = dv.derive_properties(pl, {"n_components": np.array([1])}, comp)
    return pd.DataFrame({"planet": names, "insolation": out["insolation"], "T_eq (K)": out["t_eq"],
                         "HZ": [HZ_NAMES[x] for x in out["hz"]], "t_lock (Gyr)": out["t_lock_gyr"],
                         "spin": [SPIN_NAMES[x] for x in out["spin"]],
                         "HZ extrapolated": out["hz_flags"] > 0})


def known_planets() -> pd.DataFrame:
    sol = _one_star(1.0, 5772.0, 1.0, 4.6, [0.387, 0.723, 1.0, 1.524, 5.203, 9.537],
                    [0.055, 0.815, 1.0, 0.107, 317.8, 95.2], [0.383, 0.949, 1.0, 0.532, 11.21, 9.45],
                    [0.206, 0.007, 0.017, 0.093, 0.049, 0.057], ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"])
    prox = _one_star(0.00155, 3042.0, 0.122, 4.8, [0.0485], [1.07], [1.05], [0.02], ["Proxima Cen b"])
    trap = _one_star(0.000553, 2566.0, 0.0898, 7.6, [0.02925, 0.03849, 0.04683], [0.69, 1.04, 1.32],
                     [0.92, 1.05, 1.13], [0.005, 0.01, 0.002], ["TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g"])
    return pd.concat([sol, prox, trap]).set_index("planet")


def _ms_hosts(comp):
    return planets.hmc_hosts(comp) & (np.asarray(comp["phase"]) == 0)


TYPES = {"M (< 3900 K)": (2300.0, 3900.0), "K (3900-5300 K)": (3900.0, 5300.0), "G (5300-6000 K)": (5300.0, 6000.0),
         "F (6000-7200 K)": (6000.0, 7200.0)}


def hz_census(comp, pl) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    ms = _ms_hosts(comp)
    c = pl["component"]
    s_type = pl["host_kind"] == "S"
    r = pl["planet_radius"]
    rows = []
    for name, (lo, hi) in TYPES.items():
        hosts = ms & (teff >= lo) & (teff < hi)
        sel = s_type & hosts[c]
        cons = sel & (pl["hz"] == dv.HZ_CONSERVATIVE)
        opt = sel & np.isin(pl["hz"], (dv.HZ_OPT_INNER, dv.HZ_CONSERVATIVE, dv.HZ_OPT_OUTER))
        n = hosts.sum()
        rows.append({"hosts": name, "stars": int(n),
                     "cons. HZ, 0.5-1.5 Re": (cons & (r >= 0.5) & (r <= 1.5)).sum() / n,
                     "cons. HZ, 1.5-2.5 Re": (cons & (r > 1.5) & (r <= 2.5)).sum() / n,
                     "cons. HZ, all sizes": cons.sum() / n,
                     "opt. HZ, 0.5-1.5 Re": (opt & (r >= 0.5) & (r <= 1.5)).sum() / n,
                     "% of cons. HZ planets locked": 100 * np.mean(pl["spin"][cons] > 0) if cons.any() else np.nan,
                     "% of cons. HZ planets with T_eq 200-300 K": 100 * np.mean((pl["t_eq"][cons] > 200)
                                                                                & (pl["t_eq"][cons] < 300))
                     if cons.any() else np.nan})
    return pd.DataFrame(rows).set_index("hosts")


def locking(comp, pl) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    ms = _ms_hosts(comp)
    c = pl["component"]
    rows = []
    for name, (lo, hi) in TYPES.items():
        sel = (pl["host_kind"] == "S") & (ms & (teff >= lo) & (teff < hi))[c]
        for p_lo, p_hi in ((0.0, 10.0), (10.0, 100.0), (100.0, 1000.0), (1000.0, np.inf)):
            m = sel & (pl["period"] >= p_lo) & (pl["period"] < p_hi)
            if m.sum() < 50:
                continue
            rows.append({"hosts": name, "P (d)": f"{p_lo:g}-{p_hi:g}", "planets": int(m.sum()),
                         "% synchronous": 100 * np.mean(pl["spin"][m] == dv.SPIN_SYNC),
                         "% 3:2": 100 * np.mean(pl["spin"][m] == dv.SPIN_32)})
    return pd.DataFrame(rows).set_index(["hosts", "P (d)"])


def kozai_census(comp, pl, n_without) -> pd.DataFrame:
    kz = (pl["flags"] & planets.FLAG_KOZAI) > 0
    s_type = pl["host_kind"] == "S"
    giant = pl["planet_mass"] >= 95.0
    rows = [{"quantity": "share of S-type planets cycling", "value": kz[s_type].mean()},
            {"quantity": "share of giants (>= 0.3 MJup) cycling", "value": kz[s_type & giant].mean()},
            {"quantity": "mean e, cycling planets", "value": pl["ecc"][kz].mean()},
            {"quantity": "share of cycling planets with e > 0.5", "value": np.mean(pl["ecc"][kz] > 0.5)},
            {"quantity": "planets removed by the cycles (per 1000 planets)",
             "value": 1000 * (n_without - len(pl["period"])) / max(n_without, 1)}]
    return pd.DataFrame(rows).set_index("quantity")


def ecc_in_binaries(comp, pl) -> pd.DataFrame:
    """Giant eccentricities with and without a stellar companion: model vs NASA archive."""
    t = pd.read_csv(ARCHIVE, comment="#", usecols=["pl_name", "pl_bmasse", "pl_orbsmax", "pl_orbeccen", "sy_snum",
                                                     "discoverymethod", "st_mass"])
    t = t[(t["discoverymethod"] == "Radial Velocity") & (t["pl_bmasse"] >= 95) & (t["pl_orbsmax"] >= 0.1)
          & (t["pl_orbsmax"] < 5) & np.isfinite(t["pl_orbeccen"]) & (t["st_mass"] > 0.7) & (t["st_mass"] < 1.4)]
    rows = []
    for label, m in (("single", t["sy_snum"] == 1), ("with stellar companion", t["sy_snum"] > 1)):
        e = t["pl_orbeccen"][m]
        rows.append({"sample": f"NASA archive RV, {label}", "planets": int(m.sum()), "mean e": e.mean(),
                     "e > 0.6 (%)": 100 * np.mean(e > 0.6)})
    from .. import binaries as bn

    st = bn.structure(comp)
    c = pl["component"]
    mass = np.asarray(comp["mass"])
    msini = pl["planet_mass"] * np.sin(pl["incl_sky"])
    base = ((pl["host_kind"] == "S") & (msini >= 95) & (pl["a_au"] >= 0.1) & (pl["a_au"] < 5)
            & (mass[c] > 0.7) & (mass[c] < 1.4) & (np.asarray(comp["phase"])[c] == 0))
    a_near = st["a_near"][c]
    for label, m in (("no stellar companion", np.isnan(a_near)),
                     ("stellar companion < 2000 au", a_near < 2000)):
        e = pl["ecc"][base & m]
        rows.append({"sample": f"model, {label}", "planets": int((base & m).sum()), "mean e": e.mean(),
                     "e > 0.6 (%)": 100 * np.mean(e > 0.6)})
    return pd.DataFrame(rows).set_index("sample")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.precision", 3)
    print("== Formulae against the NASA Exoplanet Archive ==")
    print(archive_check())
    print("\n== Known planets ==")
    print(known_planets())
    systems, comp = draw_systems(args.systems, seed=args.seed)
    pl = planets.attach_planets(systems, comp, seed=args.seed + 1)
    no_kz = planets.attach_planets(systems, comp, seed=args.seed + 1,
                                   binaries=planets.bn.BinaryModel(kozai=False), derived=None)
    print("\n== Habitable-zone planets per main-sequence star (S-type) ==")
    print(hz_census(comp, pl))
    print("\n== Spin states of main-sequence planets ==")
    print(locking(comp, pl))
    print("\n== Kozai-Lidov cycles ==")
    print(kozai_census(comp, pl, len(no_kz["period"])))
    print("\n== Giant eccentricities with and without a stellar companion ==")
    print(ecc_in_binaries(comp, pl))
    f = pl["flags"]
    print(f"\nplanets flagged hz_extrapolated: {np.mean((f & planets.FLAG_HZ_EXTRAP) > 0):.4f}, "
          f"kozai: {np.mean((f & planets.FLAG_KOZAI) > 0):.4f}")


if __name__ == "__main__":
    main()
