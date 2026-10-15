"""Moons: checks of :mod:`starsys.moons`.

- **Solar System analogues:** many draws of moons for planets with the masses, radii and
  orbits of the Sun's planets (age 4.6 Gyr): fraction with moons, number of moons and total
  mass ratio, against the real systems (regular satellites of Jupiter 2.1e-4, Saturn 2.5e-4,
  Uranus 1.0e-4; Triton 2.1e-4 of Neptune; the Moon 0.0123 of Earth; Phobos and Deimos are
  captured asteroids, not modelled).
- **Kepler (Teachey+2018, HEK VI):** stacking 284 warm (0.1-1 au) Kepler planets of Earth to
  Jupiter size gives an occurrence of Galilean-analogue moon systems eta < 0.38 (95%),
  0.16 +0.13/-0.10. Model: fraction of planets of 1-11 Earth radii at 0.1-1 au around FGK
  main-sequence stars with a moon of >= 0.2 Earth radii (Galilean moons are 0.29-0.41).
  Their sample is mostly sub-Neptunes: warm Kepler DR25 candidates (P < 397 d, 0.1-1 au, FGK)
  have a median radius of 2.5 Earth radii and only 8% are above 6. The like-for-like model
  value weights the model's eta per radius bin by those candidates' radii, for the whole
  sample and for its smaller and larger halves (Teachey's Table 1 splits the 284 planets at
  their median radius: eta in [0.06, 0.35] and [0.07, 0.44] at 1 sigma, < 0.48 and < 0.61 at
  95%). Their target selection favours planets that can hold large moons, so the larger
  half is the more representative comparison.
- **Survival vs orbit:** fraction of giant planets with moons and of rocky planets with an
  impact moon by semi-major axis and host type (Barnes & O'Brien 2002: close-in planets lose them).
- **Tidal heating:** resonant regular moons with a heat flux above the Earth's internal heat
  flow (Io: 2.5 W/m^2, Europa ~0.1-0.2 W/m^2).
- **Large moons in habitable zones:** moons of >= 0.1 Earth masses (able to hold an atmosphere
  for Gyr; Williams+1997, Heller+2014) around planets in the conservative habitable zone, per star.

Usage: ``python -m starsys.validation.moons [--systems 200000]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import derived as dv
from .. import moons
from .. import planets
from ..paths import CATALOGS_DIR
from ..stars.population import draw_systems

KOI = CATALOGS_DIR / "observational" / "kepler_dr25_koi.csv"
TEACHEY_EDGES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 11.0, 30.0])

SOLAR = [  # name, a (au), e, mass (Earth), radius (Earth), envelope lost, observed
    ("Mercury", 0.387, 0.206, 0.055, 0.383, True, "none"),
    ("Venus", 0.723, 0.007, 0.815, 0.949, True, "none"),
    ("Earth", 1.0, 0.017, 1.0, 1.0, True, "Moon, 0.0123 at 60 Re"),
    ("Mars", 1.524, 0.093, 0.107, 0.532, True, "Phobos, Deimos (captured, ~2e-8)"),
    ("Jupiter", 5.203, 0.049, 317.8, 11.21, False, "4 regular, 2.1e-4, 5.9-26 Rp"),
    ("Saturn", 9.537, 0.057, 95.2, 9.45, False, "Titan + 6 mid-size, 2.5e-4, 3-59 Rp"),
    ("Uranus", 19.19, 0.046, 14.5, 4.01, False, "5 regular, 1.0e-4, 5-23 Rp"),
    ("Neptune", 30.07, 0.010, 17.1, 3.88, False, "Triton (captured, retrograde), 2.1e-4, 14 Rp"),
]


def _solar_planets(n_each):
    rows = []
    for name, a, e, m, r, lost, _ in SOLAR:
        rows.append(pd.DataFrame({"name": name, "a_au": a, "ecc": e, "planet_mass": m, "planet_radius": r,
                                  "lost_envelope": lost}, index=range(n_each)))
    t = pd.concat(rows, ignore_index=True)
    n = len(t)
    comp = {"system": np.zeros(1, int), "role": np.zeros(1, int), "luminosity": np.array([1.0]),
            "teff": np.array([5772.0]), "mass": np.array([1.0]), "age_gyr": np.array([4.6]),
            "a_au": np.array([np.nan]), "ecc": np.array([np.nan])}
    pl = {"system": np.zeros(n, int), "component": np.zeros(n, int), "host_kind": np.full(n, "S", dtype=object),
          "host_mass": np.ones(n), "flags": np.zeros(n, np.int32)}
    for col in ("a_au", "ecc", "planet_mass", "planet_radius", "lost_envelope"):
        pl[col] = t[col].to_numpy()
    extra = dv.derive_properties(pl, {"n_components": np.array([1])}, comp)
    extra.pop("hz_flags")
    pl.update(extra)
    return t["name"].to_numpy(), pl, comp


def solar_system(n_each=2000, seed=0) -> pd.DataFrame:
    names, pl, comp = _solar_planets(n_each)
    m = moons.attach_moons(pl, comp, seed=seed)
    n_moons = np.bincount(m["planet"], minlength=len(names))
    tot = np.bincount(m["planet"], weights=m["mass"], minlength=len(names)) / pl["planet_mass"]
    rows = []
    for name, *_, obs in SOLAR:
        sel = names == name
        has = n_moons[sel] > 0
        kinds = m["kind"][np.isin(m["planet"], np.flatnonzero(sel))]
        rows.append({"planet": name, "% with moons": 100 * has.mean(),
                     "median N (if any)": np.median(n_moons[sel][has]) if has.any() else 0,
                     "median mass ratio (if any)": np.median(tot[sel][has]) if has.any() else np.nan,
                     "kinds": ", ".join(sorted(set(kinds))) or "-", "observed": obs})
    return pd.DataFrame(rows).set_index("planet")


def teachey(comp, pl, m) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    c = pl["component"]
    fgk = planets.hmc_hosts(comp) & (np.asarray(comp["phase"]) == 0) & (teff >= 3900) & (teff < 7200)
    warm = (pl["host_kind"] == "S") & fgk[c] & (pl["a_au"] >= 0.1) & (pl["a_au"] < 1.0)
    rows = []
    for label, r_lo, r_hi in (("1-11 Re", 1.0, 11.0), ("4-11 Re", 4.0, 11.0), ("> 6 Re", 6.0, 30.0)):
        sel = warm & (pl["planet_radius"] >= r_lo) & (pl["planet_radius"] < r_hi)
        big = np.zeros(len(pl["a_au"]), dtype=bool)
        big[m["planet"][m["radius"] >= 0.2]] = True
        rows.append({"planets": label, "n": int(sel.sum()), "model eta": big[sel].mean(),
                     "Teachey+2018": "< 0.38 (95%), 0.16 +0.13 -0.10"})
    if KOI.exists():
        r = pl["planet_radius"]
        e = TEACHEY_EDGES
        eta = np.array([big[warm & (r >= lo) & (r < hi)].mean() for lo, hi in zip(e[:-1], e[1:])])
        k = pd.read_csv(KOI, usecols=["koi_pdisposition", "koi_period", "koi_sma", "koi_steff", "koi_prad"])
        k = k[(k.koi_pdisposition == "CANDIDATE") & (k.koi_period < 397.4) & (k.koi_sma >= 0.1) & (k.koi_sma < 1.0)
              & (k.koi_steff >= 3900) & (k.koi_steff < 7200)]
        rk = k.koi_prad.dropna().to_numpy()
        rk = rk[(rk >= e[0]) & (rk < e[-1])]
        w = eta[np.clip(np.digitize(rk, e) - 1, 0, len(eta) - 1)]
        med = np.median(rk)
        for label, sel, obs in (("Kepler-weighted, all", rk > 0, "< 0.38 (95%), 0.16 +0.13 -0.10"),
                                (f"Kepler-weighted, < {med:.1f} Re", rk < med, "< 0.48 (95%), [0.06, 0.35]"),
                                (f"Kepler-weighted, >= {med:.1f} Re", rk >= med, "< 0.61 (95%), [0.07, 0.44]")):
            rows.append({"planets": label, "n": int(sel.sum()), "model eta": w[sel].mean(), "Teachey+2018": obs})
    return pd.DataFrame(rows).set_index("planets")


def survival(comp, pl, m) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    c = pl["component"]
    s = pl["host_kind"] == "S"
    has = np.zeros(len(pl["a_au"]), dtype=bool)
    has[m["planet"]] = True
    has_imp = np.zeros(len(pl["a_au"]), dtype=bool)
    has_imp[m["planet"][m["kind"] == "impact"]] = True
    giant = s & (pl["planet_mass"] >= 95.0)
    rocky = s & (dv.planet_kind(pl["planet_mass"], pl["planet_radius"], pl["lost_envelope"]) == 0) & (
        pl["planet_mass"] >= 0.1) & (pl["planet_mass"] <= 6.0)
    rows = []
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, np.inf)):
        a = (pl["a_au"] >= lo) & (pl["a_au"] < hi)
        row = {"a (au)": f"{lo:g}-{hi:g}", "giants": int((giant & a).sum()),
               "% giants with moons": 100 * has[giant & a].mean() if (giant & a).any() else np.nan}
        for name, (t_lo, t_hi) in (("M", (2300, 3900)), ("FGK", (3900, 7200))):
            r = rocky & a & (teff[c] >= t_lo) & (teff[c] < t_hi)
            row[f"% rocky with impact moon ({name})"] = 100 * has_imp[r].mean() if r.sum() >= 50 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("a (au)")


def census(comp, pl, m) -> pd.DataFrame:
    n_stars = int((planets.hmc_hosts(comp) & (np.asarray(comp["phase"]) == 0)).sum())
    rows = []
    for kind in ("regular", "impact", "captured"):
        x = m["kind"] == kind
        rows.append({"kind": kind, "moons per 1000 planets": 1000 * x.sum() / len(pl["a_au"]),
                     "median mass (Earth)": np.median(m["mass"][x]) if x.any() else np.nan,
                     "median a (planet radii)": np.median(m["a_rp"][x]) if x.any() else np.nan,
                     "% icy": 100 * np.mean(m["composition"][x] == "ice") if x.any() else np.nan})
    hz = np.zeros(len(pl["a_au"]), dtype=bool)
    hz[pl["hz"] == dv.HZ_CONSERVATIVE] = True
    big = (m["mass"] >= 0.1) & hz[m["planet"]]
    rows.append({"kind": "moons >= 0.1 Earth masses in the conservative HZ, per 1000 main-sequence stars",
                 "moons per 1000 planets": 1000 * big.sum() / max(n_stars, 1)})
    hot = m["heat_flux_w_m2"] > 0.09
    rows.append({"kind": "tidally heated above the Earth's internal heat flow (0.09 W/m^2)",
                 "moons per 1000 planets": 1000 * hot.sum() / len(pl["a_au"]),
                 "median mass (Earth)": np.median(m["mass"][hot]) if hot.any() else np.nan,
                 "median a (planet radii)": np.median(m["a_rp"][hot]) if hot.any() else np.nan,
                 "% icy": 100 * np.mean(m["composition"][hot] == "ice") if hot.any() else np.nan})
    return pd.DataFrame(rows).set_index("kind")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.precision", 3)
    print("== Solar System analogues (2000 draws each) ==")
    print(solar_system())
    systems, comp = draw_systems(args.systems, seed=args.seed)
    pl = planets.attach_planets(systems, comp, seed=args.seed + 1)
    m = moons.attach_moons(pl, comp, seed=args.seed + 2)
    print("\n== Galilean-analogue moons of warm Kepler-like planets (Teachey+2018) ==")
    print(teachey(comp, pl, m))
    print("\n== Survival vs orbit ==")
    print(survival(comp, pl, m))
    print("\n== Census ==")
    print(census(comp, pl, m))


if __name__ == "__main__":
    main()
