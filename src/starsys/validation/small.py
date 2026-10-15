"""Small planets beyond 300 d: checks of :mod:`starsys.outer.small`.

Model planets come from the stellar population (:func:`starsys.stars.population.draw_systems`)
through :func:`starsys.planets.attach_planets` (all models). Rates are intrinsic (planets
per star, no detection model).

- **Profile:** dN/d ln P of planets below 30 Earth masses (HM-C and small outer planets)
  across HM-C's 300 d edge and out to 10 snow-line radii, by host mass.
- **η⊕ (Bryson+2021):** planets of 0.5-1.5 Earth radii in the conservative (runaway to
  maximum greenhouse) and optimistic (recent Venus to early Mars) habitable zones of
  Kopparapu+2014, around main-sequence hosts of 4800-6300 K. Bryson+2021 give bounds
  from two completeness extrapolations: conservative 0.37 (+0.48 -0.21) to 0.60 (+0.90
  -0.36), optimistic 0.58 (+0.73 -0.33) to 0.88 (+1.28 -0.51). Both assume that the
  radius distribution does not change with period beyond where Kepler measures it.
- **Other long-period small-planet rates:** Hsu+2019 (SysSim, Kepler DR25 + Gaia DR2, FGK
  with 0.5 < bp-rp < 1.7): 0.75-1.5 Earth radii at 237-500 d, < 0.27 per star (84th
  percentile), plausible range 0.03-0.40. Neil & Rogers 2020 (CKS, 4700-6500 K):
  0.8-1.2 Earth radii at 292-438 d, 0.076 without envelope loss, 0.008-0.009 with it (their
  preferred models, whose retention law HM-C uses); Burke+2015: 0.1 (+1.9 -0.1).
- **Microlensing (Zang+2025):** mass-ratio function of all planets (giants included) at
  projected separations s = 0.5-2 Einstein radii, around lens-like hosts (main-sequence
  0.1-1 Msun stars weighted by M^0.5, the Einstein-radius cross-section). The Einstein
  radius is 2.5 au (M / 0.5 Msun)^0.5 (lens at 6 kpc, source at 8 kpc); projected
  separations use an isotropic orbital phase and orientation (circular orbits).
- **Wide orbits (Poleski+2021):** planets with 1e-4 < q < 0.033 at s = 2-6 around the same
  hosts: 1.4 (+0.9 -0.6) per star (default selection; Suzuki+2016 extrapolated: 0.41).

Usage: ``python -m starsys.validation.small [--systems 200000]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import planets
from ..derived import s_eff
from ..stars.population import draw_systems
from ..units import EARTH_MASS_MSUN

SMALL_MODELS = ("hmc", "hmc_m", "outer_small")

# Kopparapu+2014 limits (1 Earth mass), from :func:`starsys.derived.s_eff`.
KOPPARAPU = {"recent_venus": "recent_venus", "runaway_greenhouse": "runaway", "maximum_greenhouse": "max_greenhouse",
             "early_mars": "early_mars"}
HZ = {"conservative": ("runaway_greenhouse", "maximum_greenhouse"), "optimistic": ("recent_venus", "early_mars")}
# Bryson+2021: (lower-bound model: value, +err, -err), (upper-bound model: ...).
BRYSON2021 = {"conservative": ((0.37, 0.48, 0.21), (0.60, 0.90, 0.36)),
              "optimistic": ((0.58, 0.73, 0.33), (0.88, 1.28, 0.51))}
BRYSON_TEFF = (4800.0, 6300.0)
# (label, R range, P range (d), host Teff range, published value).
LONG_PERIOD_RATES = [
    ("Hsu+2019", (0.75, 1.5), (237.0, 500.0), (3950.0, 6800.0), "< 0.27 (84%); plausible 0.03-0.40"),
    ("Neil & Rogers 2020", (0.8, 1.2), (292.0, 438.0), (4700.0, 6500.0),
     "0.008-0.009 with envelope loss, 0.076 without; Burke+2015 0.1 (+1.9 -0.1)"),
]

# Zang+2025 Table S5 medians: power law and double Gaussian, per dex of q per dex of s.
ZANG_POWER = (0.177, -0.547)
ZANG_GAUSS = ((0.52, -4.67, -0.73), (0.058, -2.63, -1.78))
ZANG_LOG_S = (-0.3, 0.3)
POLESKI2021 = (1.39, 0.92, 0.59)  # 1e-4 < q < 0.033, 2 < s < 6, default selection
EINSTEIN_RADIUS_AU = 2.5  # at 0.5 Msun


def seff(teff, limit):
    return s_eff(teff, 1.0)[KOPPARAPU[limit]]


def simulate(n_systems: int, seed: int = 0):
    systems, comp = draw_systems(n_systems, seed=seed)
    pl = planets.attach_planets(systems, comp, seed=seed + 1)
    return systems, comp, pl


def _ms_hosts(comp):
    return planets.hmc_hosts(comp) & (np.asarray(comp["phase"]) == 0)


def profile(comp, pl) -> pd.DataFrame:
    """dN/d ln P of planets below 30 Earth masses, per host, in factor-2 period bins."""
    groups = {"early M 0.35-0.65": (0.35, 0.65), "K 0.65-0.85": (0.65, 0.85), "G 0.85-1.15": (0.85, 1.15),
              "F 1.15-1.5": (1.15, 1.5)}
    edges = 300.0 * 2.0 ** np.arange(-3, 9)
    ms = _ms_hosts(comp)
    mass = np.asarray(comp["mass"])
    small = np.isin(pl["model"], SMALL_MODELS) & (pl["planet_mass"] < 30.0)
    out = {}
    for name, (lo, hi) in groups.items():
        hosts = ms & (mass >= lo) & (mass < hi)
        sel = small & hosts[pl["component"]]
        h, _ = np.histogram(pl["period"][sel], edges)
        out[name] = h / hosts.sum() / np.log(2.0)
    df = pd.DataFrame(out, index=[f"{a:.0f}-{b:.0f}" for a, b in zip(edges[:-1], edges[1:])])
    df.index.name = "P (d)"
    return df


def eta_earth(comp, pl) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    hosts = _ms_hosts(comp) & (teff >= BRYSON_TEFF[0]) & (teff <= BRYSON_TEFF[1])
    c = pl["component"]
    s = np.asarray(comp["luminosity"])[c] / pl["a_au"] ** 2
    rocky = (pl["planet_radius"] >= 0.5) & (pl["planet_radius"] <= 1.5) & hosts[c]
    rows = []
    for name, (inner, outer) in HZ.items():
        in_hz = (s <= seff(teff[c], inner)) & (s >= seff(teff[c], outer))
        n = hosts.sum()
        (v1, u1, l1), (v2, u2, l2) = BRYSON2021[name]
        rows.append({
            "HZ": name, "model": (rocky & in_hz).sum() / n,
            "  of which HM-C": (rocky & in_hz & np.isin(pl["model"], ("hmc", "hmc_m"))).sum() / n,
            "  of which outer": (rocky & in_hz & (pl["model"] == "outer_small")).sum() / n,
            "all sizes in HZ": (in_hz & hosts[c]).sum() / n,
            "Bryson+2021": f"{v1} (+{u1} -{l1}) - {v2} (+{u2} -{l2})",
        })
    return pd.DataFrame(rows).set_index("HZ")


def long_period_rates(comp, pl) -> pd.DataFrame:
    teff = np.asarray(comp["teff"])
    c = pl["component"]
    rows = []
    for label, (r0, r1), (p0, p1), (t0, t1), published in LONG_PERIOD_RATES:
        hosts = _ms_hosts(comp) & (teff >= t0) & (teff <= t1)
        sel = (hosts[c] & (pl["planet_radius"] >= r0) & (pl["planet_radius"] < r1) & (pl["period"] >= p0)
               & (pl["period"] < p1))
        rows.append({"source": label, "R (Re)": f"{r0}-{r1}", "P (d)": f"{p0:.0f}-{p1:.0f}",
                     "model": sel.sum() / hosts.sum(), "published": published})
    return pd.DataFrame(rows).set_index("source")


def _lens_like(comp, pl, rng):
    """Host weights (lens-like hosts) and per-planet log q and projected s."""
    mass = np.asarray(comp["mass"])
    w = np.where(_ms_hosts(comp) & (mass >= 0.1) & (mass <= 1.0), np.sqrt(mass), 0.0)
    c = pl["component"]
    log_q = np.log10(pl["planet_mass"] * EARTH_MASS_MSUN / mass[c])
    cos_psi = rng.uniform(-1.0, 1.0, len(c))
    r_proj = pl["a_au"] * np.sqrt(1.0 - cos_psi**2)
    log_s = np.log10(r_proj / (EINSTEIN_RADIUS_AU * np.sqrt(mass[c] / 0.5)))
    return w, log_q, log_s


def zang(log_q):
    g = sum(a * 10.0 ** (n * (log_q - p) ** 2) for a, p, n in ZANG_GAUSS)
    return ZANG_POWER[0] * (10.0 ** log_q / 1e-4) ** ZANG_POWER[1], g


def microlensing(comp, pl, rng) -> pd.DataFrame:
    w, log_q, log_s = _lens_like(comp, pl, rng)
    pw = w[pl["component"]]
    edges = np.arange(-6.0, -1.49, 0.5)
    in_s = (log_s >= ZANG_LOG_S[0]) & (log_s < ZANG_LOG_S[1])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = in_s & (log_q >= lo) & (log_q < hi)
        dens = pw[sel].sum() / w.sum() / (hi - lo) / (ZANG_LOG_S[1] - ZANG_LOG_S[0])
        x = np.linspace(lo, hi, 21)
        power, gauss = (v.mean() for v in zang(x))
        small = (pw[sel & np.isin(pl["model"], SMALL_MODELS)].sum() / w.sum()
                 / (hi - lo) / (ZANG_LOG_S[1] - ZANG_LOG_S[0]))
        rows.append({"log q": f"{lo:.1f} to {hi:.1f}", "model": dens, "  small planets": small,
                     "Zang double Gaussian": gauss, "Zang power law": power,
                     "measured": "yes" if lo >= -5.25 else "no (extrapolated)"})
    return pd.DataFrame(rows).set_index("log q")


def poleski(comp, pl, rng) -> dict:
    w, log_q, log_s = _lens_like(comp, pl, rng)
    pw = w[pl["component"]]
    sel = (log_q >= -4.0) & (log_q < np.log10(0.033)) & (log_s >= np.log10(2.0)) & (log_s < np.log10(6.0))
    small = sel & np.isin(pl["model"], SMALL_MODELS)
    return {"model": pw[sel].sum() / w.sum(), "of which small (< 30 Me)": pw[small].sum() / w.sum(),
            "Poleski+2021": f"{POLESKI2021[0]} (+{POLESKI2021[1]} -{POLESKI2021[2]})",
            "Suzuki+2016 extrapolated": 0.41}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.precision", 3)
    _, comp, pl = simulate(args.systems, args.seed)
    rng = np.random.default_rng(args.seed + 7)
    print("== dN/d ln P, planets < 30 Earth masses (HM-C + outer small), per main-sequence host ==")
    print(profile(comp, pl))
    print("\n== eta-Earth: 0.5-1.5 Re in the habitable zone, MS hosts 4800-6300 K ==")
    print(eta_earth(comp, pl))
    print("\n== Other long-period rates of small planets, MS hosts ==")
    print(long_period_rates(comp, pl))
    print("\n== Microlensing mass-ratio function at s = 0.5-2, lens-like hosts (per dex q per dex s) ==")
    print(microlensing(comp, pl, rng))
    print("\n== Wide orbits: 1e-4 < q < 0.033, s = 2-6, per lens-like host ==")
    for k, v in poleski(comp, pl, rng).items():
        print(f"  {k}: {v if isinstance(v, str) else f'{v:.3f}'}")
    o = pl["model"] == "outer_small"
    f = pl["flags"][o]
    print(f"\nouter small planets: {o.sum()}; flagged bridge {np.mean((f & planets.FLAG_BRIDGE) > 0):.2f}, "
          f"cold_extrapolated {np.mean((f & planets.FLAG_COLD_EXTRAP) > 0):.2f}")


if __name__ == "__main__":
    main()
