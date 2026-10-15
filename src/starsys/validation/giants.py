"""Giant planets: calibration and checks of :mod:`starsys.outer.giants`.

Everything is evaluated on the California Legacy Survey stellar sample (Rosenthal+2021,
719 stars, CDS J/ApJS/255/8; published masses and [Fe/H]), replicated ``--reps`` times.
Rates are in M sin i, using each model planet's sky inclination.

Calibration (``--fit``):

- ``norm`` and ``dispersion`` jointly, minimising the sum of squared z-scores over
  :data:`FIT_TARGETS`: Fulton+2021 (2-8 and 8-32 au), Rosenthal+2022 (fraction of stars
  with a giant at 0.23-10 au), the raw CLS count above 300 Earth masses at 0.1-5 au
  (near complete; Poisson error), and the fraction of hosts with two or more giants in
  a detectable subset (M sin i >= 30 Earth masses, a < 5 au, K > 5 m/s);
- ``hj_rate``: 1.0 hot Jupiters per 100 stars;
- ``hj_outer_boost``: 51% of hot-Jupiter hosts with a 1-13 MJup companion at 1-20 au
  (Knutson+2014);
- ``ecc_beta``: the eccentricity prior, such that after the stability conditioning
  against the inner (HM-C) systems the giants at 0.1-10 au of Sun-like stars from the
  stellar population have the mean and median eccentricity of the CLS giants.

Checks: CLS raw counts and mass function, eccentricities, metallicity splits (Bryan &
Lee 2024; Van Zandt & Petigura 2024; Bonomo+2025), Johnson+2010, direct imaging
(Nielsen+2019; Vigan+2021), and the overlap with HM-C inside 300 d.

Usage: ``python -m starsys.validation.giants [--fit] [--reps 200]``
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from ..outer import giants as gp
from ..paths import CATALOGS_DIR
from ..units import a_au_to_period

CLS_DIR = CATALOGS_DIR / "observational" / "cls"
CLS_N_STARS = 719
MJUP_E = 317.83


def _cls_key(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    return key[2:] if key.startswith("hd") else key


def load_cls() -> tuple[pd.DataFrame, pd.DataFrame]:
    """CLS stars (``key``, ``mass``, ``feh``) and planets (``host``, ``msini_e``, ``a``, ``e``, ``mstar``)."""
    stars = pd.read_fwf(CLS_DIR / "table2.dat", header=None, colspecs=[(4, 13), (27, 33), (65, 70)],
                        names=["key", "feh", "mass"], dtype={"key": str})
    stars["key"] = stars["key"].map(_cls_key)
    pl = pd.read_csv(CLS_DIR / "table3.dat", sep="|", header=None).iloc[:, [0, 2, 5, 8]]
    pl.columns = ["name", "msini", "a", "e"]
    pl["host"] = pl["name"].str.strip().str.replace(r"\s*[a-z]$", "", regex=True)
    pl["msini_e"] = pl["msini"] * MJUP_E
    pl["hkey"] = pl["host"].map(_cls_key)
    pl = pl.merge(stars.rename(columns={"key": "hkey", "mass": "mstar"})[["hkey", "mstar"]], on="hkey", how="left")
    pl["mstar"] = pl["mstar"].fillna(1.0)
    return stars, pl[["host", "msini_e", "a", "e", "mstar"]]


def rv_semi_amplitude(msini_e, period_d, mstar, ecc=0.0):
    """K (m/s), planet mass neglected against the star's."""
    return 28.4329 * (msini_e / MJUP_E) * (period_d / 365.25) ** (-1 / 3) * mstar ** (-2 / 3) / np.sqrt(1 - ecc**2)


def simulate(stars: pd.DataFrame, model: gp.GiantModel, reps: int, seed: int = 0):
    """Giants for ``reps`` copies of a stellar sample; adds ``msini``, ``mstar`` and ``feh`` per planet."""
    mass = np.tile(stars["mass"].to_numpy(dtype=float), reps)
    feh = np.tile(stars["feh"].to_numpy(dtype=float), reps)
    g = gp.draw_giants(mass, feh, np.ones_like(mass), np.random.default_rng(seed), model)
    g["msini"] = g["planet_mass"] * np.sin(g["incl_sky"])
    g["mstar"] = mass[g["host"]]
    g["feh"] = feh[g["host"]]
    g["n_stars"] = len(mass)
    g["star_feh"] = feh
    return g


def _sel(g, a, m, key="msini"):
    return (g["a_au"] >= a[0]) & (g["a_au"] < a[1]) & (g[key] >= m[0]) & (g[key] < m[1])


def rate(g, a=(0, np.inf), m=(30.0, 6000.0), key="msini"):
    """Planets per star."""
    return _sel(g, a, m, key).sum() / g["n_stars"]


def host_frac(g, a=(0, np.inf), m=(30.0, 6000.0), key="msini", star_mask=None):
    """Fraction of stars (optionally of those in ``star_mask``) with at least one such planet."""
    s = _sel(g, a, m, key)
    if star_mask is None:
        return len(np.unique(g["host"][s])) / g["n_stars"]
    s &= star_mask[g["host"]]
    return len(np.unique(g["host"][s])) / star_mask.sum()


def detectable(g):
    """CLS-like detectable subset: M sin i >= 30 Earth masses, a < 5 au, K > 5 m/s."""
    p = a_au_to_period(g["a_au"], g["mstar"])
    return (g["msini"] >= 30) & (g["a_au"] < 5) & (rv_semi_amplitude(g["msini"], p, g["mstar"], g["ecc"]) > 5)


def multi_fraction(g):
    n = np.bincount(g["host"][detectable(g)])
    return (n >= 2).sum() / (n >= 1).sum()


def cls_detectable(cls_pl):
    p = a_au_to_period(cls_pl["a"], cls_pl["mstar"])
    return (cls_pl["msini_e"] >= 30) & (cls_pl["a"] < 5) & (
        rv_semi_amplitude(cls_pl["msini_e"], p, cls_pl["mstar"], cls_pl["e"]) > 5)


# --- targets ---------------------------------------------------------------------------

def fit_targets(cls_pl):
    """(label, model function, observed, +err, -err)."""
    k300 = int(((cls_pl["a"] >= 0.1) & (cls_pl["a"] < 5) & (cls_pl["msini_e"] >= 300)).sum())
    nc = cls_pl[cls_detectable(cls_pl)].groupby("host").size()
    f_multi = (nc >= 2).mean()
    e_multi = np.sqrt(f_multi * (1 - f_multi) / len(nc))
    return [
        ("2-8 au, 30-6000 Me, per 100 stars (Fulton+2021)", lambda g: 100 * rate(g, a=(2, 8)), 14.1, 2.0, 1.8),
        ("8-32 au, 30-6000 Me, per 100 stars (Fulton+2021)", lambda g: 100 * rate(g, a=(8, 32)), 8.9, 3.0, 2.4),
        ("% stars with giant 0.23-10 au (Rosenthal+2022)", lambda g: 100 * host_frac(g, a=(0.23, 10)), 17.6, 2.4, 1.9),
        (f"0.1-5 au, 300-6000 Me, per 100 stars (CLS raw, {k300})", lambda g: 100 * rate(g, a=(0.1, 5), m=(300, 6000)),
         100 * k300 / CLS_N_STARS, 100 * np.sqrt(k300) / CLS_N_STARS, 100 * np.sqrt(k300) / CLS_N_STARS),
        (f"% detectable giant hosts with >= 2 (CLS {(nc >= 2).sum()}/{len(nc)})", lambda g: 100 * multi_fraction(g),
         100 * f_multi, 100 * e_multi, 100 * e_multi),
    ]


def _z(model_value, obs, up, lo):
    return (model_value - obs) / (up if model_value >= obs else lo)


def score(g, targets):
    return sum(_z(f(g), obs, up, lo) ** 2 for _, f, obs, up, lo in targets)


# --- calibration -----------------------------------------------------------------------

HJ_TARGET = 0.010
KNUTSON_HJ_COMPANION = 0.51
DISPERSION_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, np.inf)


def calibrate(stars, cls_pl, model: gp.GiantModel, reps: int) -> gp.GiantModel:
    from scipy.optimize import minimize_scalar

    hj_rate = HJ_TARGET / gp.hj_scaling(stars["mass"], stars["feh"], model).mean()
    model = replace(model, hj_rate=round(float(hj_rate), 5))
    targets = fit_targets(cls_pl)
    best = None
    for k in DISPERSION_GRID:
        def obj(log_norm):
            return score(simulate(stars, replace(model, norm=float(np.exp(log_norm)), dispersion=k), reps, 1), targets)
        r = minimize_scalar(obj, bounds=(np.log(0.05), np.log(0.4)), method="bounded", options={"xatol": 0.01})
        print(f"  dispersion {k:5}: norm {np.exp(r.x):.4f}, score {r.fun:.2f}")
        if best is None or r.fun < best[0]:
            best = (r.fun, k, float(np.exp(r.x)))
    model = replace(model, dispersion=best[1], norm=round(best[2], 4))
    model = fit_ecc_prior(cls_pl, model)
    lo, hi = 1.0, 15.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        f = hj_companion_fraction(stars, replace(model, hj_outer_boost=mid), reps)
        lo, hi = (mid, hi) if f < KNUTSON_HJ_COMPANION else (lo, mid)
    return replace(model, hj_outer_boost=round(0.5 * (lo + hi), 2))


def sun_like_sample(n=40_000, seed=9):
    """Single main-sequence stars of 0.7-1.3 Msun from the stellar population (systems, components)."""
    from ..planets import hmc_hosts
    from ..stars.population import draw_systems

    systems, comp = draw_systems(n * 8, seed=seed)
    sel = np.flatnonzero(hmc_hosts(comp) & (comp["kind"] == "star") & (comp["mass"] >= 0.7) & (comp["mass"] < 1.3)
                         & (comp["phase"] == 0))[:n]
    sub = {key: np.asarray(v)[sel] for key, v in comp.items()}
    sub["system"] = np.arange(len(sel))
    sub["role"] = np.zeros(len(sel), int)
    return {"age_gyr": sub["age_gyr"], "n_components": np.ones(len(sel), int)}, sub


def conditioned_ecc(sample, model, seed=10):
    """Eccentricities of warm/cold giants at 0.1-10 au after attachment to the inner systems."""
    from ..planets import attach_planets

    pl = attach_planets(*sample, seed=seed, giants=model, small=None)
    m = (pl["model"] == "giant") & (pl["a_au"] >= 0.1) & (pl["a_au"] < 10)
    return pl["ecc"][m]


def cls_ecc(cls_pl):
    return cls_pl["e"][(cls_pl["msini_e"] >= 30) & (cls_pl["a"] >= 0.1) & (cls_pl["a"] < 10)].to_numpy()


def fit_ecc_prior(cls_pl, model: gp.GiantModel) -> gp.GiantModel:
    """Beta prior whose conditioned mean and median match the CLS giants."""
    from scipy.optimize import minimize

    obs = cls_ecc(cls_pl)
    target = np.array([obs.mean(), np.median(obs)])
    sigma = np.array([obs.std() / np.sqrt(len(obs)), 1.25 * obs.std() / np.sqrt(len(obs))])
    sample = sun_like_sample()

    def obj(x):
        if np.any(np.asarray(x) <= 0.05):
            return 1e9
        e = conditioned_ecc(sample, replace(model, ecc_beta=(float(x[0]), float(x[1]))))
        return float(np.sum(((np.array([e.mean(), np.median(e)]) - target) / sigma) ** 2))

    r = minimize(obj, np.array(model.ecc_beta), method="Nelder-Mead",
                 options={"xatol": 0.02, "fatol": 0.05, "maxfev": 40})
    print(f"  ecc prior: Beta{tuple(np.round(r.x, 3))}, score {r.fun:.2f}")
    return replace(model, ecc_beta=(round(float(r.x[0]), 3), round(float(r.x[1]), 3)))


def hj_companion_fraction(stars, model, reps, seed=2):
    """Among hot-Jupiter hosts: fraction with a 1-13 MJup (true mass) giant at 1-20 au."""
    g = simulate(stars, replace(model, hj_rate=model.hj_rate * 20), max(reps // 4, 1), seed=seed)
    hj_hosts = np.unique(g["host"][g["hot"]])
    comp = ~g["hot"] & _sel(g, (1, 20), (MJUP_E, 13 * MJUP_E), key="planet_mass")
    return np.isin(hj_hosts, g["host"][comp]).mean()


# --- report ----------------------------------------------------------------------------

def _row(label, model_value, obs, note=""):
    print(f"  {label:<62} model {model_value:7.2f}   obs {obs:<22} {note}")


def report(stars, cls_pl, model: gp.GiantModel, reps: int):
    g = simulate(stars, model, reps, seed=3)
    targets = fit_targets(cls_pl)
    print(f"CLS stellar sample x {reps} ({g['n_stars']} stars).\n\nFit targets:")
    for label, f, obs, up, lo in targets:
        _row(label, f(g), f"{obs:.1f} +{up:.1f}/-{lo:.1f}", f"z = {_z(f(g), obs, up, lo):+.2f}")
    print(f"  score (sum z^2) = {score(g, targets):.2f} over {len(targets)} targets")

    print("\nOther CLS-based rates (per 100 stars, M sin i):")
    _row("% stars with a hot Jupiter (calibration)", 100 * len(np.unique(g["host"][g["hot"]])) / g["n_stars"], "1.0")
    _row("3.8-7.1 au, 0.1-20 MJ (Fulton+2021 sec. 5.1)", 100 * rate(g, a=(3.8, 7.1), m=(31.8, 6357)),
         "15.5 +3.2/-3.0", "(exceeds their own 2-8 au rate)")
    p = a_au_to_period(g["a_au"], g["mstar"])
    sel = (p >= 3000) & (p < 10000) & (g["msini"] >= 95.3)
    _row("P 3000-10000 d, > 0.3 MJ (Fulton+2021)", 100 * sel.sum() / g["n_stars"], "12.6 +2.6/-2.0",
         "(Wittenmyer+2020: 6.9 +4.2/-2.1)")
    print("  CLS raw counts, M sin i 300-6000 Me (near complete inside ~5 au):")
    for lo, hi in [(0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, 5)]:
        k = ((cls_pl["a"] >= lo) & (cls_pl["a"] < hi) & (cls_pl["msini_e"] >= 300)).sum()
        _row(f"    {lo}-{hi} au", 100 * rate(g, a=(lo, hi), m=(300, 6000)),
             f"{100 * k / CLS_N_STARS:.2f} +-{100 * np.sqrt(k) / CLS_N_STARS:.2f}")
    w = (cls_pl["a"] >= 1) & (cls_pl["a"] < 5)
    obs = np.histogram(cls_pl["msini_e"][w], [300, 900, 3000, 6000])[0]
    mod = np.histogram(g["msini"][(g["a_au"] >= 1) & (g["a_au"] < 5)], [300, 900, 3000, 6000])[0]
    print(f"  mass function at 1-5 au, counts in M sin i 300-900-3000-6000 Me: CLS {obs.tolist()}, "
          f"model {np.round(mod * CLS_N_STARS / g['n_stars'], 1).tolist()}")
    e_cond, e_obs = conditioned_ecc(sun_like_sample(), model), cls_ecc(cls_pl)
    _row("giant eccentricity at 0.1-10 au after conditioning: mean", e_cond.mean(), f"{e_obs.mean():.3f} (CLS, n = {len(e_obs)})")
    _row("                                                    median", np.median(e_cond), f"{np.median(e_obs):.3f}")

    print("\n% stars with a giant at 1-10 au, 0.5-20 MJup (M sin i):")
    cj = dict(a=(1, 10), m=(0.5 * MJUP_E, 20 * MJUP_E))
    feh = g["star_feh"]
    _row("[Fe/H] > 0", 100 * host_frac(g, star_mask=feh > 0, **cj), "14.3 +2.0/-1.8 (B&L24)", "CLS: 16.4 +2.5/-2.4 (VZ&P24)")
    _row("[Fe/H] <= 0", 100 * host_frac(g, star_mask=feh <= 0, **cj), "5.0 +1.6/-1.3 (B&L24)")
    bon = stars.assign(mass=stars["mass"] * 0.994 / stars["mass"].mean(),
                       feh=stars["feh"] - stars["feh"].mean() - 0.072)
    _row("all, CLS sample shifted to M = 0.99, [Fe/H] = -0.07", 100 * host_frac(simulate(bon, model, reps, 4), **cj),
         "9.8 +0.9/-0.8 (Bonomo+2025)")
    g1 = simulate(pd.DataFrame({"mass": [1.0], "feh": [0.0]}), model, 200_000, seed=5)
    k1 = rv_semi_amplitude(g1["msini"], a_au_to_period(g1["a_au"], 1.0), 1.0, g1["ecc"])
    s1 = _sel(g1, (0, 2.5), (30, 6000)) & (k1 > 20)
    _row("1 Msun, [Fe/H] = 0: % with giant at a < 2.5 au, K > 20 m/s", 100 * len(np.unique(g1["host"][s1])) / g1["n_stars"],
         "7 +- 1 (Johnson+2010)")
    _row("HJ hosts with a 1-13 MJ companion at 1-20 au, % (calibration)", 100 * hj_companion_fraction(stars, model, reps),
         "51 +- 10 (Knutson+2014)")

    print("\nDirect imaging (true masses, [Fe/H] = 0, stellar masses uniform in the range), % of stars:")
    for (lo, hi), mr, ar, obs in [
            ((1.5, 3.0), (5, 13), (10, 100), "8.9 +5.0/-3.6 (Nielsen+2019, > 1.5 Msun)"),
            ((0.8, 1.2), (5, 13), (10, 100), "1.1 [0.7, 1.8] (Nielsen+2019 fit at 1 Msun)"),
            ((0.8, 1.2), (5, 13), (30, 100), "0.6 +0.7/-0.5 (Bowler 2016, FGK, to 300 au)"),
            ((0.5, 1.5), (2, 13), (3, 100), "< 6.9 (95%, Nielsen+2019; log-flat shape)"),
            ((0.7, 1.4), (1, 13), (5, 100), "5.8 +4.7/-2.8 (Vigan+2021, FGK, 1-75 MJ, 5-300 au)")]:
        gd = simulate(pd.DataFrame({"mass": np.linspace(lo, hi, 50), "feh": np.zeros(50)}), model, 4000, seed=6)
        _row(f"M* {lo}-{hi}: {mr[0]}-{mr[1]} MJ at {ar[0]}-{ar[1]} au",
             100 * host_frac(gd, a=ar, m=(mr[0] * MJUP_E, mr[1] * MJUP_E), key="planet_mass"), obs)


def report_hmc_overlap(model: gp.GiantModel):
    """Inside 3-300 d, planets below 10 Earth radii come from HM-C; compare its M >= 30 Me planets with CLS.

    Shown with HM-C's own masses and with the masses rescaled to RV (:mod:`starsys.inner.rv_masses`).
    """
    from ..inner import rv_masses
    from ..inner.hmc import HMCParams, generate_catalog

    n = 50_000
    _, pl = generate_catalog(HMCParams(), np.ones(n), np.ones(n), np.full(n, 5772.0), np.full(n, 0.82), seed=6)
    raw = pl["planet_mass"] >= 30
    m_rv = pl["planet_mass"] * np.where(pl["lost_envelope"], 1.0, rv_masses.correction(pl["planet_radius"]))
    big = m_rv >= 30
    a_lo, a_hi = gp.period_to_a_au(3.0, 1.0), gp.period_to_a_au(300.0, 1.0)
    cls_rate = model.norm * gp.separation_integral(a_lo, a_hi, model) * gp.mass_integral(30, 106, model)
    print("\nOverlap with HM-C (Sun-like star, P 3-300 d), planets per 100 stars:")
    _row("HM-C planets with M >= 30 Me, HM-C masses", 100 * raw.sum() / n, f"CLS model, 30-106 Me: {100 * cls_rate:.2f}")
    _row("HM-C planets with M >= 30 Me, rescaled to RV", 100 * big.sum() / n, f"CLS model, 30-106 Me: {100 * cls_rate:.2f}")
    print(f"    their HM-C radii: median {np.median(pl['planet_radius'][big]):.1f} Re, "
          f"quartiles {np.round(np.quantile(pl['planet_radius'][big], [0.25, 0.75]), 1).tolist()}")


def report_howard(n_systems: int = 200_000):
    """RV occurrence at P < 50 d by M sin i around GK dwarfs (0.54-1.28 Msun, no stellar companion):
    Howard+2010 (Eta-Earth survey, 166 stars), fit df/dlog10 M = 0.39 M^-0.48 (k +0.27/-0.16,
    exponent +0.12/-0.14), with HM-C's own masses and with the masses rescaled to RV."""
    from .. import binaries as bn
    from ..planets import attach_planets, hmc_hosts
    from ..stars.population import draw_systems

    systems, comp = draw_systems(n_systems, seed=9)
    mass = np.asarray(comp["mass"])
    gk = (hmc_hosts(comp) & (np.asarray(comp["phase"]) == 0) & (mass >= 0.54) & (mass <= 1.28)
          & np.isnan(bn.structure(comp)["a_near"]))
    fit = lambda lo, hi: 39.0 / (0.48 * np.log(10.0)) * (lo**-0.48 - hi**-0.48)  # noqa: E731
    runs = {rv: attach_planets(systems, comp, seed=10, rv_masses=rv, derived=None) for rv in (False, True)}
    print(f"\nRV occurrence at P < 50 d around {gk.sum()} GK dwarfs, % of stars (Howard+2010 fit):")
    for lo, hi in ((3, 10), (10, 30), (30, 100), (100, 1000)):
        vals = []
        for rv in (False, True):
            pl = runs[rv]
            c = pl["component"]
            sel = ((pl["host_kind"] == "S") & gk[c] & (pl["period"] < 50)
                   & (pl["planet_mass"] * np.sin(pl["incl_sky"]) >= lo) & (pl["planet_mass"] * np.sin(pl["incl_sky"]) < hi))
            vals.append(100 * (np.bincount(c[sel], minlength=len(mass)) > 0)[gk].mean())
        _row(f"M sin i {lo}-{hi} Me: HM-C masses {vals[0]:.2f}, rescaled", vals[1], f"{fit(lo, hi):.1f} (fit)")


def report_population(n_systems: int = 100_000):
    """Giants attached to the stellar population, and their effect on the inner (HM-C) systems."""
    from ..planets import attach_planets, hmc_hosts
    from ..stars.population import draw_systems

    systems, comp = draw_systems(n_systems, seed=7)
    with_g = attach_planets(systems, comp, seed=8, small=None)
    without = attach_planets(systems, comp, seed=8, giants=None, small=None)
    hosts = hmc_hosts(comp)
    print(f"\nStellar population ({n_systems} systems, {hosts.sum()} planet hosts):")
    print(f"  {'host mass (Msun)':<18}{'hosts':>8}{'% giant':>9}{'% HJ':>7}{'% wide':>8}{'inner/star':>12}{'(no giants)':>12}")
    for lo, hi in [(0.08, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 2.5), (2.5, 8.0)]:
        sel = hosts & (comp["mass"] >= lo) & (comp["mass"] < hi)
        n = sel.sum()
        if not n:
            continue
        m = sel[with_g["component"]]

        def frac(model):
            return 100 * len(np.unique(with_g["component"][m & (with_g["model"] == model)])) / n

        inner_w = (m & np.isin(with_g["model"], ["hmc", "hmc_m"])).sum() / n
        inner_wo = sel[without["component"]].sum() / n
        print(f"  {lo:>5}-{hi:<12}{n:>8}{frac('giant'):9.2f}{frac('hot_jupiter'):7.2f}{frac('wide_giant'):8.2f}"
              f"{inner_w:12.3f}{inner_wo:12.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    stars, cls_pl = load_cls()
    model = gp.GiantModel()
    if args.fit:
        print("Fitting norm and dispersion to CLS ...")
        model = calibrate(stars, cls_pl, model, args.reps)
        print("Calibrated:", {k: getattr(model, k) for k in ("norm", "dispersion", "hj_rate", "hj_outer_boost")}, "\n")
    report(stars, cls_pl, model, args.reps)
    report_hmc_overlap(model)
    report_howard()
    report_population()


if __name__ == "__main__":
    main()
