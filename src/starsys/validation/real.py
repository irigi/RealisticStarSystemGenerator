"""Real-star conditioning (:mod:`starsys.real`): checks.

- **Stellar fits** of benchmark stars against literature values that the fit does not use
  (ages, radii, white-dwarf temperatures) and the measurements it does use (pulls).
- **Injection test** (self-consistency of the conditioning): Sun-like stars get a "true"
  system drawn from the prior; a synthetic survey (RV: 60 epochs over 8 yr at 2 m/s plus 2 m/s
  jitter; transits: 4 TESS-like sectors at 100 ppm/h) detects some of its planets; the system
  is conditioned on the detected planets with the same surveys. If the conditioning is right,
  the posterior's predicted numbers of undetected planets, averaged over many stars, equal the
  true numbers of undetected planets (in bins of period and mass), and the posterior
  probability of an undetected cold giant is calibrated.

- **Hold-out test** (does the model's system architecture predict real systems?): in Kepler
  systems with at least three transiting planets, the middle planet is hidden, the system is
  conditioned on the others (no surveys), and the posterior probability of a planet within
  0.1 dex in period and radius of the hidden one is compared with the prior's (the same star,
  unconditioned). A ratio above 1 means the known planets carry real information about their
  neighbours (similar sizes, regular spacing).

Usage: ``python -m starsys.validation.real [--stars] [--injection N] [--holdout N]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .. import planets as pt
from ..real import condition as cd
from ..real import star as st
from ..real import surveys as sv
from ..real.catalog import ComponentRecord, Measurement, SystemRecord, find
from ..real.known import KnownPlanet

# (name, quantity, literature value, error, reference). Quantities the fit does not use.
LITERATURE = [
    ("Sun", "age_gyr", 4.57, 0.02, "meteoritic (Bouvier & Wadhwa 2010)"),
    ("Sirius B", "age_gyr", 0.24, 0.02, "Liebert+2005 (WD cooling + A isochrone)"),
    ("Sirius B", "teff", 25000.0, 200.0, "Barstow+2005"),
    ("Procyon B", "teff", 7740.0, 50.0, "Provencal+2002"),
    ("Procyon", "age_gyr", 2.7, 0.3, "Bond+2015"),
    ("alpha Centauri A", "radius", 1.2175, 0.0055, "Kervella+2017 (interferometry)"),
    ("alpha Centauri B", "radius", 0.8591, 0.0036, "Kervella+2017 (interferometry)"),
    ("Arcturus", "radius", 25.4, 0.2, "Ramirez & Allende Prieto 2011"),
    ("Arcturus", "age_gyr", 7.1, 1.5, "Ramirez & Allende Prieto 2011"),
    ("Vega", "age_gyr", 0.455, 0.013, "Yoon+2010"),
    ("eps Eri", "age_gyr", 0.6, 0.2, "Mamajek & Hillenbrand 2008 (activity, gyrochronology)"),
    ("40 Eri B", "mass", 0.573, 0.018, "Mason+2017 (dynamical)"),
    ("Pollux", "mass", 2.0, 0.3, "Hatzes+2012 (asteroseismology)"),
]


def sun_record() -> SystemRecord:
    c = ComponentRecord(name="Sun", measurements={
        "teff": Measurement(5772.0, 10.0, "IAU 2015"), "luminosity": Measurement(1.0, 0.01, "IAU 2015"),
        "radius": Measurement(1.0, 0.01, "IAU 2015"), "mass": Measurement(1.0, 0.01, "IAU 2015"),
        "feh": Measurement(0.0, 0.03, "definition")}, spectral_type="G2 V", distance_pc=10.0)
    return SystemRecord(name="Sun", components=[c], target=0, source="manual")


def stellar_benchmarks(n=1000, seed=1, use_gaia=True) -> pd.DataFrame:
    rows = []
    fits = {}
    for name in dict.fromkeys(r[0] for r in LITERATURE):
        rec = sun_record() if name == "Sun" else find(name, use_gaia=use_gaia)
        fits[name] = st.fit(rec, n=n, seed=seed)
    for name, q, val, err, ref in LITERATURE:
        f = fits[name]
        x = f.systems["age_gyr"] if q == "age_gyr" else f.components[q][f.target]
        lo, med, hi = np.percentile(x, [16, 50, 84])
        sig = np.hypot(0.5 * (hi - lo), err)
        rows.append({"star": name, "quantity": q, "model": f"{med:.4g} [{lo:.4g}-{hi:.4g}]",
                     "literature": f"{val:.4g} ± {err:.2g}", "pull": (med - val) / sig, "reference": ref})
    return pd.DataFrame(rows)


# --- injection test ------------------------------------------------------------------------

BINS_P = [(0.5, 10), (10, 100), (100, 1000), (1000, 1e6)]
BINS_M = [(0.0, 10.0), (10.0, 1e5)]


def _counts(period, mass) -> np.ndarray:
    out = np.zeros((len(BINS_P), len(BINS_M)))
    for i, (p0, p1) in enumerate(BINS_P):
        for j, (m0, m1) in enumerate(BINS_M):
            out[i, j] = ((period >= p0) & (period < p1) & (mass >= m0) & (mass < m1)).sum()
    return out


def _surveys():
    return [sv.RVMonitoring(p_monitored=1.0), sv.TransitCampaign(days=100.0, noise_1h=1e-4, name="TESS-like")]


def injection(n_stars=40, seed=7, n_post=50, model=cd.ConditionModel(n_prior=4000, max_prior=8000, target_ess=80)):
    """Returns (summary table, per-star records)."""
    rng = np.random.default_rng(seed)
    rec = sun_record()
    fit = st.fit(rec, n=model.max_prior + n_stars, seed=rng)
    truth_idx = np.arange(model.max_prior, model.max_prior + n_stars)
    systems, components, tgt = cd._subset_stars(fit, truth_idx)
    pl = pt.attach_planets(systems, components, seed=rng)
    surveys = _surveys()
    for s in surveys:
        s.prepare(rec, systems, components, tgt, [KnownPlanet("x", "rv", rv_detected=True)], rng)
    on = pl["component"] == tgt[pl["system"]]
    cand = {k: pl[k][on] for k in ("period", "a_au", "planet_mass", "planet_radius", "ecc", "host_mass")}
    cand["cos_i"] = np.abs(np.cos(pl["incl_sky"][on]))
    cand["r_star"] = np.asarray(components["radius"])[pl["component"][on]]
    cand["system"] = pl["system"][on]
    p_rv = surveys[0].p_detect(cand)
    p_tr = surveys[1].p_detect(cand)
    det_rv = rng.random(len(p_rv)) < p_rv
    det_tr = rng.random(len(p_tr)) < p_tr
    sub = fit_subset(fit, np.arange(model.max_prior))
    true_unseen, post_unseen, giant_truth, giant_post = [], [], [], []
    for s in range(n_stars):
        sel = np.flatnonzero(cand["system"] == s)
        known = []
        for j in sel[det_rv[sel] | det_tr[sel]]:
            sin_i = np.sqrt(1 - cand["cos_i"][j] ** 2)
            known.append(KnownPlanet(
                name=f"inj{s}-{j}", method="transit" if det_tr[j] else "rv", period=cand["period"][j],
                radius=cand["planet_radius"][j] if det_tr[j] else np.nan, radius_err=0.05 * cand["planet_radius"][j],
                msini=cand["planet_mass"][j] * sin_i if det_rv[j] and not det_tr[j] else np.nan,
                mass=cand["planet_mass"][j] if det_rv[j] and det_tr[j] else np.nan,
                transits=bool(det_tr[j]), rv_detected=bool(det_rv[j])))
        undetected = sel[~(det_rv[sel] | det_tr[sel])]
        true_unseen.append(_counts(cand["period"][undetected], cand["planet_mass"][undetected]))
        giant_truth.append(bool(((cand["period"][undetected] > 1000) & (cand["planet_mass"][undetected] > 95)).any()))
        c = cd.condition(sub, n=n_post, seed=rng, known=known, surveys=surveys, model=model, moons=False)
        p = c.planets
        mask = (p["component"] == c.target[p["system"]]) & ((p["flags"] & cd.FLAG_KNOWN) == 0)
        post_unseen.append(_counts(p["period"][mask], p["planet_mass"][mask]) / n_post)
        g = np.zeros(n_post, dtype=bool)
        big = mask & (p["period"] > 1000) & (p["planet_mass"] > 95)
        g[p["system"][big]] = True
        giant_post.append(g.mean())
    t, q = np.mean(true_unseen, axis=0), np.mean(post_unseen, axis=0)
    t_err = np.std(true_unseen, axis=0) / np.sqrt(n_stars)
    rows = []
    for i, (p0, p1) in enumerate(BINS_P):
        for j, (m0, m1) in enumerate(BINS_M):
            rows.append({"period (d)": f"{p0:g}-{p1:g}", "mass (Earth)": f"{m0:g}-{m1:g}",
                         "true undetected / star": f"{t[i, j]:.2f} ± {t_err[i, j]:.2f}",
                         "posterior predicted": f"{q[i, j]:.2f}"})
    gp, gt = np.array(giant_post), np.array(giant_truth)
    brier = np.mean((gp - gt) ** 2)
    brier_ref = np.mean((gt.mean() - gt) ** 2)
    extra = {"giant_rate_true": gt.mean(), "giant_rate_posterior": gp.mean(), "brier": brier,
             "brier_climatology": brier_ref, "detected_per_star": (det_rv | det_tr).sum() / n_stars}
    return pd.DataFrame(rows), extra


def _near(period, radius, p0, r0, dex=0.1) -> np.ndarray:
    return (np.abs(np.log10(period / p0)) < dex) & (np.abs(np.log10(radius / r0)) < dex)


def holdout(n_systems=30, seed=11, n_post=200, model=cd.ConditionModel(n_prior=5000, max_prior=20000, target_ess=150)):
    """Hold-out test on Kepler systems with >= 3 transiting planets. Returns a per-system table."""
    from ..real.catalog import nasa_hosts
    from ..real.known import known_planets

    rng = np.random.default_rng(seed)
    hosts = nasa_hosts()
    cands = hosts[(hosts["n_planets"] >= 3) & hosts["hostname"].str.match(r"^(Kepler|KOI)-")]
    rows = []
    for host in rng.permutation(cands["hostname"].to_numpy()):
        if len(rows) >= n_systems:
            break
        known = [k for k in known_planets(host) if k.transits and np.isfinite(k.period) and np.isfinite(k.radius)]
        if len(known) < 3:
            continue
        hid = known[len(known) // 2]
        rest = [k for k in known if k is not hid]
        try:
            rec = find(host, use_gaia=False)
            fit = st.fit(rec, n=model.max_prior + 4000, seed=rng, model=st.StarFitModel(n_age=1500, n_mass=100))
        except (KeyError, ValueError):
            continue
        post = cd.condition(fit_subset(fit, np.arange(model.max_prior)), n=n_post, seed=rng, known=rest,
                            surveys=None, model=model, moons=False)
        p = post.planets
        on = (p["component"] == post.target[p["system"]]) & ((p["flags"] & cd.FLAG_KNOWN) == 0)
        hit = np.zeros(n_post, dtype=bool)
        hit[p["system"][on & _near(p["period"], p["planet_radius"], hid.period, hid.radius)]] = True
        # Prior: the same star, unconditioned (systems not used above).
        systems, components, tgt = cd._subset_stars(fit, np.arange(model.max_prior, model.max_prior + 4000))
        q = pt.attach_planets(systems, components, seed=rng)
        qon = q["component"] == tgt[q["system"]]
        prior_hit = np.zeros(4000, dtype=bool)
        prior_hit[q["system"][qon & _near(q["period"], q["planet_radius"], hid.period, hid.radius)]] = True
        rows.append({"host": host, "known": len(rest), "hidden": hid.name, "P (d)": round(hid.period, 2),
                     "R (Earth)": round(hid.radius, 2), "posterior": hit.mean(), "prior": prior_hit.mean(),
                     "ESS": round(post.ess), "widening": round(post.kernel_widening, 2)})
    return pd.DataFrame(rows)


def fit_subset(fit: st.StarSystemFit, idx) -> st.StarSystemFit:
    systems, components, target = cd._subset_stars(fit, idx)
    return st.StarSystemFit(record=fit.record, systems=systems, components=components, target=target, ess=fit.ess)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stars", action="store_true", help="stellar benchmarks only")
    ap.add_argument("--injection", type=int, default=0, help="number of injected stars (0: skip)")
    ap.add_argument("--holdout", type=int, default=0, help="number of Kepler systems for the hold-out test")
    ap.add_argument("--no-gaia", action="store_true")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    t0 = time.time()
    if args.stars or not (args.injection or args.holdout):
        print(stellar_benchmarks(use_gaia=not args.no_gaia).to_string(index=False))
        print(f"[{time.time() - t0:.0f} s]")
    if args.injection:
        t0 = time.time()
        table, extra = injection(args.injection)
        print(table.to_string(index=False))
        print({k: round(float(v), 3) for k, v in extra.items()})
        print(f"[{time.time() - t0:.0f} s]")
    if args.holdout:
        t0 = time.time()
        h = holdout(args.holdout)
        print(h.to_string(index=False))
        ratio = (h["posterior"] + 1e-3) / (h["prior"] + 1e-3)
        print(f"median posterior / prior {np.median(ratio):.1f}; posterior > prior in {(ratio > 1).mean():.0%}; "
              f"mean posterior {h['posterior'].mean():.3f}, mean prior {h['prior'].mean():.3f}")
        print(f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
