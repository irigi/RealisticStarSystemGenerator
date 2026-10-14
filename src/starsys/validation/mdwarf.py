"""M-dwarf planet occurrence: fit and check of :mod:`starsys.inner.mdwarf`.

Samples of model stars drawn from the stellar population (main sequence, single):

- early M: 0.35-0.65 Msun (median ~0.5 Msun, as the Dressing & Charbonneau 2015
  Kepler sample), compared with DC15 Table 4 and the Hsu+2020 integrated rates;
- mid-to-late M: 0.1-0.3 Msun, compared with Ment & Charbonneau (2023).

Occurrence is intrinsic (planets per star, no detection model), as in those papers.

``--fit`` refits the early-M parameters to DC15 (Powell) and then the mid-to-late M
parameters to Ment & Charbonneau (grid). The score is the sum of squared log-space
residuals over measured cells (asymmetric 1-sigma errors), plus a one-sided penalty
where the model exceeds an upper limit.

Usage: ``python -m starsys.validation.mdwarf [--fit]``
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from ..inner import mdwarf as md
from ..planets import DEFAULT_MDWARF, attach_planets, hmc_hosts
from ..stars.population import draw_systems


def star_sample(mass_range, n_target=20_000, seed=0):
    """Single main-sequence stars in a mass range, as a components/systems pair."""
    systems, comp = draw_systems(int(n_target * 4), seed=seed)
    sel = np.flatnonzero(hmc_hosts(comp) & (comp["kind"] == "star") & (comp["mass"] >= mass_range[0])
                         & (comp["mass"] < mass_range[1]))[:n_target]
    n = len(sel)
    sub = {k: np.asarray(v)[sel] for k, v in comp.items()}
    sub["system"] = np.arange(n)
    sub["role"] = np.zeros(n, int)
    sys_sub = {"age_gyr": sub["age_gyr"], "n_components": np.ones(n, int)}
    return sys_sub, sub


def occurrence(pl, n_stars, p_range, r_range):
    m = ((pl["period"] >= p_range[0]) & (pl["period"] < p_range[1]) & (pl["planet_radius"] >= r_range[0])
         & (pl["planet_radius"] < r_range[1]))
    return m.sum() / n_stars


def dc15_grid(pl, n_stars):
    """Model planets per star (percent) on the DC15 grid."""
    h = np.histogram2d(pl["planet_radius"], pl["period"], [md.DC15_R_EDGES, md.DC15_P_EDGES])[0]
    return 100.0 * h / n_stars


def dc15_residuals(grid):
    """Per-cell z-scores in log space; NaN where DC15 has no measurement."""
    obs, up, lo = md.DC15_RATE, md.DC15_ERR_UP, md.DC15_ERR_LO
    model = np.maximum(grid, 1e-3)
    z = np.full(obs.shape, np.nan)
    meas = ~np.isnan(up)
    hi = meas & (model >= obs)
    z[hi] = np.log(model[hi] / obs[hi]) / np.log1p(up[hi] / obs[hi])
    low = meas & (model < obs)
    z[low] = np.log(model[low] / obs[low]) / -np.log1p(-np.minimum(lo[low] / obs[low], 0.9))
    ul = md.DC15_UPPER
    z[ul] = np.maximum(0.0, np.log(model[ul] / obs[ul])) / np.log(2.0)
    return z


def ment_residuals(pl, n_stars):
    """z-scores against Ment & Charbonneau (2023), same convention as :func:`dc15_residuals`."""
    z = []
    for p_r, r_r, val, up, lo in md.MENT2023:
        model = max(occurrence(pl, n_stars, p_r, r_r), 1e-4)
        if not np.isfinite(up):
            z.append(max(0.0, np.log(model / val)) / np.log(2.0))
        elif model >= val:
            z.append(np.log(model / val) / np.log1p(up / val))
        else:
            z.append(np.log(model / val) / -np.log1p(-min(lo / val, 0.9)))
    return np.array(z)


def run(sample, model, seed=1):
    systems, comp = sample
    return attach_planets(systems, comp, seed=seed, mdwarf=model, giants=None, small=None)


def teff_sample(teff_range, n_target=20_000, seed=0):
    """Single hosts (stars or brown dwarfs) in a Teff range."""
    systems, comp = draw_systems(int(n_target * 6), seed=seed)
    sel = np.flatnonzero(hmc_hosts(comp) & (comp["teff"] >= teff_range[0]) & (comp["teff"] < teff_range[1]))[:n_target]
    n = len(sel)
    sub = {k: np.asarray(v)[sel] for k, v in comp.items()}
    sub["system"] = np.arange(n)
    sub["role"] = np.zeros(n, int)
    return {"age_gyr": sub["age_gyr"], "n_components": np.ones(n, int)}, sub


def report_ultracool(model, sample):
    n = len(sample[1]["mass"])
    pl = run(sample, model)
    print(f"\nUltracool dwarfs (Teff {md.SESTOVIC2020_TEFF}, {n} hosts, median {np.median(sample[1]['mass']):.3f} Msun)"
          " vs Sestovic & Demory 2020, P 1-20 d")
    for r_r, val, up, lo, ul84 in md.SESTOVIC2020:
        obs = f"{val} +{up}/-{lo}" if np.isfinite(val) else f"< {ul84} (84%)"
        print(f"  R {r_r}: {obs:<20} model: {occurrence(pl, n, (1.0, 20.0), r_r):.3f}")
    short = pl["period"] < 20.0
    multi = np.bincount(pl["component"][short], minlength=n) >= 3
    print(f"  hosts with >= 3 planets inside 20 d: {multi.mean():.2f} (K2 non-detection allows ~20-30%;"
          " Muirhead+2015: 21% of mid-M dwarfs host compact multis)")


def report(model, early, late):
    n_e, n_l = len(early[1]["mass"]), len(late[1]["mass"])
    pe, pl_ = run(early, model), run(late, model)
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    grid = dc15_grid(pe, n_e)
    z = dc15_residuals(grid)
    print(f"Early M ({n_e} stars, median {np.median(early[1]['mass']):.2f} Msun) vs DC15 Table 4 "
          f"(percent; rows R {md.DC15_R_EDGES.tolist()}, cols P {md.DC15_P_EDGES.tolist()})")
    print("model:\n", grid)
    print("DC15:\n", np.where(md.DC15_UPPER, -md.DC15_RATE, md.DC15_RATE), "  (negative = upper limit)")
    print("z:\n", z, f"\nscore (sum z^2) = {np.nansum(z ** 2):.1f} over {np.isfinite(z).sum()} cells")
    print("DC15 cumulative 1-4 Re, P < 200 d: 2.5 +- 0.2   model:", round(occurrence(pe, n_e, (0.5, 200), (1, 4)), 2))
    for p_r, r_r, dirichlet, uniform in md.HSU2020_INTEGRATED:
        print(f"Hsu+2020 P {p_r} R {r_r}: Dirichlet {dirichlet[0]}, uniform {uniform[0]}   model:",
              round(occurrence(pe, n_e, p_r, r_r), 2))
    print(f"\nMid-to-late M ({n_l} stars, median {np.median(late[1]['mass']):.2f} Msun) vs Ment & Charbonneau 2023")
    for p_r, r_r, val, up, lo in md.MENT2023:
        obs = f"{val} +{up}/-{lo}" if np.isfinite(up) else f"< {val}"
        print(f"  P {p_r} R {r_r}: {obs:<20} model: {occurrence(pl_, n_l, p_r, r_r):.3f}")
    z = ment_residuals(pl_, n_l)
    print("z:", np.round(z, 2), f"\nscore (sum z^2) = {np.sum(z ** 2):.1f} over {len(z)} bins")


EARLY_FIT = ("d_log_rate_clusters", "d_mean_ln_mass", "f_bare", "min_period")
# mass_slope is fixed by prior (universal planet-to-star mass ratio): the Ment & Charbonneau
# score is flat in it. f_bare_late is capped at 0.9 since sub-Neptunes exist around mid-M.
LATE_GRID = {"f_bare_late": (0.6, 0.7, 0.8, 0.9), "min_period_late": (0.6, 0.75, 0.9)}


def fit_early(sample, base: md.MDwarfModel, starts):
    """Powell minimisation of the DC15 score over EARLY_FIT (common random numbers)."""
    from scipy.optimize import minimize

    n = len(sample[1]["mass"])

    def score(x):
        m = replace(base, **dict(zip(EARLY_FIT, map(float, x))))
        if not (0.0 <= m.f_bare <= 1.0 and 0.3 <= m.min_period <= 3.0):
            return 1e9
        return np.nansum(dc15_residuals(dc15_grid(run(sample, m), n)) ** 2)

    results = [minimize(score, x0, method="Powell", options={"xtol": 0.05, "ftol": 0.3, "maxfev": 300})
               for x0 in starts]
    for r in results:
        print("  start ->", np.round(r.x, 2), f"score {r.fun:.1f}")
    best = min(results, key=lambda r: r.fun)
    return replace(base, **{k: round(float(v), 2) for k, v in zip(EARLY_FIT, best.x)})


def fit_late(sample, base: md.MDwarfModel):
    """Grid search of the Ment & Charbonneau score over LATE_GRID."""
    n = len(sample[1]["mass"])
    best = None
    for values in np.array(np.meshgrid(*LATE_GRID.values(), indexing="ij")).reshape(len(LATE_GRID), -1).T:
        m = replace(base, **{k: float(v) for k, v in zip(LATE_GRID, values)})
        score = np.sum(ment_residuals(run(sample, m), n) ** 2)
        if best is None or score < best[0]:
            best = (score, m)
    print(f"  late best score {best[0]:.1f}:", {k: getattr(best[1], k) for k in LATE_GRID})
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--n", type=int, default=20_000)
    args = parser.parse_args()
    early = star_sample((0.35, 0.65), args.n, seed=0)
    late = star_sample((0.1, 0.3), args.n, seed=1)
    model = DEFAULT_MDWARF
    if args.fit:
        # Separate star samples for fitting, so that the report below is out of sample.
        print("Fitting early M to DC15 ...")
        model = fit_early(star_sample((0.35, 0.65), args.n // 2, seed=2), model,
                          starts=[[1.0, 0.5, 0.3, 1.0], [0.8, 0.0, 0.5, 1.5], [1.2, 0.8, 0.2, 0.7]])
        print("Fitting mid-to-late M to Ment & Charbonneau 2023 ...")
        model = fit_late(star_sample((0.1, 0.3), args.n // 2, seed=3), model)
        print(model, "\n")
    report(model, early, late)
    report_ultracool(model, teff_sample(md.SESTOVIC2020_TEFF, args.n, seed=4))


if __name__ == "__main__":
    main()
