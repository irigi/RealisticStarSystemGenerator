"""Planets inside HM-C's inner period limit (0.5 d to P_min = 3 d for FGK stars).

HM-C starts at 3 d, so FGK stars had no planets on the shortest orbits, including the
ultra-short-period planets (USPs, P < 1 d) found around ~0.5% of Sun-like stars. This
component fills 0.5 d to the host's HM-C period limit:

- **Occurrence and sizes:** Hsu+2019 (Kepler DR25 + Gaia DR2, FGK stars with 0.5 <= bp-rp <=
  1.7, the same stellar sample as HM-C; Table 2, combined detection and vetting
  efficiency), period bins 0.5-1, 1-2 and 2-4 d, radius bins 0.5-16 Earth radii (upper
  limits taken as zero). At 2-3 d only what HM-C's planets at 3-4 d leave of Hsu's 2-4 d bin is
  added (``SHORT_RATES``). Over 0.5-3 d this is ~0.05 planets per FGK star, mostly below 2
  Earth radii (the hot-Neptune desert is in the data).
- **Stellar type, P < 1 d:** Sanchis-Ojeda+2014 find USPs (P < 1 d) around 0.15 +- 0.05% of F,
  0.51 +- 0.07% of G, 0.83 +- 0.18% of K and 1.10 +- 0.40% of M dwarfs. The 0.5-1 d rate is
  scaled by their trend (steps over their Teff ranges) divided by its mean over Hsu's Kepler
  targets (0.511), so the Kepler-sample average stays Hsu's. The 1-3 d rates are the FGK
  average.
- **M dwarfs:** the M-dwarf extension of HM-C already reaches 0.75-1.5 d and was fitted to
  DC15 and Ment+2023 including their shortest periods, so this component is phased out with
  the extension's weight (zero below 3700 K).
- **Masses:** rocky below 1.6 Earth radii (Chen & Kipping 2017 Terran, R = M^0.279; most
  USPs are bare rocky cores), Neptunian above (R = 0.808 M^0.589); envelopes counted as lost
  below 1.6 Earth radii.
- **Orbits:** circular (tidally damped). Mutual inclinations to the inner system's plane are
  Rayleigh(7 deg): USPs are more inclined to their outer companions than planets in compact
  multis (Dai, Masuda & Winn 2018).
- **Placement:** a planet must keep HM-C's spacing (``num_mutual_hill_radii`` mutual Hill
  radii) from the host's innermost HM-C planet and from other short-period planets, and lie
  outside the Roche-limit proxy of 2 stellar radii; otherwise it is dropped (the data are
  occurrences over all stars, so dropping slightly undercounts: checked in
  :mod:`starsys.validation.short`).

As a transit-survey rate this is a field average over binaries and gets the same
single-star boost as HM-C when the binary model is on. Planets are flagged ``short_period``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..units import EARTH_MASS_MSUN, period_to_a_au

RSUN_AU = 1.0 / 215.032

# Hsu+2019 Table 2 (combined detection & vetting efficiency), planets per star per bin.
HSU2019_R_EDGES = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
HSU2019_P_EDGES = np.array([0.5, 1.0, 2.0, 4.0])
HSU2019_RATES = np.array([
    # 0.5-1 d
    [2.5e-3, 2.70e-3, 1.7e-3, 1.57e-3, 8.3e-4, 2.7e-4, 0.0, 0.0, 0.0, 2.9e-4, 0.0, 0.0, 0.0],
    # 1-2 d
    [2.6e-3, 2.9e-3, 1.03e-3, 2.2e-3, 9.7e-4, 6.4e-4, 2.3e-4, 2.7e-4, 0.0, 0.0, 0.0, 2.5e-4, 4.1e-4],
    # 2-4 d
    [2.65e-2, 1.20e-2, 2.5e-3, 9.2e-3, 5.6e-3, 2.9e-3, 2.46e-3, 7.5e-4, 1.28e-3, 6.1e-4, 0.0, 9.0e-4, 1.62e-3],
])
# Rates used below 3 d. 0.5-1 and 1-2 d: Hsu's. 2-4 d (used over 2-3 d, a fraction ln 1.5 /
# ln 2 = 0.585 of the bin): what Hsu's 2-4 d rate leaves after the planets HM-C already puts at
# 3-4 d (mock Kepler sample of the population, posterior-averaged), divided by 0.585, so that the
# 2-4 d bin is matched as a whole. HM-C alone exceeds Hsu at 3-4 d for 1-1.25 and 2-6 Earth
# radii; there nothing is added.
SHORT_RATES = HSU2019_RATES.copy()
SHORT_RATES[2] = np.array([32.15, 5.24, 0.0, 4.29, 2.14, 0.79, 0.0, 0.0, 0.0, 0.0, 0.0, 1.25, 2.77]) * 1e-3
# Sanchis-Ojeda+2014: USP (P < 1 d) occurrence (%) in their Teff ranges (F, G, K, M; steps,
# clamped outside 3100-7100 K).
SO2014_EDGES = np.array([3100.0, 4100.0, 5100.0, 6100.0, 7100.0])
SO2014_RATE = np.array([1.10, 0.83, 0.51, 0.15])
SO2014_KEPLER_MEAN = 0.511  # mean of the trend over Hsu's Kepler targets


@dataclass(frozen=True)
class ShortPeriodModel:
    p_min: float = 0.5  # days
    rocky_radius: float = 1.6  # Earth radii
    incl_sigma_deg: float = 7.0
    roche_stellar_radii: float = 2.0


def teff_scale(teff) -> np.ndarray:
    """USP occurrence relative to Hsu+2019's Kepler average (Sanchis-Ojeda+2014 trend)."""
    b = np.clip(np.digitize(np.asarray(teff, dtype=float), SO2014_EDGES) - 1, 0, len(SO2014_RATE) - 1)
    return SO2014_RATE[b] / SO2014_KEPLER_MEAN


def mass_from_radius(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.where(r < 1.6, r ** (1.0 / 0.279), (r / 0.808) ** (1.0 / 0.589))


def expected_number(p_max, teff, weight=1.0) -> np.ndarray:
    """Expected short-period planets per star between 0.5 d and ``p_max`` (per host)."""
    p_max = np.asarray(p_max, dtype=float)
    tot = SHORT_RATES.sum(axis=1)
    lo, hi = HSU2019_P_EDGES[:-1], HSU2019_P_EDGES[1:]
    frac = np.clip((np.log(np.maximum(p_max, 1e-9))[:, None] - np.log(lo)) / np.log(hi / lo), 0.0, 1.0)
    per_bin = frac * tot
    per_bin[:, 0] *= teff_scale(teff)
    return per_bin.sum(axis=1) * weight


def draw_short(mstar, rstar, teff, p_max, weight, rng, model: ShortPeriodModel = ShortPeriodModel()):
    """Candidates (unplaced): per-planet ``host``, ``period``, ``planet_radius``, ``planet_mass``.

    ``p_max``: per-host upper period (the host's HM-C P_min); ``weight``: per-host multiplier
    (1 - M-dwarf weight, times the single-star boost)."""
    mstar = np.asarray(mstar, dtype=float)
    n = len(mstar)
    lo, hi = HSU2019_P_EDGES[:-1], HSU2019_P_EDGES[1:]
    hosts, periods, radii = [], [], []
    r_cdf = np.cumsum(SHORT_RATES, axis=1)
    for b in range(len(lo)):
        top = np.minimum(p_max, hi[b])
        frac = np.clip(np.log(np.maximum(top, 1e-9) / lo[b]) / math.log(hi[b] / lo[b]), 0.0, 1.0)
        lam = frac * SHORT_RATES[b].sum() * np.asarray(weight, dtype=float)
        if b == 0:
            lam = lam * teff_scale(teff)
        h = np.repeat(np.arange(n), rng.poisson(lam))
        if not len(h):
            continue
        # Log-uniform within the bin, below the host's limit.
        p = np.exp(np.log(lo[b]) + rng.random(len(h)) * (np.log(top[h]) - np.log(lo[b])))
        k = np.searchsorted(r_cdf[b], rng.random(len(h)) * r_cdf[b, -1], side="right")
        k = np.minimum(k, len(HSU2019_R_EDGES) - 2)
        r = np.exp(np.log(HSU2019_R_EDGES[k]) + rng.random(len(h)) * np.log(HSU2019_R_EDGES[k + 1] /
                                                                              HSU2019_R_EDGES[k]))
        hosts.append(h)
        periods.append(p)
        radii.append(r)
    if not hosts:
        e = np.zeros(0)
        return {"host": np.zeros(0, dtype=np.int64), "period": e, "planet_radius": e, "planet_mass": e}
    host, period, radius = np.concatenate(hosts), np.concatenate(periods), np.concatenate(radii)
    # Outside ~2 stellar radii.
    ok = period_to_a_au(period, mstar[host]) > model.roche_stellar_radii * np.asarray(rstar, dtype=float)[host] * RSUN_AU
    host, period, radius = host[ok], period[ok], radius[ok]
    order = np.lexsort((period, host))
    return {"host": host[order], "period": period[order], "planet_radius": radius[order],
            "planet_mass": mass_from_radius(radius[order])}


def place(cand, inner_host, inner_period, inner_mass, mstar, n_hill):
    """Mask of candidates kept: each must be ``n_hill`` mutual Hill radii from the host's
    innermost inner planet and from the short-period planets kept before it (inside out)."""
    keep = np.zeros(len(cand["host"]), dtype=bool)
    first = {}
    for j in range(len(inner_host) - 1, -1, -1):  # innermost inner planet per host (sorted by period)
        first[int(inner_host[j])] = j
    last_p, last_m, last_h = None, None, -1
    for i in range(len(cand["host"])):
        h = int(cand["host"][i])
        p, m = cand["period"][i], cand["planet_mass"][i]
        ms = mstar[h]
        a = period_to_a_au(p, ms)

        def spaced(p2, m2):
            a2 = period_to_a_au(p2, ms)
            r_h = ((m + m2) * EARTH_MASS_MSUN / (3.0 * ms)) ** (1.0 / 3.0) * 0.5 * (a + a2)
            return abs(a2 - a) >= n_hill[h] * r_h

        ok = True
        if h == last_h and last_p is not None and not spaced(last_p, last_m):
            ok = False
        j = first.get(h)
        if ok and j is not None and not spaced(inner_period[j], inner_mass[j]):
            ok = False
        if ok:
            keep[i] = True
            last_p, last_m, last_h = p, m, h
    return keep

