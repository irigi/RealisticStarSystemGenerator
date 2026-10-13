"""Extension of HM-C to M dwarfs and to hosts below 0.1 Msun.

HM-C is fitted to FGK dwarfs (bp-rp <= 1.7, Teff >~ 3950 K on the Kepler scale).
Below that, its architecture (clusters, spacing, AMD, photoevaporation) is kept and
the following change, phased in linearly in Teff between ``teff_hi`` and ``teff_lo``:

- more planets: ``log_rate_clusters + d_log_rate_clusters``;
- planet masses: ``mean_ln_mass + d_mean_ln_mass``;
- a fraction ``f_bare`` of planets has no H/He envelope regardless of irradiation
  (HM-C gives every planet an initial envelope; stripping by irradiation alone cannot
  produce enough 1-1.5 Re planets around M dwarfs);
- the inner period limit drops from 3 d to ``min_period`` (Kepler finds M-dwarf
  planets at 0.5-3 d);
- optionally, stronger envelope stripping: ``log_alpha_pret + d_log_alpha_pret``. In
  HM-C's flux-proportional mass-loss timescale, a lifetime XUV/bolometric fluence
  higher by a factor X equals d_log_alpha_pret = -ln X. M dwarfs stay XUV-saturated
  much longer than Sun-like stars (Jackson+2012), so X > 1 is physically expected, but
  the occurrence data do not constrain it (the fit is flat in it), so the default is 0.

Early M (0.35-0.65 Msun): the first four are fitted to Dressing & Charbonneau (2015)
Table 4 (Kepler, median 0.5 Msun, revised stellar radii). Score (sum of squared
log-space z) 24 over 33 cells, against 70 for plain HM-C. The four parameters are
degenerate (fits from different starts score within noise of each other at different
points); the data pin down the occurrence they produce, not each parameter. Integrated rates fall
between Hsu, Ford & Terrien (2020)'s two priors, close to their baseline (Dirichlet).
f_swp is already 1 at these colours (HM-C's linear f_swp clips at bp-rp ~0.94),
consistent with Hsu+2020's finding that most early M dwarfs host planets.

Mid-to-late M (below ``mass_pivot`` = 0.5 Msun): planet masses scale with stellar mass,
``mean_ln_mass += mass_slope * ln(M / mass_pivot)`` (roughly universal planet-to-star
mass ratio, Pascucci+2018), and the bare fraction and inner period limit move linearly
in mass to ``f_bare_late`` and ``min_period_late`` at ``mass_late`` (constant below).
``f_bare_late`` (capped at 0.9: sub-Neptunes such as GJ 1214 b exist around 0.18 Msun
stars) and ``min_period_late`` are fitted to TESS (Ment & Charbonneau 2023, 0.1-0.3 Msun,
7 planets), with ~2 sigma tensions (too many 0.5-1 Re and 1-2.7 d planets, too
few 1-1.5 Re ones). See ``docs/validation/planets.md``.

Below 0.1 Msun (latest M and brown dwarfs) there are no occurrence measurements. The
fraction of hosts with planets is scaled with disk mass, f = (M / 0.1)^disk_mass_slope
(M_dust ~ M^1.9, Pascucci+2016), and the planets are flagged.

Refit: ``python -m starsys.validation.mdwarf --fit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmc import I_FBARE, I_FMED, I_FSLOPE, I_LALPHA, I_LRC, I_MUM, I_PMIN


@dataclass(frozen=True)
class MDwarfModel:
    # Early M, fitted to DC15.
    d_log_rate_clusters: float = 1.8
    d_mean_ln_mass: float = 0.38
    f_bare: float = 0.26
    min_period: float = 1.51
    d_log_alpha_pret: float = 0.0
    # Mid-to-late M. mass_slope is a prior (the data do not distinguish 0-1.5);
    # f_bare_late and min_period_late are fitted to Ment & Charbonneau (2023).
    mass_pivot: float = 0.5
    mass_slope: float = 1.0
    mass_late: float = 0.15
    f_bare_late: float = 0.9
    min_period_late: float = 0.75
    # Phase-in below the HM-C domain.
    teff_hi: float = 4000.0
    teff_lo: float = 3700.0
    # Below the hydrogen-burning limit region.
    substellar_mass: float = 0.1
    disk_mass_slope: float = 1.9


def mdwarf_weight(teff, model: MDwarfModel) -> np.ndarray:
    """0 in the HM-C domain, 1 for M dwarfs, linear in Teff in between."""
    return np.clip((model.teff_hi - np.asarray(teff, dtype=float)) / (model.teff_hi - model.teff_lo), 0.0, 1.0)


def adjust_params(rows: np.ndarray, mass, teff, model: MDwarfModel) -> np.ndarray:
    """Return a copy of the per-star HM-C parameter rows with the M-dwarf changes applied."""
    rows = np.array(rows, dtype=float, copy=True)
    mass = np.asarray(mass, dtype=float)
    w = mdwarf_weight(teff, model)
    rows[:, I_LRC] += w * model.d_log_rate_clusters
    rows[:, I_LALPHA] += w * model.d_log_alpha_pret
    late = np.clip((model.mass_pivot - mass) / (model.mass_pivot - model.mass_late), 0.0, 1.0)
    rows[:, I_FBARE] = w * (model.f_bare + (model.f_bare_late - model.f_bare) * late)
    p_min = model.min_period + (model.min_period_late - model.min_period) * late
    rows[:, I_PMIN] = rows[:, I_PMIN] + w * (p_min - rows[:, I_PMIN])
    rows[:, I_MUM] += w * model.d_mean_ln_mass + model.mass_slope * np.minimum(0.0, np.log(mass / model.mass_pivot))
    low = mass < model.substellar_mass
    rows[low, I_FSLOPE] = 0.0
    rows[low, I_FMED] = (mass[low] / model.substellar_mass) ** model.disk_mass_slope
    return rows


# --- observational targets --------------------------------------------------------

# Dressing & Charbonneau (2015, ApJ 807, 45) Table 4, lower block (stellar radii revised
# to the Mann+2013 Teff-radius relation). Planets per star in percent.
# Entries: (value, +err, -err); upper limits as (UL, nan, nan) with DC15_UPPER True;
# cells with injection recovery < 15% are NaN.
DC15_P_EDGES = np.array([0.5, 1.7, 5.5, 18.2, 60.3, 200.0])
DC15_R_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
_n = np.nan
DC15_RATE = np.array([
    [1.38, 8.42, 20.59, _n, _n],
    [1.95, 9.94, 26.63, 26.85, 28.85],
    [0.41, 4.15, 23.58, 24.59, 19.98],
    [0.13, 2.72, 18.73, 27.58, 18.08],
    [0.11, 1.59, 8.29, 14.51, 8.61],
    [0.12, 0.65, 3.25, 3.37, 1.97],
    [0.11, 0.38, 1.05, 0.56, 2.34],
])
DC15_ERR_UP = np.array([
    [0.93, 3.53, 8.70, _n, _n],
    [0.93, 2.82, _n, 10.70, 28.66],
    [0.51, 1.94, _n, 9.08, 16.07],
    [_n, 1.73, 5.39, 9.42, 13.21],
    [_n, 1.39, 3.96, 7.71, 9.78],
    [_n, 1.00, 2.72, 4.62, 5.87],
    [_n, 0.77, 1.82, 2.32, _n],
])
DC15_ERR_LO = np.array([
    [0.53, 2.39, 5.57, _n, _n],
    [0.61, 2.13, _n, 6.79, 10.34],
    [0.20, 1.28, _n, 6.00, 7.42],
    [_n, 1.01, 3.95, 6.31, 6.57],
    [_n, 0.69, 2.55, 4.62, 3.84],
    [_n, 0.32, 1.37, 1.62, 0.85],
    [_n, 0.19, 0.52, 0.19, _n],
])
DC15_UPPER = np.isnan(DC15_ERR_UP) & ~np.isnan(DC15_RATE)
DC15_MEDIAN_MASS = 0.50

# Hsu, Ford & Terrien (2020, MNRAS 498, 2249) integrated rates for Kepler early M
# dwarfs (planets per star): (P range d, R range Re, Dirichlet prior, uniform prior).
HSU2020_INTEGRATED = [
    ((2.0, 32.0), (1.0, 2.5), (0.9, 0.2, 0.1), (1.6, 0.3, 0.2)),
    ((0.5, 256.0), (0.5, 4.0), (4.2, 0.6, 0.6), (8.4, 1.2, 1.1)),
]

# Ment & Charbonneau (2023, AJ 165, 265) Table 7, mid-to-late M dwarfs (0.1-0.3 Msun,
# median 0.17): planets per star, (P range, R range, value, +err, -err); value with
# nan errors is a 1-sigma upper limit.
MENT2023 = [
    ((0.5, 7.0), (0.5, 4.0), 0.61, 0.24, 0.19),
    ((0.5, 7.0), (0.5, 1.0), 0.134, 0.123, 0.072),
    ((0.5, 7.0), (1.0, 1.5), 0.446, 0.194, 0.143),
    ((0.5, 7.0), (1.5, 2.0), 0.073, _n, _n),
    ((0.5, 7.0), (2.0, 99.0), 0.072, _n, _n),
    ((0.5, 1.0), (0.5, 4.0), 0.047, 0.019, 0.015),
    ((1.0, 2.7), (0.5, 4.0), 0.159, 0.064, 0.050),
    ((2.7, 7.0), (0.5, 4.0), 0.402, 0.162, 0.127),
]
MENT2023_MASS_RANGE = (0.1, 0.3)

# Sestovic & Demory (2020, A&A 641, A170): K2, 702 ultracool dwarfs (M6 and later),
# one detection. Planets per star at P = 1-20 d (uniform in log P):
# (R range, value, +err, -err, 84% upper limit).
SESTOVIC2020 = [
    ((1.0, 2.0), _n, _n, _n, 0.66),
    ((2.0, 4.0), 0.20, 0.16, 0.11, 0.36),
    ((4.0, 6.0), _n, _n, _n, 0.21),
]
SESTOVIC2020_TEFF = (2300.0, 3000.0)  # our selection standing in for "ultracool, M6-L"
