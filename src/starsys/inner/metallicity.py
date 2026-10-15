"""Metallicity dependence of the inner (HM-C) planets (backlog B3).

HM-C has no [Fe/H] dependence: it is fitted to the Kepler field as a whole. Petigura+2018
(California-Kepler Survey IV) measure, as df ∝ 10^(beta [Fe/H]):

    planets                P = 1-10 d     P = 10-100 d
    super-Earths 1-1.7 Re  +0.6 +- 0.2    -0.3 +- 0.2
    sub-Neptunes 1.7-4 Re  +1.6 +- 0.3    +0.5 +- 0.2
    sub-Saturns 4-8 Re     +5.5 +- 1.6    +2.1 +- 0.7

The dependence steepens with planet size and is stronger at short periods, as expected if
metal-rich disks build heavier cores that keep their envelopes against photoevaporation. It is
put into HM-C's own parameters, relative to the Kepler field's mean metallicity
(``feh0`` = -0.01, LAMOST; Petigura+2018 Section 2):

- planet masses: ``mean_ln_mass += d_ln_mass * ([Fe/H] - feh0)``;
- period distribution: HM-C's power-law index ``power_law_P += d_alpha_p * ([Fe/H] - feh0)``
  (Petigura+2018 Table 6: above the break, metal-rich sub-Neptunes fall off as P^-0.3, metal-poor
  ones rise as P^+0.1; the break moves inward);
- fraction of stars with planets: f_swpa multiplied by 10^(d_log_fswp ([Fe/H] - feh0)),
  capped at 1. Fitted to 0: the size and period dependence above carries the trend.

Fitted (2026-10-15, out of sample): betas 0.59 / 0.96 / 1.52 at 3-10 d and 0.15 / 0.33 / 1.22
at 10-100 d for super-Earths / sub-Neptunes / sub-Saturns; score 18.9 over the six betas
(no dependence: 31.9). The hot sub-Neptunes (2.1 sigma) and hot sub-Saturns (2.6 sigma, 7
planets) stay shallower than measured, the warm super-Earths steeper (2.3 sigma).

Planets inside 3 d (:mod:`starsys.inner.short`) are drawn from Kepler rates directly, so they take
Petigura's hot betas as weights (:func:`short_weight`).

The mass coefficient is a prior (core mass ∝ solid content ∝ 10^[Fe/H]); the other two are
fitted to the six beta values above (``python -m
starsys.validation.planets --metallicity [--fit]``). Centred on the field mean, the Kepler
averages are kept. The cold small planets beyond 300 d (:mod:`starsys.outer.small`) anchor on
HM-C's edge density without this dependence; giant planets have their own (Fischer & Valenti
2005, :mod:`starsys.outer.giants`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmc import I_ALPHAP, I_FMED, I_FSLOPE, I_MEDCOL, I_MUM

# Petigura+2018 Figure 10 / Table 7: (P range d, R range Re, beta, +err, -err).
PETIGURA2018_BETA = [
    ((1.0, 10.0), (1.0, 1.7), 0.6, 0.2, 0.2),
    ((1.0, 10.0), (1.7, 4.0), 1.6, 0.3, 0.3),
    ((1.0, 10.0), (4.0, 8.0), 5.5, 1.6, 1.5),
    ((10.0, 100.0), (1.0, 1.7), -0.3, 0.2, 0.2),
    ((10.0, 100.0), (1.7, 4.0), 0.5, 0.2, 0.2),
    ((10.0, 100.0), (4.0, 8.0), 2.1, 0.7, 0.7),
]


# Kepler field metallicity distribution (LAMOST; Petigura+2018 Table 4): RMS around the mean.
FEH_SIGMA = 0.207

# Planets inside HM-C's 3 d limit (:mod:`starsys.inner.short`, Hsu+2019 rates): Petigura's hot
# (1-10 d) betas by radius, directly: upper radius edges (Re) and beta (sub-Saturns 4-8 Re,
# Jupiters above).
HOT_BETA_R = np.array([1.7, 4.0, 8.0, np.inf])
HOT_BETA = np.array([0.6, 1.6, 5.5, 3.4])


def short_weight(radius, d_feh) -> np.ndarray:
    """Occurrence weight 10^(beta d_feh) of short-period planets, divided by its mean over the Kepler
    field's [Fe/H] distribution (Gaussian, FEH_SIGMA), so that Kepler-field rates are kept."""
    beta = HOT_BETA[np.searchsorted(HOT_BETA_R, np.asarray(radius, dtype=float))]
    return 10.0 ** (beta * np.asarray(d_feh, dtype=float)) / np.exp(0.5 * (beta * np.log(10.0) * FEH_SIGMA) ** 2)


@dataclass(frozen=True)
class MetallicityModel:
    d_ln_mass: float = 2.3026  # change of mean ln(planet mass) per dex: core mass ∝ 10^[Fe/H] (prior, ln 10)
    d_log_fswp: float = 0.0  # log10 change of f_swpa per dex (fitted: 0)
    d_alpha_p: float = -1.5  # change of HM-C's period power-law index (dN/dlog P ∝ P^alpha) per dex
    feh0: float = -0.01  # Kepler field mean [Fe/H] (LAMOST)


def adjust_params(rows: np.ndarray, feh, colour, model: MetallicityModel) -> np.ndarray:
    """Return a copy of the per-host HM-C parameter rows with the [Fe/H] dependence applied."""
    rows = np.array(rows, dtype=float, copy=True)
    d = np.asarray(feh, dtype=float) - model.feh0
    rows[:, I_MUM] += model.d_ln_mass * d
    rows[:, I_ALPHAP] += model.d_alpha_p * d
    f = np.clip(rows[:, I_FSLOPE] * (colour - rows[:, I_MEDCOL]) + rows[:, I_FMED], 0.0, 1.0)
    f_new = np.clip(f * 10.0 ** (model.d_log_fswp * d), 0.0, 1.0)
    # The linear f_swpa(colour) evaluated at this host's colour becomes f_new (also where it clipped at 1).
    rows[:, I_FMED] = np.where(f_new != f, f_new - rows[:, I_FSLOPE] * (colour - rows[:, I_MEDCOL]), rows[:, I_FMED])
    return rows
