"""Masses of HM-C's large planets, rescaled to radial-velocity measurements (backlog B9).

HM-C (:mod:`starsys.inner.hmc`) draws masses and gets radii from a fitted mass-radius
relation; Kepler constrains only the radii. Above ~4 Earth radii the relation flattens, so
the radii say little about the masses, and HM-C's masses come out 1.6-3x heavier than
measured (FGK hosts, P < 300 d, planets that keep their envelopes):

    R (Re)        4-4.5  4.5-5  5-6  6-7  7-8  8-9  9-10
    HM-C median    34     48    63   86   110  145  164   (Earth masses)
    NASA archive   22     29    25   34    37   71   99   (RV masses with < 50% errors)
    Otegi+2020     17     20    26   33    42   51   61   (M = 1.74 R^1.58, volatile-rich)

The archive is biased to heavy planets (they are easier to measure), so it bounds the true
median from above. The target median is Otegi+2020's volatile-rich relation up to 8 Earth
radii (valid below 120 Earth masses), then log-linear to Saturn (95 Earth masses at 9.45
Earth radii, also the archive median at 9-10 Earth radii). Each mass is multiplied by
c(R) = min(1, target(R) / HM-C median(R)), interpolated in R between bin centres. This keeps
HM-C's own scatter at fixed radius (0.5-1.0 in ln M, like the archive's 0.6-1.0) and every
planet's rank, and only lowers masses, so HM-C's stability conditioning still holds. Radii,
and with them the Kepler fit, are unchanged; stripped cores are not touched.
"""

from __future__ import annotations

import numpy as np

# HM-C median mass (Earth masses) of planets keeping their envelopes, by radius (bin centres),
# measured on the population (FGK hosts, P < 300 d, all posterior draws).
_R = np.array([3.5, 4.25, 4.75, 5.5, 6.5, 7.5, 8.5, 9.5])
_HMC_MEDIAN = np.array([10.0, 33.5, 47.9, 63.3, 85.8, 110.2, 144.7, 164.2])
SATURN = (9.45, 95.2)


def target_mass(radius) -> np.ndarray:
    """Median mass (Earth masses) at a given radius (Earth radii): Otegi+2020 to 8 Re, then towards Saturn."""
    r = np.asarray(radius, dtype=float)
    otegi = 1.74 * r**1.58
    m8 = 1.74 * 8.0**1.58
    slope = np.log(SATURN[1] / m8) / np.log(SATURN[0] / 8.0)
    return np.where(r <= 8.0, otegi, m8 * (r / 8.0) ** slope)


def correction(radius) -> np.ndarray:
    """Mass factor c(R) <= 1 (1 below 3.5 Earth radii)."""
    c = np.minimum(1.0, target_mass(_R) / _HMC_MEDIAN)
    return np.interp(np.asarray(radius, dtype=float), _R, c, left=1.0, right=c[-1])

