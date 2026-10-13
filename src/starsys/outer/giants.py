"""Giant planets (M >= 30 Earth masses) from 0.75 d to 100 au.

Three components, drawn per host star:

**Warm and cold giants** (P >= 10 d). Occurrence density per star

    d2N / (d ln a  d ln M) = D * h(M*, [Fe/H]) * s(a) * q(M)

- s(a) = a^beta * (1 - exp(-(a / a0)^gamma)): broken power law of Fulton+2021 (California
  Legacy Survey, CLS), Table 1: beta = -0.86, a0 = 3.6 au, gamma = 1.59, fitted to
  0.03-30 au, and used up to 30 au (wider orbits: third component).
- q(M): flat in ln M from 30 to 900 Earth masses (Fulton+2021 Fig. 6), then
  (M / 900)^``mass_slope_high`` up to 6000 Earth masses. The slope -1 reproduces the CLS
  1-5 au counts above 300 Earth masses (CLS is near complete there): see
  :mod:`starsys.validation.giants`. Masses are true masses; the CLS functions are in
  M sin i, which has the same shape for a power law, and the validation compares in
  M sin i using each planet's sky inclination.
- h(M*, [Fe/H]) = 10^(beta_feh [Fe/H]) * g(M*): g = M*^1 (Johnson+2010, 0.2-2 Msun
  main-sequence and subgiant hosts) up to the occurrence peak at 1.68 Msun, then a
  Gaussian drop of width 0.70 Msun (Wolthoff+2022, evolved hosts of 0.8-3 Msun; giant
  planets around stars above ~2.7 Msun are rare, Reffert+2015). beta_feh = 1.2
  (Johnson+2010, fitted jointly with stellar mass). No giants below 0.1 Msun.
- D and the per-star dispersion are fitted jointly to CLS-based rates on the CLS stellar
  sample (Rosenthal+2021, 719 stars): Fulton+2021 (2-8 and 8-32 au), Rosenthal+2022
  (fraction of stars with a giant), the near-complete raw CLS counts above 300 Earth
  masses, and the fraction of multi-giant hosts. See :mod:`starsys.validation.giants`.
- Number per star: Poisson with a mean multiplied by a Gamma(k, 1/k) factor per star
  (``dispersion`` = k). Giants cluster in systems beyond what metallicity explains
  (CLS: 31 of 98 giant hosts have two or more).

**Hot Jupiters** (0.75 <= P < 10 d, M >= 95 Earth masses). Rate per star
``hj_rate`` * k(M*) * 10^(beta_feh [Fe/H]), at most one per star; k(M*) from the TESS
rates of Beleznay & Kunimoto (2022) for A/F/G dwarfs and Gan+2023 for early M dwarfs,
relative to G dwarfs. ``hj_rate`` is set so that the CLS stellar sample gets 1.0 per 100
stars (RV: Mayor+2011 0.89 +- 0.36 %, Wright+2012 1.2 +- 0.4 %; TESS, single G stars:
0.98 +- 0.36 %). Period: lognormal, median 4.4 d, sigma_lnP 0.47 (RV hot Jupiters in the
NASA archive, and Kepler/TESS ones weighted by inverse transit probability, agree).
Hot Jupiters are circular (Dawson & Johnson 2018).

**Wide giants** (direct imaging, 10-100 au, 2-13 MJup), from the GPIES fit to all
stars of Nielsen+2019 (Table 3): d2N / (dm da) ~ m^-2.28 a^-1.99, with 3.5% of
1.75 Msun stars hosting a 5-13 MJup planet at 10-100 au, scaled as M*^2.0. No
metallicity dependence is assumed. At 10-30 au this overlaps the CLS component; their
sum stays within the direct-imaging errors (checked in the validation). The M*^2 scaling
is held constant above ``wide_mass_cap`` = 2.5 Msun (few GPIES stars are more massive;
hosts above 3 Msun are flagged).

Architecture:

- Hosts of a hot Jupiter get ``hj_outer_boost`` times the mean number of warm/cold
  giants (Poisson, without the per-star dispersion), calibrated to Knutson+2014 (51 +- 10 % of hot-Jupiter hosts have a 1-13 MJup companion at 1-20 au).
- Eccentricity: the 121 CLS giants beyond 0.1 au follow Beta(0.89, 2.49) (no significant
  difference inside and outside 1 au, KS p = 0.5; consistent with the Kipping 2013 global
  fit, p = 0.3). Because the stability conditioning against inner planets lowers giant
  eccentricities (below), the prior ``ecc_beta`` = Beta(0.97, 2.36) is fitted so that the
  conditioned eccentricities of giants around Sun-like stars match the CLS mean and
  median.
- Inclination relative to the inner (HM-C) system's invariable plane: Rayleigh,
  sigma = 11.8 deg (Masuda+2020, cold Jupiters vs inner super-Earths). Without an
  inner system the reference plane is isotropic.
- No correlation with the presence of inner small planets beyond what metallicity
  implies: the largest samples find none at average metallicity (Bonomo+2025; Van
  Zandt & Petigura 2024; Ruggieri+2026), against earlier claims (Zhu & Wu 2018;
  Bryan & Lee 2024).
- Stability (:func:`pair_stable`): an adjacent pair is accepted if it is AMD-Hill stable
  (Petit+2018, the criterion of the HM-C port) or satisfies the N-body-calibrated
  criterion of Petrovich (2015) for eccentric two-planet systems. Both are sufficient
  conditions that fail in different regimes: AMD-Hill rejects a small planet next to any
  eccentric giant (its critical AMD vanishes with the mass ratio), Petrovich rejects
  some observed giant pairs. Of the 62 adjacent pairs in CLS multi-planet systems, 95%
  pass AMD-Hill, 84% Petrovich, 95% either (the failures are resonant pairs).
- Giants are placed in order of decreasing mass; each must be stable with the nearest
  accepted giant on each side, otherwise its orbit (not its mass) is redrawn from its
  component's distribution, up to ``n_redraw`` times, and then it is dropped.
- Inner planets: :func:`inner_conflicts` finds inner planets unstable next to their
  nearest giants. :func:`starsys.planets.attach_planets` then alternately redraws the
  eccentricities of that host's giants (keeping a and M; :func:`redraw_eccentricities`)
  and the inner system. Giants with close inner companions thus end up on lower
  eccentricities, as observed (Dong+2014 for warm Jupiters; Bonomo+2025: inner
  multiplicity depends on the cold Jupiter's eccentricity). Semi-major axes are not
  redrawn: conditioning them on HM-C's densely filled 3-300 d range would push warm
  giants (0.1-1 au) out and cut their rate by ~3x below CLS.
- Hot-Jupiter hosts keep their inner planets with probability ``hj_keep_inner`` (Wu+2023:
  12 +- 6 % of hot Jupiters have nearby companions).
- Overlap with HM-C: inside the host's HM-C period window (P_min-300 d), giant-model
  planets below HM-C's maximum radius (10 Earth radii) are not drawn, as HM-C covers them.

Radius: cold giants follow a mass-radius relation through Neptune, Saturn and
Jupiter, R = 3.9 (M/17)^0.514 below ~140 Earth masses and R = 11.2 (M/318)^-0.04 above.
Hot giants above 150 Earth masses are inflated per Weiss+2013 (R = 2.45 M^-0.039
F^0.094, cgs flux), whichever is larger, with 5% lognormal scatter.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..inner.hmc import relative_amd_collision, relative_amd_hill
from ..units import EARTH_MASS_MSUN, a_au_to_period, period_to_a_au

SOLAR_FLUX_CGS = 1.361e6  # erg s^-1 cm^-2 at 1 au
COLD, HOT, WIDE = 0, 1, 2
COMPONENT_NAMES = {COLD: "giant", HOT: "hot_jupiter", WIDE: "wide_giant"}


@dataclass(frozen=True)
class GiantModel:
    # Separation (Fulton+2021 Table 1).
    beta: float = -0.86
    a0: float = 3.6
    gamma: float = 1.59
    a_max: float = 30.0
    p_min_cold: float = 10.0
    # Mass function (Earth masses).
    m_min: float = 30.0
    m_break: float = 900.0
    mass_slope_high: float = -1.0
    m_max: float = 6000.0
    # Host scaling.
    norm: float = 0.106  # D, per star per ln a per ln M (calibrated; see module docstring)
    dispersion: float = 1.0  # Gamma shape k of the per-star rate factor (inf = Poisson); calibrated
    mass_slope: float = 1.0
    mass_peak: float = 1.68
    mass_width: float = 0.70
    feh_slope: float = 1.2
    min_host_mass: float = 0.1
    host_mass_range: tuple = (0.2, 3.0)  # calibrated range; outside is flagged
    # Hot Jupiters.
    hj_rate: float = 0.01096  # per star at 1 Msun, [Fe/H] = 0 (calibrated)
    hj_m_min: float = 95.0
    hj_p_median: float = 4.4
    hj_p_sigma: float = 0.47
    hj_p_range: tuple = (0.75, 10.0)
    hj_keep_inner: float = 0.12
    hj_outer_boost: float = 5.21  # calibrated to Knutson+2014
    # Wide giants (Nielsen+2019, all-star fit).
    wide_rate: float = 0.035  # 5-13 MJup, 10-100 au, at wide_ref_mass
    wide_ref_mass: float = 1.75
    wide_mass_slope: float = 2.0
    wide_mass_cap: float = 2.5
    wide_m_index: float = -2.28  # dN/dm
    wide_a_index: float = -1.99  # dN/da
    wide_m_range: tuple = (2.0 * 317.83, 13.0 * 317.83)
    wide_a_range: tuple = (10.0, 100.0)
    # Orbits.
    ecc_beta: tuple = (0.967, 2.355)  # prior; conditioned on the inner system it matches CLS
    incl_sigma_deg: float = 11.8


# Hot-Jupiter rate relative to G dwarfs vs stellar mass: TESS A 0.29, F 0.36, G 0.55 %
# (Beleznay & Kunimoto 2022, 0.8-2.3 Msun); early M 0.27 % (Gan+2023, 0.45-0.65 Msun).
HJ_MASS_KNOTS = np.array([0.1, 0.55, 1.0, 1.3, 1.9, 2.3])
HJ_MASS_FACTOR = np.array([0.0, 0.49, 1.0, 0.65, 0.53, 0.53])


def host_scaling(mass, feh, model: GiantModel = GiantModel()) -> np.ndarray:
    """h(M*, [Fe/H]): giant planet occurrence relative to a solar-mass, solar-metallicity star."""
    mass = np.asarray(mass, dtype=float)
    g = np.where(mass <= model.mass_peak, np.maximum(mass, 0.0) ** model.mass_slope,
                 model.mass_peak ** model.mass_slope
                 * np.exp(-0.5 * ((mass - model.mass_peak) / model.mass_width) ** 2))
    g = np.where(mass < model.min_host_mass, 0.0, g)
    return g * 10.0 ** (model.feh_slope * np.asarray(feh, dtype=float))


def hj_scaling(mass, feh, model: GiantModel = GiantModel()) -> np.ndarray:
    k = np.interp(np.asarray(mass, dtype=float), HJ_MASS_KNOTS, HJ_MASS_FACTOR)
    return k * 10.0 ** (model.feh_slope * np.asarray(feh, dtype=float))


# --- separation and mass distributions ---------------------------------------------

_LNA = np.linspace(math.log(1e-3), math.log(1e3), 4001)


def separation_density(a, model: GiantModel = GiantModel()) -> np.ndarray:
    """s(a), per ln a (Fulton+2021 eq. 5 without the normalisation)."""
    a = np.asarray(a, dtype=float)
    return a**model.beta * -np.expm1(-((a / model.a0) ** model.gamma))


@lru_cache(maxsize=8)
def _separation_cdf(model: GiantModel) -> np.ndarray:
    s = separation_density(np.exp(_LNA), model)
    return np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(_LNA))])


def separation_integral(a_lo, a_hi, model: GiantModel = GiantModel()) -> np.ndarray:
    """Integral of s(a) d ln a between a_lo and a_hi (arrays allowed)."""
    cdf = _separation_cdf(model)
    return np.interp(np.log(a_hi), _LNA, cdf) - np.interp(np.log(a_lo), _LNA, cdf)


def mass_integral(m_lo, m_hi, model: GiantModel = GiantModel()) -> np.ndarray:
    """Integral of q(M) d ln M between m_lo and m_hi (clipped to the model's range)."""
    m_lo = np.clip(np.asarray(m_lo, dtype=float), model.m_min, model.m_max)
    m_hi = np.clip(np.asarray(m_hi, dtype=float), model.m_min, model.m_max)
    s, mb = model.mass_slope_high, model.m_break

    def prim(m):  # primitive of q in ln M, zero at m_min
        flat = np.log(np.minimum(m, mb) / model.m_min)
        x = np.maximum(m, mb) / mb
        high = np.log(x) if s == 0 else (x**s - 1.0) / s
        return flat + high

    return np.maximum(prim(m_hi) - prim(m_lo), 0.0)


def draw_mass(n, rng, model: GiantModel = GiantModel(), m_lo=None) -> np.ndarray:
    """Masses (Earth masses) from q(M) between max(m_lo, m_min) and m_max."""
    m_lo = np.full(n, model.m_min) if m_lo is None else np.broadcast_to(m_lo, (n,))
    lo = mass_integral(model.m_min, m_lo, model)
    total = mass_integral(model.m_min, model.m_max, model)
    target = lo + rng.random(n) * (total - lo)
    flat_total = math.log(model.m_break / model.m_min)
    s, mb = model.mass_slope_high, model.m_break
    out = np.where(target <= flat_total, model.m_min * np.exp(target), np.nan)
    hi = target > flat_total
    y = target[hi] - flat_total
    out[hi] = mb * (np.exp(y) if s == 0 else (1.0 + s * y) ** (1.0 / s))
    return out


def draw_separation(a_lo, a_hi, rng, model: GiantModel = GiantModel()) -> np.ndarray:
    cdf = _separation_cdf(model)
    c_lo = np.interp(np.log(a_lo), _LNA, cdf)
    c_hi = np.interp(np.log(a_hi), _LNA, cdf)
    u = c_lo + rng.random(len(c_lo)) * (c_hi - c_lo)
    return np.exp(np.interp(u, cdf, _LNA))


def _draw_power(n, index, lo, hi, rng):
    """x in [lo, hi] with dN/dx ~ x^index."""
    k = index + 1.0
    u = rng.random(n)
    if abs(k) < 1e-12:
        return lo * (hi / lo) ** u
    return (lo**k + u * (hi**k - lo**k)) ** (1.0 / k)


def _power_fraction(index, lo, hi, sub_lo, sub_hi):
    k = index + 1.0
    if abs(k) < 1e-12:
        return math.log(sub_hi / sub_lo) / math.log(hi / lo)
    return (sub_hi**k - sub_lo**k) / (hi**k - lo**k)


def expected_wide(mass, model: GiantModel = GiantModel()) -> np.ndarray:
    """Expected number of wide (direct-imaging) giants per star over the component's full range."""
    frac = (_power_fraction(model.wide_m_index, *model.wide_m_range, 5 * 317.83, 13 * 317.83)
            * _power_fraction(model.wide_a_index, *model.wide_a_range, 10.0, 100.0))
    mass = np.asarray(mass, dtype=float)
    scaled = np.clip(mass, 0.0, model.wide_mass_cap) / model.wide_ref_mass
    lam = model.wide_rate / frac * scaled**model.wide_mass_slope
    return np.where(mass < model.min_host_mass, 0.0, lam)


def giant_radius(mass, flux_cgs, rng) -> np.ndarray:
    """Radius (Earth radii) of a giant of the given mass (Earth masses) and bolometric flux."""
    mass = np.asarray(mass, dtype=float)
    flux_cgs = np.asarray(flux_cgs, dtype=float)
    # The two branches cross at 141 Earth masses.
    cold = np.minimum(3.9 * (mass / 17.0) ** 0.514, 11.2 * (mass / 318.0) ** -0.04)
    hot = 2.45 * mass**-0.039 * flux_cgs**0.094
    r = np.where(mass >= 150.0, np.maximum(cold, hot), cold)
    return r * np.exp(0.05 * rng.normal(size=len(r)))


def _giant_radius_scalar(m, flux_cgs, rng) -> float:
    """:func:`giant_radius` for one planet."""
    r = min(3.9 * (m / 17.0) ** 0.514, 11.2 * (m / 318.0) ** -0.04)
    if m >= 150.0:
        r = max(r, 2.45 * m**-0.039 * flux_cgs**0.094)
    return r * math.exp(0.05 * rng.normal())


# --- stability ------------------------------------------------------------------------

def pair_stable(m1, m2, a1, e1, a2, e2, mstar) -> bool:
    """Stability of an adjacent pair (a1 < a2; masses in Earth masses): AMD-Hill or Petrovich (2015)."""
    mu1, mu2 = m1 * EARTH_MASS_MSUN / mstar, m2 * EARTH_MASS_MSUN / mstar
    # Petrovich (2015) eq. (1): a_out (1 - e_out) / [a_in (1 + e_in)] > 2.4 max(mu)^(1/3) (a_out/a_in)^(1/2) + 1.15
    if a2 * (1.0 - e2) > a1 * (1.0 + e1) * (2.4 * max(mu1, mu2) ** (1.0 / 3.0) * math.sqrt(a2 / a1) + 1.15):
        return True
    # AMD-Hill (coplanar AMD; Petit+2018, as in the HM-C port).
    l1, l2 = mu1 * math.sqrt(a1), mu2 * math.sqrt(a2)
    amd = l1 * (1.0 - math.sqrt(1.0 - e1 * e1)) + l2 * (1.0 - math.sqrt(1.0 - e2 * e2))
    crit = l2 * min(relative_amd_hill(mu1, mu2, a1, a2), relative_amd_collision(mu1, mu2, a1, a2))
    return amd < crit


def _stable_with_neighbours(a, m, e, occ_a, occ_m, occ_e, mstar) -> bool:
    """Is a body (a, m, e) stable with the nearest occupied orbit on each side? (``occ_a`` sorted)"""
    if not len(occ_a):
        return True
    j = bisect.bisect_left(occ_a, a)
    if j > 0 and not pair_stable(occ_m[j - 1], m, occ_a[j - 1], occ_e[j - 1], a, e, mstar):
        return False
    if j < len(occ_a) and not pair_stable(m, occ_m[j], a, e, occ_a[j], occ_e[j], mstar):
        return False
    return True


# --- orientation ----------------------------------------------------------------------

def _random_unit_vectors(n, rng):
    z = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    s = np.sqrt(1.0 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


def orbit_normals(n_ref, incl, node):
    """Unit normals of orbits inclined by ``incl`` with ascending node ``node`` w.r.t. planes ``n_ref``.

    Same construction as the HM-C port (``hmc._orbit_normal``), vectorised.
    """
    t = np.where(np.abs(n_ref[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(t, n_ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n_ref, u)
    node_dir = np.cos(node)[:, None] * u + np.sin(node)[:, None] * v
    perp = np.cross(n_ref, node_dir)
    return np.cos(incl)[:, None] * n_ref - np.sin(incl)[:, None] * perp


# --- the generator ---------------------------------------------------------------------

def expected_cold(mass, feh, model: GiantModel = GiantModel()) -> np.ndarray:
    """Expected number of warm/cold giants per star (P >= p_min_cold, a <= a_max), before dispersion."""
    a_lo = period_to_a_au(model.p_min_cold, np.asarray(mass, dtype=float))
    return (model.norm * host_scaling(mass, feh, model) * separation_integral(a_lo, model.a_max, model)
            * mass_integral(model.m_min, model.m_max, model))


def _draw_orbit(component, mstar, rng, model: GiantModel):
    """One orbit (a, e) for a giant of the given component around a star of mass mstar."""
    if component == HOT:
        lo, hi = model.hj_p_range
        while True:
            p = model.hj_p_median * math.exp(model.hj_p_sigma * rng.normal())
            if lo <= p < hi:
                return float(period_to_a_au(p, mstar)), 0.0
    e = min(float(rng.beta(*model.ecc_beta)), 0.95)
    if component == COLD:
        cdf = _separation_cdf(model)
        c_lo = np.interp(math.log(period_to_a_au(model.p_min_cold, mstar)), _LNA, cdf)
        c_hi = np.interp(math.log(model.a_max), _LNA, cdf)
        return math.exp(np.interp(c_lo + rng.random() * (c_hi - c_lo), cdf, _LNA)), e
    return float(_draw_power(1, model.wide_a_index, *model.wide_a_range, rng)[0]), e


def draw_giants(mass, feh, luminosity, rng, model: GiantModel = GiantModel(), planes=None, window=None,
                n_redraw: int = 20) -> dict[str, np.ndarray]:
    """Draw giant planets for an array of host stars.

    Parameters
    ----------
    mass, feh, luminosity
        Host mass (Msun), [Fe/H] and luminosity (Lsun), one per host.
    planes
        Optional (n, 3) reference-plane normals (sky normal +z); NaN rows (no inner system)
        get an isotropic plane.
    window
        Optional HM-C window per host: (P_min, P_max, R_max) arrays. Inside it, giants
        smaller than R_max are not drawn.

    Returns
    -------
    Per-giant arrays sorted by host and period: ``host`` (index into the inputs),
    ``component`` (:data:`COLD`, :data:`HOT` or :data:`WIDE`), ``hot``, ``period`` (d),
    ``a_au``, ``planet_mass`` (Earth masses), ``planet_radius`` (Earth radii), ``ecc``,
    ``omega``, ``mean_anom``, ``incl_mut`` (to the reference plane), ``asc_node``,
    ``incl_sky`` (radians).
    """
    mass = np.asarray(mass, dtype=float)
    feh = np.asarray(feh, dtype=float)
    lum = np.asarray(luminosity, dtype=float)
    n = len(mass)
    # Candidates. Hot Jupiters: at most one per star.
    has_hj = rng.random(n) < np.minimum(model.hj_rate * hj_scaling(mass, feh, model), 1.0)
    hj_host = np.flatnonzero(has_hj)
    m_hj = draw_mass(len(hj_host), rng, model, m_lo=model.hj_m_min)
    spread = rng.gamma(model.dispersion, 1.0 / model.dispersion, n) if np.isfinite(model.dispersion) else 1.0
    lam = expected_cold(mass, feh, model) * np.where(has_hj, model.hj_outer_boost, spread)
    cold_host = np.repeat(np.arange(n), rng.poisson(lam))
    m_cold = draw_mass(len(cold_host), rng, model)
    wide_host = np.repeat(np.arange(n), rng.poisson(expected_wide(mass, model)))
    m_wide = _draw_power(len(wide_host), model.wide_m_index, *model.wide_m_range, rng)
    host = np.concatenate([hj_host, cold_host, wide_host])
    component = np.repeat(np.array([HOT, COLD, WIDE], dtype=np.int8), [len(hj_host), len(cold_host), len(wide_host)])
    m = np.concatenate([m_hj, m_cold, m_wide])
    a = np.empty(len(host))
    e = np.empty(len(host))
    for comp in (HOT, COLD, WIDE):  # vectorised first draw, same distributions as _draw_orbit
        s = np.flatnonzero(component == comp)
        if comp == HOT:
            lo, hi = model.hj_p_range
            p = np.empty(len(s))
            todo = np.arange(len(s))
            while len(todo):
                p[todo] = model.hj_p_median * np.exp(model.hj_p_sigma * rng.normal(size=len(todo)))
                todo = todo[(p[todo] < lo) | (p[todo] >= hi)]
            a[s], e[s] = period_to_a_au(p, mass[host[s]]), 0.0
            continue
        e[s] = np.minimum(rng.beta(*model.ecc_beta, len(s)), 0.95)
        if comp == COLD:
            a[s] = draw_separation(period_to_a_au(model.p_min_cold, mass[host[s]]), np.full(len(s), model.a_max),
                                   rng, model)
        else:
            a[s] = _draw_power(len(s), model.wide_a_index, *model.wide_a_range, rng)
    r = giant_radius(m, SOLAR_FLUX_CGS * lum[host] / a**2, rng)

    def in_window(h, a_, r_):
        if window is None:
            return np.zeros(np.shape(a_), dtype=bool)
        p_ = a_au_to_period(a_, mass[h])
        return (p_ >= window[0][h]) & (p_ <= window[1][h]) & (r_ < window[2][h])

    ok = ~in_window(host, a, r)
    host, component, m, a, e, r = host[ok], component[ok], m[ok], a[ok], e[ok], r[ok]

    # Placement: in order of decreasing mass per host, redrawing orbits that are unstable.
    keep = np.ones(len(host), dtype=bool)
    counts = np.bincount(host, minlength=n)
    for h in np.flatnonzero(counts > 1):
        idx = np.flatnonzero(host == h)
        occ = [[], [], []]
        for j in idx[np.argsort(-m[idx])]:
            for attempt in range(n_redraw + 1):
                if attempt:
                    a[j], e[j] = _draw_orbit(component[j], mass[h], rng, model)
                    r[j] = _giant_radius_scalar(m[j], SOLAR_FLUX_CGS * lum[h] / a[j] ** 2, rng)
                    if in_window(h, a[j], r[j]):
                        continue
                if _stable_with_neighbours(a[j], m[j], e[j], *occ, mass[h]):
                    pos = bisect.bisect_left(occ[0], a[j])
                    for lst, val in zip(occ, (a[j], m[j], e[j])):
                        lst.insert(pos, val)
                    break
            else:
                keep[j] = False
    host, component, m, a, e, r = host[keep], component[keep], m[keep], a[keep], e[keep], r[keep]

    k = len(host)
    out = {
        "host": host, "component": component, "hot": component == HOT, "period": a_au_to_period(a, mass[host]),
        "a_au": a, "planet_mass": m, "planet_radius": r, "ecc": e, "omega": 2.0 * np.pi * rng.random(k),
        "mean_anom": 2.0 * np.pi * rng.random(k),
    }
    order = np.lexsort((out["period"], out["host"]))
    out = {key: val[order] for key, val in out.items()}
    out.update(orient(out["host"], planes, n, rng, model))
    return out


def redraw_eccentricities(giants, hosts, mstar_of_host, rng, model: GiantModel = GiantModel()) -> None:
    """Redraw (in place) the eccentricities of the non-hot giants of ``hosts``; a host's new
    values are reverted if they make an adjacent pair of its giants unstable."""
    sel = np.flatnonzero(np.isin(giants["host"], hosts) & ~giants["hot"])
    if not len(sel):
        return
    old = giants["ecc"][sel].copy()
    giants["ecc"][sel] = np.minimum(rng.beta(*model.ecc_beta, len(sel)), 0.95)
    gh = giants["host"]
    for h in np.unique(gh[sel]):
        lo, hi = np.searchsorted(gh, h, "left"), np.searchsorted(gh, h, "right")
        if hi - lo < 2:
            continue
        o = lo + np.argsort(giants["a_au"][lo:hi])
        a, m, e = giants["a_au"][o], giants["planet_mass"][o], giants["ecc"][o]
        if not all(pair_stable(m[i], m[i + 1], a[i], e[i], a[i + 1], e[i + 1], mstar_of_host[h])
                   for i in range(len(o) - 1)):
            mine = (sel >= lo) & (sel < hi)
            giants["ecc"][sel[mine]] = old[mine]


def inner_conflicts(host_in, a_in, m_in, e_in, giants, mstar_of_host) -> np.ndarray:
    """Mask of inner planets that are unstable with the nearest giant on either side."""
    bad = np.zeros(len(host_in), dtype=bool)
    gh = giants["host"]
    for k in np.flatnonzero(np.isin(host_in, gh)):
        lo, hi = np.searchsorted(gh, host_in[k], "left"), np.searchsorted(gh, host_in[k], "right")
        ga = giants["a_au"][lo:hi]
        o = np.argsort(ga)
        bad[k] = not _stable_with_neighbours(a_in[k], m_in[k], e_in[k], ga[o], giants["planet_mass"][lo:hi][o],
                                             giants["ecc"][lo:hi][o], mstar_of_host[host_in[k]])
    return bad


def orient(host, planes, n_hosts, rng, model: GiantModel = GiantModel()) -> dict[str, np.ndarray]:
    """``incl_mut``, ``asc_node`` (w.r.t. the host's reference plane) and ``incl_sky`` for giants of ``host``.

    ``planes``: (n_hosts, 3) reference-plane normals (sky normal +z) or None; NaN rows
    (no inner system) get an isotropic plane.
    """
    n_ref = np.full((n_hosts, 3), np.nan) if planes is None else np.array(planes, dtype=float)
    missing = np.isnan(n_ref[:, 0])
    n_ref[missing] = _random_unit_vectors(missing.sum(), rng)
    k = len(host)
    incl = np.minimum(np.radians(model.incl_sigma_deg) * np.hypot(rng.normal(size=k), rng.normal(size=k)), np.pi)
    node = 2.0 * np.pi * rng.random(k)
    normals = orbit_normals(n_ref[host], incl, node)
    return {"incl_mut": incl, "asc_node": node, "incl_sky": np.arccos(np.clip(normals[:, 2], -1.0, 1.0))}
