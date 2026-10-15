"""Derived planet properties: insolation, equilibrium temperature, habitable zone, spin.

Called by :func:`starsys.planets.attach_planets` (``derived=`` option) on the finished planet
table; uses no random numbers, so the other columns do not depend on it.

- **Insolation** ``insolation`` (Earth = 1): present-day bolometric flux averaged over the
  orbit, L / (a^2 sqrt(1 - e^2)), from the host plus every other star of the system. A
  companion's light reaches an S-type planet from the separation of the wider of the two
  stars' orbits (hierarchy: component k orbits the centre of mass of components 0..k-1),
  averaged over that orbit; circumbinary (P-type) planets see components 0 and 1 from their
  barycentre. Evolved hosts shine with their present luminosity (giants are bright, white
  dwarfs faint). Neutron stars and black holes have no luminosity in the population: NaN.
- **Equilibrium temperature** ``t_eq`` (K): full heat redistribution with a Bond albedo
  ``albedo`` (0.3, Earth-like), T_eq = 278.3 K ((1 - A) S)^(1/4); ``t_eq0`` for A = 0.
- **Habitable zone** ``hz`` (Kopparapu+2014, 1D cloud-free climate models): S_eff(Teff) =
  S_eff,sun + a T + b T^2 + c T^3 + d T^4, T = Teff - 5780 K, for recent Venus, runaway
  greenhouse (planet-mass dependent: 0.1, 1 and 5 Earth masses, interpolated in log mass and
  held outside), maximum greenhouse and early Mars. Several stars are combined as
  Haghighipour & Kaltenegger (2013): a boundary is crossed when sum_i F_i / S_eff(Teff_i) = 1.
  Codes: 0 too hot (inside recent Venus), 1 optimistic inner (recent Venus to runaway
  greenhouse), 2 conservative (runaway to maximum greenhouse), 3 optimistic outer (maximum
  greenhouse to early Mars), 4 too cold, -1 undefined (no luminous star). The zone is given
  for every planet; whether a planet can be habitable also depends on its size and envelope.
  The fits cover 2600-7200 K; a planet whose flux comes more than 10% from stars outside that
  range (white dwarfs, A stars, brown dwarfs) is flagged ``hz_extrapolated`` and uses the fit
  at the nearest valid Teff.
- **Tidal locking** (Gladman+1996): despinning time t_lock = w a^6 alpha M_p Q / (3 G M*^2 k2
  R_p^3), with initial spin w, moment-of-inertia factor alpha, tidal Q and Love number k2 by
  planet type: rocky (R <= 1.6 Earth radii, or envelope lost): Q = 100, k2 = 0.3, alpha =
  0.33, 12 h; sub-Neptunes and ice giants: Q = 1e4, k2 = 0.1, alpha = 0.23, 16 h; giants
  (>= 50 Earth masses or >= 6 Earth radii): Q = 1e5, k2 = 0.5, alpha = 0.25, 10 h (Goldreich &
  Soter 1966; Solar System values). ``t_lock_gyr``; ``spin``: 0 free, 1 synchronous (locked,
  e < 0.1), 2 3:2 resonance (locked, e >= 0.1, Mercury-like; capture into 3:2 becomes likely
  above e ~ 0.1, Makarov 2012). Locked means t_lock below the system age. Order-of-magnitude:
  Q is uncertain by ~10x, and atmospheric thermal tides (Venus) are not modelled. With these
  values Mercury is locked (t_lock 0.6 Gyr), Venus not (22 Gyr), Earth not (160 Gyr).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .flags import FLAG_HZ_EXTRAP
from .units import AU_M, G_SI, MSUN_KG

EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_M = 6.3781e6
GYR_S = 3.15576e16
T_EQ_EARTH_FLUX = (1361.0 / (4 * 5.670374e-8)) ** 0.25  # 278.3 K: T_eq at 1 Earth flux, A = 0

# Kopparapu+2014 Table 1: S_eff,sun, a, b, c, d. Rows: recent Venus, runaway greenhouse (0.1, 1,
# 5 Earth masses), maximum greenhouse, early Mars.
HZ_RECENT_VENUS = (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15)
HZ_RUNAWAY = {0.1: (0.99, 1.209e-4, 1.404e-8, -7.418e-12, -1.713e-15),
              1.0: (1.107, 1.332e-4, 1.58e-8, -8.308e-12, -1.931e-15),
              5.0: (1.188, 1.433e-4, 1.707e-8, -8.968e-12, -2.084e-15)}
HZ_MAX_GREENHOUSE = (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16)
HZ_EARLY_MARS = (0.32, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16)
HZ_TEFF_RANGE = (2600.0, 7200.0)

HZ_TOO_HOT, HZ_OPT_INNER, HZ_CONSERVATIVE, HZ_OPT_OUTER, HZ_TOO_COLD, HZ_UNDEFINED = 0, 1, 2, 3, 4, -1
SPIN_FREE, SPIN_SYNC, SPIN_32 = 0, 1, 2


@dataclass(frozen=True)
class DerivedModel:
    albedo: float = 0.3
    hz_extrap_fraction: float = 0.1  # flux share from stars outside the HZ fits that sets the flag
    # Tidal parameters (Q, k2, alpha, initial spin period in hours): rocky, sub-Neptune, giant.
    tides_rocky: tuple = (100.0, 0.3, 0.33, 12.0)
    tides_subneptune: tuple = (1e4, 0.1, 0.23, 16.0)
    tides_giant: tuple = (1e5, 0.5, 0.25, 10.0)
    rocky_radius: float = 1.6  # Earth radii
    giant_mass: float = 50.0  # Earth masses
    giant_radius: float = 6.0  # Earth radii
    resonance_ecc: float = 0.1


def _poly(coef, teff):
    t = np.clip(teff, *HZ_TEFF_RANGE) - 5780.0
    s0, a, b, c, d = coef
    return s0 + a * t + b * t**2 + c * t**3 + d * t**4


def s_eff(teff, planet_mass=1.0) -> dict[str, np.ndarray]:
    """Kopparapu+2014 boundary fluxes (Earth = 1) for host Teff (clamped to 2600-7200 K) and planet
    mass (Earth masses): ``recent_venus``, ``runaway``, ``max_greenhouse``, ``early_mars``."""
    teff = np.asarray(teff, dtype=float)
    lm = np.clip(np.log10(np.broadcast_to(np.asarray(planet_mass, dtype=float), teff.shape)), -1.0, math.log10(5.0))
    r01, r1, r5 = (_poly(HZ_RUNAWAY[m], teff) for m in (0.1, 1.0, 5.0))
    w_lo = lm + 1.0  # 0.1 -> 1 Earth mass
    w_hi = lm / math.log10(5.0)  # 1 -> 5 Earth masses
    runaway = np.where(lm < 0, r01 + w_lo * (r1 - r01), r1 + w_hi * (r5 - r1))
    return {"recent_venus": _poly(HZ_RECENT_VENUS, teff), "runaway": runaway,
            "max_greenhouse": _poly(HZ_MAX_GREENHOUSE, teff), "early_mars": _poly(HZ_EARLY_MARS, teff)}


def _sources(planets, systems, components):
    """Per planet and per star of its system: flux (Earth = 1) and that star's Teff, as arrays
    (n_planets, n_roles); missing stars have flux 0."""
    c_sys = np.asarray(components["system"])
    n_c = len(c_sys)
    teff = np.asarray(components["teff"], dtype=float)
    lum = components.get("luminosity")
    if lum is None:
        lum = np.asarray(components["radius"], dtype=float) ** 2 * (teff / 5772.0) ** 4
    lum = np.nan_to_num(np.asarray(lum, dtype=float), nan=0.0)
    a_c = np.asarray(components.get("a_au", np.full(n_c, np.nan)), dtype=float)
    e_c = np.nan_to_num(np.asarray(components.get("ecc", np.zeros(n_c)), dtype=float))
    role_c = np.asarray(components["role"])
    first = np.searchsorted(c_sys, np.arange(len(systems["n_components"])))
    n_comp = np.asarray(systems["n_components"])
    comp = np.asarray(planets["component"])
    sys_p = np.asarray(planets["system"])
    p_type = np.asarray(planets["host_kind"]) == "P"
    a = np.asarray(planets["a_au"], dtype=float)
    e = np.clip(np.asarray(planets["ecc"], dtype=float), 0.0, 0.999)
    r_host = role_c[comp]
    n_roles = int(n_comp.max(initial=1))
    flux = np.zeros((len(comp), n_roles))
    t_src = np.full((len(comp), n_roles), np.nan)
    orbit_avg = 1.0 / (a**2 * np.sqrt(1.0 - e**2))
    for j in range(n_roles):
        has = j < n_comp[sys_p]
        idx = np.where(has, first[sys_p] + j, 0)
        own = has & ((j == r_host) | (p_type & (j <= 1)))
        wide = np.where(p_type, j, np.maximum(j, r_host))  # orbit separating the planet's host from star j
        d_idx = np.where(has, first[sys_p] + np.minimum(wide, n_comp[sys_p] - 1), 0)
        d = a_c[d_idx] * np.sqrt(np.sqrt(1.0 - np.clip(e_c[d_idx], 0.0, 0.999) ** 2))  # <1/d^2> = 1/(d^2 sqrt(1-e^2))
        f = np.where(own, lum[idx] * orbit_avg, np.where(has, lum[idx] / np.where(np.isfinite(d) & (d > 0), d, np.inf) ** 2, 0.0))
        flux[:, j] = np.where(has, f, 0.0)
        t_src[:, j] = np.where(has, teff[idx], np.nan)
    host_lum = lum[comp] + np.where(p_type, lum[np.minimum(comp + 1, len(lum) - 1)], 0.0)
    return flux, t_src, host_lum > 0


def tidal_lock_time(a_au, planet_mass, planet_radius, host_mass, kind, model: DerivedModel = DerivedModel()):
    """Despinning time (Gyr) from the initial spin; ``kind`` 0 rocky, 1 sub-Neptune, 2 giant."""
    pars = np.array([model.tides_rocky, model.tides_subneptune, model.tides_giant])[kind]
    q, k2, alpha, p0 = pars.T
    omega = 2 * np.pi / (p0 * 3600.0)
    a = np.asarray(a_au, dtype=float) * AU_M
    m = np.asarray(planet_mass, dtype=float) * EARTH_MASS_KG
    r = np.asarray(planet_radius, dtype=float) * EARTH_RADIUS_M
    ms = np.asarray(host_mass, dtype=float) * MSUN_KG
    return omega * a**6 * alpha * m * q / (3.0 * G_SI * ms**2 * k2 * r**3) / GYR_S


def planet_kind(planet_mass, planet_radius, lost_envelope, model: DerivedModel = DerivedModel()) -> np.ndarray:
    """0 rocky, 1 sub-Neptune / ice giant, 2 gas giant."""
    m = np.asarray(planet_mass, dtype=float)
    r = np.asarray(planet_radius, dtype=float)
    giant = (m >= model.giant_mass) | (r >= model.giant_radius)
    rocky = ~giant & ((r <= model.rocky_radius) | np.asarray(lost_envelope, dtype=bool))
    return np.where(giant, 2, np.where(rocky, 0, 1)).astype(np.int8)


def derive_properties(planets, systems, components, model: DerivedModel = DerivedModel()) -> dict[str, np.ndarray]:
    """New per-planet columns (see module docstring) and the ``flags`` to OR in (``hz_flags``)."""
    n = len(planets["a_au"])
    if not n:
        e = np.zeros(0)
        return {"insolation": e, "t_eq": e, "t_eq0": e, "hz": np.zeros(0, np.int8), "t_lock_gyr": e,
                "spin": np.zeros(0, np.int8), "hz_flags": np.zeros(0, np.int32)}
    flux, t_src, lit = _sources(planets, systems, components)
    s = flux.sum(axis=1)
    lit = lit & (s > 0)
    s = np.where(lit, s, np.nan)
    t_eq0 = T_EQ_EARTH_FLUX * s**0.25
    t_eq = t_eq0 * (1.0 - model.albedo) ** 0.25

    # Habitable zone: sum_i F_i / S_eff,b(T_i) per boundary b.
    mp = np.asarray(planets["planet_mass"], dtype=float)
    x = {}
    t_fill = np.nan_to_num(t_src, nan=5780.0)
    for j in range(flux.shape[1]):
        lim = s_eff(t_fill[:, j], mp)
        for key, v in lim.items():
            x[key] = x.get(key, 0.0) + flux[:, j] / v
    hz = np.full(n, HZ_TOO_COLD, dtype=np.int8)
    hz[x["early_mars"] >= 1.0] = HZ_OPT_OUTER
    hz[x["max_greenhouse"] >= 1.0] = HZ_CONSERVATIVE
    hz[x["runaway"] > 1.0] = HZ_OPT_INNER
    hz[x["recent_venus"] > 1.0] = HZ_TOO_HOT
    hz[~lit] = HZ_UNDEFINED
    outside = (t_src < HZ_TEFF_RANGE[0]) | (t_src > HZ_TEFF_RANGE[1])
    share = np.where(outside, flux, 0.0).sum(axis=1) / np.where(lit, flux.sum(axis=1), 1.0)
    hz_flags = np.where(lit & (share > model.hz_extrap_fraction), FLAG_HZ_EXTRAP, 0).astype(np.int32)

    # Spin.
    kind = planet_kind(mp, planets["planet_radius"], planets["lost_envelope"], model)
    t_lock = tidal_lock_time(planets["a_au"], mp, planets["planet_radius"], planets["host_mass"], kind, model)
    age = np.asarray(components["age_gyr"], dtype=float)[np.asarray(planets["component"])]
    locked = t_lock < age
    spin = np.where(locked, np.where(np.asarray(planets["ecc"]) >= model.resonance_ecc, SPIN_32, SPIN_SYNC),
                    SPIN_FREE).astype(np.int8)
    return {"insolation": s, "t_eq": t_eq, "t_eq0": t_eq0, "hz": hz, "t_lock_gyr": t_lock, "spin": spin,
            "hz_flags": hz_flags}
