"""Moons of the planets drawn by :func:`starsys.planets.attach_planets`.

No exomoon has been confirmed, so every moon is theory-driven (flag ``moon``); the planet's
own flags are carried over. Three kinds:

- **Regular satellites** (``regular``) of planets with an H/He envelope and >= 10 Earth masses
  (ice and gas giants), formed in the circumplanetary disk:
  - total mass: log-normal around 2e-4 of the planet (Jupiter, Saturn and Uranus all have
    1-2.5e-4; Canup & Ward 2006), 0.5 dex wide (Cilibrasi+2018's population synthesis is
    broader, peaking at 1e-4-1e-3);
  - number of moons that survive the disk phase: P(0..5) = 0.20, 0.07, 0.08, 0.40, 0.22,
    0.03, read off Cilibrasi+2018 (Fig. 15: 3 most common, 4 next, often none, a minimum at
    1-2, never more than 5);
  - masses split uniformly over the simplex (Dirichlet(1)), covering Galilean-like systems
    of similar moons and Saturn-like ones dominated by one;
  - orbits: the innermost at 4-8 planet radii (log-uniform: Io 5.9, Miranda 5.1 radii),
    each next one at a period ratio of 2 (half the time, like the Laplace chain) or 1.5-3,
    circular, in the planet's equator (inclination < 1 deg);
  - composition: icy (density 1.9 g/cm^3, like Ganymede, Callisto, Titan) outside the
    circumplanetary ice line, 12 R_p (M_p / M_Jup)^0.5 (more massive planets are hotter;
    Heller & Pudritz 2015), if the planet's zero-albedo T_eq is below 150 K; rocky (3.4
    g/cm^3, Io) otherwise. Cilibrasi+2018 find 85% of Jovian moons icy.
- **Giant-impact moons** (``impact``) of rocky planets of 0.1-6 Earth masses: Elser+2011
  find more than 1 in 12 terrestrial planets with a massive moon (range 1 in 45 to 1 in 4);
  vapour-rich disks around larger rocky planets do not make fractionally large moons
  (Nakajima+2022). Mass ratio log-uniform 0.004-0.03 (the Moon: 0.0123), formed at 3-5
  planet radii, rocky, inclination up to 10 deg. The impact leaves the planet spinning in
  ~5 h (Canup 2004), so the moon starts outside the synchronous orbit and migrates out.
- **A captured large moon** (``captured``, Triton-like; Agnor & Hamilton 2006, binary-exchange
  capture) for 25% of envelope planets of 10-50 Earth masses (ice giants) beyond 1 au (one of
  the Sun's two ice giants): mass ratio log-uniform 3e-5-3e-4 (Triton 2.1e-4), retrograde (inclination
  110-170 deg), circularised at 10-20 planet radii, icy. It disrupts the regular satellites, so
  that planet keeps none.

**Survival** (both tidal evolution and the planet's Hill sphere):
- moons migrate by the tide they raise on the planet, d(a^13/2)/dt = +-(39/2) (k2/Q) m R^5
  sqrt(G / M_p) (Q, k2 by planet type from :class:`starsys.derived.DerivedModel`): outward while
  the planet spins faster than the moon orbits (prograde moons outside the synchronous radius
  of the planet's initial spin), inward otherwise (retrograde moons always);
- once the star has despun the planet (t_lock, :mod:`starsys.derived`), prograde moons also
  spiral in (Barnes & O'Brien 2002);
- a moon is lost if it reaches the planet's Roche limit (2.44 R_p (rho_p/rho_m)^1/3) or the
  stability limit 0.4895 (1 - 1.0305 e_p) R_Hill for prograde and 0.9309 (1 - 1.0764 e_p)
  R_Hill for retrograde orbits (Domingos+2006).
Close-in planets therefore lose their moons, and impact moons migrate out like the Moon (an
Earth analogue's moon ends near 50 Earth radii after 4.6 Gyr with Q = 100, k2 = 0.3).

Planets of evolved hosts are treated on their present orbits (a moon that survives there is
kept). Moons are not given to circumbinary or pulsar planets differently from others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import derived as dv
from .flags import FLAG_MOON
from .units import AU_M, G_SI, MSUN_KG

EARTH_MASS_KG = dv.EARTH_MASS_KG
EARTH_RADIUS_M = dv.EARTH_RADIUS_M
JUPITER_MASS_EARTH = 317.83
GYR_S = dv.GYR_S
KIND_REGULAR, KIND_IMPACT, KIND_CAPTURED = "regular", "impact", "captured"


@dataclass(frozen=True)
class MoonModel:
    # Regular satellites.
    regular_min_mass: float = 10.0  # Earth masses, with an envelope
    regular_mass_ratio: float = 2e-4  # median total mass / planet mass
    regular_mass_sigma_dex: float = 0.5
    regular_n_probs: tuple = (0.20, 0.07, 0.08, 0.40, 0.22, 0.03)  # 0..5 moons (Cilibrasi+2018)
    regular_inner_rp: tuple = (4.0, 8.0)  # innermost orbit, planet radii
    regular_resonant: float = 0.5  # share of 2:1 period ratios
    regular_period_ratio: tuple = (1.5, 3.0)
    ice_line_rp: float = 12.0  # circumplanetary ice line at M_Jup (planet radii), scales as M^0.5
    ice_teq: float = 150.0  # K: planets warmer than this have rocky moons
    rho_ice: float = 1.9  # g / cm^3
    rho_rock: float = 3.4
    # Giant-impact moons (Elser+2011; Nakajima+2022).
    impact_prob: float = 1.0 / 12.0
    impact_mass_range: tuple = (0.1, 6.0)  # rocky planets, Earth masses
    impact_ratio_range: tuple = (0.004, 0.03)
    impact_a_rp: tuple = (3.0, 5.0)
    impact_spin_h: float = 5.0  # the impact spins the planet up (Earth after the Moon-forming impact; Canup 2004)
    # Captured (Triton-like) moons.
    capture_prob: float = 0.25
    capture_mass_range: tuple = (10.0, 50.0)  # ice giants
    capture_min_a_au: float = 1.0
    capture_ratio_range: tuple = (3e-5, 3e-4)
    capture_a_rp: tuple = (10.0, 20.0)
    # Stability (Domingos+2006).
    hill_prograde: tuple = (0.4895, 1.0305)
    hill_retrograde: tuple = (0.9309, 1.0764)


def _radius_earth(mass_earth, rho):
    m = np.asarray(mass_earth, dtype=float) * EARTH_MASS_KG * 1e3  # g
    return (3.0 * m / (4.0 * math.pi * rho)) ** (1.0 / 3.0) / (EARTH_RADIUS_M * 100.0)


def _log_uniform(lo, hi, n, rng):
    return np.exp(math.log(lo) + rng.random(n) * (math.log(hi) - math.log(lo)))


def _draw(planets, rng, model: MoonModel, t_eq0):
    """Candidate moons before survival: dict of per-moon arrays (``planet`` = planet row)."""
    mp = np.asarray(planets["planet_mass"], dtype=float)
    rp = np.asarray(planets["planet_radius"], dtype=float)
    kind = dv.planet_kind(mp, rp, planets["lost_envelope"])
    n = len(mp)
    out = {k: [] for k in ("planet", "kind", "mass", "a_rp", "incl", "icy")}

    def add(planet, kind_name, mass, a_rp, incl, icy):
        out["planet"].append(planet)
        out["kind"].append(np.full(len(planet), kind_name, dtype=object))
        out["mass"].append(mass)
        out["a_rp"].append(a_rp)
        out["incl"].append(incl)
        out["icy"].append(icy)

    a_au = np.asarray(planets["a_au"], dtype=float)
    # Captured moons first: they rule out regular satellites.
    lo, hi = model.capture_mass_range
    cap = np.flatnonzero((kind > 0) & (mp >= lo) & (mp <= hi) & (a_au >= model.capture_min_a_au)
                         & (rng.random(n) < model.capture_prob))
    k = len(cap)
    add(cap, KIND_CAPTURED, mp[cap] * _log_uniform(*model.capture_ratio_range, k, rng),
        _log_uniform(*model.capture_a_rp, k, rng), np.radians(110.0 + 60.0 * rng.random(k)), np.ones(k, dtype=bool))
    # Regular satellites.
    reg = np.flatnonzero((kind > 0) & (mp >= model.regular_min_mass))
    reg = reg[~np.isin(reg, cap)]
    probs = np.array(model.regular_n_probs, dtype=float)
    n_moons = rng.choice(len(probs), size=len(reg), p=probs / probs.sum())
    host = np.repeat(reg, n_moons)
    if len(host):
        has = n_moons > 0
        first = np.cumsum(n_moons)[has] - n_moons[has]  # first moon of each system
        counts = n_moons[has]
        total = mp[reg[has]] * model.regular_mass_ratio * 10 ** (model.regular_mass_sigma_dex
                                                                   * rng.normal(size=len(first)))
        g = rng.exponential(size=len(host))  # Dirichlet(1) shares
        mass = np.repeat(total, counts) * g / np.repeat(np.add.reduceat(g, first), counts)
        ratio = np.where(rng.random(len(host)) < model.regular_resonant, 2.0,
                         _log_uniform(*model.regular_period_ratio, len(host), rng))
        ratio[first] = 1.0
        # Period ratios accumulate outward along each system: a = a_inner prod(ratio)^(2/3).
        cum = np.cumsum(np.log(ratio))
        a_inner = np.repeat(_log_uniform(*model.regular_inner_rp, len(first), rng), counts)
        a_rp = a_inner * np.exp((cum - np.repeat(cum[first], counts)) * 2.0 / 3.0)
        ice_line = model.ice_line_rp * np.sqrt(mp[host] / JUPITER_MASS_EARTH)
        icy = (a_rp > ice_line) & (t_eq0[host] < model.ice_teq)
        add(host, KIND_REGULAR, mass, a_rp, np.radians(rng.random(len(host))), icy)
    # Giant-impact moons.
    lo, hi = model.impact_mass_range
    imp = np.flatnonzero((kind == 0) & (mp >= lo) & (mp <= hi) & (rng.random(n) < model.impact_prob))
    k = len(imp)
    add(imp, KIND_IMPACT, mp[imp] * _log_uniform(*model.impact_ratio_range, k, rng),
        model.impact_a_rp[0] + rng.random(k) * (model.impact_a_rp[1] - model.impact_a_rp[0]),
        np.radians(10.0 * rng.random(k)), np.zeros(k, dtype=bool))
    return {key: np.concatenate(v) if v else np.zeros(0) for key, v in out.items()}


def _survive(m, planets, t_lock, age, model: MoonModel, dmodel: dv.DerivedModel):
    """Tidal evolution and stability: (kept mask, final a in planet radii)."""
    p = m["planet"]
    mp_e = np.asarray(planets["planet_mass"], dtype=float)[p]
    rp_e = np.asarray(planets["planet_radius"], dtype=float)[p]
    kind = dv.planet_kind(mp_e, rp_e, np.asarray(planets["lost_envelope"])[p], dmodel)
    q, k2, _, p0 = np.array([dmodel.tides_rocky, dmodel.tides_subneptune, dmodel.tides_giant])[kind].T
    mp = mp_e * EARTH_MASS_KG
    rp = rp_e * EARTH_RADIUS_M
    mm = m["mass"] * EARTH_MASS_KG
    retro = m["incl"] > math.pi / 2
    # d(a^13/2)/dt = (39/2) (k2 / Q) m R^5 sqrt(G / M_p)
    rate = 19.5 * k2 / q * mm * rp**5 * np.sqrt(G_SI / mp)
    a0 = m["a_rp"] * rp
    p0 = np.where(m["kind"] == KIND_IMPACT, model.impact_spin_h, p0)
    a_sync = (G_SI * mp * (p0 * 3600.0) ** 2 / (4 * math.pi**2)) ** (1.0 / 3.0)
    t1 = np.minimum(t_lock[p], age[p]) * GYR_S
    t2 = np.maximum(age[p] * GYR_S - t1, 0.0)
    out = (a0 > a_sync) & ~retro
    x = a0**6.5 + np.where(out, rate * t1, -rate * (t1 + t2))
    x = np.where(out, x - rate * t2, x)  # despun planet: back inward
    # Stability limit and Roche limit.
    a_star = np.asarray(planets["a_au"], dtype=float)[p] * AU_M
    m_star = np.asarray(planets["host_mass"], dtype=float)[p] * MSUN_KG
    e_p = np.clip(np.asarray(planets["ecc"], dtype=float)[p], 0.0, 0.9)
    r_hill = a_star * (mp / (3.0 * m_star)) ** (1.0 / 3.0)
    f, c = np.where(retro[:, None], np.array(model.hill_retrograde)[None, :], np.array(model.hill_prograde)[None, :]).T
    a_crit = f * np.maximum(1.0 - c * e_p, 0.0) * r_hill
    rho_p = mp / (4.0 / 3.0 * math.pi * rp**3)
    rho_m = np.where(m["icy"], model.rho_ice, model.rho_rock) * 1e3
    roche = 2.44 * rp * (rho_p / rho_m) ** (1.0 / 3.0)
    # Outward-migrating moons that cross a_crit at any time are lost (the maximum of a is reached at t1).
    a_max = np.where(out, (a0**6.5 + rate * t1) ** (1.0 / 6.5), a0)
    a_end = np.maximum(x, 0.0) ** (1.0 / 6.5)
    keep = (a_max < a_crit) & (a_end > np.maximum(roche, rp)) & (a0 < a_crit)
    return keep, a_end / rp


def attach_moons(planets, components, seed=None, model: MoonModel = MoonModel(),
                 derived: dv.DerivedModel = dv.DerivedModel()) -> dict[str, np.ndarray]:
    """Moons of ``planets`` (output of :func:`starsys.planets.attach_planets`).

    Returns a dict of per-moon arrays sorted by planet and orbit: ``planet`` (row index into
    ``planets``), ``system``, ``component``, ``kind`` ("regular", "impact", "captured"),
    ``mass`` (Earth masses), ``radius`` (Earth radii), ``a_rp`` (planet radii), ``a_km``,
    ``period`` (days), ``ecc``, ``incl`` (rad, to the planet's equator; > pi/2 retrograde),
    ``composition`` ("ice" or "rock") and ``flags`` (the planet's flags plus ``moon``).
    """
    rng = np.random.default_rng(seed)
    n = len(planets["a_au"])
    if "t_lock_gyr" in planets:
        t_lock = np.asarray(planets["t_lock_gyr"], dtype=float)
        t_eq0 = np.asarray(planets["t_eq0"], dtype=float)
    else:
        extra = dv.derive_properties(planets, {"n_components": _n_components(components)}, components, derived)
        t_lock, t_eq0 = extra["t_lock_gyr"], extra["t_eq0"]
    t_eq0 = np.nan_to_num(t_eq0, nan=0.0)
    age = np.asarray(components["age_gyr"], dtype=float)[np.asarray(planets["component"])] if n else np.zeros(0)
    m = _draw(planets, rng, model, t_eq0)
    keep, a_rp = _survive(m, planets, t_lock, age, model, derived) if len(m["planet"]) else (np.zeros(0, bool), np.zeros(0))
    m = {key: v[keep] for key, v in m.items()}
    a_rp = a_rp[keep]
    p = m["planet"].astype(np.int64)
    rp_km = np.asarray(planets["planet_radius"], dtype=float)[p] * EARTH_RADIUS_M / 1e3
    mp = np.asarray(planets["planet_mass"], dtype=float)[p] * EARTH_MASS_KG
    a_m = a_rp * rp_km * 1e3
    period = 2 * math.pi * np.sqrt(a_m**3 / (G_SI * (mp + m["mass"] * EARTH_MASS_KG))) / 86400.0
    radius = _radius_earth(m["mass"], np.where(m["icy"], model.rho_ice, model.rho_rock))
    res = {"planet": p, "system": np.asarray(planets["system"])[p], "component": np.asarray(planets["component"])[p],
           "kind": m["kind"], "mass": m["mass"], "radius": radius, "a_rp": a_rp, "a_km": a_rp * rp_km,
           "period": period, "ecc": np.zeros(len(p)), "incl": m["incl"],
           "composition": np.where(m["icy"], "ice", "rock").astype(object),
           "flags": (np.asarray(planets["flags"])[p] | FLAG_MOON).astype(np.int32)}
    order = np.lexsort((res["a_rp"], res["planet"]))
    return {key: v[order] for key, v in res.items()}


def _n_components(components):
    return np.bincount(np.asarray(components["system"]))
