"""Planets in multiple-star systems: suppression, dynamical limits, circumbinary hosts.

Used by :func:`starsys.planets.attach_planets` through :mod:`starsys.hosts`.

**Suppression of S-type planets (Moe & Kratter 2021, MNRAS 507, 3593; MK21).** A star with a
stellar companion at separation a keeps its planetary system with probability S_bin(a):
0 for a <= 1 au, 0.15 at 10 au, 1 for a >= 200 au, linear in log a in between (their
adopted segmented fit to RV and imaging surveys of planet hosts; independent of host mass
and planet size). Applied to every planet model, as a whole-system Bernoulli draw.

**Single-star rates.** Transit surveys (Kepler, TESS) make no cut against close binaries, so
the rates fitted to them (HM-C, :mod:`starsys.inner.mdwarf`) are averages over all primaries
including suppressed ones (MK21's R_AllStars). RV surveys remove close binaries (R_SingleStars),
and so do direct-imaging samples. Microlensing rates are per lens, and binary lenses far from
the Einstein ring (most of them) look single, so they are field averages too: the
microlensing anchor of :mod:`starsys.outer.small` gets the same boost B(M) (without HM-C's
efficiency correction below). The transit-fitted models are therefore boosted by B = 1 / <S_bin> over a mock magnitude-limited Kepler sample drawn from
this population (Malmquist weight (L_tot/L_1)^1.5, systems brighter than 1.75 L_1 removed
like HM-C's L <= 1.75 L_MS cut). B goes into HM-C's f_swpa (clipped at 1) and the remainder
into the cluster rate lambda_c (see :func:`apply_boost`). MK21 find 33 +- 5% (volume-limited)
and 43 +- 7% (magnitude-limited) of solar-type primaries without close planets; this is
checked in :mod:`starsys.validation.binaries`.

**Dynamical limits (Holman & Wiegert 1999, AJ 117, 621; HW99).** S-type planets must lie
inside a_crit = a_b (0.464 - 0.380 mu - 0.631 e + 0.586 mu e + 0.150 e^2 - 0.198 mu e^2)
(eq. 1, mu = m_perturber / (m_host + m_perturber)); circumbinary planets outside
a_crit = a_b (1.60 + 5.10 e - 2.22 e^2 + 4.12 mu - 4.27 e mu - 5.09 mu^2 + 4.61 e^2 mu^2)
(eq. 3, mu = m_2 / (m_1 + m_2)). Fitted for 0.1 <= mu <= 0.9 and e <= 0.8 (S) / 0.7 (P).
Outside: mu > 0.9 (a planet around a much lighter companion) follows the Hill scaling
((1 - mu) / 0.1)^(1/3) from mu = 0.9; mu < 0.1 uses the formulae (smooth, the limits of the
test-particle problem); e beyond the fitted range scales with the pericentre (S) or
apocentre (P) distance from the edge value. The population's hierarchy is "companion k
orbits the centre of mass of components 0..k-1", so a planet around component r is limited
by the inner subsystem (mass M_<r at a_r) and by every outer companion k > r.

**Circumbinary (P-type) planets.** The inner pair (components 0 and 1) is a host of mass
m_0 + m_1, with the primary's Teff and colour and the radius that gives the pair's total
luminosity. Only pairs of main-sequence or pre-main-sequence stars and brown dwarfs are
circumbinary hosts. Two parts:

- the single-star models (HM-C, giants, small outer planets) outside the HW99 limit, except
  HM-C planets of 4 Earth radii and more, which the next part replaces;
- an empirical component for the planets Kepler measured (Armstrong+2014, MNRAS 444, 1873;
  eclipsing binaries, coplanar case, P < 300 d): 14% of binaries with a 4-6, 3.5% with a
  6-8 and 6.4% with an 8-10 Earth-radius planet (maximum likelihood; none above 10, < 5.9%
  at 95%). These are 3-8 times the single-star rates of the same sizes, while giants are
  rarer. They confirm the shortfall around binaries with P_bin < 7 d (Martin+2015 attribute
  it to tertiary-driven shrinking of those binaries), so no such planets are drawn there.
  Planets are placed at P/P_crit log-uniform in 1-2.5, where most Kepler circumbinary
  planets lie (P/P_crit 1.1-1.4, plus Kepler-1647 b farther out; whether this is a pile-up
  is debated: Li, Holman & Tao 2016, Quarles+2018). The per-binary rates are calibrated so
  that the eclipse-weighted rate over P_bin = 2-100 d and P < 300 d equals Armstrong's
  (:mod:`starsys.validation.binaries`). Masses from the radius with Chen & Kipping (2017)'s
  Neptunian relation (R = 0.808 M^0.589) with 0.3 dex scatter. Wider binaries, whose limit lies
  beyond 300 d, get the same planets near their limit: an extrapolation.

Orbits are near the binary plane (Kepler circumbinary planets are within a few degrees).

**Planes.** S-type systems with a companion inside 700 au are aligned with the companion's
orbit in half the cases (Gaussian, sigma 10 deg), isotropic otherwise (Christian+2022, AJ 163,
207: a 50/50 aligned/isotropic mixture, 26-72% aligned at 90% confidence, with the excess
below ~700 au). Wider systems are isotropic.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import astuple, dataclass
from functools import lru_cache

import numpy as np

from .paths import DATA_DIR

BOOST_DIR = DATA_DIR / "binaries"
BOOST_VERSION = 1  # bump when the mock-Kepler measurement changes
HBL = 0.075  # hydrogen-burning limit (Msun): lighter companions truncate but do not suppress


@dataclass(frozen=True)
class BinaryModel:
    # MK21 suppression: (separation in au, S_bin), interpolated in log a.
    s_bin_knots: tuple = ((1.0, 0.0), (10.0, 0.15), (200.0, 1.0))
    suppress: bool = True
    # Single-star boost of the transit-fitted models (mock magnitude-limited Kepler sample).
    boost: bool = True
    # Setting (B21): share (in log) of the boost that goes into HM-C's cluster count where f_swpa
    # is already 1. 1 (default) keeps the Kepler planets per star (+28-45% systems with two
    # transiting planets per single); 0 keeps Kepler's multiplicity mix (14-29% fewer planets).
    boost_cluster_share: float = 1.0
    malmquist_exponent: float = 1.5
    luminosity_cut: float = 1.75
    boost_mass_edges: tuple = (0.01, 0.05, 0.08, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35,
                               1.5, 1.75, 2.0, 3.0)
    boost_systems: int = 1_000_000
    # Efficiency correction of the boost (Teff knots, factor on B): where f_swpa is already 1
    # the boost goes into extra clusters, of which a fraction fail HM-C's stability tests.
    # Fitted so that the mock-Kepler average equals the field average
    # (``python -m starsys.validation.binaries --fit``).
    boost_correction: tuple = ((2600.0, 1.023), (3000.0, 1.159), (3400.0, 1.178), (3800.0, 1.157), (4300.0, 1.082),
                               (4800.0, 1.062), (5300.0, 1.059), (5800.0, 0.999), (6300.0, 1.018), (7000.0, 1.004))
    # Circumbinary hosts.
    circumbinary: bool = True
    cb_incl_sigma_deg: float = 2.0
    # Empirical circumbinary component (Armstrong+2014): radius bins (Earth radii) and
    # per-binary rates, calibrated to the eclipse-weighted P < 300 d rates.
    cb_radius_bins: tuple = ((4.0, 6.0), (6.0, 8.0), (8.0, 10.0))
    cb_rates: tuple = (0.46, 0.11, 0.21)
    cb_p_bin_min: float = 7.0  # days
    # Setting (B22): multiplier of the empirical rates (capped at 1 per bin). They rest on ~10
    # Kepler planets (uncertain by ~3x): 1/3-3 spans the range; 1 is Armstrong+2014.
    cb_rate_factor: float = 1.0
    cb_period_ratio: tuple = (1.0, 2.5)  # P / P_crit, log-uniform
    cb_ecc_sigma: float = 0.03
    cb_mass_scatter_dex: float = 0.3
    # Alignment of S-type systems with the nearest companion's orbit (Christian+2022).
    align_a_max: float = 700.0
    align_fraction: float = 0.5
    align_sigma_deg: float = 10.0
    # Planets around hosts with a companion inside this separation are flagged ``multiple``.
    flag_a_max: float = 200.0
    # Kozai-Lidov cycles of S-type planets driven by the nearest companion.
    kozai: bool = True

    def key(self) -> str:
        return hashlib.sha1(repr((BOOST_VERSION, self.s_bin_knots, self.malmquist_exponent, self.luminosity_cut,
                                  self.boost_mass_edges, self.boost_systems)).encode()).hexdigest()[:16]


# --- formulae ---------------------------------------------------------------------------

def s_bin(a_au, model: BinaryModel = BinaryModel()) -> np.ndarray:
    """MK21 suppression factor: fraction of planetary systems kept at binary separation a (au)."""
    knots = np.array(model.s_bin_knots, dtype=float)
    a = np.asarray(a_au, dtype=float)
    out = np.interp(np.log(np.where(np.isfinite(a), a, 1e30)), np.log(knots[:, 0]), knots[:, 1])
    return np.where(np.isnan(a), 1.0, out)


def hw_s_type(a_b, e, mu) -> np.ndarray:
    """HW99 eq. 1: largest stable S-type semi-major axis (same unit as a_b).
    ``mu`` = perturber mass / (host + perturber)."""
    a_b, e, mu = (np.asarray(x, dtype=float) for x in (a_b, e, mu))
    m = np.clip(mu, 0.0, 0.9)
    ee = np.minimum(e, 0.8)
    f = 0.464 - 0.380 * m - 0.631 * ee + 0.586 * m * ee + 0.150 * ee**2 - 0.198 * m * ee**2
    f = f * np.where(mu > 0.9, (np.maximum(1.0 - mu, 0.0) / 0.1) ** (1.0 / 3.0), 1.0)
    f = f * np.where(e > 0.8, (1.0 - np.minimum(e, 1.0)) / 0.2, 1.0)
    return a_b * np.maximum(f, 0.0)


def hw_p_type(a_b, e, mu) -> np.ndarray:
    """HW99 eq. 3: smallest stable circumbinary semi-major axis. ``mu`` = m_2 / (m_1 + m_2) <= 0.5."""
    a_b, e, mu = (np.asarray(x, dtype=float) for x in (a_b, e, mu))
    m = np.clip(mu, 0.0, 0.5)
    ee = np.minimum(e, 0.7)
    f = (1.60 + 5.10 * ee - 2.22 * ee**2 + 4.12 * m - 4.27 * ee * m - 5.09 * m**2 + 4.61 * ee**2 * m**2)
    f = f * np.where(e > 0.7, (1.0 + np.minimum(e, 1.0)) / 1.7, 1.0)
    return a_b * f


def _unit(incl, node):
    """Orbit normal from inclination to the sky normal (+z) and node."""
    s = np.sin(incl)
    return np.stack([s * np.sin(node), -s * np.cos(node), np.cos(incl)], axis=-1)


def _tilt(n_ref, sigma, rng):
    """Normals at Rayleigh(sigma) angles around ``n_ref`` (same construction as the planet orbits)."""
    from .outer.giants import orbit_normals

    k = len(n_ref)
    incl = np.minimum(sigma * np.hypot(rng.normal(size=k), rng.normal(size=k)), np.pi)
    return orbit_normals(n_ref, incl, 2.0 * np.pi * rng.random(k))


# --- system structure -------------------------------------------------------------------

def structure(components) -> dict[str, np.ndarray]:
    """Per component: S-type limit, nearest stellar companion and its orbit; per system, the
    inner pair's circumbinary limits.

    Returns a dict with per-component ``a_max`` (au, inf for singles), ``a_near`` (separation
    of the nearest companion of >= 0.075 Msun, NaN if none), ``a_near_any`` (nearest companion
    of any mass), ``near_normal`` (orbit normal of that nearest companion, (n, 3)), and
    per-component (meaningful on primaries) ``pair_a_min``/``pair_a_max`` (circumbinary limits
    of components 0+1, NaN without a companion) and ``pair_normal``. ``near_mass`` and ``near_ecc``
    are the mass (the inner subsystem's for a companion's own orbit) and eccentricity of the orbit
    at ``a_near_any``.
    """
    sys_ = np.asarray(components["system"])
    role = np.asarray(components["role"])
    mass = np.asarray(components["mass"], dtype=float)
    n = len(sys_)
    a = np.asarray(components.get("a_au", np.full(n, np.nan)), dtype=float)
    e = np.nan_to_num(np.asarray(components.get("ecc", np.zeros(n)), dtype=float))
    if "incl_sky" in components:
        normal = _unit(np.asarray(components["incl_sky"], dtype=float), np.asarray(components["node"], dtype=float))
    else:
        normal = np.full((n, 3), np.nan)
    start = np.searchsorted(sys_, sys_, "left")  # row of each system's primary (sorted by system, role)
    # Mass inside each companion's orbit (components 0..k-1).
    cum = np.cumsum(mass)
    inner_mass = cum - mass - np.where(start > 0, cum[start - 1], 0.0)

    a_max = np.full(n, np.inf)
    a_near = np.full(n, np.nan)
    a_near_any = np.full(n, np.nan)
    near_normal = np.full((n, 3), np.nan)
    near_mass = np.full(n, np.nan)
    near_ecc = np.full(n, np.nan)

    def consider(host, sep, pert_mass, ecc, nrm, mu):
        nonlocal a_max
        a_max[host] = np.minimum(a_max[host], hw_s_type(sep, ecc, mu))
        closer = ~(a_near_any[host] <= sep)
        a_near_any[host] = np.where(closer, sep, a_near_any[host])
        near_normal[host] = np.where(closer[:, None], nrm, near_normal[host])
        near_mass[host] = np.where(closer, pert_mass, near_mass[host])
        near_ecc[host] = np.where(closer, ecc, near_ecc[host])
        stellar = (pert_mass >= HBL) & ~(a_near[host] <= sep)
        a_near[host] = np.where(stellar, sep, a_near[host])

    max_role = role.max(initial=0)
    for r in range(max_role + 1):
        hosts = np.flatnonzero(role == r)
        for k in range(1, max_role + 1):
            if k == r:
                # The inner subsystem (components 0..r-1) seen from component r, on orbit r.
                h = hosts
                consider(h, a[h], inner_mass[h], e[h], normal[h], inner_mass[h] / (inner_mass[h] + mass[h]))
                continue
            if k < r:
                continue
            # Outer companion k of the same system.
            idx = start[hosts] + k
            ok = (idx < n) & (sys_[np.minimum(idx, n - 1)] == sys_[hosts])
            h, c = hosts[ok], idx[ok]
            consider(h, a[c], mass[c], e[c], normal[c], mass[c] / (mass[h] + mass[c]))

    # Circumbinary limits of the inner pair (on the primary's row).
    pair_a_min = np.full(n, np.nan)
    pair_a_max = np.full(n, np.nan)
    pair_normal = np.full((n, 3), np.nan)
    prim = np.flatnonzero(role == 0)
    sec = prim + 1
    has = (sec < n) & (sys_[np.minimum(sec, n - 1)] == sys_[prim])
    p, s = prim[has], sec[has]
    m_pair = mass[p] + mass[s]
    pair_a_min[p] = hw_p_type(a[s], e[s], np.minimum(mass[p], mass[s]) / m_pair)
    pair_a_max[p] = np.inf
    pair_normal[p] = normal[s]
    for k in range(2, max_role + 1):
        c = p + k
        ok = (c < n) & (sys_[np.minimum(c, n - 1)] == sys_[p])
        pp, cc = p[ok], c[ok]
        mp = mass[pp] + mass[pp + 1]
        pair_a_max[pp] = np.minimum(pair_a_max[pp], hw_s_type(a[cc], e[cc], mass[cc] / (mp + mass[cc])))
    return {"a_max": a_max, "a_near": a_near, "a_near_any": a_near_any, "near_normal": near_normal,
            "near_mass": near_mass, "near_ecc": near_ecc,
            "pair_a_min": pair_a_min, "pair_a_max": pair_a_max, "pair_normal": pair_normal}


def s_type_planes(a_near_any, near_normal, rng, model: BinaryModel = BinaryModel()) -> np.ndarray:
    """Reference planes of S-type systems: aligned with the nearest companion's orbit for a
    fraction of hosts with a companion inside ``align_a_max``; NaN (= isotropic) otherwise."""
    n = len(a_near_any)
    out = np.full((n, 3), np.nan)
    cand = np.flatnonzero((a_near_any < model.align_a_max) & np.isfinite(near_normal[:, 0]))
    sel = cand[rng.random(len(cand)) < model.align_fraction]
    out[sel] = _tilt(near_normal[sel], math.radians(model.align_sigma_deg), rng)
    return out


def circumbinary_planes(pair_normal, rng, model: BinaryModel = BinaryModel()) -> np.ndarray:
    return _tilt(pair_normal, math.radians(model.cb_incl_sigma_deg), rng)


# --- single-star boost ----------------------------------------------------------------

def mock_kepler(systems, components, model: BinaryModel = BinaryModel()):
    """Primaries of a mock magnitude-limited transit survey: (row index, Malmquist weight, S_bin).

    Main-sequence (or pre-MS) stellar and substellar primaries; weight (L_tot / L_1)^1.5 for
    the light of all components (unresolved at Kepler distances); systems brighter than
    ``luminosity_cut`` x L_1 are removed (HM-C's sample cut L <= 1.75 L_MS)."""
    sys_ = np.asarray(components["system"])
    role = np.asarray(components["role"])
    lum = np.nan_to_num(np.asarray(components["luminosity"], dtype=float))
    kind = np.asarray(components["kind"])
    phase = np.asarray(components["phase"], dtype=float)
    l_tot = np.bincount(sys_, weights=lum, minlength=len(systems["age_gyr"]))
    prim = np.flatnonzero((role == 0) & (((kind == "star") & np.isin(phase, (-1, 0))) | (kind == "BD")))
    ratio = l_tot[sys_[prim]] / np.maximum(lum[prim], 1e-30)
    keep = ratio <= model.luminosity_cut
    prim, ratio = prim[keep], ratio[keep]
    st = structure(components)
    return prim, ratio**model.malmquist_exponent, s_bin(st["a_near"][prim], model)


def boost_table(model: BinaryModel = BinaryModel()) -> np.ndarray:
    """(mass-bin centres, B) from a mock Kepler sample of ``boost_systems`` systems; cached."""
    path = BOOST_DIR / f"boost_{model.key()}.npy"
    if path.exists():
        return np.load(path)
    from .stars.population import draw_systems

    systems, comp = draw_systems(model.boost_systems, seed=20261014)
    prim, w, s = mock_kepler(systems, comp, model)
    m = np.asarray(comp["mass"], dtype=float)[prim]
    edges = np.array(model.boost_mass_edges)
    b = np.digitize(m, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    sw = np.bincount(b[ok], weights=w[ok], minlength=len(edges) - 1)
    sws = np.bincount(b[ok], weights=(w * s)[ok], minlength=len(edges) - 1)
    centres = np.sqrt(edges[1:] * edges[:-1])
    table = np.stack([centres, sw / np.maximum(sws, 1e-30)])
    BOOST_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, table)
    return table


def boost_factor(mass, teff=None, model: BinaryModel = BinaryModel()) -> np.ndarray:
    """Single-star boost B(M) (1 when ``model.boost`` is off). With ``teff``, times HM-C's
    efficiency correction at the host's Teff."""
    mass = np.asarray(mass, dtype=float)
    if not model.boost:
        return np.ones_like(mass)
    centres, b = boost_table(model)
    out = np.interp(np.log(mass), np.log(centres), b)
    if teff is not None:
        corr = np.array(model.boost_correction, dtype=float)
        out = out * np.interp(teff, corr[:, 0], corr[:, 1])
    return out


def _ztp_mean(lam):
    return lam / -np.expm1(-lam)


def _ztp_inverse(target):
    """lambda whose zero-truncated Poisson mean is ``target`` (>= 1)."""
    lam = np.maximum(target - 1.0, 1e-6) * 2.0
    for _ in range(60):  # Newton on f(lam) = lam / (1 - e^-lam) - target
        em = np.exp(-lam)
        f = lam / (1.0 - em) - target
        df = (1.0 - em - lam * em) / (1.0 - em) ** 2
        lam = np.maximum(lam - f / df, 1e-9)
    return lam


def apply_boost(rows, colour, boost, cluster_share: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Scale the expected number of inner planets per star by ``boost`` (per host).

    f_swpa is raised to min(1, B f_swpa) through ``f_swpa_med``; what is left (hosts where
    f_swpa would exceed 1) multiplies the mean number of clusters, through lambda_c.
    ``cluster_share`` (0-1): only rest^cluster_share goes into the cluster count (B21).
    Returns the new rows and the factor that went into the cluster count."""
    from .inner import hmc

    rows = np.array(rows, dtype=float, copy=True)
    f = np.clip(rows[:, hmc.I_FSLOPE] * (colour - rows[:, hmc.I_MEDCOL]) + rows[:, hmc.I_FMED], 0.0, 1.0)
    f_new = np.minimum(f * boost, 1.0)
    rows[:, hmc.I_FMED] += f_new - f  # the linear f_swpa(colour) at this host's colour becomes f_new
    rest = np.where(f_new > 0, f * boost / np.where(f_new > 0, f_new, 1.0), 1.0)
    lam = np.exp(rows[:, hmc.I_LRC])
    rest = rest ** cluster_share
    need = rest > 1.0 + 1e-12
    rows[need, hmc.I_LRC] = np.log(_ztp_inverse(_ztp_mean(lam[need]) * rest[need]))
    return rows, rest


# --- empirical circumbinary component ------------------------------------------------

def draw_circumbinary(m_pair, a_min, a_max, p_bin, rng, model: BinaryModel = BinaryModel()) -> dict[str, np.ndarray]:
    """Kepler-like circumbinary planets (Armstrong+2014) for pairs of total mass ``m_pair``,
    HW99 limit ``a_min`` and outer limit ``a_max`` (au), binary period ``p_bin`` (d)."""
    from .units import a_au_to_period, period_to_a_au

    n = len(m_pair)
    host, radius = [], []
    ok = p_bin >= model.cb_p_bin_min
    for (lo, hi), rate in zip(model.cb_radius_bins, model.cb_rates):
        has = np.flatnonzero(ok & (rng.random(n) < min(rate * model.cb_rate_factor, 1.0)))
        host.append(has)
        radius.append(lo + rng.random(len(has)) * (hi - lo))
    host, radius = np.concatenate(host), np.concatenate(radius)
    p_crit = a_au_to_period(a_min[host], m_pair[host])
    lo, hi = np.log(model.cb_period_ratio)
    period = p_crit * np.exp(lo + rng.random(len(host)) * (hi - lo))
    a = period_to_a_au(period, m_pair[host])
    inside = a <= a_max[host]
    host, radius, period, a = host[inside], radius[inside], period[inside], a[inside]
    # At most one planet per host (the rates are fractions of binaries with such a planet);
    # keep a random one.
    perm = rng.permutation(len(host))
    host, radius, period, a = host[perm], radius[perm], period[perm], a[perm]
    _, first = np.unique(host, return_index=True)
    host, radius, period, a = host[first], radius[first], period[first], a[first]
    k = len(host)
    mass = (radius / 0.808) ** (1.0 / 0.589) * 10 ** (model.cb_mass_scatter_dex * rng.normal(size=k))
    ecc = np.minimum(model.cb_ecc_sigma * np.hypot(rng.normal(size=k), rng.normal(size=k)), 0.3)
    return {"host": host, "period": period, "a_au": a, "planet_mass": mass, "planet_radius": radius, "ecc": ecc,
            "omega": 2 * np.pi * rng.random(k), "mean_anom": 2 * np.pi * rng.random(k)}


# --- Kozai-Lidov cycles ------------------------------------------------------------------

KOZAI_VERSION = 1
KL_COS_MAX = math.cos(math.radians(39.23))  # cycles need 39.2 < i < 140.8 deg (quadrupole)
KL_COS_GRID = np.linspace(0.02, KL_COS_MAX, 30)
KL_E0_GRID = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
KL_N_QUANTILES = 201
C_AU_YR = 63241.077  # speed of light, au / yr
GM_SUN_AU_YR = 4.0 * math.pi**2  # au^3 / yr^2


def _kl_rhs(t, y):
    """Test-particle quadrupole Kozai-Lidov equations (time in units of the KL timescale;
    Fabrycky & Tremaine 2007 eqs. with the outer orbit fixed; Naoz 2016 sec. 3)."""
    e, inc, w = y
    e = min(max(e, 1e-9), 1.0 - 1e-9)
    j = math.sqrt(1.0 - e * e)
    s2i, s2w = math.sin(inc) ** 2, math.sin(2.0 * w)
    de = 15.0 / 8.0 * e * j * s2i * s2w
    dw = 0.75 * (2.0 * j * j + 5.0 * math.sin(w) ** 2 * (e * e - s2i)) / j
    di = -15.0 / 16.0 * e * e * s2w * math.sin(2.0 * inc) / j
    return [de, di, dw]


def _kl_quantiles(cos_i0, e0, n_omega=8, t_end=200.0, n_t=20000) -> np.ndarray:
    """Quantiles of e over a uniformly sampled time, pooled over initial arguments of pericentre."""
    from scipy.integrate import solve_ivp

    t = np.linspace(0.0, t_end, n_t)
    samples = []
    for w0 in np.linspace(0.0, np.pi, n_omega, endpoint=False):
        sol = solve_ivp(_kl_rhs, (0.0, t_end), [e0, math.acos(cos_i0), w0], t_eval=t, rtol=1e-8, atol=1e-10,
                        method="DOP853")
        samples.append(np.clip(sol.y[0], 0.0, 1.0))
    return np.quantile(np.concatenate(samples), np.linspace(0.0, 1.0, KL_N_QUANTILES))


@lru_cache(maxsize=1)
def kozai_table() -> np.ndarray:
    """(len(KL_COS_GRID), len(KL_E0_GRID), KL_N_QUANTILES) quantiles of e over a KL cycle; cached."""
    path = BOOST_DIR / f"kozai_v{KOZAI_VERSION}.npy"
    if path.exists():
        return np.load(path)
    table = np.array([[_kl_quantiles(c, e0) for e0 in KL_E0_GRID] for c in KL_COS_GRID])
    BOOST_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, table)
    return table


def kozai_timescale(a, m_host, m_planet_earth, a_c, m_c, e_c) -> np.ndarray:
    """Quadrupole KL timescale (yr; Antognini 2015): 16/(30 pi) (M + m_c)/m_c P_c^2 / P (1 - e_c^2)^1.5."""
    from .units import EARTH_MASS_MSUN

    m_in = m_host + m_planet_earth * EARTH_MASS_MSUN
    p_in = np.sqrt(a**3 / m_in)
    p_out = np.sqrt(a_c**3 / (m_in + m_c))
    return 16.0 / (30.0 * math.pi) * (m_in + m_c) / m_c * p_out**2 / p_in * (1.0 - np.clip(e_c, 0.0, 0.99) ** 2) ** 1.5


def gr_precession_time(a, e, m_host) -> np.ndarray:
    """Time (yr) for general-relativistic apsidal precession to advance by one radian."""
    rate = 3.0 * GM_SUN_AU_YR**1.5 * m_host**1.5 / (C_AU_YR**2 * a**2.5 * (1.0 - np.clip(e, 0.0, 0.99) ** 2))
    return 1.0 / rate


@lru_cache(maxsize=1)
def _laplace_b32_1():
    alpha = np.linspace(0.0, 0.995, 400)
    psi = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    b = (np.cos(psi)[None, :] / (1.0 - 2.0 * alpha[:, None] * np.cos(psi)[None, :] + alpha[:, None] ** 2) ** 1.5)
    return alpha, b.mean(axis=1) * 2.0


def secular_rate(a_i, a_j, m_j, m_host) -> np.ndarray:
    """Laplace-Lagrange apsidal precession rate (rad / yr) of orbit i forced by planet j
    (Murray & Dermott 1999 eq. 7.55): n_i / 4 (m_j / M) alpha alpha_bar b_{3/2}^(1)(alpha)."""
    from .units import EARTH_MASS_MSUN

    inner = a_i < a_j
    alpha = np.where(inner, a_i / a_j, a_j / a_i)
    alpha_bar = np.where(inner, alpha, 1.0)
    grid, b = _laplace_b32_1()
    n_i = 2.0 * math.pi * np.sqrt(m_host / a_i**3)
    return 0.25 * n_i * m_j * EARTH_MASS_MSUN / m_host * alpha * alpha_bar * np.interp(np.minimum(alpha, grid[-1]), grid, b)


def kozai_draw(cos_i0, e0, rng) -> tuple[np.ndarray, np.ndarray]:
    """Eccentricity and cos(inclination to the companion orbit) at a random phase of the KL
    cycle that starts at (e0, i0); the Kozai constant sqrt(1 - e^2) cos i is conserved."""
    table = kozai_table()
    c = np.clip(np.abs(cos_i0), KL_COS_GRID[0], KL_COS_GRID[-1])
    j = np.abs(np.log(np.maximum(e0, 1e-3))[:, None] - np.log(KL_E0_GRID)[None, :]).argmin(axis=1)
    x = np.interp(c, KL_COS_GRID, np.arange(len(KL_COS_GRID)))
    i0 = np.minimum(np.floor(x).astype(int), len(KL_COS_GRID) - 2)
    w = x - i0
    u = rng.random(len(c)) * (KL_N_QUANTILES - 1)
    q0 = np.floor(u).astype(int)
    q1 = np.minimum(q0 + 1, KL_N_QUANTILES - 1)
    fq = u - q0
    lo = table[i0, j, q0] * (1 - fq) + table[i0, j, q1] * fq
    hi = table[i0 + 1, j, q0] * (1 - fq) + table[i0 + 1, j, q1] * fq
    e = np.clip((1 - w) * lo + w * hi, 0.0, 0.999)
    theta = np.sqrt(1.0 - np.clip(e0, 0.0, 0.999) ** 2) * np.abs(cos_i0)
    cos_i = np.sign(cos_i0) * np.clip(theta / np.sqrt(1.0 - e**2), 0.0, 1.0)
    return e, cos_i
