"""Small planets (below 30 Earth masses) beyond HM-C's outer period limit (300 d).

Occurrence density per star, f(M, a) = d2N / (d ln M  d ln a), from two anchors:

**Beyond the snow line: microlensing.** The planet-to-star mass-ratio function of Zang+2025
(KMTNet 2016-2019, 63 planets, Science 388, 400), lower ("super-Earth") component of their
double-Gaussian fit (Table S5):

    dN / (d log q  d log s) = A1 * 10^(n1 (log q - log q_peak)^2)

with log q_peak = -4.67 (+0.19 -0.33) (7.4 Earth-to-Sun mass ratios) and n1 = -0.73
(+0.31 -0.41) (a width of 0.55 dex), normalised by the integral over -5.2 < log q < -1.5,
0.57 +- 0.13 planets per star per dex of s. The upper (giant) component is left to
:mod:`starsys.outer.giants`; below 30 Earth masses it adds < 1% for any host. The rate is
flat in log s (Suzuki+2016 find s^m with m = 0.49 +0.47 -0.49), so it is flat in ln a.
Microlensing hosts are mostly 0.1-1 Msun; using a universal mass ratio, the planet masses
scale with M* (Pascucci+2018 find a host-independent mass-ratio break in Kepler). The
separation is in units of the snow line, a_sl = 2.7 au (M* / Msun) (Kennedy & Kenyon 2008,
the microlensing convention). For a typical lens (0.5 Msun at 6 kpc, source at 8 kpc;
Einstein radius ~2.5 au) the sensitive range s ~ 0.5-2 corresponds to ~1-4 a_sl. The
rate is used flat out to ``a_max`` = 10 a_sl (the Solar System's ice giants sit at 7-11
a_sl). Poleski+2021 (OGLE, s = 2-6) find more wide-orbit planets than this (1.4 +0.9 -0.6
per star with q > 1e-4, rising with s); see :mod:`starsys.validation.small`.

Zang+2025's main-text numbers are per dex of q and per dex of s (their eq. 1 integrates to
the quoted 0.65 per star over -5.2 < log q < -1.5). Their sensitivity omits the
binary-lens degeneracies, which bias the rates low by 5-10% (Zang+2025b, arXiv
2509.16191). That is well within the quoted errors and is not corrected.

**At 300 d: HM-C.** The inner anchor is HM-C's own output at its outer edge: the number of
planets per star per ln P at P_max/2-P_max and the mean and standard deviation of their
ln(initial mass), measured once per posterior catalog and host Teff (with the M-dwarf
extension) by running the HM-C port (:func:`edge_table`, cached in ``data/outer``). Per
host, the rate is scaled by the host's own f_swp. The mass distribution is treated as
lognormal.

**Bridge, P_max to a_sl: log-linear interpolation.** Between ln a(P_max) and ln a_sl, the
log density is interpolated linearly in ln a for every mass (x = 0 at P_max, 1 at a_sl):

    ln f(M, a) = (1 - x) ln f_HMC(M) + x ln f_ML(M)

Both anchors are Gaussian in ln M, so the bridge is Gaussian in ln M at every a, with a
mean moving from HM-C's (~1.5 Earth masses for Sun-like stars) to the microlensing peak
(~7 M* / Msun Earth masses). Pascucci+2018 find the same 3-10x shift in typical
mass ratio between Kepler (inside the snow line) and microlensing (outside). Nothing
measures this region directly: Kepler is nearly blind beyond ~300 d, and η⊕ estimates
(Bryson+2021) are extrapolations of power laws fitted at shorter periods. The bridge is
checked against those (:mod:`starsys.validation.small`) and flagged. When the snow line
is inside a(P_max) (hosts below ~0.3 Msun) there is no bridge: the microlensing rate
starts at P_max.

**Uncertainty.** The microlensing parameters (A1 through the integrated rate, log q_peak,
n1) are drawn per system from split-normal approximations of Zang+2025's posteriors
(independent, so correlations are not kept). The HM-C anchor follows the system's HM-C
posterior catalog. ``sample_uncertainty=False`` uses the central values.

**Hosts below 0.1 Msun:** rate scaled by (M / 0.1)^1.9 (disk mass, Pascucci+2016), as in
:mod:`starsys.inner.mdwarf`.

**Masses and radii:** 0.1-30 Earth masses (HM-C's lower mass limit; giants above). The
radius uses HM-C's mass-radius relation (initial H/He envelope) and its
photoevaporation, with the host's parameters (born-bare fraction for M dwarfs included).
Ices are not modelled (phase 5).

**Orbits:** eccentricity Rayleigh(``ecc_sigma`` = 0.05) and mutual inclination to the
system's reference plane (the inner invariable plane, else random) Rayleigh(1.4 deg).
These continue HM-C's multi-planet distributions (mean e 0.056, mean inclination to the
invariable plane 0.0275 rad), close to the Solar System's (mean e of the terrestrial and ice giant
planets 0.04). No data constrain them beyond 300 d.

**Packing:** planets are placed in order of decreasing mass. Each must keep HM-C's
spacing of ``num_mutual_hill_radii`` (10) mutual Hill radii from the outermost inner
planet and from other small planets, and be stable (:func:`starsys.outer.giants.pair_stable`,
AMD-Hill or Petrovich 2015) with each neighbour, giants included. Otherwise its orbit
and mass are redrawn from the host's density, up to ``n_redraw`` times, after which it is
dropped. Giants and inner planets are already placed; they are not moved.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import ndtr, ndtri

from ..inner import hmc
from ..inner.mdwarf import MDwarfModel, adjust_params
from ..paths import DATA_DIR
from ..stars.spectral import dwarf_mass_radius
from ..units import EARTH_MASS_MSUN, a_au_to_period, period_to_a_au
from .giants import orient, pair_stable

LN10 = math.log(10.0)
EDGE_DIR = DATA_DIR / "outer"
EDGE_VERSION = 1  # bump when the edge measurement changes
# Teff knots of the HM-C edge table; hosts outside are clamped.
EDGE_TEFF = np.array([2400.0, 2700.0, 3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0, 4300.0, 4700.0, 5100.0,
                      5500.0, 5900.0, 6300.0, 6800.0, 7200.0])


@dataclass(frozen=True)
class SmallPlanetModel:
    # Zang+2025, lower Gaussian: (median, +err, -err).
    ml_rate: tuple = (0.57, 0.13, 0.13)  # per star per dex of s, integrated over ml_log_q_range
    ml_log_q_peak: tuple = (-4.67, 0.19, 0.33)
    ml_n: tuple = (-0.73, 0.31, 0.41)
    ml_log_q_range: tuple = (-5.2, -1.5)  # measured range (lowest detection at -5.17)
    ml_log_q_peak_bounds: tuple = (-6.0, -3.5)  # Zang+2025's prior bounds
    ml_n_max: float = -0.15  # caps the width of a drawn Gaussian at ~1.1 dex
    sample_uncertainty: bool = True
    # Separation, in units of the snow line.
    snow_line_au: float = 2.7  # at 1 Msun, proportional to M*
    ml_a_range: tuple = (1.0, 4.0)  # probed by microlensing
    a_max: float = 10.0
    ml_host_mass_range: tuple = (0.1, 1.0)
    # Planet masses (Earth masses).
    m_min: float = 0.1
    m_max: float = 30.0
    # HM-C edge window: P_max * edge_fraction to P_max.
    edge_fraction: float = 0.5
    edge_stars: int = 2000  # per posterior catalog and Teff knot (noise ~3%, posterior spread ~13%)
    # Orbits.
    ecc_sigma: float = 0.05
    incl_sigma_deg: float = 1.4
    n_redraw: int = 20
    # Hosts below the hydrogen-burning limit.
    substellar_mass: float = 0.1
    disk_mass_slope: float = 1.9
    n_grid: int = 64


# --- inner anchor: HM-C at its outer edge ------------------------------------------------

def _edge_stats(rows, mstar, rstar, teff, n_stars, frac, seed) -> np.ndarray:
    """(k, 3): planets per host per ln P at P_max*frac-P_max, mean ln(initial mass) minus the
    row's mean_ln_mass, and SD of ln(initial mass), for k parameter rows (f_swp forced to 1)."""
    k = len(rows)
    r = np.repeat(rows, n_stars, axis=0)
    r[:, hmc.I_FMED], r[:, hmc.I_FSLOPE] = 1.0, 0.0
    rep = [np.repeat(x, n_stars) for x in (mstar, rstar, teff)]
    _, p = hmc.generate_catalog(r, *rep, np.ones(len(r)), seed=seed)
    knot = p["star_index"] // n_stars
    pmax = rows[knot, hmc.I_PMAX]
    sel = (p["period"] >= frac * pmax) & (p["period"] <= pmax)
    out = np.empty((k, 3))
    y = np.log(p["mass_init"])
    for j in range(k):
        s = sel & (knot == j)
        out[j] = s.sum() / n_stars / math.log(1.0 / frac), y[s].mean() - rows[j, hmc.I_MUM], y[s].std()
    return out


_EDGE_MEMO: dict[str, np.ndarray] = {}


def edge_table(base_rows, mdwarf: MDwarfModel | None, model: SmallPlanetModel | None = None) -> np.ndarray:
    """HM-C edge statistics, shape (n_rows, len(EDGE_TEFF), 3) (see :func:`_edge_stats`), for
    each HM-C parameter row (e.g. the 100 posterior catalogs) on main-sequence hosts at the
    Teff knots. Cached in memory and in ``data/outer``."""
    model = model or SmallPlanetModel()
    base_rows = np.ascontiguousarray(base_rows, dtype=float)
    tag = repr((mdwarf, EDGE_TEFF.tolist(), model.edge_stars, model.edge_fraction, EDGE_VERSION)).encode()
    key = hashlib.sha1(base_rows.tobytes() + tag).hexdigest()[:16]
    if key in _EDGE_MEMO:
        return _EDGE_MEMO[key]
    path = EDGE_DIR / f"edge_{key}.npy"
    if path.exists():
        table = np.load(path)
    else:
        mass, radius = dwarf_mass_radius(EDGE_TEFF)
        table = np.empty((len(base_rows), len(EDGE_TEFF), 3))
        for i, row in enumerate(base_rows):
            rows = np.broadcast_to(row, (len(EDGE_TEFF), len(row)))
            if mdwarf is not None:
                rows = adjust_params(rows, mass, EDGE_TEFF, mdwarf)
            table[i] = _edge_stats(np.array(rows), mass, radius, EDGE_TEFF, model.edge_stars, model.edge_fraction,
                                   seed=1000 + i)
        EDGE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, table)
    _EDGE_MEMO[key] = table
    return table


def inner_anchor(rows, table_rows, teff, colour):
    """Per host: HM-C planets per star per ln a at P_max, and mean and SD of their ln mass.

    ``rows``: the host's (M-dwarf adjusted) HM-C parameters; ``table_rows``: its slice of
    :func:`edge_table`, shape (n, knots, 3); ``colour``: the host colour used for f_swp."""
    f_swp = np.clip(rows[:, hmc.I_FSLOPE] * (colour - rows[:, hmc.I_MEDCOL]) + rows[:, hmc.I_FMED], 0.0, 1.0)
    t = np.clip(np.asarray(teff, dtype=float), EDGE_TEFF[0], EDGE_TEFF[-1])
    j = np.clip(np.searchsorted(EDGE_TEFF, t) - 1, 0, len(EDGE_TEFF) - 2)
    w = (t - EDGE_TEFF[j]) / (EDGE_TEFF[j + 1] - EDGE_TEFF[j])
    n = np.arange(len(t))
    lo, hi = table_rows[n, j], table_rows[n, j + 1]
    rate = np.exp((1 - w) * np.log(np.maximum(lo[:, 0], 1e-12)) + w * np.log(np.maximum(hi[:, 0], 1e-12)))
    mu = rows[:, hmc.I_MUM] + (1 - w) * lo[:, 1] + w * hi[:, 1]
    sd = (1 - w) * lo[:, 2] + w * hi[:, 2]
    return 1.5 * f_swp * rate, mu, sd  # d ln P = 1.5 d ln a


# --- outer anchor: microlensing ------------------------------------------------------------

def _split_normal(centre, up, lo, rng, n):
    z = rng.normal(size=n)
    return centre + np.where(z >= 0, up, lo) * z


def draw_ml_params(n, rng, model: SmallPlanetModel = SmallPlanetModel()) -> dict[str, np.ndarray]:
    """Per-system microlensing parameters: integrated rate, log q_peak and n1."""
    if not model.sample_uncertainty:
        return {"rate": np.full(n, model.ml_rate[0]), "log_q_peak": np.full(n, model.ml_log_q_peak[0]),
                "n": np.full(n, model.ml_n[0])}
    rate = model.ml_rate[0] * np.exp(rng.normal(size=n) * model.ml_rate[1] / model.ml_rate[0])  # lognormal
    peak = np.clip(_split_normal(*model.ml_log_q_peak, rng, n), *model.ml_log_q_peak_bounds)
    nn = np.minimum(_split_normal(*model.ml_n, rng, n), model.ml_n_max)
    return {"rate": rate, "log_q_peak": peak, "n": nn}


def outer_anchor(mstar, ml, model: SmallPlanetModel = SmallPlanetModel()):
    """Per host: (ln of the peak density per ln M per ln a, mean and SD of ln M (Earth masses))."""
    sd_dex = 1.0 / np.sqrt(2.0 * -ml["n"] * LN10)
    lo, hi = model.ml_log_q_range
    integral = sd_dex * math.sqrt(2 * math.pi) * (ndtr((hi - ml["log_q_peak"]) / sd_dex)
                                                  - ndtr((lo - ml["log_q_peak"]) / sd_dex))
    amp = ml["rate"] / integral  # A1, per dex of q per dex of s
    mstar = np.asarray(mstar, dtype=float)
    disk = np.where(mstar < model.substellar_mass, (mstar / model.substellar_mass) ** model.disk_mass_slope, 1.0)
    c = np.log(amp * disk / LN10**2)
    mu = ml["log_q_peak"] * LN10 + np.log(mstar / EARTH_MASS_MSUN)
    return c, mu, sd_dex * LN10


# --- density ---------------------------------------------------------------------------

def _bridge(u, u_in, u_sl, c_in, mu_in, s_in, c_out, mu_out, s_out):
    """Gaussian in ln M at ln a = u: (ln peak density, mean, SD). Inputs broadcast."""
    span = u_sl - u_in
    x = np.where(span > 0, np.clip((u - u_in) / np.where(span > 0, span, 1.0), 0.0, 1.0), 1.0)
    t_in, t_out = (1 - x) / s_in**2, x / s_out**2
    tau = t_in + t_out
    mu = (t_in * mu_in + t_out * mu_out) / tau
    with np.errstate(invalid="ignore"):
        c = np.where(x < 1, (1 - x) * c_in, 0.0) + np.where(x > 0, x * c_out, 0.0)
    c = c - 0.5 * (t_in * mu_in**2 + t_out * mu_out**2 - tau * mu**2)
    return c, mu, 1.0 / np.sqrt(tau)


def _marginal(c, mu, sd, y_lo, y_hi):
    """Density per ln a: the Gaussian integrated over ln M in [y_lo, y_hi]."""
    frac = ndtr((y_hi - mu) / sd) - ndtr((y_lo - mu) / sd)
    return np.exp(c) * sd * math.sqrt(2 * math.pi) * frac


class _Hosts:
    """Per-host density parameters and the ln a grid with its cumulative counts."""

    def __init__(self, mstar, a_in, anchors_in, anchors_out, model: SmallPlanetModel):
        self.model = model
        self.mstar = np.asarray(mstar, dtype=float)
        self.a_sl = model.snow_line_au * self.mstar
        self.u_in = np.log(a_in)
        self.u_sl = np.log(self.a_sl)
        self.u_out = np.log(model.a_max * self.a_sl)
        n_in, mu_in, s_in = anchors_in
        self.p_in = (np.log(np.maximum(n_in, 1e-12) / (s_in * math.sqrt(2 * math.pi))), mu_in, s_in)
        self.p_out = anchors_out
        self.y_lo, self.y_hi = math.log(model.m_min), math.log(model.m_max)
        g = np.linspace(0.0, 1.0, model.n_grid)
        span = np.maximum(self.u_out - self.u_in, 0.0)
        self.grid = self.u_in[:, None] + span[:, None] * g[None, :]
        dens = _marginal(*self._gauss(self.grid, slice(None), extra_dim=True), self.y_lo, self.y_hi)
        steps = 0.5 * (dens[:, 1:] + dens[:, :-1]) * np.diff(self.grid, axis=1)
        self.cum = np.concatenate([np.zeros((len(self.mstar), 1)), np.cumsum(steps, axis=1)], axis=1)
        self.expected = self.cum[:, -1]
        # Rows offset so that one searchsorted inverts every host's cumulative count.
        self._step = self.expected.max(initial=0.0) + 1.0
        self._flat = (self.cum + self._step * np.arange(len(self.cum))[:, None]).ravel()

    def _gauss(self, u, idx, extra_dim=False):
        def take(v):
            v = np.asarray(v)[idx]
            return v[:, None] if extra_dim else v
        return _bridge(u, take(self.u_in), take(self.u_sl), *(take(v) for v in self.p_in),
                       *(take(v) for v in self.p_out))

    def cum_at(self, u, idx):
        """Expected number of planets between ln a(P_max) and ln a = u (clipped to the grid)."""
        idx = np.asarray(idx)
        g = self.grid[idx]
        du = (g[:, -1] - g[:, 0]) / (self.model.n_grid - 1)
        pos = np.clip((u - g[:, 0]) / np.where(du > 0, du, 1.0), 0.0, self.model.n_grid - 1)
        j = np.minimum(pos.astype(int), self.model.n_grid - 2)
        c = self.cum[idx]
        r = np.arange(len(idx))
        return c[r, j] + (pos - j) * (c[r, j + 1] - c[r, j])

    def draw(self, idx, rng):
        """(a in au, mass in Earth masses) for hosts ``idx`` (one draw each)."""
        idx = np.asarray(idx, dtype=np.int64)
        target = rng.random(len(idx)) * self.expected[idx]
        # Inverse CDF on each host's grid, vectorised by offsetting the rows.
        ng = self.model.n_grid
        j = np.searchsorted(self._flat, target + self._step * idx, side="right") - idx * ng - 1
        j = np.clip(j, 0, ng - 2)
        c0, c1 = self.cum[idx, j], self.cum[idx, j + 1]
        w = np.where(c1 > c0, (target - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
        u = self.grid[idx, j] + np.clip(w, 0.0, 1.0) * (self.grid[idx, j + 1] - self.grid[idx, j])
        _, mu, sd = self._gauss(u, idx)
        lo, hi = ndtr((self.y_lo - mu) / sd), ndtr((self.y_hi - mu) / sd)
        p = np.clip(lo + rng.random(len(idx)) * (hi - lo), 1e-300, 1 - 1e-16)
        y = np.clip(mu + sd * ndtri(p), self.y_lo, self.y_hi)
        return np.exp(u), np.exp(y)


# --- radii -----------------------------------------------------------------------------

@njit(cache=True)
def _radii(m_init, period, rows, mstar, rstar, teff, age, seed):
    """HM-C radius (initial envelope) and photoevaporation for given initial masses."""
    np.random.seed(seed)
    n = len(m_init)
    r_init = np.empty(n)
    r_final = np.empty(n)
    m_final = m_init.copy()
    lost = np.zeros(n, dtype=np.bool_)
    f_oplus = hmc.bolometric_flux(365.25, 1.0, hmc.luminosity_cgs(1.0, 5780.0))
    for k in range(n):
        p = rows[k]
        m = m_init[k]
        mu_r, s_r = hmc.mean_radius_and_scatter(m, p[hmc.I_RNORM], p[hmc.I_MBREAK], p[hmc.I_G0], p[hmc.I_G1],
                                                p[hmc.I_S0], p[hmc.I_S1])
        r = mu_r
        for _ in range(10):
            r = mu_r + s_r * mu_r * np.random.normal()
            if p[hmc.I_RMIN] < r < p[hmc.I_RMAX]:
                break
        r = min(max(r, p[hmc.I_RMIN]), p[hmc.I_RMAX])
        r_init[k] = r
        r_final[k] = r
        m_env = hmc.envelope_mass(m)
        lum = hmc.luminosity_cgs(rstar[k], teff[k])
        t_loss = hmc.mass_loss_timescale_gyr(m_env, r, hmc.bolometric_flux(period[k], mstar[k], lum), f_oplus)
        p_ret = min(math.exp(p[hmc.I_LALPHA]) * t_loss / age[k], 1.0) * (1.0 - p[hmc.I_FBARE])
        if not (np.random.random() < p_ret):
            lost[k] = True
            m_final[k] = m - m_env
            r_si = hmc.radius_pure_silicate_seager2007(m_final[k])
            r_final[k] = r_si + 0.05 * r_si * np.random.normal()
    return m_final, r_final, r_init, lost


# --- packing ---------------------------------------------------------------------------

def _hill_ok(m1, m2, a1, a2, mstar, n_hill) -> bool:
    """HM-C's mutual-Hill spacing (circular orbits), a1 < a2, masses in Earth masses."""
    mu = (m1 + m2) * EARTH_MASS_MSUN / mstar
    return a2 - a1 >= n_hill * 0.5 * (a1 + a2) * (mu / 3.0) ** (1.0 / 3.0)


def _fits(a, m, e, occ, mstar, n_hill) -> bool:
    """Is (a, m, e) stable with its nearest neighbours in ``occ`` = sorted lists (a, m, e, giant)?"""
    occ_a, occ_m, occ_e, occ_g = occ
    j = bisect.bisect_left(occ_a, a)
    if j > 0:
        a1, m1, e1, g1 = occ_a[j - 1], occ_m[j - 1], occ_e[j - 1], occ_g[j - 1]
        if not g1 and not _hill_ok(m1, m, a1, a, mstar, n_hill):
            return False
        if not pair_stable(m1, m, a1, e1, a, e, mstar):
            return False
    if j < len(occ_a):
        a2, m2, e2, g2 = occ_a[j], occ_m[j], occ_e[j], occ_g[j]
        if not g2 and not _hill_ok(m, m2, a, a2, mstar, n_hill):
            return False
        if not pair_stable(m, m2, a, e, a2, e2, mstar):
            return False
    return True


def _draw_ecc(n, rng, model):
    return np.minimum(model.ecc_sigma * np.hypot(rng.normal(size=n), rng.normal(size=n)), 0.5)


# --- the generator ---------------------------------------------------------------------

def draw_small(mstar, rstar, teff, rows, table_rows, colour, ml, rng, model: SmallPlanetModel = SmallPlanetModel(),
               occupied=None, planes=None, age=None, a_limits=None) -> dict[str, np.ndarray]:
    """Draw small planets beyond HM-C's P_max for an array of hosts.

    Parameters
    ----------
    mstar, rstar, teff
        Host mass (Msun), radius (Rsun) and Teff (K).
    rows, table_rows, colour
        Per host: (M-dwarf adjusted) HM-C parameters, the matching slice of
        :func:`edge_table` and the colour used for f_swp.
    ml
        Per-host microlensing parameters (:func:`draw_ml_params`).
    occupied
        Optional dict of per-body arrays already placed (``host``, ``a_au``, ``planet_mass``,
        ``ecc``, ``giant``): the outermost inner planet and the giants of each host.
    planes
        Optional (n, 3) reference-plane normals; NaN rows get a random plane.
    age
        Per-host age (Gyr) for photoevaporation; default the HM-C ``system_age``.
    a_limits
        Optional (a_min, a_max) arrays per host (au), dynamical limits set by stellar
        companions: candidates outside are dropped and redraws must fall inside.

    Returns
    -------
    Per-planet arrays sorted by host and period: ``host``, ``period``, ``a_au``,
    ``planet_mass``, ``planet_radius``, ``mass_init``, ``radius_init``, ``lost_envelope``,
    ``ecc``, ``omega``, ``mean_anom``, ``incl_mut``, ``asc_node``, ``incl_sky``,
    ``bridge`` (inside the snow line) and ``extrapolated`` (beyond the snow line but outside
    the microlensing-measured mass ratios, separations or host masses).
    """
    mstar = np.asarray(mstar, dtype=float)
    n = len(mstar)
    a_in = period_to_a_au(rows[:, hmc.I_PMAX], mstar)
    hosts = _Hosts(mstar, a_in, inner_anchor(rows, table_rows, teff, colour), outer_anchor(mstar, ml, model), model)
    host = np.repeat(np.arange(n), rng.poisson(hosts.expected))
    a, m = hosts.draw(host, rng)
    e = _draw_ecc(len(host), rng, model)
    if a_limits is not None:
        inside = (a >= a_limits[0][host]) & (a <= a_limits[1][host])
        host, a, m, e = host[inside], a[inside], m[inside], e[inside]

    def outside(h, a_):
        return a_limits is not None and not (a_limits[0][h] <= a_ <= a_limits[1][h])

    # Placement, by host in order of decreasing mass.
    keep = np.ones(len(host), dtype=bool)
    occ_by_host: dict[int, list] = {}
    if occupied is not None:
        oh = np.asarray(occupied["host"])
        o = np.lexsort((occupied["a_au"], oh))
        cols = [np.asarray(occupied[key])[o] for key in ("a_au", "planet_mass", "ecc", "giant")]
        oh = oh[o]
        for h in np.intersect1d(np.unique(host), oh):
            lo, hi = np.searchsorted(oh, h, "left"), np.searchsorted(oh, h, "right")
            occ_by_host[int(h)] = [list(c[lo:hi]) for c in cols]
    starts = np.searchsorted(host, np.arange(n + 1))
    for h in np.flatnonzero(np.diff(starts) > 0):
        idx = np.arange(starts[h], starts[h + 1])
        occ = occ_by_host.get(int(h), [[], [], [], []])
        if len(idx) == 1 and not occ[0]:
            continue
        n_hill = rows[h, hmc.I_NHILL]
        for j in idx[np.argsort(-m[idx])]:
            for attempt in range(model.n_redraw + 1):
                if attempt:
                    aa, mm = hosts.draw(np.array([h]), rng)
                    a[j], m[j], e[j] = aa[0], mm[0], _draw_ecc(1, rng, model)[0]
                    if outside(h, a[j]):
                        continue
                if _fits(a[j], m[j], e[j], occ, mstar[h], n_hill):
                    pos = bisect.bisect_left(occ[0], a[j])
                    for lst, val in zip(occ, (a[j], m[j], e[j], False)):
                        lst.insert(pos, val)
                    break
            else:
                keep[j] = False
    host, a, m, e = host[keep], a[keep], m[keep], e[keep]

    period = a_au_to_period(a, mstar[host])
    if age is None:
        age = rows[:, hmc.I_AGE]
    m_final, r_final, r_init, lost = _radii(m, period, np.ascontiguousarray(rows[host]), mstar[host],
                                            np.asarray(rstar, dtype=float)[host], np.asarray(teff, dtype=float)[host],
                                            np.asarray(age, dtype=float)[host], int(rng.integers(2**31)))
    a_sl = hosts.a_sl[host]
    log_q = np.log10(m * EARTH_MASS_MSUN / mstar[host])
    lo, hi = model.ml_host_mass_range
    bridge = a < a_sl
    extrap = ~bridge & ((log_q < model.ml_log_q_range[0]) | (a > model.ml_a_range[1] * a_sl)
                        | (mstar[host] < lo) | (mstar[host] > hi))
    k = len(host)
    out = {
        "host": host, "period": period, "a_au": a, "planet_mass": m_final, "planet_radius": r_final,
        "mass_init": m, "radius_init": r_init, "lost_envelope": lost, "ecc": e,
        "omega": 2.0 * np.pi * rng.random(k), "mean_anom": 2.0 * np.pi * rng.random(k),
        "bridge": bridge, "extrapolated": extrap,
    }
    order = np.lexsort((out["period"], out["host"]))
    out = {key: val[order] for key, val in out.items()}
    out.update(orient(out["host"], planes, n, rng, model))
    return out


def expected_number(mstar, rows, table_rows, teff, colour, ml, model: SmallPlanetModel = SmallPlanetModel(),
                    a_range=None):
    """Expected number of small planets per host (before packing), optionally between two
    semi-major axes (au; arrays allowed)."""
    mstar = np.asarray(mstar, dtype=float)
    a_in = period_to_a_au(rows[:, hmc.I_PMAX], mstar)
    hosts = _Hosts(mstar, a_in, inner_anchor(rows, table_rows, teff, colour), outer_anchor(mstar, ml, model), model)
    if a_range is None:
        return hosts.expected
    idx = np.arange(len(mstar))
    lo, hi = (np.log(np.broadcast_to(np.asarray(x, dtype=float), mstar.shape)) for x in a_range)
    return hosts.cum_at(hi, idx) - hosts.cum_at(lo, idx)
