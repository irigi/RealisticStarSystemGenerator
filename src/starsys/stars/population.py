"""Random stellar systems of the solar neighbourhood.

Ingredients (all solar-neighbourhood, volume-limited):

- Component initial-mass function: the 20-pc census of Kirkpatrick et al. (2024,
  ApJS 271, 55), Table 18 (space density per initial-mass bin; white dwarfs are
  counted at their progenitor masses, brown-dwarf masses come from their
  simulations). Finest available binning is used (0.005 / 0.02 / 0.1 Msun).
  Above 8 Msun (no such stars within 20 pc) a Salpeter slope is appended.
- Multiplicity fraction vs primary mass: the same census for 0.1-1.3 Msun, where it
  is well sampled, and Duchene & Kraus (2013, ARA&A 51, 269; Table 1) at the
  low- and high-mass ends where close companions are under-counted in the census.
- Separations and mass ratios: Duchene & Kraus (2013) Table 1 (log-normal period
  distributions; mass-ratio power-law index gamma), interpolated in log primary mass.
- Eccentricities: tidal circularisation envelope of Moe & Di Stefano (2017),
  e_max = 1 - (P / 2 d)^(-2/3), with f(e) ~ e^0.4 below it (approximate).
- Ages: thin disk 0-10 Gyr with a present-day local distribution p(age) ~ exp(-age/5.5 Gyr)
  plus a 4% thick-disk component at 10-12.5 Gyr. The decline and thick-disk fraction
  were fitted to the census white-dwarf count and white-dwarf Teff distribution.
  Components of a system are coeval.
- [Fe/H]: thin disk N(0.03 - 0.025 * age/Gyr, 0.18); thick disk N(-0.5, 0.2)
  (approximate local age-metallicity relation).

The primary-mass function is obtained by iteratively deconvolving the census
component mass function so that primaries + companions reproduce it.

Bright subsets (for the galaxy's brightness tiers, :mod:`starsys.galaxy`): ``draw_systems``
can draw only systems whose primary is a living star (age below its MIST lifetime) with an
initial mass in a range, or only the others. The same model is sampled, conditioned:
masses from the primary-mass function times P(age < lifetime) of the population's age
distribution, ages from that distribution truncated at the lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from astropy.table import Table

from ..paths import CATALOGS_DIR, DATA_DIR
from ..units import YEAR_DAYS
from .evolution import evolve

CENSUS_DIR = CATALOGS_DIR / "observational" / "stellar_census"
MIN_MASS = 0.01  # Msun; ~10 Jupiter masses, the lower edge of the brown-dwarf IMF used here
MAX_MASS = 120.0
SALPETER = 2.35

# --- multiplicity anchors: (log10 M1, value) -------------------------------------
# Multiplicity fraction (at least one companion).
_MF = np.array([[np.log10(0.05), 0.22], [np.log10(0.15), 0.24], [np.log10(0.3), 0.26],
                [np.log10(0.45), 0.31], [np.log10(0.6), 0.40], [np.log10(0.85), 0.44],
                [np.log10(1.15), 0.50], [np.log10(1.6), 0.55], [np.log10(2.5), 0.62],
                [np.log10(5.0), 0.72], [np.log10(12.0), 0.85]])
# P(>= 2 companions | multiple): ~0.1 for M dwarfs, ~0.25 FGK (Raghavan+2010), higher for A/B.
_P_HIGHER = np.array([[np.log10(0.1), 0.10], [np.log10(0.4), 0.12], [np.log10(1.0), 0.25],
                      [np.log10(3.0), 0.35], [np.log10(12.0), 0.45]])
# Mass-ratio power-law index gamma (f(q) ~ q^gamma).
_GAMMA = np.array([[np.log10(0.05), 4.2], [np.log10(0.3), 0.4], [np.log10(1.0), 0.3], [np.log10(3.0), -0.5]])
# Log-normal period distribution: log10 of peak separation (au) and sigma of log10 P(days).
_LOGA = np.array([[np.log10(0.05), np.log10(4.5)], [np.log10(0.3), np.log10(5.3)], [np.log10(1.0), np.log10(45.0)],
                  [np.log10(3.0), np.log10(350.0)]])
_SIGP = np.array([[np.log10(0.05), 0.5], [np.log10(0.3), 1.3], [np.log10(1.0), 2.3], [np.log10(3.0), 2.0]])
# Intermediate/high-mass primaries have an additional close (P ~ 10 d) mode (DK13).
_CLOSE_MODE = np.array([[np.log10(1.3), 0.0], [np.log10(2.0), 0.3], [np.log10(12.0), 0.5]])


def _interp(table, log_m):
    return np.interp(log_m, table[:, 0], table[:, 1])


@lru_cache(maxsize=1)
def census_imf():
    """Bin edges and space densities (pc^-3 per bin) of the 20-pc component IMF."""
    t = Table.read(CENSUS_DIR / "J_ApJS_271_55_table18.ecsv").to_pandas()
    lo, hi, rho = t["MassBin0"].to_numpy(), t["MassBin1"].to_numpy(), t["SpaceDensity"].to_numpy()
    width = np.round(hi - lo, 4)
    pick = ((width == 0.005) & (hi <= 0.4)) | ((width == 0.02) & (lo >= 0.4) & (hi <= 1.6)) | \
           ((width == 0.1) & (lo >= 1.6))
    lo, hi, rho = lo[pick], hi[pick], rho[pick]
    order = np.argsort(lo)
    lo, hi, rho = lo[order], hi[order], rho[order]
    keep = hi > MIN_MASS
    lo, hi, rho = np.maximum(lo[keep], MIN_MASS), hi[keep], rho[keep]
    # Salpeter extension 8 -> 120 Msun, continuous in dN/dM at 8 Msun.
    dndm_8 = rho[-1] / (hi[-1] - lo[-1])
    edges = np.geomspace(8.0, MAX_MASS, 25)
    ext = dndm_8 * 8.0**SALPETER / (1 - SALPETER) * (edges[1:] ** (1 - SALPETER) - edges[:-1] ** (1 - SALPETER))
    return np.concatenate([lo, edges[:-1]]), np.concatenate([hi, edges[1:]]), np.concatenate([rho, ext])


def _sample_binned(rng, lo, hi, weights, n):
    k = rng.choice(len(weights), size=n, p=weights / weights.sum())
    u = rng.random(n)
    # log-uniform inside each bin (bins are narrow; this avoids piling up at edges for wide bins)
    return np.exp(np.log(lo[k]) + u * (np.log(hi[k]) - np.log(lo[k])))


@dataclass
class PopulationModel:
    """Tunable assumptions; defaults are the documented choices above."""

    thin_disk_max_age: float = 10.0
    # youngest thin-disk age (Gyr); the galaxy uses 1 Myr so that the most massive stars
    # (lifetimes < 10 Myr) exist somewhere (starsys.galaxy)
    thin_disk_min_age: float = 0.01
    # Present-day local age distribution of the thin disk, p(age) ~ exp(-age / tau).
    # A constant star-formation rate combined with vertical disk heating (older stars
    # spend less time near the mid-plane) makes the local sample younger than the SFH.
    # Calibrated against the 20-pc white-dwarf count and Teff distribution (docs/validation/census.md).
    age_decline_gyr: float = 5.5
    thick_disk_fraction: float = 0.04
    thick_disk_ages: tuple[float, float] = (10.0, 12.5)
    feh_thin_intercept: float = 0.03
    feh_thin_slope: float = -0.025
    feh_thin_sigma: float = 0.18
    feh_thick: tuple[float, float] = (-0.5, 0.2)
    min_mass: float = MIN_MASS
    primary_weights: np.ndarray | None = field(default=None, repr=False)

    # --- elementary draws ---------------------------------------------------------
    def ages_and_feh(self, rng, n):
        thick = rng.random(n) < self.thick_disk_fraction
        age = np.where(thick, rng.uniform(*self.thick_disk_ages, n), self._thin_ages(rng, n))
        feh = np.where(thick, rng.normal(*self.feh_thick, n),
                       rng.normal(self.feh_thin_intercept + self.feh_thin_slope * age, self.feh_thin_sigma))
        return age, np.clip(feh, -2.0, 0.5)

    def _thin_ages(self, rng, n):
        return self._thin_inv(rng.random(n))

    def _thin_inv(self, u):
        lo, hi, tau = self.thin_disk_min_age, self.thin_disk_max_age, self.age_decline_gyr
        if not np.isfinite(tau):
            return lo + u * (hi - lo)
        # inverse CDF of a truncated exponential
        a, b = np.exp(-lo / tau), np.exp(-hi / tau)
        return -tau * np.log(a - u * (a - b))

    def _thin_cdf(self, t):
        lo, hi, tau = self.thin_disk_min_age, self.thin_disk_max_age, self.age_decline_gyr
        t = np.clip(np.asarray(t, float), lo, hi)
        if not np.isfinite(tau):
            return (t - lo) / (hi - lo)
        a, b = np.exp(-lo / tau), np.exp(-hi / tau)
        return (a - np.exp(-t / tau)) / (a - b)

    def _thick_cdf(self, t):
        t0, t1 = self.thick_disk_ages
        return np.clip((np.asarray(t, float) - t0) / (t1 - t0), 0.0, 1.0)

    def age_cdf(self, t):
        """P(age < t) of the population (t in Gyr)."""
        f = self.thick_disk_fraction
        return (1 - f) * self._thin_cdf(t) + f * self._thick_cdf(t)

    def ages_and_feh_below(self, rng, tmax):
        """Ages and [Fe/H] conditioned on age < ``tmax`` (array, Gyr; needs age_cdf(tmax) > 0)."""
        tmax = np.asarray(tmax, float)
        n = len(tmax)
        f = self.thick_disk_fraction
        c_thin, c_thick = (1 - f) * self._thin_cdf(tmax), f * self._thick_cdf(tmax)
        thick = rng.random(n) * np.maximum(c_thin + c_thick, 1e-300) < c_thick
        u = rng.random(n)
        t0, t1 = self.thick_disk_ages
        age = np.where(thick, t0 + u * self._thick_cdf(tmax) * (t1 - t0),
                       self._thin_inv(u * self._thin_cdf(tmax)))
        feh = np.where(thick, rng.normal(*self.feh_thick, n),
                       rng.normal(self.feh_thin_intercept + self.feh_thin_slope * age, self.feh_thin_sigma))
        return np.minimum(age, tmax), np.clip(feh, -2.0, 0.5)

    def _alive_bins(self, m_lo, m_hi):
        lo, hi, rho = census_imf()
        w = rho if self.primary_weights is None else rho * self.primary_weights
        blo, bhi = np.maximum(lo, m_lo), np.minimum(hi, m_hi)
        ok = bhi > blo
        blo, bhi, w = blo[ok], bhi[ok], w[ok] * np.log(bhi[ok] / blo[ok]) / np.log(hi[ok] / lo[ok])
        life = lifetime_gyr(np.sqrt(blo * bhi), np.full(len(blo), FEH_REF))
        return blo, bhi, w * self.age_cdf(life), w

    def alive_fraction(self, m_lo, m_hi=np.inf) -> float:
        """Fraction of systems whose primary has an initial mass in [m_lo, m_hi) and is still a
        star (not a remnant) today."""
        lo, hi, rho = census_imf()
        w_all = rho if self.primary_weights is None else rho * self.primary_weights
        return float(self._alive_bins(m_lo, m_hi)[2].sum() / w_all.sum())

    def draw_alive_primaries(self, rng, n, m_lo, m_hi=np.inf):
        """Initial masses, ages and [Fe/H] of ``n`` primaries in [m_lo, m_hi) alive today."""
        blo, bhi, wa, _ = self._alive_bins(m_lo, m_hi)
        m1 = _sample_binned(rng, blo, bhi, wa, n)
        age, feh = np.zeros(n), np.zeros(n)
        todo = np.arange(n)
        for _ in range(40):
            life = lifetime_gyr(m1[todo], np.full(len(todo), FEH_REF))
            a, z = self.ages_and_feh_below(rng, life)
            # the lifetime depends (weakly) on [Fe/H]: keep the draws that are alive with theirs
            good = a < lifetime_gyr(m1[todo], z)
            age[todo[good]], feh[todo[good]] = a[good], z[good]
            todo = todo[~good]
            if not len(todo):
                break
        if len(todo):
            feh[todo] = FEH_REF
            age[todo] = 0.5 * lifetime_gyr(m1[todo], feh[todo])
        return m1, age, feh

    def draw_primaries(self, rng, n):
        lo, hi, rho = census_imf()
        w = rho if self.primary_weights is None else rho * self.primary_weights
        return _sample_binned(rng, lo, hi, w, n)

    def draw_companion_count(self, rng, m1):
        lm = np.log10(m1)
        multiple = rng.random(len(m1)) < _interp(_MF, lm)
        higher = rng.random(len(m1)) < _interp(_P_HIGHER, lm)
        quad = rng.random(len(m1)) < 0.2
        return multiple * (1 + higher * (1 + quad))

    def draw_mass_ratio(self, rng, m1):
        gamma = _interp(_GAMMA, np.log10(m1))
        qmin = np.clip(self.min_mass / m1, 0.0, 0.99)
        u = rng.random(len(m1))
        g1 = gamma + 1.0
        return (qmin**g1 + u * (1.0 - qmin**g1)) ** (1.0 / g1)

    def draw_log_period(self, rng, m1):
        lm = np.log10(m1)
        mtot_guess = 1.5 * m1
        loga = _interp(_LOGA, lm)
        # log P(days) at the peak separation, Kepler's third law.
        logp_peak = np.log10(YEAR_DAYS) + 1.5 * loga - 0.5 * np.log10(mtot_guess)
        logp = rng.normal(logp_peak, _interp(_SIGP, lm))
        close = rng.random(len(m1)) < _interp(_CLOSE_MODE, lm)
        logp = np.where(close, rng.normal(1.0, 0.5, len(m1)), logp)
        return logp

    @staticmethod
    def draw_eccentricity(rng, period_days):
        emax = np.clip(1.0 - (period_days / 2.0) ** (-2.0 / 3.0), 0.0, 0.95)
        return emax * rng.random(len(period_days)) ** (1.0 / 1.4)


def calibrate_primary_weights(model: PopulationModel, n=400_000, iterations=8, seed=0):
    """Iteratively reweight the primary-mass function so that the mass function of all
    components (primaries + companions) matches the census IMF."""
    rng = np.random.default_rng(seed)
    lo, hi, rho = census_imf()
    target = rho / rho.sum()
    model.primary_weights = np.ones_like(rho)
    for _ in range(iterations):
        m1 = model.draw_primaries(rng, n)
        nc = model.draw_companion_count(rng, m1)
        comp = [m1]
        for k in range(1, 4):
            sel = nc >= k
            comp.append(m1[sel] * model.draw_mass_ratio(rng, m1[sel]))
        allm = np.concatenate(comp)
        hist = np.histogram(allm, bins=np.append(lo, hi[-1]))[0].astype(float)
        hist /= hist.sum()
        ratio = np.where(hist > 0, target / np.maximum(hist, 1e-12), 1.0)
        model.primary_weights *= np.clip(ratio, 0.2, 5.0)
    return model


@lru_cache(maxsize=1)
def default_model() -> PopulationModel:
    cache = DATA_DIR / "stellar" / "primary_weights.npy"
    model = PopulationModel()
    if cache.exists() and len(np.load(cache)) == len(census_imf()[2]):
        model.primary_weights = np.load(cache)
    else:
        calibrate_primary_weights(model)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, model.primary_weights)
    return model


FEH_REF = 0.0  # [Fe/H] for the lifetime used to weight masses (the exact one decides "alive")


def lifetime_gyr(initial_mass, feh) -> np.ndarray:
    """MIST lifetime (Gyr) of stars of initial mass >= 0.1 Msun; lower masses never die."""
    from . import mist
    m = np.atleast_1d(np.asarray(initial_mass, float))
    out = np.full(len(m), np.inf)
    hi = m >= 0.1
    if hi.any():
        out[hi] = 10 ** mist.lifetime(m[hi], np.broadcast_to(np.asarray(feh, float), m.shape)[hi]) / 1e9
    return out


MA_COEFF = 2.8   # Mardling & Aarseth (2001), eq. 90, without the inclination factor (conservative)


def stable_triple(a_in, a_out, e_out, q_out):
    """Mardling & Aarseth (2001) stability of a hierarchical triple: outer periastron over
    inner semi-major axis above 2.8 [(1 + q_out)(1 + e_out) / sqrt(1 - e_out)]^(2/5), with
    q_out = m_out / m_inner. (The inclination factor 1 - 0.3 i/pi, which only relaxes the
    limit, is left out.)"""
    lim = MA_COEFF * ((1 + q_out) * (1 + e_out) / np.sqrt(1 - e_out)) ** 0.4
    return a_out * (1 - e_out) > lim * a_in


def _stabilise(model, rng, m1, sys_idx, mi, period, a_au, ecc, mass_in):
    """Redraw (period, eccentricity) of the outer orbits that are unstable by
    :func:`stable_triple` against the next inner orbit, in place, level by level. The redraws
    come from a child stream of ``rng`` (its state is not advanced), so stable systems are
    exactly as before this check existed. Gives up after 200 tries with a circular orbit at
    the limit (rare: very wide inner pairs)."""
    if len(sys_idx) < 3:
        return
    fix = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
    for k in range(2, len(sys_idx)):
        inner = np.searchsorted(sys_idx[k - 1], sys_idx[k])
        a_in = a_au[k - 1][inner]
        q = mi[k] / mass_in[k]
        mtot = mass_in[k] + mi[k]
        bad = np.flatnonzero(~stable_triple(a_in, a_au[k], ecc[k], q))
        for _ in range(200):
            if not len(bad):
                break
            m = m1[sys_idx[k][bad]]
            logp = model.draw_log_period(fix, m)
            a = (mtot[bad] * (10**logp / YEAR_DAYS) ** 2) ** (1 / 3)
            e = model.draw_eccentricity(fix, 10**logp)
            ok = stable_triple(a_in[bad], a, e, q[bad]) & (a <= 4e4)
            i = bad[ok]
            a_au[k][i], period[k][i], ecc[k][i] = a[ok], 10**logp[ok], e[ok]
            bad = bad[~ok]
        if len(bad):
            lim = MA_COEFF * (1 + q[bad]) ** 0.4 * a_in[bad] * 1.01
            a_au[k][bad] = np.maximum(a_au[k][bad], lim)
            ecc[k][bad] = 0.0
            period[k][bad] = YEAR_DAYS * np.sqrt(a_au[k][bad] ** 3 / mtot[bad])


def draw_systems(n, seed=None, model: PopulationModel | None = None, alive_range=None, exclude_alive_above=None):
    """Draw ``n`` stellar systems.

    ``alive_range=(m_lo, m_hi)``: only systems whose primary has an initial mass in that range
    and is still a star today; ``exclude_alive_above=m``: no system whose primary is a living
    star of initial mass >= m (the complement). Default: all systems.

    Returns ``(systems, components)`` as dicts of arrays. Components carry
    ``system`` (index), ``role`` (0 primary, k companion), the evolved properties
    from :func:`starsys.stars.evolution.evolve`, and the companion's orbit around
    the inner subsystem (``period_days``, ``a_au``, ``ecc`` and the orientation ``incl_sky``,
    ``node``, ``omega`` in radians, isotropic; NaN for primaries).
    Hierarchy: companion k orbits the centre of mass of components 0..k-1, with
    orbits at least 3x wider than the previous one and, with eccentricities, stable by the
    Mardling & Aarseth (2001) criterion (see :func:`_stabilise`).
    """
    model = model or default_model()
    rng = np.random.default_rng(seed)
    if alive_range is not None:
        m1, age, feh = model.draw_alive_primaries(rng, n, *alive_range)
    else:
        m1 = model.draw_primaries(rng, n)
        age, feh = model.ages_and_feh(rng, n)
        if exclude_alive_above is not None:
            for _ in range(100):
                bad = np.flatnonzero((m1 >= exclude_alive_above) & (age < lifetime_gyr(m1, feh)))
                if not len(bad):
                    break
                m1[bad] = model.draw_primaries(rng, len(bad))
                age[bad], feh[bad] = model.ages_and_feh(rng, len(bad))
    ncomp = model.draw_companion_count(rng, m1)

    sys_idx, role, mi = [np.arange(n)], [np.zeros(n, int)], [m1]
    period, a_au, ecc = [np.full(n, np.nan)], [np.full(n, np.nan)], [np.full(n, np.nan)]
    inner_mass = m1.copy()
    last_a = np.zeros(n)
    mass_in = [None]
    for k in range(1, 4):
        sel = np.flatnonzero(ncomp >= k)
        if len(sel) == 0:
            break
        mk = m1[sel] * model.draw_mass_ratio(rng, m1[sel])
        mtot = inner_mass[sel] + mk
        logp = model.draw_log_period(rng, m1[sel])
        a = (mtot * (10**logp / YEAR_DAYS) ** 2) ** (1 / 3)
        # Enforce hierarchy and a minimum separation of 0.02 au (contact) and maximum
        # of 0.2 pc (Galactic tidal limit ~ 1-2e5 au for ~1 Msun binaries).
        for _ in range(50):
            bad = (a < np.maximum(3 * last_a[sel], 0.02)) | (a > 4e4)
            if not bad.any():
                break
            logp[bad] = model.draw_log_period(rng, m1[sel][bad])
            a[bad] = (mtot[bad] * (10 ** logp[bad] / YEAR_DAYS) ** 2) ** (1 / 3)
        a = np.clip(a, np.maximum(3 * last_a[sel], 0.02), 4e4)
        p = YEAR_DAYS * np.sqrt(a**3 / mtot)
        sys_idx.append(sel)
        role.append(np.full(len(sel), k))
        mi.append(mk)
        period.append(p)
        a_au.append(a)
        ecc.append(model.draw_eccentricity(rng, p))
        mass_in.append(inner_mass[sel].copy())
        inner_mass[sel] = mtot
        last_a[sel] = a
    _stabilise(model, rng, m1, sys_idx, mi, period, a_au, ecc, mass_in)

    sys_idx = np.concatenate(sys_idx)
    order = np.lexsort((np.concatenate(role), sys_idx))
    comp = {
        "system": sys_idx[order],
        "role": np.concatenate(role)[order],
        "period_days": np.concatenate(period)[order],
        "a_au": np.concatenate(a_au)[order],
        "ecc": np.concatenate(ecc)[order],
    }
    mi_all = np.concatenate(mi)[order]
    comp.update(evolve(mi_all, age[comp["system"]], feh[comp["system"]], rng=rng))
    # Orientation of each companion's orbit (isotropic; sky normal = +z). Drawn last so that
    # everything above is unchanged by it.
    n_c = len(mi_all)
    companion = comp["role"] > 0
    comp["incl_sky"] = np.where(companion, np.arccos(2.0 * rng.random(n_c) - 1.0), np.nan)
    comp["node"] = np.where(companion, 2.0 * np.pi * rng.random(n_c), np.nan)
    comp["omega"] = np.where(companion, 2.0 * np.pi * rng.random(n_c), np.nan)
    systems = {"age_gyr": age, "feh": feh, "n_components": ncomp + 1, "primary_initial_mass": m1}
    return systems, comp
