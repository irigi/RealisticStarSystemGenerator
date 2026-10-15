"""Stellar systems around a real star: the model's stars, fitted to the measured ones.

Each realisation is a draw from the posterior of the system's age and [Fe/H] (shared by its
components) and each component's initial mass, given every measurement of every component:

    p(age, feh, m_1..m_n | data) ∝ prior(age, feh) Π_c IMF(m_c) L_c(evolve(m_c, age, feh))

The prior is the population model's (:mod:`starsys.stars.population`): the local disk's age and
age-metallicity distribution and the 20-pc census IMF. The present state comes from the same
evolution code as random systems (MIST, white-dwarf cooling, substellar tracks), so the star is
the model's star closest to the real one, not a copy of the catalogue values. Measured Teff,
luminosity, radius, mass, log g, [Fe/H] and age enter as Gaussians (log-normal for luminosity,
radius and masses) with the measurement error and a floor for the models' systematics
(``StarFitModel``).

The age-metallicity part is importance-sampled (``n_age`` draws from a proposal around the
measured values); for each draw the components' masses are integrated on a log grid, first
coarse, then refined around the posterior. Because components share the age, a white dwarf's
cooling age or a subgiant companion constrains every star of the system.

Companions are the census's: their projected separation (from the census positions and the
distance) is turned into an orbit by drawing the unseen orbital phase, eccentricity and
inclination. Unresolved pairs get separations from the population model below the resolution
limit. The census hierarchy is mapped onto the generator's chain (companion k orbits
components 0..k-1); 2+2 quadruples become chains, which is flagged. Stars beyond 20 pc are
single unless the NASA archive lists companions (drawn from the population model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..stars import population as pop
from ..stars.evolution import evolve, spectral_type_of
from ..units import YEAR_DAYS
from .catalog import ComponentRecord, Measurement, SystemRecord, find

MAX_AGE = 13.5
# Quantities fitted in log (relative errors) and linearly.
_LOG_QUANTITIES = ("luminosity", "radius", "mass", "initial_mass")
_LIN_QUANTITIES = ("teff", "log_g")
# Unresolved companions (no WDS or census separation) are drawn below this separation (arcsec).
RESOLUTION_ARCSEC = 0.5
FLAG_HIERARCHY_APPROX = 1  # system flag: the census hierarchy is not a chain


@dataclass(frozen=True)
class StarFitModel:
    n_age: int = 4000  # (age, [Fe/H]) draws
    n_mass: int = 160  # initial-mass grid points per component and pass
    min_ess: float = 300.0  # below this, a second, adapted round of (age, [Fe/H]) draws
    # Error floors: model systematics (MIST vs interferometric radii and Teff ~1-3%; bolometric
    # corrections; metallicity scales).
    floor_rel: dict = field(default_factory=lambda: {"teff": 0.02, "luminosity": 0.05, "radius": 0.04,
                                                     "mass": 0.03, "initial_mass": 0.03})
    floor_abs: dict = field(default_factory=lambda: {"log_g": 0.1, "feh": 0.05, "age": 0.1})
    peculiar_feh_err: float = 0.3  # Am/Ap stars: surface abundances are not the bulk metallicity


@dataclass
class StarSystemFit:
    """Posterior draws of a real system, ready for :func:`starsys.planets.attach_planets`."""

    record: SystemRecord
    systems: dict
    components: dict
    target: np.ndarray  # component row of the looked-up star in each realisation
    ess: float  # effective sample size of the (age, [Fe/H]) importance sampling
    flags: int = 0

    def summary(self) -> str:
        c, rec = self.components, self.record
        lines = [f"{rec.name}: {len(rec.components)} component(s), {len(self.target)} realisations, "
                 f"ESS {self.ess:.0f}"]
        age, feh = self.systems["age_gyr"], self.systems["feh"]
        lines.append(f"  system age {_q(age)} Gyr, [Fe/H] {_q(feh)}")
        n_comp = len(rec.components)
        for j, comp in enumerate(rec.components):
            rows = np.arange(len(self.target)) * n_comp + j
            mark = "*" if j == rec.target else " "
            kinds, counts = np.unique(c["kind"][rows], return_counts=True)
            kind = ", ".join(f"{k} {n / len(rows):.0%}" for k, n in zip(kinds, counts))
            lines.append(f" {mark}{comp.name} [{kind}]: M {_q(c['mass'][rows])} "
                         f"(initial {_q(c['initial_mass'][rows])}),"
                         f" Teff {_q(c['teff'][rows], 0)}, L {_q(c['luminosity'][rows])}, R {_q(c['radius'][rows])}")
            if j > 0:
                lines.append(f"     a {_q(c['a_au'][rows])} au, e {_q(c['ecc'][rows])}")
            for name, m in comp.measurements.items():
                key = {"age": None, "feh": None}.get(name, name)
                if key is None or key not in c:
                    continue
                lines.append(f"       measured {name} {m}")
        return "\n".join(lines)


def _q(x, digits=3) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "n/a"
    lo, med, hi = np.percentile(x, [16, 50, 84])
    fmt = f"{{:.{digits}g}}" if digits else "{:.0f}"
    return f"{fmt.format(med)} [{fmt.format(lo)}-{fmt.format(hi)}]"


# --- priors -----------------------------------------------------------------------------

def _norm_pdf(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)


def age_feh_prior(age, feh, model: pop.PopulationModel) -> np.ndarray:
    """Density of the population model's (age, [Fe/H]) distribution (clipping of [Fe/H] ignored)."""
    age, feh = np.asarray(age, dtype=float), np.asarray(feh, dtype=float)
    lo, hi, tau = 0.01, model.thin_disk_max_age, model.age_decline_gyr
    norm = tau * (np.exp(-lo / tau) - np.exp(-hi / tau))
    thin = np.where((age >= lo) & (age <= hi), np.exp(-age / tau) / norm, 0.0)
    thin = thin * _norm_pdf(feh, model.feh_thin_intercept + model.feh_thin_slope * age, model.feh_thin_sigma)
    a0, a1 = model.thick_disk_ages
    thick = np.where((age >= a0) & (age <= a1), 1.0 / (a1 - a0), 0.0) * _norm_pdf(feh, *model.feh_thick)
    return (1 - model.thick_disk_fraction) * thin + model.thick_disk_fraction * thick


def imf_density(m) -> np.ndarray:
    """dN/dm of the census component mass function (per Msun, unnormalised)."""
    lo, hi, rho = pop.census_imf()
    m = np.asarray(m, dtype=float)
    k = np.clip(np.searchsorted(hi, m), 0, len(hi) - 1)
    inside = (m >= lo[k]) & (m <= hi[k])
    return np.where(inside, rho[k] / (hi[k] - lo[k]), 0.0)


# --- measurements -----------------------------------------------------------------------

def _effective(meas: dict[str, Measurement], comp: ComponentRecord, model: StarFitModel) -> dict:
    """Measurements used in the fit, with error floors: {quantity: (value, sigma)} (sigma relative for
    log quantities)."""
    out = {}
    for q, m in meas.items():
        if not np.isfinite(m.value):
            continue
        if q in _LOG_QUANTITIES:
            if m.value <= 0:
                continue
            rel = np.hypot(m.err / m.value, model.floor_rel.get(q, 0.0))
            out[q] = (np.log(m.value), rel)
        elif q in _LIN_QUANTITIES:
            err = m.err
            if q == "teff":
                err = np.hypot(err, model.floor_rel["teff"] * m.value)
            else:
                err = np.hypot(err, model.floor_abs.get(q, 0.0))
            out[q] = (m.value, err)
        elif q in ("feh", "age"):
            err = np.hypot(m.err, model.floor_abs[q] * (m.value if q == "age" else 1.0))
            if q == "feh" and _peculiar(comp.spectral_type):
                err = max(err, model.peculiar_feh_err)
            out[q] = (m.value, err)
    # The census adopts a measured (dynamical) mass also as the initial mass: count it once. Initial
    # masses of white dwarfs (IFMR) are kept next to their present masses.
    if "mass" in out and "initial_mass" in out and comp.kind_hint != "WD":
        out.pop("initial_mass")
    return out


def _peculiar(sptype: str) -> bool:
    return bool(re.search(r"^[AB]\d?(\.\d)?\s*[mp]|[AB]\d(\.\d)?.*\b(p|m[AF])", sptype or ""))


def _mass_guess(comp: ComponentRecord) -> tuple[float, float, float]:
    """(centre, low, high) of the initial-mass search range."""
    ms = comp.measurements
    if comp.kind_hint == "WD":
        if "initial_mass" in ms:
            m = ms["initial_mass"]
            return m.value, max(0.75, m.value - 4 * m.err), min(8.0, m.value + 4 * m.err)
        return 2.0, 0.75, 8.0
    for q in ("mass", "initial_mass"):
        if q in ms:
            m = ms[q]
            f = np.exp(max(5 * m.err / m.value, 0.25))
            hi = m.value * f
            return m.value, max(0.01, m.value / f), min(hi, 20.0) if m.value < 15 else hi
    if comp.kind_hint == "BD":
        return 0.05, 0.01, 0.12
    return 1.0, 0.07, 15.0


# --- the fit ----------------------------------------------------------------------------

def _component_loglike(state, eff, hint) -> np.ndarray:
    """log L of model states (dict of arrays from ``evolve``) for one component's measurements."""
    ll = np.zeros(len(state["mass"]))
    for q, (v, s) in eff.items():
        if q in ("feh", "age"):
            continue
        x = state[q]
        if q in _LOG_QUANTITIES:
            ll += -0.5 * ((np.log(np.maximum(x, 1e-300)) - v) / s) ** 2
        else:
            ll += -0.5 * ((x - v) / s) ** 2
    kind = state["kind"]
    if hint == "WD":
        ll = np.where(kind == "WD", ll, -np.inf)
    elif hint == "star":
        ll = np.where((kind == "star") | (kind == "BD"), ll, -np.inf)
    else:  # brown-dwarf spectral types: substellar or the lowest-mass stars
        ll = np.where(np.isin(kind, ["BD", "star"]), ll, -np.inf)
    return np.where(np.isfinite(ll), ll, -np.inf)


@dataclass
class _MassGrid:
    """Initial masses (flat) for every (age, feh) draw: draw i owns ``m[start[i]:start[i + 1]]``."""

    m: np.ndarray
    start: np.ndarray
    lw: np.ndarray  # log(likelihood x IMF x dm)

    def log_evidence(self) -> np.ndarray:
        """log of the mass integral per draw (-inf where nothing fits)."""
        n = len(self.start) - 1
        out = np.full(n, -np.inf)
        size = np.diff(self.start)
        has = size > 0
        mx = np.full(n, -np.inf)
        mx[has] = np.maximum.reduceat(self.lw, self.start[:-1][has])
        ok = np.isfinite(mx)
        if not ok.any():
            return out
        rep = np.repeat(np.where(ok, mx, 0.0), size)
        with np.errstate(over="ignore"):
            e = np.exp(self.lw - rep)
        tot = np.zeros(n)
        tot[has] = np.add.reduceat(e, self.start[:-1][has])
        out[ok] = np.log(tot[ok]) + mx[ok]
        return out

    def sample(self, draws, rng) -> np.ndarray:
        """One mass per selected draw, by weight; continuous within a cell of the uniform grid."""
        out = np.empty(len(draws))
        for j, i in enumerate(draws):
            a, b = self.start[i], self.start[i + 1]
            lw = self.lw[a:b]
            w = np.exp(lw - lw.max())
            k = rng.choice(b - a, p=w / w.sum())
            m = self.m[a:b]
            lo = 0.5 * (m[k] + m[k - 1]) if k > 0 else m[k]
            hi = 0.5 * (m[k] + m[k + 1]) if k < b - a - 1 else m[k]
            # Inside a fast phase the nodes are dense and the cell tiny; on the main sequence the
            # mass is spread over the cell.
            out[j] = lo + (hi - lo) * rng.random()
        return out

    def mass_range(self, lo_q=1e-4, hi_q=1 - 1e-4, weights=None):
        draw_of = np.repeat(np.arange(len(self.start) - 1), np.diff(self.start))
        w = np.exp(self.lw - np.max(self.lw))
        if weights is not None:
            w = w * weights[draw_of]
        if not np.isfinite(w).any() or w.sum() <= 0:
            return None
        order = np.argsort(self.m)
        cdf = np.cumsum(w[order]) / w.sum()
        return self.m[order][min(np.searchsorted(cdf, lo_q), len(cdf) - 1)], \
            self.m[order][min(np.searchsorted(cdf, hi_q), len(cdf) - 1)]


CHUNK = 1_000_000


def _mass_pass(age, feh, lo, hi, n_mass, eff, hint, nodes: bool) -> _MassGrid:
    """Likelihood x IMF on a log-mass grid (plus the MIST isochrone nodes of the nearest isochrone in
    [lo, hi] with ``nodes``) for every (age, feh) draw."""
    uni = np.geomspace(lo, hi, n_mass)
    per = []
    if nodes:
        from ..stars import mist
        fehs, log_ages, offsets, counts, table = mist.load_grid()
        fi = np.abs(fehs[None, :] - np.asarray(feh)[:, None]).argmin(axis=1)
        ai = np.abs(log_ages[None, :] - np.log10(np.asarray(age) * 1e9)[:, None]).argmin(axis=1)
        cache = {}
        for f, a in zip(fi, ai):
            k = (f, a)
            if k not in cache:
                m = table[offsets[f, a]:offsets[f, a] + counts[f, a], 0]
                cache[k] = np.unique(np.concatenate([uni, m[(m > lo) & (m < hi)]]))
            per.append(cache[k])
    else:
        per = [uni] * len(age)
    size = np.array([len(x) for x in per])
    start = np.concatenate([[0], np.cumsum(size)])
    m = np.concatenate(per)
    draw_of = np.repeat(np.arange(len(age)), size)
    # Trapezoid widths within each draw's grid.
    dm = np.empty_like(m)
    for i in range(len(age)):
        x = m[start[i]:start[i + 1]]
        g = np.gradient(x) if len(x) > 1 else np.ones(1)
        dm[start[i]:start[i + 1]] = g
    ll = np.empty(len(m))
    for c0 in range(0, len(m), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        st = evolve(m[sl], np.asarray(age)[draw_of[sl]], np.asarray(feh)[draw_of[sl]], with_spectral_type=False)
        ll[sl] = _component_loglike(st, eff, hint)
    with np.errstate(divide="ignore"):
        lw = ll + np.log(imf_density(m) * dm)
    return _MassGrid(m=m, start=start, lw=lw)


def _evolved_hint(comp: ComponentRecord) -> bool:
    """Whether the measurements point at a star off the main sequence (fast phases need the
    isochrone nodes)."""
    ms = comp.measurements
    if comp.kind_hint != "star":
        return False
    if "log_g" in ms and ms["log_g"].value < 3.9:
        return True
    r = ms["radius"].value if "radius" in ms else np.nan
    if not np.isfinite(r) and "luminosity" in ms and "teff" in ms:
        r = np.sqrt(ms["luminosity"].value) / (ms["teff"].value / 5772.0) ** 2
    t = ms["teff"].value if "teff" in ms else np.nan
    return bool(np.isfinite(r) and (r > 3.0 or (np.isfinite(t) and t < 7500 and r > 1.8)))


def fit_masses(age, feh, comp: ComponentRecord, eff, model: StarFitModel) -> _MassGrid:
    """Initial-mass grids and log weights of one component for every (age, feh) draw."""
    centre, lo, hi = _mass_guess(comp)
    nodes = _evolved_hint(comp)
    g = _mass_pass(age, feh, lo, hi, model.n_mass, eff, comp.kind_hint, nodes)
    # Refine around the posterior (marginalised over age): the 0.01-99.99% range of the first pass.
    r = g.mass_range()
    if r is not None:
        lo2, hi2 = r[0] / 1.02, r[1] * 1.02
        if np.log(hi2 / lo2) < 0.9 * np.log(hi / lo):
            g = _mass_pass(age, feh, max(lo2, lo), min(hi2, hi), model.n_mass, eff, comp.kind_hint, nodes)
    return g


def fit(record: SystemRecord, n: int = 1000, seed=None, model: StarFitModel = StarFitModel(),
        population: pop.PopulationModel | None = None) -> StarSystemFit:
    """Posterior realisations of ``record``'s stellar system."""
    rng = np.random.default_rng(seed)
    population = population or pop.default_model()
    comps = list(record.components)
    effs = [_effective(c.measurements, c, model) for c in comps]

    # System-level measurements: [Fe/H] and age from any component (precision-weighted).
    feh_meas = [e["feh"] for e in effs if "feh" in e]
    age_meas = [e["age"] for e in effs if "age" in e]

    def evaluate(age, feh, log_q):
        logw = np.log(np.maximum(age_feh_prior(age, feh, population), 1e-300)) - log_q
        for m, s in feh_meas:
            logw = logw - 0.5 * ((feh - m) / s) ** 2
        for m, s in age_meas:
            logw = logw - 0.5 * ((age - m) / s) ** 2
        grids = []
        for c, e in zip(comps, effs):
            g = fit_masses(age, feh, c, e, model)
            logw = logw + g.log_evidence()
            grids.append(g)
        return logw, grids

    age, feh, log_q = _propose_age_feh(rng, model.n_age, feh_meas, age_meas, population)
    logw, grids = evaluate(age, feh, log_q)
    age1, feh1 = age, feh
    if not np.isfinite(logw).any():
        raise ValueError(f"no model star matches the measurements of {record.name}")
    ess = _ess(logw)
    if ess < model.min_ess:
        # Second round: a Gaussian fitted to the first round's posterior (covariance x4), mixed with
        # the prior.
        age2, feh2, log_q = _adapted_proposal(rng, model.n_age, age1, feh1, logw, population)
        logw2, grids2 = evaluate(age2, feh2, log_q)
        if np.isfinite(logw2).any() and _ess(logw2) > ess:
            age, feh, logw, grids, ess = age2, feh2, logw2, grids2, _ess(logw2)
    na = len(age)
    w = np.exp(logw - np.max(logw))
    w /= w.sum()
    pick = rng.choice(na, size=n, p=w)
    sys_age, sys_feh = age[pick], feh[pick]
    masses = np.column_stack([g.sample(pick, rng) for g in grids])  # n x n_comp
    systems, components, flags = _build(record, sys_age, sys_feh, masses, rng)
    target = np.arange(n) * len(comps) + record.target
    return StarSystemFit(record=record, systems=systems, components=components, target=target, ess=ess, flags=flags)


def _combine(meas):
    w = np.array([1.0 / s**2 for _, s in meas])
    v = np.array([m for m, _ in meas])
    return float((w * v).sum() / w.sum()), float(1.0 / np.sqrt(w.sum()))


def age_feh_prior_age(age, model: pop.PopulationModel) -> np.ndarray:
    """Marginal age density of the population model."""
    age = np.asarray(age, dtype=float)
    lo, hi, tau = 0.01, model.thin_disk_max_age, model.age_decline_gyr
    norm = tau * (np.exp(-lo / tau) - np.exp(-hi / tau))
    thin = np.where((age >= lo) & (age <= hi), np.exp(-age / tau) / norm, 0.0)
    a0, a1 = model.thick_disk_ages
    thick = np.where((age >= a0) & (age <= a1), 1.0 / (a1 - a0), 0.0)
    return (1 - model.thick_disk_fraction) * thin + model.thick_disk_fraction * thick


def _draw_feh_given_age(rng, age, model: pop.PopulationModel) -> np.ndarray:
    """[Fe/H] from the population model's conditional distribution at each age."""
    a0, a1 = model.thick_disk_ages
    lo, hi, tau = 0.01, model.thin_disk_max_age, model.age_decline_gyr
    norm = tau * (np.exp(-lo / tau) - np.exp(-hi / tau))
    thin_d = (1 - model.thick_disk_fraction) * np.where(age <= hi, np.exp(-age / tau) / norm, 0.0)
    thick_d = model.thick_disk_fraction * np.where((age >= a0) & (age <= a1), 1.0 / (a1 - a0), 0.0)
    p_thick = thick_d / np.maximum(thin_d + thick_d, 1e-300)
    thick = rng.random(len(age)) < p_thick
    return np.where(thick, rng.normal(*model.feh_thick, len(age)),
                    rng.normal(model.feh_thin_intercept + model.feh_thin_slope * age, model.feh_thin_sigma))


def _propose_age_feh(rng, n, feh_meas, age_meas, population: pop.PopulationModel):
    """(age, feh, log proposal density).

    Proposal: an equal mixture of the prior and a distribution centred on the measurements
    (age: normal of twice the error, truncated to the prior's range; [Fe/H]: normal of twice the
    error plus 0.05; an unmeasured one follows the prior)."""
    from scipy.stats import truncnorm

    a_lo, a_hi = 0.01, max(population.thick_disk_ages[1], population.thin_disk_max_age)
    age_p, feh_p = population.ages_and_feh(rng, n)
    if not feh_meas and not age_meas:
        return age_p, feh_p, np.log(np.maximum(age_feh_prior(age_p, feh_p, population), 1e-300))
    if age_meas:
        mu_a, s_a = _combine(age_meas)
        ta = truncnorm((a_lo - mu_a) / (2 * s_a), (a_hi - mu_a) / (2 * s_a), loc=mu_a, scale=2 * s_a)
        age_g = ta.rvs(size=n, random_state=rng)
        g_age = ta.pdf
    else:
        age_g, _ = population.ages_and_feh(rng, n)
        def g_age(x):
            return age_feh_prior_age(x, population)
    if feh_meas:
        mu_f, s_f = _combine(feh_meas)
        feh_g = rng.normal(mu_f, 2 * s_f + 0.05, n)
        def g_feh(f, a):
            return _norm_pdf(f, mu_f, 2 * s_f + 0.05)
    else:
        feh_g = _draw_feh_given_age(rng, age_g, population)
        def g_feh(f, a):
            return age_feh_prior(a, f, population) / np.maximum(age_feh_prior_age(a, population), 1e-300)
    pick = rng.random(n) < 0.5
    age = np.where(pick, age_g, age_p)
    feh = np.where(pick, feh_g, feh_p)
    q = 0.5 * age_feh_prior(age, feh, population) + 0.5 * g_age(age) * g_feh(feh, age)
    return age, feh, np.log(np.maximum(q, 1e-300))


def _ess(logw) -> float:
    w = np.exp(logw - np.max(logw))
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))


def _adapted_proposal(rng, n, age, feh, logw, population: pop.PopulationModel, prior_share=0.1):
    """Draws from 0.9 N(mean, 4 cov) of the weighted (age, feh) + 0.1 prior; returns (age, feh, log q)."""
    w = np.exp(logw - np.max(logw))
    w /= w.sum()
    x = np.column_stack([age, feh])
    mu = w @ x
    cov = (x - mu).T @ ((x - mu) * w[:, None]) * 4.0
    cov += np.diag([0.02 * max(mu[0], 0.05), 0.01]) ** 2  # floor for single-point posteriors
    a_lo, a_hi = 0.01, max(population.thick_disk_ages[1], population.thin_disk_max_age)
    g = rng.multivariate_normal(mu, cov, size=4 * n)
    g = g[(g[:, 0] > a_lo) & (g[:, 0] < a_hi)]
    inside = len(g) / (4 * n)
    g = g[:n]
    age_p, feh_p = population.ages_and_feh(rng, n)
    from_prior = rng.random(len(g)) < prior_share
    age2 = np.where(from_prior, age_p[:len(g)], g[:, 0])
    feh2 = np.where(from_prior, feh_p[:len(g)], g[:, 1])
    d = np.column_stack([age2, feh2]) - mu
    inv = np.linalg.inv(cov)
    gauss = np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, inv, d)) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    q = prior_share * age_feh_prior(age2, feh2, population) + (1 - prior_share) * gauss / max(inside, 1e-6)
    return age2, feh2, np.log(np.maximum(q, 1e-300))


# --- the system's components and orbits ---------------------------------------------------

def _deproject(s_au, mtot, rng, population: pop.PopulationModel):
    """Orbits whose projection at a random epoch and orientation is ``s_au``: (a, e, incl, omega)."""
    n = len(s_au)
    a, e = np.empty(n), np.empty(n)
    inc = np.arccos(2 * rng.random(n) - 1)
    om = 2 * np.pi * rng.random(n)
    # Eccentricity from the population model at the projected separation's period.
    p_guess = YEAR_DAYS * np.sqrt(s_au**3 / mtot)
    e = population.draw_eccentricity(rng, p_guess)
    mean = 2 * np.pi * rng.random(n)
    big_e = mean.copy()
    for _ in range(30):
        big_e = big_e - (big_e - e * np.sin(big_e) - mean) / (1 - e * np.cos(big_e))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(big_e / 2), np.sqrt(1 - e) * np.cos(big_e / 2))
    r_over_a = 1 - e * np.cos(big_e)
    proj = r_over_a * np.sqrt(1 - np.sin(om + nu) ** 2 * np.sin(inc) ** 2)
    a = s_au / np.maximum(proj, 1e-3)
    return a, e, inc, om


def _build(record: SystemRecord, age, feh, masses, rng, population: pop.PopulationModel | None = None):
    population = population or pop.default_model()
    n, n_comp = masses.shape
    comps = record.components
    flags = 0
    st = evolve(masses.ravel(), np.repeat(age, n_comp), np.repeat(feh, n_comp), rng=rng)
    mass_now = st["mass"].reshape(n, n_comp)
    a = np.full((n, n_comp), np.nan)
    ecc = np.full((n, n_comp), np.nan)
    inc = np.full((n, n_comp), np.nan)
    om = np.full((n, n_comp), np.nan)
    node = np.full((n, n_comp), np.nan)
    last_a = np.zeros(n)
    for k in range(1, n_comp):
        m_in = mass_now[:, :k].sum(axis=1)
        mtot = m_in + mass_now[:, k]
        sep = record.separations.get(k, (np.nan, ""))[0]
        dist = comps[k].distance_pc if np.isfinite(comps[k].distance_pc) else comps[0].distance_pc
        if np.isfinite(sep) and np.isfinite(dist):
            ak, ek, ik, ok = _deproject(np.full(n, sep * dist), mtot, rng, population)
        else:  # unresolved: population separations below the resolution limit
            amax = RESOLUTION_ARCSEC * dist if np.isfinite(dist) else 4e4
            ak = np.full(n, np.nan)
            todo = np.arange(n)
            for _ in range(200):
                logp = population.draw_log_period(rng, masses[todo, 0])
                trial = (mtot[todo] * (10**logp / YEAR_DAYS) ** 2) ** (1 / 3)
                good = (trial < amax) & (trial >= np.maximum(3 * last_a[todo], 0.02))
                ak[todo[good]] = trial[good]
                todo = todo[~good]
                if not len(todo):
                    break
            ak[todo] = np.maximum(np.minimum(amax, 1.0), 3.01 * last_a[todo])
            ek = population.draw_eccentricity(rng, YEAR_DAYS * np.sqrt(ak**3 / mtot))
            ik = np.arccos(2 * rng.random(n) - 1)
            ok = 2 * np.pi * rng.random(n)
        bad = ak < 3 * last_a
        if bad.any():
            flags |= FLAG_HIERARCHY_APPROX
            ak = np.where(bad, 3.01 * last_a, ak)
        a[:, k], ecc[:, k], inc[:, k], om[:, k] = ak, ek, ik, ok
        node[:, k] = 2 * np.pi * rng.random(n)
        last_a = ak
    m_tot_inner = np.cumsum(mass_now, axis=1)
    period = np.full((n, n_comp), np.nan)
    period[:, 1:] = YEAR_DAYS * np.sqrt(a[:, 1:] ** 3 / m_tot_inner[:, 1:])
    components = {
        "system": np.repeat(np.arange(n), n_comp), "role": np.tile(np.arange(n_comp), n),
        "period_days": period.ravel(), "a_au": a.ravel(), "ecc": ecc.ravel(),
    }
    components.update(st)
    components["incl_sky"], components["node"], components["omega"] = inc.ravel(), node.ravel(), om.ravel()
    components["name"] = np.tile(np.array([c.name for c in comps], dtype=object), n)
    systems = {"age_gyr": age, "feh": feh, "n_components": np.full(n, n_comp),
               "primary_initial_mass": masses[:, 0]}
    return systems, components, flags


def realise(name: str, n: int = 1000, seed=None, use_gaia: bool = True,
            model: StarFitModel = StarFitModel()) -> StarSystemFit:
    """Look up ``name`` and return ``n`` posterior realisations of its stellar system."""
    return fit(find(name, use_gaia=use_gaia), n=n, seed=seed, model=model)


__all__ = ["StarFitModel", "StarSystemFit", "fit", "realise", "spectral_type_of"]
