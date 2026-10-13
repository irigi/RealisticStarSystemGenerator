"""Python/numba port of the He & Ford (2026) hybrid model with clustered masses (HM-C).

Valid domain (as fitted to Kepler): FGK dwarfs with 0.5 <= bp-rp-E* <= 1.7,
periods 3-300 d, radii 0.5-10 R_earth, initial masses 0.1-1000 M_earth.

The port follows ``generate_planetary_system_clustered_periods_and_sizes_photoevap_distribute_amd``
in ``src/models_hybrid.jl`` of the ``He_Ford_2026`` branch of
https://github.com/ExoJulia/SysSimExClusters, step by step:

1. star hosts planets with probability f_swpa(colour) (linear, clipped to [0, 1]);
2. N_clusters ~ ZTP(lambda_c), N_planets per cluster ~ ZTP(lambda_p);
3. per cluster: lognormal mass scale, lognormal initial masses around it, initial radii
   from a smoothed broken power-law R(M) with fractional scatter (redrawn until
   0.5 < R < 10, max 10 tries, otherwise the planet is dropped);
4. unscaled periods ~ truncated lognormal(0, n*sigma_P), redrawn until the cluster is
   mutual-Hill stable and passes the circular MMR-overlap criterion (max 1000 tries);
5. period scale ~ power law in [P_min/min P', P_max/max P'], redrawn until the whole
   system is mutual-Hill stable (max 100 tries, otherwise the cluster is dropped);
6. photoevaporation (Lopez+2012 timescale, Neil & Rogers 2020 retention probability);
   stripped planets get pure-silicate radii (Seager+2007) with 5% scatter. The
   timescale scales with the planet's bolometric flux from the star's own R and Teff
   (no separate XUV/bolometric ratio); retention probability is
   min(alpha * t_loss / age, 1) with the fixed age ``system_age`` (5 Gyr), unless a
   per-star age is passed;
7. eccentricities and mutual inclinations by distributing the critical AMD per unit
   mass (singles: Rayleigh(sigma_hk) eccentricity, zero inclination);
8. isotropic orientation of the system reference plane.

Faithfulness note: in the authors' ``amd_stability.jl`` the function
``relative_AMD_MMR_overlap`` is defined twice with the same signature, so the
second (Hill-stability, Petit+2018 eq. 26) definition silently replaces the
MMR-overlap one. The published catalogs therefore use min(C_coll, C_Hill).
``amd_pair_criterion="hill"`` reproduces that (default); ``"mmr"`` uses the
MMR-overlap criterion that the paper describes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from numba import njit

from ..units import EARTH_MASS_MSUN, EARTH_RADIUS_RSUN

# --- SysSim constants (ExoplanetsSysSim.jl src/constants.jl) -------------------
_AU_M = 149597870700.0
_G_MKS = 6.67384e-11
_SUN_RADIUS_M = 6.9566e8
_EARTH_RADIUS_M = 6.3781e6
_SUN_MASS_KG = 1.988547e30
_SEC_IN_DAY = 86400.0
_DAY_IN_YEAR = 365.25
_GRAV_CONST = _G_MKS * _SEC_IN_DAY**2 * _SUN_MASS_KG / _AU_M**3  # AU^3 / (day^2 Msun)
_SIGMA_SB_CGS = 5.670374419e-5

MAX_PLANETS_OUT = 64  # hard cap on planets kept per system (never reached in practice)


@dataclass(frozen=True)
class HMCParams:
    """Model parameters, named after the catalog header keys."""

    f_swpa_med: float = 0.88
    f_swpa_slope: float = 0.9
    med_color: float = 0.81
    log_rate_clusters: float = -0.41
    max_clusters_in_sys: int = 20
    log_rate_planets_per_cluster: float = 0.61
    max_planets_in_clusters: int = 20
    min_period: float = 3.0
    max_period: float = 300.0
    power_law_P: float = -0.16
    sigma_logperiod_per_pl_in_cluster: float = 0.25
    min_mass: float = 0.1
    max_mass: float = 1000.0
    min_radius: float = 0.5
    max_radius: float = 10.0
    mean_ln_mass: float = 0.32
    sigma_ln_mass: float = 1.77
    sigma_ln_mass_in_cluster: float = 0.46
    norm_radius: float = 2.07
    break_mass: float = 20.0
    gamma0: float = 0.089
    gamma1: float = 0.5
    sigma0: float = 0.164
    sigma1: float = 0.3
    system_age: float = 5.0
    log_alpha_pret: float = 1.69
    sigma_hk: float = 0.25
    num_mutual_hill_radii: float = 10.0
    f_amd_crit: float = 1.0
    # Not in the published model (always 0 there): probability that a planet has no
    # H/He envelope regardless of irradiation (used by the M-dwarf extension).
    f_bare: float = 0.0

    @classmethod
    def from_header(cls, h: dict) -> "HMCParams":
        return cls(
            f_swpa_med=h["f_stars_with_planets_attempted_at_med_color"],
            f_swpa_slope=h["f_stars_with_planets_attempted_color_slope"],
            med_color=h["med_color"],
            log_rate_clusters=h["log_rate_clusters"],
            max_clusters_in_sys=int(h["max_clusters_in_sys"]),
            log_rate_planets_per_cluster=h["log_rate_planets_per_cluster"],
            max_planets_in_clusters=int(h["max_planets_in_clusters"]),
            min_period=h["min_period"],
            max_period=h["max_period"],
            power_law_P=h["power_law_P"],
            sigma_logperiod_per_pl_in_cluster=h["sigma_logperiod_per_pl_in_cluster"],
            min_mass=h["min_mass (M_earth)"],
            max_mass=h["max_mass (M_earth)"],
            min_radius=h["min_radius (R_earth)"],
            max_radius=h["max_radius (R_earth)"],
            mean_ln_mass=h["mean_ln_mass (ln M_earth)"],
            sigma_ln_mass=h["sigma_ln_mass (ln M_earth)"],
            sigma_ln_mass_in_cluster=h["sigma_ln_mass_in_cluster (ln M_earth)"],
            norm_radius=h["norm_radius (R_earth)"],
            break_mass=h["break_mass (M_earth)"],
            gamma0=h["power_law_γ0"],
            gamma1=h["power_law_γ1"],
            sigma0=h["power_law_σ0"],
            sigma1=h["power_law_σ1"],
            system_age=h["system_age (Gyr)"],
            log_alpha_pret=h["log_α_pret"],
            sigma_hk=h["sigma_hk"],
            num_mutual_hill_radii=h["num_mutual_hill_radii"],
            f_amd_crit=h["f_amd_crit"],
        )

    def as_array(self) -> np.ndarray:
        return np.array([float(getattr(self, f.name)) for f in fields(self)])


# Positions of the parameters in HMCParams.as_array() (same order as the fields).
(I_FMED, I_FSLOPE, I_MEDCOL, I_LRC, I_MAXC, I_LRP, I_MAXP, I_PMIN, I_PMAX, I_ALPHAP, I_SIGP,
 I_MMIN, I_MMAX, I_RMIN, I_RMAX, I_MUM, I_SIGM, I_SIGMC, I_RNORM, I_MBREAK, I_G0, I_G1, I_S0,
 I_S1, I_AGE, I_LALPHA, I_SIGHK, I_NHILL, I_FAMD, I_FBARE) = range(len(fields(HMCParams)))


# --- small numerical helpers ----------------------------------------------------

@njit(cache=True)
def semimajor_axis(period_days, mass_msun):
    return (_GRAV_CONST / (4.0 * math.pi**2) * mass_msun * period_days**2) ** (1.0 / 3.0)


@njit(cache=True)
def draw_ztp(lam, kmin, kmax):
    """Poisson truncated to [kmin, kmax] by inverse CDF (as draw_truncated_poisson)."""
    total = 0.0
    log_lam = math.log(lam)
    for k in range(kmin, kmax + 1):
        total += math.exp(k * log_lam - lam - math.lgamma(k + 1.0))
    u = np.random.random() * total
    cum = 0.0
    for k in range(kmin, kmax + 1):
        cum += math.exp(k * log_lam - lam - math.lgamma(k + 1.0))
        if cum >= u:
            return k
    return kmax


@njit(cache=True)
def draw_trunc_lognormal(mu, sigma, lo, hi):
    """LogNormal(mu, sigma) truncated to [lo, hi], by rejection with an inverse-CDF fallback."""
    for _ in range(10000):
        x = math.exp(mu + sigma * np.random.normal())
        if lo <= x <= hi:
            return x
    # Truncation region far in a tail: sample uniformly in ln x, weighted (rare path).
    while True:
        lx = math.log(lo) + np.random.random() * (math.log(hi) - math.log(lo))
        z = (lx - mu) / sigma
        zl = (math.log(lo) - mu) / sigma
        zh = (math.log(hi) - mu) / sigma
        zmax = 0.0 if zl <= 0.0 <= zh else min(zl * zl, zh * zh)
        if np.random.random() < math.exp(-0.5 * (z * z - zmax)):
            return math.exp(lx)


@njit(cache=True)
def draw_power_law(n, x0, x1):
    if n != -1.0:
        return ((x1 ** (n + 1.0) - x0 ** (n + 1.0)) * np.random.random() + x0 ** (n + 1.0)) ** (1.0 / (n + 1.0))
    return x0 * (x1 / x0) ** np.random.random()


@njit(cache=True)
def mean_radius_and_scatter(m, c, m_break, g0, g1, s0, s1):
    mu0 = c * m**g0
    mu1 = c * m_break ** (g0 - g1) * m**g1
    s = 1.0 / (1.0 + math.exp(-5.0 * (math.log(m) - math.log(m_break))))
    return (1.0 - s) * mu0 + s * mu1, (1.0 - s) * s0 + s * s1


@njit(cache=True)
def radius_pure_silicate_seager2007(m):
    return 3.90 * 10.0 ** (-0.209594 + (1.0 / 3.0) * math.log10(m / 10.55) - 0.0799 * (m / 10.55) ** 0.413)


@njit(cache=True)
def envelope_mass(m, m_transition=20.0):
    s = 1.0 / (1.0 + math.exp(-5.0 * (math.log(m) - math.log(m_transition))))
    return (1.0 - s) * 0.1 * m + s * (m - math.sqrt(m))


@njit(cache=True)
def luminosity_cgs(r_star_rsun, teff):
    r_cm = r_star_rsun * _SUN_RADIUS_M * 100.0
    return 4.0 * math.pi * r_cm**2 * _SIGMA_SB_CGS * teff**4


@njit(cache=True)
def bolometric_flux(period_days, mstar, lum_cgs):
    a_cm = semimajor_axis(period_days, mstar) * _AU_M * 100.0
    return lum_cgs / (4.0 * math.pi * a_cm**2)


@njit(cache=True)
def mass_loss_timescale_gyr(m_env, r_prim, f_p, f_oplus):
    """Lopez+2012 / NR20 envelope mass-loss timescale, exactly as in models_hybrid.jl."""
    g_cgs = _G_MKS * 1000.0
    m_env_g = m_env * (3.986004e14 / 1.3271244e20) * _SUN_MASS_KG * 1000.0
    r_cm = r_prim * _EARTH_RADIUS_M * 100.0
    t_s = (g_cgs * m_env_g**2) / (math.pi * 0.1 * r_cm**3 * 504.0) * (f_oplus / f_p)
    return t_s / (_SEC_IN_DAY * _DAY_IN_YEAR * 1e9)


# --- stability ------------------------------------------------------------------

@njit(cache=True)
def hill_stable(P, M, n, mstar, n_hill):
    """Mutual-Hill spacing test on circular orbits; NaN entries are ignored (as in Julia)."""
    order = np.argsort(P[:n])
    for k in range(n - 1):
        i, j = order[k], order[k + 1]
        if math.isnan(P[i]) or math.isnan(P[j]):
            continue
        a1 = semimajor_axis(P[i], M[i] + mstar)
        a2 = semimajor_axis(P[j], M[j] + mstar)
        mu = (M[i] + M[j]) / mstar
        rh = 0.5 * (a1 + a2) * (mu / 3.0) ** (1.0 / 3.0)
        if a2 - a1 < n_hill * rh:
            return False
    return True


@njit(cache=True)
def circular_mmr_overlap_stable(P, M, n, mstar):
    order = np.argsort(P[:n])
    for k in range(n - 1):
        i, j = order[k], order[k + 1]
        a1 = semimajor_axis(P[i], M[i] + mstar)
        a2 = semimajor_axis(P[j], M[j] + mstar)
        eps = (M[i] + M[j]) / mstar
        if not (a1 / a2 < 1.0 - 1.46 * eps ** (2.0 / 7.0)):
            return False
    return True


@njit(cache=True)
def critical_eccentricity(gamma, alpha, tol=1e-15):
    """Root of F(e) from Laskar & Petit (2017) eq. 35 (Newton + bisection)."""
    def F(e):
        return alpha * e + gamma * e / math.sqrt(alpha * (1 - e) * (1 + e) + gamma**2 * e**2) - 1.0 + alpha

    def dF(e):
        return alpha + alpha * gamma / (alpha * (1 - e) * (1 + e) + gamma**2 * e**2) ** 1.5

    e0 = 0.5
    F0 = F(e0)
    if F0 >= 0:
        eR, FR = e0, F0
        eL = ex = 0.0
        FL = F(eL)
    else:
        eL, FL = e0, F0
        eR = ex = 1.0
        FR = F(eR)
    it = 0
    while abs(ex - e0) > tol and it < 200:
        it += 1
        ex = e0 - F0 / dF(e0)
        if ex <= eL or ex >= eR:
            m = (FR - FL) / (eR - eL)
            b = F0 - m * e0
            ex = -b / m
        e0, ex = ex, e0
        F0 = F(e0)
        if F0 >= 0:
            eR, FR = e0, F0
        else:
            eL, FL = e0, F0
    return ex


@njit(cache=True)
def relative_amd_collision(mu1, mu2, a1, a2):
    gamma = mu1 / mu2
    alpha = a1 / a2
    e1 = critical_eccentricity(gamma, alpha)
    e2 = 1.0 - alpha - alpha * e1
    return gamma * math.sqrt(alpha) * (1.0 - math.sqrt((1 - e1) * (1 + e1))) + (1.0 - math.sqrt((1 - e2) * (1 + e2)))


@njit(cache=True)
def relative_amd_mmr_overlap(mu1, mu2, a1, a2):
    """Petit, Laskar & Boue (2017) eq. 74."""
    gamma = mu1 / mu2
    alpha = a1 / a2
    eps = mu1 + mu2
    r = 0.80199
    g = (3**4 * (1 - alpha) ** 5) / (2**9 * r * eps) - (32 * r * eps) / (9 * (1 - alpha) ** 2)
    return (g**2 * gamma * math.sqrt(alpha)) / (2 + 2 * gamma * math.sqrt(alpha))


@njit(cache=True)
def relative_amd_hill(mu1, mu2, a1, a2):
    """Petit, Laskar & Boue (2018) eq. 26 (what the published catalogs actually used)."""
    gamma = mu1 / mu2
    alpha = a1 / a2
    eps = mu1 + mu2
    c = gamma * math.sqrt(alpha) + 1 - (gamma + 1) ** 1.5 * math.sqrt(
        alpha / (gamma + alpha) * (1 + (9 * eps) ** (2.0 / 3.0) * gamma / (1 + gamma) ** 2))
    return max(c, 0.0)


@njit(cache=True)
def critical_amd_system(mu, a, n, use_hill):
    best = mu[0] * math.sqrt(a[0])
    for k in range(1, n):
        lam = mu[k] * math.sqrt(a[k])
        c_coll = relative_amd_collision(mu[k - 1], mu[k], a[k - 1], a[k])
        if use_hill:
            c_other = relative_amd_hill(mu[k - 1], mu[k], a[k - 1], a[k])
        else:
            c_other = relative_amd_mmr_overlap(mu[k - 1], mu[k], a[k - 1], a[k])
        best = min(best, lam * min(c_coll, c_other))
    return best


# --- geometry -------------------------------------------------------------------

@njit(cache=True)
def _random_unit_vector():
    z = 2.0 * np.random.random() - 1.0
    phi = 2.0 * math.pi * np.random.random()
    s = math.sqrt(1.0 - z * z)
    return np.array([s * math.cos(phi), s * math.sin(phi), z])


@njit(cache=True)
def _orbit_normal(i_m, node, n_ref):
    """Unit normal of an orbit inclined by i_m with ascending node `node` w.r.t. plane n_ref."""
    # Orthonormal basis (u, v) in the reference plane.
    if abs(n_ref[2]) < 0.9:
        t = np.array([0.0, 0.0, 1.0])
    else:
        t = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, n_ref)
    u /= np.linalg.norm(u)
    v = np.cross(n_ref, u)
    node_dir = math.cos(node) * u + math.sin(node) * v
    perp = np.cross(n_ref, node_dir)
    return math.cos(i_m) * n_ref - math.sin(i_m) * perp


@njit(cache=True)
def _angle(a, b):
    c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.acos(min(1.0, max(-1.0, c)))


@njit(cache=True)
def _node_in_plane(n_orb, n_plane):
    """Longitude of the ascending node of orbit n_orb measured in the plane n_plane."""
    if abs(n_plane[2]) < 0.9:
        t = np.array([0.0, 0.0, 1.0])
    else:
        t = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, n_plane)
    u /= np.linalg.norm(u)
    v = np.cross(n_plane, u)
    line = np.cross(n_plane, n_orb)
    nrm = np.linalg.norm(line)
    if nrm < 1e-15:
        return 0.0
    line /= nrm
    return math.atan2(line[0] * v[0] + line[1] * v[1] + line[2] * v[2],
                      line[0] * u[0] + line[1] * u[1] + line[2] * u[2]) % (2 * math.pi)


# --- the generator --------------------------------------------------------------

# Output columns (per planet).
COLUMNS = ("period", "planet_mass", "planet_radius", "clusterid", "ecc", "omega", "incl_mut",
           "incl_invariable", "asc_node_invariable", "incl_sky", "mean_anom", "mass_init",
           "radius_init", "lost_envelope")
N_COL = len(COLUMNS)


@njit(cache=True)
def generate_system(p, mstar, rstar, teff, color, age, use_hill, out, plane):
    """Draw one planetary system. Fills ``out[:n, :]`` (columns: COLUMNS) and returns n.

    ``plane`` (length 3) receives the unit normal of the system's invariable plane in
    the frame where the sky normal is +z; it is left untouched when n = 0.

    ``age`` (Gyr) is the age used in the photoevaporation retention probability; the
    published model uses the fixed ``system_age`` parameter for every star.
    """
    # 1. Does the star host planets?
    f_swpa = min(1.0, max(0.0, p[I_FSLOPE] * (color - p[I_MEDCOL]) + p[I_FMED]))
    if np.random.random() > f_swpa:
        return 0
    nc = draw_ztp(math.exp(p[I_LRC]), 1, int(p[I_MAXC]))
    n_in = np.empty(nc, dtype=np.int64)
    for c in range(nc):
        n_in[c] = draw_ztp(math.exp(p[I_LRP]), 1, int(p[I_MAXP]))
    ntot = int(n_in.sum())

    P = np.full(ntot, np.nan)
    M = np.full(ntot, np.nan)  # initial masses, solar units
    R = np.full(ntot, np.nan)  # initial radii, Earth radii
    cid = np.zeros(ntot, dtype=np.int64)
    max_ratio = p[I_PMAX] / p[I_PMIN]
    start = 0
    for c in range(nc):
        n = n_in[c]
        stop = start + n
        cid[start:stop] = c + 1
        # 2a. Clustered initial masses and radii.
        mu_c = math.log(draw_trunc_lognormal(p[I_MUM], p[I_SIGM], p[I_MMIN], p[I_MMAX]))
        Mc = np.full(n, np.nan)
        Rc = np.full(n, np.nan)
        for i in range(n):
            for _ in range(10):
                m = draw_trunc_lognormal(mu_c, p[I_SIGMC], p[I_MMIN], p[I_MMAX])
                mu_r, s_r = mean_radius_and_scatter(m, p[I_RNORM], p[I_MBREAK], p[I_G0], p[I_G1], p[I_S0], p[I_S1])
                r = mu_r + s_r * mu_r * np.random.normal()
                if p[I_RMIN] < r < p[I_RMAX]:
                    Mc[i] = m * EARTH_MASS_MSUN
                    Rc[i] = r
                    break
        valid = ~np.isnan(Rc)
        nv = int(valid.sum())
        Pc = np.full(n, np.nan)
        # 2b. Unscaled periods for a stable cluster.
        if nv == 1:
            Pc[valid] = 1.0
        elif nv > 1:
            Mv = Mc[valid]
            lo, hi = 1.0 / math.sqrt(max_ratio), math.sqrt(max_ratio)
            good = False
            for _ in range(1000):
                Pv = np.empty(nv)
                for k in range(nv):
                    Pv[k] = draw_trunc_lognormal(0.0, p[I_SIGP] * n, lo, hi)
                if hill_stable(Pv, Mv, nv, mstar, p[I_NHILL]) and circular_mmr_overlap_stable(Pv, Mv, nv, mstar):
                    good = True
                    break
            if good:
                Pc[valid] = Pv
        M[start:stop] = Mc
        R[start:stop] = Rc
        # 2c. Period scale such that the whole system remains stable.
        ok = ~np.isnan(Pc)
        if ok.sum() >= 1:
            pmin_scale = p[I_PMIN] / np.nanmin(Pc)
            pmax_scale = p[I_PMAX] / np.nanmax(Pc)
            placed = False
            for _ in range(100):
                scale = draw_power_law(p[I_ALPHAP], pmin_scale, pmax_scale)
                P[start:stop] = Pc * scale
                if hill_stable(P[:stop], M[:stop], stop, mstar, p[I_NHILL]):
                    placed = True
                    break
            if not placed:
                P[start:stop] = np.nan
        start = stop

    keep = np.where(~np.isnan(P))[0]
    nk = len(keep)
    if nk == 0:
        return 0
    order = keep[np.argsort(P[keep])]
    nk = min(nk, out.shape[0])
    order = order[:nk]
    P, M, R, cid = P[order], M[order], R[order], cid[order]

    # 3. Photoevaporation.
    m_init_e = M / EARTH_MASS_MSUN
    lum = luminosity_cgs(rstar, teff)
    f_oplus = bolometric_flux(365.25, 1.0, luminosity_cgs(1.0, 5780.0))
    alpha_ret = math.exp(p[I_LALPHA])
    m_final = m_init_e.copy()
    r_final = R.copy()
    lost = np.zeros(nk)
    for k in range(nk):
        m_env = envelope_mass(m_init_e[k])
        t_loss = mass_loss_timescale_gyr(m_env, R[k], bolometric_flux(P[k], mstar, lum), f_oplus)
        p_ret = min(alpha_ret * t_loss / age, 1.0) * (1.0 - p[I_FBARE])
        if not (np.random.random() < p_ret):
            lost[k] = 1.0
            m_final[k] = m_init_e[k] - m_env
            r_si = radius_pure_silicate_seager2007(m_final[k])
            r_final[k] = r_si + 0.05 * r_si * np.random.normal()

    # 4. Eccentricities and mutual inclinations (initial masses, as in the original).
    mu = M / mstar
    a = np.empty(nk)
    for k in range(nk):
        a[k] = semimajor_axis(P[k], mstar)
    ecc = np.zeros(nk)
    omega = np.zeros(nk)
    i_m = np.zeros(nk)
    if nk == 1:
        while True:
            h = p[I_SIGHK] * np.random.normal()
            kk = p[I_SIGHK] * np.random.normal()
            if h * h + kk * kk < 1.0:
                break
        ecc[0] = math.sqrt(h * h + kk * kk)
        omega[0] = math.atan2(kk, h)
    else:
        f_amd = p[I_FAMD] + (1.0 - p[I_FAMD]) * np.random.random()
        amd_tot = f_amd * critical_amd_system(mu, a, nk, use_hill)
        musum = mu.sum()
        for k in range(nk):
            lam = mu[k] * math.sqrt(a[k])
            x = (amd_tot * mu[k] / musum) / lam
            sumsq = x * (2.0 - x)
            u1, u2 = np.random.random(), np.random.random()
            lo_, hi_ = min(u1, u2), max(u1, u2)
            xsq, ysq, zsq = lo_ * sumsq, (hi_ - lo_) * sumsq, (1.0 - hi_) * sumsq
            xs = math.sqrt(xsq) * (1.0 if np.random.normal() >= 0 else -1.0)
            ys = math.sqrt(ysq) * (1.0 if np.random.normal() >= 0 else -1.0)
            ecc[k] = math.sqrt(xsq + ysq)
            omega[k] = math.atan2(xs, ys)
            i_m[k] = math.asin(math.sqrt(zsq))

    # 5-6. Orientation: reference plane isotropic; nodes and mean anomalies uniform.
    n_ref = _random_unit_vector()
    z_sky = np.array([0.0, 0.0, 1.0])
    L = np.zeros(3)
    normals = np.empty((nk, 3))
    for k in range(nk):
        normals[k] = _orbit_normal(i_m[k], 2.0 * math.pi * np.random.random(), n_ref)
        b = a[k] * math.sqrt((1.0 - ecc[k]) * (1.0 + ecc[k]))
        L += M[k] * b * math.sqrt(mstar / a[k]) * normals[k]
    n_inv = L / np.linalg.norm(L) if nk > 1 else n_ref
    plane[:] = n_inv
    for k in range(nk):
        out[k, 0] = P[k]
        out[k, 1] = m_final[k]
        out[k, 2] = r_final[k]
        out[k, 3] = cid[k]
        out[k, 4] = ecc[k]
        out[k, 5] = omega[k]
        out[k, 6] = i_m[k]
        out[k, 7] = _angle(n_inv, normals[k])
        out[k, 8] = _node_in_plane(normals[k], n_inv)
        out[k, 9] = _angle(z_sky, normals[k])
        out[k, 10] = 2.0 * math.pi * np.random.random()
        out[k, 11] = m_init_e[k]
        out[k, 12] = R[k]
        out[k, 13] = lost[k]
    return nk


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _generate_many(params, mstar, rstar, teff, color, age, use_hill):
    n_star = len(mstar)
    counts = np.zeros(n_star, dtype=np.int64)
    cap = max(16, 4 * n_star)
    buf = np.empty((cap, N_COL))
    owner = np.empty(cap, dtype=np.int64)
    work = np.empty((MAX_PLANETS_OUT, N_COL))
    planes = np.full((n_star, 3), np.nan)
    used = 0
    for s in range(n_star):
        nk = generate_system(params[s], mstar[s], rstar[s], teff[s], color[s], age[s], use_hill, work, planes[s])
        counts[s] = nk
        if used + nk > cap:
            new_cap = 2 * cap + nk
            nb = np.empty((new_cap, N_COL))
            nb[:used] = buf[:used]
            no = np.empty(new_cap, dtype=np.int64)
            no[:used] = owner[:used]
            buf, owner, cap = nb, no, new_cap
        buf[used:used + nk] = work[:nk]
        owner[used:used + nk] = s
        used += nk
    return counts, buf[:used], owner[:used], planes


def generate_catalog(params, mstar, rstar, teff, color, *, age=None, seed: int | None = None,
                     amd_pair_criterion: str = "hill", return_planes: bool = False):
    """Draw planetary systems for an array of stars.

    Returns ``(counts, planets)``: planets per star, and a dict of per-planet arrays
    (keys: ``star_index`` plus :data:`COLUMNS`), sorted by star then period.
    Masses are in Earth masses, radii in Earth radii, angles in radians.

    ``params`` is one :class:`HMCParams` for all stars, or an array of shape
    (n_star, n_params) with one row per star (columns as :meth:`HMCParams.as_array`).
    ``age`` (Gyr, per star) replaces the fixed ``params.system_age`` in the
    photoevaporation step. ``None`` (default) keeps the published behaviour.
    ``return_planes=True`` also returns an (n_star, 3) array of invariable-plane
    normals (sky normal = +z; NaN for stars without planets).
    """
    if amd_pair_criterion not in ("hill", "mmr"):
        raise ValueError("amd_pair_criterion must be 'hill' or 'mmr'")
    if seed is not None:
        _seed(seed)
    n = len(mstar)
    if isinstance(params, HMCParams):
        rows = np.broadcast_to(params.as_array(), (n, len(fields(HMCParams))))
    else:
        rows = np.asarray(params, dtype=np.float64)
        if rows.shape != (n, len(fields(HMCParams))):
            raise ValueError("per-star params must have shape (n_star, n_params)")
    if age is None:
        age = rows[:, I_AGE]
    arrs = [np.ascontiguousarray(x, dtype=np.float64) for x in (rows, mstar, rstar, teff, color, age)]
    counts, buf, owner, planes = _generate_many(*arrs, amd_pair_criterion == "hill")
    planets = {"star_index": owner}
    for j, name in enumerate(COLUMNS):
        planets[name] = buf[:, j]
    planets["clusterid"] = planets["clusterid"].astype(np.int32)
    planets["lost_envelope"] = planets["lost_envelope"].astype(bool)
    if return_planes:
        return counts, planets, planes
    return counts, planets
