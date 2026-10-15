"""Planets of evolved stars and stellar remnants.

Used by :func:`starsys.planets.attach_planets` through :mod:`starsys.hosts`.

**Hosts.** Subgiants, giants (MIST phases 2-5), post-AGB stars and white dwarfs keep the
planets they had on the main sequence: the planet models are run on the **progenitor's
main-sequence state** (MIST at the middle of its main-sequence life, log age = log lifetime
- 0.3, for its initial mass and [Fe/H]), and the result is then evolved. Progenitors bluer
than HM-C's colour range are flagged ``colour`` as usual; progenitors above the giant
calibration's 3 Msun are flagged ``giant_host``.

**Maximum past radius** (from the MIST isochrones, :func:`radius_tips`): the largest
radius on the red-giant branch (R_RGB) and on the AGB (R_AGB) of a star of the given initial
mass, read off the last isochrone on which it is still alive (stars a few per cent heavier
are then on the giant branches, so the short tip phases are resolved although the age grid
is 0.05 dex). A star now on the RGB or subgiant branch (phase 2) has its present radius as
its maximum; core-He-burning stars (3) have passed the RGB tip; AGB stars (4, 5) the RGB tip
and their present radius; post-AGB stars and white dwarfs the full AGB.

**Engulfment** (Mustill & Villaver 2012, ApJ 761, 121; MV12): tides pull planets into the
envelope from beyond the maximum stellar radius, more so for heavier planets, while mass
loss pushes them out, more so for heavier stars. MV12 find (circular orbits, through the
thermally pulsing AGB) that Jovian planets are engulfed from initial orbits up to ~3 au
around a 1 Msun star and ~5 au around 5 Msun, Terrestrial planets from 1.5-3 au; around
5 Msun Terrestrials beyond 2.8 au survive although the star exceeds 5 au. Expressed in
units of the maximum stellar radius (1.5 au and ~5.3 au in their models), the engulfment
radius is f_T = 1.0 -> 0.53 and f_J = 2.0 -> 0.95 times R_max between 1 and 5 Msun; it is
interpolated in log M* (clamped outside) and in planet mass between Earth (f_T) and
Jupiter (f_J) as w = (ln(M_p/M_E) / ln 318)^2 (MV12's Neptunes are only slightly above the
Terrestrials). These ratios are applied to the MIST R_max. White dwarfs also pass their RGB
tip: a_engulf = max(f R_max, R_RGB).

**Tidal decay** (:func:`tidal_decay`; Villaver & Livio 2009, the prescription MV12 use): the
equilibrium tide in the star's convective envelope is integrated along each host's past
post-main-sequence evolution, read off a MIST isochrone, together with the orbital expansion
from mass loss. Stars still on the giant branches engulf what is inside their present (or,
after the RGB tip, maximum) radius plus what this decay has dragged in; for white dwarfs and
post-AGB stars a planet is lost if either this integration (to the end of the AGB) or the MV12
ratios say so (they agree to ~20%: integrated, Jovian planets around a 1 Msun star are lost
from 2.4 au and Terrestrial ones from 1.2 au). The decay is weak for hot Jupiters around
subgiants (the convective turnover time exceeds half the orbital period, which suppresses
the dissipation) and negligible beyond ~5 stellar radii.

**Orbital expansion:** survivors' semi-major axes grow by M_initial / M_now (adiabatic mass
loss), times the tidal shrinkage from the integration; periods follow. Planet radii and
atmospheres are recomputed in :mod:`starsys.derived` only for giant planets (re-inflation).

**Neutron stars:** pulsar planets are rare. Nitu et al. (2022, MNRAS; 800 pulsars timed at
Jodrell Bank) find < 0.5% of pulsars host planets above a few Earth masses with periods of
20 d - 17 yr; PSR B1257+12 is the only secure system. A flagged rate of 0.2% is used, with
1-3 planets of 0.01-10 Earth masses at 0.1-1 au (log-uniform, B1257+12-like), nearly circular
and coplanar. Black holes get none.

**Post-interaction binaries:** a component whose maximum past radius exceeds 0.3 of the
separation of its nearest companion would have filled its Roche lobe; mass transfer and
common envelopes are not modelled, so all planets of such a system are flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

from .stars import mist
from .units import EARTH_MASS_MSUN

RSUN_AU = 1.0 / 215.032
MSUN_G, RSUN_CM, LSUN_CGS, YEAR_S = 1.98841e33, 6.957e10, 3.828e33, 3.15576e7
JUPITER_MASS_EARTH = 317.83

# MIST phases.
PHASE_RGB, PHASE_CHEB, PHASE_EAGB, PHASE_TPAGB, PHASE_POSTAGB = 2, 3, 4, 5, 6
EVOLVED_PHASES = (PHASE_RGB, PHASE_CHEB, PHASE_EAGB, PHASE_TPAGB, PHASE_POSTAGB)


@dataclass(frozen=True)
class EvolvedModel:
    progenitor_offset_dex: float = 0.3  # progenitor state at log lifetime - offset
    max_initial_mass: float = 8.0  # WD progenitors; heavier stars end as NS/BH
    # MV12 engulfment radius / R_max at (1 Msun, 5 Msun): Terrestrial and Jovian planets.
    f_terrestrial: tuple = (1.0, 0.53)
    f_jovian: tuple = (2.0, 0.95)
    f_masses: tuple = (1.0, 5.0)
    # Pulsar planets.
    pulsar_rate: float = 0.002
    pulsar_n_max: int = 3
    pulsar_mass_range: tuple = (0.01, 10.0)  # Earth masses
    pulsar_a_range: tuple = (0.1, 1.0)  # au
    pulsar_ecc_sigma: float = 0.02
    pulsar_incl_sigma_deg: float = 1.4
    # Roche-lobe proxy for post-interaction binaries.
    interaction_fraction: float = 0.3
    # Tidal decay on the giant branches (Villaver & Livio 2009). False: stars still alive engulf
    # inside max(1, f_MV12) times their maximum radius instead.
    tides: bool = True
    conv_teff: tuple = (6000.0, 5000.0)  # convective envelope switched on between these Teff (K)
    tide_substeps: int = 4


# --- stellar side -----------------------------------------------------------------------

def evolved_hosts(components, model: EvolvedModel | None = EvolvedModel()) -> np.ndarray:
    """Mask of evolved stars and white dwarfs whose (main-sequence) planets are modelled."""
    kind = np.asarray(components["kind"])
    if model is None:
        return np.zeros(len(kind), dtype=bool)
    phase = np.asarray(components["phase"], dtype=float)
    mi = np.asarray(components.get("initial_mass", components["mass"]), dtype=float)
    giant = (kind == "star") & np.isin(phase, EVOLVED_PHASES)
    return (giant | (kind == "WD")) & (mi <= model.max_initial_mass)


def progenitor_state(initial_mass, feh, model: EvolvedModel = EvolvedModel()) -> dict[str, np.ndarray]:
    """Mass, radius, Teff and luminosity at the middle of the main sequence (MIST)."""
    mi = np.atleast_1d(np.asarray(initial_mass, dtype=float))
    feh = np.broadcast_to(np.asarray(feh, dtype=float), mi.shape)
    log_age = mist.lifetime(mi, feh) - model.progenitor_offset_dex
    m = mist.evolve(mi, log_age, feh)
    return {"mass": m["star_mass"], "radius": 10 ** m["log_R"], "teff": 10 ** m["log_Teff"],
            "luminosity": 10 ** m["log_L"]}


@lru_cache(maxsize=1)
def _tip_grid():
    """Per MIST [Fe/H]: initial-mass grid and log10 R_RGB, log10 R_AGB tips (Rsun)."""
    fehs, ages, off, cnt, tab = mist.load_grid()
    masses = np.geomspace(0.5, 8.5, 160)
    rgb = np.full((len(fehs), len(masses)), np.nan)
    agb = np.full((len(fehs), len(masses)), np.nan)
    for f in range(len(fehs)):
        mmax = np.array([tab[off[f, j] + cnt[f, j] - 1, 0] if cnt[f, j] else 0.0 for j in range(len(ages))])
        for i, m in enumerate(masses):
            alive = np.flatnonzero(mmax >= m)
            if not len(alive):
                continue
            j = alive.max()
            rows = tab[off[f, j]:off[f, j] + cnt[f, j]]
            ph = rows[:, 6]
            for out, sel in ((rgb, (ph == PHASE_RGB) | (ph == PHASE_CHEB)), (agb, (ph >= PHASE_EAGB) & (ph <= PHASE_POSTAGB))):
                if sel.any():
                    out[f, i] = rows[sel, 4].max()
        for out in (rgb, agb):  # fill gaps along mass
            good = np.isfinite(out[f])
            if good.any():
                out[f] = np.interp(np.log(masses), np.log(masses[good]), out[f, good])
    return fehs, masses, rgb, agb


def radius_tips(initial_mass, feh) -> tuple[np.ndarray, np.ndarray]:
    """Largest radius (au) on the RGB (incl. core-He burning) and on the AGB, nearest [Fe/H]."""
    fehs, masses, rgb, agb = _tip_grid()
    mi = np.atleast_1d(np.asarray(initial_mass, dtype=float))
    f = np.abs(np.asarray(feh, dtype=float).reshape(-1)[:, None] - fehs[None, :]).argmin(axis=1)
    f = np.broadcast_to(f, mi.shape)
    x = np.log(np.clip(mi, masses[0], masses[-1]))
    lm = np.log(masses)
    j = np.clip(np.searchsorted(lm, x) - 1, 0, len(lm) - 2)
    w = (x - lm[j]) / (lm[j + 1] - lm[j])
    r_rgb = 10 ** ((1 - w) * rgb[f, j] + w * rgb[f, j + 1]) * RSUN_AU
    r_agb = 10 ** ((1 - w) * agb[f, j] + w * agb[f, j + 1]) * RSUN_AU
    return r_rgb, r_agb


def max_past_radius(components, idx) -> tuple[np.ndarray, np.ndarray]:
    """For components ``idx``: maximum radius reached so far (au) and the RGB-tip radius (au)."""
    kind = np.asarray(components["kind"])[idx]
    phase = np.asarray(components["phase"], dtype=float)[idx]
    radius = np.asarray(components["radius"], dtype=float)[idx] * RSUN_AU
    r_rgb, r_agb = radius_tips(np.asarray(components["initial_mass"], dtype=float)[idx],
                               np.asarray(components["feh"], dtype=float)[idx])
    alive_giant = (kind == "star") & (phase <= PHASE_TPAGB)
    r = np.where(alive_giant, radius, 0.0)
    r = np.where(alive_giant & (phase >= PHASE_CHEB), np.maximum(r, r_rgb), r)
    done = ~alive_giant  # post-AGB and white dwarfs
    r = np.where(done, np.maximum(r_rgb, r_agb), r)
    return r, r_rgb


def engulfment_radius(planet_mass, initial_mass, r_max, r_rgb, alive, model: EvolvedModel = EvolvedModel()):
    """Initial semi-major axis (au) inside which a planet has been engulfed."""
    x = np.clip((np.log(initial_mass) - math.log(model.f_masses[0]))
                / (math.log(model.f_masses[1]) - math.log(model.f_masses[0])), 0.0, 1.0)
    f_t = model.f_terrestrial[0] + x * (model.f_terrestrial[1] - model.f_terrestrial[0])
    f_j = model.f_jovian[0] + x * (model.f_jovian[1] - model.f_jovian[0])
    w = np.clip(np.log(np.maximum(planet_mass, 1e-6)) / math.log(JUPITER_MASS_EARTH), 0.0, 1.0) ** 2
    f = f_t + w * (f_j - f_t)
    f_alive = 1.0 if model.tides else np.maximum(f, 1.0)  # with tides, the decay is integrated instead
    return np.where(alive, r_max * f_alive, np.maximum(f * r_max, r_rgb))


# --- tidal decay on the giant branches ----------------------------------------------------

TAU_CONV_YR = (MSUN_G * RSUN_CM**2 / LSUN_CGS) ** (1.0 / 3.0) / YEAR_S  # (Msun Rsun^2 / Lsun)^(1/3), years


@njit(cache=True)
def _match_state(tab, off, cnt, f, j, phase, log_l, log_teff):
    """Initial mass of the row of isochrone (f, j) in the given phase closest to (log L, log Teff)."""
    out = np.full(len(f), np.inf)
    for h in range(len(f)):
        o, n = off[f[h], j[h]], cnt[f[h], j[h]]
        best = np.inf
        for r in range(n):
            if tab[o + r, 6] != phase[h]:
                continue
            d = (tab[o + r, 2] - log_l[h]) ** 2 + (10.0 * (tab[o + r, 3] - log_teff[h])) ** 2
            if d < best:
                best = d
                out[h] = tab[o + r, 0]
    return out


def _tracks(initial_mass, feh, age_gyr, alive, phase, luminosity, teff):
    """Per host: MIST [Fe/H] index, isochrone index, the reference time and initial mass that
    anchor the host's clock (host time = t_ref (m / m_ref)^beta for the isochrone row of initial
    mass m, homology with lifetime ~ M^-beta), and the last row to integrate to (``m_end``).

    Alive hosts: the isochrone just younger than the host, on which the row in the host's
    phase closest in (L, Teff) is its present state (t_ref = present age). White dwarfs and
    post-AGB stars: the last isochrone on which the progenitor is alive, to its end."""
    fehs, ages, off, cnt, tab = mist.load_grid()
    mi = np.asarray(initial_mass, dtype=float)
    feh = np.asarray(feh, dtype=float)
    alive = np.asarray(alive, dtype=bool)
    f = np.abs(feh[:, None] - fehs[None, :]).argmin(axis=1)
    last = np.where(cnt > 0, off + cnt - 1, 0)
    mmax = np.where(cnt > 0, tab[last, 0], 0.0)  # heaviest star alive on each isochrone
    lt = mist.lifetime(mi, feh)
    beta = (lt - mist.lifetime(1.05 * mi, feh)) / math.log10(1.05)
    beta = np.clip(np.nan_to_num(beta, nan=3.0, posinf=3.0, neginf=3.0), 1.0, 6.0)
    log_t = np.log10(np.maximum(age_gyr, 1e-3) * 1e9)
    j_alive = np.clip(np.searchsorted(ages, log_t, side="right") - 1, 0, len(ages) - 1)
    alive_on = mmax[f] >= mi[:, None]
    j_dead = np.where(alive_on.any(axis=1), len(ages) - 1 - np.argmax(alive_on[:, ::-1], axis=1), 0)
    j = np.where(alive, j_alive, j_dead)
    m_end = mmax[f, j].copy()
    t_ref, m_ref = 10.0 ** ages[j], mi.copy()
    a = np.flatnonzero(alive)
    if len(a):
        m_now = _match_state(tab, off, cnt, f[a], j[a], np.asarray(phase, dtype=float)[a],
                             np.log10(np.asarray(luminosity, dtype=float)[a]), np.log10(np.asarray(teff, dtype=float)[a]))
        ok = np.isfinite(m_now)
        m_end[a] = np.where(ok, m_now, m_end[a])
        t_ref[a] = np.where(ok, 10.0 ** log_t[a], t_ref[a])
        m_ref[a] = np.where(ok, m_now, m_ref[a])
    return f, j, t_ref, m_ref, m_end, beta


@njit(cache=True)
def _host_track(tab, o, n, mi, m_end, beta, t_ref, m_ref, teff_hi, teff_lo, nsub, tau_unit, out):
    """Fill ``out`` (rows: dt yr, mass Msun, M_env Msun, R au, tau yr) with the host's post-main-
    sequence history; returns the number of steps."""
    rsun_au = 1.0 / 215.032
    k = 0
    for r in range(n - 1):
        m1, m2 = tab[o + r, 0], tab[o + r + 1, 0]
        if m1 >= m_end:
            break
        ph1, ph2 = tab[o + r, 6], tab[o + r + 1, 6]
        if ph1 < 2 or ph2 < 2:
            continue  # still on the main sequence
        if ph2 > 6:
            break  # remnant rows
        top = min(m2, m_end)
        t1 = t_ref * (m1 / m_ref) ** beta
        t2 = t_ref * (top / m_ref) ** beta
        for s in range(nsub):
            w = (s + 0.5) / nsub * (top - m1) / (m2 - m1)
            m_row = (1 - w) * m1 + w * m2
            mass = mi / m_row * ((1 - w) * tab[o + r, 1] + w * tab[o + r + 1, 1])
            core = mi / m_row * ((1 - w) * tab[o + r, 7] + w * tab[o + r + 1, 7])
            lum = 10.0 ** ((1 - w) * tab[o + r, 2] + w * tab[o + r + 1, 2])
            teff = 10.0 ** ((1 - w) * tab[o + r, 3] + w * tab[o + r + 1, 3])
            rad = 10.0 ** ((1 - w) * tab[o + r, 4] + w * tab[o + r + 1, 4])  # Rsun
            g = min(1.0, max(0.0, (teff_hi - teff) / (teff_hi - teff_lo)))
            m_env = g * max(mass - core, 0.0)
            out[k, 0] = (t2 - t1) / nsub
            out[k, 1] = mass
            out[k, 2] = m_env
            out[k, 3] = rad * rsun_au
            out[k, 4] = tau_unit * (m_env * rad * rad / (3.0 * lum)) ** (1.0 / 3.0) if m_env > 0 else 1.0
            k += 1
    return k


@njit(cache=True)
def _decay_kernel(tab, off, cnt, f, j, mi, m_end, beta, t_ref, m_ref, start, a0, mp, teff_hi, teff_lo, nsub, tau_unit,
                  engulfed, shrink):
    """Hosts h own planets start[h]:start[h+1] (sorted by host)."""
    track = np.empty((cnt.max() * nsub, 5))
    for h in range(len(mi)):
        o, n = off[f[h], j[h]], cnt[f[h], j[h]]
        k = _host_track(tab, o, n, mi[h], m_end[h], beta[h], t_ref[h], m_ref[h], teff_hi, teff_lo, nsub, tau_unit,
                        track)
        # Upper bound on the decay of a^8 (f <= 1, q at the lowest mass) per unit q(1+q).
        bound, r_max, m_min = 0.0, 0.0, mi[h]
        for s in range(k):
            m_min = min(m_min, track[s, 1])
            r_max = max(r_max, track[s, 3])
            if track[s, 2] > 0:
                bound += 8.0 * track[s, 2] / track[s, 1] / track[s, 4] * track[s, 3] ** 8 * track[s, 0]
        for i in range(start[h], start[h + 1]):
            q_hi = mp[i] / m_min
            # The star overfills its Roche lobe (Eggleton 1983) when R > r_L a: engulfed.
            big_q = (m_min / mp[i]) ** (1.0 / 3.0)
            r_lobe = 0.49 * big_q**2 / (0.6 * big_q**2 + math.log(1.0 + big_q))
            if r_lobe * a0[i] > 2.0 * r_max and bound * q_hi * (1 + q_hi) < 1e-8 * a0[i] ** 8:
                engulfed[i] = False
                shrink[i] = 1.0
                continue
            a = a0[i]
            tide = 1.0
            m_prev = -1.0
            done = False
            for s in range(k):
                mass, m_env, r_au, tau = track[s, 1], track[s, 2], track[s, 3], track[s, 4]
                if m_prev > 0:
                    a *= m_prev / mass  # adiabatic mass loss
                m_prev = mass
                if m_env > 0.0:
                    period = math.sqrt(a**3 / mass)
                    fac = 1.0 if tau <= 0.5 * period else (0.5 * period / tau) ** 2
                    q = mp[i] / mass
                    a8 = a**8 - 8.0 * fac / tau * (m_env / mass) * q * (1.0 + q) * r_au**8 * track[s, 0]
                    if a8 <= (r_au / r_lobe) ** 8:
                        done = True
                        break
                    a_new = a8 ** 0.125
                    tide *= a_new / a
                    a = a_new
                if r_lobe * a <= r_au:
                    done = True
                    break
            engulfed[i] = done
            shrink[i] = tide


def tidal_decay(a0, planet_mass, initial_mass, feh, age_gyr, alive, phase, luminosity, teff,
                model: EvolvedModel = EvolvedModel()):
    """Orbital decay by the equilibrium tide in the convective envelope of an evolved host
    (Zahn 1977; Villaver & Livio 2009, eqs. 7-8, with the envelope base at the centre), integrated
    from the end of the main sequence to today (alive hosts) or to the end of the AGB (white
    dwarfs), together with the adiabatic expansion from mass loss.

    da/dt = -(f / tau) (M_env / M) q (1 + q) (R / a)^8 a, tau = [M_env R^2 / (3 L)]^(1/3),
    f = 1, or (P / 2 tau)^2 when tau > P / 2.

    The host's past is read off one MIST isochrone: stars slightly lighter than the host are
    in the host's earlier states (:func:`_tracks`).
    M_env = M - M_He-core, switched on as Teff falls from 6000 to 5000 K (stars crossing the
    Hertzsprung gap have radiative envelopes). A planet is engulfed once the star fills its
    Roche lobe in the star-planet pair (R > r_L a, Eggleton 1983; r_L ~ 0.8 for planets).
    Main-sequence decay is not included (the hot-Jupiter rates are measured on the main
    sequence).

    ``a0`` (au, main-sequence orbit), ``planet_mass`` (Earth masses), per planet; host columns
    (initial mass, [Fe/H], age, alive, and the present MIST phase, L/Lsun and Teff, used for
    alive hosts) per planet. Returns (engulfed mask, tidal shrink factor of the surviving orbits).
    """
    a0 = np.asarray(a0, dtype=float)
    n = len(a0)
    engulfed, shrink = np.zeros(n, dtype=bool), np.ones(n)
    if not n:
        return engulfed, shrink
    fehs, ages, off, cnt, tab = mist.load_grid()
    cols = np.stack([np.broadcast_to(np.asarray(x, dtype=float), a0.shape)
                     for x in (initial_mass, feh, age_gyr, alive, phase, luminosity, teff)], axis=1)
    hosts, inv = np.unique(cols, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    start = np.searchsorted(inv[order], np.arange(len(hosts) + 1))
    f, j, t_ref, m_ref, m_end, beta = _tracks(hosts[:, 0], hosts[:, 1], hosts[:, 2], hosts[:, 3] > 0, hosts[:, 4],
                                              hosts[:, 5], hosts[:, 6])
    mp = np.asarray(planet_mass, dtype=float)[order] * EARTH_MASS_MSUN
    eng, shr = np.zeros(n, dtype=bool), np.ones(n)
    _decay_kernel(tab, off, cnt, f, j, np.ascontiguousarray(hosts[:, 0]), m_end, beta, t_ref, m_ref, start,
                  np.ascontiguousarray(a0[order]), mp, float(model.conv_teff[0]), float(model.conv_teff[1]),
                  int(model.tide_substeps), TAU_CONV_YR, eng, shr)
    engulfed[order], shrink[order] = eng, shr
    return engulfed, shrink


# --- remnants -----------------------------------------------------------------------------

def draw_pulsar_planets(n_hosts, mass, rng, model: EvolvedModel = EvolvedModel()) -> dict[str, np.ndarray]:
    """Planets of neutron stars (index into the ``n_hosts`` inputs in ``host``)."""
    from .inner.hmc import radius_pure_silicate_seager2007
    from .units import a_au_to_period

    has = np.flatnonzero(rng.random(n_hosts) < model.pulsar_rate)
    host = np.repeat(has, rng.integers(1, model.pulsar_n_max + 1, len(has)))
    k = len(host)
    lm = np.log(model.pulsar_mass_range)
    la = np.log(model.pulsar_a_range)
    m = np.exp(lm[0] + rng.random(k) * (lm[1] - lm[0]))
    a = np.exp(la[0] + rng.random(k) * (la[1] - la[0]))
    r = np.array([radius_pure_silicate_seager2007(x) for x in m])
    e = np.minimum(model.pulsar_ecc_sigma * np.hypot(rng.normal(size=k), rng.normal(size=k)), 0.3)
    out = {"host": host, "a_au": a, "period": a_au_to_period(a, np.asarray(mass, dtype=float)[host]),
           "planet_mass": m, "planet_radius": r, "ecc": e, "omega": 2 * np.pi * rng.random(k),
           "mean_anom": 2 * np.pi * rng.random(k)}
    order = np.lexsort((out["period"], out["host"]))
    return {key: v[order] for key, v in out.items()}
