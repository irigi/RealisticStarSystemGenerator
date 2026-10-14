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
Terrestrials). These ratios are applied to the MIST R_max. Stars still on the giant
branches have lost little mass, so for them the factor is at least 1. White dwarfs also
pass their RGB tip: a_engulf = max(f R_max, R_RGB).

**Orbital expansion:** survivors' semi-major axes grow by M_initial / M_now (adiabatic mass
loss); periods follow. MV12 show that tides shrink orbits further only within ~0.5 au of the
engulfment radius. Planet radii and atmospheres are not changed (phase 5).

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

from .stars import mist

RSUN_AU = 1.0 / 215.032
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
    return np.where(alive, r_max * np.maximum(f, 1.0), np.maximum(f * r_max, r_rgb))


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
