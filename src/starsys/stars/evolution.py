"""Present-day state of a star (or substellar object) from initial mass, age and [Fe/H].

Dispatches to:
- ATMO 2020 / BHAC15 for M < 0.1 Msun (:mod:`.substellar`);
- MIST for M >= 0.1 Msun (:mod:`.mist`);
- remnants (IFMR + Montreal WD cooling, NS, BH) once a star has left the MIST
  isochrones (:mod:`.remnants`).
"""

from __future__ import annotations

import numpy as np

from . import mist, remnants, spectral, substellar

# Hydrogen-burning limit at solar metallicity (~0.075 Msun); objects below are brown dwarfs.
HBL = 0.075
TEFF_SUN = 5772.0


def evolve(initial_mass, age_gyr, feh, rng=None, with_spectral_type=True):
    """Return a dict of arrays describing each object today.

    Keys: kind ('BD', 'star', 'WD', 'NS', 'BH'), mass, teff, luminosity, radius,
    log_g, phase (MIST code, NaN otherwise), spectral_type, hydrogen_atmosphere
    (WDs only), cooling_age_gyr.
    """
    rng = np.random.default_rng(rng)
    mi = np.atleast_1d(np.asarray(initial_mass, dtype=float))
    age = np.atleast_1d(np.asarray(age_gyr, dtype=float))
    feh = np.atleast_1d(np.asarray(feh, dtype=float))
    n = len(mi)
    log_age = np.log10(age * 1e9)

    kind = np.full(n, "star", dtype=object)
    mass = mi.copy()
    log_teff = np.full(n, np.nan)
    log_l = np.full(n, np.nan)
    log_r = np.full(n, np.nan)
    log_g = np.full(n, np.nan)
    phase = np.full(n, np.nan)
    h_atm = np.zeros(n, dtype=bool)
    t_cool = np.full(n, np.nan)

    low = mi < 0.1
    if low.any():
        s = substellar.evolve_substellar(mi[low], log_age[low])
        log_teff[low], log_l[low], log_r[low] = s["log_Teff"], s["log_L"], s["log_R"]
        kind[low & (mi < HBL)] = "BD"

    high = ~low
    if high.any():
        m = mist.evolve(mi[high], log_age[high], feh[high])
        idx = np.flatnonzero(high)
        alive = m["phase"] > -50
        ia = idx[alive]
        mass[ia] = m["star_mass"][alive]
        log_teff[ia], log_l[ia], log_r[ia], log_g[ia] = (m["log_Teff"][alive], m["log_L"][alive],
                                                        m["log_R"][alive], m["log_g"][alive])
        phase[ia] = m["phase"][alive]
        # MIST phase 6 = post-AGB and early white-dwarf cooling: keep MIST's values, call it a WD.
        young_wd = ia[m["phase"][alive] == 6]
        kind[young_wd] = "WD"
        h_atm[young_wd] = rng.random(len(young_wd)) < remnants.DA_FRACTION
        t_cool[young_wd] = 0.0
        dead = idx[~alive]
        if len(dead):
            rtype = remnants.remnant_type(mi[dead])
            kind[dead] = rtype
            mass[dead] = remnants.remnant_mass(mi[dead])
            life = 10 ** mist.lifetime(mi[dead], feh[dead]) / 1e9
            t_cool[dead] = np.maximum(age[dead] - life, 1e-4)
            wd = dead[rtype == "WD"]
            if len(wd):
                h_atm[wd] = rng.random(len(wd)) < remnants.DA_FRACTION
                c = remnants.cool_white_dwarf(mass[wd], t_cool[wd] * 1e9, h_atm[wd])
                log_teff[wd], log_l[wd], log_r[wd], log_g[wd] = c["log_Teff"], c["log_L"], c["log_R"], c["log_g"]

    # Fill log g where missing (substellar objects): g = G M / R^2, solar log g = 4.438.
    miss = np.isnan(log_g) & np.isfinite(log_r)
    log_g[miss] = 4.438 + np.log10(mass[miss]) - 2.0 * log_r[miss]

    teff = 10**log_teff
    out = {
        "kind": kind, "initial_mass": mi, "mass": mass, "age_gyr": age, "feh": feh,
        "teff": teff, "luminosity": 10**log_l, "radius": 10**log_r, "log_g": log_g,
        "phase": phase, "hydrogen_atmosphere": h_atm, "cooling_age_gyr": t_cool,
    }
    if with_spectral_type:
        out["spectral_type"] = spectral_type_of(out)
    return out


def spectral_type_of(state) -> np.ndarray:
    """Spectral types of the objects in a state dict returned by :func:`evolve`."""
    return spectral.spectral_type(state["kind"], state["teff"], state["phase"], state["log_g"],
                                  np.log10(state["luminosity"]), state["hydrogen_atmosphere"])
