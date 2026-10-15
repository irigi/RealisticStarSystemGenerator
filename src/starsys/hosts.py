"""Planet hosts: the rows the planet models are run on.

One row per **S-type host** (a component that gets planets: pre-main-sequence and
main-sequence stars, brown dwarfs and, with the evolved-star model, subgiants, giants and
white dwarfs, whose planets are drawn on the progenitor's main-sequence state) and, with
the binary model, one per **circumbinary host** (the inner pair of a multiple, see
:mod:`starsys.binaries`). Each row carries the host properties the planet models use, the
dynamical limits [a_min, a_max] of its planets, a reference plane (NaN = isotropic) and the
host-level flags. Rows are ordered by system, component, then S before P.
"""

from __future__ import annotations

import numpy as np

from . import binaries as bn
from . import evolved as ev
from .flags import (FLAG_CIRCUMBINARY, FLAG_COLOUR, FLAG_EVOLVED, FLAG_MULTIPLE, FLAG_POST_INTERACTION,
                    FLAG_SUBSTELLAR, FLAG_YOUNG)
from .inner.mdwarf import MDwarfModel
from .stars.spectral import bp_rp_from_teff

# HM-C fitted domain in host colour (de-reddened Gaia bp-rp).
HMC_COLOUR_RANGE = (0.5, 1.7)
YOUNG_AGE_GYR = 0.1
TEFF_SUN = 5772.0

# MIST phases: -1 pre-MS, 0 MS. Stars below 0.1 Msun (BHAC15) have no MIST phase.
_PHASE_PRE_MS, _PHASE_MS = -1, 0

S_TYPE, P_TYPE = 0, 1


def hmc_hosts(components) -> np.ndarray:
    """Boolean mask of the components that get planets as (pre-)main-sequence stars or brown dwarfs."""
    kind = np.asarray(components["kind"])
    phase = np.asarray(components["phase"], dtype=float)
    dwarf = (kind == "star") & (np.isin(phase, (_PHASE_PRE_MS, _PHASE_MS)) | np.isnan(phase))
    return dwarf | (kind == "BD")


def base_flags(teff, age, mass, multiple, mdwarf: MDwarfModel | None) -> np.ndarray:
    """Host-level extrapolation flags of the inner-system model."""
    colour = bp_rp_from_teff(teff)
    flags = np.where(colour < HMC_COLOUR_RANGE[0], FLAG_COLOUR, 0)
    flags |= np.where(np.asarray(age) < YOUNG_AGE_GYR, FLAG_YOUNG, 0)
    flags |= np.where(multiple, FLAG_MULTIPLE, 0)
    if mdwarf is not None:
        flags |= np.where(np.asarray(mass) < mdwarf.substellar_mass, FLAG_SUBSTELLAR, 0)
    else:
        # Without the M-dwarf extension, HM-C is extrapolated on the red side as well.
        flags |= np.where(colour > HMC_COLOUR_RANGE[1], FLAG_COLOUR, 0)
    return flags.astype(np.int32)


def host_flags(components, systems, mdwarf: MDwarfModel | None) -> np.ndarray:
    """Per-component flags without the binary model (any companion counts as ``multiple``)."""
    n_comp = np.asarray(systems["n_components"])[components["system"]]
    return base_flags(components["teff"], components["age_gyr"], components["mass"], n_comp > 1, mdwarf)


def build_hosts(systems, components, rng, mdwarf: MDwarfModel | None = None,
                binaries: bn.BinaryModel | None = None, evolved: ev.EvolvedModel | None = None) -> dict:
    """Host rows (dict of arrays). Without ``binaries`` and ``evolved`` this is exactly the
    (pre-)main-sequence components and brown dwarfs, and no random numbers are used."""
    c = {key: np.asarray(components[key]) for key in ("system", "role", "kind", "phase", "mass", "radius", "teff",
                                                      "age_gyr")}
    n_c = len(c["system"])
    c["luminosity"] = np.asarray(components.get("luminosity", c["radius"] ** 2 * (c["teff"] / TEFF_SUN) ** 4))
    c["feh"] = np.asarray(components.get("feh", np.zeros(n_c)))
    c["initial_mass"] = np.asarray(components.get("initial_mass", c["mass"]))
    n_comp = np.asarray(systems["n_components"])[c["system"]]
    dwarf = hmc_hosts(components)
    old = ev.evolved_hosts(components, evolved)
    idx = np.flatnonzero(dwarf | old)
    n = len(idx)
    h = {
        "component": idx, "kind": np.full(n, S_TYPE, dtype=np.int8), "system": c["system"][idx],
        "mass": c["mass"][idx].astype(float), "radius": c["radius"][idx].astype(float),
        "teff": c["teff"][idx].astype(float), "luminosity": c["luminosity"][idx].astype(float),
        "feh": c["feh"][idx].astype(float), "age": c["age_gyr"][idx].astype(float),
        "a_min": np.zeros(n), "a_max": np.full(n, np.inf), "plane": np.full((n, 3), np.nan),
        "multiple": n_comp[idx] > 1, "extra_flags": np.zeros(n, dtype=np.int32),
        # Evolved hosts: maximum past radius, RGB-tip radius (au), initial and present mass.
        "evolved": old[idx], "r_max": np.zeros(n), "r_rgb": np.zeros(n), "mass_init": c["mass"][idx].astype(float),
        "mass_now": c["mass"][idx].astype(float), "alive": np.ones(n, dtype=bool),
        "p_bin": np.zeros(n),  # circumbinary hosts: period of the inner binary (d)
        # Present state (differs from mass/radius/teff/luminosity for evolved hosts).
        "phase_now": c["phase"][idx].astype(float), "teff_now": c["teff"][idx].astype(float),
        "lum_now": c["luminosity"][idx].astype(float),
    }
    st = None
    if evolved is not None and old.any():
        e = np.flatnonzero(h["evolved"])
        ce = idx[e]
        prog = ev.progenitor_state(c["initial_mass"][ce], c["feh"][ce], evolved)
        for key in ("mass", "radius", "teff", "luminosity"):
            h[key][e] = prog[key]
        h["mass_init"][e] = c["initial_mass"][ce]
        h["r_max"][e], h["r_rgb"][e] = ev.max_past_radius(components, ce)
        h["alive"][e] = c["kind"][ce] == "star"
        h["extra_flags"][e] |= FLAG_EVOLVED
        # Post-interaction binaries: an evolved component that overfilled its Roche lobe.
        st = bn.structure(components)
        r_all = np.zeros(len(c["system"]))
        r_all[ce] = h["r_max"][e]
        inter = r_all > evolved.interaction_fraction * np.nan_to_num(st["a_near_any"], nan=np.inf)
        bad_sys = np.zeros(len(systems["age_gyr"]), dtype=bool)
        bad_sys[c["system"][inter]] = True
        h["extra_flags"] |= np.where(bad_sys[h["system"]], FLAG_POST_INTERACTION, 0).astype(np.int32)

    if binaries is None:
        return h

    st = st or bn.structure(components)
    h["a_max"] = st["a_max"][idx]
    h["multiple"] = st["a_near_any"][idx] < binaries.flag_a_max
    keep = np.ones(n, dtype=bool)
    if binaries.suppress:
        keep = rng.random(n) < bn.s_bin(st["a_near"][idx], binaries)
    h["plane"] = bn.s_type_planes(st["a_near_any"][idx], st["near_normal"][idx], rng, binaries)
    h = {key: v[keep] for key, v in h.items()}

    if binaries.circumbinary:
        p = np.flatnonzero((c["role"] == 0) & np.isfinite(st["pair_a_min"]))
        s = p + 1
        ok = dwarf[p] & dwarf[s]
        m_pair = c["mass"][p] + c["mass"][s]
        reach = np.maximum(100.0, 27.0 * m_pair)  # beyond the reach of every planet model
        ok &= (st["pair_a_min"][p] < reach) & (st["pair_a_min"][p] < st["pair_a_max"][p])
        p, s, m_pair = p[ok], s[ok], m_pair[ok]
        k = len(p)
        lum = c["luminosity"][p] + c["luminosity"][s]
        teff = c["teff"][p].astype(float)
        # Tertiary inside the flag separation: the role-2 companion's orbit.
        t = p + 2
        has_t = (t < len(c["system"])) & (c["system"][np.minimum(t, len(c["system"]) - 1)] == c["system"][p])
        a_comp = np.asarray(components.get("a_au", np.full(n_c, np.nan)), dtype=float)
        a_t = np.where(has_t, a_comp[np.minimum(t, n_c - 1)], np.inf)
        cb = {
            "component": p, "kind": np.full(k, P_TYPE, dtype=np.int8), "system": c["system"][p], "mass": m_pair,
            "radius": np.sqrt(lum) / (teff / TEFF_SUN) ** 2, "teff": teff, "luminosity": lum,
            "feh": c["feh"][p].astype(float), "age": c["age_gyr"][p].astype(float),
            "a_min": st["pair_a_min"][p], "a_max": st["pair_a_max"][p],
            "plane": bn.circumbinary_planes(st["pair_normal"][p], rng, binaries),
            "multiple": a_t < binaries.flag_a_max, "extra_flags": np.full(k, FLAG_CIRCUMBINARY, dtype=np.int32),
            "evolved": np.zeros(k, dtype=bool), "r_max": np.zeros(k), "r_rgb": np.zeros(k), "mass_init": m_pair,
            "mass_now": m_pair, "alive": np.ones(k, dtype=bool),
            "p_bin": np.asarray(components.get("period_days", np.full(n_c, np.nan)), dtype=float)[s],
            "phase_now": c["phase"][p].astype(float), "teff_now": teff, "lum_now": lum,
        }
        h = {key: np.concatenate([h[key], cb[key]]) for key in h}
        order = np.lexsort((h["kind"], h["component"]))
        h = {key: v[order] for key, v in h.items()}
    return h


def flags_of(h, mdwarf: MDwarfModel | None) -> np.ndarray:
    """Host-level flags of the rows of :func:`build_hosts`."""
    return base_flags(h["teff"], h["age"], h["mass"], h["multiple"], mdwarf) | h["extra_flags"]
