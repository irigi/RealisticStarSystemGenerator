"""Attach planets to the stellar components drawn by :func:`starsys.stars.population.draw_systems`.

Every component is assigned the planet models that apply to it; their outputs are
merged into one planet table keyed by (system, component). So far that is the inner
system:

- **HM-C** (He & Ford 2026; port in :mod:`starsys.inner.hmc`), P = 3-300 d, fitted to
  Kepler FGK dwarfs with de-reddened 0.5 <= bp-rp <= 1.7. Host colour comes from Teff
  via :func:`starsys.stars.spectral.bp_rp_from_teff` (Mamajek dwarf sequence on the
  Kepler colour scale) plus a random draw of the Kepler colour scatter at fixed Teff.
  f_swp was fitted on measured colours, so without that scatter the host fraction of
  Kepler stars comes out 0.003 too high. Model parameters are a random draw from the
  100 ABC posterior catalogs per system (components of a system share the draw),
  unless a fixed parameter set is given.
- **M dwarfs and below** (:mod:`starsys.inner.mdwarf`): the same model with the
  M-dwarf calibration (more planets, smaller planets, periods down to 0.5 d) phased
  in below 4000 K, and a disk-mass scaling of occurrence below 0.1 Msun
  (``model`` = "hmc_m").

Hosts: hydrogen-burning stars on the pre-main sequence or main sequence, and brown
dwarfs. Evolved stars and remnants get no planets yet (phase 4.6). Planets are drawn
around every component of a multiple independently; binary pruning is phase 4.5.

Extrapolation is recorded per planet in the ``flags`` bit mask (``FLAG_*`` below).
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

import numpy as np

from .inner import hmc
from .inner.mdwarf import MDwarfModel, adjust_params, mdwarf_weight
from .stars.spectral import bp_rp_from_teff, kepler_bp_rp_scatter
from .syssim_catalogs import MODELS, available_catalog_ids, read_physical_header
from .units import period_to_a_au

# HM-C fitted domain in host colour (de-reddened Gaia bp-rp).
HMC_COLOUR_RANGE = (0.5, 1.7)

# Extrapolation flags (bit mask, per planet).
FLAG_COLOUR = 1  # host colour outside the calibrated range: bp-rp < 0.5 (~F3 and hotter),
                 # or > 1.7 when the M-dwarf extension is off
FLAG_YOUNG = 2  # system younger than 100 Myr: atmospheric loss still under way
FLAG_MULTIPLE = 4  # host has stellar companions; S-type stability not yet applied (phase 4.5)
FLAG_SUBSTELLAR = 8  # host below 0.1 Msun: occurrence from disk-mass scaling, no data

FLAG_NAMES = {FLAG_COLOUR: "colour", FLAG_YOUNG: "young", FLAG_MULTIPLE: "multiple", FLAG_SUBSTELLAR: "substellar"}
YOUNG_AGE_GYR = 0.1

DEFAULT_MDWARF = MDwarfModel()

# MIST phases: -1 pre-MS, 0 MS. Stars below 0.1 Msun (BHAC15) have no MIST phase.
_PHASE_PRE_MS, _PHASE_MS = -1, 0


@lru_cache(maxsize=1)
def hmc_posterior() -> tuple[tuple[int, ...], tuple[hmc.HMCParams, ...]]:
    """Catalog ids and HM-C parameters of the 100 published posterior draws."""
    model = MODELS["hmc"]
    ids = tuple(available_catalog_ids(model))
    return ids, tuple(hmc.HMCParams.from_header(read_physical_header(model, cid)) for cid in ids)


def hmc_hosts(components) -> np.ndarray:
    """Boolean mask of the components that get inner-system planets."""
    kind = np.asarray(components["kind"])
    phase = np.asarray(components["phase"], dtype=float)
    dwarf = (kind == "star") & (np.isin(phase, (_PHASE_PRE_MS, _PHASE_MS)) | np.isnan(phase))
    return dwarf | (kind == "BD")


def host_flags(components, systems, mdwarf: MDwarfModel | None = DEFAULT_MDWARF) -> np.ndarray:
    """Per-component extrapolation flags for the inner-system model."""
    colour = bp_rp_from_teff(components["teff"])
    flags = np.where(colour < HMC_COLOUR_RANGE[0], FLAG_COLOUR, 0)
    flags |= np.where(np.asarray(components["age_gyr"]) < YOUNG_AGE_GYR, FLAG_YOUNG, 0)
    n_comp = np.asarray(systems["n_components"])[components["system"]]
    flags |= np.where(n_comp > 1, FLAG_MULTIPLE, 0)
    if mdwarf is not None:
        flags |= np.where(np.asarray(components["mass"]) < mdwarf.substellar_mass, FLAG_SUBSTELLAR, 0)
    else:
        # Without the M-dwarf extension, HM-C is extrapolated on the red side as well.
        flags |= np.where(colour > HMC_COLOUR_RANGE[1], FLAG_COLOUR, 0)
    return flags.astype(np.int32)


def attach_planets(systems, components, seed=None, *, params: hmc.HMCParams | None = None,
                   mdwarf: MDwarfModel | None = DEFAULT_MDWARF, age: str = "fixed",
                   amd_pair_criterion: str = "hill") -> dict[str, np.ndarray]:
    """Draw planets for every component of every system.

    Parameters
    ----------
    systems, components
        Output of :func:`starsys.stars.population.draw_systems` (dicts of arrays).
    seed
        Seed or ``numpy.random.Generator``.
    params
        Fixed HM-C parameters for all systems. ``None`` draws one of the 100 posterior
        catalogs per system (its id goes into ``hmc_catalog``).
    mdwarf
        M-dwarf extension (:class:`starsys.inner.mdwarf.MDwarfModel`). ``None`` runs plain
        HM-C everywhere (published model; M dwarfs are then flagged ``colour``).
    age
        ``"fixed"``: photoevaporation uses HM-C's ``system_age`` (5 Gyr) for every star, as
        published. ``"star"``: use each star's own age (the authors list the fixed age as
        a limitation; this is outside what was fitted).
    amd_pair_criterion
        Passed to :func:`starsys.inner.hmc.generate_catalog`.

    Returns
    -------
    dict of per-planet arrays, sorted by system, component and period: ``system``,
    ``component`` (row index into ``components``), ``role``, ``model`` ("hmc" or
    "hmc_m"), ``hmc_catalog`` (posterior catalog id, 0 for fixed params), ``flags``,
    ``a_au``, and every column of :data:`starsys.inner.hmc.COLUMNS`.
    """
    if age not in ("fixed", "star"):
        raise ValueError("age must be 'fixed' or 'star'")
    rng = np.random.default_rng(seed)
    n_sys = len(systems["age_gyr"])
    host = np.flatnonzero(hmc_hosts(components))
    sys_of_host = np.asarray(components["system"])[host]
    mass = np.asarray(components["mass"], dtype=float)[host]
    teff = np.asarray(components["teff"], dtype=float)[host]

    if params is None:
        ids, posterior = hmc_posterior()
        table = np.array([p.as_array() for p in posterior])
        draw = rng.integers(len(posterior), size=n_sys)[sys_of_host]
        rows, cat_id = table[draw], np.asarray(ids, dtype=np.int32)[draw]
    else:
        rows = np.broadcast_to(params.as_array(), (len(host), len(fields(hmc.HMCParams))))
        cat_id = np.zeros(len(host), dtype=np.int32)
    if mdwarf is not None:
        rows = adjust_params(rows, mass, teff, mdwarf)
        is_m = mdwarf_weight(teff, mdwarf) > 0
    else:
        is_m = np.zeros(len(host), dtype=bool)

    colour = bp_rp_from_teff(teff) + rng.normal(0.0, 1.0, len(teff)) * kepler_bp_rp_scatter(teff)
    star_age = np.asarray(components["age_gyr"], dtype=float)[host] if age == "star" else None
    _, pl = hmc.generate_catalog(rows, mass, np.asarray(components["radius"], dtype=float)[host], teff, colour,
                                 age=star_age, seed=int(rng.integers(2**31)),
                                 amd_pair_criterion=amd_pair_criterion)
    k = pl.pop("star_index")
    comp = host[k]
    flags = host_flags(components, systems, mdwarf)
    result = {
        "system": np.asarray(components["system"])[comp],
        "component": comp,
        "role": np.asarray(components["role"])[comp],
        "model": np.where(is_m[k], "hmc_m", "hmc").astype(object),
        "hmc_catalog": cat_id[k],
        "flags": flags[comp],
        "a_au": period_to_a_au(pl["period"], np.asarray(components["mass"])[comp]),
    }
    result.update(pl)
    return result


def describe_flags(flags) -> list[str]:
    """Human-readable names of the bits set in each flag value."""
    return [",".join(name for bit, name in FLAG_NAMES.items() if f & bit) for f in np.atleast_1d(flags)]
