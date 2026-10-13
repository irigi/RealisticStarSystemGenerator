"""Attach planets to the stellar components drawn by :func:`starsys.stars.population.draw_systems`.

Every component is assigned the planet models that apply to it; their outputs are
merged into one planet table keyed by (system, component). Phase 4.1 wires up the
inner-system model only:

- **HM-C** (He & Ford 2026; port in :mod:`starsys.inner.hmc`), P = 3-300 d. Fitted to
  Kepler FGK dwarfs with de-reddened 0.5 <= bp-rp <= 1.7. Host colour is derived from
  Teff with :func:`starsys.stars.spectral.bp_rp_from_teff` (Mamajek dwarf sequence on
  the Kepler colour scale) plus a random draw of the Kepler colour scatter at fixed
  Teff, because f_swp was fitted on measured colours (without the scatter, the host
  fraction of Kepler stars comes out 0.003 too high). Model parameters are a random draw from the 100 ABC
  posterior catalogs per system (components of a system share the draw), unless a
  fixed parameter set is given.

Hosts: hydrogen-burning stars on the pre-main sequence or main sequence. Brown dwarfs,
evolved stars and remnants get no planets yet (phases 4.2 and 4.6). Planets are
drawn around every component of a multiple independently; binary pruning is phase 4.5.

Extrapolation is recorded per planet in the ``flags`` bit mask (``FLAG_*`` below).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .inner import hmc
from .stars.spectral import bp_rp_from_teff, kepler_bp_rp_scatter
from .syssim_catalogs import MODELS, available_catalog_ids, read_physical_header
from .units import period_to_a_au

# HM-C fitted domain in host colour (de-reddened Gaia bp-rp).
HMC_COLOUR_RANGE = (0.5, 1.7)

# Extrapolation flags (bit mask, per planet).
FLAG_COLOUR = 1  # host colour outside HM-C's fitted 0.5-1.7 (hotter than ~F2 or cooler than ~K7)
FLAG_PRE_MS = 2  # host is on the pre-main sequence (HM-C was fitted to main-sequence dwarfs)
FLAG_MULTIPLE = 4  # host has stellar companions; S-type stability not yet applied (phase 4.5)

FLAG_NAMES = {FLAG_COLOUR: "colour", FLAG_PRE_MS: "pre_ms", FLAG_MULTIPLE: "multiple"}

# MIST phases: -1 pre-MS, 0 MS. Stars below 0.1 Msun (BHAC15) have no MIST phase.
_PHASE_PRE_MS, _PHASE_MS = -1, 0


@lru_cache(maxsize=1)
def hmc_posterior() -> tuple[tuple[int, ...], tuple[hmc.HMCParams, ...]]:
    """Catalog ids and HM-C parameters of the 100 published posterior draws."""
    model = MODELS["hmc"]
    ids = tuple(available_catalog_ids(model))
    return ids, tuple(hmc.HMCParams.from_header(read_physical_header(model, cid)) for cid in ids)


def hmc_hosts(components) -> np.ndarray:
    """Boolean mask of the components that get HM-C planets."""
    kind = np.asarray(components["kind"])
    phase = np.asarray(components["phase"], dtype=float)
    return (kind == "star") & (np.isin(phase, (_PHASE_PRE_MS, _PHASE_MS)) | np.isnan(phase))


def host_flags(components, systems) -> np.ndarray:
    """Per-component extrapolation flags for HM-C."""
    colour = bp_rp_from_teff(components["teff"])
    lo, hi = HMC_COLOUR_RANGE
    flags = np.where((colour < lo) | (colour > hi), FLAG_COLOUR, 0)
    flags |= np.where(np.asarray(components["phase"]) == _PHASE_PRE_MS, FLAG_PRE_MS, 0)
    n_comp = np.asarray(systems["n_components"])[components["system"]]
    flags |= np.where(n_comp > 1, FLAG_MULTIPLE, 0)
    return flags.astype(np.int32)


def attach_planets(systems, components, seed=None, *, params: hmc.HMCParams | None = None,
                   age: str = "fixed", amd_pair_criterion: str = "hill") -> dict[str, np.ndarray]:
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
    age
        ``"fixed"``: photoevaporation uses HM-C's ``system_age`` (5 Gyr) for every star, as
        published. ``"star"``: use each star's own age (the authors list the fixed age as
        a limitation; this is outside what was fitted).
    amd_pair_criterion
        Passed to :func:`starsys.inner.hmc.generate_catalog`.

    Returns
    -------
    dict of per-planet arrays, sorted by system, component and period: ``system``,
    ``component`` (row index into ``components``), ``role``, ``model`` ("hmc"),
    ``hmc_catalog`` (posterior catalog id, 0 for fixed params), ``flags``, ``a_au``,
    and every column of :data:`starsys.inner.hmc.COLUMNS`.
    """
    if age not in ("fixed", "star"):
        raise ValueError("age must be 'fixed' or 'star'")
    rng = np.random.default_rng(seed)
    n_sys = len(systems["age_gyr"])
    host = np.flatnonzero(hmc_hosts(components))
    sys_of_host = np.asarray(components["system"])[host]

    if params is None:
        ids, posterior = hmc_posterior()
        draw = rng.integers(len(posterior), size=n_sys)[sys_of_host]
        groups = [(ids[k], posterior[k], host[draw == k]) for k in range(len(posterior))]
    else:
        groups = [(0, params, host)]

    teff = np.asarray(components["teff"], dtype=float)
    colour = bp_rp_from_teff(teff) + rng.normal(0.0, 1.0, len(teff)) * kepler_bp_rp_scatter(teff)
    flags = host_flags(components, systems)
    parts = []
    for cat_id, p, idx in groups:
        if len(idx) == 0:
            continue
        star_age = np.asarray(components["age_gyr"])[idx] if age == "star" else None
        _, pl = hmc.generate_catalog(p, components["mass"][idx], components["radius"][idx], teff[idx], colour[idx],
                                     age=star_age, seed=int(rng.integers(2**31)),
                                     amd_pair_criterion=amd_pair_criterion)
        comp = idx[pl.pop("star_index")]
        pl["component"] = comp
        pl["hmc_catalog"] = np.full(len(comp), cat_id, dtype=np.int32)
        parts.append(pl)

    cols = ("component", "hmc_catalog") + hmc.COLUMNS
    if parts:
        out = {c: np.concatenate([pl[c] for pl in parts]) for c in cols}
    else:
        out = {c: np.empty(0) for c in cols}
        out["component"] = np.empty(0, dtype=np.int64)
    comp = out["component"]
    order = np.lexsort((out["period"], comp))
    out = {c: v[order] for c, v in out.items()}
    comp = out["component"]
    result = {
        "system": np.asarray(components["system"])[comp],
        "component": comp,
        "role": np.asarray(components["role"])[comp],
        "model": np.full(len(comp), "hmc", dtype=object),
        "flags": flags[comp],
        "a_au": period_to_a_au(out["period"], np.asarray(components["mass"])[comp]),
    }
    result.update(out)
    return result


def describe_flags(flags) -> list[str]:
    """Human-readable names of the bits set in each flag value."""
    return [",".join(name for bit, name in FLAG_NAMES.items() if f & bit) for f in np.atleast_1d(flags)]
