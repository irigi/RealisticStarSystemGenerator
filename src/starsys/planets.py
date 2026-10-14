"""Attach planets to the stellar components drawn by :func:`starsys.stars.population.draw_systems`.

Every component is assigned the planet models that apply to it; their outputs are
merged into one planet table keyed by (system, component):

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
- **Giant planets** (:mod:`starsys.outer.giants`), M >= 30 Earth masses: warm and cold
  giants from the California Legacy Survey to 30 au (``model`` = "giant"), hot Jupiters
  ("hot_jupiter") and wide giants from direct imaging to 100 au ("wide_giant"), scaled
  with stellar mass and [Fe/H].
- **Small planets beyond HM-C's P_max** (:mod:`starsys.outer.small`), 0.1-30 Earth
  masses out to 10 snow-line radii ("outer_small"): the microlensing mass-ratio function
  beyond the snow line, bridged to HM-C's density at P_max, and placed around the inner
  planets and giants with the same stability criteria.

HM-C and the giant model overlap inside HM-C's period window (P_min-300 d), where HM-C was fitted to
planets below 10 Earth radii: there, giant-model planets are kept only if their radius
is 10 Earth radii or more. Hot-Jupiter hosts lose their inner system unless it is kept
with probability ``hj_keep_inner``. An inner system that is not stable next to the
host's giants (:func:`starsys.outer.giants.pair_stable`) is redrawn, alternating with
the eccentricities of its giants, up to ``GIANT_REDRAWS`` times, after which the
offending planets are removed. Redrawing rather
than removing keeps most of HM-C's inner occurrence (fitted to Kepler stars, which
include giant hosts) and agrees with RV surveys, which find inner small planets about as
common with or without cold Jupiters (Bonomo+2025; Ruggieri+2026). Small planets beyond
P_max are drawn last and fitted around the inner planets and giants, which do not move.

Hosts: hydrogen-burning stars on the pre-main sequence or main sequence, and brown
dwarfs (no giants below 0.1 Msun). Evolved stars and remnants get no planets yet (phase 4.6). Planets are drawn
around every component of a multiple independently; binary pruning is phase 4.5.

Extrapolation is recorded per planet in the ``flags`` bit mask (``FLAG_*`` below).
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

import numpy as np

from .inner import hmc
from .inner.mdwarf import MDwarfModel, adjust_params, mdwarf_weight
from .outer import giants as gp
from .outer import small as sp
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
FLAG_GIANT_HOST = 16  # giant planet around a host outside the giant calibration's 0.2-3 Msun
FLAG_BRIDGE = 32  # small planet between HM-C's P_max and the snow line: interpolated, not measured
FLAG_COLD_EXTRAP = 64  # small planet beyond the snow line outside the microlensing-measured mass ratios
                       # (log q < -5.2), separations (> 4 snow-line radii) or host masses (0.1-1 Msun)

FLAG_NAMES = {FLAG_COLOUR: "colour", FLAG_YOUNG: "young", FLAG_MULTIPLE: "multiple", FLAG_SUBSTELLAR: "substellar",
              FLAG_GIANT_HOST: "giant_host", FLAG_BRIDGE: "bridge", FLAG_COLD_EXTRAP: "cold_extrapolated"}
YOUNG_AGE_GYR = 0.1
GIANT_CLUSTER_ID = -1  # ``clusterid`` of giant-model planets
SMALL_CLUSTER_ID = -2  # ``clusterid`` of small planets beyond HM-C's P_max
GIANT_REDRAWS = 20  # inner-system redraws for hosts whose inner system is unstable next to its giants

DEFAULT_MDWARF = MDwarfModel()
DEFAULT_GIANTS = gp.GiantModel()
DEFAULT_SMALL = sp.SmallPlanetModel()

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
                   mdwarf: MDwarfModel | None = DEFAULT_MDWARF, giants: gp.GiantModel | None = DEFAULT_GIANTS,
                   small: sp.SmallPlanetModel | None = DEFAULT_SMALL, age: str = "fixed",
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
    giants
        Giant-planet model (:class:`starsys.outer.giants.GiantModel`); ``None`` for no giants.
    small
        Small planets beyond HM-C's P_max (:class:`starsys.outer.small.SmallPlanetModel`);
        ``None`` for none.
    age
        ``"fixed"``: photoevaporation uses HM-C's ``system_age`` (5 Gyr) for every star, as
        published. ``"star"``: use each star's own age (the authors list the fixed age as
        a limitation; this is outside what was fitted).
    amd_pair_criterion
        Passed to :func:`starsys.inner.hmc.generate_catalog`.

    Returns
    -------
    dict of per-planet arrays, sorted by system, component and period: ``system``,
    ``component`` (row index into ``components``), ``role``, ``model`` ("hmc",
    "hmc_m", "giant", "hot_jupiter", "wide_giant" or "outer_small"), ``hmc_catalog`` (posterior
    catalog id of the system, 0 for fixed params), ``flags``, ``a_au``, and every
    column of :data:`starsys.inner.hmc.COLUMNS`. For giants (``clusterid`` -1) and small
    planets beyond P_max (``clusterid`` -2), the angles ``incl_mut``/``incl_invariable``/
    ``asc_node_invariable`` refer to the inner system's invariable plane (or a random plane
    without one). For giants, ``mass_init`` and ``radius_init`` equal the final values.
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
        table = params.as_array()[None, :]
        draw = np.zeros(len(host), dtype=np.int64)
        rows = np.broadcast_to(table[0], (len(host), len(fields(hmc.HMCParams))))
        cat_id = np.zeros(len(host), dtype=np.int32)
    if mdwarf is not None:
        rows = adjust_params(rows, mass, teff, mdwarf)
        is_m = mdwarf_weight(teff, mdwarf) > 0
    else:
        is_m = np.zeros(len(host), dtype=bool)

    colour = bp_rp_from_teff(teff) + rng.normal(0.0, 1.0, len(teff)) * kepler_bp_rp_scatter(teff)
    radius = np.asarray(components["radius"], dtype=float)[host]
    star_age = np.asarray(components["age_gyr"], dtype=float)[host] if age == "star" else None

    def inner(idx):
        """HM-C systems for hosts ``idx`` (positions in ``host``): (host position per planet, planets, planes)."""
        _, p, planes = hmc.generate_catalog(rows[idx], mass[idx], radius[idx], teff[idx], colour[idx],
                                            age=None if star_age is None else star_age[idx],
                                            seed=int(rng.integers(2**31)), amd_pair_criterion=amd_pair_criterion,
                                            return_planes=True)
        return idx[p.pop("star_index")], p, planes

    k, pl, planes = inner(np.arange(len(host)))
    if giants is not None:
        k, pl = _add_giants(k, pl, planes, inner, rows, mass, host, components, giants, rng)
    if small is not None:
        edge = sp.edge_table(table, mdwarf, small)[draw]
        ml = {key: v[sys_of_host] for key, v in sp.draw_ml_params(n_sys, rng, small).items()}
        k, pl = _add_small(k, pl, planes, rows, edge, colour, ml, mass, radius, teff, star_age, small, rng)
    model = np.where(is_m[k], "hmc_m", "hmc").astype(object)
    if giants is not None:
        names = np.array([gp.COMPONENT_NAMES[c] for c in (gp.COLD, gp.HOT, gp.WIDE)], dtype=object)
        model = np.where(pl["clusterid"] == GIANT_CLUSTER_ID, names[pl["giant_component"]], model)
    pl.pop("giant_component", None)
    model = np.where(pl["clusterid"] == SMALL_CLUSTER_ID, "outer_small", model)
    small_flags = pl.pop("small_flags", np.zeros(len(k), dtype=np.int32))

    comp = host[k]
    flags = host_flags(components, systems, mdwarf)[comp]
    if giants is not None:
        lo, hi = giants.host_mass_range
        is_giant = pl["clusterid"] == GIANT_CLUSTER_ID
        flags |= np.where(is_giant & ((mass[k] < lo) | (mass[k] > hi)), FLAG_GIANT_HOST, 0).astype(np.int32)
    flags |= small_flags
    result = {
        "system": np.asarray(components["system"])[comp],
        "component": comp,
        "role": np.asarray(components["role"])[comp],
        "model": model,
        "hmc_catalog": cat_id[k],
        "flags": flags,
        "a_au": period_to_a_au(pl["period"], mass[k]),
    }
    result.update(pl)
    return result


def _add_giants(k, pl, planes, inner, rows, mass, host, components, giants: gp.GiantModel, rng):
    """Draw giants, reconcile the inner systems with them and merge; sorted by host position and period."""
    window = (rows[:, hmc.I_PMIN], rows[:, hmc.I_PMAX], rows[:, hmc.I_RMAX])
    g = gp.draw_giants(mass, np.asarray(components["feh"], dtype=float)[host],
                       np.asarray(components["luminosity"], dtype=float)[host], rng, giants, window=window)

    def drop(sel):
        return {key: v[~sel] for key, v in pl.items()}

    # Hot-Jupiter hosts usually lose their inner system.
    hj = np.unique(g["host"][g["hot"]])
    cleared = np.isin(k, hj[rng.random(len(hj)) >= giants.hj_keep_inner])
    k, pl = k[~cleared], drop(cleared)

    def unstable(k, pl):
        return gp.inner_conflicts(k, period_to_a_au(pl["period"], mass[k]), pl["planet_mass"], pl["ecc"], g, mass)

    # Alternately redraw the giants' eccentricities and the inner systems of unstable hosts;
    # remove what is still unstable.
    bad = unstable(k, pl)
    for _ in range(GIANT_REDRAWS):
        redo = np.unique(k[bad])
        if not len(redo):
            break
        gp.redraw_eccentricities(g, redo, mass, rng, giants)
        sel = np.isin(k, redo)
        bad[sel] = unstable(k[sel], {key: v[sel] for key, v in pl.items()})
        redo = np.unique(k[bad])
        sel = np.isin(k, redo)
        k, pl = k[~sel], drop(sel)
        k_new, pl_new, planes_new = inner(redo)
        planes[redo] = planes_new
        bad = np.concatenate([np.zeros(len(k), dtype=bool), unstable(k_new, pl_new)])
        k = np.concatenate([k, k_new])
        pl = {key: np.concatenate([pl[key], pl_new[key]]) for key in pl}
    k, pl = k[~bad], drop(bad)

    # Hosts without an inner system get one random reference plane, shared with later components.
    missing = np.isnan(planes[:, 0])
    planes[missing] = gp._random_unit_vectors(missing.sum(), rng)
    o = gp.orient(g["host"], planes, len(host), rng, giants)
    n = len(g["host"])
    gcols = {
        "period": g["period"], "planet_mass": g["planet_mass"], "planet_radius": g["planet_radius"],
        "clusterid": np.full(n, GIANT_CLUSTER_ID, dtype=pl["clusterid"].dtype), "ecc": g["ecc"],
        "omega": g["omega"], "incl_mut": o["incl_mut"], "incl_invariable": o["incl_mut"],
        "asc_node_invariable": o["asc_node"], "incl_sky": o["incl_sky"], "mean_anom": g["mean_anom"],
        "mass_init": g["planet_mass"], "radius_init": g["planet_radius"], "lost_envelope": np.zeros(n, dtype=bool),
        "giant_component": g["component"],
    }
    pl["giant_component"] = np.full(len(k), -1, dtype=np.int8)
    k = np.concatenate([k, g["host"]])
    pl = {key: np.concatenate([pl[key], gcols[key]]) for key in pl}
    order = np.lexsort((pl["period"], k))
    return k[order], {key: v[order] for key, v in pl.items()}


def _add_small(k, pl, planes, rows, edge, colour, ml, mass, radius, teff, star_age, small: sp.SmallPlanetModel, rng):
    """Draw small planets beyond P_max around the inner planets and giants; merged and sorted."""
    inner_pl = np.flatnonzero(pl["clusterid"] >= 0)
    last = inner_pl[np.r_[k[inner_pl][1:] != k[inner_pl][:-1], True]] if len(inner_pl) else inner_pl  # outermost
    giant = np.flatnonzero(pl["clusterid"] == GIANT_CLUSTER_ID)
    occ = np.concatenate([last, giant])
    occupied = {"host": k[occ], "a_au": period_to_a_au(pl["period"][occ], mass[k[occ]]),
                "planet_mass": pl["planet_mass"][occ], "ecc": pl["ecc"][occ],
                "giant": np.r_[np.zeros(len(last), dtype=bool), np.ones(len(giant), dtype=bool)]}
    s = sp.draw_small(mass, radius, teff, rows, edge, colour, ml, rng, small, occupied=occupied, planes=planes,
                      age=star_age)
    n = len(s["host"])
    flags = np.where(s["bridge"], FLAG_BRIDGE, 0) | np.where(s["extrapolated"], FLAG_COLD_EXTRAP, 0)
    scols = {
        "period": s["period"], "planet_mass": s["planet_mass"], "planet_radius": s["planet_radius"],
        "clusterid": np.full(n, SMALL_CLUSTER_ID, dtype=pl["clusterid"].dtype), "ecc": s["ecc"],
        "omega": s["omega"], "incl_mut": s["incl_mut"], "incl_invariable": s["incl_mut"],
        "asc_node_invariable": s["asc_node"], "incl_sky": s["incl_sky"], "mean_anom": s["mean_anom"],
        "mass_init": s["mass_init"], "radius_init": s["radius_init"], "lost_envelope": s["lost_envelope"],
        "giant_component": np.full(n, -1, dtype=np.int8), "small_flags": flags.astype(np.int32),
    }
    pl.setdefault("giant_component", np.full(len(k), -1, dtype=np.int8))
    pl["small_flags"] = np.zeros(len(k), dtype=np.int32)
    k = np.concatenate([k, s["host"]])
    pl = {key: np.concatenate([pl[key], scols[key]]) for key in pl}
    order = np.lexsort((pl["period"], k))
    return k[order], {key: v[order] for key, v in pl.items()}


def describe_flags(flags) -> list[str]:
    """Human-readable names of the bits set in each flag value."""
    return [",".join(name for bit, name in FLAG_NAMES.items() if f & bit) for f in np.atleast_1d(flags)]
