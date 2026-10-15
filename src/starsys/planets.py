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
- **Planets inside HM-C's inner limit** (:mod:`starsys.inner.short`), 0.5 d to P_min
  (3 d for FGK stars; ``model`` = "short_period"): Hsu+2019's Kepler occurrence and
  Sanchis-Ojeda+2014's ultra-short-period trend with stellar type, phased out for M dwarfs.
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

Hosts (:mod:`starsys.hosts`): hydrogen-burning stars on the pre-main sequence or main
sequence, and brown dwarfs (no giants below 0.1 Msun).

- **Multiples** (:mod:`starsys.binaries`, ``binaries=`` option): close stellar companions
  suppress planetary systems (Moe & Kratter 2021); planets must lie inside the Holman &
  Wiegert (1999) S-type limit; the inner pair of a multiple also hosts circumbinary planets
  (``host_kind`` "P", flag ``circumbinary``), drawn from the same models outside the P-type
  limit. Transit-fitted rates (HM-C and the M-dwarf extension) are boosted to single-star
  rates. Without the binary model, every component is treated as single and the rates are
  field averages.
- **Evolved stars and remnants** (:mod:`starsys.evolved`, ``evolved=`` option): subgiants,
  giants and white dwarfs get the planets of their main-sequence progenitor, minus those
  engulfed (Mustill & Villaver 2012), on orbits widened by the mass lost; neutron stars get
  rare pulsar planets ("pulsar"); black holes none.

Extrapolation is recorded per planet in the ``flags`` bit mask (``FLAG_*`` below).
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

import numpy as np

from . import binaries as bn
from . import evolved as ev
from .flags import (FLAG_BRIDGE, FLAG_CIRCUMBINARY, FLAG_COLD_EXTRAP, FLAG_COLOUR, FLAG_EVOLVED,  # noqa: F401
                    FLAG_GIANT_HOST, FLAG_MULTIPLE, FLAG_NAMES, FLAG_POST_INTERACTION, FLAG_PULSAR,
                    FLAG_SHORT_PERIOD, FLAG_SUBSTELLAR, FLAG_YOUNG)
from .hosts import (HMC_COLOUR_RANGE, P_TYPE, YOUNG_AGE_GYR, build_hosts, flags_of, hmc_hosts,  # noqa: F401
                    host_flags)
from .inner import hmc
from .inner import short as sh
from .inner.mdwarf import MDwarfModel, adjust_params, mdwarf_weight
from .outer import giants as gp
from .outer import small as sp
from .stars.spectral import bp_rp_from_teff, kepler_bp_rp_scatter
from .syssim_catalogs import MODELS, available_catalog_ids, read_physical_header
from .units import EARTH_MASS_MSUN, a_au_to_period, period_to_a_au

GIANT_CLUSTER_ID = -1  # ``clusterid`` of giant-model planets
SMALL_CLUSTER_ID = -2  # ``clusterid`` of small planets beyond HM-C's P_max
PULSAR_CLUSTER_ID = -3  # ``clusterid`` of pulsar planets
CB_CLUSTER_ID = -4  # ``clusterid`` of the empirical circumbinary component
SHORT_CLUSTER_ID = -5  # ``clusterid`` of planets inside HM-C's inner period limit
CB_SPLIT_RADIUS = 4.0  # circumbinary HM-C planets this large are replaced by the empirical component
GIANT_REDRAWS = 20  # inner-system redraws for hosts whose inner system is unstable next to its giants

DEFAULT_MDWARF = MDwarfModel()
DEFAULT_GIANTS = gp.GiantModel()
DEFAULT_SMALL = sp.SmallPlanetModel()
DEFAULT_BINARIES = bn.BinaryModel()
DEFAULT_EVOLVED = ev.EvolvedModel()
DEFAULT_SHORT = sh.ShortPeriodModel()


@lru_cache(maxsize=1)
def hmc_posterior() -> tuple[tuple[int, ...], tuple[hmc.HMCParams, ...]]:
    """Catalog ids and HM-C parameters of the 100 published posterior draws."""
    model = MODELS["hmc"]
    ids = tuple(available_catalog_ids(model))
    return ids, tuple(hmc.HMCParams.from_header(read_physical_header(model, cid)) for cid in ids)


def attach_planets(systems, components, seed=None, *, params: hmc.HMCParams | None = None,
                   mdwarf: MDwarfModel | None = DEFAULT_MDWARF, giants: gp.GiantModel | None = DEFAULT_GIANTS,
                   small: sp.SmallPlanetModel | None = DEFAULT_SMALL,
                   binaries: bn.BinaryModel | None = DEFAULT_BINARIES,
                   evolved: ev.EvolvedModel | None = DEFAULT_EVOLVED,
                   short: sh.ShortPeriodModel | None = DEFAULT_SHORT, age: str = "fixed",
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
    binaries
        Multiple-star model (:class:`starsys.binaries.BinaryModel`). ``None`` treats every
        component as single, with field-average (transit-survey) inner rates.
    evolved
        Evolved stars and remnants (:class:`starsys.evolved.EvolvedModel`); ``None`` gives
        them no planets.
    short
        Planets between 0.5 d and HM-C's inner period limit (:class:`starsys.inner.short.ShortPeriodModel`);
        ``None`` for none.
    age
        ``"fixed"``: photoevaporation uses HM-C's ``system_age`` (5 Gyr) for every star, as
        published. ``"star"``: use each star's own age (the authors list the fixed age as
        a limitation; this is outside what was fitted).
    amd_pair_criterion
        Passed to :func:`starsys.inner.hmc.generate_catalog`.

    Returns
    -------
    dict of per-planet arrays, sorted by system, component, host kind and period: ``system``,
    ``component`` (row index into ``components``; the primary for circumbinary planets),
    ``host_kind`` ("S", or "P" for planets orbiting components 0 and 1), ``host_mass`` (Msun,
    today), ``role``, ``model`` ("hmc", "hmc_m", "giant", "hot_jupiter", "wide_giant",
    "outer_small" or "pulsar"), ``hmc_catalog`` (posterior catalog id of the system, 0 for
    fixed params and pulsars), ``flags``, ``a_au``, and every column of
    :data:`starsys.inner.hmc.COLUMNS`. For giants (``clusterid`` -1), small planets beyond P_max
    (-2) and pulsar planets (-3), the angles ``incl_mut``/``incl_invariable``/
    ``asc_node_invariable`` refer to the inner system's invariable plane (or the host's
    reference plane without one). For giants, ``mass_init`` and ``radius_init`` equal the
    final values. Planets of evolved hosts are on their present (expanded) orbits.
    """
    if age not in ("fixed", "star"):
        raise ValueError("age must be 'fixed' or 'star'")
    rng = np.random.default_rng(seed)
    n_sys = len(systems["age_gyr"])
    h = build_hosts(systems, components, rng, mdwarf, binaries, evolved)
    host = h["component"]
    sys_of_host = h["system"]
    mass, teff, radius = h["mass"], h["teff"], h["radius"]

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
    star_age = h["age"] if age == "star" else None
    rest = np.ones(len(host))
    if binaries is not None and binaries.boost:
        rows, rest = bn.apply_boost(rows, colour, bn.boost_factor(mass, teff, binaries))
    limits = (h["a_min"], h["a_max"]) if binaries is not None else None
    fixed_plane = np.isfinite(h["plane"][:, 0])
    if short is not None:
        short_weight = 1.0 - (mdwarf_weight(teff, mdwarf) if mdwarf is not None else 0.0)
        short_weight = np.where(h["kind"] == P_TYPE, 0.0, short_weight)
        if binaries is not None:
            short_weight = short_weight * bn.boost_factor(mass, None, binaries)

    def inner(idx):
        """HM-C systems for hosts ``idx`` (positions in ``host``): (host position per planet, planets, planes)."""
        _, p, planes = hmc.generate_catalog(rows[idx], mass[idx], radius[idx], teff[idx], colour[idx],
                                            age=None if star_age is None else star_age[idx],
                                            seed=int(rng.integers(2**31)), amd_pair_criterion=amd_pair_criterion,
                                            return_planes=True)
        kk = idx[p.pop("star_index")]
        if limits is not None:
            a = period_to_a_au(p["period"], mass[kk])
            ok = (a >= limits[0][kk]) & (a <= limits[1][kk])
            kk, p = kk[ok], {key: v[ok] for key, v in p.items()}
        fix = fixed_plane[idx]
        if fix.any():
            # Systems aligned with a binary orbit: rotate the HM-C system onto the binary's plane.
            planes[fix] = h["plane"][idx[fix]]
            sel = fixed_plane[kk]
            nrm = gp.orbit_normals(h["plane"][kk[sel]], p["incl_invariable"][sel], p["asc_node_invariable"][sel])
            p["incl_sky"][sel] = np.arccos(np.clip(nrm[:, 2], -1.0, 1.0))
        if short is not None:
            kk, p = _short_period(kk, p, idx, planes, h, rows, short_weight[idx], limits, short, rng)
        if binaries is not None and binaries.circumbinary:
            kk, p = _circumbinary(kk, p, idx, h, rows, binaries, rng)
        return kk, p, planes

    k, pl, planes = inner(np.arange(len(host)))
    if giants is not None:
        k, pl = _add_giants(k, pl, planes, inner, rows, mass, h, giants, rng, limits)
    if small is not None:
        edge = sp.edge_table(table, mdwarf, small)[draw]
        if binaries is not None:
            edge = edge.copy()
            edge[:, :, 0] *= rest[:, None]
        ml = {key: v[sys_of_host] for key, v in sp.draw_ml_params(n_sys, rng, small).items()}
        if binaries is not None:
            ml["rate"] = ml["rate"] * bn.boost_factor(mass, None, binaries)  # microlensing: field average
        k, pl = _add_small(k, pl, planes, rows, edge, colour, ml, mass, radius, teff, star_age, small, rng, limits)
    model = np.where(is_m[k], "hmc_m", "hmc").astype(object)
    if giants is not None:
        names = np.array([gp.COMPONENT_NAMES[c] for c in (gp.COLD, gp.HOT, gp.WIDE)], dtype=object)
        model = np.where(pl["clusterid"] == GIANT_CLUSTER_ID, names[pl["giant_component"]], model)
    pl.pop("giant_component", None)
    model = np.where(pl["clusterid"] == SMALL_CLUSTER_ID, "outer_small", model)
    model = np.where(pl["clusterid"] == CB_CLUSTER_ID, "circumbinary", model)
    model = np.where(pl["clusterid"] == SHORT_CLUSTER_ID, "short_period", model)
    small_flags = pl.pop("small_flags", np.zeros(len(k), dtype=np.int32))

    flags = flags_of(h, mdwarf)[k]
    if giants is not None:
        lo, hi = giants.host_mass_range
        is_giant = pl["clusterid"] == GIANT_CLUSTER_ID
        flags |= np.where(is_giant & ((mass[k] < lo) | (mass[k] > hi)), FLAG_GIANT_HOST, 0).astype(np.int32)
    flags |= small_flags
    flags |= np.where(pl["clusterid"] == SHORT_CLUSTER_ID, FLAG_SHORT_PERIOD, 0).astype(np.int32)
    a_au = period_to_a_au(pl["period"], mass[k])
    host_mass = mass[k]
    if evolved is not None and h["evolved"].any():
        a_au, host_mass, keep = _evolve_orbits(k, pl, a_au, h, evolved)
        pl = {key: v[keep] for key, v in pl.items()}
        k, model, flags, a_au, host_mass = k[keep], model[keep], flags[keep], a_au[keep], host_mass[keep]
        pl["period"] = a_au_to_period(a_au, host_mass)
    comp = host[k]
    result = {
        "system": sys_of_host[k],
        "component": comp,
        "host_kind": np.where(h["kind"][k] == P_TYPE, "P", "S").astype(object),
        "host_mass": host_mass,
        "role": np.asarray(components["role"])[comp],
        "model": model,
        "hmc_catalog": cat_id[k],
        "flags": flags,
        "a_au": a_au,
    }
    result.update(pl)
    if evolved is not None:
        result = _add_pulsar_planets(result, systems, components, evolved, rng)
    return result


def _short_period(kk, p, idx, planes, h, rows, weight, limits, short: sh.ShortPeriodModel, rng):
    """Add planets between 0.5 d and each host's HM-C P_min for hosts ``idx`` (``planes``:
    their reference planes, filled in place where missing). Returns the merged, sorted planets."""
    c = sh.draw_short(h["mass"][idx], h["radius"][idx], h["teff"][idx], rows[idx, hmc.I_PMIN], weight, rng, short)
    if limits is not None and len(c["host"]):
        hh = idx[c["host"]]
        a = period_to_a_au(c["period"], h["mass"][hh])
        ok = (a >= limits[0][hh]) & (a <= limits[1][hh])
        c = {key: v[ok] for key, v in c.items()}
    if not len(c["host"]):
        return kk, p
    pos = np.full(len(h["mass"]), -1)
    pos[idx] = np.arange(len(idx))
    local = pos[kk]  # positions in idx of the existing planets' hosts
    keep = sh.place(c, local, p["period"], p["planet_mass"], h["mass"][idx], rows[idx, hmc.I_NHILL])
    c = {key: v[keep] for key, v in c.items()}
    n = len(c["host"])
    if not n:
        return kk, p
    missing = np.isnan(planes[:, 0])
    planes[missing] = gp._random_unit_vectors(missing.sum(), rng)
    o = gp.orient(c["host"], planes, len(idx), rng, gp.GiantModel(incl_sigma_deg=short.incl_sigma_deg))
    r = c["planet_radius"]
    add = {
        "period": c["period"], "planet_mass": c["planet_mass"], "planet_radius": r,
        "clusterid": np.full(n, SHORT_CLUSTER_ID, dtype=p["clusterid"].dtype), "ecc": np.zeros(n),
        "omega": 2 * np.pi * rng.random(n), "incl_mut": o["incl_mut"], "incl_invariable": o["incl_mut"],
        "asc_node_invariable": o["asc_node"], "incl_sky": o["incl_sky"], "mean_anom": 2 * np.pi * rng.random(n),
        "mass_init": c["planet_mass"], "radius_init": r, "lost_envelope": r < short.rocky_radius,
    }
    kk = np.concatenate([kk, idx[c["host"]]])
    p = {key: np.concatenate([p[key], add[key]]) for key in p}
    order = np.lexsort((p["period"], kk))
    return kk[order], {key: v[order] for key, v in p.items()}


def _circumbinary(kk, p, idx, h, rows, binaries: bn.BinaryModel, rng):
    """For the circumbinary hosts among ``idx``: replace HM-C planets of >= CB_SPLIT_RADIUS by
    the empirical component, removing HM-C planets within 10 mutual Hill radii of it.
    Returns the merged (host position, planets), sorted by host and period."""
    cb_hosts = idx[h["kind"][idx] == P_TYPE]
    if not len(cb_hosts):
        return kk, p
    big = (h["kind"][kk] == P_TYPE) & (p["planet_radius"] >= CB_SPLIT_RADIUS)
    kk, p = kk[~big], {key: v[~big] for key, v in p.items()}
    c = bn.draw_circumbinary(h["mass"][cb_hosts], h["a_min"][cb_hosts], h["a_max"][cb_hosts], h["p_bin"][cb_hosts],
                             rng, binaries)
    host = cb_hosts[c["host"]]
    n = len(host)
    if not n:
        return kk, p
    # Make room: HM-C planets too close to the new planet are removed.
    mstar = h["mass"]
    a_in = period_to_a_au(p["period"], mstar[kk])
    clash = np.zeros(len(kk), dtype=bool)
    for j in range(n):
        sel = np.flatnonzero(kk == host[j])
        if not len(sel):
            continue
        a2, m2 = c["a_au"][j], c["planet_mass"][j]
        r_h = ((p["planet_mass"][sel] + m2) * EARTH_MASS_MSUN / (3.0 * mstar[host[j]])) ** (1.0 / 3.0) \
            * 0.5 * (a_in[sel] + a2)
        clash[sel] |= np.abs(a_in[sel] - a2) < rows[host[j], hmc.I_NHILL] * r_h
    kk, p = kk[~clash], {key: v[~clash] for key, v in p.items()}
    o = gp.orient(np.arange(n), h["plane"][host], n, rng, gp.GiantModel(incl_sigma_deg=1.4))
    add = {
        "period": c["period"], "planet_mass": c["planet_mass"], "planet_radius": c["planet_radius"],
        "clusterid": np.full(n, CB_CLUSTER_ID, dtype=p["clusterid"].dtype), "ecc": c["ecc"], "omega": c["omega"],
        "incl_mut": o["incl_mut"], "incl_invariable": o["incl_mut"], "asc_node_invariable": o["asc_node"],
        "incl_sky": o["incl_sky"], "mean_anom": c["mean_anom"], "mass_init": c["planet_mass"],
        "radius_init": c["planet_radius"], "lost_envelope": np.zeros(n, dtype=bool),
    }
    kk = np.concatenate([kk, host])
    p = {key: np.concatenate([p[key], add[key]]) for key in p}
    order = np.lexsort((p["period"], kk))
    return kk[order], {key: v[order] for key, v in p.items()}


def _evolve_orbits(k, pl, a_au, h, evolved: ev.EvolvedModel):
    """Engulfment and adiabatic orbit expansion for planets of evolved hosts.
    Returns (a_au, host mass today, mask of surviving planets)."""
    old = h["evolved"][k]
    r_eng = ev.engulfment_radius(pl["planet_mass"], h["mass_init"][k], h["r_max"][k], h["r_rgb"][k], h["alive"][k],
                                 evolved)
    keep = ~(old & (a_au < r_eng))
    shrink = np.ones(len(k))
    if evolved.tides:
        e = np.flatnonzero(old & keep)
        ke = k[e]
        eng, shrink[e] = ev.tidal_decay(a_au[e], pl["planet_mass"][e], h["mass_init"][ke], h["feh"][ke], h["age"][ke],
                                        h["alive"][ke], h["phase_now"][ke], h["lum_now"][ke], h["teff_now"][ke],
                                        evolved)
        keep[e[eng]] = False
    host_mass = np.where(old, h["mass_now"][k], h["mass"][k])
    return np.where(old, a_au * h["mass"][k] / host_mass * shrink, a_au), host_mass, keep


def _add_pulsar_planets(result, systems, components, evolved: ev.EvolvedModel, rng):
    """Append the planets of neutron stars and re-sort."""
    ns = np.flatnonzero(np.asarray(components["kind"]) == "NS")
    p = ev.draw_pulsar_planets(len(ns), np.asarray(components["mass"], dtype=float)[ns], rng, evolved)
    n = len(p["host"])
    if not n:
        return result
    comp = ns[p["host"]]
    o = gp.orient(p["host"], None, len(ns), rng, gp.GiantModel(incl_sigma_deg=evolved.pulsar_incl_sigma_deg))
    n_comp = np.asarray(systems["n_components"])[np.asarray(components["system"])[comp]]
    add = {
        "system": np.asarray(components["system"])[comp], "component": comp,
        "host_kind": np.full(n, "S", dtype=object), "host_mass": np.asarray(components["mass"], dtype=float)[comp],
        "role": np.asarray(components["role"])[comp], "model": np.full(n, "pulsar", dtype=object),
        "hmc_catalog": np.zeros(n, dtype=result["hmc_catalog"].dtype),
        "flags": (FLAG_PULSAR | np.where(n_comp > 1, FLAG_MULTIPLE, 0)).astype(np.int32), "a_au": p["a_au"],
        "period": p["period"], "planet_mass": p["planet_mass"], "planet_radius": p["planet_radius"],
        "clusterid": np.full(n, PULSAR_CLUSTER_ID, dtype=result["clusterid"].dtype), "ecc": p["ecc"],
        "omega": p["omega"], "incl_mut": o["incl_mut"], "incl_invariable": o["incl_mut"],
        "asc_node_invariable": o["asc_node"], "incl_sky": o["incl_sky"], "mean_anom": p["mean_anom"],
        "mass_init": p["planet_mass"], "radius_init": p["planet_radius"], "lost_envelope": np.zeros(n, dtype=bool),
    }
    out = {key: np.concatenate([result[key], add[key]]) for key in result}
    kind = (out["host_kind"] == "P").astype(np.int8)
    order = np.lexsort((out["period"], kind, out["component"]))
    return {key: v[order] for key, v in out.items()}


def _add_giants(k, pl, planes, inner, rows, mass, h, giants: gp.GiantModel, rng, limits=None):
    """Draw giants, reconcile the inner systems with them and merge; sorted by host position and period."""
    window = (rows[:, hmc.I_PMIN], rows[:, hmc.I_PMAX], rows[:, hmc.I_RMAX])
    g = gp.draw_giants(mass, h["feh"], h["luminosity"], rng, giants, window=window, a_limits=limits)

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
    o = gp.orient(g["host"], planes, len(mass), rng, giants)
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


def _add_small(k, pl, planes, rows, edge, colour, ml, mass, radius, teff, star_age, small: sp.SmallPlanetModel, rng,
               limits=None):
    """Draw small planets beyond P_max around the inner planets and giants; merged and sorted."""
    inner_pl = np.flatnonzero((pl["clusterid"] >= 0) | (pl["clusterid"] == CB_CLUSTER_ID)
                               | (pl["clusterid"] == SHORT_CLUSTER_ID))
    last = inner_pl[np.r_[k[inner_pl][1:] != k[inner_pl][:-1], True]] if len(inner_pl) else inner_pl  # outermost
    giant = np.flatnonzero(pl["clusterid"] == GIANT_CLUSTER_ID)
    occ = np.concatenate([last, giant])
    occupied = {"host": k[occ], "a_au": period_to_a_au(pl["period"][occ], mass[k[occ]]),
                "planet_mass": pl["planet_mass"][occ], "ecc": pl["ecc"][occ],
                "giant": np.r_[np.zeros(len(last), dtype=bool), np.ones(len(giant), dtype=bool)]}
    s = sp.draw_small(mass, radius, teff, rows, edge, colour, ml, rng, small, occupied=occupied, planes=planes,
                      age=star_age, a_limits=limits)
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
