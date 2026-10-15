"""Export a :class:`starsys.generate.Systems` bundle.

- **JSON** (:func:`system_dict`, :func:`write_json`): one nested record per system, stars with
  their planets and the planets with their moons, readable labels and explicit units
  (``units`` at the top). The schema is ``starsys.system/1``:

  ``{"schema", "units", "meta", "systems": [{"id", "age_gyr", "feh", "stars": [{"name",
  "role", "kind", "spectral_type", ..., "orbit": {...} | null, "habitable_zone_au": {...},
  "planets": [...]}], "circumbinary_planets": [...]}]}``

  Every planet carries ``name``, ``class``, orbit, size, ``insolation``,
  ``t_eq_k`` (equilibrium temperature, no greenhouse: see A9 in :mod:`starsys.describe`),
  ``t_surface_illustrative_k`` and ``climate``, spin, flags and ``moons``. Angles are in
  degrees. Non-finite numbers become ``null``.
- **Tables** (:func:`write_tables`): the four tables as CSV or Parquet, one row per system,
  component, planet and moon, with the names and labels added as columns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import derived as dv
from . import describe as ds
from .flags import FLAG_NAMES

SCHEMA = "starsys.system/1"
FLAG_KNOWN, FLAG_KNOWN_INSERTED = 32768, 65536  # starsys.real.condition (not imported: optional deps)
ALL_FLAGS = {**FLAG_NAMES, FLAG_KNOWN: "known", FLAG_KNOWN_INSERTED: "known_inserted"}

UNITS = {
    "age_gyr": "Gyr", "feh": "dex", "mass": "Msun (stars), Earth masses (planets, moons)",
    "radius": "Rsun (stars), Earth radii (planets, moons)", "teff": "K", "luminosity": "Lsun",
    "a_au": "au", "a_km": "km", "a_rp": "planet radii", "period_days": "days", "angles": "degrees (JSON), radians (tables)",
    "insolation": "Earth = 1 (1361 W/m^2)",
    "t_eq_k": "K; equilibrium temperature, Bond albedo 0.3, full redistribution, no greenhouse (Earth 255 K)",
    "t_eq0_k": "K; as t_eq_k with albedo 0",
    "t_surface_illustrative_k": "K; rough surface temperature of rocky planets, Earth-like atmosphere assumed "
                                "(starsys.describe.surface_temperature); null without a surface",
    "rotation_h": "hours (sidereal)", "t_lock_gyr": "Gyr", "t_runaway_myr": "Myr",
    "tidal_heating_w": "W", "heat_flux_w_m2": "W/m^2",
}


def _num(x, digits: int = 6):
    """JSON-safe float rounded to ``digits`` significant figures (None for NaN/inf)."""
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def _deg(x):
    return _num(np.degrees(float(x)), 5)


def _flags(f) -> list[str]:
    return [name for bit, name in ALL_FLAGS.items() if int(f) & bit]


def _star(bundle, r, name) -> dict:
    c = bundle.components
    role = int(c["role"][r])
    d = {"name": name, "role": role, "kind": ds.STAR_KINDS.get(str(c["kind"][r]), str(c["kind"][r])),
         "spectral_type": str(c["spectral_type"][r]) if "spectral_type" in c else "",
         "mass": _num(c["mass"][r]), "initial_mass": _num(c["initial_mass"][r]), "radius": _num(c["radius"][r]),
         "teff": _num(c["teff"][r], 5), "luminosity": _num(c["luminosity"][r]), "log_g": _num(c["log_g"][r], 4),
         "age_gyr": _num(c["age_gyr"][r], 4), "feh": _num(c["feh"][r], 3),
         "mist_phase": _num(c["phase"][r]) if np.isfinite(c["phase"][r]) else None,
         "cooling_age_gyr": _num(c["cooling_age_gyr"][r], 4)}
    if bundle.target is not None:
        d["is_target"] = bool(np.any(np.asarray(bundle.target) == r))
    if role > 0:
        d["orbit"] = {"around": [ds.star_name(bundle, k) for k in range(role)], "a_au": _num(c["a_au"][r]),
                      "ecc": _num(c["ecc"][r], 4), "period_days": _num(c["period_days"][r]),
                      "incl_sky_deg": _deg(c["incl_sky"][r]), "node_sky_deg": _deg(c["node"][r]),
                      "omega_deg": _deg(c["omega"][r])}
    else:
        d["orbit"] = None
    hz = ds.hz_distances(c["teff"][r], c["luminosity"][r]) if str(c["kind"][r]) in ("star", "BD", "WD") else {}
    d["habitable_zone_au"] = ({k: _num(v, 4) for k, v in hz.items()}
                              if hz and math.isfinite(hz["runaway"]) else None)
    return d


def _moon(bundle, r, name) -> dict:
    m = bundle.moons
    return {"name": name, "kind": str(m["kind"][r]), "composition": str(m["composition"][r]),
            "mass": _num(m["mass"][r]), "radius": _num(m["radius"][r]), "a_km": _num(m["a_km"][r]),
            "a_rp": _num(m["a_rp"][r], 4), "period_days": _num(m["period"][r]), "ecc": _num(m["ecc"][r], 4),
            "incl_deg": _deg(m["incl"][r]), "tidal_heating_w": _num(m["tidal_heating_w"][r], 4),
            "heat_flux_w_m2": _num(m["heat_flux_w_m2"][r], 4), "flags": _flags(m["flags"][r])}


def _planet(bundle, r, name, labels, moons) -> dict:
    p = bundle.planets
    cls, t_surf, climate, kind = labels
    g = (lambda k, digits=6: _num(p[k][r], digits) if k in p else None)
    d = {"name": name, "class": str(cls[r]),
         "type": ("rocky", "envelope", "giant")[int(kind[r])],
         "known": bool(int(p["flags"][r]) & FLAG_KNOWN),
         "a_au": g("a_au"), "period_days": g("period"), "ecc": g("ecc", 4),
         "omega_deg": _deg(p["omega"][r]), "mean_anomaly_deg": _deg(p["mean_anom"][r]),
         "incl_invariable_deg": _deg(p["incl_invariable"][r]),
         "node_invariable_deg": _deg(p["asc_node_invariable"][r]),
         "incl_sky_deg": _deg(p["incl_sky"][r]), "node_sky_deg": _deg(p["node_sky"][r]),
         "mass": g("planet_mass"), "radius": g("planet_radius", 4),
         "lost_envelope": bool(p["lost_envelope"][r]),
         "insolation": g("insolation"), "t_eq_k": g("t_eq", 4), "t_eq0_k": g("t_eq0", 4),
         "t_surface_illustrative_k": _num(t_surf[r], 3), "climate": ds.CLIMATE_NAMES[int(climate[r])],
         "habitable_zone": ds.HZ_NAMES.get(int(p["hz"][r]), "") if "hz" in p else None,
         "spin": ds.SPIN_NAMES.get(int(p["spin"][r]), "") if "spin" in p else None,
         "rotation_h": g("rotation_h", 4), "obliquity_deg": g("obliquity_deg", 4),
         "t_lock_gyr": g("t_lock_gyr", 4), "t_runaway_myr": g("t_runaway_myr", 4),
         "model": str(p["model"][r]), "flags": _flags(p["flags"][r]), "moons": moons}
    return d


def planet_labels(planets) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class, illustrative surface temperature, climate code and type (0 rocky, 1 envelope,
    2 giant) of every planet."""
    if not len(planets["a_au"]):
        e = np.zeros(0)
        return e.astype(object), e, e.astype(np.int8), e.astype(np.int8)
    t, climate = ds.surface_temperature(planets)
    kind = dv.planet_kind(planets["planet_mass"], planets["planet_radius"], planets["lost_envelope"])
    return ds.planet_class(planets), t, climate, kind


def system_dict(bundle, i: int, labels=None) -> dict:
    """Nested, JSON-ready record of system ``i`` (see the module docstring)."""
    s, c, p = bundle.systems, bundle.components, bundle.planets
    labels = labels if labels is not None else planet_labels(p)
    comp, pl, mo = bundle.rows(i)
    cn, pn, mn = ds.names(bundle, i)
    moons_of: dict[int, list] = {}
    for r in sorted(mo, key=lambda r: bundle.moons["a_km"][r]):
        moons_of.setdefault(int(bundle.moons["planet"][r]), []).append(_moon(bundle, int(r), mn[int(r)]))
    stars = []
    for r in comp:
        d = _star(bundle, int(r), cn[int(r)])
        d["planets"] = [_planet(bundle, int(q), pn[int(q)], labels, moons_of.get(int(q), []))
                        for q in pl if int(p["component"][q]) == r and str(p["host_kind"][q]) == "S"]
        stars.append(d)
    cb = [_planet(bundle, int(q), pn[int(q)], labels, moons_of.get(int(q), []))
          for q in pl if str(p["host_kind"][q]) == "P"]
    return {"id": int(i), "system_id": ds.system_label(bundle, i), "age_gyr": _num(s["age_gyr"][i], 4), "feh": _num(s["feh"][i], 3),
            "n_stars": len(comp), "n_planets": len(pl), "n_moons": len(mo), "stars": stars,
            "circumbinary_planets": cb,
            "circumbinary_host": [cn[int(comp[0])], cn[int(comp[1])]] if cb else None}


def to_json_obj(bundle, which=None) -> dict:
    which = range(len(bundle)) if which is None else which
    labels = planet_labels(bundle.planets)
    return {"schema": SCHEMA, "units": UNITS, "meta": bundle.meta,
            "systems": [system_dict(bundle, int(i), labels) for i in which]}


def write_json(bundle, path, which=None, indent: int | None = 1) -> Path:
    """Write systems ``which`` (default: all) as JSON to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_obj(bundle, which), indent=indent, ensure_ascii=False))
    return path


def tables(bundle) -> dict:
    """The four tables as pandas DataFrames with names and labels added (angles stay radians,
    as in the generator's tables)."""
    import pandas as pd

    def frame(d):
        return pd.DataFrame({k: np.asarray(v) for k, v in d.items()
                             if np.asarray(v).ndim == 1})

    sysdf, comp, pl = frame(bundle.systems), frame(bundle.components), frame(bundle.planets)
    moons = frame(bundle.moons) if bundle.moons else pd.DataFrame()
    sysdf.insert(0, "system", np.arange(len(sysdf)))
    comp.insert(1, "name", [ds.star_name(bundle, r) for r in comp["role"]])
    if len(pl):
        pnames = np.empty(len(pl), dtype=object)
        mnames = np.empty(len(moons), dtype=object)
        for i in np.unique(pl["system"]):
            _, pn, mn = ds.names(bundle, int(i))
            for r, n in pn.items():
                pnames[r] = n
            for r, n in mn.items():
                mnames[r] = n
        cls, t_surf, climate, kind = planet_labels(bundle.planets)
        pl.insert(2, "name", pnames)
        pl["class"] = cls
        pl["type"] = np.array(["rocky", "envelope", "giant"])[kind]
        pl["t_surface_illustrative"] = t_surf
        pl["climate"] = [ds.CLIMATE_NAMES[int(x)] for x in climate]
        pl["flag_names"] = [",".join(_flags(f)) for f in pl["flags"]]
        if len(moons):
            moons.insert(1, "name", mnames)
    return {"systems": sysdf, "components": comp, "planets": pl, "moons": moons}


def write_tables(bundle, directory, fmt: str = "csv") -> list[Path]:
    """Write ``systems``, ``components``, ``planets`` and ``moons`` as CSV or Parquet files."""
    if fmt not in ("csv", "parquet"):
        raise ValueError("fmt must be 'csv' or 'parquet'")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for name, df in tables(bundle).items():
        path = directory / f"{name}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        out.append(path)
    (directory / "units.json").write_text(json.dumps(UNITS, indent=1))
    out.append(directory / "units.json")
    return out


__all__ = ["SCHEMA", "UNITS", "system_dict", "to_json_obj", "write_json", "tables", "write_tables", "planet_labels"]
