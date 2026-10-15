"""Known planets of a real star, from the local copy of the NASA Exoplanet Archive
(``pscomppars``: one row per planet with the archive's adopted parameters)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import nasa_planets

TRANSIT, RV, IMAGING, OTHER = "transit", "rv", "imaging", "other"
_METHODS = {"Transit": TRANSIT, "Radial Velocity": RV, "Imaging": IMAGING}


@dataclass
class KnownPlanet:
    """A detected planet. Masses in Earth masses, radii in Earth radii; NaN where not measured.

    ``mass`` is a true mass (transit + RV, astrometry, TTVs); ``msini`` a minimum mass (RV only).
    ``radius`` is measured only for transiting planets (the archive's radii of other planets come
    from mass-radius relations and are kept in ``radius_mr``, as is ``mass_mr`` for transiting
    planets without a mass)."""

    name: str
    method: str  # discovery method: "transit", "rv", "imaging" or "other"
    period: float = np.nan  # days
    period_err: float = np.nan
    a_au: float = np.nan
    radius: float = np.nan
    radius_err: float = np.nan
    mass: float = np.nan
    mass_err: float = np.nan
    msini: float = np.nan
    msini_err: float = np.nan
    ecc: float = np.nan
    incl_deg: float = np.nan  # sky inclination
    transits: bool = False
    rv_detected: bool = False
    circumbinary: bool = False
    controversial: bool = False
    radius_mr: float = np.nan
    mass_mr: float = np.nan


def _f(row, col) -> float:
    v = row.get(col)
    return float(v) if pd.notna(v) else np.nan


def _err(row, col) -> float:
    e = [abs(_f(row, col + "err1")), abs(_f(row, col + "err2"))]
    e = [x for x in e if np.isfinite(x)]
    return float(np.mean(e)) if e else np.nan


def from_row(row: pd.Series) -> KnownPlanet:
    prov = str(row.get("pl_bmassprov") or "")
    transits = bool(row.get("tran_flag") == 1)
    k = KnownPlanet(
        name=str(row["pl_name"]), method=_METHODS.get(row.get("discoverymethod"), OTHER),
        period=_f(row, "pl_orbper"), period_err=_err(row, "pl_orbper"), a_au=_f(row, "pl_orbsmax"),
        ecc=_f(row, "pl_orbeccen"), incl_deg=_f(row, "pl_orbincl"), transits=transits,
        rv_detected=bool(row.get("rv_flag") == 1), circumbinary=bool(row.get("cb_flag") == 1),
        controversial=bool(row.get("pl_controv_flag") == 1),
    )
    if transits and pd.notna(row.get("pl_rade")) and not row.get("pl_radelim"):
        k.radius, k.radius_err = _f(row, "pl_rade"), _err(row, "pl_rade")
    else:
        k.radius_mr = _f(row, "pl_rade")
    lim = row.get("pl_bmasselim")
    if pd.notna(row.get("pl_bmasse")) and not (pd.notna(lim) and lim != 0):
        if prov in ("Mass", "Msin(i)/sin(i)"):
            k.mass, k.mass_err = _f(row, "pl_bmasse"), _err(row, "pl_bmasse")
        elif prov == "Msini":
            k.msini, k.msini_err = _f(row, "pl_bmasse"), _err(row, "pl_bmasse")
        else:
            k.mass_mr = _f(row, "pl_bmasse")
    if np.isnan(k.msini) and pd.notna(row.get("pl_msinie")):
        k.msini, k.msini_err = _f(row, "pl_msinie"), _err(row, "pl_msinie")
    return k


def known_planets(hostname: str, include_controversial: bool = False) -> list[KnownPlanet]:
    """Known planets of the archive host ``hostname``, by period (or semi-major axis)."""
    if not hostname:
        return []
    d = nasa_planets()
    rows = d[d["hostname"] == hostname]
    out = [from_row(r) for _, r in rows.iterrows()]
    if not include_controversial:
        out = [k for k in out if not k.controversial]
    return sorted(out, key=lambda k: (k.period if np.isfinite(k.period) else 1e12, k.a_au))
