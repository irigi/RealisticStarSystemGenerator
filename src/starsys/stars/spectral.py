"""Spectral classification from physical parameters.

Temperature class uses the dwarf Teff sequence of Pecaut & Mamajek (2013,
updated table "EEM_dwarf_UBVIJHK_colors_Teff.txt", O3 to Y4). Giants and
subgiants get the same temperature class (adequate to about one subclass) with
a luminosity class from log g / luminosity. White dwarfs use the standard
D-type notation with temperature index 50400/Teff.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..paths import CATALOGS_DIR

MAMAJEK_FILE = CATALOGS_DIR / "stellar_models" / "Mamajek" / "EEM_dwarf_UBVIJHK_colors_Teff.txt"


@lru_cache(maxsize=1)
def dwarf_sequence():
    types, teffs = [], []
    for line in MAMAJEK_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or line.startswith("#") or not parts[0].endswith("V"):
            continue
        try:
            teff = float(parts[1])
        except ValueError:
            continue
        if types and parts[0] == types[-1]:
            continue
        types.append(parts[0][:-1])  # drop the trailing luminosity class "V"
        teffs.append(teff)
        if parts[0] == "Y4V":
            break
    return np.array(types), np.array(teffs)


def temperature_class(teff):
    types, teffs = dwarf_sequence()
    teff = np.atleast_1d(np.asarray(teff, dtype=float))
    # Table is ordered hot -> cool; pick the nearest entry in log Teff.
    idx = np.abs(np.log10(teff)[:, None] - np.log10(teffs)[None, :]).argmin(axis=1)
    return types[idx]


def luminosity_class(kind, phase, log_g, log_l):
    """V dwarf, IV subgiant, III giant, II bright giant, I supergiant."""
    kind = np.asarray(kind)
    out = np.full(kind.shape, "V", dtype=object)
    star = kind == "star"
    evolved = star & np.isin(phase, [2, 3, 4, 5, 6, 9])
    out[evolved & (log_g >= 3.5)] = "IV"
    out[evolved & (log_g < 3.5)] = "III"
    out[evolved & (log_l >= 3.5)] = "II"
    out[evolved & (log_l >= 4.5)] = "I"
    return out


def spectral_type(kind, teff, phase, log_g, log_l, hydrogen_atmosphere=None):
    kind = np.atleast_1d(np.asarray(kind))
    teff = np.atleast_1d(np.asarray(teff, dtype=float))
    out = np.empty(kind.shape, dtype=object)
    tc = temperature_class(teff)
    lc = luminosity_class(kind, np.atleast_1d(phase), np.atleast_1d(log_g), np.atleast_1d(log_l))
    for i, k in enumerate(kind):
        if k == "WD":
            letter = "DA" if (hydrogen_atmosphere is None or hydrogen_atmosphere[i]) else "DB"
            if teff[i] < 12000 and letter == "DB":
                letter = "DC"  # He-atmosphere WDs show no lines below ~11-12 kK
            out[i] = f"{letter}{min(50400.0 / teff[i], 9.9):.1f}"
        elif k == "NS":
            out[i] = "neutron star"
        elif k == "BH":
            out[i] = "black hole"
        else:
            out[i] = f"{tc[i]}{lc[i]}"
    return out
