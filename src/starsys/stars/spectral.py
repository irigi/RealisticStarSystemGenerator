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


# Median and robust (MAD) scatter of (Kepler de-reddened bp-rp) - (Mamajek dwarf bp-rp
# at the same Teff) in 200 K bins, from the HFR2020b Kepler stellar table (Berger+2020
# Teff). Bins outside 4200-6400 K are biased by the table's 0.5 < bp-rp < 1.7 cut and
# are not used; the offset tapers to zero at 3800 K and 6800 K, the scatter is held at
# the edge values. Regenerate with ``python -m starsys.validation.planets --colour``.
# Columns: Teff, offset, scatter.
_KEPLER_BPRP = np.array([
    [3800, 0.0, 0.070], [4200, -0.073, 0.070], [4400, -0.104, 0.065], [4600, -0.112, 0.047],
    [4800, -0.084, 0.043], [5000, -0.081, 0.043], [5200, -0.031, 0.048], [5400, -0.033, 0.052],
    [5600, -0.040, 0.065], [5800, -0.063, 0.073], [6000, -0.060, 0.076], [6200, -0.041, 0.082],
    [6400, -0.024, 0.084], [6800, 0.0, 0.084]])


@lru_cache(maxsize=1)
def dwarf_bp_rp_sequence():
    """(Teff, Gaia bp-rp) of the Pecaut & Mamajek dwarf sequence, ascending in Teff."""
    teffs, colours = [], []
    for line in MAMAJEK_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) < 12 or line.startswith("#") or not parts[0].endswith("V"):
            continue
        try:
            teff, colour = float(parts[1]), float(parts[11])
        except ValueError:
            continue
        teffs.append(teff)
        colours.append(colour)
        if parts[0] == "Y4V":
            break
    order = np.argsort(teffs)
    return np.array(teffs)[order], np.array(colours)[order]


@lru_cache(maxsize=1)
def dwarf_mass_radius_sequence():
    """(Teff, mass in Msun, radius in Rsun) of the Pecaut & Mamajek dwarf sequence where the
    table gives a mass (F0-L2, 7220-2060 K), ascending in Teff."""
    rows = []
    for line in MAMAJEK_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) < 31 or line.startswith("#") or not parts[0].endswith("V"):
            continue
        try:
            rows.append((float(parts[1]), float(parts[30]), float(parts[6])))
        except ValueError:
            continue
    t, m, r = np.array(sorted(rows)).T
    return t, m, r


def dwarf_mass_radius(teff):
    """Main-sequence mass and radius at a given Teff (Mamajek; clamped outside 2060-7220 K)."""
    t, m, r = dwarf_mass_radius_sequence()
    teff = np.asarray(teff, dtype=float)
    return np.interp(teff, t, m), np.interp(teff, t, r)


def bp_rp_from_teff(teff, kepler_scale=True):
    """De-reddened Gaia bp-rp of a dwarf from Teff (Mamajek sequence, 2350-10700 K;
    clamped outside). With ``kepler_scale`` the Kepler-table offset above is added, so
    the colour is on the scale the He & Ford models were fitted on."""
    teff = np.asarray(teff, dtype=float)
    t, c = dwarf_bp_rp_sequence()
    colour = np.interp(np.log10(teff), np.log10(t), c)
    if kepler_scale:
        colour = colour + np.interp(teff, _KEPLER_BPRP[:, 0], _KEPLER_BPRP[:, 1])
    return colour


def kepler_bp_rp_scatter(teff):
    """Scatter of Kepler de-reddened bp-rp at fixed Teff (measurement, reddening and
    metallicity combined). Models fitted on measured colours should be marginalised
    over it when only Teff is known."""
    return np.interp(np.asarray(teff, dtype=float), _KEPLER_BPRP[:, 0], _KEPLER_BPRP[:, 2])
