"""Stellar remnants: white dwarfs, neutron stars, black holes.

- Initial-final mass relation: Cummings et al. (2018), MIST-based fit
  (their eqs. for 0.83-2.85, 2.85-3.60 and 3.60-7.20 Msun progenitors),
  extrapolated linearly outside that range and capped at 1.38 Msun.
- White-dwarf cooling: Montreal cooling tables (Bedard et al. 2020), pure-H (DA)
  and pure-He (DB) atmospheres, interpolated in mass and log cooling age.
- Progenitors above ~8 Msun become neutron stars (1.35 Msun) and above ~20 Msun
  black holes. These are crude placeholders: no such objects lie within 20 pc,
  so they matter little for a solar-neighbourhood generator.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..paths import CATALOGS_DIR

WD_DIR = CATALOGS_DIR / "stellar_models" / "WD_Montreal"
WD_MAX_PROGENITOR = 8.0
NS_MAX_PROGENITOR = 20.0
NS_MASS = 1.35

# Fraction of white dwarfs with hydrogen atmospheres (DA vs non-DA, roughly 3:1 in the
# local sample; e.g. Kilic et al. 2020, McCleery et al. 2020).
DA_FRACTION = 0.75


def ifmr_cummings2018_mist(mi):
    mi = np.asarray(mi, dtype=float)
    mf = np.where(mi < 2.85, 0.080 * mi + 0.489,
                  np.where(mi < 3.60, 0.187 * mi + 0.184, 0.107 * mi + 0.471))
    return np.clip(mf, 0.0, 1.38)


@lru_cache(maxsize=1)
def wd_tables():
    """{atmosphere: (masses, [array(log_age, log_Teff, log_L, log_g)])}."""
    out = {"H": ([], []), "He": ([], [])}
    for f in sorted(WD_DIR.glob("Table_Mass_*")):
        mass = float(f.name.split("_")[-1])
        blocks, current = [], []
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                if current:
                    blocks.append(np.array(current))
                    current = []
                continue
            if len(vals) > 10:
                current.append(vals)
        if current:
            blocks.append(np.array(current))
        for atm, block in zip(("H", "He"), blocks[:2]):
            teff, logg, mbol, age = block[:, 0], block[:, 1], block[:, 2], block[:, -1]
            log_l = (4.74 - mbol) / 2.5
            order = np.argsort(age)
            good = age[order] > 0
            arr = np.column_stack([np.log10(age[order][good]), np.log10(teff[order][good]),
                                   log_l[order][good], logg[order][good]])
            out[atm][0].append(mass)
            out[atm][1].append(arr)
    return out


def cool_white_dwarf(mass, cooling_age_yr, hydrogen):
    """log_Teff, log_L, log_g, log_R for white dwarfs of given mass and cooling age."""
    mass = np.atleast_1d(np.asarray(mass, dtype=float))
    log_t = np.log10(np.maximum(np.atleast_1d(cooling_age_yr), 1e5))
    hydrogen = np.broadcast_to(np.asarray(hydrogen, dtype=bool), mass.shape)
    tabs = wd_tables()
    res = np.full((len(mass), 3), np.nan)
    for atm, sel in (("H", hydrogen), ("He", ~hydrogen)):
        masses, tracks = tabs[atm]
        masses = np.array(masses)
        idx = np.flatnonzero(sel)
        m = np.clip(mass[idx], masses[0], masses[-1])
        k = np.clip(np.searchsorted(masses, m) - 1, 0, len(masses) - 2)
        w = ((m - masses[k]) / (masses[k + 1] - masses[k]))[:, None]
        for kk in np.unique(k):
            g = k == kk
            s = idx[g]
            vals = []
            for t in (tracks[kk], tracks[kk + 1]):
                x = np.clip(log_t[s], t[0, 0], t[-1, 0])
                vals.append(np.column_stack([np.interp(x, t[:, 0], t[:, c]) for c in (1, 2, 3)]))
            res[s] = (1 - w[g]) * vals[0] + w[g] * vals[1]
    log_teff, log_l, log_g = res.T
    # R from L = 4 pi R^2 sigma T^4, in solar units (Teff_sun = 5772 K).
    log_r = 0.5 * log_l - 2.0 * (log_teff - np.log10(5772.0))
    return {"log_Teff": log_teff, "log_L": log_l, "log_g": log_g, "log_R": log_r}


def remnant_type(mi):
    mi = np.asarray(mi, dtype=float)
    return np.where(mi < WD_MAX_PROGENITOR, "WD", np.where(mi < NS_MAX_PROGENITOR, "NS", "BH"))


def remnant_mass(mi):
    mi = np.asarray(mi, dtype=float)
    # Black-hole masses: crude 0.25 x progenitor mass, a placeholder (see module docstring).
    return np.where(mi < WD_MAX_PROGENITOR, ifmr_cummings2018_mist(mi),
                    np.where(mi < NS_MAX_PROGENITOR, NS_MASS, 0.25 * mi))
