"""Evolution of brown dwarfs and the lowest-mass stars (below the MIST grid).

- ATMO 2020 chemical-equilibrium tracks (Phillips et al. 2020): 0.0005-0.075 Msun,
  the best available models for cool L/T/Y dwarfs.
- BHAC15 tracks (Baraffe et al. 2015): used for 0.075-0.1 Msun (solar metallicity).

Both are solar-metallicity grids; [Fe/H] is ignored here (small effect for
brown-dwarf cooling compared with mass and age).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..paths import CATALOGS_DIR

ATMO_DIR = CATALOGS_DIR / "stellar_models" / "ATMO2020" / "evolutionary_tracks" / "ATMO_CEQ" / "MKO_WISE_IRAC"
BHAC15_FILE = CATALOGS_DIR / "stellar_models" / "BHAC15" / "BHAC15_tracks+structure"

ATMO_MAX_MASS = 0.075
SUBSTELLAR_MAX_MASS = 0.1


class _TrackSet:
    """Tracks on a mass grid; each track is (log_age, log_Teff, log_L, log_R)."""

    def __init__(self, masses, tracks):
        order = np.argsort(masses)
        self.log_m = np.log10(np.asarray(masses)[order])
        self.tracks = [tracks[i] for i in order]

    def _one(self, k, log_age):
        t = self.tracks[k]
        x = np.clip(log_age, t[:, 0][0], t[:, 0][-1])
        return np.array([np.interp(x, t[:, 0], t[:, c]) for c in (1, 2, 3)])

    def __call__(self, mass, log_age):
        lm = np.log10(mass)
        out = np.empty((len(lm), 3))
        for s, (m, a) in enumerate(zip(lm, log_age)):
            k = int(np.clip(np.searchsorted(self.log_m, m) - 1, 0, len(self.log_m) - 2))
            w = np.clip((m - self.log_m[k]) / (self.log_m[k + 1] - self.log_m[k]), 0.0, 1.0)
            out[s] = (1 - w) * self._one(k, a) + w * self._one(k + 1, a)
        return out  # log_Teff, log_L, log_R


@lru_cache(maxsize=1)
def atmo_tracks() -> _TrackSet:
    masses, tracks = [], []
    for f in sorted(ATMO_DIR.glob("*_ATMO_CEQ_vega.txt")):
        d = np.loadtxt(f, comments="#", usecols=(0, 1, 2, 3, 4))
        masses.append(d[0, 0])
        # columns: mass, age (Gyr), Teff, log L, R (Rsun)
        tracks.append(np.column_stack([np.log10(d[:, 1] * 1e9), np.log10(d[:, 2]), d[:, 3], np.log10(d[:, 4])]))
    return _TrackSet(masses, tracks)


@lru_cache(maxsize=1)
def bhac15_tracks() -> _TrackSet:
    rows = []
    for line in BHAC15_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 13:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    d = np.array(rows)  # M, log t, Teff, log L, log g, R, ...
    masses, tracks = [], []
    for m in np.unique(d[:, 0]):
        t = d[d[:, 0] == m]
        masses.append(m)
        tracks.append(np.column_stack([t[:, 1], np.log10(t[:, 2]), t[:, 3], np.log10(t[:, 5])]))
    return _TrackSet(masses, tracks)


def evolve_substellar(mass, log_age):
    """log_Teff, log_L, log_R for objects with mass < 0.1 Msun."""
    mass = np.atleast_1d(np.asarray(mass, dtype=float))
    log_age = np.atleast_1d(np.asarray(log_age, dtype=float))
    out = np.full((len(mass), 3), np.nan)
    atmo = mass < ATMO_MAX_MASS
    if atmo.any():
        out[atmo] = atmo_tracks()(mass[atmo], log_age[atmo])
    bh = ~atmo
    if bh.any():
        out[bh] = bhac15_tracks()(mass[bh], log_age[bh])
    return {"log_Teff": out[:, 0], "log_L": out[:, 1], "log_R": out[:, 2]}
