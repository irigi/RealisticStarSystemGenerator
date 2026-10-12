"""MIST v1.2 isochrones (Choi et al. 2016; Dotter 2016), v/vcrit = 0.4.

Valid for initial masses >= 0.1 Msun, log10(age/yr) 5.0-10.3 and
[Fe/H] -4.0 to +0.5. The text isochrones are parsed once into
``data/stellar/mist_grid.npz``; queries interpolate bilinearly in
([Fe/H], log age) between isochrones and linearly in initial mass along each
isochrone. Where the four bracketing isochrones disagree on the evolutionary
phase (fast phases), the nearest isochrone is used instead of blending.

MIST phase codes: -1 pre-MS, 0 MS, 2 RGB, 3 core-He burning, 4 early AGB,
5 TP-AGB, 6 post-AGB/WD, 9 Wolf-Rayet.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit

from ..paths import CATALOGS_DIR, DATA_DIR

MIST_DIR = CATALOGS_DIR / "stellar_models" / "MIST" / "MIST_v1.2_vvcrit0.4_basic_isos"
GRID_FILE = DATA_DIR / "stellar" / "mist_grid.npz"

# Columns kept from the basic isochrones (0-based indices in the file).
_COLS = {"log_age": 1, "initial_mass": 2, "star_mass": 3, "log_L": 7, "log_Teff": 10, "log_R": 11,
         "log_g": 12, "phase": 24}
QUANTITIES = ("star_mass", "log_L", "log_Teff", "log_R", "log_g")


def _feh_of(path: Path) -> float:
    m = re.search(r"feh_([mp])(\d+\.\d+)", path.name)
    return (-1 if m[1] == "m" else 1) * float(m[2])


def build_grid(mist_dir: Path = MIST_DIR, out: Path = GRID_FILE) -> None:
    files = sorted(mist_dir.glob("*.iso"), key=_feh_of)
    fehs, blocks = [], []
    ages = None
    for f in files:
        data = np.loadtxt(f, comments="#", usecols=list(_COLS.values()))
        file_ages = np.unique(data[:, 0])
        if ages is None:
            ages = file_ages
        elif not np.allclose(ages, file_ages):
            raise ValueError(f"age grid differs in {f.name}")
        fehs.append(_feh_of(f))
        blocks.append(data)
    n_f, n_a = len(fehs), len(ages)
    offsets = np.zeros((n_f, n_a), dtype=np.int64)
    counts = np.zeros((n_f, n_a), dtype=np.int64)
    flat = []
    pos = 0
    for i, data in enumerate(blocks):
        for j, age in enumerate(ages):
            rows = data[data[:, 0] == age]
            rows = rows[np.argsort(rows[:, 1], kind="stable")]
            offsets[i, j], counts[i, j] = pos, len(rows)
            flat.append(rows[:, 1:])
            pos += len(rows)
    flat = np.vstack(flat)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fehs=np.array(fehs), log_ages=ages, offsets=offsets, counts=counts, table=flat)


@lru_cache(maxsize=1)
def load_grid():
    if not GRID_FILE.exists():
        build_grid()
    g = np.load(GRID_FILE)
    # table columns: initial_mass, star_mass, log_L, log_Teff, log_R, log_g, phase
    return g["fehs"], g["log_ages"], g["offsets"], g["counts"], np.ascontiguousarray(g["table"])


@njit(cache=True)
def _bracket(grid, x):
    n = len(grid)
    if x <= grid[0]:
        return 0, 0, 0.0
    if x >= grid[n - 1]:
        return n - 1, n - 1, 0.0
    j = np.searchsorted(grid, x) - 1
    return j, j + 1, (x - grid[j]) / (grid[j + 1] - grid[j])


@njit(cache=True)
def _along_isochrone(table, off, cnt, mi, out):
    """Interpolate one isochrone at initial mass mi. Returns phase, or -99 if mi is beyond the
    isochrone (star has died) and -98 if below its minimum mass."""
    m = table[off:off + cnt, 0]
    if cnt == 0 or mi > m[cnt - 1]:
        return -99.0
    if mi < m[0]:
        return -98.0
    k = np.searchsorted(m, mi) - 1
    if k < 0:
        k = 0
    if k >= cnt - 1:
        k = cnt - 2
    w = (mi - m[k]) / (m[k + 1] - m[k]) if m[k + 1] > m[k] else 0.0
    for q in range(5):
        out[q] = (1 - w) * table[off + k, 1 + q] + w * table[off + k + 1, 1 + q]
    return table[off + k, 6] if w < 0.5 else table[off + k + 1, 6]


@njit(cache=True)
def interpolate(fehs, log_ages, offsets, counts, table, mi, log_age, feh):
    """Vectorised MIST lookup. Returns (values[n, 5], phase[n]) with phase -99 for dead stars."""
    n = len(mi)
    values = np.full((n, 5), np.nan)
    phase = np.full(n, np.nan)
    tmp = np.empty(5)
    for s in range(n):
        f0, f1, wf = _bracket(fehs, feh[s])
        a0, a1, wa = _bracket(log_ages, log_age[s])
        corners = ((f0, a0, (1 - wf) * (1 - wa)), (f0, a1, (1 - wf) * wa),
                   (f1, a0, wf * (1 - wa)), (f1, a1, wf * wa))
        acc = np.zeros(5)
        best_w, best_phase = -1.0, np.nan
        best_vals = np.full(5, np.nan)
        phases_agree = True
        first_phase = np.nan
        for c in range(4):
            fi, ai, w = corners[c]
            ph = _along_isochrone(table, offsets[fi, ai], counts[fi, ai], mi[s], tmp)
            if c == 0:
                first_phase = ph
            elif ph != first_phase:
                phases_agree = False
            if ph > -50:
                acc += w * tmp
            if w > best_w:
                best_w, best_phase = w, ph
                best_vals[:] = tmp
        if phases_agree and first_phase > -50:
            values[s] = acc
            phase[s] = first_phase
        else:
            phase[s] = best_phase
            if best_phase > -50:
                values[s] = best_vals
    return values, phase


@njit(cache=True)
def lifetime_log_age(fehs, log_ages, offsets, counts, table, mi, feh):
    """log10(age/yr) at which a star of initial mass mi leaves the isochrones (nearest [Fe/H])."""
    n = len(mi)
    out = np.full(n, np.inf)
    nf = len(fehs)
    for s in range(n):
        fi = np.argmin(np.abs(fehs - feh[s]))
        prev_max = np.inf
        for j in range(len(log_ages)):
            cnt = counts[fi, j]
            mmax = table[offsets[fi, j] + cnt - 1, 0] if cnt > 0 else 0.0
            if mi[s] > mmax:
                if j == 0 or not np.isfinite(prev_max):
                    out[s] = log_ages[j]
                else:
                    # interpolate in log mass between the two isochrones' maximum masses
                    w = (np.log(prev_max) - np.log(mi[s])) / (np.log(prev_max) - np.log(mmax))
                    out[s] = log_ages[j - 1] + w * (log_ages[j] - log_ages[j - 1])
                break
            prev_max = mmax
    return out


def evolve(initial_mass, log_age, feh):
    """Present-day properties of stars with initial mass >= 0.1 Msun.

    Returns a dict of arrays (QUANTITIES + ``phase``); phase == -99 marks stars that
    have already died (remnants are handled in :mod:`starsys.stars.remnants`).
    """
    grid = load_grid()
    args = [np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64) for x in (initial_mass, log_age, feh)]
    values, phase = interpolate(*grid, *args)
    out = {q: values[:, i] for i, q in enumerate(QUANTITIES)}
    out["phase"] = phase
    return out


def lifetime(initial_mass, feh):
    grid = load_grid()
    args = [np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64) for x in (initial_mass, feh)]
    return lifetime_log_age(*grid, *args)
