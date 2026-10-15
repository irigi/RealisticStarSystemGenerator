import math

import numpy as np
import pytest

from starsys import binaries as bn
from starsys import planets


def test_cycle_table_reaches_the_analytic_maximum():
    table = bn.kozai_table()
    j = 0  # e0 = 0.01
    for i, c in enumerate(bn.KL_COS_GRID):
        e_max = math.sqrt(max(0.0, 1.0 - 5.0 / 3.0 * c * c))
        if e_max > 0.2:  # near 39.2 deg the finite e0 = 0.01 dominates
            assert table[i, j, -1] == pytest.approx(e_max, abs=0.02)
        assert np.all(np.diff(table[i, j]) >= 0)


def test_draw_conserves_the_kozai_constant():
    rng = np.random.default_rng(1)
    cos0 = np.full(5000, math.cos(math.radians(70.0)))
    e0 = np.full(5000, 0.05)
    e, c = bn.kozai_draw(cos0, e0, rng)
    assert e.max() <= math.sqrt(1 - 5 / 3 * cos0[0] ** 2) + 0.02 and e.max() > 0.7
    assert np.allclose(np.sqrt(1 - e**2) * c, np.sqrt(1 - 0.05**2) * cos0, atol=1e-9)


def test_timescales():
    # A Jupiter at 5 au around a Sun with a 0.5 Msun companion at 500 au: t_KL ~ 1e7 yr.
    t = bn.kozai_timescale(np.array([5.0]), np.array([1.0]), np.array([318.0]), np.array([500.0]), np.array([0.5]),
                           np.array([0.0]))[0]
    assert 3e6 < t < 3e7
    # Mercury's GR precession is 43" per century: one radian in ~4.8e5 centuries.
    t_gr = bn.gr_precession_time(np.array([0.387]), np.array([0.206]), np.array([1.0]))[0]
    assert t_gr == pytest.approx(206265 / 43 * 100, rel=0.02)
    # Earth forced by Jupiter: ~8 arcsec / yr scale (Laplace-Lagrange, Murray & Dermott).
    nu = bn.secular_rate(np.array([1.0]), np.array([5.2]), np.array([317.8]), np.array([1.0]))[0]
    assert 3 < np.degrees(nu) * 3600 < 15


def _binary(n, a_c, m_c=0.5):
    comp = {"system": np.repeat(np.arange(n), 2), "role": np.tile([0, 1], n),
            "kind": np.full(2 * n, "star", dtype=object), "phase": np.zeros(2 * n),
            "mass": np.tile([1.0, m_c], n), "initial_mass": np.tile([1.0, m_c], n),
            "radius": np.tile([1.0, 0.5], n), "teff": np.tile([5772.0, 3800.0], n),
            "age_gyr": np.full(2 * n, 5.0), "feh": np.zeros(2 * n), "luminosity": np.tile([1.0, 0.05], n),
            "a_au": np.tile([np.nan, a_c], n), "ecc": np.tile([np.nan, 0.3], n),
            "period_days": np.tile([np.nan, 365.25 * math.sqrt(a_c**3 / (1 + m_c))], n)}
    rng = np.random.default_rng(0)
    comp["incl_sky"] = np.where(comp["role"] > 0, np.arccos(2 * rng.random(2 * n) - 1), np.nan)
    comp["node"] = np.where(comp["role"] > 0, 2 * np.pi * rng.random(2 * n), np.nan)
    comp["omega"] = np.where(comp["role"] > 0, 2 * np.pi * rng.random(2 * n), np.nan)
    return {"age_gyr": np.full(n, 5.0), "n_components": np.full(n, 2)}, comp


def test_kozai_in_the_pipeline():
    systems, comp = _binary(4000, 300.0)
    pl = planets.attach_planets(systems, comp, seed=3)
    kz = (pl["flags"] & planets.FLAG_KOZAI) > 0
    assert kz.any()
    assert set(np.unique(pl["host_kind"][kz])) == {"S"}
    # Cycling planets are dynamically detached: mostly lone or outer giants and wide small planets.
    assert np.median(pl["a_au"][kz]) > 1.0
    assert pl["ecc"][kz].mean() > pl["ecc"][~kz & (pl["a_au"] > 1.0)].mean()
    off = planets.attach_planets(systems, comp, seed=3, binaries=planets.bn.BinaryModel(kozai=False))
    assert not ((off["flags"] & planets.FLAG_KOZAI) > 0).any()
