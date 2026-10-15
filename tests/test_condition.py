import numpy as np
import pytest

from starsys.real import condition as cd
from starsys.real import star
from starsys.real.known import known_planets

SMALL = cd.ConditionModel(n_prior=3000, max_prior=6000, target_ess=60)
FAST_STAR = star.StarFitModel(n_age=1200, n_mass=100)


def test_geometry_round_trip():
    rng = np.random.default_rng(0)
    inc, node = np.arccos(rng.uniform(-1, 1, 50)), rng.uniform(0, 2 * np.pi, 50)
    i2, n2 = cd.sky_angles(cd.normals_from_sky(inc, node), np.array([0.0, 0.0, 1.0]))
    nrm = cd.normals_from_sky(inc, node)
    assert np.allclose(cd.normals_from_sky(i2, n2), nrm @ cd._frame(np.array([0.0, 0.0, 1.0])).T)
    assert np.allclose(i2, inc)
    v = cd._band_views(nrm[0], 0.05, 200, rng)
    assert np.all(np.abs(v @ nrm[0]) <= 0.05 + 1e-12) and np.allclose(np.linalg.norm(v, axis=1), 1)


def test_tempering():
    rng = np.random.default_rng(1)
    chi2 = rng.uniform(0, 30, 5000)
    none = np.zeros(5000, dtype=int)
    lam, mu, w, ess = cd.tempering(chi2, np.zeros(5000), none, 25.0, 2000.0)  # ESS 670 at lambda = 1
    assert 1 < lam < 2.5 and mu == 1 and ess >= 1999 and w.sum() == pytest.approx(1)
    assert cd.tempering(np.zeros(100), np.zeros(100), np.zeros(100, int), 25.0, 50.0)[:2] == (1.0, 1.0)
    # Insertions only matter once the kernels are at their widest.
    n_ins = rng.integers(0, 4, 5000)
    lam, mu, w, ess = cd.tempering(np.zeros(5000), np.zeros(5000), n_ins, 25.0, 2000.0)
    assert lam == 1.0 and mu > 1 and ess >= 1999


def test_known_planets_rv():
    c = cd.condition("Proxima Centauri", n=25, seed=3, model=SMALL, star_model=FAST_STAR, use_gaia=False)
    pl = c.planets
    known = (pl["flags"] & cd.FLAG_KNOWN) > 0
    assert np.all(pl["component"][known] == c.target[pl["system"][known]])
    for k in known_planets("Proxima Cen"):
        sel = pl["known_name"] == k.name
        assert np.bincount(pl["system"][sel], minlength=25).tolist() == [1] * 25
        assert np.allclose(pl["period"][sel], k.period)
        # Minimum masses are kept at the system's orientation.
        assert np.allclose(pl["planet_mass"][sel] * np.maximum(np.sin(pl["incl_sky"][sel]), 0.05), k.msini,
                           rtol=1e-6)
    assert np.isfinite(pl["insolation"][known]).all()


def test_known_transiting_planet_transits():
    c = cd.condition("HD 209458", n=20, seed=4, model=SMALL, star_model=FAST_STAR, use_gaia=False)
    pl = c.planets
    sel = pl["known_name"] == "HD 209458 b"
    assert sel.sum() == 20
    r_star = c.components["radius"][pl["component"][sel]] * cd.RSUN_AU
    assert np.all(np.abs(np.cos(pl["incl_sky"][sel])) < (r_star + pl["planet_radius"][sel] * cd.EARTH_RADIUS_AU)
                  / pl["a_au"][sel] + 1e-9)
    assert np.allclose(pl["planet_radius"][sel], known_planets("HD 209458")[0].radius)
