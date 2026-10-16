import numpy as np
import pytest

from starsys.stars import evolution, population, remnants


def test_sun_today():
    r = evolution.evolve([1.0], [4.6], [0.0])
    assert r["kind"][0] == "star"
    assert 5600 < r["teff"][0] < 6000
    assert 0.9 < r["luminosity"][0] < 1.2
    assert r["spectral_type"][0].startswith("G")


def test_old_intermediate_mass_star_is_white_dwarf():
    r = evolution.evolve([3.0], [2.0], [0.0])
    assert r["kind"][0] == "WD"
    assert r["mass"][0] == pytest.approx(remnants.ifmr_cummings2018_mist(3.0), rel=1e-6)
    assert 0.005 < r["radius"][0] < 0.02


def test_brown_dwarf_cools():
    r = evolution.evolve([0.05, 0.05], [1.0, 8.0], [0.0, 0.0])
    assert (r["kind"] == "BD").all()
    assert r["teff"][1] < r["teff"][0]


def test_ifmr_monotonic():
    mi = np.linspace(0.9, 7.0, 50)
    assert np.all(np.diff(remnants.ifmr_cummings2018_mist(mi)) > 0)


def test_draw_systems_structure():
    systems, comp = population.draw_systems(3000, seed=4)
    assert len(systems["age_gyr"]) == 3000
    assert np.bincount(comp["system"]).tolist() == systems["n_components"].tolist()
    comps = comp["role"] > 0
    assert np.all(comp["a_au"][comps] > 0) and np.all(comp["ecc"][comps] < 1)
    # Components of a system are coeval.
    assert np.allclose(comp["age_gyr"], systems["age_gyr"][comp["system"]])
    assert set(np.unique(comp["kind"])) <= {"star", "BD", "WD", "NS", "BH"}


def test_draw_systems_bright_subsets():
    from starsys.stars.population import default_model, draw_systems, lifetime_gyr
    s, c = draw_systems(400, seed=3, alive_range=(3.0, 8.0))
    m1 = s["primary_initial_mass"]
    assert (m1 >= 3.0).all() and (m1 < 8.0).all()
    assert (s["age_gyr"] < lifetime_gyr(m1, s["feh"])).all()
    s, c = draw_systems(3000, seed=4, exclude_alive_above=1.5)
    m1 = s["primary_initial_mass"]
    assert not ((m1 >= 1.5) & (s["age_gyr"] < lifetime_gyr(m1, s["feh"]))).any()
    # the fraction agrees with plain draws
    s, _ = draw_systems(200000, seed=5)
    m1 = s["primary_initial_mass"]
    alive = ((m1 >= 1.5) & (m1 < 3.0) & (s["age_gyr"] < lifetime_gyr(m1, s["feh"]))).mean()
    assert alive == pytest.approx(default_model().alive_fraction(1.5, 3.0), rel=0.1)


def test_hierarchical_multiples_are_stable():
    from starsys.stars import population as pp
    s, c = pp.draw_systems(40000, seed=11)
    role, sysi, m, a, e = c["role"], c["system"], c["initial_mass"], c["a_au"], c["ecc"]
    outer = np.flatnonzero(role >= 2)
    assert len(outer) > 500
    start = np.searchsorted(sysi, sysi[outer])
    m_in = np.array([m[i0:i].sum() for i0, i in zip(start, outer)])
    assert pp.stable_triple(a[outer - 1], a[outer], e[outer], m[outer] / m_in).all()
    # the check only redraws unstable orbits: everything else is as drawn
    inner = role <= 1
    assert np.isfinite(a[role == 1]).all() and (e[outer] < 1).all() and inner.sum() > 0
