import math

import numpy as np
import pytest

from starsys import derived as dv
from starsys import moons
from starsys import planets

RSUN_AU = 1.0 / 215.032


def _planets(n, a, mass, radius, host_mass=1.0, lum=1.0, teff=5772.0, age=4.6, lost=False):
    comp = {"system": np.zeros(1, int), "role": np.zeros(1, int), "luminosity": np.array([lum]),
            "teff": np.array([teff]), "mass": np.array([host_mass]), "age_gyr": np.array([age]),
            "a_au": np.array([np.nan]), "ecc": np.array([np.nan])}
    pl = {"system": np.zeros(n, int), "component": np.zeros(n, int), "host_kind": np.full(n, "S", dtype=object),
          "host_mass": np.full(n, host_mass), "a_au": np.full(n, a), "ecc": np.zeros(n),
          "planet_mass": np.full(n, mass), "planet_radius": np.full(n, radius), "lost_envelope": np.full(n, lost),
          "flags": np.zeros(n, np.int32)}
    pl.update({k: v for k, v in dv.derive_properties(pl, {"n_components": np.array([1])}, comp).items()
               if k != "hz_flags"})
    return pl, comp


def test_jupiter_analogues():
    pl, comp = _planets(4000, 5.2, 317.8, 11.2)
    m = moons.attach_moons(pl, comp, seed=1)
    assert set(np.unique(m["kind"])) <= {"regular"}  # too massive for capture, not rocky
    n = np.bincount(m["planet"], minlength=4000)
    assert n.max() <= 5 and 0.7 < np.mean(n > 0) < 0.85
    tot = np.bincount(m["planet"], weights=m["mass"], minlength=4000)[n > 0] / 317.8
    assert 1e-4 < np.median(tot) < 4e-4
    r_hill = 5.2 * (317.8 * 3.0035e-6 / 3.0) ** (1 / 3) * 1.496e8  # km
    assert (m["a_km"] < 0.4895 * r_hill).all() and (m["a_rp"] > 3.0).all()
    assert ((m["flags"] & planets.FLAG_MOON) > 0).all()
    # Jupiter's T_eq0 is ~120 K: moons formed outside 12 R_J are icy, inside rocky (massive inner
    # moons migrate out afterwards).
    assert (m["a_rp"][m["composition"] == "ice"] > 11.9).all() and (m["composition"][m["a_rp"] < 11] == "rock").all()
    # Galilean moons' periods: Io 1.77 d at 5.9 R_J.
    io = (m["a_rp"] > 5.8) & (m["a_rp"] < 6.0)
    if io.any():
        assert m["period"][io].mean() == pytest.approx(1.77, rel=0.05)


def test_earth_analogues_and_the_moon():
    pl, comp = _planets(20000, 1.0, 1.0, 1.0, lost=True)
    m = moons.attach_moons(pl, comp, seed=2)
    assert set(np.unique(m["kind"])) == {"impact"}
    assert len(m["planet"]) / 20000 == pytest.approx(1 / 12, rel=0.1)
    # A Moon-mass moon formed at 3.8 Earth radii ends near 50 Earth radii after 4.6 Gyr.
    cand = {"planet": np.array([0]), "kind": np.array(["impact"], dtype=object), "mass": np.array([0.0123]),
            "a_rp": np.array([3.8]), "incl": np.array([0.05]), "icy": np.array([False])}
    keep, a = moons._survive(cand, pl, pl["t_lock_gyr"], np.full(20000, 4.6), moons.MoonModel(), dv.DerivedModel())
    assert keep[0] and 40 < a[0] < 70


def test_close_in_planets_lose_their_moons():
    pl, comp = _planets(3000, 0.04, 317.8, 12.0)
    assert len(moons.attach_moons(pl, comp, seed=3)["planet"]) == 0  # Barnes & O'Brien 2002
    # A rocky planet in an M dwarf's habitable zone is tidally locked; its impact moon spirals in.
    pl, comp = _planets(3000, 0.05, 1.0, 1.0, host_mass=0.15, lum=0.002, teff=3100.0, age=5.0, lost=True)
    assert (pl["spin"] > 0).all()
    assert len(moons.attach_moons(pl, comp, seed=4)["planet"]) == 0


def test_population_moons():
    from starsys.stars.population import draw_systems

    systems, comp = draw_systems(5000, seed=5)
    pl = planets.attach_planets(systems, comp, seed=6)
    m = moons.attach_moons(pl, comp, seed=7)
    assert len(m["planet"]) > 0 and np.all(np.diff(m["planet"]) >= 0)
    assert np.array_equal(m["system"], pl["system"][m["planet"]])
    assert np.isfinite(m["radius"]).all() and (m["period"] > 0).all()
    captured = m["kind"] == "captured"
    assert (m["incl"][captured] > math.pi / 2).all()
    # A captured moon replaces the regular satellites.
    assert not np.isin(m["planet"][captured], m["planet"][m["kind"] == "regular"]).any()


def test_resonant_moons_are_tidally_heated():
    pl, comp = _planets(4000, 5.2, 317.8, 11.2)
    m = moons.attach_moons(pl, comp, seed=3)
    res = m["ecc"] > 0
    assert res.any() and (~res).any()
    assert ((m["ecc"][res] >= 0.001) & (m["ecc"][res] <= 0.01)).all()
    assert (m["tidal_heating_w"][~res] == 0).all() and (m["tidal_heating_w"][res] > 0).all()
    # Io-like moons (inner, resonant, ~0.015 Earth masses) dissipate ~1e13-1e15 W.
    io = res & (m["a_rp"] < 8) & (m["mass"] > 0.005)
    assert 1e12 < np.median(m["tidal_heating_w"][io]) < 1e16
