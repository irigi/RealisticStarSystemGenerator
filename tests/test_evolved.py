import numpy as np
import pytest

from starsys import evolved as ev
from starsys import planets


def test_progenitor_and_radius_tips_of_the_sun():
    p = ev.progenitor_state(1.0, 0.0)
    assert 5500 < p["teff"][0] < 6000 and 0.9 < p["radius"][0] < 1.2
    r_rgb, r_agb = ev.radius_tips(1.0, 0.0)
    assert 0.6 < r_rgb[0] < 1.0 and 1.2 < r_agb[0] < 2.0  # ~170 Rsun RGB tip, ~1.5 au AGB


def test_engulfment_radius():
    r_rgb, r_agb = ev.radius_tips(1.0, 0.0)
    r = ev.engulfment_radius(np.array([1.0, 17.0, 318.0]), np.ones(3), np.full(3, r_agb[0]), np.full(3, r_rgb[0]),
                             np.zeros(3, dtype=bool))
    assert np.all(np.diff(r) > 0)  # heavier planets feel stronger tides
    assert r[0] > 1.0 and r[2] < 5.2  # Earth engulfed, Jupiter survives
    # A star still on the giant branch engulfs at least everything inside its maximum radius.
    alive = ev.engulfment_radius(np.array([1.0]), np.array([5.0]), np.array([0.5]), np.array([0.5]), np.array([True]))
    assert alive[0] >= 0.5


def _stars(kind, phase, initial_mass, mass, radius, teff, n):
    comp = {"system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, kind, dtype=object),
            "phase": np.full(n, phase), "initial_mass": np.full(n, initial_mass), "mass": np.full(n, mass),
            "radius": np.full(n, radius), "teff": np.full(n, teff), "age_gyr": np.full(n, 10.0), "feh": np.zeros(n),
            "luminosity": np.full(n, radius**2 * (teff / 5772.0) ** 4)}
    return {"age_gyr": np.full(n, 10.0), "n_components": np.ones(n, int)}, comp


def test_white_dwarfs_keep_expanded_outer_planets():
    systems, comp = _stars("WD", np.nan, 1.0, 0.54, 0.012, 8000.0, 3000)
    pl = planets.attach_planets(systems, comp, seed=1)
    assert len(pl["period"]) > 0
    assert ((pl["flags"] & planets.FLAG_EVOLVED) > 0).all()
    # Every survivor started outside its engulfment radius and moved out by M_prog / M_WD.
    m_prog = ev.progenitor_state(1.0, 0.0)["mass"][0]
    a0 = pl["a_au"] * 0.54 / m_prog
    r_rgb, r_agb = ev.radius_tips(1.0, 0.0)
    r_eng = ev.engulfment_radius(pl["planet_mass"], np.ones(len(a0)), np.full(len(a0), max(r_rgb[0], r_agb[0])),
                                 np.full(len(a0), r_rgb[0]), np.zeros(len(a0), dtype=bool))
    assert np.all(a0 >= r_eng * (1 - 1e-9))
    assert pl["a_au"].min() > 2.5
    assert pl["host_mass"] == pytest.approx(0.54)
    # Periods follow the present-day mass.
    assert np.allclose(pl["period"], 365.25 * np.sqrt(pl["a_au"] ** 3 / 0.54))


def test_subgiants_get_main_sequence_planets():
    systems, comp = _stars("star", 2.0, 1.3, 1.29, 2.5, 5200.0, 2000)
    pl = planets.attach_planets(systems, comp, seed=2)
    assert np.isin(["hmc", "giant"], pl["model"]).all()
    assert pl["a_au"].min() >= 2.5 / 215.032  # nothing inside the star
    assert evolved_without_model_is_empty(systems, comp)


def evolved_without_model_is_empty(systems, comp):
    return len(planets.attach_planets(systems, comp, seed=2, evolved=None)["period"]) == 0


def test_pulsar_planets_are_rare_and_flagged():
    n = 20000
    systems, comp = _stars("NS", np.nan, 12.0, 1.35, 1.4e-5, 1e5, n)
    pl = planets.attach_planets(systems, comp, seed=3)
    assert set(np.unique(pl["model"])) <= {"pulsar"}
    hosts = len(np.unique(pl["component"]))
    assert 0.0005 * n < hosts < 0.005 * n
    assert ((pl["flags"] & planets.FLAG_PULSAR) > 0).all()
