import numpy as np
import pytest

from starsys import planets
from starsys.inner.hmc import COLUMNS, HMCParams
from starsys.stars.population import draw_systems
from starsys.stars.spectral import bp_rp_from_teff


@pytest.fixture(scope="module")
def population():
    return draw_systems(3000, seed=11)


def test_bp_rp_from_teff_sun_and_monotonic():
    assert bp_rp_from_teff(5772.0, kepler_scale=False) == pytest.approx(0.82, abs=0.03)
    teff = np.linspace(3000, 9000, 200)
    assert np.all(np.diff(bp_rp_from_teff(teff)) < 0)


def test_attach_planets_structure(population):
    systems, comp = population
    pl = planets.attach_planets(systems, comp, seed=5)
    n = len(pl["period"])
    assert n > 0 and all(len(v) == n for v in pl.values())
    assert set(COLUMNS) <= set(pl)
    # Only HM-C hosts carry planets, and keys are consistent with the components table.
    assert planets.hmc_hosts(comp)[pl["component"]].all()
    assert np.array_equal(pl["system"], comp["system"][pl["component"]])
    # Sorted by component, then period.
    same = pl["component"][1:] == pl["component"][:-1]
    assert np.all(np.diff(pl["component"]) >= 0) and np.all(np.diff(pl["period"])[same] > 0)
    assert np.all((pl["period"] >= 3.0) & (pl["period"] <= 300.0))
    assert set(np.unique(pl["hmc_catalog"])) <= set(planets.hmc_posterior()[0])


def test_flags(population):
    systems, comp = population
    pl = planets.attach_planets(systems, comp, seed=5)
    host_teff = comp["teff"][pl["component"]]
    colour_flag = (pl["flags"] & planets.FLAG_COLOUR) > 0
    assert colour_flag[host_teff < 3500].all()  # M dwarfs are redder than bp-rp 1.7
    assert not colour_flag[(host_teff > 4500) & (host_teff < 6000)].any()
    multiple = (pl["flags"] & planets.FLAG_MULTIPLE) > 0
    assert np.array_equal(multiple, systems["n_components"][pl["system"]] > 1)


def test_reproducible_and_fixed_params(population):
    systems, comp = population
    a = planets.attach_planets(systems, comp, seed=9)
    b = planets.attach_planets(systems, comp, seed=9)
    assert np.array_equal(a["period"], b["period"])
    c = planets.attach_planets(systems, comp, seed=9, params=HMCParams(), age="star")
    assert np.all(c["hmc_catalog"] == 0)


def test_per_star_age_reduces_stripping_of_young_hosts():
    # Two identical sets of Sun-like stars at 0.5 Gyr: the per-star age should strip fewer envelopes.
    n = 4000
    comp = {"system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, "star", dtype=object),
            "phase": np.zeros(n), "mass": np.ones(n), "radius": np.ones(n), "teff": np.full(n, 5772.0),
            "age_gyr": np.full(n, 0.5)}
    systems = {"age_gyr": np.full(n, 0.5), "n_components": np.ones(n, int)}
    fixed = planets.attach_planets(systems, comp, seed=1, params=HMCParams())
    young = planets.attach_planets(systems, comp, seed=1, params=HMCParams(), age="star")
    assert young["lost_envelope"].mean() < 0.5 * fixed["lost_envelope"].mean()
