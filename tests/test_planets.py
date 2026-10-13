import numpy as np
import pytest

from starsys import planets
from starsys.inner import hmc
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
    assert np.all((pl["period"] >= 0.5) & (pl["period"] <= 300.0))
    fgk = pl["model"] == "hmc"
    assert np.all(pl["period"][fgk] >= 3.0)
    assert set(np.unique(pl["hmc_catalog"])) <= set(planets.hmc_posterior()[0])


def test_flags(population):
    systems, comp = population
    pl = planets.attach_planets(systems, comp, seed=5)
    host_teff = comp["teff"][pl["component"]]
    colour_flag = (pl["flags"] & planets.FLAG_COLOUR) > 0
    assert colour_flag[host_teff > 7000].all()  # bluer than HM-C's bp-rp 0.5
    assert not colour_flag[host_teff < 6000].any()  # M dwarfs are covered by the extension
    multiple = (pl["flags"] & planets.FLAG_MULTIPLE) > 0
    assert np.array_equal(multiple, systems["n_components"][pl["system"]] > 1)
    substellar = (pl["flags"] & planets.FLAG_SUBSTELLAR) > 0
    assert np.array_equal(substellar, comp["mass"][pl["component"]] < 0.1)
    young = (pl["flags"] & planets.FLAG_YOUNG) > 0
    assert np.array_equal(young, comp["age_gyr"][pl["component"]] < planets.YOUNG_AGE_GYR)
    # Without the extension, M dwarfs are out of HM-C's colour domain.
    plain = planets.attach_planets(systems, comp, seed=5, mdwarf=None)
    cool = comp["teff"][plain["component"]] < 3500
    assert ((plain["flags"][cool] & planets.FLAG_COLOUR) > 0).all()
    assert set(np.unique(plain["model"])) == {"hmc"}


def test_mdwarf_extension_leaves_fgk_untouched():
    from starsys.inner.mdwarf import MDwarfModel, adjust_params

    p = HMCParams()
    rows = np.tile(p.as_array(), (3, 1))
    out = adjust_params(rows, mass=np.array([1.0, 0.45, 0.05]), teff=np.array([5800.0, 3600.0, 2500.0]),
                        model=MDwarfModel())
    assert np.array_equal(out[0], rows[0])
    assert out[1, hmc.I_PMIN] < p.min_period and out[1, hmc.I_LRC] > p.log_rate_clusters
    # Below 0.1 Msun the host fraction follows the disk-mass scaling.
    assert out[2, hmc.I_FSLOPE] == 0.0 and out[2, hmc.I_FMED] == pytest.approx(0.5**1.9)


def test_brown_dwarfs_get_few_small_planets():
    n = 4000
    comp = {"system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, "BD", dtype=object),
            "phase": np.full(n, np.nan), "mass": np.full(n, 0.05), "radius": np.full(n, 0.1),
            "teff": np.full(n, 2000.0), "age_gyr": np.full(n, 3.0)}
    systems = {"age_gyr": np.full(n, 3.0), "n_components": np.ones(n, int)}
    pl = planets.attach_planets(systems, comp, seed=2)
    hosts = len(np.unique(pl["component"]))
    assert 0 < hosts < 0.5 * n
    assert np.median(pl["planet_radius"]) < 1.5
    assert (pl["flags"] & planets.FLAG_SUBSTELLAR).all()


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
