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
    inner = np.isin(pl["model"], ["hmc", "hmc_m"])
    assert np.all((pl["period"][inner] >= 0.5) & (pl["period"][inner] <= 300.0))
    assert set(np.unique(pl["model"])) <= {"hmc", "hmc_m", "giant", "hot_jupiter", "wide_giant"}
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
    assert set(np.unique(plain["model"])) <= {"hmc", "giant", "hot_jupiter", "wide_giant"}


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
            "teff": np.full(n, 2000.0), "age_gyr": np.full(n, 3.0), "feh": np.zeros(n), "luminosity": np.full(n, 1e-4)}
    systems = {"age_gyr": np.full(n, 3.0), "n_components": np.ones(n, int)}
    pl = planets.attach_planets(systems, comp, seed=2)
    hosts = len(np.unique(pl["component"]))
    assert 0 < hosts < 0.5 * n
    assert np.median(pl["planet_radius"]) < 1.5
    assert (pl["flags"] & planets.FLAG_SUBSTELLAR).all()
    assert set(np.unique(pl["model"])) == {"hmc_m"}  # no giants below 0.1 Msun


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
    fixed = planets.attach_planets(systems, comp, seed=1, params=HMCParams(), giants=None)
    young = planets.attach_planets(systems, comp, seed=1, params=HMCParams(), age="star", giants=None)
    assert young["lost_envelope"].mean() < 0.5 * fixed["lost_envelope"].mean()


def _sun_like(n, feh=0.0):
    comp = {"system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, "star", dtype=object),
            "phase": np.zeros(n), "mass": np.ones(n), "radius": np.ones(n), "teff": np.full(n, 5772.0),
            "age_gyr": np.full(n, 4.6), "feh": np.full(n, feh), "luminosity": np.ones(n)}
    return {"age_gyr": np.full(n, 4.6), "n_components": np.ones(n, int)}, comp


def test_giants_merge_with_inner_system():
    from starsys.outer.giants import pair_stable

    systems, comp = _sun_like(20_000, feh=0.2)
    pl = planets.attach_planets(systems, comp, seed=4, params=HMCParams())
    giant = np.isin(pl["model"], ["giant", "hot_jupiter", "wide_giant"])
    assert giant.sum() > 1000 and (pl["clusterid"][giant] == planets.GIANT_CLUSTER_ID).all()
    # Inside HM-C's 3-300 d window, giant-model planets are the ones HM-C cannot make (R >= 10 Re).
    window = giant & (pl["period"] >= 3.0) & (pl["period"] <= 300.0)
    assert (pl["planet_radius"][window] >= 10.0).all()
    # Sorted, and every adjacent giant/any pair is AMD-Hill stable.
    same = pl["component"][1:] == pl["component"][:-1]
    assert np.all(np.diff(pl["period"])[same] > 0)
    for i in np.flatnonzero(same & (giant[1:] | giant[:-1]))[:2000]:
        args = (pl["planet_mass"][i], pl["planet_mass"][i + 1], pl["a_au"][i], pl["ecc"][i], pl["a_au"][i + 1],
                pl["ecc"][i + 1], 1.0)
        assert pair_stable(*args)
    # Hot-Jupiter hosts mostly have no inner (HM-C) planets.
    hj = np.unique(pl["component"][pl["model"] == "hot_jupiter"])
    inner_hosts = np.unique(pl["component"][pl["model"] == "hmc"])
    assert np.isin(hj, inner_hosts).mean() < 0.4


def test_giant_occurrence_scales_with_metallicity_and_mass():
    from starsys.outer import giants as gp

    assert gp.host_scaling(1.0, 0.0) == pytest.approx(1.0)
    assert gp.host_scaling(1.0, 0.3) > 2.0
    assert gp.host_scaling(0.05, 0.0) == 0.0
    assert gp.host_scaling(1.68, 0.0) > gp.host_scaling(1.0, 0.0) > gp.host_scaling(0.5, 0.0)
    assert gp.host_scaling(3.5, 0.0) < 0.3 * gp.host_scaling(1.68, 0.0)
    rich = planets.attach_planets(*_sun_like(10_000, 0.3), seed=1, params=HMCParams())
    poor = planets.attach_planets(*_sun_like(10_000, -0.3), seed=1, params=HMCParams())
    assert (rich["model"] == "giant").sum() > 2.5 * (poor["model"] == "giant").sum()


def test_giant_samplers_match_their_densities():
    from starsys.outer import giants as gp

    rng = np.random.default_rng(0)
    model = gp.GiantModel()
    m = gp.draw_mass(200_000, rng, model)
    assert m.min() >= model.m_min and m.max() <= model.m_max
    frac = gp.mass_integral(300, 3000, model) / gp.mass_integral(model.m_min, model.m_max, model)
    assert ((m >= 300) & (m < 3000)).mean() == pytest.approx(frac, abs=0.005)
    a = gp.draw_separation(np.full(200_000, 0.1), np.full(200_000, 30.0), rng, model)
    frac = gp.separation_integral(1, 5, model) / gp.separation_integral(0.1, 30, model)
    assert ((a >= 1) & (a < 5)).mean() == pytest.approx(frac, abs=0.005)
