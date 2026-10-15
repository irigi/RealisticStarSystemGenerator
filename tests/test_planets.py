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
    # Only main-sequence stars, brown dwarfs, evolved stars and neutron stars carry planets
    # (circumbinary planets are listed under the primary), and keys match the components table.
    from starsys.evolved import evolved_hosts

    allowed = planets.hmc_hosts(comp) | evolved_hosts(comp) | (comp["kind"] == "NS")
    assert allowed[pl["component"]].all()
    assert np.array_equal(pl["system"], comp["system"][pl["component"]])
    assert (comp["role"][pl["component"][pl["host_kind"] == "P"]] == 0).all()
    # Sorted by component, host kind, then period.
    same = (pl["component"][1:] == pl["component"][:-1]) & (pl["host_kind"][1:] == pl["host_kind"][:-1])
    assert np.all(np.diff(pl["component"]) >= 0) and np.all(np.diff(pl["period"])[same] > 0)
    ms = planets.hmc_hosts(comp)[pl["component"]] & (pl["host_kind"] == "S")
    inner = np.isin(pl["model"], ["hmc", "hmc_m"]) & ms
    assert np.all((pl["period"][inner] >= 0.5) & (pl["period"][inner] <= 300.0))
    assert set(np.unique(pl["model"])) <= {"hmc", "hmc_m", "giant", "hot_jupiter", "wide_giant", "outer_small",
                                           "circumbinary", "pulsar", "short_period"}
    fgk = (pl["model"] == "hmc") & ms
    assert np.all(pl["period"][fgk] >= 3.0)
    assert set(np.unique(pl["hmc_catalog"])) <= set(planets.hmc_posterior()[0])


def test_flags(population):
    systems, comp = population
    pl = planets.attach_planets(systems, comp, seed=5, binaries=None, evolved=None)
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
    # With the binary model, ``multiple`` marks companions inside 200 au; circumbinary and
    # evolved hosts carry their own flags.
    full = planets.attach_planets(systems, comp, seed=5)
    multiple = (full["flags"] & planets.FLAG_MULTIPLE) > 0
    assert (systems["n_components"][full["system"]][multiple] > 1).all()
    assert np.array_equal((full["flags"] & planets.FLAG_CIRCUMBINARY) > 0, full["host_kind"] == "P")
    from starsys.evolved import evolved_hosts

    assert np.array_equal((full["flags"] & planets.FLAG_EVOLVED) > 0, evolved_hosts(comp)[full["component"]]
                          & (full["host_kind"] == "S"))
    # Without the extension, M dwarfs are out of HM-C's colour domain.
    plain = planets.attach_planets(systems, comp, seed=5, mdwarf=None, small=None)
    cool = comp["teff"][plain["component"]] < 3500
    assert ((plain["flags"][cool] & planets.FLAG_COLOUR) > 0).all()
    assert set(np.unique(plain["model"])) <= {"hmc", "giant", "hot_jupiter", "wide_giant", "circumbinary", "pulsar",
                                              "short_period"}


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
    assert set(np.unique(pl["model"])) <= {"hmc_m", "outer_small", "short_period"}  # no giants below 0.1 Msun
    assert np.mean(pl["model"] == "short_period") < 0.02


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
    fixed = planets.attach_planets(systems, comp, seed=1, params=HMCParams(), giants=None, small=None)
    young = planets.attach_planets(systems, comp, seed=1, params=HMCParams(), age="star", giants=None, small=None)
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


def test_intermediate_mass_hosts_have_giants_near_800_days():
    from starsys.outer import giants as gp
    from starsys.units import a_au_to_period

    rng = np.random.default_rng(1)
    model = gp.GiantModel()
    assert gp.im_weight(np.array([1.0, 1.35, 2.0]), model).tolist() == pytest.approx([0.0, 0.5, 1.0])
    a1 = gp.draw_cold_separation(np.full(100_000, 1.0), rng, model)
    a2 = gp.draw_cold_separation(np.full(100_000, 2.0), rng, model)
    p2 = a_au_to_period(a2, 2.0)
    # Wolthoff+2022: log-normal in P, mode 793 d, sigma 0.89 (the Fulton s(a) peaks at ~3.7 au)
    assert np.median(np.log(p2)) == pytest.approx(np.log(793.0), abs=0.02)
    assert np.std(np.log(p2)) == pytest.approx(0.89, abs=0.02)
    assert np.mean(p2 < 90) < 0.01 < np.mean(a_au_to_period(a1, 1.0) < 90)
    assert (a2 <= model.a_max).all() and (p2 >= model.p_min_cold).all()


def test_rv_mass_rescaling():
    from starsys.inner import rv_masses

    c = rv_masses.correction(np.array([2.0, 3.5, 5.0, 7.0, 9.5, 10.0]))
    assert c[:2].tolist() == [1.0, 1.0] and (c <= 1.0).all() and (c[2:] < 0.7).all()
    assert rv_masses.target_mass(5.0) == pytest.approx(1.74 * 5.0**1.58)  # Otegi+2020
    assert rv_masses.target_mass(9.45) == pytest.approx(95.2, rel=1e-6)  # Saturn
    systems, comp = _sun_like(3000)
    kw = dict(seed=3, params=HMCParams(), giants=None, small=None, derived=None)
    a = planets.attach_planets(systems, comp, rv_masses=False, **kw)
    b = planets.attach_planets(systems, comp, **kw)
    assert np.array_equal(a["planet_radius"], b["planet_radius"])
    big = (a["planet_radius"] > 5) & ~a["lost_envelope"] & (a["model"] == "hmc")
    assert big.any() and (b["planet_mass"][big] < 0.7 * a["planet_mass"][big]).all()
    small = a["planet_radius"] < 3.5
    assert np.array_equal(a["planet_mass"][small], b["planet_mass"][small])


def test_angles_refer_to_the_whole_system_invariable_plane():
    systems, comp = _sun_like(5000, feh=0.3)
    pl = planets.attach_planets(systems, comp, seed=5, params=HMCParams(), derived=None)
    e = pl["ecc"]
    w = pl["planet_mass"] * np.sqrt(pl["a_au"] * (1 - e * e))
    i, node = pl["incl_invariable"], pl["asc_node_invariable"]
    n = len(comp["mass"])
    # In-plane components of the total angular momentum vanish for every host.
    for comp_xy in (np.sin(i) * np.cos(node), np.sin(i) * np.sin(node)):
        resid = np.bincount(pl["component"], weights=w * comp_xy, minlength=n)
        assert np.abs(resid).max() < 1e-9 * np.bincount(pl["component"], weights=w, minlength=n).max()
    # Inner planets of giant hosts are tilted to the giant-dominated plane (Masuda+2020: ~12 deg).
    giant_host = np.isin(pl["component"], pl["component"][pl["model"] == "giant"])
    inner = giant_host & (pl["model"] == "hmc")
    assert np.median(np.degrees(i[inner])) > 3.0


def test_small_planets_beyond_pmax():
    from starsys.outer.giants import pair_stable
    from starsys.outer.small import _hill_ok

    systems, comp = _sun_like(20_000, feh=0.2)
    pl = planets.attach_planets(systems, comp, seed=6, params=HMCParams())
    o = pl["model"] == "outer_small"
    assert o.sum() > 10_000 and (pl["clusterid"][o] == planets.SMALL_CLUSTER_ID).all()
    assert (pl["period"][o] >= 300.0 * (1 - 1e-9)).all()
    assert (pl["mass_init"][o] >= 0.1).all() and (pl["mass_init"][o] <= 30.0).all()
    assert (pl["a_au"][o] <= 27.0 * (1 + 1e-9)).all()  # 10 snow-line radii of a 1 Msun star
    bridge = (pl["flags"] & planets.FLAG_BRIDGE) > 0
    assert np.array_equal(bridge[o], pl["a_au"][o] < 2.7)
    assert not bridge[~o].any()
    log_q = np.log10(pl["mass_init"] * 3.0035e-6)
    extrap = (pl["flags"] & planets.FLAG_COLD_EXTRAP) > 0
    assert np.array_equal(extrap[o], ((pl["a_au"] >= 2.7) & ((log_q < -5.2) | (pl["a_au"] > 10.8)))[o])
    # Every adjacent pair involving a small outer planet is stable, with HM-C's Hill spacing
    # unless the neighbour is a giant.
    giant = pl["clusterid"] == planets.GIANT_CLUSTER_ID
    same = pl["component"][1:] == pl["component"][:-1]
    pairs = np.flatnonzero(same & (o[1:] | o[:-1]))
    assert len(pairs) > 5000
    for i in pairs[:3000]:
        m1, m2 = pl["mass_init"][i] if o[i] else pl["planet_mass"][i], pl["mass_init"][i + 1]
        a1, a2, e1, e2 = pl["a_au"][i], pl["a_au"][i + 1], pl["ecc"][i], pl["ecc"][i + 1]
        assert pair_stable(m1, m2, a1, e1, a2, e2, 1.0)
        if not (giant[i] or giant[i + 1]):
            assert _hill_ok(m1, m2, a1, a2, 1.0, 10.0)


def test_small_planet_density_is_continuous_and_normalised():
    from starsys.outer import small as sp

    model = sp.SmallPlanetModel(sample_uncertainty=False)
    ml = sp.draw_ml_params(2, np.random.default_rng(0), model)
    mstar = np.array([1.0, 0.2])
    c, mu, sd = sp.outer_anchor(mstar, ml, model)
    # The lower Gaussian integrates to Zang+2025's rate over the measured log q range.
    lo, hi = (x * np.log(10) + np.log(mstar[0] / 3.0035e-6) for x in model.ml_log_q_range)
    y = np.linspace(lo, hi, 4001)
    integral = np.trapezoid(np.exp(c[0] - 0.5 * ((y - mu[0]) / sd[0]) ** 2), y) * np.log(10)
    assert integral == pytest.approx(model.ml_rate[0], rel=1e-3)
    # Continuity: at P_max the density is HM-C's, beyond the snow line the microlensing one.
    n_in, mu_in, s_in = np.array([1.2, 1.2]), np.array([0.4, -1.0]), np.array([1.5, 1.4])
    a_in = np.array([0.85, 0.6])  # the 0.2 Msun snow line (0.54 au) is inside the second
    h = sp._Hosts(mstar, a_in, (n_in, mu_in, s_in), (c, mu, sd), model)
    ylo, yhi = np.log(model.m_min), np.log(model.m_max)
    at_in = sp._marginal(*h._gauss(np.log(a_in), np.arange(2)), ylo, yhi)
    trunc = [np.diff(sp.ndtr((np.array([ylo, yhi]) - m) / s))[0] for m, s in zip(mu_in, s_in)]
    assert at_in[0] == pytest.approx(n_in[0] * trunc[0], rel=1e-6)
    out = sp._marginal(*h._gauss(np.log(3 * h.a_sl), np.arange(2)), ylo, yhi)
    direct = sp._marginal(c, mu, sd, ylo, yhi)
    assert np.allclose(out, direct)
    # No bridge when the snow line is inside a(P_max): microlensing density from P_max on.
    assert at_in[1] == pytest.approx(direct[1], rel=1e-6)
    # Sampler matches the cumulative counts.
    rng = np.random.default_rng(1)
    a, m = h.draw(np.zeros(200_000, dtype=int), rng)
    frac = (h.cum_at(np.log(2.7 * 4), [0]) - h.cum_at(np.log(2.7), [0]))[0] / h.expected[0]
    assert ((a >= 2.7) & (a < 10.8)).mean() == pytest.approx(frac, abs=0.005)
    assert (m >= model.m_min).all() and (m <= model.m_max).all()
    far = a > 5.0
    z = (yhi - mu[0]) / sd[0]  # upper truncation at 30 Earth masses (the lower one is negligible)
    truncated_mean = mu[0] - sd[0] * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) / sp.ndtr(z)
    assert np.log(m[far]).mean() == pytest.approx(truncated_mean, abs=0.02)
