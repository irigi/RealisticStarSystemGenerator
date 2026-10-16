import numpy as np
import pytest

from starsys import galaxy as g


def test_local_density_matches_census():
    r, n = 35.0, 0
    for k in g.sectors_within((0, 0, 0), r):
        n += int((np.linalg.norm(g.sector(*k).xyz, axis=1) < r).sum())
    expected = g.LOCAL_DENSITY * 4 / 3 * np.pi * r**3
    assert n == pytest.approx(expected, rel=0.05)


def test_sectors_are_deterministic_and_systems_addressable():
    a = g.sector(3, -2, 1)
    g.sector.cache_clear()
    b = g.sector(3, -2, 1)
    assert np.array_equal(a.xyz, b.xyz) and np.array_equal(a.components["mass"], b.components["mass"])
    assert (np.floor(a.xyz / g.SECTOR_PC) == [3, -2, 1]).all()
    sid = a.sid(2)
    sys1 = g.system(sid, moons=False)
    assert sys1.components["mass"][0] == a.components["mass"][a.components["system"] == 2][0]
    assert g.parse(sid) == (0, (3, -2, 1), 2) and g.parse("G2:1.-1.0-7") == (2, (1, -1, 0), 7)


def test_structure():
    t = g.galaxy_totals()
    m = g.mean_system_mass()
    assert 5e10 < sum(t.values()) < 2e11
    assert 1e9 < t["halo"] * m < 2e9 and 1.4e10 < t["bulge"] * m < 1.8e10
    centre = g.densities(np.array([[g.R0_PC, 0.0, -g.Z_SUN_PC]]))
    assert centre["bulge"][0] > 1.0
    # the Sun lies near the Local arm; points far between arms have S ~ 0
    assert g.arm_strength(np.zeros((1, 3)))[0] > 0.3
    R, beta, z = g.galactocentric(np.zeros((1, 3)))
    assert R[0] == pytest.approx(g.R0_PC) and beta[0] == pytest.approx(0) and z[0] == pytest.approx(g.Z_SUN_PC)


def test_populations_vary_with_position():
    bulge = g.sector(814, 0, 0)  # at the Galactic centre
    assert (bulge.population == "bulge").mean() > 0.9 and len(bulge) > 1000
    assert np.median(bulge.systems["age_gyr"]) > 8
    outer = g.population("thin", 12000.0)
    inner = g.population("thin", 5000.0)
    assert outer.feh_thin_intercept < inner.feh_thin_intercept


def test_random_system_is_fixed_and_follows_the_galaxy():
    import collections
    a = [g.random_system(k) for k in range(60)]
    assert a == [g.random_system(k) for k in range(60)] and len(set(a)) == 60
    pops = collections.Counter(g.cell(g.parse(s)[0], *g.parse(s)[1]).population[g.parse(s)[2]] for s in a)
    # thin disc and bulge dominate by number (galaxy_totals: 46 % and 41 %)
    assert pops["thin"] > 10 and pops["bulge"] > 10


def test_brightness_tiers():
    # deterministic, addressable, and only bright living primaries
    a = g.cell(2, 0, 0, 0)
    g.cell.cache_clear()
    b = g.cell(2, 0, 0, 0)
    assert len(a) > 20 and np.array_equal(a.xyz, b.xyz)
    assert (np.floor(a.xyz / 160.0) == 0).all()
    sid = a.sid(1)
    assert sid.startswith("G2:0.0.0-") and g.parse(sid) == (2, (0, 0, 0), 1)
    sys1 = g.system(sid, moons=False)
    assert 3.0 <= sys1.systems["primary_initial_mass"][0] < 8.0 and sys1.components["kind"][0] == "star"
    # no living massive primary in the old bulge and halo
    assert all(g.tier_fraction(t, c) == 0 for t in g.TIERS for c in ("thick", "bulge", "halo"))
    # tier-1 density near the Sun: the thin-disc fraction of the local density (young stars
    # follow the arms, so allow for the local arm factor)
    r, n = 150.0, 0
    for k in g.cells_within(1, (0, 0, 0), r):
        n += int((np.linalg.norm(g.cell(1, *k).xyz, axis=1) < r).sum())
    d = g.densities(np.zeros((1, 3)))
    expected = float(d["thin"][0]) * g.tier_fraction(1, "thin") * 4 / 3 * np.pi * r**3
    assert n == pytest.approx(expected, rel=0.15)
    # every system belongs to exactly one tier: the galaxy totals are unchanged
    assert sum(g.tier_totals().values()) == pytest.approx(sum(g.galaxy_totals().values()))
