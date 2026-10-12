import math

import numpy as np
import pytest

from starsys.inner import hmc


@pytest.fixture(autouse=True)
def _seed():
    hmc._seed(12345)


def test_semimajor_axis_earth():
    assert hmc.semimajor_axis(365.25, 1.0) == pytest.approx(1.0, rel=1e-4)


def test_critical_eccentricity_solves_collision_condition():
    gamma, alpha = 0.7, 0.8
    e1 = hmc.critical_eccentricity(gamma, alpha)
    f = alpha * e1 + gamma * e1 / math.sqrt(alpha * (1 - e1**2) + gamma**2 * e1**2) - 1 + alpha
    assert abs(f) < 1e-12
    # The two orbits touch: a1(1+e1) = a2(1-e2).
    e2 = 1 - alpha - alpha * e1
    assert alpha * (1 + e1) == pytest.approx(1 - e2)


def test_ztp_matches_truncated_poisson():
    lam = 1.8
    draws = np.array([hmc.draw_ztp(lam, 1, 20) for _ in range(200_000)])
    assert draws.min() >= 1
    expected_mean = lam / (1 - math.exp(-lam))
    assert draws.mean() == pytest.approx(expected_mean, rel=0.01)


def test_power_law_bounds_and_shape():
    x = np.array([hmc.draw_power_law(-0.5, 3.0, 300.0) for _ in range(100_000)])
    assert x.min() >= 3.0 and x.max() <= 300.0
    # CDF of x^-0.5 between x0 and x1 at the geometric midpoint.
    mid = 30.0
    cdf = (mid**0.5 - 3**0.5) / (300**0.5 - 3**0.5)
    assert (x < mid).mean() == pytest.approx(cdf, abs=0.01)


def test_trunc_lognormal_respects_bounds_even_in_far_tail():
    x = np.array([hmc.draw_trunc_lognormal(10.0, 0.3, 0.1, 1000.0) for _ in range(2000)])
    assert x.min() >= 0.1 and x.max() <= 1000.0


def test_pure_silicate_radius_earth_like():
    # Seager+2007 pure-silicate planet of one Earth mass is ~1.0-1.1 Earth radii.
    assert 0.95 < hmc.radius_pure_silicate_seager2007(1.0) < 1.15


def test_critical_amd_hill_vs_mmr_differ():
    mu = np.array([3e-5, 3e-5])
    a = np.array([0.1, 0.13])
    hill = hmc.critical_amd_system(mu, a, 2, True)
    mmr = hmc.critical_amd_system(mu, a, 2, False)
    assert hill > 0 and mmr > 0 and hill != mmr


def test_generated_catalog_respects_model_bounds():
    p = hmc.HMCParams()
    n = 5000
    counts, pl = hmc.generate_catalog(p, np.ones(n), np.ones(n), np.full(n, 5780.0), np.full(n, 0.82), seed=7)
    assert counts.sum() == len(pl["period"])
    assert pl["period"].min() >= p.min_period and pl["period"].max() <= p.max_period
    assert pl["radius_init"].min() > p.min_radius and pl["radius_init"].max() < p.max_radius
    assert np.all(pl["ecc"] < 1.0)
    # Systems are sorted by period within each star.
    idx = pl["star_index"]
    same = idx[1:] == idx[:-1]
    assert np.all(np.diff(pl["period"])[same] > 0)


def test_generated_systems_are_hill_stable():
    p = hmc.HMCParams()
    n = 3000
    _, pl = hmc.generate_catalog(p, np.ones(n), np.ones(n), np.full(n, 5780.0), np.full(n, 0.82), seed=3)
    idx = pl["star_index"]
    same = idx[1:] == idx[:-1]
    m = pl["mass_init"] * hmc.EARTH_MASS_MSUN
    a = (hmc._GRAV_CONST / (4 * math.pi**2) * (1 + m) * pl["period"] ** 2) ** (1 / 3)
    rh = 0.5 * (a[1:] + a[:-1]) * ((m[1:] + m[:-1]) / 3) ** (1 / 3)
    assert np.all((a[1:] - a[:-1])[same] >= p.num_mutual_hill_radii * rh[same] * (1 - 1e-9))
