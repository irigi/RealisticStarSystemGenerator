import numpy as np
import pytest

from starsys import planets
from starsys.inner import hmc
from starsys.inner import short as sh
from starsys.units import EARTH_MASS_MSUN, period_to_a_au


def test_rates_and_teff_trend():
    # Sun-like star, 0.5-3 d: Hsu+2019 below 2 d plus the 2-3 d residual.
    n = sh.expected_number(np.array([3.0]), np.array([5700.0]))[0]
    assert 0.03 < n < 0.07
    assert sh.expected_number(np.array([0.5]), np.array([5700.0]))[0] == 0.0
    s = sh.teff_scale(np.array([3500.0, 4500.0, 5500.0, 6500.0]))
    assert np.all(np.diff(s) < 0)  # USPs are more common around cooler stars
    assert s[3] == pytest.approx(0.15 / sh.SO2014_KEPLER_MEAN)


def _sun_like(n):
    comp = {"system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, "star", dtype=object),
            "phase": np.zeros(n), "mass": np.ones(n), "radius": np.ones(n), "teff": np.full(n, 5772.0),
            "age_gyr": np.full(n, 4.6), "feh": np.zeros(n), "luminosity": np.ones(n)}
    return {"age_gyr": np.full(n, 4.6), "n_components": np.ones(n, int)}, comp


def test_short_period_planets_fill_inside_hmc():
    systems, comp = _sun_like(20_000)
    pl = planets.attach_planets(systems, comp, seed=4, giants=None, small=None)
    s = pl["model"] == "short_period"
    assert 0.02 * 20_000 < s.sum() < 0.08 * 20_000
    assert np.all((pl["period"][s] >= 0.5) & (pl["period"][s] < 3.0))
    assert ((pl["flags"][s] & planets.FLAG_SHORT_PERIOD) > 0).all()
    assert not ((pl["flags"][~s] & planets.FLAG_SHORT_PERIOD) > 0).any()
    assert np.all(pl["ecc"][s] == 0.0) and np.median(pl["planet_radius"][s]) < 1.5
    # Spacing from the next planet out (the innermost HM-C planet or another short-period planet).
    nxt = np.flatnonzero(s)[:-1] + 1
    nxt = nxt[pl["component"][nxt] == pl["component"][nxt - 1]]
    a1 = period_to_a_au(pl["period"][nxt - 1], 1.0)
    a2 = period_to_a_au(pl["period"][nxt], 1.0)
    r_h = ((pl["mass_init"][nxt - 1] + pl["mass_init"][nxt]) * EARTH_MASS_MSUN / 3.0) ** (1 / 3) * 0.5 * (a1 + a2)
    spaced = (a2 - a1) >= hmc.HMCParams().num_mutual_hill_radii * r_h * 0.999
    inner = pl["model"][nxt] == "short_period"
    assert spaced[inner].all()
    # Without the component, nothing inside HM-C's 3 d.
    plain = planets.attach_planets(systems, comp, seed=4, giants=None, small=None, short=None)
    assert plain["period"].min() >= 3.0
