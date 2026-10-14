import numpy as np
import pytest

from starsys import binaries as bn
from starsys import planets
from starsys.inner import hmc
from starsys.inner.hmc import HMCParams
from starsys.stars.population import draw_systems


@pytest.fixture(scope="module")
def population():
    systems, comp = draw_systems(6000, seed=21)
    return systems, comp, planets.attach_planets(systems, comp, seed=22, evolved=None)


def test_holman_wiegert_limits():
    # HW99 Table 3 / eq. 1 (S-type) and Table 7 / eq. 3 (P-type) at e = 0.
    assert bn.hw_s_type(1.0, 0.0, 0.5) == pytest.approx(0.274, abs=1e-3)
    assert bn.hw_s_type(1.0, 0.0, 0.1) == pytest.approx(0.426, abs=1e-3)
    assert bn.hw_p_type(1.0, 0.0, 0.5) == pytest.approx(2.39, abs=0.01)
    # Eccentric binaries: tighter S-type and wider P-type limits; continuous beyond the fit.
    assert bn.hw_s_type(1.0, 0.5, 0.3) < bn.hw_s_type(1.0, 0.0, 0.3)
    assert bn.hw_p_type(1.0, 0.5, 0.3) > bn.hw_p_type(1.0, 0.0, 0.3)
    assert bn.hw_s_type(1.0, 0.8, 0.3) == pytest.approx(bn.hw_s_type(1.0, 0.8 - 1e-9, 0.3), rel=1e-6)
    assert bn.hw_s_type(1.0, 0.99, 0.3) > 0 and bn.hw_s_type(1.0, 0.0, 0.99) > 0


def test_suppression_factor():
    s = bn.s_bin([0.5, 1.0, 10.0, 200.0, 5000.0, np.nan])
    assert s == pytest.approx([0.0, 0.0, 0.15, 1.0, 1.0, 1.0])
    a = np.geomspace(1.0, 200.0, 50)
    assert np.all(np.diff(bn.s_bin(a)) > 0)


def test_boost_preserves_expected_planet_count():
    rows = np.tile(HMCParams().as_array(), (3, 1))
    colour = np.array([0.6, 0.81, 1.5])  # f_swpa < 1, at the median colour, clipped at 1
    out, rest = bn.apply_boost(rows, colour, np.array([1.3, 1.3, 1.3]))

    def expected(r):
        f = np.clip(r[:, hmc.I_FSLOPE] * (colour - r[:, hmc.I_MEDCOL]) + r[:, hmc.I_FMED], 0, 1)
        lam = np.exp(r[:, hmc.I_LRC])
        return f * lam / -np.expm1(-lam)

    assert expected(out) == pytest.approx(1.3 * expected(rows), rel=1e-6)
    assert rest[0] == pytest.approx(1.0) and rest[2] == pytest.approx(1.3)


def test_planets_respect_dynamical_limits(population):
    systems, comp, pl = population
    st = bn.structure(comp)
    s = pl["host_kind"] == "S"
    assert np.all(pl["a_au"][s] <= st["a_max"][pl["component"][s]] * (1 + 1e-9))
    p = ~s
    assert p.any()
    assert np.all(pl["a_au"][p] >= st["pair_a_min"][pl["component"][p]] * (1 - 1e-9))
    assert np.all(pl["a_au"][p] <= st["pair_a_max"][pl["component"][p]] * (1 + 1e-9))
    assert ((pl["flags"][p] & planets.FLAG_CIRCUMBINARY) > 0).all()
    # Circumbinary HM-C planets >= 4 Earth radii are replaced by the empirical component.
    cbm = pl["model"][p]
    assert not np.any(np.isin(cbm, ["hmc", "hmc_m"]) & (pl["planet_radius"][p] >= planets.CB_SPLIT_RADIUS))
    assert np.all(pl["planet_radius"][p][cbm == "circumbinary"] >= 4.0)


def test_close_binaries_lose_planets(population):
    systems, comp, pl = population
    st = bn.structure(comp)
    has = np.bincount(pl["component"][pl["host_kind"] == "S"], minlength=len(comp["mass"])) > 0
    ms = planets.hmc_hosts(comp) & (comp["kind"] == "star")
    single = ms & ~np.isfinite(st["a_near"])
    close = ms & (st["a_near"] < 1.0)
    assert has[single].mean() > 0.5
    assert has[close].sum() == 0
