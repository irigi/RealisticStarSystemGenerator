import numpy as np
import pytest

from starsys import derived as dv
from starsys import planets


def _system(lum, teff, mass, age=4.6, a_comp=None, ecc_comp=None):
    """One system; components given as lists (role order)."""
    k = len(lum)
    comp = {"system": np.zeros(k, int), "role": np.arange(k), "luminosity": np.array(lum, float),
            "teff": np.array(teff, float), "mass": np.array(mass, float), "age_gyr": np.full(k, age),
            "a_au": np.array([np.nan] + list(a_comp or []), float), "ecc": np.array([np.nan] + list(ecc_comp or []), float)}
    return {"n_components": np.array([k])}, comp


def _planets(a, m, r, e=None, host_mass=1.0, kind="S", lost=None):
    n = len(a)
    return {"system": np.zeros(n, int), "component": np.zeros(n, int), "host_kind": np.full(n, kind, dtype=object),
            "host_mass": np.full(n, host_mass), "a_au": np.array(a, float), "ecc": np.zeros(n) if e is None else np.array(e),
            "planet_mass": np.array(m, float), "planet_radius": np.array(r, float),
            "lost_envelope": np.zeros(n, bool) if lost is None else np.array(lost)}


def test_kopparapu_coefficients_for_the_sun():
    s = dv.s_eff(5780.0, 1.0)
    assert s["recent_venus"] == pytest.approx(1.776) and s["runaway"] == pytest.approx(1.107)
    assert s["max_greenhouse"] == pytest.approx(0.356) and s["early_mars"] == pytest.approx(0.32)
    assert dv.s_eff(5780.0, 5.0)["runaway"] == pytest.approx(1.188)
    assert dv.s_eff(5780.0, 0.1)["runaway"] == pytest.approx(0.99)
    assert dv.s_eff(3000.0, 1.0)["runaway"] < 1.107  # cooler stars: HZ at lower flux


def test_solar_system():
    systems, comp = _system([1.0], [5772.0], [1.0])
    names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]
    pl = _planets([0.387, 0.723, 1.0, 1.524, 5.20], [0.055, 0.815, 1.0, 0.107, 317.8], [0.383, 0.949, 1.0, 0.532, 11.2],
                  e=[0.206, 0.007, 0.017, 0.093, 0.049])
    out = dv.derive_properties(pl, systems, comp)
    got = dict(zip(names, out["t_eq"]))
    assert got["Earth"] == pytest.approx(254.6, abs=1.0)
    assert out["insolation"][2] == pytest.approx(1.0, rel=1e-3)
    assert dict(zip(names, out["hz"])) == {"Mercury": dv.HZ_TOO_HOT, "Venus": dv.HZ_TOO_HOT,
                                           "Earth": dv.HZ_CONSERVATIVE, "Mars": dv.HZ_CONSERVATIVE,
                                           "Jupiter": dv.HZ_TOO_COLD}
    spin = dict(zip(names, out["spin"]))
    assert spin["Mercury"] == dv.SPIN_32 and spin["Earth"] == dv.SPIN_FREE and spin["Venus"] == dv.SPIN_FREE
    assert 0.1 < out["t_lock_gyr"][0] < 4.6 and out["t_lock_gyr"][2] > 50
    assert out["rotation_h"][0] == pytest.approx(1407.5, rel=0.01)  # Mercury, 3:2
    assert 2.0 < out["rotation_h"][2] < 5.0  # primordial (Kokubo & Genda 2010); no Moon in this model
    assert out["rotation_h"][4] == pytest.approx(10.0)  # nominal giant spin
    assert out["obliquity_deg"][0] == 0.0 and np.isnan(out["obliquity_deg"][2])
    assert not out["hz_flags"].any()


def test_m_dwarf_planets_in_the_hz_are_locked():
    # Proxima Cen b and TRAPPIST-1 e, f, g.
    systems, comp = _system([0.00155], [3042.0], [0.122], age=4.8)
    pl = _planets([0.0485], [1.07], [1.05], host_mass=0.122)
    out = dv.derive_properties(pl, systems, comp)
    assert out["insolation"][0] == pytest.approx(0.66, rel=0.05)
    assert out["hz"][0] == dv.HZ_CONSERVATIVE and out["spin"][0] == dv.SPIN_SYNC
    systems, comp = _system([0.000553], [2566.0], [0.0898], age=7.6)
    pl = _planets([0.02925, 0.03849, 0.04683], [0.69, 1.04, 1.32], [0.92, 1.05, 1.13], host_mass=0.0898)
    out = dv.derive_properties(pl, systems, comp)
    assert (out["hz"] == dv.HZ_CONSERVATIVE).all() and (out["spin"] == dv.SPIN_SYNC).all()
    assert (out["hz_flags"] == planets.FLAG_HZ_EXTRAP).all()  # 2566 K is below Kopparapu's 2600 K


def test_primordial_spins_and_obliquities():
    rng = np.random.default_rng(0)
    n = 20_000
    kind = np.repeat(np.array([0, 2], dtype=np.int8), n)
    m = np.repeat([1.0, 317.8], n)
    r = np.repeat([1.0, 11.2], n)
    p0, ob = dv.primordial_spin(m, r, kind, rng=rng)
    rocky = kind == 0
    w = 2 * np.pi / (p0[rocky] * 3600.0)
    assert np.sqrt(np.mean(w**2)) * 3600.0 == pytest.approx(2.35, rel=0.03)  # Kokubo & Genda 2010
    assert (w <= dv.breakup_rate(1.0, 1.0) * (1 + 1e-9)).all()
    assert np.mean(np.cos(np.radians(ob[rocky]))) == pytest.approx(0.0, abs=0.02)  # isotropic
    assert np.median(p0[~rocky]) == pytest.approx(10.0, rel=0.03) and (ob[~rocky] < 90).mean() > 0.99


def test_rotation_with_a_generator():
    systems, comp = _system([1.0], [5772.0], [1.0])
    pl = _planets([0.05, 1.0, 5.2], [1.0, 1.0, 317.8], [1.0, 1.0, 11.2])
    out = dv.derive_properties(pl, systems, comp, rng=np.random.default_rng(1))
    p_orb_h = 2 * np.pi * np.sqrt((pl["a_au"] * 1.495978707e11) ** 3 / (6.674e-11 * 1.989e30)) / 3600
    assert out["spin"][0] == dv.SPIN_SYNC and out["rotation_h"][0] == pytest.approx(p_orb_h[0], rel=1e-3)
    assert out["spin"][1] == dv.SPIN_FREE and out["rotation_h"][1] < p_orb_h[1]
    assert np.isfinite(out["obliquity_deg"]).all() and out["obliquity_deg"][0] == 0.0


def test_early_runaway_greenhouse():
    # A late M dwarf (Luger & Barnes 2015): mid-HZ planets spend tens to hundreds of Myr in a runaway greenhouse.
    comp = {"system": np.zeros(2, int), "role": np.arange(2), "kind": np.array(["star", "star"], dtype=object),
            "initial_mass": np.array([0.15, 1.0]), "feh": np.zeros(2), "age_gyr": np.array([5.0, 4.6])}
    pl = {"component": np.array([0, 0, 1]), "planet_mass": np.array([1.0, 1.0, 1.0])}
    t = dv.runaway_time(pl, comp, np.array([0.6, 0.05, 1.0]))  # mid-HZ and far out of the M dwarf; the Earth
    assert 10.0 < t[0] < 1000.0 and t[1] == 0.0 and t[2] == 0.0
    assert np.isnan(dv.runaway_time(pl, {"age_gyr": comp["age_gyr"]}, np.ones(3))).all()


def test_binary_light():
    # Alpha Cen A with B at 23.5 au: B adds ~0.1% to an Earth analogue around A.
    systems, comp = _system([1.52, 0.50], [5790.0, 5260.0], [1.1, 0.9], a_comp=[23.5], ecc_comp=[0.52])
    pl = _planets([1.23], [1.0], [1.0], host_mass=1.1)
    out = dv.derive_properties(pl, systems, comp)
    assert out["insolation"][0] == pytest.approx(1.52 / 1.23**2 + 0.50 / (23.5**2 * np.sqrt(1 - 0.52**2)), rel=1e-9)
    # A circumbinary planet sees both stars from the barycentre.
    systems, comp = _system([1.0, 0.25], [5800.0, 4500.0], [1.0, 0.7], a_comp=[0.2], ecc_comp=[0.1])
    pl = _planets([1.5], [100.0], [8.0], host_mass=1.7, kind="P")
    out = dv.derive_properties(pl, systems, comp)
    assert out["insolation"][0] == pytest.approx(1.25 / 1.5**2)


def test_attach_planets_adds_columns():
    n = 300
    comp = {"system": np.arange(n), "role": np.zeros(n, int), "kind": np.full(n, "star", dtype=object),
            "phase": np.zeros(n), "mass": np.ones(n), "radius": np.ones(n), "teff": np.full(n, 5772.0),
            "age_gyr": np.full(n, 4.6), "feh": np.zeros(n), "luminosity": np.ones(n),
            "a_au": np.full(n, np.nan), "ecc": np.full(n, np.nan)}
    systems = {"age_gyr": np.full(n, 4.6), "n_components": np.ones(n, int)}
    pl = planets.attach_planets(systems, comp, seed=1)
    for col in ("insolation", "t_eq", "t_eq0", "hz", "t_lock_gyr", "spin"):
        assert len(pl[col]) == len(pl["period"])
    assert np.allclose(pl["insolation"], 1.0 / (pl["a_au"] ** 2 * np.sqrt(1 - pl["ecc"] ** 2)))
    bare = planets.attach_planets(systems, comp, seed=1, derived=None)
    assert "insolation" not in bare
    assert all(np.array_equal(bare[c], pl[c]) for c in bare)
