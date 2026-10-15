import numpy as np
import pytest

from starsys.real import surveys as sv


def _cand(**kw):
    n = len(np.atleast_1d(kw.get("period", [1.0])))
    base = {"period": np.full(n, 10.0), "a_au": np.full(n, 0.09), "planet_mass": np.full(n, 1.0),
            "planet_radius": np.full(n, 1.0), "ecc": np.zeros(n), "host_mass": np.full(n, 1.0),
            "r_star": np.full(n, 1.0), "cos_i": np.zeros(n), "system": np.zeros(n, dtype=int)}
    base.update({k: np.asarray(v, dtype=float) if k != "system" else v for k, v in kw.items()})
    return base


def test_semi_amplitude():
    assert sv.semi_amplitude(317.83, 365.25, 1.0) == pytest.approx(28.43, rel=1e-3)
    assert sv.semi_amplitude(1.0, 365.25, 1.0) == pytest.approx(0.0895, rel=0.01)


def test_signal_power():
    t = np.sort(np.random.default_rng(0).uniform(0, 3000, 200))
    w = np.ones(200)
    s = sv.signal_power(t, w, np.zeros(200, int), np.array([3.7, 50.0, 1e5]))
    assert s[0] == pytest.approx(100.0, rel=0.1) and s[1] == pytest.approx(100.0, rel=0.1)  # N K^2 / 2, K = 1
    assert s[2] < 0.05 * s[0]  # far beyond the baseline: absorbed by the offset
    s_trend = sv.signal_power(t, w, np.zeros(200, int), np.array([5000.0]), trend=True)
    assert s_trend[0] < sv.signal_power(t, w, np.zeros(200, int), np.array([5000.0]))[0]


def test_transit_campaign_and_noise():
    assert 1e-4 < sv.tess_noise_1h(10.0) < 3e-4 and sv.tess_noise_1h(14.0) > sv.tess_noise_1h(10.0)
    camp = sv.TransitCampaign(days=20, noise_1h=3e-4)
    # An Earth around a 0.12 Rsun star at 6 d: deep transit, detected when it transits.
    c = _cand(period=[6.1, 6.1], a_au=[0.029, 0.029], r_star=[0.12, 0.12], host_mass=[0.09, 0.09],
              cos_i=[0.0, 0.2])
    p = camp.p_detect(c)
    assert p[0] > 0.99 and p[1] == 0.0


def test_gaia_astrometry():
    g = sv.GaiaAstrometry()
    g._on, g._sigma, g._dist = True, 0.1, 5.0
    c = _cand(period=[365.0, 365.0, 3e5], a_au=[1.0, 1.0, 40.0], planet_mass=[3000.0, 3.0, 3000.0])
    p = g.p_detect(c)
    assert p[0] > 0.99 and p[1] < 0.01 and p[2] < 0.5
