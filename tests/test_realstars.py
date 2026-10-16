import numpy as np
import pytest

from starsys import galaxy as gx
from starsys import realstars as rs

pytestmark = pytest.mark.skipif(not rs.HIP_FILE.exists(), reason="Hipparcos catalogue not downloaded")


def test_blend_replaces_exactly_what_the_catalogues_hold():
    d_real = np.linalg.norm(rs.xyz(), axis=1)
    n_cat = n_real = 0
    for lo, hi in ((25, 40), (40, 55)):
        for tier in [0, *gx.TIERS]:  # the bright stars are in the brightness-tier cells
            for k in gx.cells_within(tier, (0, 0, 0), hi):
                sec = gx.cell(tier, *k)
                d = np.linalg.norm(sec.xyz, axis=1)
                n_cat += int((rs.in_catalogue(sec) & (d > lo) & (d < hi)).sum())
        n_real += int(((d_real > lo) & (d_real < hi) & (rs.stars()["source"] == "hipparcos").to_numpy()).sum())
    assert n_cat == pytest.approx(n_real, rel=0.1)
    # within the census volume every procedural system is replaced
    sec = gx.sector(0, 0, 0)
    assert rs.in_catalogue(sec).all() and len(rs.procedural_kept((0, 0, 0))) == 0


def test_search_and_magnitudes():
    s = rs.stars()
    for q, name in (("Vega", "Vega"), ("Sirius", "Sirius"), ("HIP 91262", "Vega"), ("Deneb", "Deneb")):
        assert s["name"].iloc[rs.search(q)[0]] == name
    sun_like = rs.abs_v(np.array([5772.0]), np.array([1.0]))[0]
    assert sun_like == pytest.approx(4.81, abs=0.1)
