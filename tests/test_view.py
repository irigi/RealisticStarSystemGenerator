import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from starsys import export as ex  # noqa: E402
from starsys import ids  # noqa: E402
from starsys.view import art  # noqa: E402
from starsys.view import model as md  # noqa: E402


def test_art_is_deterministic_and_varied():
    giant = {"type": "giant", "t_eq_k": 120, "rotation_h": 10, "mass": 300}
    a, b = art.render(giant, 48, 5), art.render(giant, 48, 5)
    assert a.shape == (48, 48, 4) and np.array_equal(a, b)
    assert not np.array_equal(a, art.render(giant, 48, 6))
    for body in ({"teff": 5800, "kind": "star"}, {"teff": 0, "kind": "black hole"},
                 {"type": "rocky", "climate": "temperate", "t_eq_k": 255, "mass": 1},
                 {"type": "envelope", "t_eq_k": 80}, {"composition": "ice", "heat_flux_w_m2": 0.01}):
        img = art.render(body, 32, 1)
        assert img[16, 16, 3] == 255  # opaque centre
    r, g, bl = art.star_rgb(3000)
    assert r > g > bl  # cool stars are red-orange
    r, g, bl = art.star_rgb(20000)
    assert bl > r


def test_model_positions_follow_kepler():
    m = md.build(ex.system_dict(ids.system("S1-4"), 0))
    assert m.order[0] == "A" and "AB b" in m.bodies and m.bodies["AB b"].parent == "A+B"
    p0 = md.positions(m, 0.0)
    per = m.bodies["AB b"].orbit[3]
    p1 = md.positions(m, per)
    assert np.allclose(p0["AB b"], p1["AB b"], atol=1e-2 * m.bodies["AB b"].orbit[0])
    # barycentre at the origin: stars weighted by mass cancel
    w = {s.name: s.d["mass"] for s in m.stars}
    com = sum(w[n] * np.array(p0[n]) for n in w) / sum(w.values())
    assert np.allclose(com, 0, atol=1e-9)
    e = md.kepler(np.array([0.3, 2.0]), np.array([0.5, 0.9]))
    assert np.allclose(e - np.array([0.5, 0.9]) * np.sin(e), [0.3, 2.0])


def test_window_navigation_and_selection(tmp_path):
    from PySide6.QtWidgets import QApplication

    from starsys.view.app import SequenceSource, SystemWindow
    app = QApplication.instance() or QApplication([])
    w = SystemWindow(SequenceSource(1), 3)
    w.show()
    app.processEvents()
    w.load(3)
    first = w.model.d["stars"][0]["mass"]
    w.step(+1)
    assert w.k == 4 and w.id_edit.text() == "S1-4"
    w.step(-1)
    assert w.k == 3 and w.model.d["stars"][0]["mass"] == first
    w.filter.setCurrentText("multiple stars")
    w.step(+1)
    assert w.model.d["n_stars"] > 1
    name = w.model.order[-1]
    w.select(name)
    assert name in w.info.toHtml() or w.model.bodies[name].short in w.info.toHtml()
    w.id_edit.setText("S1-75")
    w.goto_text()
    assert w.k == 75
    assert not w.grab().isNull()
    w.close()


def test_galaxy_walk_and_star_appearance():
    from PySide6.QtWidgets import QApplication

    from starsys import galaxy as gx
    from starsys import realstars as rs
    from starsys.view.app import SystemWindow
    from starsys.view.starmap import GalaxyWalkSource, appearance, teff_colour
    app = QApplication.instance() or QApplication([])
    sec = gx.sector(20, 5, 0)
    src = GalaxyWalkSource(sec.sid(0))
    seen = []
    w = SystemWindow(src, 0, on_load=lambda sid, b, i: seen.append(sid))
    w.load(0)
    assert w.id_edit.text() == sec.sid(0) and not w.prev_b.isEnabled()
    w.step(+1)
    w.step(+1)
    w.step(-1)
    w.step(-1)
    assert w.k == 0 and seen[0] == seen[-1] == sec.sid(0) and seen[1] == seen[3] != seen[0]
    # the walk is fixed by its first star
    assert GalaxyWalkSource(sec.sid(0)).label(2) == seen[2]
    assert src.index_of(seen[1]) == 1 and src.index_of(sec.sid(3)) == 0 and src.label(0) == sec.sid(3)
    assert rs.real_seed("HD 197345") == rs.real_seed("HD 197345") != rs.real_seed("HD 10700")
    w.close()
    app.processEvents()
    col = teff_colour(np.array([3000.0, 6000.0, 20000.0]))
    assert col[0, 0] > col[0, 2] and col[2, 2] > col[2, 0]
    xyz = np.array([[0.0, 0, 0], [5, 0, 0], [500, 0, 0]])
    size, c = appearance(np.array([5.0, 15.0, 5.0]), xyz, np.array([0.0, 0, 50]), 50.0, col)
    assert size[0] > size[1] and size[2] == 0  # bright near star big, far star outside the window hidden


def test_star_map_mouse_centre():
    from PySide6.QtWidgets import QApplication

    from starsys.view.starmap import StarMapWindow
    app = QApplication.instance() or QApplication([])
    w = StarMapWindow(cloud_n=2000)
    v = w.view
    v.resize(1000, 800)  # not shown: the offscreen platform has no OpenGL; the projection is pure maths
    w.fly((10, 20, 0), 100, elevation=30)
    cx, cy = v.width() / 2, v.height() / 2
    assert np.allclose(v.plane_point(cx, cy, 0.0), [10, 20, 0], atol=0.5)
    p = v.plane_point(cx + 150, cy - 80, 0.0)
    assert abs(p[2]) < 1e-6
    sc, ok = v.project(p[None, :])
    assert ok[0] and np.allclose(sc[0], [cx + 150, cy - 80], atol=1.0)
    w.double_click(cx + 150, cy - 80)  # empty place (or a star near it): the centre moves there
    for _ in range(40):
        app.processEvents()
        if w._flight is None:
            break
        import time
        time.sleep(0.03)
    assert np.linalg.norm(w._centre() - (10, 20, 0)) > 5
    w.close()
