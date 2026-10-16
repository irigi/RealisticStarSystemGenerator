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
    from starsys.view.starmap import GalaxyWalkSource, appearance, effective_mag, magnitude_limit, teff_colour
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
    m = effective_mag(np.array([5.0, 15.0, 5.0]), xyz, np.array([0.0, 0, 50]), 50.0, np.zeros(3))
    size, c = appearance(m, magnitude_limit(m, 50.0), col)
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


def test_artwork_is_rendered_at_the_zoom():
    import time

    from PySide6.QtWidgets import QApplication

    from starsys.view import sysmap
    from starsys.view.app import SequenceSource, SystemWindow
    app = QApplication.instance() or QApplication([])
    w = SystemWindow(SequenceSource(1), 3)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    w.load(3)
    m = w.map
    b = m._art[0][1]
    pm = sysmap.pixmap(b, 40, 2.0)
    assert pm.width() == 80 and pm.devicePixelRatio() == 2.0
    it, _, lpx, n0 = m._art[0]
    m.centerOn(it.sceneBoundingRect().center())
    m.scale(4, 4)
    m.centerOn(it.sceneBoundingRect().center())
    m.sharpen()
    for _ in range(200):
        app.processEvents()
        if not m.loader.pending:
            break
        time.sleep(0.02)
    app.processEvents()
    need = sysmap.render_px(lpx, m.transform().m11() * m.devicePixelRatioF())
    assert need > 1.5 * n0 and it.pixmap().width() == need
    assert abs(it.pixmap().width() / it.pixmap().devicePixelRatio() - lpx) < 1  # same size on the map
    w.close()


def test_star_map_shows_a_budget_of_stars_at_every_zoom():
    from PySide6.QtWidgets import QApplication

    from starsys.view.starmap import StarMapWindow, star_budget, tier_weight
    app = QApplication.instance() or QApplication([])
    w = StarMapWindow(cloud_n=2000)
    w.view.resize(1000, 800)  # not shown (no OpenGL offscreen)
    assert tier_weight(0, 40.0) == 1 and tier_weight(0, 2000.0) == 0 and tier_weight(3, 1e5) == 1
    for d in (50.0, 2000.0, 40000.0):
        w.fly((gx_r0() if d > 10000 else 0.0, 0, 0), d, 40)
        w._last_cam = None
        for _ in range(3000):
            w._tick()
            if not w.queue:
                break
        w._tick()
        while w._ease_limit():
            pass
        shown = int(w.real_vis.sum() + w.proc_vis.sum())
        # (fewer where the faint tiers have faded and the bright ones are sparse)
        assert 0.4 * star_budget(d) <= shown <= 1.02 * star_budget(d), (d, shown)  # limit eased to within 0.02 mag
        if d > 1000:  # no faint tier is shown once the view has outgrown it
            assert not (w.proc_vis & (w.proc["tier"] == 0)).any()
    w.close()


def gx_r0():
    from starsys import galaxy as gx
    return gx.R0_PC


def test_star_sizes_change_smoothly_with_zoom_and_close_views_show_every_star():
    from starsys.view.starmap import appearance, base_size, magnitude_limit, reference_mag
    ds = np.geomspace(5, 50000, 400)
    ref = np.array([reference_mag(d) for d in ds])
    s0 = np.array([base_size(d) for d in ds])
    assert np.abs(np.diff(ref)).max() < 0.1 and np.abs(np.diff(s0)).max() < 0.02
    # a bright star's size does not depend on how many fainter stars there are
    col = np.ones((2, 4), np.float32)
    a, _ = appearance(np.array([3.0, 9.0]), 10.0, col, 1000.0)
    b, _ = appearance(np.array([3.0, 9.0]), 14.0, col, 1000.0)
    assert a[0] == b[0] and a[1] < b[1]
    # close views (Sun to Procyon): no limit short of the faintest dwarfs; Barnard's Star
    # (M_V 13.2, 1.8 pc) a few pixels across
    assert magnitude_limit(np.zeros(10), 10.0) > 25
    size, _ = appearance(np.array([13.2 + 5 * np.log10(10 / 10)]), magnitude_limit(np.zeros(10), 10.0),
                         col[:1], 10.0)
    assert size[0] >= 3.0


def test_stellar_orbits_are_drawn_barycentric():
    m = md.build(ex.system_dict(ids.system("S1-1"), 0))  # a triple: (A+B)+C
    assert len(m.stars) == 3
    for t in (0.0, 1234.5, 5e5):
        pos = md.positions(m, t)
        for s in m.stars:
            path = md.orbit_path(m, s.name, pos, 4000)
            span = np.ptp(path, axis=0).max()
            assert np.linalg.norm(path - np.array(pos[s.name]), axis=1).min() < 0.01 * span
        (com, path), = md.barycentre_paths(m, pos, 4000)
        assert np.linalg.norm(path - com, axis=1).min() < 0.01 * np.ptp(path, axis=0).max()
        # A and B on opposite sides of their centre of mass, which C and A+B straddle in turn
        mass = {s.name: s.d["mass"] for s in m.stars}
        ab = (mass["A"] * np.array(pos["A"]) + mass["B"] * np.array(pos["B"])) / (mass["A"] + mass["B"])
        assert np.allclose(ab, com)
        assert np.dot(np.array(pos["A"]) - ab, np.array(pos["B"]) - ab) < 0
        assert np.dot(ab, np.array(pos["C"])) < 0


def test_solar_system_is_real():
    from starsys import realstars as rs
    from starsys.view.app import passes
    b = rs.real_system("Sun")
    d = b.system_dict(0)
    m = md.build(d)
    planets = [n for n in m.order if m.bodies[n].cat == "planet"]
    assert planets == ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    assert abs(m.bodies["Earth"].orbit[3] - 365.256) < 0.01
    assert m.bodies["Earth"].d["habitable_zone"] == "conservative HZ"
    assert [c for c in m.bodies["Jupiter"].children] == ["Io", "Europa", "Ganymede", "Callisto"]
    assert m.bodies["Io"].d["heat_flux_w_m2"] > 1  # drawn as volcanic
    # J2000: the Sun's geocentric longitude is 280.46 deg, so Earth is at ~100.5 deg heliocentric
    x, y = md.positions(m, 0.0)["Earth"]
    assert abs(np.degrees(np.arctan2(y, x)) % 360 - 100.5) < 2.5
    assert passes(b, 0, {"multiple": True}) and b.json_obj()["systems"][0]["system_id"] == "R:Sun"


def test_orbit_view_keeps_bodies_on_their_drawn_orbits_at_any_zoom():
    from PySide6.QtWidgets import QApplication

    from starsys.view.orbits import OrbitView
    app = QApplication.instance() or QApplication([])  # noqa: F841
    m = md.build(ex.system_dict(ids.system("S1-1"), 0))
    v = OrbitView()
    v.resize(800, 600)
    v.set_system(m)
    v.t = 777.0
    pos = md.positions(m, v.t)
    for log in (False, True):
        v.log = log
        for zoom in (1.0, 1e3, 1e6):
            v.fit()
            v.scale *= zoom if not log else 1.0
            if log:
                v.r0 /= zoom
            for s in m.stars:
                v.centre = np.array(pos[s.name])
                pts = v.refined(lambda E: md.orbit_path(m, s.name, pos, E))
                q = v.to_screen(np.array(pos[s.name]))[0]
                # distance from the body to the polyline (segments)
                a, b = pts[:-1], pts[1:]
                ab = b - a
                t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
                d = np.hypot(*(a + t[:, None] * ab - q).T).min()
                assert d < 0.5, (s.name, log, zoom, d)
