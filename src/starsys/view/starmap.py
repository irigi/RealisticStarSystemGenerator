"""3D star map of the procedural Milky Way with the real stars blended in.

Layers (heliocentric Galactic coordinates in parsecs, see :mod:`starsys.galaxy`):

* the galaxy as a point cloud sampled from the density model (far view; fades out near stars);
* the real stars (:mod:`starsys.realstars`: 20-pc census + Hipparcos Survey);
* the procedural systems of the sectors around the view centre (level of detail: loaded a
  few sectors per timer tick, only within a radius that shrinks as the camera closes in),
  minus those the catalogues already hold.

Mouse: left-drag rotates, middle-drag (or Ctrl+left) pans, wheel zooms, click selects,
double-click centres on a star. Stars are coloured by temperature and sized by absolute
magnitude (brighter = bigger).
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QVector3D
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QPushButton, QSplitter, QStatusBar, QTextBrowser, QVBoxLayout, QWidget)

from .. import galaxy as gx
from .. import ids
from .. import realstars as rs
from . import info as info_mod
from .art import star_rgb
from .sysmap import BG

_T = np.geomspace(1500, 40000, 200)
_RGB = np.array([star_rgb(t) for t in _T])


def teff_colour(teff: np.ndarray, alpha=1.0) -> np.ndarray:
    t = np.clip(np.nan_to_num(np.asarray(teff, float), nan=5000.0), _T[0], _T[-1])
    i = np.clip(np.searchsorted(_T, t), 0, len(_T) - 1)
    rgb = _RGB[i]
    rgb = np.clip(rgb.mean(1, keepdims=True) + 2.4 * (rgb - rgb.mean(1, keepdims=True)), 0, 1)
    return np.concatenate([rgb, np.full((len(t), 1), alpha)], axis=1).astype(np.float32)


def mag_size(abs_v: np.ndarray) -> np.ndarray:
    """Point size (px) from absolute V magnitude."""
    return np.clip(7.5 - 0.42 * np.nan_to_num(abs_v, nan=10.0), 1.5, 14.0).astype(np.float32)


def appearance(abs_v, xyz, eye, view_dist, base_col, centre=None):
    """Sizes and colours of stars as seen from the camera: apparent magnitude from the eye
    position against a limit that follows the zoom, and a spatial window around the view
    centre (a map, not a sky: only the region being looked at is shown in full)."""
    d = np.maximum(np.linalg.norm(xyz - eye, axis=1), 0.05)
    m = np.nan_to_num(abs_v, nan=12.0) + 5 * np.log10(d / 10.0)
    m_lim = 8.5 + 5 * math.log10(max(view_dist, 1.0) / 10.0)
    x = m_lim - m
    if centre is not None:
        rc = np.linalg.norm(xyz - centre, axis=1)
        x = x - np.clip((rc / view_dist - 0.5) * 6.0, 0.0, None) ** 1.5
    size = np.clip(1.0 + 0.5 * x, 0.0, 11.0).astype(np.float32)
    col = base_col.copy()
    col[:, 3] = np.clip(x / 3.0, 0.0, 1.0) * base_col[:, 3]
    size[x <= 0] = 0.0
    return size, col


class MapView(gl.GLViewWidget):
    picked = Signal(float, float)
    doublePicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._press = None
        self.label_xyz = np.zeros((0, 3))
        self.label_text: list[str] = []

    def paintGL(self):
        super().paintGL()
        if not len(self.label_xyz):
            return
        from PySide6.QtGui import QPainter
        sc, ok = self.project(self.label_xyz)
        p = QPainter(self)
        font = QFont()
        font.setPointSizeF(8.5)
        p.setFont(font)
        p.setPen(QColor(175, 190, 215))
        for (x, y), good, t in zip(sc, ok, self.label_text):
            if good:
                p.drawText(int(x) + 6, int(y) - 4, t)
        p.end()

    def mousePressEvent(self, ev):
        self._press = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._press is not None and (ev.position() - self._press).manhattanLength() < 4:
            self.picked.emit(ev.position().x(), ev.position().y())
        self._press = None

    def mouseDoubleClickEvent(self, ev):
        self.doublePicked.emit(ev.position().x(), ev.position().y())

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen coordinates (px) of points and a mask of those in front of the camera."""
        vp = self.getViewport()
        m = self.projectionMatrix(vp, vp) * self.viewMatrix()
        a = np.array(m.data(), dtype=np.float64).reshape(4, 4).T  # column-major
        h = np.c_[xyz, np.ones(len(xyz))] @ a.T
        w = h[:, 3]
        ok = w > 1e-9
        ndc = h[:, :2] / np.where(ok, w, 1.0)[:, None]
        sx = (ndc[:, 0] + 1) / 2 * self.width()
        sy = (1 - ndc[:, 1]) / 2 * self.height()
        return np.stack([sx, sy], 1), ok & (np.abs(ndc) <= 1.05).all(1)


class _Conditioner(QObject):
    done = Signal(object, str)
    failed = Signal(str)

    def __init__(self, lookup, name):
        super().__init__()
        self.lookup, self.name = lookup, name

    def run(self):
        try:
            self.done.emit(rs.real_system(self.lookup, self.name), self.name)
        except Exception as e:  # shown in the status bar
            self.failed.emit(f"{type(e).__name__}: {e}")


class GalaxyWalkSource:
    """Previous / Next for the system viewer opened from the map: system 0 is the star that was
    opened (real or procedural), systems 1, 2, ... are random systems of the galaxy
    (:func:`starsys.galaxy.random_system`) from a seed fixed by the first star, so Previous
    always walks back the same way."""

    def __init__(self, origin: str, real: tuple[str, str] | None = None):
        """``origin``: the first system's ID (a galaxy ID, or ``R:<name>`` for a real star with
        ``real = (catalogue lookup, name)``)."""
        self._set(origin, real)

    def _set(self, origin, real):
        import zlib
        self.origin, self.real = origin, real
        self.seed = ids.mix(gx.GALAXY_SEED, zlib.crc32(origin.encode()))
        self._sids: dict[int, str] = {0: origin}

    def __len__(self):
        return 10**12

    def sid(self, k: int) -> str:
        if k not in self._sids:
            self._sids[k] = gx.random_system(ids.mix(self.seed, k))
        return self._sids[k]

    def get(self, k):
        if k == 0 and self.real is not None:
            return rs.real_system(*self.real), 0
        return gx.system(self.sid(k)), 0

    def label(self, k):
        return self.sid(k)

    def index_of(self, text):
        t = text.strip()
        for k, v in self._sids.items():
            if v == t:
                return k
        try:
            gx.parse(t)
        except ValueError:
            return None
        self._set(t, None)  # a new walk from the typed system
        return 0


class StarMapWindow(QMainWindow):
    def __init__(self, cloud_n: int = 150000):
        super().__init__()
        from .app import DARK
        self.setWindowTitle("starsys — galaxy map")
        self.resize(1500, 920)
        self.setStyleSheet(DARK)
        self.view = MapView()
        self.view.setBackgroundColor(QColor(BG))
        self.view.opts["fov"] = 50
        self.view.picked.connect(self.pick)
        self.view.doublePicked.connect(lambda x, y: (self.pick(x, y), self.centre_on_selection()))

        # galaxy cloud
        pts, comp, arm = gx.sample_cloud(cloud_n)
        self.cloud_xyz = pts.astype(np.float32)
        col = np.empty((len(pts), 4), np.float32)
        warm = np.array([1.0, 0.82, 0.55], np.float32)
        white = np.array([0.95, 0.92, 0.85], np.float32)
        blue = np.array([0.55, 0.7, 1.0], np.float32)
        col[:, :3] = np.where((comp == 0)[:, None], white, warm)
        young = (comp == 0) & (arm > 0.55)
        col[young, :3] = blue
        col[:, 3] = np.where(comp == 2, 0.12, np.where(comp == 0, 0.4, 0.25))
        self.cloud_col = col
        self.cloud = gl.GLScatterPlotItem(pos=self.cloud_xyz, color=col, size=2.0, pxMode=True)
        self.cloud.setGLOptions("additive")
        self.view.addItem(self.cloud)
        # arm centre lines (faint)
        for name, xy in gx.arm_lines():
            line = np.c_[xy, np.full(len(xy), -gx.Z_SUN_PC)].astype(np.float32)
            it = gl.GLLinePlotItem(pos=line[::10], color=(0.3, 0.45, 0.8, 0.25), width=1, antialias=True)
            it.setGLOptions("additive")
            self.view.addItem(it)

        # real stars
        s = rs.stars()
        self.real = s
        self.real_xyz = s[["x", "y", "z"]].to_numpy(np.float32)
        d = np.maximum(np.linalg.norm(self.real_xyz, axis=1), 1e-3)
        absv = s["vmag"].to_numpy(float) - 5 * np.log10(d / 10) - rs.extinction(self.real_xyz.astype(float))
        absv[0] = 4.83  # the Sun
        self.real_absv = absv
        self.real_col = teff_colour(s["teff"].to_numpy(float))
        self.real_size = mag_size(absv)
        self.real_item = gl.GLScatterPlotItem(pos=self.real_xyz, color=self.real_col, size=self.real_size, pxMode=True)
        self.real_item.setGLOptions("additive")
        self.real_item.setGLOptions("additive")
        self.view.addItem(self.real_item)

        # procedural neighbourhood
        self.loaded: dict[tuple, tuple] = {}
        self.queue: list[tuple] = []
        self.proc_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=0.0, pxMode=True)
        self.proc_item.setGLOptions("additive")
        self.view.addItem(self.proc_item)
        self.proc = {"xyz": np.zeros((0, 3), np.float32), "sid": [], "absv": np.zeros(0), "teff": np.zeros(0),
                     "spt": [], "n": np.zeros(0, int), "pop": []}

        self.sel_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=0.0, color=(0.4, 1, 0.6, 1),
                                             pxMode=True)
        self.view.addItem(self.sel_item)
        self.selection = None
        self.drops = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode="lines", color=(0.3, 0.4, 0.6, 0.2))
        self.drops.setGLOptions("additive")
        self.view.addItem(self.drops)
        self.grid = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode="lines", color=(0.3, 0.4, 0.6, 0.3))
        self.grid.setGLOptions("additive")
        self.view.addItem(self.grid)
        self.hud_scale = 10.0

        # side panels
        self.search = QLineEdit()
        self.search.setPlaceholderText("search: Vega, tau Ceti, HIP 91262, G0.0.0-3 …")
        self.search.returnPressed.connect(self.do_search)
        self.results = QListWidget()
        self.results.itemClicked.connect(lambda it: self.select_real(it.data(Qt.ItemDataRole.UserRole), True))
        b_sun = QPushButton("Sun")
        b_sun.clicked.connect(lambda: self.fly((0, 0, 0), 60))
        b_gal = QPushButton("Whole galaxy")
        b_gal.clicked.connect(lambda: self.fly((gx.R0_PC, 0, 0), 45000, elevation=55))
        b_hood = QPushButton("1 kpc")
        b_hood.clicked.connect(lambda: self.fly(self._centre(), 2500))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Find a star"))
        ll.addWidget(self.search)
        ll.addWidget(self.results, 1)
        row = QHBoxLayout()
        for b in (b_sun, b_hood, b_gal):
            row.addWidget(b)
        ll.addLayout(row)
        legend = QLabel("<small>Bright points: real stars (census, Hipparcos).<br>Fainter points near the "
                        "view centre: procedural systems filling what the catalogues miss.<br>"
                        "Left-drag rotate · Ctrl/middle-drag pan · wheel zoom · double-click centre</small>")
        legend.setWordWrap(True)
        ll.addWidget(legend)

        self.info = QTextBrowser()
        self.info.document().setDefaultStyleSheet(info_mod.STYLE)
        self.open_b = QPushButton("Open system ▶")
        self.open_b.setEnabled(False)
        self.open_b.clicked.connect(self.open_system)
        self.centre_b = QPushButton("Centre on it")
        self.centre_b.clicked.connect(self.centre_on_selection)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.info, 1)
        rl.addWidget(self.centre_b)
        rl.addWidget(self.open_b)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.view)
        split.addWidget(right)
        split.setSizes([280, 900, 320])
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())
        self.hud = QLabel(self.view)
        self.hud.setStyleSheet("color: #8fa0bc; background: transparent;")
        self.hud.move(10, 8)
        self.hud.resize(600, 20)

        self.windows = []
        self._thread = None
        self._last_cam = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)
        self.fly((0, 0, 0), 60, elevation=25)

    # --- camera -------------------------------------------------------------------------
    def _centre(self) -> np.ndarray:
        c = self.view.opts["center"]
        return np.array([c.x(), c.y(), c.z()])

    def fly(self, centre, distance, elevation=None):
        self.view.setCameraPosition(pos=QVector3D(*map(float, centre)), distance=float(distance),
                                    elevation=elevation)
        self._last_cam = None

    def load_radius(self) -> float:
        d = self.view.opts["distance"]
        return 0.0 if d > 1500 else float(np.clip(0.5 * d, 12.0, 45.0))

    # --- level of detail ---------------------------------------------------------------------
    def _tick(self):
        c, d = self._centre(), self.view.opts["distance"]
        cam = (tuple(np.round(c / 3)), round(math.log(d) * 8))
        if cam != self._last_cam:
            self._last_cam = cam
            r = self.load_radius()
            want = set(gx.sectors_within(c, r)) if r > 0 else set()
            for k in [k for k in self.loaded if k not in want]:
                del self.loaded[k]
            self.queue = sorted((k for k in want if k not in self.loaded),
                                key=lambda k: np.linalg.norm((np.array(k) + 0.5) * gx.SECTOR_PC - c))
            self._update_cloud_alpha(d)
            self._rebuild_proc()
            self._restyle_real()
            self._update_labels()
            self.hud.setText(f"view centre ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) pc · camera distance "
                             f"{d:,.0f} pc · rings every {self.hud_scale:g} pc · "
                             f"{len(self.proc['sid'])} procedural systems loaded")
        if self.queue:
            for _ in range(12):
                if not self.queue:
                    break
                k = self.queue.pop(0)
                self.loaded[k] = self._sector_points(k)
            self._rebuild_proc()
            if not self.queue:
                self._update_labels()
            self.hud.setText(f"loading sectors… {len(self.queue)} left")

    def _sector_points(self, key):
        sec = gx.sector(*key)
        keep = rs.procedural_kept(key)
        if not len(keep):
            return None
        c = sec.components
        first = np.searchsorted(c["system"], np.arange(len(sec)))
        mv = rs.abs_v(c["teff"], c["luminosity"])
        flux = np.bincount(c["system"], weights=10 ** (-0.4 * mv), minlength=len(sec))
        absv = -2.5 * np.log10(np.maximum(flux, 1e-30))
        teff = np.asarray(c["teff"])[first]
        spt = np.asarray(c["spectral_type"])[first]
        n = np.asarray(sec.systems["n_components"])
        return (sec.xyz[keep].astype(np.float32), [sec.sid(int(j)) for j in keep], absv[keep], teff[keep],
                list(spt[keep]), n[keep], list(sec.population[keep]))

    def _rebuild_proc(self):
        parts = [v for v in self.loaded.values() if v is not None]
        if not parts:
            self.proc = {"xyz": np.zeros((0, 3), np.float32), "sid": [], "absv": np.zeros(0), "teff": np.zeros(0),
                         "spt": [], "n": np.zeros(0, int), "pop": []}
            self.proc_item.setData(pos=np.zeros((1, 3), np.float32), size=0.0)
            return
        self.proc = {"xyz": np.concatenate([p[0] for p in parts]), "sid": sum((p[1] for p in parts), []),
                     "absv": np.concatenate([p[2] for p in parts]), "teff": np.concatenate([p[3] for p in parts]),
                     "spt": sum((p[4] for p in parts), []), "n": np.concatenate([p[5] for p in parts]),
                     "pop": sum((p[6] for p in parts), [])}
        size, col = appearance(self.proc["absv"], self.proc["xyz"], self._eye(), self.view.opts["distance"],
                               teff_colour(self.proc["teff"], 0.85), self._centre())
        self.proc_item.setData(pos=self.proc["xyz"], color=col, size=size)

    def _eye(self) -> np.ndarray:
        e = self.view.cameraPosition()
        return np.array([e.x(), e.y(), e.z()])

    def _restyle_real(self):
        if self.view.opts["distance"] > 3000:  # galaxy scale: individual stars are not resolved
            self.real_vis = np.zeros(len(self.real_xyz), bool)
            self.real_item.setVisible(False)
            return
        self.real_item.setVisible(True)
        size, col = appearance(self.real_absv, self.real_xyz, self._eye(), self.view.opts["distance"], self.real_col,
                               self._centre())
        self.real_vis = size > 0
        self.real_item.setData(color=col, size=size)

    def _update_cloud_alpha(self, d):
        a = float(np.clip((math.log10(max(d, 1)) - 2.7) / 1.0, 0.0, 1.0))
        col = self.cloud_col.copy()
        col[:, 3] *= a
        self.cloud.setData(color=col, size=2.0 if d > 8000 else 1.5)
        self.cloud.setVisible(a > 0.01)

    def _update_labels(self):
        c, d = self._centre(), self.view.opts["distance"]
        r = np.linalg.norm(self.real_xyz - c, axis=1)
        vis = getattr(self, "real_vis", np.ones(len(r), bool))
        named = ~self.real["name"].str.startswith(("HIP ", "HD ", "2MASS", "SDSS", "WISE", "LP ", "G ")).to_numpy()
        cand = np.flatnonzero(named & vis & (r < 0.9 * d))
        app = self.real_absv[cand] + 5 * np.log10(np.maximum(np.linalg.norm(self.real_xyz[cand] - self._eye(), axis=1),
                                                             0.1) / 10)
        cand = cand[np.argsort(app)][:30]
        if self.selection is not None:
            sel_label = ([self.selection[2]], [self._sel_name()])
        else:
            sel_label = ([], [])
        if d > 3000:
            cand = cand[:0]
            sel_label = (sel_label[0] + [np.zeros(3), np.array([gx.R0_PC, 0, -gx.Z_SUN_PC])],
                         sel_label[1] + ["Sun", "Galactic centre"])
        self.view.label_xyz = np.array(list(self.real_xyz[cand]) + sel_label[0], float).reshape(-1, 3)
        self.view.label_text = [str(n) for n in self.real["name"].iloc[cand]] + sel_label[1]
        # drop lines to the plane through the view centre, and a ring grid (Elite-style depth cues)
        eye = self._eye()
        near = np.flatnonzero(vis & (r < 0.45 * d))
        appm = self.real_absv[near] + 5 * np.log10(np.maximum(np.linalg.norm(self.real_xyz[near] - eye, axis=1), 0.1))
        pp = [self.real_xyz[near[np.argsort(appm)][:250]]]
        if len(self.proc["xyz"]):
            rp = np.linalg.norm(self.proc["xyz"] - c, axis=1)
            sel = np.flatnonzero((rp < 0.3 * d) & (self.proc["absv"] < 9.0))[:150]
            pp.append(self.proc["xyz"][sel])
        top = np.concatenate(pp).astype(np.float32)
        bottom = top.copy()
        bottom[:, 2] = c[2]
        seg = np.empty((2 * len(top), 3), np.float32)
        seg[0::2], seg[1::2] = top, bottom
        self.drops.setData(pos=seg if len(seg) else np.zeros((2, 3), np.float32), mode="lines",
                           color=(0.35, 0.5, 0.75, 0.16))
        step = 10 ** math.floor(math.log10(max(d / 3, 1)))
        rings = []
        th = np.linspace(0, 2 * np.pi, 97)
        for k in range(1, 6):
            rr = k * step * (2 if d / step > 12 else 1)
            ring = np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), np.full(len(th), c[2])]
            rings.append(np.repeat(ring, 2, axis=0)[1:-1])
        self.grid.setData(pos=np.concatenate(rings).astype(np.float32), mode="lines", color=(0.25, 0.35, 0.55, 0.35))
        self.hud_scale = step * (2 if d / step > 12 else 1)

    def _sel_name(self) -> str:
        kind, i, _ = self.selection
        if kind == "id":
            return i
        return str(self.real["name"].iloc[i]) if kind == "real" else self.proc["sid"][i]

    # --- selection ----------------------------------------------------------------------
    def pick(self, x, y):
        best = None
        vis = {"real": getattr(self, "real_vis", np.ones(len(self.real_xyz), bool))}
        for kind, xyz in (("real", self.real_xyz), ("proc", self.proc["xyz"])):
            if not len(xyz):
                continue
            sc, ok = self.view.project(xyz.astype(np.float64))
            if kind in vis:
                ok &= vis[kind]
            dist = np.hypot(sc[:, 0] - x, sc[:, 1] - y)
            dist[~ok] = np.inf
            i = int(np.argmin(dist))
            if dist[i] < 9 and (best is None or dist[i] < best[0] - (2 if kind == "real" else 0)):
                best = (dist[i], kind, i)
        if best is None:
            return
        if best[1] == "real":
            self.select_real(best[2])
        else:
            self.select_proc(best[2])

    def _mark(self, xyz):
        self.sel_item.setData(pos=np.array([xyz], np.float32), size=18.0, color=(0.4, 1, 0.6, 0.6))
        self._last_cam = None  # refresh labels

    def select_real(self, i, centre=False):
        i = int(i)
        row = self.real.iloc[i]
        xyz = self.real_xyz[i]
        self.selection = ("real", i, xyz)
        self._mark(xyz)
        d = float(np.linalg.norm(xyz))
        l = math.degrees(math.atan2(xyz[1], xyz[0])) % 360
        b = math.degrees(math.asin(xyz[2] / d)) if d > 0 else 0.0
        src = {"census": "20-pc census (Kirkpatrick et al. 2024)", "hipparcos": "Hipparcos (ESA 1997)"}[row["source"]]
        rows = [("Catalogue name", row["lookup"]), ("Spectral type", row["sptype"] or "—"),
                ("V magnitude", f"{row['vmag']:.2f}" if np.isfinite(row["vmag"]) else "—"),
                ("Absolute V", f"{self.real_absv[i]:.2f}"), ("Distance", f"{d:.2f} pc ({d * 3.2616:.1f} ly)"),
                ("Galactic l, b", f"{l:.2f}°, {b:+.2f}°"), ("Source", src)]
        html = "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows)
        self.info.setHtml(f"<h2>{row['name']}</h2><p class='desc'>Real star system. Opening it generates planetary "
                          f"systems conditioned on its known properties and planets.</p><table>{html}</table>")
        self.open_b.setEnabled(True)
        self.open_b.setText("Generate its systems ▶")
        if centre:
            self.centre_on_selection()

    def select_proc(self, i):
        p = self.proc
        xyz = p["xyz"][i]
        self.selection = ("proc", i, xyz)
        self._mark(xyz)
        d = float(np.linalg.norm(xyz))
        rows = [("ID", p["sid"][i]), ("Primary", p["spt"][i]), ("Stars", str(int(p["n"][i]))),
                ("Absolute V", f"{p['absv'][i]:.2f}"), ("Distance", f"{d:.2f} pc ({d * 3.2616:.1f} ly)"),
                ("Population", p["pop"][i] + " disc" if p["pop"][i] in ("thin", "thick") else p["pop"][i])]
        html = "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows)
        self.info.setHtml(f"<h2>{p['sid'][i]}</h2><p class='desc'>Procedural system (too faint for the "
                          f"catalogues).</p><table>{html}</table>")
        self.open_b.setEnabled(True)
        self.open_b.setText("Open system ▶")

    def centre_on_selection(self):
        if self.selection is not None:
            self.fly(self.selection[2], min(self.view.opts["distance"], 40.0))

    def do_search(self):
        t = self.search.text().strip()
        self.results.clear()
        if t.startswith("G"):
            try:
                key, j = gx.parse(t)
                sec = gx.sector(*key)
                self.fly(sec.xyz[j], 25)
                self.statusBar().showMessage(f"{t}: {sec.population[j]} population")
                self.selection = ("id", t, sec.xyz[j])
                self._mark(sec.xyz[j])
                self._open_walk(GalaxyWalkSource(t))
                return
            except (ValueError, IndexError):
                pass
        for i in rs.search(t):
            r = self.real.iloc[i]
            it = QListWidgetItem(f"{r['name']}  ·  {r['sptype']}  ·  {np.linalg.norm(self.real_xyz[i]):.1f} pc")
            it.setData(Qt.ItemDataRole.UserRole, int(i))
            self.results.addItem(it)
        if self.results.count():
            self.select_real(self.results.item(0).data(Qt.ItemDataRole.UserRole), True)
        else:
            self.statusBar().showMessage(f"no star matches {t!r}")

    # --- opening systems ----------------------------------------------------------------
    def open_system(self):
        if self.selection is None:
            return
        kind, i, xyz = self.selection
        if kind == "proc":
            self._open_walk(GalaxyWalkSource(self.proc["sid"][i]))
            return
        if kind == "id":
            self._open_walk(GalaxyWalkSource(i))
            return
        lookup, name = str(self.real["lookup"].iloc[i]), str(self.real["name"].iloc[i])
        self.statusBar().showMessage(f"conditioning a planetary system on {name} (takes a few seconds)…")
        self.open_b.setEnabled(False)
        th = QThread(self)
        worker = _Conditioner(lookup, name)
        worker.moveToThread(th)
        th.started.connect(worker.run)
        worker.done.connect(lambda _b, nm, lk=lookup: self._open_walk(GalaxyWalkSource(f"R:{nm}", (lk, nm))))
        worker.failed.connect(lambda msg: (self.statusBar().showMessage(f"could not condition: {msg}"),
                                           self.open_b.setEnabled(True)))
        worker.done.connect(th.quit)
        worker.failed.connect(th.quit)
        self._thread = (th, worker)
        th.start()

    def _open_walk(self, source):
        from .app import SystemWindow
        self.open_b.setEnabled(True)
        w = SystemWindow(source, on_load=self._followed)
        w.show()
        self.windows.append(w)
        self.statusBar().showMessage(f"{source.origin}: Next / Previous walk through random systems of the galaxy")

    def _followed(self, sid, bundle, i):
        """The system viewer moved to ``sid``: mark it on the map and fly there."""
        if sid.startswith("R:"):
            return
        pos = bundle.meta.get("position_pc")
        if pos is None:
            return
        xyz = np.asarray(pos, float)
        if self.selection is not None and np.allclose(self.selection[2], xyz):
            return
        self.selection = ("id", sid, xyz)
        self._mark(xyz)
        self.fly(xyz, min(self.view.opts["distance"], 40.0))
        self.info.setHtml(f"<h2>{sid}</h2><p class='desc'>Procedural system ({bundle.meta.get('population', '')} "
                          f"population), shown in the system viewer.</p>")
        self.open_b.setEnabled(True)
        self.open_b.setText("Open system ▶")


__all__ = ["GalaxyWalkSource", "StarMapWindow", "mag_size", "teff_colour"]
