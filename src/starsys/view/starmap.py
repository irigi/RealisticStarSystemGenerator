"""3D star map of the procedural Milky Way with the real stars blended in.

Layers (heliocentric Galactic coordinates in parsecs, see :mod:`starsys.galaxy`):

* the real stars (:mod:`starsys.realstars`: 20-pc census + Hipparcos Survey);
* the procedural systems, minus those the catalogues already hold, from the galaxy's
  brightness tiers: the 10-pc sectors (all faint stars) near the view centre, the A/F-star
  and B-star cells out to a few hundred pc, and the O/B stars (>= 8 Msun) of the whole galaxy;
* the unresolved light of the rest (old disc, bulge, halo) as a faint point cloud sampled
  from the density model, which brightens smoothly as the camera moves out.

What is drawn: every star gets an apparent magnitude from the camera plus a penalty outside
the region being looked at (a map, not a sky); the :data:`STAR_BUDGET` brightest are shown,
graded in size and brightness relative to the faintest shown, so the field stays equally
full at every zoom. A tier is shown only while its cells cover the view (it fades out as a
whole when the view grows past them, never at a radius); real stars follow the tier of their
absolute magnitude. Zooming out, the faint stars give way to ever brighter ones until only
the O and B stars of the spiral arms are left, against the glow of the old stars.

Mouse: left-drag rotates, right- or middle-drag moves the map, wheel zooms, click selects,
double-click centres on a star or on the place under the cursor.
"""

from __future__ import annotations

import math
import time

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

STAR_BUDGET = 6000      # stars shown in a local view; up to 5x more for the whole galaxy
# largest radius (pc) to which each tier's cells are loaded around the view centre (tier 3:
# the whole galaxy)
TIER_RADIUS = {0: 45.0, 1: 250.0, 2: 700.0, 3: math.inf}
# a real star's tier (as the procedural ones: living primaries of 1.5-3, 3-8, >= 8 Msun):
# main-sequence-like stars by Teff (F0 ~ 1.5, B9 ~ 3, B2 ~ 8 Msun; Pecaut & Mamajek 2013),
# supergiants (class I) by absolute V, cool giants as tier 1 (mostly 1-3 Msun)
REAL_TIER_TEFF = (7200.0, 10500.0, 20000.0)


def teff_colour(teff: np.ndarray, alpha=1.0) -> np.ndarray:
    t = np.clip(np.nan_to_num(np.asarray(teff, float), nan=5000.0), _T[0], _T[-1])
    i = np.clip(np.searchsorted(_T, t), 0, len(_T) - 1)
    rgb = _RGB[i]
    rgb = np.clip(rgb.mean(1, keepdims=True) + 2.4 * (rgb - rgb.mean(1, keepdims=True)), 0, 1)
    return np.concatenate([rgb, np.full((len(t), 1), alpha)], axis=1).astype(np.float32)


def mag_size(abs_v: np.ndarray) -> np.ndarray:
    """Point size (px) from absolute V magnitude."""
    return np.clip(7.5 - 0.42 * np.nan_to_num(abs_v, nan=10.0), 1.5, 14.0).astype(np.float32)


def tier_weight(tier, view_dist: float) -> np.ndarray:
    """1 while a tier's cells cover the view, fading to 0 as the view outgrows them."""
    r = np.array([TIER_RADIUS[t] for t in range(4)])[np.asarray(tier, int)]
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(r), 1.0, np.clip((r - 0.6 * view_dist) / (0.3 * view_dist), 0.0, 1.0))


def real_tier(abs_v: np.ndarray, teff: np.ndarray, sptype) -> np.ndarray:
    """Brightness tier of real stars, comparable to the procedural tiers (which go by mass)."""
    import re
    t = np.nan_to_num(np.asarray(teff, float), nan=5000.0)
    a = np.nan_to_num(np.asarray(abs_v, float), nan=99.0)
    tier = sum((t >= x).astype(int) for x in REAL_TIER_TEFF)
    cls = [re.search(r"(I{1,3}|IV|V)(?:a|b|ab)?\b", str(x) or "") for x in sptype]
    lum = np.array([m.group(1) if m else "" for m in cls], object)
    giant = (lum == "II") | (lum == "III")
    tier = np.where(giant & (t < REAL_TIER_TEFF[0]), 1, tier)
    tier = np.where(lum == "I", np.where(a < -4.0, 3, 2), tier)
    return tier.astype(int)


def star_budget(view_dist: float) -> int:
    """More stars for wider views: the whole galaxy needs its O/B stars to show the arms."""
    return int(STAR_BUDGET * np.clip((view_dist / 3000.0) ** 0.7, 1.0, 5.0))


def effective_mag(abs_v, xyz, eye, view_dist, centre, tier=None):
    """Apparent V from the camera plus the penalties (mag) for lying outside the region looked
    at and for belonging to a tier that is fading out (inf: not shown)."""
    d = np.maximum(np.linalg.norm(xyz - eye, axis=1), 0.05)
    m = np.nan_to_num(abs_v, nan=12.0) + 5 * np.log10(d / 10.0)
    rc = np.linalg.norm(xyz - centre, axis=1)
    m = m + np.clip((rc / view_dist - 0.5) * 6.0, 0.0, None) ** 1.5
    if tier is not None:
        w = tier_weight(tier, view_dist)
        m = np.where(w > 0, m + 1.5 * (1 - w), np.inf)
    return m


# reference apparent magnitude of the point sizes against log10(camera distance / pc): about
# the faintest shown in a Sun-centred view (see magnitude_limit), but smooth in the zoom, so
# that a star's size never jumps when the set of stars shown changes
_REF_LOGD = np.array([1.0, 1.6, 2.2, 3.2, 4.65])
_REF_MAG = np.array([13.5, 13.5, 10.0, 10.5, 13.5])


def reference_mag(view_dist: float) -> float:
    return float(np.interp(math.log10(max(view_dist, 1.0)), _REF_LOGD, _REF_MAG))


def base_size(view_dist: float) -> float:
    """Point size (px) of a star at the reference magnitude: larger in close views, where
    single faint dwarfs must be easy to see, smaller for the crowded galaxy."""
    return float(np.interp(math.log10(max(view_dist, 1.0)), [1.6, 3.7], [3.4, 2.2]))


def magnitude_limit(m_eff: np.ndarray, view_dist: float, budget: int | None = None) -> float:
    """Faintest effective magnitude shown: the ``budget``-th brightest (default
    :func:`star_budget`), and no fainter than a cap above the reference magnitude (so a sparse
    region is not filled with the faintest dwarfs). In close views (< ~40 pc) the cap opens
    up to every star, brown dwarfs included."""
    budget = star_budget(view_dist) if budget is None else budget
    near = float(np.clip((math.log10(100.0) - math.log10(max(view_dist, 1.0))) / 0.4, 0.0, 1.0))
    cap = reference_mag(view_dist) + 3.0 + 12.0 * near
    if len(m_eff) > budget:
        return float(min(np.partition(m_eff, budget)[budget], cap))
    return cap


def appearance(m_eff, m_lim, base_col, view_dist=None, weight=None):
    """Sizes and colours. The size follows the star's effective magnitude against the smooth
    :func:`reference_mag` of the zoom; the limit only fades the faintest out (over 1.5 mag),
    as does a tier's ``weight`` while the view outgrows its cells."""
    m = np.asarray(m_eff, float)
    if view_dist is None:  # sizes relative to the limit
        ref, s0 = m_lim, 1.0
    else:
        ref, s0 = reference_mag(view_dist), base_size(view_dist)
    fade = np.clip((m_lim - m) / 1.5, 0.0, 1.0)
    if weight is not None:
        fade = fade * weight
    size = np.clip(s0 + 0.4 * (ref - m), 0.75 * s0, 12.0) * (0.55 + 0.45 * fade)
    size = np.where(fade > 0, size, 0.0).astype(np.float32)
    col = base_col.copy()
    col[:, 3] = fade * base_col[:, 3]
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
        drawn: list[tuple[float, float]] = []
        # the selection's label (last) first, then the others where they do not overlap
        for (x, y), good, t in reversed(list(zip(sc, ok, self.label_text))):
            if good and all(abs(x - u) > 70 or abs(y - v) > 13 for u, v in drawn):
                p.drawText(int(x) + 6, int(y) - 4, t)
                drawn.append((x, y))
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

    def mouseMoveEvent(self, ev):
        if ev.buttons() == Qt.MouseButton.RightButton:
            # right-drag: move the map in the galactic plane (as middle-drag)
            lpos = ev.position()
            diff = lpos - getattr(self, "mousePos", lpos)
            self.mousePos = lpos
            self.pan(diff.x(), diff.y(), 0, relative="view-upright")
            return
        super().mouseMoveEvent(ev)

    def _matrix(self) -> np.ndarray:
        vp = self.getViewport()
        m = self.projectionMatrix(vp, vp) * self.viewMatrix()
        return np.array(m.data(), dtype=np.float64).reshape(4, 4).T  # column-major

    def ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction of the line of sight through the screen point (px)."""
        inv = np.linalg.inv(self._matrix())
        ndc = np.array([2 * x / self.width() - 1, 1 - 2 * y / self.height()])
        a, b = (inv @ np.r_[ndc, z, 1.0] for z in (-1.0, 1.0))
        a, b = a[:3] / a[3], b[:3] / b[3]
        return a, (b - a) / np.linalg.norm(b - a)

    def plane_point(self, x: float, y: float, z0: float) -> np.ndarray:
        """Where the line of sight through (x, y) meets the horizontal plane z = z0 (or, when
        looking along the plane, the point at the current centre's distance)."""
        o, d = self.ray(x, y)
        c = self.opts["center"]
        dist = float((self.cameraPosition() - c).length())
        if abs(d[2]) > 1e-3:
            t = (z0 - o[2]) / d[2]
            if 0 < t < 50 * dist:
                return o + t * d
        e = self.cameraPosition()
        return np.array([e.x(), e.y(), e.z()]) + dist * d

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen coordinates (px) of points and a mask of those in front of the camera."""
        h = np.c_[xyz, np.ones(len(xyz))] @ self._matrix().T
        w = h[:, 3]
        ok = w > 1e-9
        ndc = h[:, :2] / np.where(ok, w, 1.0)[:, None]
        sx = (ndc[:, 0] + 1) / 2 * self.width()
        sy = (1 - ndc[:, 1]) / 2 * self.height()
        return np.stack([sx, sy], 1), ok & (np.abs(ndc) <= 1.05).all(1)


class _Conditioner(QObject):
    done = Signal(object, str, str)
    failed = Signal(str)

    def __init__(self, lookup, name):
        super().__init__()
        self.lookup, self.name = lookup, name

    def run(self):
        try:
            self.done.emit(rs.real_system(self.lookup, self.name), self.name, self.lookup)
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


def _empty_proc() -> dict:
    return {"xyz": np.zeros((0, 3), np.float32), "sid": np.zeros(0, object), "absv": np.zeros(0),
            "teff": np.zeros(0), "spt": np.zeros(0, object), "n": np.zeros(0, int), "pop": np.zeros(0, object),
            "tier": np.zeros(0, np.int8)}


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
        self.view.doublePicked.connect(self.double_click)

        # unresolved light of the old stars (the young ones are resolved: brightness tier 3)
        pts, comp, arm = gx.sample_cloud(cloud_n)
        self.cloud_xyz = pts.astype(np.float32)
        col = np.empty((len(pts), 4), np.float32)
        warm = np.array([1.0, 0.8, 0.55], np.float32)
        white = np.array([0.92, 0.88, 0.8], np.float32)
        col[:, :3] = np.where((comp == 0)[:, None], white, warm)
        col[:, 3] = np.where(comp == 2, 0.045, np.where(comp == 0, 0.11, 0.07))
        self.cloud_col = col
        self.cloud = gl.GLScatterPlotItem(pos=self.cloud_xyz, color=col, size=2.0, pxMode=True)
        self.cloud.setGLOptions("additive")
        self.view.addItem(self.cloud)
        # arm centre lines (faint, galaxy scale only)
        self.arm_items = []
        for name, xy in gx.arm_lines():
            line = np.c_[xy, np.full(len(xy), -gx.Z_SUN_PC)].astype(np.float32)
            it = gl.GLLinePlotItem(pos=line[::10], color=(0.3, 0.45, 0.8, 0.0), width=1, antialias=True)
            it.setGLOptions("additive")
            self.view.addItem(it)
            self.arm_items.append(it)

        # real stars
        s = rs.stars()
        self.real = s
        self.real_xyz = s[["x", "y", "z"]].to_numpy(np.float32)
        d = np.maximum(np.linalg.norm(self.real_xyz, axis=1), 1e-3)
        absv = s["vmag"].to_numpy(float) - 5 * np.log10(d / 10) - rs.extinction(self.real_xyz.astype(float))
        absv[0] = 4.83  # the Sun
        self.real_absv = absv
        self.real_tier = real_tier(absv, s["teff"].to_numpy(float), s["sptype"].fillna("").to_numpy())
        self.real_col = teff_colour(s["teff"].to_numpy(float))
        self.real_vis = np.zeros(len(absv), bool)
        self.real_item = gl.GLScatterPlotItem(pos=self.real_xyz, color=self.real_col, size=mag_size(absv), pxMode=True)
        self.real_item.setGLOptions("additive")
        self.view.addItem(self.real_item)

        # procedural systems: cells of the brightness tiers, loaded around the view centre
        # (tier 3, the O/B stars, for the whole galaxy)
        self.loaded: dict[tuple, dict | None] = {}
        self.queue: list[tuple] = []
        self.tier3 = [(3, k) for k in gx.cells_within(3, (gx.R0_PC, 0, 0), 16000) if abs(k[2]) <= 2]
        self.proc_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=0.0, pxMode=True)
        self.proc_item.setGLOptions("additive")
        self.view.addItem(self.proc_item)
        self.proc = _empty_proc()
        self.proc_col = teff_colour(np.zeros(0))
        self.proc_vis = np.zeros(0, bool)
        self._dirty = False
        self._last_rebuild = 0.0
        self._mlim = None
        self._mlim_target = 0.0

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
        b_sun.clicked.connect(lambda: self.fly((0, 0, 0), 60, animate=True))
        b_gal = QPushButton("Whole galaxy")
        b_gal.clicked.connect(lambda: self.fly((gx.R0_PC, 0, 0), 45000, elevation=55, animate=True))
        b_hood = QPushButton("1 kpc")
        b_hood.clicked.connect(lambda: self.fly(self._centre(), 2500, animate=True))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Find a star"))
        ll.addWidget(self.search)
        ll.addWidget(self.results, 1)
        row = QHBoxLayout()
        for b in (b_sun, b_hood, b_gal):
            row.addWidget(b)
        ll.addLayout(row)
        legend = QLabel("<small>Real stars (20-pc census, Hipparcos) and procedural systems filling what the "
                        "catalogues miss. The brightest stars in view are shown; zooming out leaves the "
                        "O and B stars of the arms against the glow of the old disc and bulge.<br>"
                        "Left-drag rotate · right- or middle-drag move · wheel zoom · double-click a star or any "
                        "place to centre there</small>")
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
        self._flight = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)
        self.fly((0, 0, 0), 60, elevation=25)

    # --- camera -------------------------------------------------------------------------
    def _centre(self) -> np.ndarray:
        c = self.view.opts["center"]
        return np.array([c.x(), c.y(), c.z()])

    def fly(self, centre, distance, elevation=None, animate=False, ms=350):
        """Move the view centre and camera distance (smoothly with ``animate``: the centre
        linearly, the distance logarithmically)."""
        self._flight = None
        if not animate:
            self.view.setCameraPosition(pos=QVector3D(*map(float, centre)), distance=float(distance),
                                        elevation=elevation)
            self._last_cam = None
            return
        e0 = self.view.opts["elevation"]
        self._flight = (self._centre(), np.asarray(centre, float), math.log(self.view.opts["distance"]),
                        math.log(float(distance)), e0, e0 if elevation is None else elevation, 0, max(ms // 20, 1))
        QTimer.singleShot(0, self._fly_step)

    def _fly_step(self):
        if self._flight is None:
            return
        c0, c1, l0, l1, e0, e1, k, n = self._flight
        k += 1
        u = k / n
        u = u * u * (3 - 2 * u)  # ease in and out
        self.view.setCameraPosition(pos=QVector3D(*map(float, c0 + (c1 - c0) * u)),
                                    distance=math.exp(l0 + (l1 - l0) * u), elevation=e0 + (e1 - e0) * u)
        self._last_cam = None
        if k < n:
            self._flight = (c0, c1, l0, l1, e0, e1, k, n)
            QTimer.singleShot(20, self._fly_step)
        else:
            self._flight = None

    def double_click(self, x, y):
        """Centre on the star under the cursor, or on the point of the plane through the
        current centre under the cursor."""
        if self.pick(x, y):
            self.centre_on_selection()
        else:
            self.fly(self.view.plane_point(x, y, self._centre()[2]), self.view.opts["distance"], animate=True)

    # --- level of detail ---------------------------------------------------------------------
    def wanted_cells(self, centre, d) -> list[tuple]:
        """(tier, cell) keys to load for a view: tiers 0-2 within their radius while they are
        shown, nearest first; then the whole galaxy's tier-3 cells."""
        out = []
        for t in (0, 1, 2):
            if tier_weight(t, d) > 0 or (t == 0 and d < 80):
                r = min(TIER_RADIUS[t], d)
                ks = gx.cells_within(t, centre, r)
                size = gx.cell_size(t)
                dist = [np.linalg.norm((np.array(k) + 0.5) * size - centre) for k in ks]
                out += [(t, ks[i]) for i in np.argsort(dist)]
        size = gx.cell_size(3)
        dist = [np.linalg.norm((np.array(k) + 0.5) * size - centre) for _, k in self.tier3]
        return out + [self.tier3[i] for i in np.argsort(dist)]

    def _tick(self):
        c, d = self._centre(), self.view.opts["distance"]
        cam = (tuple(np.round(c / max(d / 30, 1.0))), round(math.log(d) * 8))
        if cam != self._last_cam:
            self._last_cam = cam
            want = self.wanted_cells(c, d)
            keep = set(want)
            for k in [k for k in self.loaded if k not in keep]:
                del self.loaded[k]
                self._dirty = True
            self.queue = [k for k in want if k not in self.loaded]
            self._update_cloud_alpha(d)
            if self._dirty:
                self._rebuild_proc()
            self._restyle()
            self._update_labels()
            self._hud()
        if self.queue:
            t0 = time.perf_counter()
            while self.queue and time.perf_counter() - t0 < 0.035:
                k = self.queue.pop(0)
                self.loaded[k] = self._cell_points(*k)
                self._dirty = True
            if not self.queue or time.perf_counter() - self._last_rebuild > 0.4:
                self._rebuild_proc()
                self._restyle()
                if not self.queue:
                    self._update_labels()
            self._hud()
        if self._ease_limit() and not self.queue:
            self._hud()

    def _hud(self):
        c, d = self._centre(), self.view.opts["distance"]
        load = f" · loading {len(self.queue)} cells…" if self.queue else ""
        self.hud.setText(f"view centre ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) pc · camera distance {d:,.0f} pc · "
                         f"rings every {self.hud_scale:g} pc · {int(self.proc_vis.sum() + self.real_vis.sum())} "
                         f"stars shown of {len(self.proc['xyz']) + len(self.real_xyz):,}{load}")

    def _cell_points(self, tier, key):
        sec = gx.cell(tier, *key)
        keep = rs.procedural_kept(key, tier)
        if not len(keep):
            return None
        c = sec.components
        first = np.searchsorted(c["system"], np.arange(len(sec)))
        mv = rs.abs_v(c["teff"], c["luminosity"])
        flux = np.bincount(c["system"], weights=10 ** (-0.4 * mv), minlength=len(sec))
        absv = -2.5 * np.log10(np.maximum(flux, 1e-30))
        return {"xyz": sec.xyz[keep].astype(np.float32), "sid": np.array([sec.sid(int(j)) for j in keep], object),
                "absv": absv[keep], "teff": np.asarray(c["teff"])[first][keep],
                "spt": np.asarray(c["spectral_type"], object)[first][keep],
                "n": np.asarray(sec.systems["n_components"])[keep], "pop": np.asarray(sec.population, object)[keep],
                "tier": np.full(len(keep), tier, np.int8)}

    def _rebuild_proc(self):
        self._dirty = False
        self._last_rebuild = time.perf_counter()
        parts = [v for v in self.loaded.values() if v is not None]
        self.proc = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else _empty_proc()
        self.proc_col = teff_colour(self.proc["teff"], 0.9)

    def _eye(self) -> np.ndarray:
        e = self.view.cameraPosition()
        return np.array([e.x(), e.y(), e.z()])

    def _restyle(self):
        """One magnitude limit for the real and procedural stars (the STAR_BUDGET brightest).
        The limit follows its target over a few frames (see _tick), so stars fade in and out
        rather than pop when cells load or tiers fade."""
        eye, d, c = self._eye(), self.view.opts["distance"], self._centre()
        self._m_real = effective_mag(self.real_absv, self.real_xyz, eye, d, c, self.real_tier)
        self._m_proc = effective_mag(self.proc["absv"], self.proc["xyz"], eye, d, c, self.proc["tier"])
        self._mlim_target = magnitude_limit(np.concatenate([self._m_real, self._m_proc]), d)
        if self._mlim is None:
            self._mlim = self._mlim_target
        self._apply_style()

    def _apply_style(self):
        d = self.view.opts["distance"]
        size, col = appearance(self._m_real, self._mlim, self.real_col, d, tier_weight(self.real_tier, d))
        self.real_vis = size > 0
        self.real_item.setData(color=col, size=size)
        if len(self._m_proc):
            size, col = appearance(self._m_proc, self._mlim, self.proc_col, d, tier_weight(self.proc["tier"], d))
            self.proc_vis = size > 0
            self.proc_item.setData(pos=self.proc["xyz"], color=col, size=size)
        else:
            self.proc_vis = np.zeros(0, bool)
            self.proc_item.setData(pos=np.zeros((1, 3), np.float32), size=0.0)

    def _ease_limit(self) -> bool:
        """Move the magnitude limit a step towards its target; True if it moved."""
        if self._mlim is None or abs(self._mlim - self._mlim_target) < 0.02:
            return False
        step = self._mlim_target - self._mlim
        self._mlim += float(np.clip(0.3 * step, -0.6, 0.6)) if abs(step) > 0.05 else step
        self._apply_style()
        return True

    def _update_cloud_alpha(self, d):
        """The unresolved light brightens smoothly as the view widens (no switch-over)."""
        a = float(np.clip((math.log10(max(d, 1)) - 2.5) / 1.6, 0.0, 1.0)) ** 1.5
        col = self.cloud_col.copy()
        col[:, 3] *= a
        self.cloud.setData(color=col, size=1.5 + 0.5 * a)
        self.cloud.setVisible(a > 0.005)
        arm_a = 0.22 * float(np.clip((math.log10(max(d, 1)) - 3.3) / 0.7, 0.0, 1.0))
        for it in self.arm_items:
            it.setData(color=(0.3, 0.45, 0.8, arm_a))
            it.setVisible(arm_a > 0.005)

    def _update_labels(self):
        c, d = self._centre(), self.view.opts["distance"]
        r = np.linalg.norm(self.real_xyz - c, axis=1)
        vis = self.real_vis
        named = ~self.real["name"].str.startswith(("HIP ", "HD ", "2MASS", "SDSS", "WISE", "LP ", "G ")).to_numpy()
        cand = np.flatnonzero(named & vis & (r < 0.9 * d))
        app = self.real_absv[cand] + 5 * np.log10(np.maximum(np.linalg.norm(self.real_xyz[cand] - self._eye(), axis=1),
                                                             0.1) / 10)
        n_lab = int(np.clip(30 * (1500.0 / max(d, 1.0)) ** 0.5, 0, 30)) if d < 3000 else 0
        cand = cand[np.argsort(app)][:n_lab]
        if self.selection is not None:
            sel_label = ([self.selection[2]], [self._sel_name()])
        else:
            sel_label = ([], [])
        if d > 3000:
            sel_label = (sel_label[0] + [np.zeros(3), np.array([gx.R0_PC, 0, -gx.Z_SUN_PC])],
                         sel_label[1] + ["Sun", "Galactic centre"])
        self.view.label_xyz = np.array(list(self.real_xyz[cand]) + sel_label[0], float).reshape(-1, 3)
        self.view.label_text = [str(n) for n in self.real["name"].iloc[cand]] + sel_label[1]
        # drop lines to the plane through the view centre, and a ring grid (Elite-style depth cues)
        # (fewer as the view widens; none at galaxy scale, where they would only scratch the disc)
        eye = self._eye()
        n_drop = int(np.clip(250 * (60.0 / max(d, 1.0)) ** 0.6, 0, 250)) if d < 1500 else 0
        near = np.flatnonzero(vis & (r < 0.45 * d))
        appm = self.real_absv[near] + 5 * np.log10(np.maximum(np.linalg.norm(self.real_xyz[near] - eye, axis=1), 0.1))
        pp = [self.real_xyz[near[np.argsort(appm)][:n_drop]]]
        if len(self.proc["xyz"]) and n_drop:
            rp = np.linalg.norm(self.proc["xyz"] - c, axis=1)
            sel = np.flatnonzero((rp < 0.3 * d) & self.proc_vis)
            sel = sel[np.argsort(self.proc["absv"][sel])][: n_drop // 2]
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
        vis = {"real": self.real_vis, "proc": self.proc_vis}
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
            return False
        if best[1] == "real":
            self.select_real(best[2])
        else:
            self.select_proc(best[2])
        return True

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
            self.fly(self.selection[2], min(self.view.opts["distance"], 40.0), animate=True)

    def do_search(self):
        t = self.search.text().strip()
        self.results.clear()
        if t.startswith("G"):
            try:
                tier, key, j = gx.parse(t)
                sec = gx.cell(tier, *key)
                self.fly(sec.xyz[j], 25, animate=True)
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
        # bound methods of this window (not lambdas): Qt then runs them in the GUI thread
        worker.done.connect(self._real_ready)
        worker.failed.connect(self._real_failed)
        worker.done.connect(th.quit)
        worker.failed.connect(th.quit)
        self._thread = (th, worker)
        th.start()

    def _real_ready(self, _bundle, name, lookup):
        self._open_walk(GalaxyWalkSource(f"R:{name}", (lookup, name)))

    def _real_failed(self, msg):
        self.statusBar().showMessage(f"could not condition: {msg}")
        self.open_b.setEnabled(True)

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
            hits = rs.search(sid[2:], limit=1)
            if hits and (self.selection is None or self.selection[:2] != ("real", hits[0])):
                self.select_real(hits[0], centre=True)
            return
        pos = bundle.meta.get("position_pc")
        if pos is None:
            return
        xyz = np.asarray(pos, float)
        if self.selection is not None and np.allclose(self.selection[2], xyz):
            return
        self.selection = ("id", sid, xyz)
        self._mark(xyz)
        self.fly(xyz, min(self.view.opts["distance"], 40.0), animate=True)
        self.info.setHtml(f"<h2>{sid}</h2><p class='desc'>Procedural system ({bundle.meta.get('population', '')} "
                          f"population), shown in the system viewer.</p>")
        self.open_b.setEnabled(True)
        self.open_b.setText("Open system ▶")


__all__ = ["GalaxyWalkSource", "STAR_BUDGET", "StarMapWindow", "appearance", "effective_mag", "mag_size",
           "magnitude_limit", "real_tier", "teff_colour", "tier_weight"]
