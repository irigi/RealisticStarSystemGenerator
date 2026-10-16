"""Top-down orbit view at true separations.

Distances are to scale (linear) or logarithmic around the view centre (``log`` mode, for
systems spanning 0.01-10^4 au); bodies are drawn on their own size scale (the same
``R^0.5`` law as the system map, times a user factor), since at true scale every planet
would be far below a pixel. Habitable zones are shaded around each star (conservative zone
green, optimistic edges lighter). Orbits are face-on (see :mod:`starsys.view.model`).
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import model as md
from .sysmap import BG, SELECT, SUBTEXT, TEXT, display_px, pixmap

ORBIT = QColor(80, 105, 150)
ORBIT_STAR = QColor(200, 170, 90)
HZ_CONS = QColor(60, 170, 90, 60)
HZ_OPT = QColor(60, 170, 90, 28)


class OrbitView(QWidget):
    bodyClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.m: md.SystemModel | None = None
        self.t = 0.0
        self.log = False
        self.scale = 100.0  # px per au (linear) or px per decade (log)
        self.r0 = 0.01      # log mode: au at which the log scale starts
        self.centre = np.zeros(2)  # au, view centre in world coordinates
        self.follow: str | None = None
        self.size_factor = 1.0
        self.selected: str | None = None
        self.show_moons = True
        self._drag = None
        self._hit: list[tuple[float, float, float, str]] = []
        self.setMouseTracking(False)

    # --- set-up -------------------------------------------------------------------------
    def set_system(self, m: md.SystemModel):
        self.m = m
        self.t = 0.0
        self.follow = None
        self.selected = None
        self.fit()

    def extent(self) -> tuple[float, float]:
        """Smallest and largest orbital distances (au) in the system."""
        a = []
        for b in self.m.bodies.values():
            o = b.orbit
            if o and b.cat != "moon":
                a.append(o[0] * (1 + o[1]))
        if not a:
            return 0.01, 1.0
        return max(min(a), 1e-3), max(a)

    def fit(self):
        if self.m is None:
            return
        lo, hi = self.extent()
        self.centre = np.zeros(2)
        half = 0.45 * min(self.width(), self.height())
        if self.log:
            self.r0 = lo * 0.5
            self.scale = half / max(math.log10(1 + hi / self.r0), 0.5)
        else:
            self.scale = half / hi
        self.update()

    def set_log(self, on: bool):
        self.log = on
        self.fit()

    def focus(self, name: str | None):
        """Centre the view on a body (and follow it in time)."""
        self.follow = name
        if self.m is None or name is None:
            return
        b = self.m.bodies[name]
        pos = md.positions(self.m, self.t)
        self.centre = np.array(pos[name])
        # zoom to its children's orbits, if any
        kids = [self.m.bodies[c].orbit for c in b.children if self.m.bodies[c].orbit]
        half = 0.42 * min(self.width(), self.height())
        if kids:
            hi = max(o[0] * (1 + o[1]) for o in kids)
            lo = min(o[0] for o in kids)
            if self.log:
                self.r0 = lo * 0.5
                self.scale = half / max(math.log10(1 + hi / self.r0), 0.5)
            else:
                self.scale = half / hi
        self.update()

    # --- coordinates --------------------------------------------------------------------
    def to_screen(self, xy: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(xy) - self.centre
        c = np.array([self.width() / 2, self.height() / 2])
        if self.log:
            r = np.hypot(d[:, 0], d[:, 1])
            k = np.where(r > 0, self.scale * np.log10(1 + r / self.r0) / np.maximum(r, 1e-30), 0)
            d = d * k[:, None]
        else:
            d = d * self.scale
        return np.stack([c[0] + d[:, 0], c[1] - d[:, 1]], axis=1)

    def px_per_au_at(self, r: float) -> float:
        if not self.log:
            return self.scale
        return self.scale / (math.log(10) * (self.r0 + r))

    # --- painting -----------------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)
        if self.m is None:
            p.end()
            return
        m = self.m
        pos = md.positions(m, self.t)
        if self.follow and self.follow in pos:
            self.centre = np.array(pos[self.follow])
        self._hit = []
        font = QFont()
        font.setPointSizeF(8.5)
        p.setFont(font)
        # habitable zones
        for s in m.stars:
            hz = s.d.get("habitable_zone_au") or {}
            if not hz:
                continue
            c = np.array(pos[s.name])
            for (r1, r2), col in (((hz["recent_venus"], hz["early_mars"]), HZ_OPT),
                                  ((hz["runaway"], hz["max_greenhouse"]), HZ_CONS)):
                path = QPainterPath()
                path.setFillRule(Qt.FillRule.OddEvenFill)
                for r in (r2, r1):
                    th = np.linspace(0, 2 * np.pi, 160)
                    pts = self.to_screen(c + r * np.stack([np.cos(th), np.sin(th)], 1))
                    path.addPolygon(QPolygonF([QPointF(*q) for q in pts]))
                p.fillPath(path, QBrush(col))
        # orbits
        for name in m.order:
            b = m.bodies[name]
            if b.cat == "moon" and not self.show_moons:
                continue
            o = b.orbit
            if o is None:
                continue
            if b.cat == "moon":
                par = np.array(pos[b.parent])
                rpar = float(np.hypot(*(par - self.centre)))
                if o[0] * self.px_per_au_at(rpar) < 6:
                    continue
            path = md.orbit_path(m, name, pos)
            pts = self.to_screen(path)
            col = ORBIT_STAR if b.cat == "star" else ORBIT
            if b.cat == "moon":
                col = QColor(70, 85, 110)
            pen = QPen(SELECT if name == self.selected else col, 1.6 if name == self.selected else 1.0)
            p.setPen(pen)
            p.drawPolyline(QPolygonF([QPointF(*q) for q in pts]))
        # centres of mass of inner stellar pairs (hierarchical triples, quadruples)
        pen = QPen(ORBIT_STAR, 1.0, Qt.PenStyle.DashLine)
        for com, path in md.barycentre_paths(m, pos):
            p.setPen(pen)
            p.drawPolyline(QPolygonF([QPointF(*q) for q in self.to_screen(path)]))
            q = self.to_screen(com)[0]
            p.setPen(QPen(ORBIT_STAR, 1))
            p.drawLine(QPointF(q[0] - 3, q[1] - 3), QPointF(q[0] + 3, q[1] + 3))
            p.drawLine(QPointF(q[0] - 3, q[1] + 3), QPointF(q[0] + 3, q[1] - 3))
        # bodies (stars last so they stay visible over close-in planets? no: stars first)
        order = sorted(m.order, key=lambda n: {"star": 0, "barycentre": 1, "planet": 2, "moon": 3}[m.bodies[n].cat])
        labels: list[tuple[float, float]] = []
        for name in order:
            b = m.bodies[name]
            if b.cat == "barycentre":
                q = self.to_screen(np.array(pos[name]))[0]
                p.setPen(QPen(ORBIT_STAR, 1))
                p.drawLine(QPointF(q[0] - 4, q[1]), QPointF(q[0] + 4, q[1]))
                p.drawLine(QPointF(q[0], q[1] - 4), QPointF(q[0], q[1] + 4))
                continue
            if b.cat == "moon":
                if not self.show_moons:
                    continue
                par = np.array(pos[b.parent])
                rpar = float(np.hypot(*(par - self.centre)))
                if b.orbit[0] * self.px_per_au_at(rpar) < 6:
                    continue
            q = self.to_screen(np.array(pos[name]))[0]
            if not (-200 < q[0] < self.width() + 200 and -200 < q[1] < self.height() + 200):
                continue
            size = display_px(b.radius_earth, b.cat == "star") * self.size_factor * (0.2 if b.cat == "star" else 0.4)
            size = max(size, 4.0)
            rp = int(size / 0.72) if b.cat == "star" else int(size)
            pm = pixmap(b, max(rp, 4), self.devicePixelRatioF())
            p.drawPixmap(QRectF(q[0] - rp / 2, q[1] - rp / 2, rp, rp), pm, QRectF(pm.rect()))
            if name == self.selected:
                p.setPen(QPen(SELECT, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(*q), size / 2 + 5, size / 2 + 5)
            if b.cat != "moon" or self.selected == name:
                lx, ly = q[0] + size / 2 + 4, q[1] - 3
                # declutter: skip labels that would overlap one already drawn (selection first)
                if name == self.selected or all(abs(lx - x) > 60 or abs(ly - y) > 12 for x, y in labels):
                    p.setPen(TEXT if b.cat != "moon" else SUBTEXT)
                    p.drawText(QPointF(lx, ly), b.short or name)
                    labels.append((lx, ly))
            self._hit.append((q[0], q[1], size / 2 + 4, name))
        self._draw_scale_bar(p)
        p.end()

    def _draw_scale_bar(self, p: QPainter):
        p.setPen(QPen(SUBTEXT, 1))
        y = self.height() - 18
        if self.log:
            p.drawText(QPointF(12, y), "logarithmic distances (from the view centre)")
            return
        target = 120 / self.scale
        k = 10 ** math.floor(math.log10(target))
        for f in (1, 2, 5, 10):
            if f * k >= target * 0.7:
                k *= f
                break
        w = k * self.scale
        p.drawLine(QPointF(12, y), QPointF(12 + w, y))
        p.drawLine(QPointF(12, y - 4), QPointF(12, y + 4))
        p.drawLine(QPointF(12 + w, y - 4), QPointF(12 + w, y + 4))
        label = f"{k:g} au" if k >= 0.01 else f"{k * 1.496e8:.3g} km"
        p.drawText(QPointF(16 + w, y + 4), label + "   (bodies not to scale)")

    # --- interaction --------------------------------------------------------------------
    def wheelEvent(self, ev):
        f = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
        if self.log:
            self.r0 /= f
            self.scale *= 1.0 + 0.05 * (f > 1) - 0.05 * (f < 1)
        else:
            # keep the world point under the cursor fixed
            c = np.array([self.width() / 2, self.height() / 2])
            mpos = np.array([ev.position().x(), ev.position().y()])
            w = self.centre + np.array([1, -1]) * (mpos - c) / self.scale
            self.scale *= f
            self.centre = w - np.array([1, -1]) * (mpos - c) / self.scale
            self.follow = None
        self.update()

    def mousePressEvent(self, ev):
        self._drag = (ev.position(), self.centre.copy())

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        d = ev.position() - self._drag[0]
        if self.log:
            k = self.px_per_au_at(0)
        else:
            k = self.scale
        self.centre = self._drag[1] - np.array([d.x(), -d.y()]) / k
        self.follow = None
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None and (ev.position() - self._drag[0]).manhattanLength() < 5:
            x, y = ev.position().x(), ev.position().y()
            best = None
            for hx, hy, r, name in self._hit:
                d = math.hypot(hx - x, hy - y)
                if d <= max(r, 8) and (best is None or d < best[0]):
                    best = (d, name)
            if best:
                self.bodyClicked.emit(best[1])
        self._drag = None

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.m is not None and self.follow is None and not self.centre.any():
            self.fit()


__all__ = ["OrbitView"]
