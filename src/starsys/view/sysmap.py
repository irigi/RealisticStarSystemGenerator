"""Hierarchical schematic system map (in the style of Pioneer Space Simulator's system view).

One row per star (and one for circumbinary planets around a pair): the star on the left, its
planets to the right in order of distance, each planet's moons stacked below it. Body sizes
follow ``R^0.5`` (monotonic, so relative sizes keep their order and Jupiter/Earth/Moon stay
recognisably different; stars are capped). Lines show the hierarchy: companions are joined
to the stars they orbit, planets to their host.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsView

from . import art
from .model import SystemModel

BG = QColor(8, 11, 20)
GRID = QColor(30, 40, 62)
LINE = QColor(70, 90, 125)
TEXT = QColor(205, 214, 230)
SUBTEXT = QColor(130, 145, 170)
SELECT = QColor(120, 190, 255)

_cache: dict[tuple[int, int], QPixmap] = {}


def display_px(radius_earth: float, star: bool = False) -> float:
    px = 22.0 * max(radius_earth, 1e-3) ** 0.5
    return float(np.clip(px, 26 if star else 8, 180 if star else 120))


def fmt_km(km: float) -> str:
    return f"{km:,.0f} km" if km < 1e7 else f"{km / 1.496e8:.3g} au"


def pixmap(body, px: int) -> QPixmap:
    """Cached artwork of a body (render size includes the stellar glow)."""
    key = (body.seed, px)
    pm = _cache.get(key)
    if pm is None:
        if len(_cache) > 2000:
            _cache.clear()
        arr = np.ascontiguousarray(art.render(body.d, px, body.seed))
        img = QImage(arr.data, px, px, 4 * px, QImage.Format.Format_RGBA8888).copy()
        pm = QPixmap.fromImage(img)
        _cache[key] = pm
    return pm


def _au(a):
    return f"{a:.3g} au" if a >= 0.01 else f"{a * 1.496e8:.3g} km"


class SystemMapView(QGraphicsView):
    bodyClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(BG))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.items_by_name: dict[str, QGraphicsEllipseItem] = {}
        self.ring = None
        self.model: SystemModel | None = None
        self._press = None

    # --- drawing ------------------------------------------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, BG)
        painter.setPen(QPen(GRID, 0))
        step = 60
        x0 = int(rect.left() // step) * step
        y0 = int(rect.top() // step) * step
        x = x0
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = y0
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

    def set_system(self, m: SystemModel):
        self.model = m
        sc = self.scene()
        sc.clear()
        self.items_by_name.clear()
        self.ring = None
        font, small = QFont(), QFont()
        font.setPointSizeF(9.5)
        small.setPointSizeF(8)
        y = 0.0
        row_y: dict[str, float] = {}
        rows_left = 40.0
        for root in m.roots:
            rb = m.bodies[root]
            planets = [m.bodies[c] for c in sorted(rb.children, key=lambda c: m.bodies[c].orbit[0]
                                                    if m.bodies[c].orbit else 0) if m.bodies[c].cat == "planet"]
            star_px = display_px(rb.radius_earth, True) if rb.cat == "star" else 26
            sizes = [display_px(p.radius_earth) for p in planets]
            moon_h = [sum(display_px(m.bodies[c].radius_earth) + 30 for c in p.children) + 16 * bool(p.children) for p in planets]
            top_h = max([star_px] + sizes)
            cy = y + top_h / 2 + 10
            row_y[root] = cy
            # central body
            if rb.cat == "star":
                glow = int(star_px / 0.72)
                self._add_body(rb, rows_left + star_px / 2, cy, star_px, glow, font, small,
                               rb.d.get("spectral_type", ""))
            else:
                self._add_barycentre(rb, rows_left + star_px / 2, cy, font)
            x = rows_left + star_px + 50
            if planets:
                sc.addLine(rows_left + star_px, cy, x + sum(sizes) + 70 * len(planets) - 70, cy,
                           QPen(LINE, 1.2))
            for p, s, mh in zip(planets, sizes, moon_h):
                w = max(s, 64)
                px_x = x + w / 2
                self._add_body(p, px_x, cy, s, int(s), font, small, _au(p.d["a_au"]))
                my = cy + s / 2 + 46
                moons = [m.bodies[c] for c in p.children]
                if moons:
                    sc.addLine(px_x, cy + s / 2 + 36, px_x, my + mh - 24, QPen(LINE, 1))
                for mo in moons:
                    ms = display_px(mo.radius_earth)
                    self._add_body(mo, px_x + 0, my + ms / 2, ms, int(max(ms, 4)), small, small,
                                   fmt_km(mo.d['a_km']), label_right=True)
                    my += ms + 30
                x += w + 70
            y = cy + top_h / 2 + max([0] + moon_h) + 60
        # hierarchy brackets between stars
        for b in m.stars:
            o = b.d.get("orbit")
            if not o:
                continue
            inner = [n for n in o["around"] if n in row_y]
            if not inner:
                continue
            y0, y1 = min(row_y[n] for n in inner), row_y[b.name]
            xb = rows_left - 18 - 8 * b.d.get("role", 1)
            pen = QPen(SELECT.darker(150), 1.5)
            sc.addLine(xb, y0, xb, y1, pen)
            sc.addLine(xb, y0, rows_left - 4, y0, pen)
            sc.addLine(xb, y1, rows_left - 4, y1, pen)
            t = sc.addSimpleText(f"{_au(o['a_au'])}, e {o.get('ecc') or 0:.2f}", small)
            t.setBrush(SUBTEXT)
            t.setRotation(-90)
            t.setPos(xb - 16, (y0 + y1) / 2 + t.boundingRect().width() / 2)
        for root in m.roots:
            rb = m.bodies[root]
            if rb.cat == "barycentre":
                mem = [row_y[n] for n in rb.d["members"] if n in row_y]
                if mem:
                    pen = QPen(QColor(200, 170, 90), 1.2, Qt.PenStyle.DashLine)
                    xb = rows_left - 30
                    sc.addLine(xb, min(mem), xb, row_y[root], pen)
                    sc.addLine(xb, row_y[root], rows_left, row_y[root], pen)
        r = sc.itemsBoundingRect().adjusted(-60, -40, 60, 40)
        sc.setSceneRect(r)
        self.resetTransform()
        self.fit()

    def fit(self):
        r = self.scene().sceneRect()
        if r.width() <= 0:
            return
        k = max(min(self.viewport().width() / r.width(), self.viewport().height() / r.height(), 1.0), 0.4)
        self.resetTransform()
        self.scale(k, k)
        vw, vh = self.viewport().width() / k, self.viewport().height() / k
        cx = r.center().x() if r.width() <= vw else r.left() + vw / 2
        cy = r.center().y() if r.height() <= vh else r.top() + vh / 2
        self.centerOn(cx, cy)

    def _add_body(self, b, x, y, size, render_px, font, small, sub, label_right=False):
        sc = self.scene()
        render_px = max(int(render_px), 6)
        pm = pixmap(b, render_px)
        it = sc.addPixmap(pm)
        it.setOffset(x - render_px / 2, y - render_px / 2)
        it.setData(0, b.name)
        it.setCursor(Qt.CursorShape.PointingHandCursor)
        it.setToolTip(f"{b.name}: {b.d.get('class') or b.d.get('spectral_type') or b.d.get('kind', '')}")
        hit = QGraphicsEllipseItem(x - size / 2 - 3, y - size / 2 - 3, size + 6, size + 6)
        hit.setPen(QPen(Qt.PenStyle.NoPen))
        hit.setData(0, b.name)
        sc.addItem(hit)
        self.items_by_name[b.name] = hit
        name = sc.addSimpleText(b.short or b.name, font)
        name.setBrush(TEXT)
        sub_t = sc.addSimpleText(sub, small)
        sub_t.setBrush(SUBTEXT)
        if label_right:
            name.setPos(x + size / 2 + 8, y - name.boundingRect().height() + 2)
            sub_t.setPos(x + size / 2 + 8, y)
        else:
            nb = name.boundingRect()
            name.setPos(x - nb.width() / 2, y + size / 2 + 4)
            sub_t.setPos(x - sub_t.boundingRect().width() / 2, y + size / 2 + 4 + nb.height())

    def _add_barycentre(self, b, x, y, font):
        sc = self.scene()
        pen = QPen(QColor(220, 190, 110), 1.5)
        for dx in (-6, 6):
            sc.addEllipse(x + dx - 5, y - 5, 10, 10, pen, QBrush(QColor(255, 220, 140)))
        hit = QGraphicsEllipseItem(x - 16, y - 12, 32, 24)
        hit.setPen(QPen(Qt.PenStyle.NoPen))
        hit.setData(0, b.name)
        sc.addItem(hit)
        self.items_by_name[b.name] = hit
        t = sc.addSimpleText(b.name.replace("+", ""), font)
        t.setBrush(TEXT)
        t.setPos(x - t.boundingRect().width() / 2, y + 14)

    # --- selection & interaction --------------------------------------------------------------
    def select(self, name: str | None):
        sc = self.scene()
        if self.ring is not None:
            sc.removeItem(self.ring)
            self.ring = None
        it = self.items_by_name.get(name) if name else None
        if it is None:
            return
        r = it.rect().adjusted(-4, -4, 4, 4)
        self.ring = sc.addEllipse(r, QPen(SELECT, 2))
        self.ring.setZValue(10)
        self.ensureVisible(r, 80, 80)

    def mousePressEvent(self, ev):
        self._press = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._press is not None and (ev.position() - self._press).manhattanLength() < 5:
            for it in self.items(ev.position().toPoint()):
                name = it.data(0)
                if name:
                    self.bodyClicked.emit(str(name))
                    break
        self._press = None

    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)


__all__ = ["SystemMapView", "display_px", "pixmap"]
