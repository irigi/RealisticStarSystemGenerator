"""The system viewer window: system map, orbit view, body tree and information panel, with
Previous / Next over a source of systems (a fixed seed sequence, or a bundle such as the
conditioned realisations of a real star)."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QSlider, QSplitter, QStatusBar, QTextBrowser, QToolBar,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import describe as ds
from .. import export as ex
from . import info
from . import model as md
from .orbits import OrbitView
from .sysmap import SystemMapView, fmt_km

# ------------------------------------------------------------------------------- sources


class SequenceSource:
    """Systems ``S<master>-<k>`` of :mod:`starsys.ids` (unbounded)."""

    def __init__(self, master: int = 1):
        self.master = master
        self._last = None

    def __len__(self):
        return 10**12

    def bundle(self, k: int):
        from .. import ids
        if self._last and self._last[0] == k:
            return self._last[1]
        b = ids.system(ids.SeqID(self.master, k))
        self._last = (k, b)
        return b

    def get(self, k: int):
        return self.bundle(k), 0

    def label(self, k: int) -> str:
        return f"S{self.master}-{k}"

    def index_of(self, text: str) -> int | None:
        from .. import ids
        try:
            sid = ids.parse(text)
        except ValueError:
            return None
        if sid.master != self.master:
            self.master = sid.master
            self._last = None
        return sid.k


class BundleSource:
    """The systems of one :class:`starsys.generate.Systems` bundle."""

    def __init__(self, bundle, title: str = ""):
        self.b = bundle
        self.title = title

    def __len__(self):
        return len(self.b)

    def get(self, k: int):
        return self.b, k

    def label(self, k: int) -> str:
        return ds.system_label(self.b, k) if self.b.meta.get("ids") else f"{self.title} #{k}"

    def index_of(self, text: str) -> int | None:
        t = text.strip().split("#")[-1]
        return int(t) if t.isdigit() and int(t) < len(self.b) else None


FILTERS = {
    "all systems": {},
    "single star": {"single": True},
    "F/G/K primary": {"spectral": "FGK"},
    "G primary": {"spectral": "G"},
    "M dwarf primary": {"spectral": "M"},
    "habitable-zone rocky planet": {"habitable": True},
    "≥ 5 planets": {"min_planets": 5},
    "multiple stars": {"multiple": True},
    "giant planet": {"giant": True},
}


def passes(bundle, i: int, f: dict) -> bool:
    import numpy as np

    from ..__main__ import select
    if not f:
        return True
    g = dict(f)
    multiple, giant = g.pop("multiple", False), g.pop("giant", False)
    if multiple and bundle.systems["n_components"][i] < 2:
        return False
    if giant:
        sel = np.asarray(bundle.planets["system"]) == i
        if not (np.asarray(bundle.planets["planet_mass"])[sel] > 50).any():
            return False
    return (not g) or i in set(select(bundle, **g).tolist())


# ------------------------------------------------------------------------------- window

DARK = """
QMainWindow, QWidget { background: #0b0f1a; color: #d8dde8; }
QToolBar { background: #121a2b; border: 0; spacing: 6px; padding: 4px; }
QPushButton { background: #1d2a44; border: 1px solid #2c3d60; padding: 4px 10px; border-radius: 3px; }
QPushButton:hover { background: #26375a; }
QPushButton:checked { background: #34507e; }
QLineEdit, QComboBox { background: #141d30; border: 1px solid #2c3d60; padding: 3px; }
QTreeWidget, QTextBrowser { background: #0e1422; border: 0; }
QTreeWidget::item:selected { background: #24406e; }
QHeaderView::section { background: #121a2b; color: #8fa0bc; border: 0; padding: 3px; }
QSplitter::handle { background: #1a2438; }
QStatusBar { background: #121a2b; color: #8fa0bc; }
QTabBar::tab { background: #121a2b; padding: 5px 12px; }
QTabBar::tab:selected { background: #24406e; }
"""


class SystemWindow(QMainWindow):
    def __init__(self, source, start: int = 0, filt: str = "all systems"):
        super().__init__()
        self.source = source
        self.k = start
        self.model: md.SystemModel | None = None
        self.selected: str | None = None
        self.setWindowTitle("starsys — system viewer")
        self.resize(1500, 920)
        self.setStyleSheet(DARK)

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        self.prev_b = QPushButton("◀ Previous")
        self.next_b = QPushButton("Next ▶")
        self.id_edit = QLineEdit()
        self.id_edit.setFixedWidth(150)
        self.id_edit.setToolTip("system ID; type one and press Enter")
        self.filter = QComboBox()
        self.filter.addItems(list(FILTERS))
        self.filter.setCurrentText(filt)
        self.filter.setToolTip("Previous / Next skip systems that do not match")
        exp_b = QPushButton("Export JSON…")
        txt_b = QPushButton("Copy summary")
        for w in (self.prev_b, self.id_edit, self.next_b, QLabel("  show:"), self.filter, exp_b, txt_b):
            tb.addWidget(w)
        self.prev_b.clicked.connect(lambda: self.step(-1))
        self.next_b.clicked.connect(lambda: self.step(+1))
        self.id_edit.returnPressed.connect(self.goto_text)
        exp_b.clicked.connect(self.export_json)
        txt_b.clicked.connect(self.copy_text)
        for seq, d in ((QKeySequence(Qt.Key.Key_PageDown), +1), (QKeySequence(Qt.Key.Key_PageUp), -1)):
            a = QAction(self)
            a.setShortcut(seq)
            a.triggered.connect(lambda _=False, d=d: self.step(d))
            self.addAction(a)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["body", "distance"])
        self.tree.setColumnWidth(0, 150)
        self.tree.itemClicked.connect(lambda it, _c: self.select(it.data(0, Qt.ItemDataRole.UserRole)))

        self.map = SystemMapView()
        self.map.bodyClicked.connect(self.select)
        self.orbits = OrbitView()
        self.orbits.bodyClicked.connect(self.select)

        ob = QWidget()
        ol = QHBoxLayout(ob)
        ol.setContentsMargins(6, 2, 6, 2)
        self.log_cb = QCheckBox("log distances")
        self.log_cb.toggled.connect(self.orbits.set_log)
        self.moon_cb = QCheckBox("moons")
        self.moon_cb.setChecked(True)
        self.moon_cb.toggled.connect(self._moons)
        fit_b = QPushButton("Fit")
        fit_b.clicked.connect(lambda: (setattr(self.orbits, "follow", None), self.orbits.fit()))
        focus_b = QPushButton("Centre on selection")
        focus_b.clicked.connect(lambda: self.orbits.focus(self.selected))
        self.size_s = QSlider(Qt.Orientation.Horizontal)
        self.size_s.setRange(2, 40)
        self.size_s.setValue(10)
        self.size_s.setFixedWidth(110)
        self.size_s.valueChanged.connect(lambda v: (setattr(self.orbits, "size_factor", v / 10), self.orbits.update()))
        self.speed_b = []
        self.time_l = QLabel()
        ol.addWidget(self.log_cb)
        ol.addWidget(self.moon_cb)
        ol.addWidget(QLabel("body size"))
        ol.addWidget(self.size_s)
        ol.addWidget(fit_b)
        ol.addWidget(focus_b)
        ol.addStretch(1)
        for label, rate in (("⏸", 0.0), ("▶", 1.0), ("▶▶", 10.0), ("▶▶▶", 100.0), ("▶▶▶▶", 1000.0)):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip("pause" if rate == 0 else f"{rate:g} days per second")
            b.clicked.connect(lambda _=False, r=rate, b=b: self.set_rate(r, b))
            self.speed_b.append(b)
            ol.addWidget(b)
        ol.addWidget(self.time_l)
        self.speed_b[0].setChecked(True)
        self.rate = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

        lower = QWidget()
        ll = QVBoxLayout(lower)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        ll.addWidget(ob)
        ll.addWidget(self.orbits, 1)

        centre = QSplitter(Qt.Orientation.Vertical)
        centre.addWidget(self.map)
        centre.addWidget(lower)
        centre.setSizes([520, 400])

        self.info = QTextBrowser()
        self.info.document().setDefaultStyleSheet(info.STYLE)
        self.info.setMinimumWidth(320)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(centre)
        split.addWidget(self.info)
        split.setSizes([260, 900, 340])
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())
        QTimer.singleShot(0, lambda: self.load(self.k))

    # --- navigation ---------------------------------------------------------------------
    def load(self, k: int):
        bundle, i = self.source.get(k)
        d = ex.system_dict(bundle, i)
        d["system_id"] = self.source.label(k)
        self.k = k
        self.bundle, self.i = bundle, i
        self.model = md.build(d)
        self.id_edit.setText(self.source.label(k))
        self.map.set_system(self.model)
        self.orbits.set_system(self.model)
        lo, hi = self.orbits.extent()
        self.log_cb.setChecked(hi / lo > 30)  # wide systems start in log scale
        self._fill_tree()
        self.selected = None
        self.info.setHtml(info.system_html(self.model))
        self.prev_b.setEnabled(k > 0)
        self.next_b.setEnabled(k + 1 < len(self.source))
        self.statusBar().showMessage(f"{self.source.label(k)}: {d['n_stars']} star(s), {d['n_planets']} planet(s), "
                                     f"{d['n_moons']} moon(s) — click a body for details")

    def step(self, direction: int):
        f = FILTERS[self.filter.currentText()]
        k = self.k
        for tries in range(20000):
            k += direction
            if k < 0 or k >= len(self.source):
                self.statusBar().showMessage("no further system matches the filter")
                return
            bundle, i = self.source.get(k)
            if passes(bundle, i, f):
                self.load(k)
                return
            if tries % 50 == 49:
                self.statusBar().showMessage(f"searching… {tries + 1} systems skipped")
                QApplication.processEvents()
        self.statusBar().showMessage("no match within 20000 systems")

    def goto_text(self):
        k = self.source.index_of(self.id_edit.text())
        if k is None:
            self.statusBar().showMessage(f"not an ID of this list: {self.id_edit.text()}")
            return
        self.load(k)

    # --- selection ----------------------------------------------------------------------
    def select(self, name: str | None):
        if not name or self.model is None or name not in self.model.bodies:
            return
        self.selected = name
        b = self.model.bodies[name]
        self.info.setHtml(info.body_html(self.model, b))
        self.map.select(name)
        self.orbits.selected = name
        # follow the selected body's host in the orbit view
        host = b.parent if b.cat in ("planet", "moon") else None
        if host and self.orbits.follow != host and b.cat == "planet":
            self.orbits.focus(host)
        elif b.cat == "moon" and self.orbits.follow != b.parent:
            self.orbits.focus(b.parent)
        self.orbits.update()
        it = self._items.get(name)
        if it is not None:
            self.tree.setCurrentItem(it)

    def _fill_tree(self):
        self.tree.clear()
        self._items = {}
        m = self.model
        for name in m.order:
            b = m.bodies[name]
            parent = self._items.get(b.parent) if b.cat != "star" else None
            if b.cat == "star" and b.parent:
                # companions hang under the first star they orbit, as in the hierarchy
                parent = self._items.get(b.parent.split("+")[0])
            o = b.orbit
            if b.cat == "moon":
                dist = fmt_km(b.d["a_km"])
            elif o:
                dist = f"{o[0]:.3g} au"
            else:
                dist = ""
            label = b.short or name if b.cat != "barycentre" else f"{name.replace('+', ' + ')} (barycentre)"
            it = QTreeWidgetItem([label, dist])
            it.setData(0, Qt.ItemDataRole.UserRole, name)
            if b.cat == "star":
                it.setForeground(0, QColor(255, 225, 150))
            if parent is None:
                self.tree.addTopLevelItem(it)
            else:
                parent.addChild(it)
            self._items[name] = it
        self.tree.expandAll()

    # --- time ---------------------------------------------------------------------------
    def set_rate(self, rate, button):
        self.rate = rate
        for b in self.speed_b:
            b.setChecked(b is button)

    def _tick(self):
        if self.rate:
            self.orbits.t += self.rate * self.timer.interval() / 1000.0
            self.orbits.update()
        t = self.orbits.t
        self.time_l.setText(f"  t = {t:,.1f} d" if t < 3650 else f"  t = {t / 365.25:,.1f} yr")

    def _moons(self, on):
        self.orbits.show_moons = on
        self.orbits.update()

    # --- export -------------------------------------------------------------------------
    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export system", f"{self.source.label(self.k)}.json",
                                              "JSON (*.json)")
        if path:
            obj = ex.to_json_obj(self.bundle, [self.i])
            with open(path, "w") as fh:
                json.dump(obj, fh, indent=1)
            self.statusBar().showMessage(f"wrote {path}")

    def copy_text(self):
        QApplication.clipboard().setText(ds.system_text(self.bundle, self.i))
        self.statusBar().showMessage("summary copied to the clipboard")


def dark_palette(app: QApplication):
    pal = QPalette()
    for role, col in ((QPalette.ColorRole.Window, "#0b0f1a"), (QPalette.ColorRole.Base, "#0e1422"),
                      (QPalette.ColorRole.Text, "#d8dde8"), (QPalette.ColorRole.WindowText, "#d8dde8"),
                      (QPalette.ColorRole.Button, "#1d2a44"), (QPalette.ColorRole.ButtonText, "#d8dde8"),
                      (QPalette.ColorRole.Highlight, "#24406e"), (QPalette.ColorRole.HighlightedText, "#ffffff")):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)


__all__ = ["BundleSource", "SequenceSource", "SystemWindow", "dark_palette"]
