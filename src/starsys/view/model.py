"""Viewer-side model of one system: a flat list of bodies built from the JSON dict of
:func:`starsys.export.system_dict`, and their positions at a given time.

Orbits are drawn face-on in one plane: each orbit keeps its size, eccentricity and the
in-plane angle (node + argument of pericentre), but mutual inclinations are ignored, which
is what a schematic top-down view can show. Stellar companions without a phase in the
generator get a mean anomaly from the body's art seed (decorative, but fixed).
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

AU_KM = 1.495978707e8
R_EARTH_AU = 6371.0 / AU_KM
R_SUN_AU = 695700.0 / AU_KM


@dataclass
class Body:
    name: str
    cat: str            # "star", "barycentre", "planet", "moon"
    d: dict             # the export dict (empty for barycentres)
    parent: str | None  # name of the body or barycentre it orbits
    seed: int = 0
    children: list[str] = field(default_factory=list)

    short: str = ""     # label within its star's row (planet "b" of star "Proxima Centauri")

    @property
    def radius_earth(self) -> float:
        """Radius in Earth radii (stars converted from solar radii)."""
        if self.cat == "star":
            return float(self.d.get("radius") or 0.01) * 109.08
        return float(self.d.get("radius") or 0.0)

    @property
    def orbit(self) -> tuple[float, float, float, float, float] | None:
        """(a_au, ecc, angle_rad, period_days, mean_anomaly0_rad) or None (central body)."""
        d = self.d
        if self.cat == "star":
            o = d.get("orbit")
            if not o:
                return None
            ang = math.radians((o.get("node_sky_deg") or 0) + (o.get("omega_deg") or 0))
            return o["a_au"], o.get("ecc") or 0.0, ang, o["period_days"], (self.seed % 6283) / 1000.0
        if self.cat == "planet":
            ang = math.radians((d.get("node_invariable_deg") or 0) + (d.get("omega_deg") or 0))
            return (d["a_au"], d.get("ecc") or 0.0, ang, d["period_days"],
                    math.radians(d.get("mean_anomaly_deg") or 0.0))
        if self.cat == "moon":
            return (d["a_km"] / AU_KM, d.get("ecc") or 0.0, (self.seed % 6283) / 1000.0, d["period_days"],
                    (self.seed // 7 % 6283) / 1000.0)
        return None


@dataclass
class SystemModel:
    sid: str
    d: dict
    bodies: dict[str, Body]
    order: list[str]       # display order: depth-first
    roots: list[str]

    @property
    def stars(self) -> list[Body]:
        return [b for b in self.bodies.values() if b.cat == "star"]


def _seed(sid: str, name: str) -> int:
    return zlib.crc32(f"{sid}/{name}".encode())


def build(d: dict) -> SystemModel:
    """Model of the system dict ``d`` (one entry of the JSON export's ``systems``)."""
    sid = str(d.get("system_id", d.get("id", "")))
    bodies: dict[str, Body] = {}

    def add(b: Body, host: str | None = None):
        b.seed = _seed(sid, b.name)
        b.short = b.name[len(host) + 1:] if host and b.name.startswith(host + " ") else b.name
        if host and b.short == b.name:
            # catalogue spellings of the host differ ("Proxima Cen b" around "Proxima Centauri"):
            # keep the planet letter (and moon numeral)
            m = re.search(r"\s([a-z](?:\s[IVXLC]+)?)$", b.name)
            if m:
                b.short = m.group(1)
        bodies[b.name] = b
        if b.parent is not None and b.parent in bodies:
            bodies[b.parent].children.append(b.name)
        return b

    stars = d.get("stars", [])
    for s in stars:
        around = (s.get("orbit") or {}).get("around")
        add(Body(s["name"], "star", s, None))
        bodies[s["name"]].parent = "+".join(around) if around else None
        for p in s.get("planets", []):
            add(Body(p["name"], "planet", p, s["name"]), s["name"] if len(s["name"]) > 2 else None)
            for m in p.get("moons", []):
                add(Body(m["name"], "moon", m, p["name"]), s["name"] if len(s["name"]) > 2 else None)
    cb_host = d.get("circumbinary_host") or []
    if d.get("circumbinary_planets"):
        bc = "+".join(cb_host)
        add(Body(bc, "barycentre", {"members": cb_host}, None))
        for p in d["circumbinary_planets"]:
            add(Body(p["name"], "planet", p, bc))
            for m in p.get("moons", []):
                add(Body(m["name"], "moon", m, p["name"]))
    order: list[str] = []

    def walk(n):
        order.append(n)
        for c in sorted(bodies[n].children, key=lambda c: _sort_a(bodies[c])):
            walk(c)

    roots = [s["name"] for s in stars] + ([ "+".join(cb_host)] if d.get("circumbinary_planets") else [])
    for r in roots:
        walk(r)
    return SystemModel(sid, d, bodies, order, roots)


def _sort_a(b: Body) -> float:
    o = b.orbit
    return o[0] if o else 0.0


def kepler(mean_anom, ecc, tol=1e-10):
    """Eccentric anomaly (vectorised Newton iteration)."""
    m = np.mod(np.asarray(mean_anom, float), 2 * np.pi)
    e = np.asarray(ecc, float)
    E = np.where(e > 0.8, np.pi, m)
    for _ in range(30):
        f = E - e * np.sin(E) - m
        E = E - f / (1 - e * np.cos(E))
        if np.all(np.abs(f) < tol):
            break
    return E


def orbit_xy(a, e, ang, E):
    """Position on a face-on orbit (focus at origin) for eccentric anomalies E."""
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e * e) * np.sin(E)
    c, s = math.cos(ang), math.sin(ang)
    return x * c - y * s, x * s + y * c


def positions(m: SystemModel, t_days: float) -> dict[str, tuple[float, float]]:
    """Positions (au, face-on plane, barycentre of the system at the origin) at time ``t_days``."""
    pos: dict[str, tuple[float, float]] = {}
    mass = {}
    stars = sorted(m.stars, key=lambda b: b.d.get("role", 0))
    com, m_in = np.zeros(2), 0.0
    for b in stars:
        mk = float(b.d.get("mass") or 0.1)
        o = b.orbit
        if o is None or m_in == 0:
            p = np.zeros(2)
        else:
            a, e, ang, per, m0 = o
            E = kepler(m0 + 2 * np.pi * t_days / per, e)
            # companion around the inner group's centre of mass
            p = com + np.array(orbit_xy(a, e, ang, E))
        pos[b.name] = (float(p[0]), float(p[1]))
        mass[b.name] = mk
        com = (com * m_in + p * mk) / (m_in + mk)
        m_in += mk
    # inner groups are displaced opposite to their companions so that the total COM is fixed;
    # for display the total barycentre is moved to the origin.
    for b in stars:
        pos[b.name] = (pos[b.name][0] - com[0], pos[b.name][1] - com[1])
    for name in m.order:
        b = m.bodies[name]
        if b.cat == "barycentre":
            mem = [n for n in b.d["members"] if n in pos]
            w = np.array([mass[n] for n in mem])
            xy = np.array([pos[n] for n in mem])
            pos[name] = tuple((w[:, None] * xy).sum(0) / w.sum()) if len(mem) else (0.0, 0.0)
        elif b.cat in ("planet", "moon"):
            a, e, ang, per, m0 = b.orbit
            E = kepler(m0 + 2 * np.pi * t_days / per, e)
            x, y = orbit_xy(a, e, ang, E)
            px, py = pos.get(b.parent, (0.0, 0.0))
            pos[name] = (px + float(x), py + float(y))
    return pos


def orbit_path(m: SystemModel, name: str, pos: dict, n: int = 240) -> np.ndarray | None:
    """Points (n x 2, au) of a body's orbit around its parent's current position."""
    b = m.bodies[name]
    o = b.orbit
    if o is None:
        return None
    a, e, ang, _, _ = o
    E = np.linspace(0, 2 * np.pi, n)
    x, y = orbit_xy(a, e, ang, E)
    if b.cat == "star":
        # companion orbit drawn around the inner group's centre of mass (approximate: current COM)
        inner = [s for s in m.stars if s.d.get("role", 0) < b.d.get("role", 0)]
        w = np.array([float(s.d.get("mass") or 0.1) for s in inner])
        c = (w[:, None] * np.array([pos[s.name] for s in inner])).sum(0) / w.sum()
    else:
        c = np.array(pos.get(b.parent, (0.0, 0.0)))
    return np.stack([x + c[0], y + c[1]], axis=1)


__all__ = ["Body", "SystemModel", "build", "kepler", "orbit_path", "orbit_xy", "positions"]
