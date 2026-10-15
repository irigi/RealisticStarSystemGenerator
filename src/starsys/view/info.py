"""Plain-language information about one body (HTML for the viewer's info panel)."""

from __future__ import annotations

import html
import math

from .. import describe as ds
from .model import Body, SystemModel


def _f(x, digits=3, unit=""):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    x = float(x)
    if x == 0:
        s = "0"
    elif abs(x) >= 1e5 or abs(x) < 1e-3:
        s = f"{x:.{digits - 1}e}"
    else:
        s = f"{x:.{digits}g}"
    return f"{s} {unit}".strip()


def _period(days):
    return "—" if days is None else ds.fmt_period(float(days))


def _hours(h):
    if h is None:
        return "—"
    return f"{h:.3g} h" if h < 48 else ds.fmt_period(h / 24.0)


def _row(k, v):
    return f"<tr><td class='k'>{html.escape(k)}</td><td>{v}</td></tr>"


def sentence(b: Body) -> str:
    d = b.d
    if b.cat == "star":
        kind = d.get("kind", "star")
        if kind == "star":
            return f"{d.get('spectral_type', '')} star, {_f(d.get('age_gyr'), 2)} Gyr old."
        return f"{kind.capitalize()} ({d.get('spectral_type', '')})."
    if b.cat == "barycentre":
        return "Barycentre of " + " and ".join(d["members"]) + "; circumbinary planets orbit it."
    if b.cat == "moon":
        kind = {"regular": "Regular", "impact": "Impact-generated", "captured": "Captured"}.get(d.get("kind"), "")
        comp = "icy" if d.get("composition") == "ice" else "rocky"
        s = f"{kind} {comp} moon"
        flux = d.get("heat_flux_w_m2") or 0
        if flux > 30:
            s += ", tidally molten (magma ocean)"
        elif flux > 1:
            s += ", volcanically active (Io-like tidal heating)"
        elif flux > 0.05 and comp == "icy":
            s += ", tidally heated (possible subsurface ocean)"
        return s + "."
    cls = d.get("class", "planet")
    parts = [cls[0].upper() + cls[1:]]
    typ = d.get("type")
    parts.append({"rocky": "rocky surface", "envelope": "hydrogen-helium or volatile envelope",
                  "giant": "gas giant"}.get(typ, ""))
    hz = d.get("habitable_zone")
    if hz and hz not in ("too hot", "too cold", "undefined"):
        parts.append(f"in the {hz} habitable zone")
    if d.get("climate") and typ == "rocky":
        parts.append(f"climate: {d['climate']}")
    if d.get("spin") == "locked":
        parts.append("tidally locked")
    if d.get("known"):
        parts.append("known planet")
    return ", ".join(p for p in parts if p) + "."


def body_html(m: SystemModel, b: Body) -> str:
    d = b.d
    rows = []
    if b.cat == "star":
        rows += [_row("Type", html.escape(str(d.get("spectral_type")))),
                 _row("Mass", _f(d.get("mass"), 3, "M☉")), _row("Radius", _f(d.get("radius"), 3, "R☉")),
                 _row("Temperature", _f(d.get("teff"), 4, "K")),
                 _row("Luminosity", _f(d.get("luminosity"), 3, "L☉")),
                 _row("Age", _f(d.get("age_gyr"), 3, "Gyr")), _row("[Fe/H]", _f(d.get("feh"), 2))]
        if d.get("cooling_age_gyr"):
            rows.append(_row("Cooling age", _f(d["cooling_age_gyr"], 3, "Gyr")))
        o = d.get("orbit")
        if o:
            rows += [_row("Orbits", " + ".join(o["around"])), _row("Semi-major axis", _f(o["a_au"], 4, "au")),
                     _row("Eccentricity", _f(o.get("ecc"), 3)), _row("Period", _period(o["period_days"]))]
        hz = d.get("habitable_zone_au") or {}
        if hz:
            rows.append(_row("Habitable zone", f"{_f(hz['runaway'])}–{_f(hz['max_greenhouse'])} au "
                                               f"(optimistic {_f(hz['recent_venus'])}–{_f(hz['early_mars'])})"))
    elif b.cat == "barycentre":
        rows.append(_row("Members", " + ".join(d["members"])))
    elif b.cat == "planet":
        g = None
        if d.get("mass") and d.get("radius"):
            g = 9.81 * d["mass"] / d["radius"] ** 2
        rows += [_row("Mass", _f(d.get("mass"), 3, "M⊕") + f" ({_f((d.get('mass') or 0) / 317.8, 3)} M<sub>J</sub>)"
                      if (d.get("mass") or 0) > 30 else _f(d.get("mass"), 3, "M⊕")),
                 _row("Radius", _f(d.get("radius"), 3, "R⊕")),
                 _row("Surface gravity" if d.get("type") == "rocky" else "Gravity (1 bar)",
                      _f(g, 3, "m/s²") + (f" ({g / 9.81:.2f} g)" if g else "")),
                 _row("Semi-major axis", _f(d.get("a_au"), 4, "au")),
                 _row("Periapsis / apoapsis", f"{_f(d['a_au'] * (1 - (d.get('ecc') or 0)), 3)} / "
                                              f"{_f(d['a_au'] * (1 + (d.get('ecc') or 0)), 3)} au"),
                 _row("Eccentricity", _f(d.get("ecc"), 3)), _row("Orbital period", _period(d.get("period_days"))),
                 _row("Insolation", _f(d.get("insolation"), 3, "× Earth")),
                 _row("Equilibrium temperature*", _f(d.get("t_eq_k"), 3, "K")),
                 _row("Surface temperature†", _f(d.get("t_surface_illustrative_k"), 3, "K")),
                 _row("Habitable zone", html.escape(str(d.get("habitable_zone", "—")))),
                 _row("Spin", html.escape(str(d.get("spin", "—")))),
                 _row("Day length (sidereal)", _hours(d.get("rotation_h"))),
                 _row("Axial tilt", _f(d.get("obliquity_deg"), 3, "°")),
                 _row("Moons", str(len(b.children)))]
        if d.get("t_runaway_myr"):
            rows.append(_row("Early runaway phase", _f(d["t_runaway_myr"], 3, "Myr")))
        rows.append(_row("Origin (model)", html.escape(str(d.get("model", "—")))))
        if d.get("flags"):
            rows.append(_row("Flags", html.escape(", ".join(d["flags"]))))
    else:
        rows += [_row("Mass", _f(d.get("mass"), 3, "M⊕") + f" ({_f((d.get('mass') or 0) / 0.0123, 3)} Moon)"),
                 _row("Radius", _f(d.get("radius"), 3, "R⊕") + f" ({_f((d.get('radius') or 0) * 6371, 4, 'km')})"),
                 _row("Distance", f"{d['a_km']:,.0f} km ({_f(d.get('a_rp'), 3)} planet radii)"),
                 _row("Orbital period", _period(d.get("period_days"))), _row("Eccentricity", _f(d.get("ecc"), 3)),
                 _row("Inclination", _f(d.get("incl_deg"), 3, "°") +
                      (" (retrograde)" if (d.get("incl_deg") or 0) > 90 else "")),
                 _row("Tidal heating", _f(d.get("tidal_heating_w"), 3, "W")),
                 _row("Surface heat flux", _f(d.get("heat_flux_w_m2"), 3, "W/m²") + " (Io ≈ 2)")]
    foot = ""
    if b.cat == "planet":
        foot = ("<p class='foot'>* no greenhouse effect, Bond albedo 0.3 (Earth 255 K).<br>"
                "† illustrative; Earth-like atmosphere assumed, rocky planets only.</p>")
    return (f"<h2>{html.escape(b.name)}</h2><p class='desc'>{html.escape(sentence(b))}</p>"
            f"<table>{''.join(rows)}</table>{foot}")


def system_html(m: SystemModel) -> str:
    d = m.d
    rows = [_row("ID", html.escape(m.sid)), _row("Age", _f(d.get("age_gyr"), 3, "Gyr")),
            _row("[Fe/H]", _f(d.get("feh"), 2)), _row("Stars", str(d.get("n_stars"))),
            _row("Planets", str(d.get("n_planets"))), _row("Moons", str(d.get("n_moons")))]
    return f"<h2>System {html.escape(m.sid)}</h2><table>{''.join(rows)}</table>"


STYLE = """
body { color: #d8dde8; font-family: sans-serif; }
h2 { color: #ffffff; font-weight: normal; margin: 2px 0 4px 0; }
p.desc { color: #c0c8d8; margin: 0 0 8px 0; }
td.k { color: #8fa0bc; padding-right: 12px; }
td { padding: 1px 0; }
p.foot { color: #7a879c; font-size: small; }
"""

__all__ = ["STYLE", "body_html", "sentence", "system_html"]
