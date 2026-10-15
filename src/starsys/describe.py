"""Readable labels and per-system summaries of a :class:`starsys.generate.Systems` bundle.

Labels (used by the text summaries and added to every export):

- **Names.** Stars are lettered by role (A primary, B, C ... companions; real stars keep their
  catalogue names). Planets take their host's letter and b, c, d ... in order of period (the
  real convention is discovery order); circumbinary planets are named after the inner pair
  ("AB b"). Known exoplanets of a real star keep their names. Moons: the planet's name and a
  Roman numeral in order of distance.
- **Planet class** (:func:`planet_class`): size word from :func:`starsys.derived.planet_kind`
  (rocky / envelope / giant) and radius or mass, with a temperature word from the
  equilibrium temperature and habitable zone: "hot" T_eq >= 1000 K, "warm" >= 400 K,
  "temperate" inside the optimistic habitable zone, "cold" otherwise, "frozen" < 150 K.
- **Temperatures (A9).** ``t_eq`` is the *equilibrium* temperature: Bond albedo 0.3, full heat
  redistribution, no greenhouse effect (the Earth: 255 K; its real mean surface is 288 K).
  :func:`surface_temperature` adds a rough, **illustrative** surface temperature for rocky
  planets (see there); envelope planets and giants have no surface.
"""

from __future__ import annotations

import math

import numpy as np

from . import derived as dv
from .flags import FLAG_KOZAI, FLAG_NAMES

EARTH_PER_JUPITER = 317.83
JUPITER_RADIUS_EARTH = 11.209
DAYS_PER_YEAR = 365.25

HZ_NAMES = {dv.HZ_TOO_HOT: "too hot", dv.HZ_OPT_INNER: "optimistic HZ (inner)",
            dv.HZ_CONSERVATIVE: "conservative HZ", dv.HZ_OPT_OUTER: "optimistic HZ (outer)",
            dv.HZ_TOO_COLD: "too cold", dv.HZ_UNDEFINED: "unlit"}
SPIN_NAMES = {dv.SPIN_FREE: "free", dv.SPIN_SYNC: "tidally locked", dv.SPIN_32: "3:2 spin-orbit"}
STAR_KINDS = {"star": "star", "BD": "brown dwarf", "WD": "white dwarf", "NS": "neutron star", "BH": "black hole"}

# --- illustrative surface temperature (A9) ----------------------------------------------------
# Earth's greenhouse factor: 288 K / 255 K (its T_eq with Bond albedo 0.3).
EARTH_GREENHOUSE = 288.0 / 255.0
# Inside the runaway-greenhouse limit: scaled to Venus, 737 K surface against 299 K T_eq (albedo 0.3,
# 1.91 Earth fluxes). Steam atmospheres of runaway planets reach 1500+ K (Goldblatt+2013), so this
# is a floor more than an estimate.
VENUS_GREENHOUSE = 737.0 / (dv.T_EQ_EARTH_FLUX * (0.7 * 1.91) ** 0.25)
# The carbonate-silicate cycle holds habitable-zone planets above freezing out to the maximum
# greenhouse limit (Kasting+1993; Kopparapu+2013 define that limit by a 273 K surface).
FREEZING = 273.0
AIRLESS_MASS = 0.1  # Earth masses: below this (Mars 0.107, Mercury 0.055) an atmosphere is thin or absent

(CLIMATE_NO_SURFACE, CLIMATE_AIRLESS, CLIMATE_RUNAWAY, CLIMATE_MARGINAL, CLIMATE_TEMPERATE, CLIMATE_FROZEN,
 CLIMATE_UNLIT) = range(7)
CLIMATE_NAMES = {CLIMATE_NO_SURFACE: "no surface (envelope)", CLIMATE_AIRLESS: "thin or no atmosphere",
                 CLIMATE_RUNAWAY: "runaway greenhouse", CLIMATE_MARGINAL: "marginal (near the runaway limit)",
                 CLIMATE_TEMPERATE: "temperate", CLIMATE_FROZEN: "frozen", CLIMATE_UNLIT: "unlit"}


def surface_temperature(planets) -> tuple[np.ndarray, np.ndarray]:
    """Illustrative mean surface temperature (K) and climate code of each planet (A9).

    Not a climate model; it assumes an Earth-like atmosphere wherever one is plausible:

    - envelope planets and giants: no surface (NaN);
    - rocky planets below 0.1 Earth masses: thin or no atmosphere, surface ~ T_eq;
    - inside the recent-Venus limit (habitable-zone code 0): runaway greenhouse, Venus-scaled,
      T_eq x 2.46 (a lower bound: steam atmospheres are hotter);
    - between recent Venus and the runaway limit (code 1, the optimistic inner zone): marginal,
      Earth's greenhouse, T_eq x 288/255 (Venus may have kept water this close until ~1 Gyr ago);
    - inside the conservative and outer optimistic zone (codes 2 and 3): Earth's greenhouse, held
      at least at 273 K by the carbonate-silicate cycle;
    - beyond it: Earth's greenhouse factor (a snowball).

    Tidally locked planets have large day-night contrasts that a mean temperature hides; early
    runaway phases may have removed their water (``t_runaway_myr``).
    """
    kind = dv.planet_kind(planets["planet_mass"], planets["planet_radius"], planets["lost_envelope"])
    t_eq = np.asarray(planets["t_eq"], dtype=float)
    hz = np.asarray(planets["hz"])
    mass = np.asarray(planets["planet_mass"], dtype=float)
    code = np.full(len(t_eq), CLIMATE_NO_SURFACE, dtype=np.int8)
    rocky = kind == 0
    code[rocky & (hz == dv.HZ_TOO_HOT)] = CLIMATE_RUNAWAY
    code[rocky & (hz == dv.HZ_OPT_INNER)] = CLIMATE_MARGINAL
    code[rocky & ((hz == dv.HZ_CONSERVATIVE) | (hz == dv.HZ_OPT_OUTER))] = CLIMATE_TEMPERATE
    code[rocky & (hz == dv.HZ_TOO_COLD)] = CLIMATE_FROZEN
    code[rocky & (mass < AIRLESS_MASS)] = CLIMATE_AIRLESS
    code[rocky & ((hz == dv.HZ_UNDEFINED) | ~np.isfinite(t_eq))] = CLIMATE_UNLIT
    t = np.full(len(t_eq), np.nan)
    t = np.where(code == CLIMATE_AIRLESS, t_eq, t)
    t = np.where(code == CLIMATE_RUNAWAY, t_eq * VENUS_GREENHOUSE, t)
    t = np.where(code == CLIMATE_TEMPERATE, np.maximum(t_eq * EARTH_GREENHOUSE, FREEZING), t)
    t = np.where((code == CLIMATE_FROZEN) | (code == CLIMATE_MARGINAL), t_eq * EARTH_GREENHOUSE, t)
    return t, code


def planet_class(planets) -> np.ndarray:
    """Short readable class of each planet, e.g. "temperate Earth-size", "hot Jupiter"."""
    m = np.asarray(planets["planet_mass"], dtype=float)
    r = np.asarray(planets["planet_radius"], dtype=float)
    kind = dv.planet_kind(m, r, planets["lost_envelope"])
    t_eq = np.asarray(planets["t_eq"], dtype=float) if "t_eq" in planets else np.full(len(m), np.nan)
    hz = np.asarray(planets["hz"]) if "hz" in planets else np.full(len(m), dv.HZ_UNDEFINED)
    out = np.empty(len(m), dtype=object)
    for i in range(len(m)):
        if kind[i] == 0:
            size = ("Mars-size" if r[i] < 0.6 else "sub-Earth" if r[i] < 0.85 else
                    "Earth-size" if r[i] < 1.25 else "super-Earth")
        elif kind[i] == 1:
            size = "sub-Neptune" if r[i] < 3.0 else "Neptune"
        else:
            mj = m[i] / EARTH_PER_JUPITER
            size = ("sub-Saturn" if m[i] < 60 else "Saturn" if mj < 0.6 else "Jupiter" if mj < 4 else
                    "super-Jupiter" if mj < 13 else "brown-dwarf-mass object")
        if not np.isfinite(t_eq[i]):
            temp = "dark"
        elif t_eq[i] >= 1000:
            temp = "hot"
        elif t_eq[i] >= 400 or hz[i] == dv.HZ_TOO_HOT:
            temp = "warm"
        elif hz[i] in (dv.HZ_OPT_INNER, dv.HZ_CONSERVATIVE, dv.HZ_OPT_OUTER):
            temp = "temperate"
        elif t_eq[i] < 150:
            temp = "frozen" if kind[i] == 0 else "cold"
        else:
            temp = "cool"
        if kind[i] == 1 and size == "Neptune" and temp in ("cold", "frozen"):
            size = "ice giant"
        out[i] = f"{temp} {size}"
    return out


# --- names -----------------------------------------------------------------------------------

def _letters(k: int) -> str:
    s = ""
    k += 1
    while k:
        k, rem = divmod(k - 1, 26)
        s = chr(ord("A") + rem) + s
    return s


def _roman(k: int) -> str:
    out = ""
    for val, sym in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                     (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while k >= val:
            out += sym
            k -= val
    return out


def star_name(bundle, role: int) -> str:
    return bundle.star_names.get(int(role), _letters(int(role)))


def names(bundle, i: int) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Names of the components, planets and moons of system ``i``, keyed by table row."""
    comp, pl, mo = bundle.rows(i)
    c, p = bundle.components, bundle.planets
    cn = {int(r): star_name(bundle, c["role"][r]) for r in comp}
    pn: dict[int, str] = {}
    known = np.asarray(p["known_name"]) if "known_name" in p else None
    groups: dict[tuple, list[int]] = {}
    for r in pl:
        groups.setdefault((int(p["component"][r]), str(p["host_kind"][r])), []).append(int(r))
    for (host, kind), rows in groups.items():
        rows = sorted(rows, key=lambda r: p["period"][r])
        if kind == "P":
            stem = cn[host] + star_name(bundle, c["role"][host] + 1) if not bundle.star_names else \
                f"{cn[host]}+{star_name(bundle, c['role'][host] + 1)}"
        else:
            stem = cn[host]
        used = {str(known[r]) for r in rows} if known is not None else set()
        letter = ord("b")
        for r in rows:
            if known is not None and known[r]:
                pn[r] = str(known[r])
                continue
            while f"{stem} {chr(letter)}" in used and letter < ord("z"):
                letter += 1
            pn[r] = f"{stem} {chr(letter)}"
            letter += 1
    mn: dict[int, str] = {}
    if len(mo):
        m = bundle.moons
        by_planet: dict[int, list[int]] = {}
        for r in mo:
            by_planet.setdefault(int(m["planet"][r]), []).append(int(r))
        for planet, rows in by_planet.items():
            for j, r in enumerate(sorted(rows, key=lambda r: m["a_km"][r]), start=1):
                mn[r] = f"{pn.get(planet, '?')} {_roman(j)}"
    return cn, pn, mn


# --- habitable-zone distances ----------------------------------------------------------------

def hz_distances(teff, luminosity) -> dict[str, float]:
    """Kopparapu+2014 habitable-zone boundaries (au) of a single star of Earth-mass planets:
    ``recent_venus``, ``runaway``, ``max_greenhouse``, ``early_mars`` (NaN for dark remnants)."""
    if not (np.isfinite(luminosity) and luminosity > 0 and np.isfinite(teff)):
        return {k: math.nan for k in ("recent_venus", "runaway", "max_greenhouse", "early_mars")}
    s = dv.s_eff(np.array([teff]), 1.0)
    return {k: float(np.sqrt(luminosity / v[0])) for k, v in s.items()}


# --- formatting -----------------------------------------------------------------------------

def _g(x, digits=3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    if abs(x) >= 10 ** digits or abs(x) < 10 ** -(digits - 1):
        return f"{x:.{digits - 1}e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return f"{x:.{digits}g}"


def fmt_period(days) -> str:
    if not np.isfinite(days):
        return "—"
    if days < 2:
        return f"{days * 24:.3g} h"
    if days < 1000:
        return f"{days:.3g} d"
    return f"{_g(days / DAYS_PER_YEAR)} yr"


def fmt_myr(myr) -> str:
    return f"{_g(myr / 1000)} Gyr" if myr >= 1000 else f"{myr:.0f} Myr"


def fmt_mass_earth(m) -> str:
    return f"{_g(m)} M⊕" if m < 50 else f"{_g(m / EARTH_PER_JUPITER)} MJ"


def fmt_radius_earth(r) -> str:
    return f"{_g(r)} R⊕" if r < 6 else f"{_g(r / JUPITER_RADIUS_EARTH)} RJ"


def _star_line(c, r, name) -> str:
    kind = STAR_KINDS.get(str(c["kind"][r]), str(c["kind"][r]))
    sp = str(c["spectral_type"][r]) if "spectral_type" in c else ""
    lum = c["luminosity"][r]
    parts = [f"{name}  {sp or kind}" + (f" ({kind})" if kind != "star" and sp else ""),
             f"{_g(c['mass'][r])} M☉", f"{_g(c['radius'][r])} R☉" if np.isfinite(c["radius"][r]) else None,
             f"{c['teff'][r]:.0f} K" if np.isfinite(c["teff"][r]) else None,
             f"{_g(lum)} L☉" if np.isfinite(lum) and lum > 0 else None]
    return " · ".join(p for p in parts if p)


def system_label(bundle, i: int) -> str:
    """The system's stable ID (``bundle.meta["ids"]``, see :mod:`starsys.ids`) or its row number."""
    ids = bundle.meta.get("ids") or []
    return str(ids[i]) if i < len(ids) and ids[i] else str(i)


def system_text(bundle, i: int, moons: bool = True) -> str:
    """Readable multi-line summary of system ``i``."""
    s, c, p = bundle.systems, bundle.components, bundle.planets
    comp, pl, mo = bundle.rows(i)
    cn, pn, mn = names(bundle, i)
    feh = s["feh"][i]
    kinds, counts = np.unique([str(c["kind"][r]) for r in comp], return_counts=True)
    order = list(STAR_KINDS)
    bodies = ", ".join(f"{n} {STAR_KINDS.get(k, k)}{'s' * int(n > 1)}"
                       for k, n in sorted(zip(kinds, counts), key=lambda kn: order.index(kn[0]) if kn[0] in order else 9))
    lines = [f"System {system_label(bundle, i)} · age {_g(s['age_gyr'][i])} Gyr · [Fe/H] {feh:+.2f} · "
             f"{bodies} · {len(pl)} planet{'s' * (len(pl) != 1)}"
             f" · {len(mo)} moon{'s' * (len(mo) != 1)}"]
    if "real_star" in bundle.meta:
        lines[0] = f"{bundle.meta['real_star']} (realisation {i}) · " + lines[0].split(" · ", 1)[1]
    t_surf, climate = surface_temperature(p) if len(p["a_au"]) and "t_eq" in p else (None, None)
    cls = planet_class(p) if len(p["a_au"]) else None
    hosts = [(int(r), "S") for r in comp]
    if len(comp) > 1:
        hosts.insert(2, (int(comp[0]), "P"))  # after the inner pair
    moon_rows: dict[int, list[int]] = {}
    for r in mo:
        moon_rows.setdefault(int(bundle.moons["planet"][r]), []).append(int(r))

    for host, kind in hosts:
        if kind == "S":
            lines.append("")
            lines.append(_star_line(c, host, cn[host]))
            role = int(c["role"][host])
            if role > 0:
                inner = "+".join(cn[int(r)] for r in comp[:role]) if role > 1 else cn[int(comp[0])]
                lines.append(f"   orbits {inner}: a {_g(c['a_au'][host])} au, e {c['ecc'][host]:.2f}, "
                             f"P {fmt_period(c['period_days'][host])}")
            hz = hz_distances(c["teff"][host], c["luminosity"][host])
            if np.isfinite(hz["runaway"]) and str(c["kind"][host]) in ("star", "BD", "WD"):
                lines.append(f"   habitable zone {_g(hz['runaway'])}–{_g(hz['max_greenhouse'])} au "
                             f"(optimistic {_g(hz['recent_venus'])}–{_g(hz['early_mars'])} au)")
        rows = [int(r) for r in pl if int(p["component"][r]) == host and str(p["host_kind"][r]) == kind]
        if not rows:
            continue
        if kind == "P":
            lines.append("")
            lines.append(f"Circumbinary planets (orbiting {cn[host]} and {cn[int(comp[1])]}):")
        wn = max(12, *(len(pn[r]) for r in rows))
        wc = max(5, *(len(cls[r]) for r in rows))
        lines.append(f"   {'planet':<{wn}} {'class':<{wc}} {'a':>10} {'P':>9} {'e':>5} {'mass':>9} {'radius':>9} "
                     f"{'S':>7} {'T_eq*':>6} {'T_surf†':>7}  notes")
        for r in sorted(rows, key=lambda r: p["a_au"][r]):
            notes = []
            if "hz" in p and p["hz"][r] in (dv.HZ_OPT_INNER, dv.HZ_CONSERVATIVE, dv.HZ_OPT_OUTER):
                notes.append(HZ_NAMES[int(p["hz"][r])])
            if "spin" in p and p["spin"][r] != dv.SPIN_FREE:
                notes.append(SPIN_NAMES[int(p["spin"][r])])
            elif "rotation_h" in p and np.isfinite(p["rotation_h"][r]):
                notes.append(f"day {fmt_period(p['rotation_h'][r] / 24)}")
            if (climate is not None and climate[r] in (CLIMATE_MARGINAL, CLIMATE_TEMPERATE)
                    and p["t_runaway_myr"][r] > 100):
                notes.append(f"runaway for {fmt_myr(p['t_runaway_myr'][r])}: water loss likely")
            if p["flags"][r] & FLAG_KOZAI:
                notes.append("Kozai")
            if "known_name" in p and p["known_name"][r]:
                notes.append("known")
            n_m = len(moon_rows.get(r, []))
            if n_m:
                notes.append(f"{n_m} moon{'s' * (n_m > 1)}")
            ts = t_surf[r] if t_surf is not None else np.nan
            lines.append(f"   {pn[r]:<{wn}} {cls[r]:<{wc}} {_g(p['a_au'][r]) + ' au':>10} {fmt_period(p['period'][r]):>9} "
                         f"{p['ecc'][r]:5.2f} {fmt_mass_earth(p['planet_mass'][r]):>9} "
                         f"{fmt_radius_earth(p['planet_radius'][r]):>9} "
                         f"{_g(p['insolation'][r]) if 'insolation' in p else '—':>7} "
                         f"{(f'{p['t_eq'][r]:.0f}' if 't_eq' in p and np.isfinite(p['t_eq'][r]) else '—'):>6} "
                         f"{(f'{ts:.0f}' if np.isfinite(ts) else '—'):>7}  {', '.join(notes)}")
            if moons:
                for mr in sorted(moon_rows.get(r, []), key=lambda m: bundle.moons["a_km"][m]):
                    m = bundle.moons
                    heat = (f", tidal heating {_g(m['heat_flux_w_m2'][mr])} W/m²"
                            if m["heat_flux_w_m2"][mr] > 0.01 else "")
                    lines.append(f"      {mn[mr]:<{wn + 2}} {m['kind'][mr]} {m['composition'][mr]} moon, "
                                 f"{fmt_mass_earth(m['mass'][mr])}, {fmt_radius_earth(m['radius'][mr])}, "
                                 f"a {_g(m['a_km'][mr])} km ({_g(m['a_rp'][mr])} planet radii), "
                                 f"P {fmt_period(m['period'][mr])}{heat}")
    if len(pl):
        lines.append("")
        lines.append("   S: insolation (Earth = 1). * T_eq: equilibrium temperature (K), Bond albedo 0.3, no greenhouse"
                     " (Earth 255 K).")
        lines.append("   † T_surf: illustrative surface temperature (K) of rocky planets, Earth-like atmosphere assumed"
                     " (see starsys.describe).")
    return "\n".join(lines)


def flag_names(flags: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES.items() if flags & bit]


__all__ = ["planet_class", "surface_temperature", "names", "system_text", "hz_distances", "HZ_NAMES",
           "SPIN_NAMES", "CLIMATE_NAMES", "STAR_KINDS", "flag_names"]
