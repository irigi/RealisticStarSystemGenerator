"""The real Solar System, in the export schema of :func:`starsys.export.system_dict`.

Opening the Sun shows this instead of a conditioned realisation. It holds the eight planets
and Pluto with their major moons.

Sources:

* **Planet orbits:** J2000 mean elements (a, e, i, mean longitude L, longitude of perihelion
  varpi, node Omega; ecliptic and equinox J2000) from Standish, "Keplerian elements for
  approximate positions of the major planets", JPL SSD, Table 1. The mean anomaly is
  M = L - varpi and the argument of perihelion is omega = varpi - Omega. Time 0 of the viewer
  is J2000.0. Angles are referred to the ecliptic, not to the invariable plane (1.6 deg apart).
* **Planet physical data:** NASA Planetary Fact Sheet (Williams, NSSDCA): mass, mean radius,
  sidereal rotation (negative = retrograde), obliquity and mean surface temperature for the
  four terrestrial planets.
* **Moons:** JPL SSD planetary satellite mean elements (a, e, period, inclination to the
  planet's equator or Laplace plane) and physical parameters (GM, mean radius).
* **Tidal heat:** Io 1e14 W (Lainey et al. 2009), Enceladus 1.5e10 W (Howett et al. 2011),
  Europa ~1e12 W (order of magnitude, model-dependent). Other moons get None (negligible).
* **Derived values:** equilibrium temperatures, insolation and habitable-zone labels use the
  generator's own conventions (Bond albedo 0.3, Kopparapu et al. 2014 limits) so that they
  compare with generated planets. The real values differ (Earth's actual albedo is 0.30, so
  this convention gives 255 K for Earth).
* **Artwork:** ``art`` gives the procedural renderer a palette hint (the planets' real
  colours); the surfaces themselves are still procedural (no real maps; no rings).
"""

from __future__ import annotations

import math

from .. import describe as ds

EARTH_KG = 5.9722e24
EARTH_KM = 6371.0
SIDEREAL_YEAR_D = 365.25636
T_EQ_1AU = 254.6  # K: 278.6 (0.7)^(1/4), the generator's convention (Bond albedo 0.3)

SUN = {"name": "Sun", "role": 0, "kind": "star", "spectral_type": "G2V", "mass": 1.0, "initial_mass": 1.0,
       "radius": 1.0, "teff": 5772.0, "luminosity": 1.0, "log_g": 4.438, "age_gyr": 4.57, "feh": 0.0,
       "mist_phase": 0.0, "cooling_age_gyr": None, "is_target": True, "orbit": None}

# name: (a au, e, i deg, L deg, varpi deg, Omega deg, mass Earth, radius Earth, rotation h,
#        obliquity deg, type, class, climate, surface K, art hint)
PLANETS = {
    "Mercury": (0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593,
                0.0553, 0.383, 1407.6, 0.034, "rocky", "sub-Earth", "thin or no atmosphere", 440.0,
                {"base": "#8c8680"}),
    "Venus": (0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255,
              0.815, 0.949, -5832.5, 177.4, "rocky", "Earth-size", "runaway greenhouse", 737.0, {}),
    "Earth": (1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
              1.0, 1.0, 23.9345, 23.44, "rocky", "temperate Earth-size", "temperate", 288.0, {}),
    "Mars": (1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891,
             0.107, 0.532, 24.6229, 25.19, "rocky", "Mars-size", "thin or no atmosphere", 210.0,
             {"base": "#b0623a"}),
    "Jupiter": (5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909,
                317.83, 10.97, 9.925, 3.13, "giant", "cold Jupiter", "no surface (envelope)", None, {"palette": 0}),
    "Saturn": (9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448,
               95.16, 9.14, 10.656, 26.73, "giant", "cold Saturn", "no surface (envelope)", None, {"palette": 1}),
    "Uranus": (19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503,
               14.54, 3.98, -17.24, 97.77, "envelope", "ice giant", "no surface (envelope)", None,
               {"base": "#7ac8d0"}),
    "Neptune": (30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574,
                17.15, 3.86, 16.11, 28.32, "envelope", "ice giant", "no surface (envelope)", None,
                {"base": "#3a68c8"}),
    "Pluto": (39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684,
              0.00218, 0.1868, -153.29, 122.53, "rocky", "dwarf planet", "frozen", 44.0, {"base": "#d8c8b0"}),
}

# planet: [(name, a km, period d, e, incl deg, GM km^3/s^2 -> mass, radius km, kind, composition,
#           tidal heat W, art hint)]
MOONS = {
    "Earth": [("Moon", 384400, 27.321661, 0.0549, 5.145, 4902.80, 1737.4, "impact", "rock", None, {})],
    "Mars": [("Phobos", 9376, 0.31891, 0.0151, 1.08, 7.087e-4, 11.08, "captured", "rock", None, {}),
             ("Deimos", 23463, 1.26244, 0.00033, 1.79, 9.62e-5, 6.2, "captured", "rock", None, {})],
    "Jupiter": [("Io", 421700, 1.769138, 0.0041, 0.050, 5959.92, 1821.5, "regular", "rock", 1.0e14, {}),
                ("Europa", 671034, 3.551181, 0.0094, 0.470, 3202.72, 1560.8, "regular", "ice", 1.0e12,
                 {"base": "#d8ccb8"}),
                ("Ganymede", 1070412, 7.154553, 0.0013, 0.204, 9887.83, 2631.2, "regular", "ice", None,
                 {"base": "#a89c8c"}),
                ("Callisto", 1882709, 16.689018, 0.0074, 0.205, 7179.29, 2410.3, "regular", "ice", None,
                 {"base": "#7c7064"})],
    "Saturn": [("Mimas", 185539, 0.942422, 0.0196, 1.574, 2.503, 198.2, "regular", "ice", None, {}),
               ("Enceladus", 237948, 1.370218, 0.0047, 0.009, 7.211, 252.1, "regular", "ice", 1.5e10,
                {"base": "#f0f4f8"}),
               ("Tethys", 294619, 1.887802, 0.0001, 1.091, 41.21, 531.0, "regular", "ice", None, {}),
               ("Dione", 377396, 2.736915, 0.0022, 0.028, 73.11, 561.4, "regular", "ice", None, {}),
               ("Rhea", 527108, 4.518212, 0.0013, 0.331, 153.94, 763.5, "regular", "ice", None, {}),
               ("Titan", 1221870, 15.945421, 0.0288, 0.306, 8978.14, 2574.7, "regular", "ice", None,
                {"base": "#d0a050"}),
               ("Iapetus", 3560820, 79.330183, 0.0283, 8.298, 120.52, 734.3, "regular", "ice", None, {})],
    "Uranus": [("Miranda", 129900, 1.413479, 0.0013, 4.338, 4.32, 235.8, "regular", "ice", None, {}),
               ("Ariel", 190900, 2.520379, 0.0012, 0.041, 83.46, 578.9, "regular", "ice", None, {}),
               ("Umbriel", 266000, 4.144177, 0.0039, 0.128, 85.09, 584.7, "regular", "ice", None,
                {"base": "#6c6a68"}),
               ("Titania", 436300, 8.705872, 0.0011, 0.079, 226.94, 788.9, "regular", "ice", None, {}),
               ("Oberon", 583500, 13.463239, 0.0014, 0.068, 205.32, 761.4, "regular", "ice", None, {})],
    "Neptune": [("Proteus", 117647, 1.122315, 0.0005, 0.524, 2.58, 210.0, "regular", "rock", None, {}),
                ("Triton", 354759, -5.876854, 0.0000, 156.865, 1427.6, 1353.4, "captured", "ice", None,
                 {"base": "#d8c8c0"}),
                ("Nereid", 5513818, 360.13619, 0.7507, 7.090, 2.06, 178.5, "captured", "ice", None, {})],
    "Pluto": [("Charon", 19591, 6.387230, 0.0002, 0.080, 106.1, 606.0, "impact", "ice", None, {"base": "#a8a0a0"})],
}
GM_EARTH = 398600.435  # km^3/s^2


def _hz_label(a_au, hz) -> str:
    s = 1.0 / a_au**2
    flux = {k: 1.0 / v**2 for k, v in hz.items()}
    if s > flux["recent_venus"]:
        return "too hot"
    if s > flux["runaway"]:
        return "optimistic HZ (inner)"
    if s > flux["max_greenhouse"]:
        return "conservative HZ"
    if s > flux["early_mars"]:
        return "optimistic HZ (outer)"
    return "too cold"


def _planet(name, row, hz) -> dict:
    a, e, i, L, varpi, node, mass, radius, rot, obl, typ, cls, climate, t_surf, art = row
    t_eq = T_EQ_1AU / math.sqrt(a)
    moons = []
    for mn, a_km, per, me, inc, gm, r_km, kind, comp, heat, mart in MOONS.get(name, []):
        r = r_km / EARTH_KM
        flux = heat / (4 * math.pi * (r_km * 1e3) ** 2) if heat else None
        moons.append({"name": mn, "kind": kind, "composition": comp, "mass": round(gm / GM_EARTH, 9),
                      "radius": round(r, 5), "a_km": float(a_km), "a_rp": round(a_km / (radius * EARTH_KM), 3),
                      "period_days": abs(per), "ecc": me, "incl_deg": inc, "tidal_heating_w": heat,
                      "heat_flux_w_m2": flux, "flags": ["real"], "art": mart})
    return {"name": name, "class": cls, "type": typ, "known": True, "a_au": a,
            "period_days": SIDEREAL_YEAR_D * a**1.5 / math.sqrt(1 + mass * 3.0035e-6), "ecc": e,
            "omega_deg": (varpi - node) % 360, "mean_anomaly_deg": (L - varpi) % 360,
            "incl_invariable_deg": i, "node_invariable_deg": node, "mass": mass, "radius": radius,
            "insolation": 1.0 / a**2, "t_eq_k": round(t_eq, 1), "t_surface_illustrative_k": t_surf,
            "climate": climate, "habitable_zone": _hz_label(a, hz), "spin": "free",
            "rotation_h": abs(rot), "obliquity_deg": obl, "model": "real (Solar System)", "flags": ["real"],
            "art": art, "moons": moons}


def system_dict() -> dict:
    """The Solar System as one entry of the JSON export's ``systems`` (ID ``R:Sun``)."""
    hz = ds.hz_distances(SUN["teff"], SUN["luminosity"])
    sun = dict(SUN, habitable_zone_au={k: round(v, 4) for k, v in hz.items()},
               planets=[_planet(n, row, hz) for n, row in PLANETS.items()])
    n_moons = sum(len(p["moons"]) for p in sun["planets"])
    return {"id": 0, "system_id": "R:Sun", "age_gyr": SUN["age_gyr"], "feh": 0.0, "n_stars": 1,
            "n_planets": len(PLANETS), "n_moons": n_moons, "stars": [sun], "circumbinary_planets": [],
            "circumbinary_host": None}


class SolarSystem:
    """Stands in for a :class:`starsys.generate.Systems` bundle of one system in the viewer."""

    meta = {"ids": ["R:Sun"]}

    def __len__(self):
        return 1

    def system_dict(self, i: int = 0) -> dict:
        return system_dict()

    def json_obj(self) -> dict:
        from ..export import SCHEMA, UNITS
        return {"schema": SCHEMA, "units": UNITS, "meta": self.meta, "systems": [system_dict()]}

    def text(self) -> str:
        d = system_dict()
        lines = ["R:Sun — the Solar System (real data; J2000 orbits)"]
        for p in d["stars"][0]["planets"]:
            moons = ", ".join(m["name"] for m in p["moons"])
            lines.append(f"  {p['name']}: {p['class']}, a = {p['a_au']:.3f} au, {p['mass']:.3g} Earth masses"
                         + (f"; moons {moons}" if moons else ""))
        return "\n".join(lines)


def is_sun(lookup: str) -> bool:
    return str(lookup).strip().lower() in ("sun", "sol", "the sun")


__all__ = ["PLANETS", "MOONS", "SolarSystem", "is_sun", "system_dict"]
