"""Look up a real star and its stellar system in local catalogues.

Sources, in order of precedence for each measured quantity:

- **NASA Exoplanet Archive** (``pscomppars``, local copy): planet hosts anywhere, with the
  archive's adopted stellar parameters and errors (Teff, radius, luminosity, mass, log g,
  [Fe/H], age) and photometry.
- **PASTEL** (Soubiran et al. 2016, VizieR B/pastel): literature Teff, log g and [Fe/H] of ~30 000
  stars; the median of the determinations published since 2000 (all if none) is used, with
  the scatter as error.
- **20-pc census** (Kirkpatrick et al. 2024, ApJS 271, 55, Table 4): every known object within
  20 pc with its system hierarchy, adopted initial mass (dynamical masses where measured,
  progenitor masses for white dwarfs), spectral type, Gaia photometry, parallax and RUWE.
- **Bright Star Catalogue** (Hoffleit & Warren 1991, V/50): Bayer and Flamsteed names, V, B-V,
  spectral type and parallax of the ~9000 stars brighter than V = 6.5.
- **Washington Double Star Catalog** (Mason et al. 2001, VizieR B/wds): the latest measured
  separation of pairs whose components the census does not resolve.
- **Gaia DR3** (online, optional, cached in ``data/real/gaia``): GSP-Phot and FLAME parameters
  for stars with no other measurement.

Where Teff is not measured it is taken from the spectral type or the BP-RP colour along the
Mamajek dwarf sequence (dwarfs only), and the luminosity from the absolute G or V magnitude with
the sequence's bolometric correction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy.table import Table

from ..paths import CATALOGS_DIR, DATA_DIR
from . import names

OBS_DIR = CATALOGS_DIR / "observational"
CENSUS_FILE = OBS_DIR / "stellar_census" / "J_ApJS_271_55_table4.ecsv"
NASA_FILE = OBS_DIR / "nasa_pscomppars.csv"
BSC_FILE = OBS_DIR / "stars" / "bsc5.ecsv"
PASTEL_FILE = OBS_DIR / "stars" / "pastel.ecsv"
WDS_FILE = OBS_DIR / "stars" / "wds.ecsv"
MAMAJEK_FILE = CATALOGS_DIR / "stellar_models" / "Mamajek" / "EEM_dwarf_UBVIJHK_colors_Teff.txt"
GAIA_CACHE = DATA_DIR / "real" / "gaia"
PASTEL_MODERN_YEAR = 2000

M_BOL_SUN = 4.74


@dataclass
class Measurement:
    value: float
    err: float
    source: str

    def __repr__(self):
        return f"{self.value:.4g} ± {self.err:.2g} ({self.source})"


@dataclass
class ComponentRecord:
    """One star, brown dwarf or white dwarf of a real system."""

    name: str
    measurements: dict[str, Measurement] = field(default_factory=dict)
    spectral_type: str = ""
    kind_hint: str = "star"  # "star", "WD" or "BD", from the spectral type
    ra: float = np.nan  # deg, J2000 (epoch 2000 for census objects)
    dec: float = np.nan
    distance_pc: float = np.nan
    hierarchy: str = ""  # census "Syst" code, e.g. "1674.1.2"
    ids: dict[str, str] = field(default_factory=dict)
    photometry: dict[str, float] = field(default_factory=dict)  # V, G, BP, RP, T, J, Ks
    ruwe: float = np.nan
    nasa_host: str = ""  # hostname in the NASA Exoplanet Archive
    n_known_planets: int = 0


@dataclass
class SystemRecord:
    """A real stellar system: its components (census order) and the looked-up target."""

    name: str
    components: list[ComponentRecord]
    target: int  # index of the looked-up component
    source: str  # "census", "nasa", "bsc" or "gaia"
    n_stars_reported: int = 1  # NASA sy_snum where the system is not in the census
    # Projected separation (arcsec) of component k from the nearest of components 0..k-1, with its
    # source ("WDS" or "census positions"); NaN where unresolved.
    separations: dict[int, tuple[float, str]] = field(default_factory=dict)

    @property
    def star(self) -> ComponentRecord:
        return self.components[self.target]


# --- catalogue loading -------------------------------------------------------------------

TABLE_CACHE = DATA_DIR / "real" / "tables"


def _read_table(path, reader) -> pd.DataFrame:
    """``reader(path)`` as a DataFrame, cached as a pickle next to the other caches (rebuilt when
    the catalogue file is newer)."""
    cache = TABLE_CACHE / (path.name + ".pkl")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_pickle(cache)
    d = reader(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(cache)
    return d


def _ecsv(path) -> pd.DataFrame:
    return Table.read(path).to_pandas()


def _csv(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)

_CENSUS_ID_COLS = ("Name", "OName", "GJ", "HD", "HIP", "Gaia", "TIC", "2MASS", "Var", "WD", "LHS", "LP", "Wolf",
                   "Ross", "BD", "CD", "CPD", "HR", "Karmn", "LTT", "NLTT", "LSPM", "Giclas")


def _split_ids(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    out = []
    for part in value.split(","):
        part = re.sub(r"\s*\((?:[A-D][a-d]?|[A-D]{1,2})\)\s*$", "", part.strip())  # "Castor (Aa)" -> "Castor"
        if part:
            out.append(part)
    return out


@lru_cache(maxsize=1)
def census() -> pd.DataFrame:
    t = _read_table(CENSUS_FILE, _ecsv)
    t["Syst"] = t["Syst"].astype(str)
    t["Name"] = t["Name"].str.replace(r"^-+", "", regex=True).str.strip()
    return t


@lru_cache(maxsize=1)
def nasa_hosts() -> pd.DataFrame:
    d = nasa_planets()
    n = d.groupby("hostname").size()
    h = d.drop_duplicates("hostname").reset_index(drop=True).copy()
    h["n_planets"] = h["hostname"].map(n).to_numpy()
    return h


@lru_cache(maxsize=1)
def nasa_planets() -> pd.DataFrame:
    return _read_table(NASA_FILE, _csv)


@lru_cache(maxsize=1)
def bsc() -> pd.DataFrame:
    return _read_table(BSC_FILE, _ecsv)


@lru_cache(maxsize=1)
def pastel() -> pd.DataFrame:
    p = _read_table(PASTEL_FILE, _ecsv)
    p["key"] = p["ID"].map(names.key)
    p["year"] = pd.to_numeric(p["bibcode"].str[:4], errors="coerce")
    return p


@lru_cache(maxsize=1)
def wds() -> pd.DataFrame:
    w = _read_table(WDS_FILE, _ecsv)
    w["disc_key"] = w["Disc"].astype(str).str.replace(" ", "").str.upper()
    w["comp_key"] = w["Comp"].fillna("").astype(str).str.replace(r"[\s,]", "", regex=True).str.upper()
    return w


_MULT = re.compile(r"\*\*\s*([A-Z]{1,4})\s*(\d+)([A-Za-z]*)")


def _wds_codes(value) -> list[tuple[str, str]]:
    """(discoverer code, component letters) of a census ``Mult`` entry such as "** STF 518B"."""
    if not isinstance(value, str):
        return []
    return [(m.group(1) + m.group(2), m.group(3).upper()) for m in _MULT.finditer(value)]


def wds_separation(disc: str, letters_a: str, letters_b: str) -> float:
    """Latest WDS separation (arcsec) between components ``letters_a`` and ``letters_b`` of the pair
    ``disc`` (e.g. "STF518", "B", "C"); NaN if not listed."""
    w = wds()
    rows = w[w["disc_key"] == disc]
    keys = rows["comp_key"].to_numpy()
    seps = pd.to_numeric(rows["sep2"], errors="coerce").to_numpy()
    best = None
    for a, b in ((letters_a, letters_b), (letters_b, letters_a)):
        for key, sep in zip(keys, seps):
            # "AB", or a component against a subsystem: "A,BC" (key "ABC") for A and B.
            if key.startswith(a) and b in key[len(a):] and np.isfinite(sep) and sep < 999.0:
                if best is None or len(key) < len(best[0]):
                    best = (key, sep)
    return float(best[1]) if best else np.nan


@lru_cache(maxsize=1)
def _census_index() -> dict[str, int]:
    """Name key -> census row (leaf or subsystem row)."""
    c = census()
    idx: dict[str, int] = {}
    for col in _CENSUS_ID_COLS:
        if col not in c:
            continue
        for row, val in zip(c.index, c[col]):
            for ident in _split_ids(val):
                k = names.key(ident)
                if k and k not in idx:
                    idx[k] = row
    # "GJ 65" for the system of "GJ 65 A" and "GJ 65 B": the A component.
    for k, row in list(idx.items()):
        base, letter = names.split_component(k)
        if letter == "a" and base and base not in idx:
            idx[base] = row
    return idx


@lru_cache(maxsize=1)
def _nasa_index() -> dict[str, int]:
    h = nasa_hosts()
    idx: dict[str, int] = {}
    for col in ("hostname", "hd_name", "hip_name", "tic_id", "gaia_dr3_id", "gaia_dr2_id"):
        for row, val in zip(h.index, h[col]):
            k = names.key(val) if isinstance(val, str) else ""
            if k and k not in idx:
                idx[k] = row
    # Hosts known under another name, through their planets' names ("Kepler-90 i" of KOI-351).
    row_of = {hn: r for r, hn in zip(h.index, h["hostname"])}
    p = nasa_planets()
    for pl, hn in zip(p["pl_name"], p["hostname"]):
        k = names.key(re.sub(r"\s+[a-z]$", "", str(pl)))
        if k and k not in idx and hn in row_of:
            idx[k] = row_of[hn]
    return idx


@lru_cache(maxsize=1)
def _bsc_index() -> dict[str, int]:
    b = bsc()
    idx: dict[str, int] = {}
    for row, (nm, hr, hd) in enumerate(zip(b["Name"], b["HR"], b["HD"])):
        for k in (names.bsc_name_key(nm), names.bsc_flamsteed_key(nm), f"hr{hr}",
                  f"hd{int(hd)}" if pd.notna(hd) else ""):
            if k and k not in idx:
                idx[k] = row
    return idx


@lru_cache(maxsize=1)
def _pastel_groups() -> dict[str, np.ndarray]:
    p = pastel()
    return {k: np.asarray(v) for k, v in p.groupby("key").indices.items()}


# --- Mamajek dwarf sequence ----------------------------------------------------------------

@lru_cache(maxsize=1)
def mamajek() -> pd.DataFrame:
    rows = []
    cols = None
    for line in MAMAJEK_FILE.read_text().splitlines():
        if line.startswith("#SpT") and cols is None:
            cols = line[1:].split()
            continue
        parts = line.split()
        if cols is None or len(parts) < len(cols) or not parts[0].endswith("V"):
            continue
        rows.append(parts[:len(cols)])
        if parts[0] == "Y4V":
            break
    d = pd.DataFrame(rows, columns=cols).drop_duplicates("SpT")
    for c in d.columns[1:-1]:
        d[c] = pd.to_numeric(d[c].str.replace(r"\.{2,}", "", regex=True), errors="coerce")
    return d.reset_index(drop=True)


def teff_from_spectral_type(sptype: str) -> Measurement | None:
    """Teff of a dwarf spectral type (e.g. "M4 V", "K2.5V", "A0mA1 Va") from Mamajek's table."""
    if not isinstance(sptype, str):
        return None
    m = re.match(r"^\s*(?:sd|d|k|g)?([OBAFGKMLTY])(\d(?:\.\d+)?)?", sptype.strip())
    if not m:
        return None
    d = mamajek()
    letter, sub = m.group(1), float(m.group(2) or 5.0)
    code = d["SpT"].str[0]
    num = pd.to_numeric(d["SpT"].str[1:-1], errors="coerce")
    sel = (code == letter) & num.notna()
    if not sel.any():
        return None
    teff = float(np.interp(sub, num[sel], d["Teff"][sel]))
    # About one subtype of error (5% for M and later, 3% earlier).
    return Measurement(teff, teff * (0.05 if letter in "MLTY" else 0.03), "spectral type")


def teff_from_bp_rp(bp_rp: float) -> Measurement | None:
    d = mamajek().dropna(subset=["Bp-Rp", "Teff"])
    x, y = d["Bp-Rp"].to_numpy(), d["Teff"].to_numpy()
    if not np.isfinite(bp_rp) or bp_rp < x.min() or bp_rp > x.max():
        return None
    order = np.argsort(x)
    teff = float(np.interp(bp_rp, x[order], y[order]))
    return Measurement(teff, 0.04 * teff, "BP-RP")


def luminosity_from_mag(mag: float, band: str, distance_pc: float, teff: float) -> Measurement | None:
    """L/Lsun from an apparent V or G magnitude, a distance and the dwarf sequence's bolometric
    correction at Teff (BC_V; BC_G = BC_V + (G-V))."""
    if not all(np.isfinite(x) for x in (mag, distance_pc, teff)):
        return None
    d = mamajek().dropna(subset=["Teff", "BCv"])
    order = np.argsort(d["Teff"].to_numpy())
    t = d["Teff"].to_numpy()[order]
    if teff < t.min() or teff > t.max():
        return None
    bc = float(np.interp(teff, t, d["BCv"].to_numpy()[order]))
    if band == "G":
        gv = d.dropna(subset=["G-V"])
        o2 = np.argsort(gv["Teff"].to_numpy())
        bc -= float(np.interp(teff, gv["Teff"].to_numpy()[o2], gv["G-V"].to_numpy()[o2]))
    m_abs = mag - 5.0 * np.log10(distance_pc / 10.0)
    log_l = (M_BOL_SUN - (m_abs + bc)) / 2.5
    return Measurement(10**log_l, 10**log_l * 0.08, f"{band} mag + BC")


# --- per-source measurements ---------------------------------------------------------------

def _err(e1, e2, value, rel_default):
    e = np.nanmean(np.abs([e1, e2])) if np.isfinite([e1, e2]).any() else np.nan
    return float(e) if np.isfinite(e) and e > 0 else abs(value) * rel_default


def nasa_measurements(row: pd.Series) -> dict[str, Measurement]:
    out = {}

    def take(col, name, rel_default, transform=None):
        v = row.get(col)
        if pd.isna(v):
            return
        lim = row.get(col + "lim")
        if pd.notna(lim) and lim != 0:  # upper/lower limits are not used
            return
        e = _err(row.get(col + "err1"), row.get(col + "err2"), v, rel_default)
        if transform == "log":  # st_lum is log10(L/Lsun)
            out[name] = Measurement(10**v, 10**v * np.log(10) * e, "NASA")
        else:
            out[name] = Measurement(float(v), e, "NASA")

    take("st_teff", "teff", 0.02)
    take("st_rad", "radius", 0.05)
    take("st_lum", "luminosity", 0.05, "log")
    take("st_mass", "mass", 0.05)
    take("st_logg", "log_g", 0.02)
    take("st_met", "feh", 0.0)
    take("st_age", "age", 0.3)
    if "feh" in out and out["feh"].err == 0:
        out["feh"].err = 0.1
    return out


def pastel_measurements(keys) -> dict[str, Measurement]:
    groups = _pastel_groups()
    rows = np.unique(np.concatenate([groups[k] for k in keys if k in groups] or [np.array([], dtype=int)]))
    if not len(rows):
        return {}
    p = pastel().iloc[rows]
    modern = p[p["year"] >= PASTEL_MODERN_YEAR]
    p = modern if len(modern) else p
    out = {}
    for col, name, floor in (("Teff", "teff", 50.0), ("logg", "log_g", 0.1), ("[Fe/H]", "feh", 0.05)):
        v = pd.to_numeric(p[col], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(v):
            continue
        med = float(np.median(v))
        mad = 1.4826 * float(np.median(np.abs(v - med))) if len(v) > 2 else 0.0
        out[name] = Measurement(med, max(mad, floor), f"PASTEL ({len(v)})")
    return out


def census_measurements(row: pd.Series) -> dict[str, Measurement]:
    out = {}
    im = row.get("IMass")
    if pd.notna(im) and im > 0:
        e = row.get("e_IMass")
        e = float(e) if pd.notna(e) and e > 0 else 0.1 * im
        out["initial_mass"] = Measurement(float(im), e, f"census ({row.get('n_IMass', '')})")
    m = row.get("Mass")
    if pd.notna(m) and m > 0:
        e = row.get("e_Mass")
        out["mass"] = Measurement(float(m), float(e) if pd.notna(e) and e > 0 else 0.05 * m,
                                  f"census ({row.get('n_Mass', '')})")
    t = row.get("Teff")
    if pd.notna(t) and t > 0:
        e = row.get("e_Teff")
        out["teff"] = Measurement(float(t), float(e) if pd.notna(e) and e > 0 else 100.0, "census")
    return out


def kind_from_spectral_type(sptype: str) -> str:
    if not isinstance(sptype, str):
        return "star"
    s = sptype.strip()
    if re.match(r"^(D[ABOQZCX]|WD|DA|DB|DC|DQ|DZ)", s):
        return "WD"
    if re.match(r"^(sd|d)?[LTY]\d", s):
        return "BD"
    return "star"


# --- Gaia DR3 (online, cached) -------------------------------------------------------------

def gaia_dr3(source_id: str) -> dict | None:
    """GSP-Phot and FLAME parameters plus astrometry of a Gaia DR3 source (cached JSON)."""
    sid = re.sub(r"\D", "", str(source_id))
    if not sid:
        return None
    path = GAIA_CACHE / f"{sid}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        from astroquery.gaia import Gaia
        q = ("SELECT g.source_id, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, "
             "a.teff_gspphot, a.teff_gspphot_lower, a.teff_gspphot_upper, a.logg_gspphot, a.mh_gspphot, "
             "a.radius_flame, a.radius_flame_lower, a.radius_flame_upper, a.lum_flame, a.lum_flame_lower, "
             "a.lum_flame_upper, a.mass_flame, a.mass_flame_lower, a.mass_flame_upper, a.age_flame, "
             "a.age_flame_lower, a.age_flame_upper "
             "FROM gaiadr3.gaia_source AS g LEFT JOIN gaiadr3.astrophysical_parameters AS a "
             f"ON a.source_id = g.source_id WHERE g.source_id = {sid}")
        r = Gaia.launch_job(q).get_results()
    except Exception:  # offline or service down
        return None
    if not len(r):
        return None
    rec = {c: (None if np.ma.is_masked(r[c][0]) else float(r[c][0])) for c in r.colnames if c != "source_id"}
    GAIA_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def gaia_measurements(rec: dict) -> dict[str, Measurement]:
    out = {}

    def take(name, col, transform=None):
        v, lo, hi = rec.get(col), rec.get(col + "_lower"), rec.get(col + "_upper")
        if v is None:
            return
        e = 0.5 * (hi - lo) if lo is not None and hi is not None else 0.1 * abs(v)
        out[name] = Measurement(float(v), max(float(e), 1e-3), "Gaia DR3")

    take("teff", "teff_gspphot")
    take("radius", "radius_flame")
    take("luminosity", "lum_flame")
    take("mass", "mass_flame")
    take("age", "age_flame")
    if rec.get("mh_gspphot") is not None:  # GSP-Phot metallicities are biased low; broad error
        out["feh"] = Measurement(float(rec["mh_gspphot"]), 0.3, "Gaia DR3")
    if "teff" in out:  # GSP-Phot Teff: ~2-3% systematic
        out["teff"].err = max(out["teff"].err, 0.03 * out["teff"].value)
    return out


# --- assembling a component ---------------------------------------------------------------

def _id_keys(ids) -> list[str]:
    return [k for k in (names.key(i) for i in ids) if k]


def _merge(target: dict, extra: dict):
    for k, v in extra.items():
        target.setdefault(k, v)


def _fill_derived(c: ComponentRecord):
    """Teff from spectral type or colour, luminosity from photometry, where not measured."""
    ms = c.measurements
    if c.kind_hint == "star" and "teff" not in ms:
        bp, rp = c.photometry.get("BP", np.nan), c.photometry.get("RP", np.nan)
        t = teff_from_bp_rp(bp - rp) if np.isfinite(bp - rp) else None
        t = t or teff_from_spectral_type(c.spectral_type)
        if t is not None:
            ms["teff"] = t
    if c.kind_hint == "star" and "luminosity" not in ms and "teff" in ms and not (
            "radius" in ms and ms["radius"].source == ms["teff"].source):
        for band in ("G", "V"):
            lum = luminosity_from_mag(c.photometry.get(band, np.nan), band, c.distance_pc, ms["teff"].value)
            if lum is not None:
                ms["luminosity"] = lum
                break


def _component_from_census(row: pd.Series) -> ComponentRecord:
    ids = {col: row[col] for col in _CENSUS_ID_COLS if col in row and isinstance(row[col], str) and row[col].strip()}
    plx = row.get("Plx")
    c = ComponentRecord(
        name=row["Name"], spectral_type=str(row.get("SpTOpt") or row.get("SpTNIR") or ""),
        ra=float(row.get("RAPdeg")) if pd.notna(row.get("RAPdeg")) else _sexa(row.get("RAJ2000"), 15.0),
        dec=float(row.get("DEPdeg")) if pd.notna(row.get("DEPdeg")) else _sexa(row.get("DEJ2000"), 1.0),
        distance_pc=1000.0 / plx if pd.notna(plx) and plx > 0 else np.nan, hierarchy=row["Syst"], ids=ids,
        ruwe=float(row["RUWE"]) if pd.notna(row.get("RUWE")) else np.nan,
    )
    # Positions in the J2000 columns are per component; the RAPdeg columns can repeat the primary's.
    ra2, de2 = _sexa(row.get("RAJ2000"), 15.0), _sexa(row.get("DEJ2000"), 1.0)
    if np.isfinite(ra2):
        c.ra, c.dec = ra2, de2
    sp = row.get("SpTOpt") if isinstance(row.get("SpTOpt"), str) else row.get("SpTNIR")
    c.spectral_type = sp if isinstance(sp, str) else ""
    c.kind_hint = kind_from_spectral_type(c.spectral_type)
    if c.kind_hint == "star" and pd.notna(row.get("IMass")) and row["IMass"] == -999:
        c.kind_hint = "BD"
    for band, col in (("G", "Gmag"), ("BP", "BPmag"), ("RP", "RPmag"), ("J", "Jmag"), ("Ks", "Ksmag")):
        if pd.notna(row.get(col)):
            c.photometry[band] = float(row[col])
    c.measurements = census_measurements(row)
    return c


def _sexa(value, scale) -> float:
    if not isinstance(value, str) or not value.strip():
        return np.nan
    parts = [float(x) for x in value.replace("+", " +").split()]
    sign = -1.0 if value.strip().startswith("-") else 1.0
    d = abs(parts[0]) + parts[1] / 60.0 + parts[2] / 3600.0
    return sign * d * scale


def _add_external(c: ComponentRecord, use_gaia: bool):
    """NASA, BSC, PASTEL and (optionally) Gaia measurements for a census or NASA component."""
    keys = set(_id_keys(c.ids.values())) | {names.key(c.name)}
    for k in list(keys):  # the primary is also known without its letter ("alf CMa" for alpha CMa A)
        base, letter = names.split_component(k)
        if letter == "a":
            keys.add(base)
    hosts = nasa_hosts()
    ni = _nasa_index()
    for k in keys:
        if k in ni:
            row = hosts.loc[ni[k]]
            c.nasa_host = row["hostname"]
            c.n_known_planets = int(row["n_planets"])
            # Archive values are adopted over the census's model-based initial masses.
            nm = nasa_measurements(row)
            for q, m in nm.items():
                model_based = ("census (M_", "census (T", "census (s")  # census masses from relations
                if q not in c.measurements or c.measurements[q].source.startswith(model_based):
                    c.measurements[q] = m
            for band, col in (("V", "sy_vmag"), ("G", "sy_gaiamag"), ("T", "sy_tmag")):
                if pd.notna(row.get(col)) and band not in c.photometry:
                    c.photometry[band] = float(row[col])
            if not np.isfinite(c.distance_pc) and pd.notna(row.get("sy_dist")):
                c.distance_pc = float(row["sy_dist"])
            for col in ("tic_id", "gaia_dr3_id", "hd_name", "hip_name"):
                if isinstance(row.get(col), str):
                    c.ids.setdefault(col, row[col])
            break
    keys |= set(_id_keys(c.ids.values()))
    bi = _bsc_index()
    for k in keys:
        if k in bi:
            b = bsc().iloc[bi[k]]
            if pd.notna(b["Vmag"]):
                c.photometry.setdefault("V", float(b["Vmag"]))
            if not c.spectral_type and isinstance(b["SpType"], str):
                c.spectral_type = b["SpType"]
            if not np.isfinite(c.distance_pc) and pd.notna(b["Parallax"]) and b["Parallax"] > 0:
                c.distance_pc = 1.0 / float(b["Parallax"])
            if np.isnan(c.ra):
                c.ra, c.dec = _sexa(b["RAJ2000"], 15.0), _sexa(b["DEJ2000"], 1.0)
            break
    _merge(c.measurements, pastel_measurements(keys))
    if use_gaia and ("teff" not in c.measurements or "feh" not in c.measurements):
        gid = next((v for k, v in c.ids.items() if k in ("Gaia", "gaia_dr3_id")), None)
        rec = gaia_dr3(gid) if gid else None
        if rec:
            _merge(c.measurements, gaia_measurements(rec))
            if not np.isfinite(c.ruwe) and rec.get("ruwe") is not None:
                c.ruwe = rec["ruwe"]
    _fill_derived(c)


def _census_system(row_index: int, use_gaia: bool) -> tuple[list[ComponentRecord], int, dict]:
    c = census()
    syst = c.loc[row_index, "Syst"]
    top = syst.split(".")[0]
    members = c[(c["Syst"] == top) | c["Syst"].str.startswith(top + ".")]
    leaves = members[members["NcTR"] == 1]
    if not len(leaves):
        leaves = members.iloc[:1]
    leaves = leaves.loc[_chain_order(list(leaves["Syst"]), leaves.index)]
    comps = [_component_from_census(r) for _, r in leaves.iterrows()]
    _inherit_ids(members, leaves, comps)
    # Target: the leaf itself, or the most massive leaf of the subsystem whose name was given.
    sub = leaves[(leaves["Syst"] == syst) | leaves["Syst"].str.startswith(syst + ".")]
    if len(sub) == 1:
        target = list(leaves.index).index(sub.index[0])
    else:
        im = sub["IMass"].where(sub["IMass"] > 0, 0.0).fillna(0.0)
        target = list(leaves.index).index(im.idxmax())
    for comp in comps:
        _add_external(comp, use_gaia)
    return comps, target, _separations(members, leaves, comps)


def _chain_order(codes: list[str], index) -> list:
    """Order of the leaves so that the generator's chain (companion k orbits components 0..k-1)
    follows the census hierarchy: the most structured subsystem first, its leaves before the wider
    ones. ``codes`` are the leaves' "Syst" codes (e.g. "477.1", "477.2.1", "477.2.2")."""
    tree: dict = {}
    for code, idx in zip(codes, index):
        node = tree
        for part in code.split(".")[1:]:
            node = node.setdefault(part, {})
        node["_leaf"] = idx

    def depth(node) -> int:
        kids = [v for k, v in node.items() if k != "_leaf"]
        return 0 if not kids else 1 + max(depth(v) for v in kids)

    def flatten(node) -> list:
        out = [node["_leaf"]] if "_leaf" in node else []
        kids = [v for k, v in sorted(node.items()) if k != "_leaf"]
        if not kids:
            return out
        # Deepest child first (stable: census order among equals), then the others.
        first = max(range(len(kids)), key=lambda i: (depth(kids[i]), -i))
        out += flatten(kids[first])
        for i, kid in enumerate(kids):
            if i != first:
                out += flatten(kid)
        return out

    return flatten(tree)


def _inherit_ids(members: pd.DataFrame, leaves: pd.DataFrame, comps: list[ComponentRecord]):
    """Identifiers of subsystem rows ("Castor (Aa)" on the row of alpha Gem A) go to the subsystem's
    first leaf, where the leaf has none of that kind."""
    for _, row in members[members["NcTR"] > 1].iterrows():
        code = row["Syst"]
        under = [i for i, c in enumerate(leaves["Syst"]) if c.startswith(code + ".")]
        if not under:
            continue
        first = min(under, key=lambda i: leaves["Syst"].iloc[i])
        for col in _CENSUS_ID_COLS:
            if col in ("Name",) or col not in row or not isinstance(row[col], str) or not row[col].strip():
                continue
            comps[first].ids.setdefault(col, row[col])


def _separations(members: pd.DataFrame, leaves: pd.DataFrame, comps: list[ComponentRecord]) -> dict:
    """Projected separation of each component from the nearest earlier one: WDS where the pair is
    listed, else the census positions (if they differ)."""
    # WDS discoverer code of the system and each leaf's component letters.
    codes = [c for v in members["Mult"] for c in _wds_codes(v)]
    disc = codes[0][0] if codes else ""
    letters = []
    for (_, row), comp in zip(leaves.iterrows(), comps):
        own = [c for c in _wds_codes(row.get("Mult")) if c[0] == disc]
        if own and len(own[0][1]) == 1:
            letters.append(own[0][1])
        else:
            m = re.search(r"\s([A-F])[a-d]?\)?$", comp.name)  # "alpha Gem Ba", "YY Gem A (alpha Gem C)"
            letters.append(m.group(1) if m else "")
    out = {}
    for k in range(1, len(comps)):
        best = (np.nan, "")
        if letters[k] and letters[k] in letters[:k]:  # e.g. Ba and Bb: an unresolved pair
            out[k] = (np.nan, "unresolved")
            continue
        for j in range(k):
            sep, src = np.nan, ""
            if disc and letters[j] and letters[k] and letters[j] != letters[k]:
                sep, src = wds_separation(disc, letters[j], letters[k]), "WDS"
            if not np.isfinite(sep):
                sep, src = _position_separation(comps[j], comps[k]), "census positions"
            if np.isfinite(sep) and not (sep >= best[0]):
                best = (sep, src)
        out[k] = best
    return out


def _position_separation(c0: ComponentRecord, c1: ComponentRecord) -> float:
    """Separation (arcsec) of two census positions; NaN if identical (unresolved in the census)."""
    if not all(np.isfinite([c0.ra, c0.dec, c1.ra, c1.dec])):
        return np.nan
    d0, d1 = np.radians(c0.dec), np.radians(c1.dec)
    cosang = np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(np.radians(c1.ra - c0.ra))
    sep = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) * 3600.0)
    return sep if sep > 0.05 else np.nan


def find(name: str, use_gaia: bool = True) -> SystemRecord:
    """Look up ``name`` (any common identifier). Raises ``KeyError`` if no catalogue has it."""
    k = names.key(name)
    candidates = [k]
    base, comp = names.split_component(k)
    m = re.match(r"^(.*\S)\s+([A-Da-d])$", name.strip())
    if m and not comp:  # "Sirius B": a component letter after a proper name
        base, comp = names.key(m.group(1)), m.group(2).lower()
    if comp:
        candidates.append(base)
    ci = _census_index()
    # Names known only to the NASA archive or the Bright Star Catalogue: bridge through their
    # HD / HIP / Gaia / TIC numbers to the census.
    ni, bi = _nasa_index(), _bsc_index()
    for cand in list(candidates):
        if cand in ci:
            continue
        aliases = []
        if cand in ni:
            row = nasa_hosts().loc[ni[cand]]
            aliases = [row.get(c) for c in ("hd_name", "hip_name", "gaia_dr3_id", "tic_id")]
        elif cand in bi:
            hd = bsc().iloc[bi[cand]]["HD"]
            aliases = [f"HD {int(hd)}"] if pd.notna(hd) else []
        for a in aliases:
            ak = names.key(a) if isinstance(a, str) else ""
            if ak in ci:
                candidates.insert(candidates.index(cand), ak)
                if cand == base:
                    base = ak
                break
    for cand in candidates:
        if cand in ci:
            row = ci[cand]
            comps, target, seps = _census_system(row, use_gaia)
            if cand == base and comp:  # "X B" given but only "X" known: pick the component by letter
                by_name = [i for i, cc in enumerate(comps) if cc.name.lower().endswith(" " + comp)]
                if by_name:
                    target = by_name[0]
                elif ord(comp[0]) - ord("a") < len(comps):
                    target = ord(comp[0]) - ord("a")
            return SystemRecord(name=name, components=comps, target=target, source="census", separations=seps)
    # Not within 20 pc: a NASA host, a bright star or a Gaia source.
    for cand in candidates:
        if cand in ni:
            row = nasa_hosts().loc[ni[cand]]
            dist = float(row["sy_dist"]) if pd.notna(row["sy_dist"]) else np.nan
            c = ComponentRecord(name=row["hostname"], distance_pc=dist,
                                ra=float(row["ra"]), dec=float(row["dec"]),
                                spectral_type=row["st_spectype"] if isinstance(row["st_spectype"], str) else "",
                                ids={"hostname": row["hostname"]})
            c.kind_hint = kind_from_spectral_type(c.spectral_type)
            _add_external(c, use_gaia)
            n_star = int(row["sy_snum"]) if pd.notna(row["sy_snum"]) else 1
            return SystemRecord(name=name, components=[c], target=0, source="nasa", n_stars_reported=n_star)
        if cand in bi:
            b = bsc().iloc[bi[cand]]
            c = ComponentRecord(name=name, ids={"HD": f"HD {int(b['HD'])}" if pd.notna(b["HD"]) else ""},
                                spectral_type=b["SpType"] if isinstance(b["SpType"], str) else "")
            c.kind_hint = kind_from_spectral_type(c.spectral_type)
            _add_external(c, use_gaia)
            n_star = 2 if isinstance(b["Multiple"], str) and b["Multiple"].strip() else 1
            return SystemRecord(name=name, components=[c], target=0, source="bsc", n_stars_reported=n_star)
    if k.startswith("gaia") and use_gaia:
        rec = gaia_dr3(k[4:])
        if rec:
            c = ComponentRecord(name=name, ids={"Gaia": k[4:]})
            if rec.get("parallax"):
                c.distance_pc = 1000.0 / rec["parallax"]
            c.photometry["G"] = rec.get("phot_g_mean_mag") or np.nan
            c.measurements = gaia_measurements(rec)
            c.ruwe = rec.get("ruwe") or np.nan
            _fill_derived(c)
            return SystemRecord(name=name, components=[c], target=0, source="gaia")
    raise KeyError(f"{name!r} not found in the 20-pc census, the NASA Exoplanet Archive or the Bright Star "
                   "Catalogue (and not a Gaia DR3 id)")
