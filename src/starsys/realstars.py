"""Real stars in the procedural galaxy: catalogue systems plus a procedural fill of exactly
what the catalogues miss.

Catalogues (both selection functions are known, so the blend is exact):

* **20-pc census** (Kirkpatrick et al. 2024): every system within 20 pc. Treated as complete,
  so no procedural system is kept within 20 pc.
* **Hipparcos** (ESA 1997, VizieR I/239) beyond 20 pc, restricted to its *Survey* part, which
  is complete to V <= 7.9 + 1.1 |sin b| for spectral types earlier than G5 and
  V <= 7.3 + 1.1 |sin b| for later types (ESA 1997, Vol. 1, §2.2), and to parallaxes measured
  better than 5 sigma; stars with IAU proper names are kept down to 1.5 sigma (Deneb, Betelgeuse:
famous, and far beyond where procedural stars are drawn in practice).

A procedural system is removed when the catalogues would contain it: within 20 pc, or when its
apparent V (unresolved components summed, with extinction) is within the Survey limit at its
latitude and its expected parallax signal-to-noise exceeds 5 (Hipparcos parallax error taken as
0.9 mas, a typical value for V < 9, van Leeuwen 2007). So around a bright distant star like
Vega's neighbours at 100 pc, the bright stars on the map are real and the faint M dwarfs are
procedural. Fainter Hipparcos stars are *not* shown (they would duplicate the procedural fill).

Extinction: A_V = 0.7 mag/kpc along the path inside a dust layer of 100 pc scale height
(a mean value, e.g. Whittet 2003). Bolometric corrections and absolute magnitudes: Mamajek's
dwarf table (Pecaut & Mamajek 2013, as used in starsys.real).

Names: IAU-CSN proper names (WGSN, Mamajek's list), Bright Star Catalogue names, the census
names, else "HIP n".
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import galaxy as gx
from .paths import CATALOGS_DIR, DATA_DIR

STARS_DIR = CATALOGS_DIR / "observational" / "stars"
HIP_FILE = STARS_DIR / "hip_main.ecsv"
IAU_FILE = STARS_DIR / "IAU-CSN.txt"
CACHE = DATA_DIR / "real" / "starmap.parquet"

CENSUS_PC = 20.0
HIP_SIGMA_MAS = 0.9
MIN_PLX_SNR = 5.0
AV_PER_KPC, DUST_H = 0.7, 100.0


def survey_limit(sin_b, early) -> np.ndarray:
    """Hipparcos Survey completeness limit in V (early = spectral type before G5)."""
    return np.where(early, 7.9, 7.3) + 1.1 * np.abs(sin_b)


def extinction(xyz_pc: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(xyz_pc, axis=-1)
    sin_b = np.abs(xyz_pc[..., 2]) / np.maximum(d, 1e-9)
    path = np.minimum(d, DUST_H / np.maximum(sin_b, 1e-6))
    return AV_PER_KPC * path / 1000.0


@lru_cache(maxsize=1)
def _bc_table():
    from .real.catalog import mamajek
    m = mamajek()
    ok = np.isfinite(m["BCv"]) & np.isfinite(m["Teff"])
    t, bc = m["Teff"][ok].to_numpy(float), m["BCv"][ok].to_numpy(float)
    o = np.argsort(t)
    return t[o], bc[o]


def abs_v(teff, lum) -> np.ndarray:
    """Absolute V magnitude from Teff and luminosity (bolometric correction from Mamajek)."""
    t, bc = _bc_table()
    teff = np.asarray(teff, float)
    lum = np.maximum(np.asarray(lum, float), 1e-12)
    b = np.interp(teff, t, bc, left=bc[0] - 3.0, right=bc[-1])
    b = np.where(teff < t[0], bc[0] - 8.0 * (t[0] - teff) / t[0], b)  # brown dwarfs: very red, V faint
    return 4.74 - 2.5 * np.log10(lum) - b


def system_vmag(sec: gx.Sector) -> np.ndarray:
    """Apparent V of each procedural system of a sector (components summed, extinction added)."""
    c = sec.components
    mv = abs_v(c["teff"], c["luminosity"])
    flux = np.bincount(c["system"], weights=10 ** (-0.4 * mv), minlength=len(sec))
    d = np.maximum(np.linalg.norm(sec.xyz, axis=1), 1e-3)
    with np.errstate(divide="ignore"):
        m_abs = -2.5 * np.log10(np.maximum(flux, 1e-30))
    return m_abs + 5 * np.log10(d / 10.0) + extinction(sec.xyz)


def in_catalogue(sec: gx.Sector) -> np.ndarray:
    """True for the procedural systems the real catalogues would already contain."""
    if not len(sec):
        return np.zeros(0, bool)
    d = np.linalg.norm(sec.xyz, axis=1)
    v = system_vmag(sec)
    first = np.searchsorted(sec.components["system"], np.arange(len(sec)))
    teff = np.asarray(sec.components["teff"])[first]
    early = teff > 5600.0  # before G5
    sin_b = sec.xyz[:, 2] / np.maximum(d, 1e-9)
    hip = (v <= survey_limit(sin_b, early)) & ((1000.0 / np.maximum(d, 1e-9)) / HIP_SIGMA_MAS > MIN_PLX_SNR)
    return (d < CENSUS_PC) | hip


# ------------------------------------------------------------------------------- catalogues

def _iau() -> pd.DataFrame:
    rows = []
    for line in IAU_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("$"):
            continue
        tail = line[36:].split()
        if tail and tail[-1] == "*":
            tail = tail[:-1]
        if len(tail) < 7:
            continue
        hip, hd = tail[-5], tail[-4]
        rows.append({"proper": line[:18].strip(), "hip": int(hip) if hip.isdigit() else -1,
                     "hd": int(hd) if hd.isdigit() else -1})
    return pd.DataFrame(rows)


def _galactic(ra_deg, dec_deg, dist_pc) -> np.ndarray:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    g = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs").galactic
    lr, br = g.l.radian, g.b.radian
    d = np.asarray(dist_pc, float)
    return np.stack([d * np.cos(br) * np.cos(lr), d * np.cos(br) * np.sin(lr), d * np.sin(br)], axis=1)


def _num(s) -> np.ndarray:
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"(\d+)")[0], errors="coerce").to_numpy()


def _census_systems() -> pd.DataFrame:
    from .real.catalog import _sexa, census
    c = census()
    syst = c["Syst"].astype(str)
    top = c[~syst.str.contains(r"\.")].copy()
    comps = c[syst.str.contains(r"\.")]
    first = comps.groupby(comps["Syst"].astype(str).str.split(".").str[0]).first()
    key = top["Syst"].astype(str)

    def pick(col):
        v = top[col].copy()
        alt = first.reindex(key)[col].to_numpy() if col in first else None
        if alt is not None:
            v = v.where(v.notna(), pd.Series(alt, index=v.index))
        return v

    ra = top["RAJ2000"].map(lambda s: _sexa(s, 15.0))
    dec = top["DEJ2000"].map(lambda s: _sexa(s, 1.0))
    plx = pick("Plx").astype(float)
    dist = np.where(plx > 0, 1000.0 / plx, np.nan)
    sun = top["Name"].eq("Sun").to_numpy()
    xyz = np.zeros((len(top), 3))
    ok = np.isfinite(dist) & np.isfinite(ra) & np.isfinite(dec)
    xyz[ok] = _galactic(ra[ok], dec[ok], dist[ok])
    keep = ok | sun
    name = top["OName"].where(top["OName"].notna(), top["Name"]).astype(str)
    sp = pick("SpTOpt").where(pick("SpTOpt").notna(), pick("SpTNIR"))
    return pd.DataFrame({
        "name": name.to_numpy()[keep], "lookup": top["Name"].astype(str).to_numpy()[keep],
        "x": xyz[keep, 0], "y": xyz[keep, 1], "z": xyz[keep, 2],
        "vmag": np.where(sun, -26.74, pick("Gmag").astype(float).to_numpy())[keep],
        "sptype": sp.astype(str).replace("nan", "").to_numpy()[keep],
        "teff": pd.to_numeric(pick("Teff"), errors="coerce").to_numpy()[keep],
        "n_comp": pd.to_numeric(top["Ncomp"], errors="coerce").fillna(1).to_numpy()[keep],
        "hip": _num(pick("HIP"))[keep], "source": "census"})


def _hipparcos(census_hip: set) -> pd.DataFrame:
    from astropy.table import Table

    from .real.catalog import bsc
    from .real.catalog import teff_from_spectral_type
    t = Table.read(HIP_FILE).to_pandas()
    plx, eplx = t["Plx"].astype(float), t["e_Plx"].astype(float)
    named = t["HIP"].isin(set(_iau()["hip"]))
    snr = plx / eplx
    ok = (plx > 0) & ((snr > MIN_PLX_SNR) | (named & (snr > 1.5))) & (plx < 1000.0 / CENSUS_PC) & t["Vmag"].notna()
    t = t[ok & ~t["HIP"].isin(census_hip)].copy()
    t["dist"] = 1000.0 / t["Plx"]
    xyz = _galactic(t["RAICRS"], t["DEICRS"], t["dist"])
    sin_b = xyz[:, 2] / t["dist"].to_numpy()
    sp = t["SpType"].fillna("").astype(str)
    teff = np.array([(m.value if (m := teff_from_spectral_type(s)) else np.nan) for s in sp])
    bv = t["B-V"].astype(float).to_numpy()
    teff = np.where(np.isfinite(teff), teff, 4600.0 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62)))  # Ballesteros 2012
    early = np.where(np.isfinite(teff), teff > 5600.0, True)
    keep = (t["Vmag"].to_numpy() <= survey_limit(sin_b, early)) | t["HIP"].isin(set(_iau()["hip"])).to_numpy()
    t, xyz, teff, sp = t[keep], xyz[keep], teff[keep], sp[keep]
    iau = _iau()
    proper = dict(zip(iau["hip"], iau["proper"]))
    b = bsc()
    bname = {int(h): re.sub(r"\s+", " ", str(n)).strip() for h, n in zip(b["HD"], b["Name"])
             if pd.notna(h) and isinstance(n, str) and n.strip()}
    hd = t["HD"].fillna(-1).astype(int)
    name = [proper.get(h) or bname.get(d) or f"HIP {h}" for h, d in zip(t["HIP"], hd)]
    lookup = [f"HD {d}" if d > 0 else f"HIP {h}" for h, d in zip(t["HIP"], hd)]
    return pd.DataFrame({"name": name, "lookup": lookup, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                         "vmag": t["Vmag"].to_numpy(float), "sptype": sp.to_numpy(), "teff": teff,
                         "n_comp": 1, "hip": t["HIP"].to_numpy(), "source": "hipparcos"})


@lru_cache(maxsize=1)
def stars() -> pd.DataFrame:
    """All real systems of the map (cached in data/real/starmap.parquet)."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    cen = _census_systems()
    iau = _iau()
    proper = dict(zip(iau["hip"], iau["proper"]))
    cen["name"] = [proper.get(int(h), n) if np.isfinite(h) else n for h, n in zip(cen["hip"], cen["name"])]
    from astropy.table import Table
    hv = Table.read(HIP_FILE)[["HIP", "Vmag"]].to_pandas().set_index("HIP")["Vmag"]
    v_hip = np.array([hv.get(int(h), np.nan) if np.isfinite(h) else np.nan for h in cen["hip"]])
    cen["vmag"] = np.where(np.isfinite(v_hip), v_hip, cen["vmag"])
    hip = _hipparcos(set(int(h) for h in cen["hip"] if np.isfinite(h)))
    df = pd.concat([cen, hip], ignore_index=True)
    df["teff"] = df["teff"].astype(float)
    df["hip"] = pd.to_numeric(df["hip"], errors="coerce")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def xyz() -> np.ndarray:
    s = stars()
    return s[["x", "y", "z"]].to_numpy()


@lru_cache(maxsize=1)
def _search_keys() -> dict[str, int]:
    from .real import names
    out = {}
    for i, (n, lk, h) in enumerate(zip(stars()["name"], stars()["lookup"], stars()["hip"])):
        for v in (n, lk) + ((f"HIP {int(h)}",) if np.isfinite(h) else ()):
            out.setdefault(names.key(str(v)), i)
            out.setdefault(str(v).lower(), i)
    return out


def search(text: str, limit: int = 20) -> list[int]:
    """Rows whose name or identifier matches ``text`` (exact key first, then substrings)."""
    from .real import names
    k = _search_keys()
    hits = []
    for key in (names.key(text), text.lower().strip()):
        if key in k and k[key] not in hits:
            hits.append(k[key])
    t = text.lower().strip()
    if t:
        nm = stars()["name"].str.lower() + " | " + stars()["lookup"].str.lower()
        for i in np.flatnonzero(nm.str.contains(re.escape(t)).to_numpy()):
            if i not in hits:
                hits.append(int(i))
            if len(hits) >= limit:
                break
    return hits[:limit]


@lru_cache(maxsize=4096)
def procedural_kept(key, tier: int = 0) -> np.ndarray:
    """Indices of the procedural systems of a cell (default: a 10-pc sector) that are shown
    (not in the catalogues)."""
    sec = gx.cell(tier, *key)
    return np.flatnonzero(~in_catalogue(sec))


REAL_SEED = 20261016


def real_seed(lookup: str) -> int:
    """The fixed seed of a real star's planetary system (from its catalogue identifier)."""
    import zlib

    from . import ids
    return ids.mix(REAL_SEED, zlib.crc32(str(lookup).encode()))


@lru_cache(maxsize=64)
def real_system(lookup: str, name: str | None = None):
    """The one planetary system of a real star: a single realisation conditioned on the star
    (:func:`starsys.real.condition.condition`) with a seed fixed by its catalogue identifier,
    so it is the same every time (and in every session). ID ``R:<name>``."""
    from .generate import Systems
    from .real import condition as cd
    from .real.catalog import find
    rec = find(lookup)
    b = Systems.from_conditioned(cd.condition(rec, n=1, seed=real_seed(lookup)))
    b.meta["ids"] = [f"R:{name or rec.name}"]
    return b


__all__ = ["abs_v", "extinction", "in_catalogue", "procedural_kept", "real_seed", "real_system", "search",
           "stars", "survey_limit", "system_vmag", "xyz"]
