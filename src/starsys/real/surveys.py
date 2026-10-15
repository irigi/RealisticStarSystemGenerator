"""Survey sensitivity for the real star: the probability that each model planet would have been
detected, used by :func:`starsys.real.condition.condition` as the non-detection weight
prod(1 - p_det) over the planets that are not known.

Each survey has ``prepare(record, systems, components, target, known, rng)`` (once per batch of
prior systems) and ``p_detect(cand)``, where ``cand`` holds the target star's model planets as
arrays: ``period`` (d), ``a_au``, ``planet_mass`` (Earth masses), ``planet_radius`` (Earth
radii), ``ecc``, ``host_mass`` (Msun), ``r_star`` (Rsun), ``cos_i`` (|cos| of the sky
inclination in the chosen view), ``age_gyr`` and ``system`` (batch system index).

- :class:`RVArchive`: the star's own archival radial velocities (HARPS RVBank, Trifonov et al.
  2020; HIRES, Butler et al. 2017; CARMENES DR1, Ribas et al. 2023, whose velocities are not
  public in VizieR: their epochs are spread evenly over 2016-2020 at 2 m/s). Per instrument
  segment (HARPS before/after the 2015 fibre upgrade, HIRES before/after the 2004 CCD upgrade)
  an offset is fitted; stars with a stellar companion inside 2000 au also get a quadratic
  trend. The phase-averaged expected Delta chi^2 of a planet of semi-amplitude K at period P is
  lambda = K^2 S(P), S being the noise-weighted signal power left after the nuisance fit; the
  detection probability is that of a noncentral chi^2 (2 d.o.f., lambda) exceeding 30, about
  a 1% false-alarm probability over a blind periodogram. The noise is the quoted errors plus
  a jitter estimated from the residual scatter, less the known planets' signals (at least
  1 m/s).
- :class:`RVMonitoring`: stars without archival velocities. Whether the star is monitored at
  all is drawn per prior system with the probability that a 20-pc census star of that
  brightness and type is in one of the three archives; monitored stars get a generic survey
  (60 epochs over 8 years at 2 m/s plus 2 m/s of jitter, 3 m/s for M dwarfs; the error grows
  above 6000 K as lines broaden and thin out). Stars with RV-detected known planets count as
  monitored.
- :class:`TESS`: sectors observed up to the catalogue date (tess-point); 1-hour noise from the
  photon, background and systematics budget of Sullivan et al. (2015, Sect. 6); detection at a
  multiple-event SNR of 7.3 with at least two transits.
- :class:`Kepler`: Kepler planet hosts; 4.5-hour CDPP of the star (DR25 stellar table; for stars
  outside its FGK sample, a typical value at the Kepler magnitude: 40 ppm at Kp = 12, x2.5 per 2
  mag, 30 ppm floor), 4 years
  at 92% duty cycle, DR25 detection efficiency (gamma CDF in MES, a = 30.87, b = 0.271,
  maximum 0.940; Christiansen 2017, KSCI-19110) with at least three transits.
- :class:`GaiaAstrometry`: Gaia DR3's RUWE below 1.4 excludes companions whose wobble would
  have raised it: expected RUWE^2 = 1 + alpha_eff^2 / (2 sigma^2), alpha_eff the astrometric
  signature (reduced by (34 months / P)^2 beyond the 34-month baseline, where the orbit is
  absorbed by proper motion), sigma the per-transit along-scan error at the star's G
  (Lindegren et al. 2021).
- :class:`TransitCampaign`: any other photometric campaign, by duration and 1-hour precision
  (e.g. Spitzer's 20 days on TRAPPIST-1).
- :class:`DirectImaging`: on for stars with planets found by imaging (or on request): planets
  brighter than a typical deep SPHERE/GPI 5-sigma contrast curve in H (11 mag at 0.3", +4.5 mag
  per dex of separation, at most 15.5 mag, nothing inside 0.12" or beyond 6"), with the planet's
  H magnitude from the ATMO 2020 hot-start tracks (Phillips et al. 2020) at the system's age,
  averaged over orbital phase.

Not modelled: microlensing, and RV data not public in VizieR (ESPRESSO, recent HARPS and
HIRES). The detection probabilities are thus lower bounds for well-observed stars.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import gamma, ncx2, norm

from ..paths import CATALOGS_DIR, DATA_DIR
from . import names
from .catalog import SystemRecord, census
from .known import KnownPlanet

RV_DIR = CATALOGS_DIR / "observational" / "rv"
STELLAR_DIR = CATALOGS_DIR / "stellar_tables"
KOI_FILE = CATALOGS_DIR / "observational" / "kepler_dr25_koi.csv"
RV_CACHE = DATA_DIR / "real" / "rv_archive.pkl"
K_JUP = 28.4329  # m/s for 1 MJup at 1 yr around 1 Msun
MJUP_EARTH = 317.83
RSUN_AU = 0.00465047
EARTH_RADIUS_RSUN = 0.0091577
HARPS_UPGRADE_BJD = 2457174.5  # 2015 June 1: fibre upgrade (new offset)
HIRES_UPGRADE_BJD = 2453237.0  # 2004 August: CCD upgrade
TESS_START = datetime.date(2018, 7, 25)
TESS_SECTOR_DAYS = 27.4
TESS_DUTY_DAYS = 25.0  # usable days per sector (downlink gaps)


def catalogue_date() -> datetime.date:
    p = CATALOGS_DIR / "observational" / "RETRIEVED_DATE"
    try:
        return datetime.date.fromisoformat(p.read_text().strip())
    except (OSError, ValueError):
        return datetime.date.today()


# --- archival RVs -------------------------------------------------------------------------

@dataclass
class RVSeries:
    instrument: str
    t: np.ndarray  # BJD
    err: np.ndarray  # m/s
    rv: np.ndarray | None  # m/s (None when only the epochs are known)
    segment: np.ndarray  # integer segment per epoch (separate offsets)


@lru_cache(maxsize=1)
def rv_archive() -> dict[str, list[RVSeries]]:
    """Name key -> archival RV series (cached as a pickle)."""
    if RV_CACHE.exists():
        return pd.read_pickle(RV_CACHE)
    from .cds import read

    out: dict[str, list[RVSeries]] = {}
    d = RV_DIR / "J_A_A_636_A74"
    h = read(d / "ReadMe", d / "rvbank.dat.gz")
    h = h[np.isfinite(h["DRVmlcnzp"]) & np.isfinite(h["e_DRVmlcnzp"]) & (h["e_DRVmlcnzp"] > 0)]
    for name, g in h.groupby("Name"):
        t = g["BJD"].to_numpy()
        out.setdefault(names.key(name), []).append(RVSeries(
            "HARPS", t, g["e_DRVmlcnzp"].to_numpy(), g["DRVmlcnzp"].to_numpy(), (t > HARPS_UPGRADE_BJD).astype(int)))
    d = RV_DIR / "J_AJ_153_208"
    k = read(d / "ReadMe", d / "table1.dat.gz")
    k = k[np.isfinite(k["RV"]) & (k["e_RV"] > 0)]
    for name, g in k.groupby("Name"):
        t = g["BJD"].to_numpy()
        out.setdefault(names.key(name), []).append(RVSeries(
            "HIRES", t, g["e_RV"].to_numpy(), g["RV"].to_numpy(), (t > HIRES_UPGRADE_BJD).astype(int)))
    d = RV_DIR / "J_A_A_670_A139"
    c = read(d / "ReadMe", d / "table1.dat")
    t0, t1 = 2457388.5, 2459215.5  # 2016-01-01 .. 2020-12-31
    for _, row in c.iterrows():
        nrv = int(row["NAVC"]) if np.isfinite(row["NAVC"]) else 0
        if nrv < 2:
            continue
        t = np.linspace(t0, t1, nrv)
        series = RVSeries("CARMENES", t, np.full(nrv, 2.0), None, np.zeros(nrv, dtype=int))
        for ident in (row["Karmn"], row["Name"]):
            if ident:
                out.setdefault(names.key(ident), []).append(series)
    RV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, RV_CACHE)
    return out


def star_keys(record: SystemRecord) -> set[str]:
    c = record.star
    keys = {names.key(v) for v in c.ids.values() if isinstance(v, str)} | {names.key(c.name)}
    if c.nasa_host:
        keys.add(names.key(c.nasa_host))
    keys |= {names.key(v) for col in ("Karmn",) for v in [c.ids.get(col, "")] if v}
    return {k for k in keys if k}


def rv_series_of(record: SystemRecord) -> list[RVSeries]:
    arch = rv_archive()
    out, seen = [], set()
    for k in star_keys(record):
        for s in arch.get(k, []):
            if id(s) not in seen:
                seen.add(id(s))
                out.append(s)
    return out


def semi_amplitude(mass_earth, period_d, m_star, ecc=0.0, sin_i=1.0) -> np.ndarray:
    """RV semi-amplitude (m/s)."""
    mp = np.asarray(mass_earth, dtype=float) / MJUP_EARTH
    return (K_JUP * mp * sin_i * np.asarray(m_star, dtype=float) ** (-2.0 / 3.0)
            * (np.asarray(period_d, dtype=float) / 365.25) ** (-1.0 / 3.0) / np.sqrt(1 - np.clip(ecc, 0, 0.95) ** 2))


def signal_power(t, w, segment, periods, trend: bool = False, chunk: int = 256) -> np.ndarray:
    """S(P): phase-averaged sum over epochs of w * (unit sinusoid - nuisance fit)^2 / 2, nuisance =
    per-segment offsets (+ linear and quadratic trend)."""
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    seg = np.asarray(segment)
    cols = [(seg == s).astype(float) for s in np.unique(seg)]
    if trend:
        x = (t - t.mean()) / max(np.ptp(t), 1.0)
        cols += [x, x**2]
    b = np.column_stack(cols)
    bw = b * w[:, None]
    proj = np.linalg.pinv(b.T @ bw) @ bw.T  # (m, N): nuisance coefficients from data
    out = np.empty(len(periods))
    for i0 in range(0, len(periods), chunk):
        p = np.asarray(periods[i0:i0 + chunk])
        th = 2 * np.pi * t[:, None] / p[None, :]
        tot = np.zeros(len(p))
        for f in (np.cos, np.sin):
            x = f(th)
            r = x - b @ (proj @ x)
            tot += (w[:, None] * r**2).sum(axis=0)
        out[i0:i0 + chunk] = 0.5 * tot
    return out


class Survey:
    name = "survey"

    def prepare(self, record: SystemRecord, systems, components, target, known: list[KnownPlanet], rng):
        pass

    def p_detect(self, cand: dict) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass
class RVArchive(Survey):
    threshold: float = 30.0  # Delta chi^2 for a detection
    min_jitter: float = 1.0  # m/s
    companion_trend_au: float = 2000.0
    name: str = "RV archive"
    _grid: np.ndarray = field(default=None, repr=False)
    _power: np.ndarray = field(default=None, repr=False)
    _info: str = ""

    def prepare(self, record, systems, components, target, known, rng):
        if self._power is not None:
            return
        series = rv_series_of(record)
        self._grid = np.geomspace(0.3, 1e6, 3000)
        if not series:
            self._power = np.zeros(len(self._grid))
            self._info = "no archival RVs"
            return
        from ..binaries import structure
        a_near = structure(components)["a_near_any"][target]
        trend = bool(np.nanmedian(np.where(np.isfinite(a_near), a_near, np.inf)) < self.companion_trend_au)
        jit = self._jitter(series, record, known, trend)
        t = np.concatenate([s.t for s in series])
        err = np.concatenate([s.err for s in series])
        seg = np.concatenate([s.segment + 10 * i for i, s in enumerate(series)])
        w = 1.0 / (err**2 + jit**2)
        self._power = signal_power(t, w, seg, self._grid, trend=trend)
        span = (t.max() - t.min()) / 365.25
        inst = ", ".join(sorted({s.instrument for s in series}))
        self._info = (f"{len(t)} RVs ({inst}) over {span:.1f} yr, median error {np.median(err):.1f} m/s, "
                      f"jitter {jit:.1f} m/s" + (", with trend" if trend else ""))

    def _jitter(self, series, record, known, trend) -> float:
        """Residual scatter of the archival RVs (offsets, trend) less the errors and known planets."""
        with_rv = [s for s in series if s.rv is not None]
        if not with_rv:
            return 2.0 if (record.star.measurements.get("teff") and
                           record.star.measurements["teff"].value > 3900) else 3.0
        res, errs = [], []
        for i, s in enumerate(with_rv):
            cols = [(s.segment == g).astype(float) for g in np.unique(s.segment)]
            if trend and len(s.t) > 4:
                x = (s.t - s.t.mean()) / max(np.ptp(s.t), 1.0)
                cols += [x, x**2]
            b = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(b, s.rv, rcond=None)
            res.append(s.rv - b @ coef)
            errs.append(s.err)
        res, errs = np.concatenate(res), np.concatenate(errs)
        # Robust scatter: 1.4826 MAD.
        rms2 = (1.4826 * np.median(np.abs(res - np.median(res)))) ** 2
        m_star = record.star.measurements.get("mass")
        m_star = m_star.value if m_star else 1.0
        k2 = sum(semi_amplitude(k.mass if np.isfinite(k.mass) else k.msini, k.period, m_star) ** 2 / 2
                 for k in known if np.isfinite(k.period) and (np.isfinite(k.mass) or np.isfinite(k.msini)))
        return float(np.sqrt(max(rms2 - np.median(errs) ** 2 - k2, self.min_jitter**2)))

    def p_detect(self, cand):
        if not np.any(self._power):
            return np.zeros(len(cand["period"]))
        sin_i = np.sqrt(1 - np.clip(cand["cos_i"], 0, 1) ** 2)
        k = semi_amplitude(cand["planet_mass"], cand["period"], cand["host_mass"], cand["ecc"], sin_i)
        s = np.interp(np.log(cand["period"]), np.log(self._grid), self._power)
        return ncx2.sf(self.threshold, 2, k**2 * s)

    def describe(self):
        return f"{self.name}: {self._info}"


# --- RV monitoring prior --------------------------------------------------------------------

@lru_cache(maxsize=1)
def monitoring_table() -> pd.DataFrame:
    """Fraction of 20-pc census stars with archival RVs, by brightness (G, or V - 0.3) and type."""
    c = census()
    leaves = c[(c["NcTR"] == 1)].copy()
    arch = rv_archive()
    has = []
    for _, row in leaves.iterrows():
        keys = {names.key(v) for col in ("Name", "GJ", "HD", "HIP", "Karmn", "LHS", "Wolf", "Ross", "BD", "CD")
                for v in str(row.get(col, "")).split(",") if isinstance(row.get(col), str)}
        has.append(any(k in arch for k in keys if k))
    leaves["has_rv"] = has
    leaves["mag"] = leaves["Gmag"]
    leaves["cls"] = np.where(leaves["SpTOpt"].astype(str).str.match(r"^(sd)?M"), "M",
                             np.where(leaves["SpTOpt"].astype(str).str.match(r"^[FGK]"), "FGK", "other"))
    leaves = leaves[leaves["cls"] != "other"]
    bins = [-5, 6, 8, 10, 12, 14, 30]
    leaves["bin"] = pd.cut(leaves["mag"].fillna(4.0), bins)
    return leaves.groupby(["cls", "bin"], observed=True)["has_rv"].agg(["mean", "size"]).reset_index()


def monitoring_probability(mag: float, teff: float) -> float:
    t = monitoring_table()
    cls = "M" if teff < 3900 else "FGK"
    rows = t[t["cls"] == cls]
    for _, r in rows.iterrows():
        if r["bin"].left < (mag if np.isfinite(mag) else 4.0) <= r["bin"].right:
            return float(r["mean"])
    return 0.0


def hot_star_rv_factor(teff) -> float:
    """RV error relative to a Sun-like star: few lines and fast rotation above ~6000 K (x5 at
    7000 K, x30 at 8000 K, at most x150; cf. Lagrange et al. 2009 for A-F stars with HARPS)."""
    return float(min(np.exp(max(teff - 6000.0, 0.0) / 600.0), 150.0))


@dataclass
class RVMonitoring(Survey):
    n_epochs: int = 60
    years: float = 8.0
    error: float = 2.0  # m/s
    jitter_fgk: float = 2.0
    jitter_m: float = 3.0
    threshold: float = 30.0
    p_monitored: float | None = None  # None: from the census calibration (1 with RV-detected planets)
    name: str = "RV monitoring (generic)"
    _monitored: np.ndarray = field(default=None, repr=False)
    _power: np.ndarray = field(default=None, repr=False)
    _grid: np.ndarray = field(default=None, repr=False)
    _p: float = 0.0
    _err: float = np.nan

    def prepare(self, record, systems, components, target, known, rng):
        star = record.star
        teff = star.measurements["teff"].value if "teff" in star.measurements else 5000.0
        if self.p_monitored is not None:
            p = self.p_monitored
        elif any(k.rv_detected for k in known):
            p = 1.0
        else:
            g = star.photometry.get("G", np.nan)
            if not np.isfinite(g) and np.isfinite(star.photometry.get("V", np.nan)):
                g = star.photometry["V"] - 0.3
            p = monitoring_probability(g, teff)
        self._p = p
        self._monitored = rng.random(len(target)) < p
        if self._power is None:
            jit = self.jitter_m if teff < 3900 else self.jitter_fgk
            self._err = self.error * hot_star_rv_factor(teff)
            t = np.linspace(0.0, self.years * 365.25, self.n_epochs) + rng.uniform(-30, 30, self.n_epochs)
            self._grid = np.geomspace(0.3, 1e6, 3000)
            w = np.full(self.n_epochs, 1.0 / (self._err**2 + jit**2))
            self._power = signal_power(t, w, np.zeros(self.n_epochs, int), self._grid)

    def p_detect(self, cand):
        sin_i = np.sqrt(1 - np.clip(cand["cos_i"], 0, 1) ** 2)
        k = semi_amplitude(cand["planet_mass"], cand["period"], cand["host_mass"], cand["ecc"], sin_i)
        s = np.interp(np.log(cand["period"]), np.log(self._grid), self._power)
        return ncx2.sf(self.threshold, 2, k**2 * s) * self._monitored[cand["system"]]

    def describe(self):
        return (f"{self.name}: monitored with probability {self._p:.2f}; {self.n_epochs} epochs over "
                f"{self.years:g} yr at {self._err:.3g} m/s")


# --- transits -----------------------------------------------------------------------------

def tess_noise_1h(tmag) -> np.ndarray:
    """1-hour TESS photometric noise (fraction): photon noise of 1.5e4 e-/s at T = 10, background
    and read noise equivalent to 300 e-/s, 60 ppm systematics floor (Sullivan et al. 2015)."""
    f = 1.5e4 * 10 ** (-0.4 * (np.asarray(tmag, dtype=float) - 10.0))
    phot = np.sqrt((f + 300.0) * 3600.0) / (f * 3600.0)
    return np.hypot(phot, 60e-6)


def tmag_estimate(star) -> float:
    ph = star.photometry
    if np.isfinite(ph.get("T", np.nan)):
        return ph["T"]
    g, bp, rp = ph.get("G", np.nan), ph.get("BP", np.nan), ph.get("RP", np.nan)
    if np.isfinite(g) and np.isfinite(bp - rp):
        c = bp - rp  # Stassun et al. (2019), eq. 1
        return g - 0.00522555 * c**3 + 0.0891337 * c**2 - 0.633923 * c + 0.0324473
    if np.isfinite(g):
        return g - 0.43
    if np.isfinite(ph.get("V", np.nan)):
        return ph["V"] - 0.6
    return np.nan


def _transit_snr(cand, depth_noise_1h, days):
    r_star_au = cand["r_star"] * RSUN_AU
    k = cand["planet_radius"] * EARTH_RADIUS_RSUN / cand["r_star"]
    b = cand["cos_i"] * cand["a_au"] / r_star_au
    transits = b < 1 + k
    chord = np.sqrt(np.clip((1 + k) ** 2 - b**2, 0, None))
    dur_h = cand["period"] * 24 / np.pi * np.arcsin(np.clip(r_star_au / cand["a_au"] * chord, 0, 1))
    n_tr = days / cand["period"]
    snr = k**2 / depth_noise_1h * np.sqrt(np.maximum(n_tr, 0) * np.maximum(dur_h, 1e-3))
    return transits, snr, n_tr, dur_h


@dataclass
class TESS(Survey):
    snr_threshold: float = 7.3
    min_transits: int = 2
    name: str = "TESS"
    _n_sectors: int = 0
    _noise: float = np.nan
    _tmag: float = np.nan

    def prepare(self, record, systems, components, target, known, rng):
        if self._n_sectors or np.isfinite(self._noise):
            return
        star = record.star
        self._tmag = tmag_estimate(star)
        self._noise = float(tess_noise_1h(self._tmag)) if np.isfinite(self._tmag) else np.nan
        self._n_sectors = tess_sectors(star.ra, star.dec)

    def p_detect(self, cand):
        if not self._n_sectors or not np.isfinite(self._noise):
            return np.zeros(len(cand["period"]))
        transits, snr, n_tr, _ = _transit_snr(cand, self._noise, self._n_sectors * TESS_DUTY_DAYS)
        return transits * (n_tr >= self.min_transits) * norm.cdf(snr - self.snr_threshold)

    def describe(self):
        return (f"{self.name}: {self._n_sectors} sector(s), T = {self._tmag:.1f}, "
                f"1-h noise {self._noise * 1e6:.0f} ppm")


def tess_sectors(ra, dec, date: datetime.date | None = None) -> int:
    """Number of TESS sectors that observed (ra, dec) and ended before ``date`` (catalogue date)."""
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return 0
    try:
        from tess_stars2px import tess_stars2px_function_entry
    except ImportError:
        return 0
    date = date or catalogue_date()
    last = int((date - TESS_START).days // TESS_SECTOR_DAYS)
    r = tess_stars2px_function_entry(1, float(ra), float(dec))
    sectors = {int(s) for s in np.atleast_1d(r[3]) if int(s) > 0}
    return sum(1 for s in sectors if s <= last)


@dataclass
class TransitCampaign(Survey):
    """A photometric campaign: ``days`` of coverage at ``noise_1h`` (fractional 1-hour precision)."""

    days: float = 20.0
    noise_1h: float = 3e-4
    snr_threshold: float = 7.3
    min_transits: int = 2
    name: str = "transit campaign"

    def p_detect(self, cand):
        transits, snr, n_tr, _ = _transit_snr(cand, self.noise_1h, self.days)
        return transits * (n_tr >= self.min_transits) * norm.cdf(snr - self.snr_threshold)

    def describe(self):
        return f"{self.name}: {self.days:g} d at {self.noise_1h * 1e6:.0f} ppm/h"


@lru_cache(maxsize=1)
def _kepler_tables():
    koi = pd.read_csv(KOI_FILE, comment="#", low_memory=False, usecols=["kepid", "kepoi_name", "kepler_name"])
    st = pd.read_csv(STELLAR_DIR / "q1_q17_dr25_gaia_berger_fgk_HFR2020b_cleaned.csv",
                     usecols=["kepid", "rrmscdpp04p5"])
    return koi, st


@dataclass
class Kepler(Survey):
    days: float = 1426.0
    duty: float = 0.92
    min_transits: int = 3
    name: str = "Kepler"
    cdpp_4p5: float = np.nan  # ppm; looked up for Kepler hosts
    _on: bool = False

    def prepare(self, record, systems, components, target, known, rng):
        if self._on or np.isfinite(self.cdpp_4p5):
            self._on = True
            return
        host = record.star.nasa_host
        koi, st = _kepler_tables()
        # Host keys: "KOI-351" for K00351.xx, "Kepler-90" for Kepler-90 b.
        koi_key = "koi-" + koi["kepoi_name"].astype(str).str[1:6].str.lstrip("0")
        kep_key = koi["kepler_name"].fillna("").astype(str).str.replace(r"\s+[a-z]$", "", regex=True).map(names.key)
        keys = {names.key(host)} | {names.key(k.name.rsplit(" ", 1)[0]) for k in known}
        sel = koi[koi_key.isin(keys) | kep_key.isin(keys)]
        if not len(sel):
            return
        kepid = sel["kepid"].iloc[0]
        c = st.loc[st["kepid"] == kepid, "rrmscdpp04p5"]
        if len(c) and np.isfinite(c.iloc[0]):
            self.cdpp_4p5 = float(c.iloc[0])
        else:  # not in the FGK table: typical 4.5-h CDPP at the star's Kepler magnitude
            kp = record.star.photometry.get("Kp", np.nan)
            self.cdpp_4p5 = float(np.hypot(30.0, 40.0 * 10 ** (0.2 * (kp - 12.0)))) if np.isfinite(kp) else 100.0
        self._on = True

    def p_detect(self, cand):
        if not self._on:
            return np.zeros(len(cand["period"]))
        transits, _, n_tr, dur_h = _transit_snr(cand, 1.0, self.days * self.duty)
        k = cand["planet_radius"] * EARTH_RADIUS_RSUN / cand["r_star"]
        cdpp = self.cdpp_4p5 * 1e-6 * np.sqrt(4.5 / np.maximum(dur_h, 0.1))
        mes = k**2 / cdpp * np.sqrt(np.maximum(n_tr, 0))
        return transits * (n_tr >= self.min_transits) * 0.940 * gamma.cdf(mes, a=30.87, scale=0.271)

    def describe(self):
        return f"{self.name}: CDPP(4.5 h) {self.cdpp_4p5:.0f} ppm" if self._on else f"{self.name}: not a target"


# --- astrometry -----------------------------------------------------------------------------

def gaia_sigma_fov(g) -> np.ndarray:
    """Along-scan precision per field-of-view transit (mas) vs G (Lindegren et al. 2021, Fig. A.1,
    approximate): ~0.1 mas for G < 13, 0.3 at 15, 1 at 17, 3 at 19."""
    g = np.asarray(g, dtype=float)
    return np.maximum(0.1, 0.1 * 10 ** (0.24 * (g - 13.0)))


@dataclass
class GaiaAstrometry(Survey):
    ruwe_limit: float = 1.4
    ruwe_scatter: float = 0.07
    baseline_days: float = 1038.0  # DR3: 34 months
    name: str = "Gaia DR3 astrometry"
    _on: bool = False
    _sigma: float = np.nan
    _dist: float = np.nan
    _why: str = ""

    def prepare(self, record, systems, components, target, known, rng):
        star = record.star
        g = star.photometry.get("G", np.nan)
        if not np.isfinite(star.ruwe):
            self._why = "no RUWE"
        elif star.ruwe >= self.ruwe_limit:
            self._why = f"RUWE {star.ruwe:.2f}: excess already present"
        elif not np.isfinite(g) or g < 3.0:
            self._why = "too bright or no G"
        elif not np.isfinite(star.distance_pc):
            self._why = "no distance"
        else:
            self._on = True
            self._sigma = float(gaia_sigma_fov(g))
            self._dist = star.distance_pc
            self._why = f"RUWE {star.ruwe:.2f}, G = {g:.1f}"

    def p_detect(self, cand):
        if not self._on:
            return np.zeros(len(cand["period"]))
        q = cand["planet_mass"] * 3.0035e-6 / cand["host_mass"]
        alpha = q * cand["a_au"] / self._dist * 1000.0  # mas
        damp = np.minimum(1.0, (self.baseline_days / cand["period"]) ** 2)
        ruwe = np.sqrt(1 + (alpha * damp) ** 2 / (2 * self._sigma**2))
        return norm.cdf((ruwe - self.ruwe_limit) / self.ruwe_scatter)

    def describe(self):
        return f"{self.name}: " + ("used, " if self._on else "not used, ") + self._why


# --- direct imaging ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def atmo_h_grid():
    """(log10 mass [Msun] grid, log10 age [Gyr] grid, absolute MKO H magnitude table) from ATMO 2020."""
    from scipy.interpolate import interp1d

    from ..stars.substellar import ATMO_DIR

    ages = np.linspace(-3.0, 1.0, 81)
    masses, table = [], []
    for f in sorted(ATMO_DIR.glob("*_ATMO_CEQ_vega.txt")):
        d = np.loadtxt(f, comments="#")
        masses.append(d[0, 0])
        la = np.log10(d[:, 1])
        h = interp1d(la, d[:, 8], bounds_error=False, fill_value=(d[0, 8], d[-1, 8]))(ages)
        table.append(h)
    order = np.argsort(masses)
    return np.log10(np.array(masses)[order]), ages, np.array(table)[order]


def planet_abs_h(mass_earth, age_gyr) -> np.ndarray:
    """Absolute MKO H magnitude of a planet (hot start, ATMO 2020); 99 below 0.3 Jupiter masses."""
    from scipy.interpolate import RegularGridInterpolator

    lm_grid, la_grid, tab = atmo_h_grid()
    f = RegularGridInterpolator((lm_grid, la_grid), tab, bounds_error=False, fill_value=None)
    lm = np.log10(np.asarray(mass_earth, dtype=float) * 3.0035e-6)
    la = np.clip(np.log10(np.asarray(age_gyr, dtype=float)), la_grid[0], la_grid[-1])
    out = f(np.column_stack([np.clip(lm, lm_grid[0], lm_grid[-1]), la]))
    return np.where(lm < np.log10(0.3 * 9.546e-4), 99.0, out)


def contrast_limit(rho_arcsec) -> np.ndarray:
    """Typical deep SPHERE/GPI 5-sigma H-band contrast (mag) at separation ``rho`` (arcsec)."""
    r = np.asarray(rho_arcsec, dtype=float)
    lim = np.clip(11.0 + 4.5 * np.log10(np.maximum(r, 1e-3) / 0.3), 0.0, 15.5)
    return np.where((r < 0.12) | (r > 6.0), -np.inf, lim)


@dataclass
class DirectImaging(Survey):
    enabled: bool | None = None  # None: on if a known planet was found by imaging
    n_phase: int = 8
    name: str = "direct imaging"
    _on: bool = False
    _h_star: float = np.nan
    _dist: float = np.nan

    def prepare(self, record, systems, components, target, known, rng):
        star = record.star
        on = self.enabled if self.enabled is not None else any(k.method == "imaging" for k in known)
        ph = star.photometry
        h = ph.get("H", np.nan)
        if not np.isfinite(h):
            h = ph.get("Ks", np.nan) + 0.1 if np.isfinite(ph.get("Ks", np.nan)) else ph.get("J", np.nan) - 0.3
        self._h_star, self._dist = h, star.distance_pc
        self._on = bool(on and np.isfinite(h) and np.isfinite(self._dist))

    def p_detect(self, cand):
        if not self._on:
            return np.zeros(len(cand["period"]))
        age = cand.get("age_gyr", np.full(len(cand["period"]), 1.0))
        h_p = planet_abs_h(cand["planet_mass"], age) + 5 * np.log10(self._dist / 10.0)
        dmag = h_p - self._h_star
        sin_i2 = 1 - np.clip(cand["cos_i"], 0, 1) ** 2
        phases = (np.arange(self.n_phase) + 0.5) / self.n_phase * 2 * np.pi
        rho = (cand["a_au"] / self._dist)[:, None] * np.sqrt(1 - np.sin(phases)[None, :] ** 2 * sin_i2[:, None])
        return (dmag[:, None] < contrast_limit(rho)).mean(axis=1)

    def describe(self):
        if not self._on:
            return f"{self.name}: not used"
        return f"{self.name}: H = {self._h_star:.1f} at {self._dist:.1f} pc, typical deep contrast curve"


# --- default set ------------------------------------------------------------------------------

def default_surveys(record: SystemRecord, known: list[KnownPlanet]) -> list[Survey]:
    """The surveys that apply to ``record``'s star: archival RVs (or the monitoring prior), TESS,
    Kepler for Kepler hosts, Gaia astrometry, and direct imaging where planets were imaged."""
    out: list[Survey] = []
    out.append(RVArchive() if rv_series_of(record) else RVMonitoring())
    out.append(TESS())
    if record.star.nasa_host and (record.star.nasa_host.startswith(("Kepler", "KOI")) or
                                  any(k.name.startswith(("Kepler", "KOI")) for k in known)):
        out.append(Kepler())
    out.append(GaiaAstrometry())
    if any(k.method == "imaging" for k in known):
        out.append(DirectImaging())
    return out
