"""A procedural Milky Way: star-system density, populations, and coordinate-seeded sectors.

How the games do it (the approach followed here)
------------------------------------------------
Frontier (Elite II), Pioneer and Space Engine never store the galaxy. Space is cut into cells
(sectors, or octree nodes in Space Engine); a cell's seed is a hash of its coordinates; the
number of stars in a cell follows a galaxy *density model* (Frontier and Pioneer: a density
map with spiral arms; Space Engine: an analytic disc + bulge + halo with spiral arms and more
young blue stars in the arms); each star's seed comes from the cell seed and its index, so a
star is regenerated identically whenever its cell is visited. Space Engine and Pioneer place
catalogued real stars near the Sun and fill the rest procedurally. Sources: Space Engine
developer blog (spaceengine.org, "procedural galaxy" posts), the gamedev.net "galaxy
generation" octree write-up, the Pioneer source (github.com/pioneerspacesim/pioneer).

Here the cells are cubes of :data:`SECTOR_PC` parsecs; the counts are Poisson with the mean
given by an empirical Milky Way model (all parameters with sources below), the stars of each
component are drawn by :func:`starsys.stars.population.draw_systems` with a population model
appropriate to the component and position (ages, [Fe/H]); IMF and multiplicity are the local
ones everywhere (an assumption: no strong evidence for IMF variations within the Milky Way,
Bastian, Covey & Meyer 2010).

Brightness tiers. A map has to show the bright distant stars long before the faint nearby
ones, and they must be the *same* stars when one flies there. So systems whose primary is
a living star of 1.5 Msun or more (A, B, O stars and their giants; ~1 % of systems) are not
drawn in the 10-pc sectors but in larger cells of their own (:data:`TIERS`): 1.5-3 Msun in
80-pc cells, 3-8 Msun in 160-pc cells, >= 8 Msun in 1280-pc cells (~2e5 in the whole
galaxy, all of which a full-galaxy view can load). The 10-pc sectors hold everything else
(``draw_systems(exclude_alive_above=1.5)``) at the correspondingly reduced density. Large
cells place their stars on an 8^3 sub-grid of densities (the disc scale height is smaller
than a 1280-pc cell). IDs: ``G<ix>.<iy>.<iz>-<j>`` (sectors) and ``G<t>:<ix>.<iy>.<iz>-<j>``
(tier t cells).

Coordinates: heliocentric Galactic Cartesian, parsecs: x toward the Galactic centre
(l = 0), y toward l = 90° (direction of rotation), z toward the north Galactic pole.

Model (Bland-Hawthorn & Gerhard 2016, BHG16, unless noted)
-----------------------------------------------------------
* R0 = 8.15 kpc, Sun 5.5 pc above the plane (Reid et al. 2019).
* Thin disc: exponential, scale length 2.6 kpc, scale height 350 pc (BHG16: 300 ± 50 pc;
  the upper end so that, with the thick disc, the local stellar surface density approaches
  McKee et al. 2015's ~33 Msun/pc^2 given the census mid-plane density); a Gaussian inner
  hole of 2.5 kpc where the bar dominates (BHG16 §5); outer edge ~14 kpc
  (soft truncation; Minniti et al. 2011 found the stellar disc edge at 13.9 kpc).
* Thick disc: scale length 2.0 kpc, scale height 900 pc, 4 % of the local density
  (BHG16: 4 ± 2 %; the same 4 % as the local age model, starsys.stars.population).
* Bar/bulge: Dwek et al. (1995) G2 (boxy Gaussian) profile with scale lengths 1.6, 0.6,
  0.4 kpc (Rattenbury et al. 2007 fits of the same form), bar angle 27° from the Sun–centre
  line (BHG16), stellar mass 1.6e10 Msun (BHG16: 1.4–1.7e10).
* Stellar halo: oblate power law, rho ∝ r^-2.77, axis ratio 0.64 (Juric et al. 2008), core
  1 kpc, out to 60 kpc. Normalised to the halo's total stellar mass, 1.4e9 Msun (Deason et
  al. 2019), which gives 0.18 % of the local density; Juric et al.'s local 0.5 % would make
  the halo ~3x too massive with this profile (tension noted, mass preferred).
* Spiral arms: log-periodic spirals with kinks fitted to maser parallaxes (Reid et al. 2019,
  Table 2, :data:`ARMS`), within their measured azimuth ranges; beyond them the four major
  arms are extended with their mean pitch angle of 10° (Reid et al. 2019 §5) from 3 to
  15 kpc, the Local arm only ±25° beyond its measured range (it is an isolated segment).
  Arm width: Gaussian sigma w(Rkink) + 42 pc/kpc (R - Rkink) (Reid et al. 2019).
  Old stars show weak arm contrast: the thin-disc density is modulated by +-15 % (arms have
  ~10-20 % stellar mass contrast in near-IR, e.g. Drimmel & Spergel 2001); young stars
  (< 300 Myr) are strongly confined: their density is scaled by (0.1 + S)/(0.1 + S_sun),
  with S the arm profile (0 between arms, 1 on an arm centre).
* Local normalisation: 0.081 systems pc^-3 at the Sun (the 20-pc census component IMF of
  Kirkpatrick et al. 2024 divided by the generator's 1.37 components per system).
* Youngest thin-disc age 1 Myr (the local model's 10 Myr would leave no living star above
  ~20 Msun anywhere).
* [Fe/H]: thin-disc radial gradient -0.06 dex/kpc (Genovali et al. 2014, Cepheids; flatter
  for old stars, kept common here); thick disc as locally; bulge N(0, 0.35) dex, 10 ± 1.5 Gyr
  (old, metal-rich on average, Zoccali et al. 2017); halo N(-1.5, 0.5), 11–13 Gyr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from . import ids
from .stars.population import PopulationModel, default_model, draw_systems

GALAXY_SEED = 20261015
SECTOR_PC = 10.0
R0_PC = 8150.0
Z_SUN_PC = 5.5
LOCAL_DENSITY = 0.081  # systems per pc^3 at the Sun

THIN_L, THIN_H = 2600.0, 350.0
THICK_L, THICK_H, THICK_LOCAL = 2000.0, 900.0, 0.04
HALO_LOCAL, HALO_SLOPE, HALO_Q = 0.0018, 2.77, 0.64
BAR_SCALES = (1600.0, 600.0, 400.0)
BAR_ANGLE_DEG = 27.0
BULGE_MASS = 1.6e10
DISC_EDGE, DISC_EDGE_W = 14000.0, 800.0
DISC_HOLE = 2500.0
OLD_ARM_CONTRAST = 0.15
YOUNG_AGE_GYR = 0.3
YOUNG_BOOST_MAX = 1.5
THIN_MIN_AGE = 0.001  # Gyr

# brightness tiers: tier -> (initial-mass range of a living primary in Msun, cell size in pc)
TIERS = {1: (1.5, 3.0, 80.0), 2: (3.0, 8.0, 160.0), 3: (8.0, math.inf, 1280.0)}
TIER_MIN_MASS = 1.5
TIER_SUB = 8          # sub-grid per cell axis for the stars' positions
_TIER_KEY = 1_000_000  # seed namespace of the tier cells

# Reid et al. (2019) Table 2: name, beta range (deg), beta_kink (deg), R_kink (kpc),
# psi before / after the kink (deg), width at R_kink (kpc), major arm?
ARMS = [
    ("Norma-Outer", (5, 54), 18, 4.46, -1.0, 19.5, 0.14, True),
    ("Outer", (-16, 71), 18, 12.24, 3.0, 9.4, 0.65, False),  # joined to Norma (same arm)
    ("Scutum-Centaurus", (0, 104), 23, 4.91, 14.1, 12.1, 0.23, True),
    ("Sagittarius-Carina", (2, 97), 24, 6.04, 17.1, 1.0, 0.27, True),
    ("Local", (-8, 34), 9, 8.26, 11.4, 11.4, 0.31, False),
    ("Perseus", (-23, 115), 40, 8.87, 10.3, 8.7, 0.35, True),
]
EXTRA_PITCH = 10.0


# ------------------------------------------------------------------------------- geometry

def galactocentric(xyz_pc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, beta, z) of heliocentric points; beta = Galactocentric azimuth of Reid et al.
    (0 toward the Sun, increasing in the direction of rotation), radians."""
    xyz = np.atleast_2d(xyz_pc)
    xg = xyz[:, 0] - R0_PC
    yg = xyz[:, 1]
    R = np.hypot(xg, yg)
    beta = np.pi - np.arctan2(yg, xg)
    beta = (beta + np.pi) % (2 * np.pi) - np.pi
    return R, beta, xyz[:, 2] + Z_SUN_PC


def from_galactocentric(R, beta, z):
    th = np.pi - np.asarray(beta)
    return np.stack([R * np.cos(th) + R0_PC, R * np.sin(th), np.asarray(z) - Z_SUN_PC], axis=-1)


def _arm_curve(name, brange, bk, rk, psi_lo, psi_hi, w, major):
    """Densely sampled centre line of an arm (x, y heliocentric pc) and its width there."""
    bk_r = math.radians(bk)
    lo, hi = math.radians(brange[0]), math.radians(brange[1])

    def radius(beta):
        psi = np.where(beta <= bk_r, psi_lo, psi_hi)
        return rk * 1000.0 * np.exp(-(beta - bk_r) * np.tan(np.radians(psi)))

    if major:
        # extend at 10° pitch from both ends until R leaves 3-15 kpc
        t = math.tan(math.radians(EXTRA_PITCH))
        r_lo, r_hi = float(radius(np.array(lo))), float(radius(np.array(hi)))
        b_in = hi + math.log(r_hi / 3000.0) / t       # inward with increasing beta
        b_out = lo - math.log(15000.0 / r_lo) / t     # outward with decreasing beta
        beta = np.arange(b_out, b_in, 0.002)
        r = np.where(beta < lo, r_lo * np.exp(-(beta - lo) * t),
                     np.where(beta > hi, r_hi * np.exp(-(beta - hi) * t), radius(beta)))
        taper = np.ones_like(beta)
    else:
        m = math.radians(25)
        beta = np.arange(lo - m, hi + m, 0.002)
        r = radius(beta)
        taper = np.clip(1 - np.maximum(lo - beta, beta - hi) / m, 0, 1)
    width = np.maximum(w * 1000.0 + 42.0 * (r / 1000.0 - rk), 80.0)
    xyz = from_galactocentric(r, beta, np.zeros_like(r))
    return xyz[:, :2], width, taper


@lru_cache(maxsize=1)
def _arm_trees():
    out = []
    for arm in ARMS:
        xy, width, taper = _arm_curve(*arm)
        out.append((arm[0], cKDTree(xy), width, taper))
    return out


ARM_GRID_PC = 25.0
ARM_EXTENT_PC = 16500.0  # raster half-width around the Galactic centre


def _arm_strength_exact(xyz_pc: np.ndarray) -> np.ndarray:
    xy = np.atleast_2d(xyz_pc)[:, :2]
    s = np.zeros(len(xy))
    for _, tree, width, taper in _arm_trees():
        d, k = tree.query(xy, distance_upper_bound=3000.0)
        ok = np.isfinite(d)
        val = np.zeros(len(xy))
        val[ok] = np.exp(-0.5 * (d[ok] / width[k[ok]]) ** 2) * taper[k[ok]]
        s = np.maximum(s, val)
    return s


@lru_cache(maxsize=1)
def _arm_raster() -> np.ndarray:
    """S on a 25-pc grid (Galactocentric x, y), cached in data/galaxy/ (it takes a few s)."""
    from .paths import DATA_DIR
    n = int(round(2 * ARM_EXTENT_PC / ARM_GRID_PC)) + 1
    path = DATA_DIR / "galaxy" / f"arm_raster_{n}.npy"
    if path.exists():
        return np.load(path)
    g = np.linspace(-ARM_EXTENT_PC, ARM_EXTENT_PC, n)
    xx, yy = np.meshgrid(g + R0_PC, g, indexing="ij")
    ras = _arm_strength_exact(np.stack([xx.ravel(), yy.ravel()], 1)).reshape(n, n).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ras)
    return ras


def arm_strength(xyz_pc: np.ndarray) -> np.ndarray:
    """Spiral-arm profile S in [0, 1] (1 on the centre line of an arm), bilinear in a 25-pc
    raster of the exact profile (arms are >= 80 pc wide)."""
    xy = np.atleast_2d(np.asarray(xyz_pc, float))[:, :2]
    ras = _arm_raster()
    n = ras.shape[0]
    fx = (xy[:, 0] - R0_PC + ARM_EXTENT_PC) / ARM_GRID_PC
    fy = (xy[:, 1] + ARM_EXTENT_PC) / ARM_GRID_PC
    inside = (fx >= 0) & (fx < n - 1) & (fy >= 0) & (fy < n - 1)
    i = np.clip(np.floor(fx).astype(int), 0, n - 2)
    j = np.clip(np.floor(fy).astype(int), 0, n - 2)
    u, v = np.clip(fx - i, 0, 1), np.clip(fy - j, 0, 1)
    s = (ras[i, j] * (1 - u) * (1 - v) + ras[i + 1, j] * u * (1 - v) + ras[i, j + 1] * (1 - u) * v
         + ras[i + 1, j + 1] * u * v)
    return np.where(inside, s, 0.0).astype(float)


def arm_lines() -> list[tuple[str, np.ndarray]]:
    """Arm centre lines (name, n x 2 heliocentric pc) for drawing."""
    return [(arm[0], _arm_curve(*arm)[0]) for arm in ARMS]


# ------------------------------------------------------------------------------- densities

def _thin_raw(R, z):
    edge = 1.0 / (1.0 + np.exp((R - DISC_EDGE) / DISC_EDGE_W))
    hole = 1.0 - np.exp(-((R / DISC_HOLE) ** 2))
    return np.exp(-(R - R0_PC) / THIN_L) * np.exp(-np.abs(z) / THIN_H) * edge * hole


def _thick_raw(R, z):
    edge = 1.0 / (1.0 + np.exp((R - DISC_EDGE) / DISC_EDGE_W))
    return np.exp(-(R - R0_PC) / THICK_L) * np.exp(-np.abs(z) / THICK_H) * edge


def _halo_raw(R, z):
    r = np.sqrt(R**2 + (z / HALO_Q) ** 2)
    return (np.maximum(r, 1000.0) / R0_PC) ** -HALO_SLOPE


def _bar_m(xyz):
    """Bar-frame normalised radius of heliocentric points."""
    xyz = np.atleast_2d(xyz)
    xg, yg, zg = xyz[:, 0] - R0_PC, xyz[:, 1], xyz[:, 2] + Z_SUN_PC
    # bar major axis 27° from the Sun-centre line, near end at positive longitude (first quadrant)
    a = math.radians(BAR_ANGLE_DEG)
    xb = -xg * math.cos(a) + yg * math.sin(a)
    yb = xg * math.sin(a) + yg * math.cos(a)
    x0, y0, z0 = BAR_SCALES
    return np.sqrt((xb / x0) ** 2 + (yb / y0) ** 2 + (zg / z0) ** 2)


@lru_cache(maxsize=1)
def mean_system_mass() -> float:
    s, c = draw_systems(100000, seed=1)
    return float(np.sum(c["mass"]) / len(s["age_gyr"]))


@lru_cache(maxsize=1)
def _norms() -> tuple[float, float, float, float]:
    sun = np.array([[0.0, 0.0, 0.0]])
    R, _, z = galactocentric(sun)
    s_sun = float(arm_strength(sun)[0])
    thin_local = LOCAL_DENSITY * (1 - THICK_LOCAL - HALO_LOCAL)
    k_thin = thin_local / (_thin_raw(R, z)[0] * (1 + OLD_ARM_CONTRAST * (s_sun - 0.5) * 2))
    k_thick = LOCAL_DENSITY * THICK_LOCAL / _thick_raw(R, z)[0]
    k_halo = LOCAL_DENSITY * HALO_LOCAL / _halo_raw(R, z)[0]
    x0, y0, z0 = (v for v in BAR_SCALES)
    n_bulge = BULGE_MASS / mean_system_mass()
    k_bar = n_bulge / ((2 * math.pi) ** 1.5 * x0 * y0 * z0)
    return k_thin, k_thick, k_halo, k_bar


def densities(xyz_pc: np.ndarray) -> dict[str, np.ndarray]:
    """Number densities (systems pc^-3) of the components at heliocentric points."""
    R, beta, z = galactocentric(xyz_pc)
    k_thin, k_thick, k_halo, k_bar = _norms()
    s = arm_strength(xyz_pc)
    return {"thin": k_thin * _thin_raw(R, z) * (1 + OLD_ARM_CONTRAST * (s - 0.5) * 2),
            "thick": k_thick * _thick_raw(R, z),
            "halo": k_halo * _halo_raw(R, z),
            "bulge": k_bar * np.exp(-0.5 * _bar_m(xyz_pc) ** 2),
            "arm": s}


def total_density(xyz_pc) -> np.ndarray:
    d = densities(xyz_pc)
    return d["thin"] + d["thick"] + d["halo"] + d["bulge"]


# ------------------------------------------------------------------------------- populations

@dataclass
class _Fixed(PopulationModel):
    """Population with a fixed age/[Fe/H] distribution (bulge, halo)."""

    age_mu: float = 10.0
    age_sd: float = 1.5
    age_lim: tuple[float, float] = (6.0, 13.0)
    feh_mu: float = 0.0
    feh_sd: float = 0.35

    def ages_and_feh(self, rng, n):
        age = np.clip(rng.normal(self.age_mu, self.age_sd, n), *self.age_lim)
        return age, np.clip(rng.normal(self.feh_mu, self.feh_sd, n), -2.0, 0.5)

    def age_cdf(self, t):
        from scipy.special import ndtr
        lo, hi = self.age_lim
        t = np.asarray(t, float)
        return np.where(t <= lo, 0.0, np.where(t >= hi, 1.0, ndtr((t - self.age_mu) / self.age_sd)))

    def ages_and_feh_below(self, rng, tmax):
        from scipy.special import ndtri
        tmax = np.asarray(tmax, float)
        u = rng.random(len(tmax)) * self.age_cdf(tmax)
        age = np.clip(self.age_mu + self.age_sd * ndtri(np.clip(u, 1e-300, 1 - 1e-16)), *self.age_lim)
        return np.minimum(age, tmax), np.clip(rng.normal(self.feh_mu, self.feh_sd, len(tmax)), -2.0, 0.5)


@lru_cache(maxsize=1)
def _base() -> PopulationModel:
    return replace(default_model(), thin_disk_min_age=THIN_MIN_AGE)


FEH_GRADIENT = -0.06e-3  # dex per pc


def population(component: str, R_pc: float) -> PopulationModel:
    b = _base()
    if component == "thin":
        return replace(b, thick_disk_fraction=0.0,
                       feh_thin_intercept=b.feh_thin_intercept + FEH_GRADIENT * (R_pc - R0_PC))
    if component == "thick":
        return replace(b, thick_disk_fraction=1.0)
    kw = {k: getattr(b, k) for k in b.__dataclass_fields__}
    if component == "bulge":
        return _Fixed(**kw, age_mu=10.0, age_sd=1.5, age_lim=(5.0, 13.0), feh_mu=0.0, feh_sd=0.35)
    return _Fixed(**kw, age_mu=12.0, age_sd=0.6, age_lim=(11.0, 13.0), feh_mu=-1.5, feh_sd=0.5)


# ------------------------------------------------------------------------------- sectors

COMPONENTS = ("thin", "thick", "bulge", "halo")


def sector_of(xyz_pc) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.floor(np.asarray(xyz_pc, float) / SECTOR_PC))


@dataclass
class Sector:
    """Stars of one sector: system and component tables (as from draw_systems), positions,
    and the galactic component each system belongs to."""

    key: tuple[int, int, int]
    systems: dict
    components: dict
    xyz: np.ndarray            # n x 3 pc
    population: np.ndarray     # component name per system
    real: np.ndarray | None = None  # index into the real-star list, -1 for procedural
    tier: int = 0

    def __len__(self):
        return len(self.xyz)

    def sid(self, j: int) -> str:
        t = f"{self.tier}:" if self.tier else ""
        return f"G{t}{self.key[0]}.{self.key[1]}.{self.key[2]}-{j}"


def _concat(tables: list[dict], offsets: list[int] | None = None, key="system") -> dict:
    tables = [t for t in tables if t and len(next(iter(t.values())))]
    if not tables:
        return {}
    out = {k: np.concatenate([np.asarray(t[k]) for t in tables]) for k in tables[0]}
    return out


@lru_cache(maxsize=4096)
def sector(ix: int, iy: int, iz: int, procedural_only: bool = False) -> Sector:
    """The (procedural) stars of sector (ix, iy, iz), always the same."""
    seed = ids.mix(GALAXY_SEED, ix, iy, iz)
    rng = np.random.default_rng(seed)
    centre = (np.array([ix, iy, iz]) + 0.5) * SECTOR_PC
    d = densities(centre[None, :])
    R = float(galactocentric(centre[None, :])[0][0])
    vol = SECTOR_PC**3
    sys_t, comp_t, xyz, pop = [], [], [], []
    n_off = 0
    for k, comp in enumerate(COMPONENTS):
        lam = float(d[comp][0]) * vol * (1.0 - sum(tier_fraction(t, comp) for t in TIERS))
        n = int(rng.poisson(lam))
        if n == 0:
            continue
        s, c = draw_systems(n, seed=ids.mix(seed, k), model=population(comp, R), exclude_alive_above=TIER_MIN_MASS)
        keep = np.ones(n, bool)
        if comp == "thin":
            # young stars follow the arms
            s_here, s_sun = float(d["arm"][0]), float(arm_strength(np.zeros((1, 3)))[0])
            p_young = min((0.1 + s_here) / (0.1 + s_sun), YOUNG_BOOST_MAX)
            young = s["age_gyr"] < YOUNG_AGE_GYR
            u = rng.random(n)
            keep = ~young | (u < p_young)
            # (p > 1: duplicate is not possible; the boost above 1 is applied by extra draws)
            if p_young > 1:
                n_extra = int(rng.poisson((p_young - 1) * young.sum()))
                if n_extra:
                    s2, c2 = draw_systems(4 * n_extra + 8, seed=ids.mix(seed, k, 1), model=population(comp, R),
                                          exclude_alive_above=TIER_MIN_MASS)
                    yj = np.flatnonzero(s2["age_gyr"] < YOUNG_AGE_GYR)[:n_extra]
                    s, c = _append_young(s, c, s2, c2, yj)
                    keep = np.concatenate([keep, np.ones(len(yj), bool)])
                    n = len(keep)
        s, c = _take(s, c, np.flatnonzero(keep))
        c["system"] = c["system"] + n_off
        sys_t.append(s)
        comp_t.append(c)
        m = len(s["age_gyr"])
        xyz.append((np.array([ix, iy, iz]) + rng.random((m, 3))) * SECTOR_PC)
        pop += [comp] * m
        n_off += m
    if not sys_t:
        empty = draw_systems(1, seed=1)
        s, c = _take(*empty, np.zeros(0, int))
        return Sector((ix, iy, iz), s, c, np.zeros((0, 3)), np.array([], dtype=object))
    return Sector((ix, iy, iz), _concat(sys_t), _concat(comp_t), np.concatenate(xyz), np.array(pop, dtype=object))


def _take(s: dict, c: dict, rows: np.ndarray) -> tuple[dict, dict]:
    """Subset of systems (renumbered 0..len(rows)-1) with their components."""
    rows = np.asarray(rows, int)
    s2 = {k: np.asarray(v)[rows] for k, v in s.items()}
    new = np.full(len(s["age_gyr"]), -1)
    new[rows] = np.arange(len(rows))
    ck = new[np.asarray(c["system"])]
    sel = np.flatnonzero(ck >= 0)
    order = sel[np.lexsort((np.asarray(c["role"])[sel], ck[sel]))]
    c2 = {k: np.asarray(v)[order] for k, v in c.items()}
    c2["system"] = ck[order]
    return s2, c2


def _append_young(s, c, s2, c2, rows):
    a, b = _take(s2, c2, rows)
    b["system"] = b["system"] + len(s["age_gyr"])
    return _concat([s, a]), _concat([c, b])


def cell_size(tier: int) -> float:
    return SECTOR_PC if tier == 0 else TIERS[tier][2]


@lru_cache(maxsize=None)
def tier_fraction(tier: int, comp: str) -> float:
    """Fraction of a component's systems that belong to brightness tier ``tier``."""
    lo, hi, _ = TIERS[tier]
    return population(comp, R0_PC).alive_fraction(lo, hi)


@lru_cache(maxsize=1)
def _s_sun() -> float:
    return float(arm_strength(np.zeros((1, 3)))[0])


def cell_of(tier: int, xyz_pc) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.floor(np.asarray(xyz_pc, float) / cell_size(tier)))


@lru_cache(maxsize=2048)
def cell(tier: int, ix: int, iy: int, iz: int) -> Sector:
    """The systems of brightness tier ``tier`` in cell (ix, iy, iz) (tier 0: :func:`sector`)."""
    if tier == 0:
        return sector(ix, iy, iz)
    lo_m, hi_m, size = TIERS[tier]
    seed = ids.mix(GALAXY_SEED, _TIER_KEY + tier, ix, iy, iz)
    rng = np.random.default_rng(seed)
    sub = size / TIER_SUB
    g = (np.arange(TIER_SUB) + 0.5) * sub
    centres = np.array([ix, iy, iz]) * size + np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    d = densities(centres)
    R = float(galactocentric(((np.array([ix, iy, iz]) + 0.5) * size)[None, :])[0][0])
    sys_t, comp_t, xyz_t, pop = [], [], [], []
    for k, comp in enumerate(COMPONENTS):
        f = tier_fraction(tier, comp)
        if f <= 0:
            continue
        boost = YOUNG_BOOST_MAX if comp == "thin" else 1.0  # thinned below
        counts = rng.poisson(d[comp] * f * sub**3 * boost)
        n = int(counts.sum())
        if n == 0:
            continue
        xyz = centres[np.repeat(np.arange(len(centres)), counts)] + (rng.random((n, 3)) - 0.5) * sub
        s, c = draw_systems(n, seed=ids.mix(seed, k), model=population(comp, R), alive_range=(lo_m, hi_m))
        if comp == "thin":
            # young stars follow the arms (as in the sectors), old ones do not
            p_young = np.minimum((0.1 + arm_strength(xyz)) / (0.1 + _s_sun()), YOUNG_BOOST_MAX)
            keep = rng.random(n) < np.where(s["age_gyr"] < YOUNG_AGE_GYR, p_young, 1.0) / boost
            s, c = _take(s, c, np.flatnonzero(keep))
            xyz = xyz[keep]
        m = len(s["age_gyr"])
        if m == 0:
            continue
        c["system"] = c["system"] + sum(len(t["age_gyr"]) for t in sys_t)
        sys_t.append(s)
        comp_t.append(c)
        xyz_t.append(xyz)
        pop += [comp] * m
    if not sys_t:
        empty = draw_systems(1, seed=1)
        s, c = _take(*empty, np.zeros(0, int))
        return Sector((ix, iy, iz), s, c, np.zeros((0, 3)), np.array([], dtype=object), tier=tier)
    return Sector((ix, iy, iz), _concat(sys_t), _concat(comp_t), np.concatenate(xyz_t), np.array(pop, dtype=object),
                  tier=tier)


def tier_totals() -> dict[int, float]:
    """Expected number of systems of each brightness tier in the galaxy (tier 0: the rest)."""
    tot = galaxy_totals()
    out = {t: sum(tot[c] * tier_fraction(t, c) for c in COMPONENTS) for t in TIERS}
    out[0] = sum(tot.values()) - sum(out.values())
    return out


def cells_within(tier: int, centre_pc, radius_pc) -> list[tuple[int, int, int]]:
    """Cells of a tier that touch the sphere (centre, radius)."""
    size = cell_size(tier)
    c = np.asarray(centre_pc, float)
    lo = np.floor((c - radius_pc) / size).astype(int)
    hi = np.floor((c + radius_pc) / size).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + 1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    cen = (g + 0.5) * size
    near = np.linalg.norm(cen - c, axis=1) <= radius_pc + size * 0.87
    return [tuple(map(int, k)) for k in g[near]]


def sectors_within(centre_pc, radius_pc) -> list[tuple[int, int, int]]:
    return cells_within(0, centre_pc, radius_pc)


def system(sid: str, *, moons: bool = True):
    """The complete system with a galaxy ID ``G<ix>.<iy>.<iz>-<j>`` or ``G<t>:<ix>.<iy>.<iz>-<j>``."""
    tier, key, j = parse(sid)
    sec = cell(tier, *key)
    s, c = ids.subset(sec.systems, sec.components, j)
    seed = ids.mix(GALAXY_SEED, *key, j) if tier == 0 else ids.mix(GALAXY_SEED, _TIER_KEY + tier, *key, j)
    b = ids.complete(s, c, seed, moons=moons, meta={"ids": [sid]})
    b.meta["position_pc"] = [round(float(v), 3) for v in sec.xyz[j]]
    b.meta["population"] = str(sec.population[j])
    return b


def parse(sid: str) -> tuple[int, tuple[int, int, int], int]:
    """(tier, cell key, index) of a galaxy ID."""
    import re
    m = re.match(r"^G(?:(\d):)?(-?\d+)\.(-?\d+)\.(-?\d+)-(\d+)$", sid.strip())
    if not m or (m.group(1) and int(m.group(1)) not in TIERS):
        raise ValueError(f"not a galaxy ID: {sid!r} (expected e.g. G0.1.-2-5 or G2:0.0.0-5)")
    return (int(m.group(1) or 0), (int(m.group(2)), int(m.group(3)), int(m.group(4))), int(m.group(5)))


def random_system(seed: int, tries: int = 50) -> str:
    """ID of a pseudo-random procedural system, each system of the galaxy (about) equally
    likely: a point drawn from the galaxy's density (:func:`sample_cloud`), a brightness tier
    with the probabilities of the systems there, then a system drawn uniformly among the shown
    procedural systems of that tier's cell around the point (redrawn if there are none).
    The same ``seed`` always gives the same system."""
    from . import realstars as rs
    for t in range(tries):
        pt = sample_cloud(1, seed=ids.mix(seed, t))[0][0]
        rng = np.random.default_rng(ids.mix(seed, t, 1))
        d = densities(pt[None, :])
        tot = sum(float(d[c][0]) for c in COMPONENTS)
        p = [sum(float(d[c][0]) * tier_fraction(k, c) for c in COMPONENTS) / tot for k in TIERS]
        u = rng.random()
        tier = next((k for k, cp in zip(TIERS, np.cumsum(p)) if u < cp), 0)
        key = cell_of(tier, pt)
        kept = rs.procedural_kept(key, tier)
        if len(kept):
            return cell(tier, *key).sid(int(kept[rng.integers(len(kept))]))
    raise RuntimeError("no procedural system found")


# ------------------------------------------------------------------------------- far view

def sample_cloud(n: int = 200000, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` points (heliocentric pc) distributed like the galaxy's stars, for the far view,
    with their component index (0 thin, 1 thick, 2 bulge, 3 halo) and arm strength.

    Each component is sampled from its own analytic profile and then thinned by rejection
    against the full model (arms, hole, edge), so relative numbers match :func:`densities`.
    """
    rng = np.random.default_rng(seed)
    k_thin, k_thick, k_halo, k_bar = _norms()
    # total numbers of each component (numerical integration on a coarse grid)
    tot = galaxy_totals()
    frac = np.array([tot[c] for c in COMPONENTS])
    frac = frac / frac.sum()
    counts = rng.multinomial(n, frac)
    pts, comp, arm = [], [], []
    for ci, (cname, m) in enumerate(zip(COMPONENTS, counts)):
        got = 0
        while got < m:
            k = int((m - got) * 2.5) + 100
            if cname in ("thin", "thick"):
                L, H = (THIN_L, THIN_H) if cname == "thin" else (THICK_L, THICK_H)
                R = rng.gamma(2.0, L, k)
                beta = rng.uniform(-np.pi, np.pi, k)
                z = rng.laplace(0, H, k)
                xyz = from_galactocentric(R, beta, z)
                acc = np.ones(k)
                edge = 1.0 / (1.0 + np.exp((R - DISC_EDGE) / DISC_EDGE_W))
                acc *= edge
                if cname == "thin":
                    acc *= 1.0 - np.exp(-((R / DISC_HOLE) ** 2))
                    s = arm_strength(xyz)
                    young = rng.random(k) < 0.05
                    acc *= np.where(young, (0.1 + s) / 1.1, (1 + OLD_ARM_CONTRAST * (s - 0.5) * 2) / 1.15)
                else:
                    s = np.zeros(k)
            elif cname == "bulge":
                x0, y0, z0 = BAR_SCALES
                xb, yb, zb = rng.normal(0, x0, k), rng.normal(0, y0, k), rng.normal(0, z0, k)
                a = math.radians(BAR_ANGLE_DEG)
                xg = -xb * math.cos(a) + yb * math.sin(a)
                yg = xb * math.sin(a) + yb * math.cos(a)
                xyz = np.stack([xg + R0_PC, yg, zb - Z_SUN_PC], 1)
                acc, s = np.ones(k), np.zeros(k)
            else:
                # r^-2.5 between 1 and 60 kpc: p(r) dr ∝ r^-0.5
                u = rng.random(k)
                r = (1000.0**0.5 + u * (60000.0**0.5 - 1000.0**0.5)) ** 2
                v = rng.normal(size=(k, 3))
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                v[:, 2] *= HALO_Q
                xyz = v * r[:, None] + np.array([R0_PC, 0, -Z_SUN_PC])
                acc, s = np.ones(k), np.zeros(k)
            keep = rng.random(k) < acc
            take = np.flatnonzero(keep)[: m - got]
            pts.append(xyz[take])
            comp.append(np.full(len(take), ci))
            arm.append(s[take])
            got += len(take)
    return np.concatenate(pts), np.concatenate(comp), np.concatenate(arm)


@lru_cache(maxsize=1)
def galaxy_totals() -> dict[str, float]:
    """Total number of systems per component (thin/thick: R-azimuth grid times the analytic
    vertical integral; halo: R-z grid; bulge: analytic)."""
    k_thin, k_thick, k_halo, k_bar = _norms()
    R = np.linspace(25, 25000, 500)
    beta = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    dR, db = R[1] - R[0], beta[1] - beta[0]
    RR, BB = np.meshgrid(R, beta, indexing="ij")
    xyz = from_galactocentric(RR.ravel(), BB.ravel(), np.full(RR.size, Z_SUN_PC))
    s = arm_strength(xyz)
    zero = np.zeros(RR.size)
    area = RR.ravel() * dR * db
    thin = k_thin * np.sum(_thin_raw(RR.ravel(), zero) * (1 + OLD_ARM_CONTRAST * (s - 0.5) * 2) * area) * 2 * THIN_H
    thick = k_thick * np.sum(_thick_raw(RR.ravel(), zero) * area) * 2 * THICK_H
    Rh = np.linspace(25, 60000, 1200)
    zh = np.linspace(-60000, 60000, 2401)
    Rg, Zg = np.meshgrid(Rh, zh, indexing="ij")
    r = np.sqrt(Rg**2 + (Zg / HALO_Q) ** 2)
    halo = k_halo * np.sum(_halo_raw(Rg, Zg) * (r < 60000) * 2 * np.pi * Rg) * (Rh[1] - Rh[0]) * (zh[1] - zh[0])
    return {"thin": float(thin), "thick": float(thick), "bulge": BULGE_MASS / mean_system_mass(),
            "halo": float(halo)}


__all__ = ["ARMS", "GALAXY_SEED", "R0_PC", "SECTOR_PC", "Sector", "TIERS", "arm_lines", "arm_strength", "cell",
           "cell_of", "cell_size", "cells_within", "densities", "tier_fraction", "tier_totals",
           "galactocentric", "galaxy_totals", "parse", "random_system", "sample_cloud", "sector", "sector_of", "sectors_within",
           "system", "total_density"]
