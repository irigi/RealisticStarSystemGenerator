"""Planetary systems of a real star, conditioned on its known planets and on the surveys that
did not find more.

Target: draws from p(system | star, known planets, non-detections) ∝ prior(system | star) x
L(known planets, non-detections | system), where the prior is the generator itself run on the
star's posterior realisations (:mod:`starsys.real.star`). Importance sampling:

1. Prior systems are drawn in batches (``attach_planets`` on the stellar realisations).
2. Each prior system is weighted by
   - a kernel per known planet, exp(-chi^2 / 2) in log period (log semi-major axis where the
     period is unknown) and the measured log radius, log mass or log M sin i, maximised over the
     assignments of model planets to known planets (Hungarian algorithm on the target star's
     planets);
   - a known planet without a model counterpart is inserted at a penalty exp(-``insert_chi2`` / 2),
     like a match 5 sigma off; known planets the prior cannot make (or makes too rarely for the
     sample) are thus still present;
   - transiting known planets: the system is viewed from a direction drawn uniformly within the
     transit band of a matched planet, weighted by the band's probability R*/a and by the
     fraction of such views in which the other matched transiting planets transit too. The
     sky orientation is otherwise unconstrained, so this is exact and makes co-transiting
     neighbours come out right;
   - known planets beyond the host's dynamical (Holman-Wiegert) limit in that realisation's
     binary orbit: weight 0 (the known planet constrains the companion's orbit);
   - every other planet of the target star: x (1 - p_det) of the surveys (``surveys``,
     :mod:`starsys.real.surveys`; none by default).
3. Each batch is resampled in proportion to its weights; batches are combined in proportion to
   their total weight. Batches are added until the effective sample size reaches
   ``target_ess`` or ``max_prior`` systems were drawn. If the effective sample size is still
   short, the matched planets' kernels are widened by the smallest factor that reaches it (at
   most 2.5; an ABC-style tolerance, reported as ``kernel_widening``), and then, for systems with
   many known planets, the insertion penalty is softened (``insertion_softening``, at most 4).
4. In the chosen systems the matched planets take the measured values (period, radius, mass, or
   M sin i / sin i at the system's orientation; eccentricity where measured), inserted ones are
   placed in the system's plane, and unmatched model planets closer than ``hill_spacing``
   mutual Hill radii to a known planet (or on crossing orbits) are removed. Derived properties
   are recomputed and moons attached.

Known planets are flagged ``FLAG_KNOWN`` (``FLAG_KNOWN_INSERTED`` if inserted) and carry their
names in ``known_name``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import binaries as bn
from .. import derived as dv
from .. import moons as mn
from .. import planets as pt
from ..flags import FLAG_HZ_EXTRAP
from ..units import EARTH_MASS_MSUN, a_au_to_period, period_to_a_au
from . import star as st
from .catalog import SystemRecord, find
from .known import IMAGING, KnownPlanet, known_planets

FLAG_KNOWN = 32768  # a known exoplanet: parameters set to the measured values
FLAG_KNOWN_INSERTED = 65536  # a known exoplanet the drawn prior system had no counterpart for
KNOWN_CLUSTER_ID = -6
RSUN_AU = 0.00465047
EARTH_RADIUS_AU = 4.26352e-5


@dataclass(frozen=True)
class ConditionModel:
    sigma_logp: float = 0.08  # dex: which model planet plays a known one (values are then snapped)
    sigma_logr: float = 0.08  # dex, added to the measurement error
    sigma_logm: float = 0.15  # dex, for masses and minimum masses
    sigma_loga: float = 0.15  # dex, when only the semi-major axis is known (imaging)
    # Penalty of a known planet without a model counterpart: exp(-insert_chi2 / 2) ~ 4e-6, a 5-sigma
    # mismatch. It acts as the prior probability that the model cannot make the planet: planets the
    # model makes at rates above ~1e-5 come from natural matches.
    insert_chi2: float = 25.0
    n_views: int = 64  # viewing directions tried per system with transiting known planets
    hill_spacing: float = 2.0 * np.sqrt(3.0)  # minimum separation of a model planet from a known one
    n_prior: int = 10000  # prior systems per batch
    max_prior: int = 80000
    target_ess: float = 200.0


@dataclass
class ConditionedSystems:
    record: SystemRecord
    known: list[KnownPlanet]
    systems: dict
    components: dict
    planets: dict
    moons: dict
    target: np.ndarray  # component row of the real star in each system
    ess: float
    n_prior: int
    inserted_fraction: dict[str, float] = field(default_factory=dict)
    kernel_widening: float = 1.0  # lambda of :func:`tempering`
    insertion_softening: float = 1.0  # mu of :func:`tempering`
    n_unique: int = 0  # distinct prior systems among the output
    surveys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.target)
        pl = self.planets
        on = pl["component"] == self.target[pl["system"]]
        lines = [f"{self.record.name}: {n} systems conditioned on {len(self.known)} known planet(s); "
                 f"{self.n_prior} prior systems, effective sample size {self.ess:.0f}, "
                 f"{self.n_unique} distinct, kernels widened x{self.kernel_widening:.2f}"
                 + (f", insertion penalty softened x{self.insertion_softening:.2f}" if self.insertion_softening > 1
                    else "")]
        for sv in self.surveys:
            lines.append(f"  survey {sv}")
        for k in self.known:
            lines.append(f"  known {k.name} ({k.method}): inserted in {self.inserted_fraction.get(k.name, 0):.0%}")
        n_on = np.bincount(pl["system"][on], minlength=n)
        lines.append(f"  planets around {self.record.star.name}: {n_on.mean():.2f} per system "
                     f"(known {len(self.known)}, unseen {n_on.mean() - len(self.known):.2f})")
        unseen = on & ((pl["flags"] & FLAG_KNOWN) == 0)
        if unseen.any():
            p = pl["period"][unseen]
            m = pl["planet_mass"][unseen]
            for lo, hi, label in ((0, 10, "< 10 d"), (10, 100, "10-100 d"), (100, 1000, "100-1000 d"),
                                  (1000, np.inf, "> 1000 d")):
                sel = (p >= lo) & (p < hi)
                lines.append(f"    unseen {label}: {sel.sum() / n:.2f} per system "
                             f"(of which > 0.3 MJup: {(sel & (m > 95)).sum() / n:.2f})")
        return "\n".join(lines)


# --- geometry -----------------------------------------------------------------------------

def normals_from_sky(incl, node) -> np.ndarray:
    """Orbit normals (sky normal +z) from sky inclination and node, as in :mod:`starsys.binaries`."""
    s = np.sin(incl)
    return np.stack([s * np.sin(node), -s * np.cos(node), np.cos(incl)], axis=-1)


def _frame(v) -> np.ndarray:
    """Rows x', y', z' = v of a right-handed frame with z' along ``v``."""
    e = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = e - (e @ v) * v
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(v, x), v])


def sky_angles(normals, v):
    """(incl_sky, node_sky) of ``normals`` seen along ``v``."""
    f = _frame(v)
    n = normals @ f.T
    return np.arccos(np.clip(n[:, 2], -1.0, 1.0)), np.arctan2(n[:, 0], -n[:, 1]) % (2 * np.pi)


def _band_views(n_axis, rho, k, rng) -> np.ndarray:
    """``k`` viewing directions uniform on the sphere within |n_axis . v| < rho."""
    c = rng.uniform(-rho, rho, k)
    phi = 2 * np.pi * rng.random(k)
    f = _frame(n_axis)
    s = np.sqrt(1 - c**2)
    return (c[:, None] * f[2] + s[:, None] * (np.cos(phi)[:, None] * f[0] + np.sin(phi)[:, None] * f[1]))


# --- kernels ------------------------------------------------------------------------------

def _known_arrays(known: list[KnownPlanet], model: ConditionModel) -> dict:
    lp = np.log10([k.period for k in known]) if known else np.zeros(0)
    la = np.log10([k.a_au for k in known]) if known else np.zeros(0)
    use_p = np.isfinite(lp)
    use_a = ~use_p & np.isfinite(la)

    def rel(v, e):
        v, e = np.asarray(v, dtype=float), np.asarray(e, dtype=float)
        return np.where(np.isfinite(e) & (v > 0), e / np.maximum(v, 1e-300) / np.log(10), 0.0)

    r = np.array([k.radius for k in known], dtype=float)
    m = np.array([k.mass for k in known], dtype=float)
    ms = np.array([k.msini for k in known], dtype=float)
    return {
        "logp": lp, "loga": la, "use_p": use_p, "use_a": use_a,
        "logr": np.log10(r), "sr": np.hypot(model.sigma_logr, rel(r, [k.radius_err for k in known])),
        "logm": np.log10(m), "sm": np.hypot(model.sigma_logm, rel(m, [k.mass_err for k in known])),
        "logms": np.log10(ms), "sms": np.hypot(model.sigma_logm, rel(ms, [k.msini_err for k in known])),
        "transit": np.array([k.transits for k in known], dtype=bool),
        "cb": np.array([k.circumbinary for k in known], dtype=bool),
    }


def _chi2(ka, idx, logp, loga, logr, logm, sin_i, cb, model: ConditionModel) -> np.ndarray:
    """chi^2 of known planets ``idx`` (rows) against candidate planets (columns)."""
    out = np.zeros((len(idx), len(logp)))
    for row, k in enumerate(idx):
        c = np.zeros(len(logp))
        if ka["use_p"][k]:
            c += ((logp - ka["logp"][k]) / model.sigma_logp) ** 2
        elif ka["use_a"][k]:
            c += ((loga - ka["loga"][k]) / model.sigma_loga) ** 2
        if np.isfinite(ka["logr"][k]):
            c += ((logr - ka["logr"][k]) / ka["sr"][k]) ** 2
        if np.isfinite(ka["logm"][k]):
            c += ((logm - ka["logm"][k]) / ka["sm"][k]) ** 2
        elif np.isfinite(ka["logms"][k]) and sin_i is not None:
            c += ((logm + np.log10(np.maximum(sin_i, 1e-3)) - ka["logms"][k]) / ka["sms"][k]) ** 2
        c = np.where(cb != ka["cb"][k], np.inf, c)  # circumbinary known planets only match P-type planets
        out[row] = c
    return out


def _assign(chi2, insert_chi2) -> tuple[np.ndarray, np.ndarray]:
    """Best assignment of rows (known planets) to columns (candidates) or to insertion.
    Returns (column per row or -1, chi2 per row)."""
    n_k, n_c = chi2.shape
    if n_k == 0:
        return np.zeros(0, int), np.zeros(0)
    big = 1e6
    cost = np.full((n_k, n_c + n_k), big)
    cost[:, :n_c] = np.where(np.isfinite(chi2), np.minimum(chi2, big), big)
    cost[np.arange(n_k), n_c + np.arange(n_k)] = insert_chi2
    r, c = linear_sum_assignment(cost)
    col = np.full(n_k, -1)
    val = np.full(n_k, insert_chi2)
    for i, j in zip(r, c):
        if j < n_c and cost[i, j] < insert_chi2:
            col[i], val[i] = j, cost[i, j]
    return col, val


# --- one batch ----------------------------------------------------------------------------

@dataclass
class _Choice:
    col: np.ndarray  # planet row (batch table) per known planet, -1 = inserted
    view: np.ndarray | None  # viewing direction (old sky frame), None = unchanged


def _weigh_batch(pl, components, target, known, ka, surveys, model: ConditionModel, rng):
    """Log weights and assignment choices of the systems of one batch."""
    n_sys = len(target)
    n_known = len(known)
    logw = np.zeros(n_sys)  # geometry and non-detections
    chi2 = np.zeros(n_sys)  # kernels of the matched known planets
    n_ins = np.zeros(n_sys, dtype=int)  # inserted known planets
    choices: list[_Choice] = [None] * n_sys  # type: ignore[list-item]
    comp_mass = np.asarray(components["mass"], dtype=float)
    comp_rad = np.asarray(components["radius"], dtype=float)
    a_max = bn.structure(components)["a_max"][target]
    k_period = np.array([k.period for k in known], dtype=float)
    k_a = np.array([k.a_au for k in known], dtype=float)
    k_rad = np.array([k.radius if np.isfinite(k.radius) else (k.radius_mr if np.isfinite(k.radius_mr) else 1.0)
                      for k in known], dtype=float)

    tgt_of_sys = target
    on = pl["component"] == tgt_of_sys[pl["system"]]
    rows = np.flatnonzero(on)
    sys_of = pl["system"][rows]
    order = np.argsort(sys_of, kind="stable")
    rows, sys_of = rows[order], sys_of[order]
    starts = np.searchsorted(sys_of, np.arange(n_sys + 1))
    logp = np.log10(pl["period"][rows])
    loga = np.log10(pl["a_au"][rows])
    logr = np.log10(pl["planet_radius"][rows])
    logm = np.log10(pl["planet_mass"][rows])
    cb = pl["host_kind"][rows] == "P"
    nrm = normals_from_sky(pl["incl_sky"][rows], np.nan_to_num(pl["node_sky"][rows]))
    r_star_au = comp_rad[pl["component"][rows]] * RSUN_AU
    rho = (r_star_au + pl["planet_radius"][rows] * EARTH_RADIUS_AU) / pl["a_au"][rows]
    t_idx = np.flatnonzero(ka["transit"])
    o_idx = np.flatnonzero(~ka["transit"])
    new_cos = np.abs(nrm[:, 2]).copy()  # |cos i| in the chosen view, filled per system
    matched = np.zeros(len(rows), dtype=bool)

    for s in range(n_sys):
        a, b = starts[s], starts[s + 1]
        sl = slice(a, b)
        col = np.full(n_known, -1)
        lw = 0.0
        # Known planets beyond this realisation's dynamical limit: incompatible.
        m_host = comp_mass[tgt_of_sys[s]]
        ka_s = np.where(np.isfinite(k_period), period_to_a_au(np.nan_to_num(k_period, nan=1.0), m_host), k_a)
        if n_known and np.any(ka_s[~ka["cb"]] > a_max[s] * (1 + 1e-9)):
            logw[s] = -np.inf
            chi2[s] = np.inf
            n_ins[s] = n_known
            choices[s] = _Choice(col, None)
            continue
        # Transiting known planets (orientation-free kernels).
        if len(t_idx):
            c2 = _chi2(ka, t_idx, logp[sl], loga[sl], logr[sl], logm[sl], None, cb[sl], model)
            cc, val = _assign(c2, model.insert_chi2)
            col[t_idx] = np.where(cc >= 0, cc + a, -1)
            t_val = dict(zip(t_idx, val))
            chi2[s] += val[cc >= 0].sum()
        view = None
        mt = [r for r in col[t_idx] if r >= 0]
        if mt:
            # View within the transit band of the matched planet with the narrowest band.
            j = min(mt, key=lambda r: rho[r])
            views = _band_views(nrm[j], rho[j], model.n_views, rng)
            others = [r for r in mt if r != j]
            if others:
                ok = np.abs(views @ nrm[others].T) < rho[others][None, :]
                n_ok = ok.sum(axis=1)
                best = n_ok.max()
                good = np.flatnonzero(n_ok == best)
                frac = len(good) / model.n_views
                v = views[rng.choice(good)]
                bad = [others[i] for i in np.flatnonzero(~ok[good[0]])] if best < len(others) else []
                for r in bad:  # cannot transit together with the others: treated as inserted
                    kk = np.flatnonzero(col == r)[0]
                    col[kk] = -1
                    chi2[s] -= t_val[kk]
            else:
                frac = 1.0
                v = views[0]
            lw += np.log(rho[j] * frac)
            view = v
        # Inserted transiting known planets are placed in the plane of the view's transiting planets
        # (they transit with them); without a matched one, the view is random and the first pays
        # its transit probability.
        ins_t = [kk for kk in t_idx if col[kk] < 0]
        if ins_t and view is None:
            r_star = comp_rad[tgt_of_sys[s]] * RSUN_AU
            kk = max(ins_t, key=lambda q: ka_s[q])
            lw += np.log(min(1.0, (r_star + k_rad[kk] * EARTH_RADIUS_AU) / ka_s[kk]))
        # |cos i| of every candidate in this view.
        if view is not None and b > a:
            new_cos[sl] = np.abs(nrm[sl] @ view)
        # Other known planets (M sin i depends on the view).
        if len(o_idx):
            free = np.setdiff1d(np.arange(a, b), col[col >= 0])
            sin_i = np.sqrt(1 - new_cos[free] ** 2)
            c2 = _chi2(ka, o_idx, logp[free], loga[free], logr[free], logm[free], sin_i, cb[free], model)
            cc, val = _assign(c2, model.insert_chi2)
            col[o_idx] = np.where(cc >= 0, free[np.maximum(cc, 0)] if len(free) else -1, -1)
            chi2[s] += val[cc >= 0].sum()
        matched[col[col >= 0]] = True
        n_ins[s] = (col < 0).sum()
        logw[s] = lw
        choices[s] = _Choice(col, view)

    # Non-detections: every unmatched planet of the target star.
    if surveys:
        cand = {key: pl[key][rows] for key in ("period", "a_au", "planet_mass", "planet_radius", "ecc", "host_mass")}
        cand["cos_i"] = new_cos
        cand["r_star"] = comp_rad[pl["component"][rows]]
        cand["age_gyr"] = np.asarray(components["age_gyr"], dtype=float)[pl["component"][rows]]
        cand["system"] = sys_of
        p_det = np.zeros(len(rows))
        for sv in surveys:
            p = np.clip(sv.p_detect(cand), 0.0, 1.0)
            p_det = 1 - (1 - p_det) * (1 - p)
        miss = np.where(matched, 0.0, np.log(np.maximum(1 - p_det, 1e-300)))
        logw += np.bincount(sys_of, weights=miss, minlength=n_sys)
    # Map candidate positions to planet-table rows.
    for s in range(n_sys):
        c = choices[s].col
        choices[s].col = np.where(c >= 0, rows[np.maximum(c, 0)], -1)
    return logw, chi2, n_ins, choices


def tempering(chi2, logw, n_ins, insert_chi2, target_ess, max_lambda=2.5, max_mu=4.0):
    """Weights exp(logw - chi2 / 2 lambda^2 - n_ins insert_chi2 / 2 mu^2), with the smallest widening
    that gives an effective sample size of ``target_ess``: first the matched planets' kernels
    (lambda up to ``max_lambda``), then, if that is not enough (systems with many known planets),
    the insertion penalty too (mu up to ``max_mu``). Widening the kernels alone does not favour
    inserted planets over natural matches. Returns (lambda, mu, normalised weights, ESS)."""
    def weights(lam, mu):
        lw = logw - 0.5 * chi2 / lam**2 - 0.5 * insert_chi2 * n_ins / mu**2
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        return w, 1.0 / np.sum(w**2)

    def search(f, lo, hi):
        """Smallest x in [lo, hi] with ESS(f(x)) >= target (hi if none)."""
        e_lo = weights(*f(lo))[1]
        if e_lo >= target_ess:
            return lo
        e_hi = weights(*f(hi))[1]
        if e_hi < target_ess:
            return hi if e_hi > 1.01 * e_lo else lo  # widen only where it helps
        for _ in range(25):
            mid = np.sqrt(lo * hi)
            if weights(*f(mid))[1] >= target_ess:
                hi = mid
            else:
                lo = mid
        return hi

    lam = search(lambda x: (x, 1.0), 1.0, max_lambda)
    mu = 1.0
    if weights(lam, 1.0)[1] < target_ess:
        mu = search(lambda x: (lam, x), 1.0, max_mu)
    w, ess = weights(lam, mu)
    return lam, mu, w, ess


def _ess(logw) -> float:
    w = np.exp(logw - np.max(logw))
    w /= w.sum()
    return float(1.0 / np.sum(w**2))


# --- building the chosen systems ------------------------------------------------------------

def _subset_stars(fit: st.StarSystemFit, idx) -> tuple[dict, dict, np.ndarray]:
    """Systems ``idx`` of a stellar fit as a new (systems, components, target) set."""
    n_comp = len(fit.record.components)
    rows = (np.asarray(idx)[:, None] * n_comp + np.arange(n_comp)[None, :]).ravel()
    systems = {k: np.asarray(v)[idx] for k, v in fit.systems.items()}
    components = {k: np.asarray(v)[rows] for k, v in fit.components.items()}
    components["system"] = np.repeat(np.arange(len(idx)), n_comp)
    target = np.arange(len(idx)) * n_comp + fit.record.target
    return systems, components, target


def _radius_from_mass(m) -> float:
    """Chen & Kipping (2017) mean mass-radius relation (Earth units)."""
    if m < 2.04:
        return 1.008 * m**0.279
    if m < 132.0:
        return 1.23 * (m / 2.04) ** 0.589
    return 14.3 * (m / 132.0) ** -0.044


def _mass_from_radius(r) -> float:
    """Inverse of :func:`_radius_from_mass` below the Jupiter plateau (Jupiter mass on it)."""
    if r < 1.23:
        return (r / 1.008) ** (1 / 0.279)
    if r < 14.3:
        return 2.04 * (r / 1.23) ** (1 / 0.589)
    return 318.0


def _build_one(pl, rows_sys, comps, choice: _Choice, known, target_row, offset, model: ConditionModel, rng):
    """Planet rows of one chosen system (dict of arrays) with the known planets in place.
    ``comps``: the system's component rows (``offset``: row of its first component in the batch);
    ``target_row``: the real star's row in the batch."""
    p = {k: np.array(v[rows_sys]) for k, v in pl.items()}
    n = len(rows_sys)
    p.setdefault("known_name", np.full(n, "", dtype=object))
    p["known_name"] = np.full(n, "", dtype=object)
    pos = {r: i for i, r in enumerate(rows_sys)}
    # Orientation.
    if choice.view is not None:
        ok = np.isfinite(p["node_sky"])
        nrm = normals_from_sky(p["incl_sky"][ok], p["node_sky"][ok])
        p["incl_sky"][ok], p["node_sky"][ok] = sky_angles(nrm, choice.view)
        c_ok = np.isfinite(comps["incl_sky"])
        if c_ok.any():
            cn = normals_from_sky(comps["incl_sky"][c_ok], comps["node"][c_ok])
            comps["incl_sky"][c_ok], comps["node"][c_ok] = sky_angles(cn, choice.view)
    host_mass = float(comps["mass"][target_row - offset])
    host_mask = (p["component"] == target_row) & (p["host_kind"] == "S")
    # Plane of the target's planets (for inserted ones).
    if host_mask.any():
        nn = normals_from_sky(p["incl_sky"][host_mask], np.nan_to_num(p["node_sky"][host_mask]))
        w = p["planet_mass"][host_mask][:, None] * np.sqrt(p["a_au"][host_mask])[:, None]
        plane = (nn * w).sum(axis=0)
        plane /= max(np.linalg.norm(plane), 1e-300)
    else:
        v = rng.normal(size=3)
        plane = v / np.linalg.norm(v)
    add = []
    keep_known = np.zeros(n, dtype=bool)
    for kk, k in enumerate(known):
        r = choice.col[kk]
        if r >= 0:
            i = pos[r]
            _snap(p, i, k, host_mass)
            keep_known[i] = True
        else:
            add.append(_inserted(p, k, host_mass, target_row, plane, rng))
    if add:
        p = {key: np.concatenate([p[key]] + [a[key] for a in add]) for key in p}
        keep_known = np.concatenate([keep_known, np.ones(len(add), dtype=bool)])
    # Remove model planets too close to a known one.
    hm = (p["component"] == target_row) & (p["host_kind"] == "S")
    drop = np.zeros(len(keep_known), dtype=bool)
    for i in np.flatnonzero(keep_known & hm):
        for j in np.flatnonzero(hm & ~keep_known & ~drop):
            a1, a2 = sorted([(p["a_au"][i], p["ecc"][i]), (p["a_au"][j], p["ecc"][j])])
            rh = ((p["planet_mass"][i] + p["planet_mass"][j]) * EARTH_MASS_MSUN / (3 * host_mass)) ** (1 / 3) \
                * 0.5 * (a1[0] + a2[0])
            if a2[0] - a1[0] < model.hill_spacing * rh or a1[0] * (1 + a1[1]) > a2[0] * (1 - a2[1]):
                drop[j] = True
    p = {key: v[~drop] for key, v in p.items()}
    return p


def _snap(p, i, k: KnownPlanet, host_mass):
    if np.isfinite(k.period):
        p["period"][i] = k.period
        p["a_au"][i] = period_to_a_au(k.period, host_mass)
    elif np.isfinite(k.a_au):
        p["a_au"][i] = k.a_au
        p["period"][i] = a_au_to_period(k.a_au, host_mass)
    m_old = p["planet_mass"][i]
    sin_i = max(np.sin(p["incl_sky"][i]), 0.05)
    if np.isfinite(k.mass):
        m_new = k.mass
    elif np.isfinite(k.msini):
        m_new = k.msini / sin_i
    else:
        m_new = m_old
    if np.isfinite(k.radius):
        r_new = k.radius
    elif m_new != m_old:
        r_new = p["planet_radius"][i] * _radius_from_mass(m_new) / _radius_from_mass(m_old)
    else:
        r_new = p["planet_radius"][i]
    if not np.isfinite(k.mass) and not np.isfinite(k.msini) and np.isfinite(k.radius) and m_old > 0:
        m_new = m_old * _mass_from_radius(k.radius) / _mass_from_radius(p["planet_radius"][i])
    p["mass_init"][i] *= m_new / m_old
    p["radius_init"][i] *= r_new / max(p["planet_radius"][i], 1e-9)
    p["planet_mass"][i], p["planet_radius"][i] = m_new, r_new
    if np.isfinite(k.ecc):
        p["ecc"][i] = k.ecc
    p["flags"][i] |= FLAG_KNOWN
    p["known_name"][i] = k.name


def _inserted(p, k: KnownPlanet, host_mass, target_row, plane, rng) -> dict:
    """One row (dict of length-1 arrays) for a known planet without a model counterpart."""
    row = {key: v[:0].copy() for key, v in p.items()}
    period = k.period if np.isfinite(k.period) else a_au_to_period(k.a_au, host_mass)
    a = period_to_a_au(period, host_mass)
    # Orientation: in the target's plane; transiting ones edge-on (sky normal +z).
    if k.transits:
        u = plane - plane[2] * np.array([0.0, 0.0, 1.0])
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0.0, 0.0])
        inc = np.radians(k.incl_deg) if np.isfinite(k.incl_deg) else np.pi / 2
        nrm = np.sin(inc) * u + np.cos(inc) * np.array([0.0, 0.0, 1.0])
    else:
        nrm = plane
    incl_sky = float(np.arccos(np.clip(nrm[2], -1, 1)))
    node_sky = float(np.arctan2(nrm[0], -nrm[1]) % (2 * np.pi))
    if np.isfinite(k.mass):
        m = k.mass
    elif np.isfinite(k.msini):
        m = k.msini / max(np.sin(incl_sky), 0.05)
    elif np.isfinite(k.mass_mr):
        m = k.mass_mr
    else:
        m = _mass_from_radius(k.radius) if np.isfinite(k.radius) else 10.0
    r = k.radius if np.isfinite(k.radius) else (k.radius_mr if np.isfinite(k.radius_mr) else _radius_from_mass(m))
    vals = {
        "system": p["system"][0] if len(p["system"]) else 0, "component": target_row, "host_kind": "S",
        "host_mass": host_mass, "role": 0, "model": "known", "hmc_catalog": 0,
        "flags": FLAG_KNOWN | FLAG_KNOWN_INSERTED,
        "a_au": a, "period": period, "planet_mass": m, "planet_radius": r, "clusterid": KNOWN_CLUSTER_ID,
        "ecc": k.ecc if np.isfinite(k.ecc) else 0.0, "omega": 2 * np.pi * rng.random(), "incl_mut": 0.0,
        "incl_invariable": 0.0, "asc_node_invariable": 0.0, "incl_sky": incl_sky, "node_sky": node_sky,
        "mean_anom": 2 * np.pi * rng.random(), "mass_init": m, "radius_init": r, "lost_envelope": r < 1.6,
        "known_name": k.name,
    }
    for key in row:
        v = vals.get(key, np.nan if row[key].dtype.kind == "f" else 0)
        row[key] = np.array([v], dtype=row[key].dtype)
    return row


def _refresh_invariable(p):
    """incl_invariable / asc_node_invariable of every planet w.r.t. its host's invariable plane,
    from the sky orientations (in place)."""
    ok = np.isfinite(p["node_sky"])
    if not ok.any():
        return
    idx = np.flatnonzero(ok)
    nrm = normals_from_sky(p["incl_sky"][idx], p["node_sky"][idx])
    key = p["system"][idx].astype(np.int64) * 1_000_000 + p["component"][idx] * 2 + (p["host_kind"][idx] == "P")
    u, inv = np.unique(key, return_inverse=True)
    e = np.clip(p["ecc"][idx], 0, 0.999)
    lw = p["planet_mass"][idx] * np.sqrt(p["a_au"][idx] * (1 - e * e))
    tot = np.zeros((len(u), 3))
    np.add.at(tot, inv, lw[:, None] * nrm)
    n_inv = tot[inv] / np.maximum(np.linalg.norm(tot[inv], axis=1, keepdims=True), 1e-300)
    p["incl_invariable"][idx] = np.arccos(np.clip((nrm * n_inv).sum(axis=1), -1, 1))
    p["asc_node_invariable"][idx] = pt._node_in_plane(nrm, n_inv)


# --- driver -------------------------------------------------------------------------------

def condition(target, n: int = 100, seed=None, known: list[KnownPlanet] | None = None, surveys="auto",
              model: ConditionModel = ConditionModel(), star_model: st.StarFitModel = st.StarFitModel(),
              planet_kwargs: dict | None = None, use_gaia: bool = True,
              include_controversial: bool = False, moons: bool = True) -> ConditionedSystems:
    """``n`` planetary systems of a real star (name, :class:`SystemRecord` or :class:`StarSystemFit`),
    conditioned on its known planets (default: the NASA archive's) and ``surveys`` (``"auto"``:
    :func:`starsys.real.surveys.default_surveys`; ``None``: none; or a list of survey objects)."""
    rng = np.random.default_rng(seed)
    if isinstance(target, str):
        target = find(target, use_gaia=use_gaia)
    if isinstance(target, SystemRecord):
        fit = st.fit(target, n=model.max_prior, seed=rng, model=star_model)
    else:
        fit = target
    record = fit.record
    if known is None:
        known = known_planets(record.star.nasa_host, include_controversial)
    ka = _known_arrays(known, model)
    planet_kwargs = dict(planet_kwargs or {})
    if isinstance(surveys, str) and surveys == "auto":
        from .surveys import default_surveys
        surveys = default_surveys(record, known)
    elif surveys is not None and callable(getattr(surveys, "p_detect", None)):
        surveys = [surveys]
    surveys = list(surveys or [])

    n_total = len(fit.target)
    batches = []  # (systems, components, target, planets, choices)
    all_chi2, all_logw, all_ins = [], [], []
    drawn = 0
    order = rng.permutation(n_total)
    while drawn < min(model.max_prior, n_total):
        idx = order[drawn:drawn + model.n_prior]
        drawn += len(idx)
        systems, components, tgt = _subset_stars(fit, idx)
        pl = pt.attach_planets(systems, components, seed=rng, **planet_kwargs)
        for sv in surveys:
            sv.prepare(record, systems, components, tgt, known, rng)
        logw, chi2, n_ins, choices = _weigh_batch(pl, components, tgt, known, ka, surveys, model, rng)
        batches.append((systems, components, tgt, pl, choices))
        all_chi2.append(chi2)
        all_logw.append(logw)
        all_ins.append(n_ins)
        lw = (np.concatenate(all_logw) - 0.5 * np.concatenate(all_chi2)
              - 0.5 * model.insert_chi2 * np.concatenate(all_ins))
        if np.isfinite(lw).any() and _ess(lw) >= model.target_ess:
            break
    chi2 = np.concatenate(all_chi2)
    logw = np.concatenate(all_logw)
    n_ins = np.concatenate(all_ins)
    if not np.isfinite(logw - 0.5 * chi2).any():
        raise ValueError(f"no prior system of {record.name} is compatible with its known planets")
    lam, mu, w, ess = tempering(chi2, logw, n_ins, model.insert_chi2, model.target_ess)
    sizes = np.array([len(b[2]) for b in batches])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    picks = rng.choice(len(w), size=n, p=w)

    out_pl, out_sys, out_comp, out_tgt = [], [], [], []
    n_comp = len(record.components)
    inserted = np.zeros(len(known))
    for i, g in enumerate(picks):
        b = int(np.searchsorted(offsets, g, side="right") - 1)
        s = int(g - offsets[b])
        systems, components, tgt, pl, choices = batches[b]
        comps = {k: np.array(v[s * n_comp:(s + 1) * n_comp]) for k, v in components.items()}
        rows_sys = np.flatnonzero(pl["system"] == s)
        choice = choices[s]
        p = _build_one(pl, rows_sys, comps, choice, known, s * n_comp + record.target, s * n_comp, model, rng)
        inserted += choice.col < 0
        # Renumber to the output system i.
        p["system"] = np.full(len(p["system"]), i, dtype=pl["system"].dtype)
        p["component"] = p["component"] - s * n_comp + i * n_comp
        comps["system"] = np.full(n_comp, i)
        out_pl.append(p)
        out_comp.append(comps)
        out_sys.append({k: np.asarray(v)[[s]] for k, v in systems.items()})
        out_tgt.append(i * n_comp + record.target)
    planets = {k: np.concatenate([p[k] for p in out_pl]) for k in out_pl[0]}
    components = {k: np.concatenate([c[k] for c in out_comp]) for k in out_comp[0]}
    systems = {k: np.concatenate([s_[k] for s_ in out_sys]) for k in out_sys[0]}
    srt = np.lexsort((planets["period"], planets["host_kind"] == "P", planets["component"]))
    planets = {k: v[srt] for k, v in planets.items()}
    _refresh_invariable(planets)
    planets["host_mass"] = np.asarray(components["mass"], dtype=float)[planets["component"]]
    dmodel = planet_kwargs.get("derived", pt.DEFAULT_DERIVED)
    if dmodel is not None:
        extra = dv.derive_properties(planets, systems, components, dmodel, rng)
        planets["flags"] = (planets["flags"] & ~FLAG_HZ_EXTRAP) | extra.pop("hz_flags")
        planets.update(extra)
    moon_table = mn.attach_moons(planets, components, seed=rng) if moons else {}
    return ConditionedSystems(record=record, known=known, systems=systems, components=components, planets=planets,
                              moons=moon_table, target=np.array(out_tgt), ess=ess, n_prior=drawn,
                              inserted_fraction={k.name: inserted[j] / n for j, k in enumerate(known)},
                              kernel_widening=lam, insertion_softening=mu, n_unique=len(np.unique(picks)),
                              surveys=[sv.describe() for sv in surveys])


__all__ = ["ConditionModel", "ConditionedSystems", "condition", "FLAG_KNOWN", "FLAG_KNOWN_INSERTED", "replace"]
