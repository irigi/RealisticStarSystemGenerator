"""Planets of evolved stars and remnants: checks of :mod:`starsys.evolved`.

Model planets come from the stellar population through :func:`starsys.planets.attach_planets`
(all models). Rates are intrinsic; RV rates use M sin i with each planet's sky inclination.

- **Census of evolved hosts:** planets per host, fraction with a giant planet, innermost
  and median orbit, by evolutionary phase.
- **Giant planets around giant stars (Wolthoff+2022; Lick, EXPRESS and PPPS, 482 giants):**
  completeness-corrected fraction of evolved stars with a planet of M sin i >= 0.8 MJup at
  P > 89 d, 10.7 +2.2 -1.6% overall, 14.2 +4.1 -2.7% for RGB and 6.6 +2.0 -1.3% for
  horizontal-branch stars. Their detections span 89-3168 d (baselines up to ~12 yr), so the
  model counts planets at 89-4400 d. Their RGB stars average 1.3 Msun, their HB stars 2.1 Msun
  (1-3.5 Msun), so the subsets differ mostly by mass; their mass-metallicity fit
  (f = 0.128 exp(-(M - 1.68)^2 / (2 0.70^2)) 10^(1.38 [Fe/H])) gives 11% and 7.4% for the two
  subsets. The model is shown by mass, next to that fit averaged over the model stars, and
  reweighted to a Wolthoff-like sample: 171 RGB stars of 1.0-1.6 Msun and 311 HB stars spread
  evenly over 1-3 Msun (their HB masses are near-uniform over 1-3.5 Msun), both with their
  sample's [Fe/H] distribution (mean -0.05, sd 0.18; the model's RGB stars are older and
  0.05 dex poorer), with the expected
  number of planets they would have found at P < 89 d (they found none; completeness is
  highest there).
- **Close-in giants around giants:** RV surveys find almost no giant planets with P < 90 d
  around giant stars (Johnson+2007; Sato+2008; Wolthoff+2022: one at 89 d among 37 planets),
  partly from engulfment and tidal decay (Villaver & Livio 2009). Fraction with a giant
  inside 0.5 au and at P < 90 d, giant stars vs main-sequence stars of the same mass.
- **White dwarfs:** fraction with surviving planets, in particular at 1-10 au where they can
  perturb planetesimals onto the star; 25-50% of white dwarfs show metal pollution
  (Koester+2014: 27% of DA white dwarfs accreting, up to ~50% with diffusion timescales).
  Innermost surviving planet vs progenitor mass: Mustill & Villaver (2012) find none inside
  ~1.5 au (1 Msun progenitor) and ~10 au (5 Msun).
- **The Sun:** engulfment radii of Earth, a Neptune and Jupiter (MV12: the Sun's AGB radius
  reaches ~1.5 au; Jupiter survives).

Usage: ``python -m starsys.validation.evolved [--systems 400000]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import evolved as ev
from .. import planets
from ..stars.population import draw_systems

GIANT_MODELS = ("giant", "hot_jupiter", "wide_giant")
MJUP_E = 317.83


def simulate(n_systems, seed=0):
    systems, comp = draw_systems(n_systems, seed=seed)
    return systems, comp, planets.attach_planets(systems, comp, seed=seed + 1)


def _groups(comp):
    kind, phase, log_g = comp["kind"], np.asarray(comp["phase"], dtype=float), np.asarray(comp["log_g"], dtype=float)
    star = kind == "star"
    return {"subgiant (2, log g >= 3.5)": star & (phase == 2) & (log_g >= 3.5),
            "RGB (2, log g < 3.5)": star & (phase == 2) & (log_g < 3.5),
            "core-He burning (3)": star & (phase == 3), "AGB (4-5)": star & np.isin(phase, (4, 5)),
            "white dwarf": kind == "WD"}


def census(comp, pl) -> pd.DataFrame:
    hosts = ev.evolved_hosts(comp)
    s = pl["host_kind"] == "S"
    n_all = np.bincount(pl["component"][s], minlength=len(comp["mass"]))
    n_g = np.bincount(pl["component"][s & np.isin(pl["model"], GIANT_MODELS)], minlength=len(comp["mass"]))
    rows = []
    for name, sel in _groups(comp).items():
        idx = np.flatnonzero(sel & hosts)
        m = s & sel[pl["component"]]
        rows.append({"phase": name, "stars": len(idx), "planets/star": n_all[idx].mean(),
                     "% with planets": 100 * np.mean(n_all[idx] > 0), "% with giant": 100 * np.mean(n_g[idx] > 0),
                     "innermost a (au)": pl["a_au"][m].min() if m.any() else np.nan,
                     "median a (au)": np.median(pl["a_au"][m]) if m.any() else np.nan})
    return pd.DataFrame(rows).set_index("phase")


def wolthoff(comp, pl) -> pd.DataFrame:
    g = _groups(comp)
    giants = g["RGB (2, log g < 3.5)"] | g["core-He burning (3)"]
    msini = pl["planet_mass"] * np.sin(pl["incl_sky"])
    sel = (pl["host_kind"] == "S") & (msini >= 0.8 * MJUP_E) & (pl["period"] > 89.0) & (pl["period"] < 4400.0)
    has = np.bincount(pl["component"][sel], minlength=len(comp["mass"])) > 0
    mass = np.asarray(comp["mass"])
    fit = 12.8 * np.exp(-0.5 * ((mass - 1.68) / 0.70) ** 2) * 10 ** (1.38 * np.asarray(comp["feh"]))
    rows = []
    for name, lo, hi in (("1.0-1.5 Msun", 1.0, 1.5), ("1.5-2.0 Msun", 1.5, 2.0), ("2.0-3.0 Msun", 2.0, 3.0),
                         ("all 1-3 Msun", 1.0, 3.0)):
        for label, grp in (("RGB", g["RGB (2, log g < 3.5)"]), ("HB", g["core-He burning (3)"]), ("all", giants)):
            idx = np.flatnonzero(grp & (mass >= lo) & (mass < hi))
            if len(idx) < 30:
                continue
            rows.append({"mass": name, "stage": label, "stars": len(idx), "model %": 100 * has[idx].mean(),
                         "Wolthoff fit %": fit[idx].mean()})
    out = pd.DataFrame(rows).set_index(["mass", "stage"])
    out.attrs["obs"] = "Wolthoff+2022: all 10.7 +2.2 -1.6%, RGB 14.2 +4.1 -2.7% (fit: 11%), HB 6.6 +2.0 -1.3% (fit: 7.4%)"
    return out


def wolthoff_sample(comp, pl) -> pd.DataFrame:
    """Model rates reweighted to Wolthoff+2022's sample (171 RGB stars of ~1.3 Msun, 311 HB stars of 1-3 Msun)."""
    g = _groups(comp)
    mass = np.asarray(comp["mass"])
    msini = pl["planet_mass"] * np.sin(pl["incl_sky"])
    big = (pl["host_kind"] == "S") & (msini >= 0.8 * MJUP_E)
    n = len(mass)
    has_win = np.bincount(pl["component"][big & (pl["period"] > 89.0) & (pl["period"] < 4400.0)], minlength=n) > 0
    has_in = np.bincount(pl["component"][big & (pl["period"] <= 89.0)], minlength=n) > 0
    rgb = np.flatnonzero(g["RGB (2, log g < 3.5)"] & (mass >= 1.0) & (mass < 1.6))
    hb = np.flatnonzero(g["core-He burning (3)"] & (mass >= 1.0) & (mass < 3.0))
    # HB stars weighted to a uniform mass distribution over 1-3 Msun.
    edges = np.linspace(1.0, 3.0, 9)
    k = np.clip(np.digitize(mass[hb], edges) - 1, 0, len(edges) - 2)
    w_hb = 1.0 / np.bincount(k, minlength=len(edges) - 1)[k]
    feh = np.asarray(comp["feh"])

    def feh_weight(idx, w):  # reweight to the sample's [Fe/H] distribution, N(-0.05, 0.18)
        f_edges = np.arange(-1.0, 0.61, 0.1)
        j = np.clip(np.digitize(feh[idx], f_edges) - 1, 0, len(f_edges) - 2)
        have = np.bincount(j, weights=w, minlength=len(f_edges) - 1)
        mid = 0.5 * (f_edges[1:] + f_edges[:-1])
        want = np.exp(-0.5 * ((mid + 0.05) / 0.18) ** 2)
        return w * np.where(have[j] > 0, want[j] / np.maximum(have[j], 1e-300), 0.0)

    rows = []
    for name, idx, w, n_obs, obs in (("RGB", rgb, feh_weight(rgb, np.ones(len(rgb))), 171, "14.2 +4.1 -2.7"),
                                     ("HB", hb, feh_weight(hb, w_hb), 311, "6.6 +2.0 -1.3")):
        rows.append({"stage": name, "model stars": len(idx), "Wolthoff stars": n_obs,
                     "model % (89-4400 d)": 100 * np.average(has_win[idx], weights=w), "Wolthoff %": obs,
                     "expected hosts at P <= 89 d": n_obs * np.average(has_in[idx], weights=w)})
    tot = {"stage": "all", "model stars": len(rgb) + len(hb), "Wolthoff stars": 482,
           "model % (89-4400 d)": (171 * rows[0]["model % (89-4400 d)"] + 311 * rows[1]["model % (89-4400 d)"]) / 482,
           "Wolthoff %": "10.7 +2.2 -1.6",
           "expected hosts at P <= 89 d": rows[0]["expected hosts at P <= 89 d"] + rows[1]["expected hosts at P <= 89 d"]}
    rows.append(tot)
    out = pd.DataFrame(rows).set_index("stage")
    out.attrs["obs"] = (f"Wolthoff+2022 found no host at P <= 89 d; Poisson probability of none given the model: "
                        f"{np.exp(-tot['expected hosts at P <= 89 d']):.2f}")
    return out


def close_in(comp, pl) -> pd.DataFrame:
    g = _groups(comp)
    giants = g["RGB (2, log g < 3.5)"] | g["core-He burning (3)"] | g["AGB (4-5)"]
    ms = (comp["kind"] == "star") & (comp["phase"] == 0)
    sel = (pl["host_kind"] == "S") & np.isin(pl["model"], GIANT_MODELS)
    has_close = np.bincount(pl["component"][sel & (pl["a_au"] < 0.5)], minlength=len(comp["mass"])) > 0
    has_90 = np.bincount(pl["component"][sel & (pl["period"] < 90.0)], minlength=len(comp["mass"])) > 0
    has_any = np.bincount(pl["component"][sel], minlength=len(comp["mass"])) > 0
    mass = np.asarray(comp["mass"])
    rows = []
    for name, lo, hi in (("1.0-1.5 Msun", 1.0, 1.5), ("1.5-2.5 Msun", 1.5, 2.5)):
        for label, grp in (("giant stars", giants), ("main sequence", ms)):
            idx = np.flatnonzero(grp & (mass >= lo) & (mass < hi))
            rows.append({"mass": name, "hosts": label, "stars": len(idx), "% giant < 0.5 au": 100 * has_close[idx].mean(),
                         "% giant P < 90 d": 100 * has_90[idx].mean(), "% any giant": 100 * has_any[idx].mean()})
    return pd.DataFrame(rows).set_index(["mass", "hosts"])


def white_dwarfs(comp, pl) -> pd.DataFrame:
    wd = np.flatnonzero(comp["kind"] == "WD")
    s = pl["host_kind"] == "S"
    n = len(comp["mass"])
    has = np.bincount(pl["component"][s], minlength=n) > 0
    has_1_10 = np.bincount(pl["component"][s & (pl["a_au"] >= 1) & (pl["a_au"] < 10)], minlength=n) > 0
    has_g = np.bincount(pl["component"][s & np.isin(pl["model"], GIANT_MODELS)], minlength=n) > 0
    inner = np.full(n, np.inf)
    np.minimum.at(inner, pl["component"][s], pl["a_au"][s])
    mi = np.asarray(comp["initial_mass"])
    rows = []
    for name, lo, hi in (("progenitor 0.8-1.5", 0.8, 1.5), ("1.5-2.5", 1.5, 2.5), ("2.5-4", 2.5, 4.0),
                         ("4-8", 4.0, 8.0), ("all", 0.0, 8.0)):
        idx = wd[(mi[wd] >= lo) & (mi[wd] < hi)]
        if not len(idx):
            continue
        fin = inner[idx][np.isfinite(inner[idx])]
        rows.append({"white dwarfs": name, "n": len(idx), "% with planets": 100 * has[idx].mean(),
                     "% with planet at 1-10 au": 100 * has_1_10[idx].mean(), "% with giant": 100 * has_g[idx].mean(),
                     "innermost a, 5th pct (au)": np.percentile(fin, 5) if len(fin) else np.nan,
                     "median innermost a (au)": np.median(fin) if len(fin) else np.nan})
    return pd.DataFrame(rows).set_index("white dwarfs")


def sun(model=planets.DEFAULT_EVOLVED) -> pd.DataFrame:
    r_rgb, r_agb = ev.radius_tips(1.0, 0.0)
    r_max = np.maximum(r_rgb, r_agb)
    rows = []
    for name, mp, a0 in (("Earth", 1.0, 1.0), ("Neptune-mass", 17.1, 1.5), ("Jupiter", 317.8, 5.2)):
        r = ev.engulfment_radius(np.array([mp]), np.array([1.0]), r_max, r_rgb, np.array([False]), model)[0]
        rows.append({"planet": name, "a0 (au)": a0, "engulfed inside (au)": r, "fate": "engulfed" if a0 < r else
                     f"survives at {a0 / 0.54:.1f} au"})
    out = pd.DataFrame(rows).set_index("planet")
    out.attrs["radii"] = f"MIST 1 Msun: RGB tip {r_rgb[0]:.2f} au, AGB tip {r_agb[0]:.2f} au (WD mass ~0.54 Msun)"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.precision", 3)
    systems, comp, pl = simulate(args.systems, args.seed)
    print("== Evolved hosts (S-type planets) ==")
    print(census(comp, pl))
    w = wolthoff(comp, pl)
    print(f"\n== Giant stars with a planet of M sin i >= 0.8 MJup at P > 89 d ==\n{w.attrs['obs']}")
    print(w)
    ws = wolthoff_sample(comp, pl)
    print(f"\n== Reweighted to Wolthoff+2022's sample ==\n{ws.attrs['obs']}")
    print(ws)
    print("\n== Giant planets inside 0.5 au: giant stars vs main-sequence stars ==")
    print(close_in(comp, pl))
    print("\n== White dwarfs ==")
    print(white_dwarfs(comp, pl))
    s = sun()
    print(f"\n== The Sun's planets ({s.attrs['radii']}) ==")
    print(s)
    f = pl["flags"]
    print(f"\npulsar planets: {(pl['model'] == 'pulsar').sum()} around {(comp['kind'] == 'NS').sum()} neutron stars; "
          f"planets flagged post_interaction: {np.mean((f & planets.FLAG_POST_INTERACTION) > 0):.4f}, "
          f"evolved: {np.mean((f & planets.FLAG_EVOLVED) > 0):.4f}")


if __name__ == "__main__":
    main()
