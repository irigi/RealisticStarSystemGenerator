"""Planets in multiple systems: checks of :mod:`starsys.binaries`.

Model planets come from the stellar population through :func:`starsys.planets.attach_planets`
(all models; ``evolved=None`` so that only main-sequence effects are compared). Rates are
intrinsic (no detection model).

- **Suppressed fraction (Moe & Kratter 2021, MK21):** the fraction of solar-type primaries
  (0.8-1.2 Msun, main sequence) without close planets because of a companion, 1 - <S_bin>:
  33 +- 5% volume-limited, 43 +- 7% magnitude-limited (weight (L_tot/L_1)^1.5); M dwarfs
  28 +- 5% and A stars 52 +- 8% magnitude-limited. Shown with and without HM-C's
  luminosity cut (L_tot <= 1.75 L_1), which the boost uses.
- **Kepler-like average:** inner (HM-C) planets per primary at 3-300 d over the mock
  magnitude-limited Kepler sample, with the binary model (suppression + boost) and without
  it (every star single, the fitted field average). They should agree; the multiplicity mix
  of the boosted systems is shown as well.
- **Hot Jupiters, transit vs RV (MK21 Section 3):** the giant model's hot-Jupiter rate is
  fitted to RV surveys, which remove close binaries (R_SingleStars; ~1% for CLS). Over the
  mock transit sample (no cut against binaries) the model should give ~1/B of it; TESS and
  Kepler measure 0.4-0.6% (MK21) and Beleznay+2022 0.5-1% for FGK stars rising with mass.
- **Circumbinary planets (Armstrong+2014, Kepler eclipsing binaries, coplanar case):**
  fraction of FGK main-sequence pairs with P_bin = 2-100 d and a stability limit inside 300 d
  (Armstrong's search window), weighted by the eclipse probability (R_1 + R_2)/a, with at
  least one circumbinary planet at P < 300 d: 6-10 Earth radii
  10.0% (50% interval 8.2-16.3, 95% 3.5-28), > 10 Earth radii 0% (95% upper limit 5.9%),
  4-6 Earth radii 14% (11.7-26, 95% 4.2-48).
- **Separation of companions of planet hosts:** fraction of S-type planet hosts with a
  stellar companion in bins of separation, relative to all stars of the same mass: recovers
  S_bin (consistency check).

Usage: ``python -m starsys.validation.binaries [--systems 300000]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import binaries as bn
from .. import planets
from ..stars.population import draw_systems
from ..units import a_au_to_period

INNER_MODELS = ("hmc", "hmc_m")
ARMSTRONG2014 = {"6-10": (6.0, 10.0, "10.0 (50%: 8.2-16.3; 95%: 3.5-28)"),
                 "> 10": (10.0, 1e9, "0 (95% < 5.9)"),
                 "4-6": (4.0, 6.0, "14 (50%: 11.7-26; 95%: 4.2-48)"),
                 "8-10": (8.0, 10.0, "6.4 (50%: 5.7-11.6; 95%: 2.5-19.7)")}
MK21_F = {"G volume-limited": "33 +- 5", "G magnitude-limited": "43 +- 7", "M magnitude-limited": "28 +- 5",
          "A magnitude-limited": "52 +- 8"}


def simulate(n_systems, seed=0):
    systems, comp = draw_systems(n_systems, seed=seed)
    on = planets.attach_planets(systems, comp, seed=seed + 1, evolved=None)
    off = planets.attach_planets(systems, comp, seed=seed + 1, evolved=None, binaries=None)
    return systems, comp, on, off


def suppressed_fraction(systems, comp, model=planets.DEFAULT_BINARIES) -> pd.DataFrame:
    st = bn.structure(comp)
    s = bn.s_bin(st["a_near"], model)
    sys_ = np.asarray(comp["system"])
    lum = np.nan_to_num(np.asarray(comp["luminosity"], dtype=float))
    l_tot = np.bincount(sys_, weights=lum, minlength=len(systems["age_gyr"]))[sys_]
    w_mag = (l_tot / np.maximum(lum, 1e-30)) ** model.malmquist_exponent
    cut = l_tot <= model.luminosity_cut * lum
    prim = (np.asarray(comp["role"]) == 0) & np.isin(comp["phase"], (-1, 0)) & (comp["kind"] == "star")
    mass = np.asarray(comp["mass"])
    rows = []
    for name, lo, hi, ref in (("M 0.1-0.5", 0.1, 0.5, "M"), ("G 0.8-1.2", 0.8, 1.2, "G"), ("A 1.6-3", 1.6, 3.0, "A")):
        sel = prim & (mass >= lo) & (mass < hi)
        rows.append({"primaries": name,
                     "volume-limited": 1 - s[sel].mean(),
                     "magnitude-limited": 1 - np.average(s[sel], weights=w_mag[sel]),
                     "mag.-limited, L cut": 1 - np.average(s[sel & cut], weights=w_mag[sel & cut]),
                     "boost B": 1 / np.average(s[sel & cut], weights=w_mag[sel & cut]),
                     "MK21 (mag.-limited)": MK21_F[f"{ref} magnitude-limited"]
                     + (f"; vol. {MK21_F['G volume-limited']}" if ref == "G" else "")})
    return pd.DataFrame(rows).set_index("primaries")


def kepler_average(systems, comp, on, off, model=planets.DEFAULT_BINARIES) -> tuple[pd.DataFrame, pd.DataFrame]:
    prim, w, _ = bn.mock_kepler(systems, comp, model)
    teff = np.asarray(comp["teff"])[prim]
    rows, mult = [], []
    for name, lo, hi in (("M 3000-3900 K", 3000, 3900), ("K 3900-5300 K", 3900, 5300), ("G 5300-6000 K", 5300, 6000),
                         ("F 6000-6800 K", 6000, 6800)):
        sel = (teff >= lo) & (teff < hi)
        p, ww = prim[sel], w[sel]
        row = {"Teff": name, "primaries": len(p)}
        for label, pl in (("binaries on", on), ("binaries off", off)):
            inner = np.isin(pl["model"], INNER_MODELS) & (pl["host_kind"] == "S") & (pl["period"] >= 3) & (
                pl["period"] <= 300)
            n = np.bincount(pl["component"][inner], minlength=len(comp["mass"]))[p]
            row[label] = np.average(n, weights=ww)
            m = {"label": label, "Teff": name}
            hosts = n > 0
            m["hosts"] = np.average(hosts, weights=ww)
            for k, (a, b) in {"1": (1, 1), "2": (2, 2), "3-4": (3, 4), "5+": (5, 99)}.items():
                m[k] = np.average((n >= a) & (n <= b), weights=ww) / max(m["hosts"], 1e-12)
            mult.append(m)
        row["ratio"] = row["binaries on"] / row["binaries off"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Teff"), pd.DataFrame(mult).set_index(["Teff", "label"])


def hot_jupiters(systems, comp, on, model=planets.DEFAULT_BINARIES) -> pd.DataFrame:
    prim, w, _ = bn.mock_kepler(systems, comp, model)
    st = bn.structure(comp)
    mass = np.asarray(comp["mass"])
    hj = np.bincount(on["component"][(on["model"] == "hot_jupiter") & (on["host_kind"] == "S")],
                     minlength=len(mass)) > 0
    rows = []
    for name, lo, hi in (("0.6-1.0 Msun", 0.6, 1.0), ("1.0-1.4 Msun", 1.0, 1.4), ("FGK 0.6-1.4", 0.6, 1.4)):
        sel = (mass[prim] >= lo) & (mass[prim] < hi)
        p = prim[sel]
        single = ~(st["a_near"][p] < 200.0)
        rows.append({"hosts": name, "transit-like (all, mag.-limited) %": 100 * np.average(hj[p], weights=w[sel]),
                     "RV-like (no companion < 200 au) %": 100 * hj[p][single].mean(),
                     "ratio": np.average(hj[p], weights=w[sel]) / max(hj[p][single].mean(), 1e-12)})
    return pd.DataFrame(rows).set_index("hosts")


def circumbinary(comp, on) -> pd.DataFrame:
    role = np.asarray(comp["role"])
    sys_ = np.asarray(comp["system"])
    n = len(role)
    p = np.flatnonzero(role == 0)
    s = p + 1
    has = (s < n) & (sys_[np.minimum(s, n - 1)] == sys_[p])
    p, s = p[has], s[has]
    pb = np.asarray(comp["period_days"])[s]
    fgk = lambda i: (comp["kind"][i] == "star") & np.isin(comp["phase"][i], (-1, 0)) & (comp["teff"][i] >= 3900) & (
        comp["teff"][i] < 7000)
    # Pairs whose stability limit lies inside 300 d (Armstrong's search window is P_crit-300 d).
    st = bn.structure(comp)
    m_pair = np.asarray(comp["mass"])[p] + np.asarray(comp["mass"])[s]
    p_crit = a_au_to_period(st["pair_a_min"][p], m_pair)
    sel = fgk(p) & (pb >= 2.0) & (pb <= 100.0) & (p_crit < 300.0)
    p, s, pb = p[sel], s[sel], pb[sel]
    a = np.asarray(comp["a_au"])[s]
    w = (np.asarray(comp["radius"])[p] + np.asarray(comp["radius"])[s]) / 215.032 / a
    cb = (on["host_kind"] == "P") & (on["period"] < 300.0)
    rows = []
    for label, (lo, hi, ref) in ARMSTRONG2014.items():
        x = cb & (on["planet_radius"] >= lo) & (on["planet_radius"] < hi)
        has_pl = np.bincount(on["component"][x], minlength=n)[p] > 0
        rows.append({"radius (Re)": label, "model %": 100 * np.average(has_pl, weights=w), "Armstrong+2014 %": ref})
    out = pd.DataFrame(rows).set_index("radius (Re)")
    out.attrs["pairs"] = len(p)
    return out


def separation_check(comp, on) -> pd.DataFrame:
    st = bn.structure(comp)
    mass = np.asarray(comp["mass"])
    ms = np.isin(comp["phase"], (-1, 0)) & (comp["kind"] == "star") & (mass >= 0.6) & (mass < 1.4)
    host = np.bincount(on["component"][on["host_kind"] == "S"], minlength=len(mass)) > 0
    edges = [0.3, 1, 3, 10, 30, 100, 300, 1000]
    rows = []
    f_all = ms.sum()
    f_host = (ms & host).sum()
    single = ms & ~np.isfinite(st["a_near"])
    base = (single & host).sum() / single.sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = ms & (st["a_near"] >= lo) & (st["a_near"] < hi)
        rows.append({"a (au)": f"{lo}-{hi}", "field fraction": b.sum() / f_all,
                     "host fraction": (b & host).sum() / f_host,
                     "host rate / single-star rate": (b & host).sum() / max(b.sum(), 1) / base,
                     "S_bin at bin centre": float(bn.s_bin(np.sqrt(lo * hi)))})
    return pd.DataFrame(rows).set_index("a (au)")


CORRECTION_TEFF = (2600.0, 3000.0, 3400.0, 3800.0, 4300.0, 4800.0, 5300.0, 5800.0, 6300.0, 7000.0)


def _inner_counts(pl, n_comp, prim):
    inner = np.isin(pl["model"], INNER_MODELS) & (pl["host_kind"] == "S") & (pl["period"] >= 3) & (pl["period"] <= 300)
    return np.bincount(pl["component"][inner], minlength=n_comp)[prim]


def fit_boost_correction(n_systems=400_000, iterations=3, seed=0):
    """Fit ``BinaryModel.boost_correction`` so that the mock-Kepler inner-planet average with the
    binary model equals the field average without it (per Teff knot)."""
    from dataclasses import replace

    systems, comp = draw_systems(n_systems, seed=seed)
    model = planets.DEFAULT_BINARIES
    prim, w, _ = bn.mock_kepler(systems, comp, model)
    teff = np.asarray(comp["teff"])[prim]
    off = planets.attach_planets(systems, comp, seed=seed + 1, evolved=None, binaries=None, giants=None, small=None)
    n_off = _inner_counts(off, len(comp["mass"]), prim)
    knots = np.array(CORRECTION_TEFF)
    edges = np.concatenate([[0.0], 0.5 * (knots[1:] + knots[:-1]), [1e5]])
    corr = np.ones(len(knots))
    for it in range(iterations):
        model = replace(model, boost_correction=tuple(zip(knots.tolist(), np.round(corr, 4).tolist())))
        on = planets.attach_planets(systems, comp, seed=seed + 2 + it, evolved=None, binaries=model, giants=None,
                                    small=None)
        n_on = _inner_counts(on, len(comp["mass"]), prim)
        for j in range(len(knots)):
            sel = (teff >= edges[j]) & (teff < edges[j + 1])
            if sel.sum() < 200:
                continue
            ratio = np.average(n_on[sel], weights=w[sel]) / np.average(n_off[sel], weights=w[sel])
            corr[j] /= ratio
        print(f"iteration {it + 1}: " + ", ".join(f"{k:.0f}: {c:.3f}" for k, c in zip(knots, corr)))
    return tuple(zip(knots.tolist(), np.round(corr, 3).tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", action="store_true", help="fit the boost efficiency correction")
    args = ap.parse_args(argv)
    if args.fit:
        print("boost_correction =", fit_boost_correction())
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.precision", 3)
    systems, comp, on, off = simulate(args.systems, args.seed)
    print("== Fraction of primaries without close planets because of companions (1 - <S_bin>) ==")
    print(suppressed_fraction(systems, comp))
    avg, mult = kepler_average(systems, comp, on, off)
    print("\n== Inner (HM-C) planets per primary, 3-300 d, mock magnitude-limited Kepler sample ==")
    print(avg)
    print("\n== Multiplicity of inner systems (fraction of hosts), same sample ==")
    print(mult)
    print("\n== Hot Jupiters: transit-like vs RV-like samples ==")
    print(hot_jupiters(systems, comp, on))
    cb = circumbinary(comp, on)
    print(f"\n== Circumbinary planets, P < 300 d, FGK pairs with P_bin 2-100 d (eclipse-weighted; "
          f"{cb.attrs['pairs']} pairs) ==")
    print(cb)
    print("\n== Companions of S-type planet hosts (0.6-1.4 Msun MS) vs the field ==")
    print(separation_check(comp, on))
    n_p = (on["host_kind"] == "P").sum()
    print(f"\nplanets: {len(on['period'])} with binaries, {len(off['period'])} without; circumbinary {n_p}; "
          f"flagged multiple {np.mean((on['flags'] & planets.FLAG_MULTIPLE) > 0):.3f} "
          f"(without the binary model {np.mean((off['flags'] & planets.FLAG_MULTIPLE) > 0):.3f})")


if __name__ == "__main__":
    main()
