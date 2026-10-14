"""Phase 4 validation report: the whole planet population against the data it rests on.

One draw of the stellar population with every planet model attached (defaults of
:func:`starsys.planets.attach_planets`). Writes ``docs/validation/phase4_report.md`` and the
figures in ``docs/validation/figures/``; the per-step checks are in the other
``starsys.validation`` modules.

- **Occurrence vs period and radius by host type** (intrinsic planets per star, S-type hosts):
  main-sequence M, K, G, F and A stars, giant stars, white dwarfs, circumbinary pairs.
- **Kepler:** the inner planets (HM-C, 3-300 d, 0.5-10 Earth radii) of a mock magnitude-limited
  Kepler sample of the population (binaries included, Teff distribution reweighted to the Kepler
  targets of HM-C) against the published HM-C physical catalogs (intrinsic planets per
  Kepler target; mean and spread of 20 posterior catalogs).
- **M dwarfs:** Dressing & Charbonneau (2015) and Hsu+2020 integrated rates for the mock
  sample's early M dwarfs.
- **CLS:** giant-planet rates around effectively single Sun-like stars (no companion inside
  200 au), M sin i, against Fulton+2021 and Rosenthal+2022.
- **Giant-planet correlations:** metallicity (Bryan & Lee 2024), stellar mass, and the
  inner-small/outer-giant relation (Bonomo+2025: none at average metallicity).
- **Where the model extrapolates:** share of planets carrying each flag, and the share with
  any flag by host type and semi-major axis.

Usage: ``python -m starsys.validation.report [--systems 300000]``
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import binaries as bn  # noqa: E402
from .. import planets  # noqa: E402
from ..inner import mdwarf as md  # noqa: E402
from ..paths import REPO_ROOT  # noqa: E402
from ..stars.population import draw_systems  # noqa: E402
from ..syssim_catalogs import MODELS, read_physical_catalog, read_physical_stars  # noqa: E402
from .hmc import load_hmc_stellar_table  # noqa: E402

DOC = REPO_ROOT / "docs" / "validation" / "phase4_report.md"
FIG = REPO_ROOT / "docs" / "validation" / "figures"
INNER = ("hmc", "hmc_m")
GIANTS = ("giant", "hot_jupiter", "wide_giant")
MJUP_E = 317.83
P_EDGES = np.array([3.0, 10.0, 30.0, 100.0, 300.0])
R_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
A_EDGES = np.array([0.0, 0.1, 1.0, 10.0, 100.0, np.inf])
LOGP_EDGES = np.arange(-0.5, 6.01, 0.25)


def host_types(comp) -> dict[str, np.ndarray]:
    kind = np.asarray(comp["kind"])
    phase = np.asarray(comp["phase"], dtype=float)
    teff = np.asarray(comp["teff"], dtype=float)
    mass = np.asarray(comp["mass"], dtype=float)
    ms = (kind == "star") & np.isin(phase, (-1, 0))
    return {
        "brown dwarfs": kind == "BD",
        "M (MS, < 3900 K)": ms & (teff < 3900) & (mass >= 0.075),
        "K (3900-5300 K)": ms & (teff >= 3900) & (teff < 5300),
        "G (5300-6000 K)": ms & (teff >= 5300) & (teff < 6000),
        "F (6000-7300 K)": ms & (teff >= 6000) & (teff < 7300),
        "A and hotter (MS, > 7300 K)": ms & (teff >= 7300),
        "subgiants and giants": (kind == "star") & np.isin(phase, (2, 3, 4, 5)),
        "white dwarfs": kind == "WD",
        "neutron stars": kind == "NS",
    }


# --- occurrence -------------------------------------------------------------------------

def occurrence_by_type(comp, pl) -> pd.DataFrame:
    s = pl["host_kind"] == "S"
    n_comp = len(comp["mass"])
    rows = []
    for name, sel in host_types(comp).items():
        n = sel.sum()
        if not n:
            continue
        m = s & sel[pl["component"]]
        per = np.bincount(pl["component"][m], minlength=n_comp)[sel]
        row = {"hosts": name, "stars": int(n), "planets/star": per.mean(), "% with planets": 100 * np.mean(per > 0)}
        for label, models in (("inner (HM-C)", INNER), ("giants", GIANTS), ("outer small", ("outer_small",))):
            row[label] = np.isin(pl["model"][m], models).sum() / n
        rows.append(row)
    pairs = np.flatnonzero((np.asarray(comp["role"]) == 1))
    p = pl["host_kind"] == "P"
    rows.append({"hosts": "circumbinary (per inner pair)", "stars": len(pairs), "planets/star": p.sum() / len(pairs),
                 "% with planets": 100 * len(np.unique(pl["component"][p])) / len(pairs),
                 "inner (HM-C)": np.isin(pl["model"][p], INNER).sum() / len(pairs),
                 "giants": np.isin(pl["model"][p], GIANTS).sum() / len(pairs),
                 "outer small": (pl["model"][p] == "outer_small").sum() / len(pairs)})
    return pd.DataFrame(rows).set_index("hosts")


def figure_period_profiles(comp, pl, path):
    s = pl["host_kind"] == "S"
    types = host_types(comp)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    centres = 0.5 * (LOGP_EDGES[1:] + LOGP_EDGES[:-1])
    for ax, (title, sel_pl) in zip(axes, (("planets < 30 Earth masses", pl["planet_mass"] < 30),
                                          ("planets >= 30 Earth masses", pl["planet_mass"] >= 30))):
        for name in ("M (MS, < 3900 K)", "K (3900-5300 K)", "G (5300-6000 K)", "F (6000-7300 K)",
                     "A and hotter (MS, > 7300 K)", "subgiants and giants", "white dwarfs"):
            sel = types[name]
            m = s & sel_pl & sel[pl["component"]]
            h = np.histogram(np.log10(pl["period"][m]), LOGP_EDGES)[0] / sel.sum() / (np.diff(LOGP_EDGES) * np.log(10))
            ax.step(centres, np.where(h > 0, h, np.nan), where="mid", label=name)
        ax.set_yscale("log")
        ax.set_xlabel("log10 P (d)")
        ax.set_title(title)
        ax.axvline(np.log10(300), color="grey", lw=0.5, ls=":")
    axes[0].set_ylabel("planets per star per ln P")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_pr_grids(comp, pl, path):
    s = pl["host_kind"] == "S"
    types = host_types(comp)
    p_edges = np.logspace(-0.3, 5, 22)
    r_edges = np.logspace(np.log10(0.3), np.log10(25), 18)
    names = ("M (MS, < 3900 K)", "K (3900-5300 K)", "G (5300-6000 K)", "F (6000-7300 K)")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, name in zip(axes, names):
        sel = types[name]
        m = s & sel[pl["component"]]
        h = np.histogram2d(pl["period"][m], pl["planet_radius"][m], [p_edges, r_edges])[0] / sel.sum()
        im = ax.pcolormesh(p_edges, r_edges, np.log10(np.where(h > 0, h, np.nan)).T, vmin=-4, vmax=-0.5,
                           cmap="viridis")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("P (d)")
    axes[0].set_ylabel("R (Earth radii)")
    fig.colorbar(im, ax=axes, label="log10 planets per star per cell")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- Kepler (published HM-C) -------------------------------------------------------------

def published_grid(catalog_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-target planets in (R, P) cells: mean and SD over catalogs; Kepler target Teffs."""
    model = MODELS["hmc"]
    stellar = load_hmc_stellar_table().set_index("star_id")
    grids = []
    teff = None
    for cid in catalog_ids:
        _, pub = read_physical_catalog(model, cid)
        targets = read_physical_stars(model, cid)
        h = np.histogram2d(pub["planet_radius"], pub["period"], [R_EDGES, P_EDGES])[0]
        grids.append(h / len(targets))
        if teff is None:
            teff = stellar.loc[targets["star_id"], "teff"].to_numpy()
    g = np.array(grids)
    return g.mean(axis=0), g.std(axis=0), teff


def kepler_comparison(systems, comp, pl, catalog_ids=range(1, 21)) -> tuple[pd.DataFrame, pd.DataFrame]:
    pub_mean, pub_sd, kep_teff = published_grid(catalog_ids)
    prim, w, _ = bn.mock_kepler(systems, comp, planets.DEFAULT_BINARIES)
    teff = np.asarray(comp["teff"])[prim]
    # Reweight the mock sample to the Kepler targets' Teff distribution.
    edges = np.arange(np.floor(kep_teff.min() / 100) * 100, kep_teff.max() + 100, 100.0)
    h_kep = np.histogram(kep_teff, edges)[0] / len(kep_teff)
    b = np.digitize(teff, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    prim, w, b = prim[ok], w[ok], b[ok]
    w_bin = np.bincount(b, weights=w, minlength=len(h_kep))
    w = w * h_kep[b] / np.maximum(w_bin[b], 1e-30)
    m = np.isin(pl["model"], INNER) & (pl["host_kind"] == "S")
    weight_of = np.zeros(len(comp["mass"]))
    weight_of[prim] = w
    pw = weight_of[pl["component"][m]]
    model_grid = np.histogram2d(pl["planet_radius"][m], pl["period"][m], [R_EDGES, P_EDGES], weights=pw)[0] / w.sum()
    cols = [f"{a:g}-{b:g} d" for a, b in zip(P_EDGES[:-1], P_EDGES[1:])]
    idx = [f"{a:g}-{b:g} Re" for a, b in zip(R_EDGES[:-1], R_EDGES[1:])]
    ratio = pd.DataFrame(model_grid / pub_mean, index=idx, columns=cols)
    z = pd.DataFrame((model_grid - pub_mean) / np.maximum(pub_sd, 1e-12), index=idx, columns=cols)
    totals = pd.DataFrame({"model": [model_grid.sum()], "published HM-C": [pub_mean.sum()],
                           "posterior SD": [np.sqrt((pub_sd**2).sum())]}, index=["planets per star, 3-300 d"])
    ratio.attrs["totals"] = totals
    return ratio, z


def mdwarf_rates(systems, comp, pl) -> pd.DataFrame:
    prim, w, _ = bn.mock_kepler(systems, comp, planets.DEFAULT_BINARIES)
    mass = np.asarray(comp["mass"])[prim]
    sel = (mass >= 0.35) & (mass < 0.65)
    prim, w = prim[sel], w[sel]
    m = np.isin(pl["model"], INNER) & (pl["host_kind"] == "S")
    weight_of = np.zeros(len(comp["mass"]))
    weight_of[prim] = w

    def rate(p_range, r_range):
        x = m & (pl["period"] >= p_range[0]) & (pl["period"] < p_range[1]) & (pl["planet_radius"] >= r_range[0]) & (
            pl["planet_radius"] < r_range[1])
        return weight_of[pl["component"][x]].sum() / w.sum()

    rows = [{"rate": "DC15: 1-4 Re, P < 200 d", "model": rate((0.5, 200), (1, 4)), "observed": "2.5 +- 0.2"}]
    for p_r, r_r, dirichlet, uniform in md.HSU2020_INTEGRATED:
        rows.append({"rate": f"Hsu+2020: {r_r[0]:g}-{r_r[1]:g} Re, {p_r[0]:g}-{p_r[1]:g} d", "model": rate(p_r, r_r),
                     "observed": f"{dirichlet[0]} +{dirichlet[1]}/-{dirichlet[2]} (Dirichlet) to "
                                 f"{uniform[0]} +{uniform[1]}/-{uniform[2]} (uniform)"})
    return pd.DataFrame(rows).set_index("rate")


# --- CLS and correlations ---------------------------------------------------------------

def _sun_like_single(comp):
    st = bn.structure(comp)
    mass = np.asarray(comp["mass"])
    ms = (comp["kind"] == "star") & np.isin(comp["phase"], (-1, 0))
    return ms & (mass >= 0.7) & (mass < 1.3) & ~(st["a_near"] < 200.0)


def cls_rates(comp, pl) -> pd.DataFrame:
    hosts = _sun_like_single(comp)
    n = hosts.sum()
    s = (pl["host_kind"] == "S") & hosts[pl["component"]]
    msini = pl["planet_mass"] * np.sin(pl["incl_sky"])
    a = pl["a_au"]
    big = s & (msini >= 30) & (msini < 6000)
    rows = [
        {"rate": "2-8 au, 30-6000 Me (per 100 stars)", "model": 100 * (big & (a >= 2) & (a < 8)).sum() / n,
         "observed": "14.1 +2.0/-1.8 (Fulton+2021)"},
        {"rate": "8-32 au, 30-6000 Me (per 100 stars)", "model": 100 * (big & (a >= 8) & (a < 32)).sum() / n,
         "observed": "8.9 +3.0/-2.4 (Fulton+2021)"},
        {"rate": "% stars with a giant at 0.23-10 au",
         "model": 100 * np.mean(np.bincount(pl["component"][big & (a >= 0.23) & (a < 10)],
                                            minlength=len(hosts))[hosts] > 0),
         "observed": "17.6 +2.4/-1.9 (Rosenthal+2022)"},
        {"rate": "% stars with a hot Jupiter",
         "model": 100 * np.mean(np.bincount(pl["component"][s & (pl["model"] == "hot_jupiter")],
                                            minlength=len(hosts))[hosts] > 0),
         "observed": "~1 (RV)"},
    ]
    out = pd.DataFrame(rows).set_index("rate")
    out.attrs["n"] = int(n)
    return out


def giant_correlations(comp, pl) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = bn.structure(comp)
    mass = np.asarray(comp["mass"])
    feh = np.asarray(comp["feh"])
    ms = (comp["kind"] == "star") & np.isin(comp["phase"], (-1, 0)) & ~(st["a_near"] < 200.0)
    s = pl["host_kind"] == "S"
    msini = pl["planet_mass"] * np.sin(pl["incl_sky"])
    cj = s & np.isin(pl["model"], GIANTS) & (pl["a_au"] >= 1) & (pl["a_au"] < 10) & (msini >= 0.5 * MJUP_E) & (
        msini < 20 * MJUP_E)
    se = s & np.isin(pl["model"], INNER) & (pl["planet_radius"] >= 1) & (pl["planet_radius"] < 4)
    has_cj = np.bincount(pl["component"][cj], minlength=len(mass)) > 0
    has_se = np.bincount(pl["component"][se], minlength=len(mass)) > 0
    sun = ms & (mass >= 0.7) & (mass < 1.3)
    rows = []
    for label, sel, obs in (("[Fe/H] > 0", sun & (feh > 0), "14.3 +2.0/-1.8 (Bryan & Lee 2024)"),
                            ("[Fe/H] <= 0", sun & (feh <= 0), "5.0 +1.6/-1.3"),
                            ("all", sun, "9.8 +0.9/-0.8 (Bonomo+2025, CLS-like sample)")):
        rows.append({"Sun-like hosts (0.7-1.3 Msun)": label, "% with cold giant": 100 * has_cj[sel].mean(),
                     "% with cold giant, given an inner 1-4 Re planet": 100 * has_cj[sel & has_se].mean(),
                     "observed (% with cold giant)": obs})
    corr = pd.DataFrame(rows).set_index("Sun-like hosts (0.7-1.3 Msun)")
    corr.attrs["note"] = ("Bonomo+2025: P(CJ | SE) consistent with P(CJ) at average metallicity; Bryan & Lee 2024: "
                          "28 +- 7% vs 14% for metal-rich stars")
    rows = []
    for lo, hi in ((0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 2.5), (2.5, 4.0)):
        sel = ms & (mass >= lo) & (mass < hi)
        rows.append({"M* (Msun)": f"{lo}-{hi}", "stars": int(sel.sum()), "% with cold giant": 100 * has_cj[sel].mean()})
    return corr, pd.DataFrame(rows).set_index("M* (Msun)")


# --- extrapolation map --------------------------------------------------------------------

def flag_shares(pl) -> pd.DataFrame:
    rows = [{"flag": name, "% of planets": 100 * np.mean((pl["flags"] & bit) > 0)}
            for bit, name in planets.FLAG_NAMES.items()]
    rows.append({"flag": "any", "% of planets": 100 * np.mean(pl["flags"] > 0)})
    rows.append({"flag": "any except young", "% of planets": 100 * np.mean((pl["flags"] & ~planets.FLAG_YOUNG) > 0)})
    return pd.DataFrame(rows).set_index("flag")


def extrapolation_map(comp, pl) -> pd.DataFrame:
    types = host_types(comp)
    s = pl["host_kind"] == "S"
    flagged = (pl["flags"] & ~planets.FLAG_YOUNG) > 0
    cols = [f"{a:g}-{b:g} au" if np.isfinite(b) else f"> {a:g} au" for a, b in zip(A_EDGES[:-1], A_EDGES[1:])]
    rows, counts = {}, {}
    groups = {name: s & sel[pl["component"]] for name, sel in types.items()}
    groups["circumbinary"] = ~s
    for name, m in groups.items():
        b = np.digitize(pl["a_au"][m], A_EDGES) - 1
        n = np.bincount(b, minlength=len(cols))[:len(cols)]
        f = np.bincount(b, weights=flagged[m].astype(float), minlength=len(cols))[:len(cols)]
        rows[name] = np.where(n > 0, 100 * f / np.maximum(n, 1), np.nan)
        counts[name] = n
    out = pd.DataFrame(rows, index=cols).T
    out.attrs["counts"] = pd.DataFrame(counts, index=cols).T
    return out


def figure_extrapolation(xmap, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(xmap.to_numpy(dtype=float), cmap="magma_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(xmap.shape[1]), xmap.columns)
    ax.set_yticks(range(xmap.shape[0]), xmap.index)
    for i in range(xmap.shape[0]):
        for j in range(xmap.shape[1]):
            v = xmap.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 60 else "black", fontsize=8)
    fig.colorbar(im, label="% of planets with an extrapolation flag (excluding 'young')")
    ax.set_title("Where the model extrapolates")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- report ---------------------------------------------------------------------------------

def _md(df: pd.DataFrame, floatfmt=".3g") -> str:
    """Markdown table (index as the first column)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)

    names = df.index.names if df.index.nlevels > 1 else [df.index.name or ""]
    head = [" / ".join(str(n) for n in names if n) or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, idx in enumerate(df.index):
        label = " / ".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        lines.append("| " + " | ".join([label] + [fmt(df[c].iloc[i]) for c in df.columns]) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    systems, comp = draw_systems(args.systems, seed=args.seed)
    t1 = time.time()
    pl = planets.attach_planets(systems, comp, seed=args.seed + 1)
    t2 = time.time()
    t_stars, t_planets = (t1 - t0) * 1e5 / args.systems, (t2 - t1) * 1e5 / args.systems

    occ = occurrence_by_type(comp, pl)
    figure_period_profiles(comp, pl, FIG / "period_profiles.png")
    figure_pr_grids(comp, pl, FIG / "period_radius_ms.png")
    ratio, z = kepler_comparison(systems, comp, pl)
    mdr = mdwarf_rates(systems, comp, pl)
    cls = cls_rates(comp, pl)
    corr, by_mass = giant_correlations(comp, pl)
    shares = flag_shares(pl)
    xmap = extrapolation_map(comp, pl)
    figure_extrapolation(xmap, FIG / "extrapolation_map.png")

    out = [
        "# Phase 4 report (generated)",
        "",
        f"Generated by `python -m starsys.validation.report --systems {args.systems}` on {date.today()}: "
        f"{args.systems:,} systems from the stellar population, {len(pl['period']):,} planets, all planet models "
        "at their defaults. Rates are intrinsic (no detection model). The verdicts and context are in "
        "[planets.md](planets.md).",
        "",
        f"Runtime per 100,000 systems: stars {t_stars:.1f} s, planets {t_planets:.1f} s (first use also builds the "
        "cached HM-C edge and binary boost tables).",
        "",
        "## Planets by host type (S-type, per star)",
        "",
        _md(occ),
        "",
        "![dN/d ln P by host type](figures/period_profiles.png)",
        "",
        "![Period-radius occurrence, main-sequence hosts](figures/period_radius_ms.png)",
        "",
        "## Kepler: inner planets vs the published HM-C catalogs",
        "",
        "Mock magnitude-limited Kepler sample (binaries included, Teff reweighted to HM-C's Kepler targets); "
        "published: catalogs 1-20 (mean; z uses their spread).",
        "",
        _md(ratio.attrs["totals"]),
        "",
        "Model / published, planets per star per cell:",
        "",
        _md(ratio, ".2f"),
        "",
        "z = (model - published) / posterior SD:",
        "",
        _md(z, ".1f"),
        "",
        "## Early M dwarfs (0.35-0.65 Msun, mock Kepler sample)",
        "",
        _md(mdr),
        "",
        f"## CLS: effectively single Sun-like stars ({cls.attrs['n']:,} stars, M sin i)",
        "",
        _md(cls),
        "",
        "## Giant-planet correlations (cold giants: 1-10 au, M sin i 0.5-20 MJ; single stars)",
        "",
        _md(corr),
        "",
        corr.attrs["note"] + ".",
        "",
        _md(by_mass),
        "",
        "## Where the model extrapolates",
        "",
        _md(shares, ".1f"),
        "",
        "Share (%) of planets with any extrapolation flag other than `young`, by host and semi-major axis "
        "(present-day orbits):",
        "",
        _md(xmap, ".0f"),
        "",
        "Planets per cell:",
        "",
        _md(xmap.attrs["counts"], ".0f"),
        "",
        "![Extrapolation map](figures/extrapolation_map.png)",
        "",
    ]
    DOC.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
