"""Compare the stellar population model with the 20-pc census (Kirkpatrick et al. 2024).

The component mass function is matched by construction (calibration), so the
informative checks are the ones that depend on ages, evolution and multiplicity:
class fractions (spectral class, brown-dwarf subclasses, white dwarfs, giants),
the white-dwarf fraction as a function of progenitor mass, and the multiplicity
fraction as a function of primary mass.

Usage: ``python -m starsys.validation.census``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.table import Table

from ..stars.population import CENSUS_DIR, draw_systems
from ..stars.spectral import MAMAJEK_FILE, temperature_class

CLASSES = ["B", "A", "F", "G", "K", "M", "L", "T", "Y", "WD", "giant (III)", "subgiant (IV)"]


def _mass_to_class(mass: np.ndarray) -> np.ndarray:
    """Main-sequence spectral class letter for a mass, from the Mamajek table's Msun column."""
    rows = []
    for line in MAMAJEK_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[0].endswith("V") and not line.startswith("#"):
            try:
                rows.append((parts[0][0], float(parts[-2])))
            except ValueError:
                pass
        if parts and parts[0] == "Y4V":
            break
    letters = np.array([r[0] for r in rows])
    masses = np.array([r[1] for r in rows])
    idx = np.abs(np.log10(mass)[:, None] - np.log10(masses)[None, :]).argmin(axis=1)
    return letters[idx]


def load_census() -> pd.DataFrame:
    t = Table.read(CENSUS_DIR / "J_ApJS_271_55_table4.ecsv").to_pandas()
    t["system"] = t["Syst"].astype(str).str.split(".").str[0]
    leaf = t[t.NcTR == 1].copy()
    sp = leaf.SpTOpt.fillna("").astype(str).str.strip()
    sp = sp.where(sp != "", leaf.SpTNIR.fillna("").astype(str).str.strip())
    leaf["sptype"] = sp
    is_wd = leaf.n_IMass.astype(str).str.startswith("wd")
    giant = sp.str.contains(r"III|II", regex=True) & ~is_wd
    subgiant = sp.str.contains(r"IV", regex=True) & ~giant & ~is_wd
    letter = sp.str[0].where(sp.str[0].isin(list("OBAFGKMLTY")), "?")
    # Stars without a published spectral type: class from the adopted initial mass.
    untyped = (letter == "?") & (leaf.IMass > 0)
    letter[untyped] = _mass_to_class(leaf.IMass[untyped].to_numpy())
    # Brown dwarfs without a spectral type but with a Teff estimate.
    bd_teff = (letter == "?") & leaf.Teff.notna()
    letter[bd_teff] = [t[0] for t in temperature_class(leaf.Teff[bd_teff].to_numpy())]
    leaf["cls"] = np.where(is_wd, "WD", np.where(giant, "giant (III)", np.where(subgiant, "subgiant (IV)", letter)))
    leaf["mass"] = leaf.IMass.where(leaf.IMass > 0, 0.05)  # BDs: mass unknown, placeholder
    leaf["is_wd"] = is_wd
    return leaf


def classify_model(comp: dict) -> pd.DataFrame:
    df = pd.DataFrame({k: comp[k] for k in ("system", "kind", "initial_mass", "spectral_type")})
    sp = df.spectral_type.astype(str)
    star = df.kind == "star"
    giant = star & sp.str.contains(r"(?:III|II|I)$", regex=True) & ~sp.str.endswith("IV")
    subgiant = star & sp.str.endswith("IV")
    letter = sp.str[0]
    df["cls"] = np.where(df.kind == "WD", "WD", np.where(giant, "giant (III)",
                                                       np.where(subgiant, "subgiant (IV)", letter)))
    df.loc[df.kind.isin(["NS", "BH"]), "cls"] = "NS/BH"
    df["mass"] = df.initial_mass
    df["is_wd"] = df.kind == "WD"
    return df


def compare(n_systems=200_000, seed=0) -> dict[str, pd.DataFrame]:
    census = load_census()
    _, comp = draw_systems(n_systems, seed=seed)
    model = classify_model(comp)

    known = census[census.cls != "?"]
    frac = pd.DataFrame({
        "census_N": known.cls.value_counts().reindex(CLASSES).fillna(0).astype(int),
        "census_frac": known.cls.value_counts(normalize=True).reindex(CLASSES).fillna(0),
        "model_frac": model.cls.value_counts(normalize=True).reindex(CLASSES).fillna(0),
    })
    frac["census_unclassified"] = (census.cls == "?").sum()

    bins = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 8.0]
    wd = pd.DataFrame({
        "census_N": census.groupby(pd.cut(census.mass, bins), observed=False).size(),
        "census_wd_frac": census.groupby(pd.cut(census.mass, bins), observed=False).is_wd.mean(),
        "model_wd_frac": model.groupby(pd.cut(model.mass, bins), observed=False).is_wd.mean(),
    })

    mbins = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3, 1.8, 3, 8]

    def mult(df):
        g = df.groupby("system")
        s = pd.DataFrame({"n": g.size(), "m1": g.mass.max()})
        return s.groupby(pd.cut(s.m1, mbins), observed=False).n.agg(N="size", mf=lambda x: (x > 1).mean())

    c, m = mult(census), mult(model)
    multiplicity = pd.DataFrame({"census_N": c.N, "census_mf": c.mf, "model_mf": m.mf})
    return {"class_fractions": frac, "wd_fraction_by_progenitor_mass": wd, "multiplicity_by_primary_mass": multiplicity}


def main() -> None:
    pd.set_option("display.width", 200)
    for name, table in compare().items():
        print(f"\n== {name}")
        print(table.round(4).to_string())


if __name__ == "__main__":
    main()
