"""Readers for the He & Ford SysSim simulated catalogs (HM-C, Paper IV).

Each model directory holds 100 "physical catalogs" (the true planetary systems
around a simulated Kepler stellar sample) and the matching "observed catalogs"
(what a simulated Kepler would detect). Every catalog was generated with a
different draw of the model parameters from the ABC posterior; those parameters
are written in a ``# key: value`` header at the top of each CSV.

``star_id`` is a 1-based row index into the stellar table the model was run on;
``target_id`` is the 1-based index of the simulated target within the catalog.
Stars without planets are absent from ``physical_catalogN.csv`` but present (with
``num_planets == 0``) in ``physical_catalog_starsN.csv``.

Only HM-C is used; readers for the earlier papers' catalogs (H20, Papers I-II) and a parquet
cache were removed after phase 5 (recoverable from git, commit d7006df).
"""

from __future__ import annotations

import ast
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CATALOGS_DIR
from .units import EARTH_MASS_MSUN, EARTH_RADIUS_RSUN


@dataclass(frozen=True)
class SysSimModel:
    """Location and metadata of one published set of 100 catalogs."""

    name: str
    directory: str
    stellar_table: str
    paper: str

    @property
    def path(self) -> Path:
        return CATALOGS_DIR / self.directory


MODELS: dict[str, SysSimModel] = {
    m.name: m
    for m in [
        SysSimModel("hmc", "He_Ford_2026_HM-C/depth_min_0.29", "q1_q17_dr25_gaia_berger_fgk_HFR2020b_cleaned.csv",
                    "He & Ford 2026 (Paper IV), hybrid model HM-C"),
    ]
}

N_CATALOGS = 100

_HEADER_RE = re.compile(r"^#\s*(?P<key>[^:]+?)\s*:\s*(?P<value>.*?)\s*$")


def _parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_header(lines: list[str]) -> dict:
    """Parse ``# key: value`` lines into a dict.

    Keys may repeat (e.g. two ``mr_model`` lines); repeated values are joined
    into a list. Unit annotations such as ``(R_earth)`` are kept in the key.
    """
    params: dict = {}
    for line in lines:
        match = _HEADER_RE.match(line)
        if not match:
            continue
        key, value = match["key"], _parse_value(match["value"])
        if key in params:
            prev = params[key]
            params[key] = (prev if isinstance(prev, list) else [prev]) + [value]
        else:
            params[key] = value
    return params


def _split_header(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    n_header = 0
    while n_header < len(lines) and lines[n_header].startswith("#"):
        n_header += 1
    return parse_header(lines[:n_header]), "".join(lines[n_header:])


def _read_member(model: SysSimModel, archive: str, member: str) -> str:
    with zipfile.ZipFile(model.path / archive) as zf:
        return zf.read(member).decode()


def _read_table(model: SysSimModel, archive: str, member: str) -> tuple[dict, pd.DataFrame]:
    params, body = _split_header(_read_member(model, archive, member))
    return params, pd.read_csv(io.StringIO(body))


def available_catalog_ids(model: SysSimModel) -> list[int]:
    with zipfile.ZipFile(model.path / "physical_catalogs.zip") as zf:
        names = zf.namelist()
    ids = {
        int(m.group(1))
        for name in names
        if "__MACOSX" not in name and (m := re.fullmatch(r"physical_catalog(\d+)\.csv", Path(name).name))
    }
    return sorted(ids)


def read_physical_catalog(model: SysSimModel, catalog_id: int) -> tuple[dict, pd.DataFrame]:
    """Return (parameters, planets) with masses/radii converted to Earth units."""
    params, df = _read_table(model, "physical_catalogs.zip", f"physical_catalog{catalog_id}.csv")
    df["planet_mass"] = df["planet_mass"] / EARTH_MASS_MSUN
    df["planet_radius"] = df["planet_radius"] / EARTH_RADIUS_RSUN
    for col in ("target_id", "star_id", "clusterid"):
        df[col] = df[col].astype(np.int32)
    return params, df


def read_physical_header(model: SysSimModel, catalog_id: int) -> dict:
    """Model parameters of one catalog, reading only the CSV header."""
    member = f"physical_catalog{catalog_id}.csv"
    lines = []
    with zipfile.ZipFile(model.path / "physical_catalogs.zip") as zf, zf.open(member) as fh:
        for raw in fh:
            line = raw.decode()
            if not line.startswith("#"):
                break
            lines.append(line)
    return parse_header(lines)


def read_physical_stars(model: SysSimModel, catalog_id: int) -> pd.DataFrame | None:
    """Per-target table including stars with zero planets (None if not published)."""
    if not (model.path / "physical_catalogs_stars.zip").exists():
        return None
    _, df = _read_table(model, "physical_catalogs_stars.zip", f"physical_catalog_stars{catalog_id}.csv")
    for col in ("target_id", "star_id", "num_planets"):
        df[col] = df[col].astype(np.int32)
    return df


def read_stellar_table(name: str) -> pd.DataFrame:
    """Stellar input table with a 1-based ``star_id`` column matching the catalogs."""
    df = pd.read_csv(CATALOGS_DIR / "stellar_tables" / name)
    df.insert(0, "star_id", np.arange(1, len(df) + 1, dtype=np.int32))
    return df

