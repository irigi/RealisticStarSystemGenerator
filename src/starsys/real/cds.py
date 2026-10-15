"""Read CDS/VizieR fixed-width tables with their ReadMe ("Byte-by-byte Description" blocks) into
pandas, tolerating blank and "---" fields (astropy's CDS reader does not)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_COL = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s+([AIFE]\d+(?:\.\d+)?)\s+\S+\s+(\S+)")


def byte_layout(readme: Path, filename: str) -> list[tuple[int, int, str, str]]:
    """(start, end, format, label) of each column of ``filename`` (0-based, end exclusive)."""
    text = Path(readme).read_text(errors="replace").splitlines()
    out, inside = [], False
    for line in text:
        if line.startswith("Byte-by-byte Description of file:"):
            names = line.split(":", 1)[1].split()
            inside = filename in names or any(filename.startswith(n.rstrip("*").rstrip(".")) for n in names)
            continue
        if not inside:
            continue
        if out and (line.startswith("-----") and len(out) > 0 and _seen_body(out)):
            break
        m = _COL.match(line)
        if m:
            start = int(m.group(1)) - 1
            end = int(m.group(2) or m.group(1))
            out.append((start, end, m.group(3), m.group(4)))
    return out


def _seen_body(cols) -> bool:
    return len(cols) > 0


def read(readme: Path, datafile: Path) -> pd.DataFrame:
    name = Path(datafile).name
    base = name[:-3] if name.endswith(".gz") else name
    layout = byte_layout(readme, base)
    if not layout:
        raise ValueError(f"no byte-by-byte description of {base} in {readme}")
    colspecs = [(a, b) for a, b, _, _ in layout]
    names = [lab for _, _, _, lab in layout]
    d = pd.read_fwf(datafile, colspecs=colspecs, names=names, header=None, dtype=str,
                    compression="gzip" if name.endswith(".gz") else None)
    for (_, _, fmt, lab) in layout:
        if fmt[0] in "IFE":
            d[lab] = pd.to_numeric(d[lab].str.strip().replace({"---": None, "--": None, "": None}), errors="coerce")
        else:
            d[lab] = d[lab].fillna("").str.strip()
    return d
