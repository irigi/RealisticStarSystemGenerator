"""Download the star catalogues used by :mod:`starsys.real` from VizieR into
``catalogs/observational/stars`` (``python -m starsys.real.fetch``).

- Bright Star Catalogue, 5th ed. (Hoffleit & Warren 1991; V/50)
- PASTEL (Soubiran et al. 2016; B/pastel)
- Washington Double Star Catalog (Mason et al. 2001; B/wds)

The 20-pc census and the NASA Exoplanet Archive tables are fetched with the rest of the
project's catalogues (see ``catalogs/``).
"""

from __future__ import annotations

import datetime

from .catalog import BSC_FILE, PASTEL_FILE, WDS_FILE

CATALOGS = {BSC_FILE: "V/50/catalog", PASTEL_FILE: "B/pastel/pastel", WDS_FILE: "B/wds/wds"}


def fetch(overwrite: bool = False) -> None:
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=-1, columns=["**"])
    for path, cat in CATALOGS.items():
        if path.exists() and not overwrite:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        table = v.get_catalogs(cat)[0]
        table.write(path, overwrite=True)
        print(f"{cat}: {len(table)} rows -> {path}")
    (BSC_FILE.parent / "RETRIEVED_DATE").write_text(datetime.date.today().isoformat() + "\n")


if __name__ == "__main__":
    import sys

    fetch(overwrite="--overwrite" in sys.argv)
