"""Filesystem locations used across the package."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGS_DIR = Path(os.environ.get("STARSYS_CATALOGS", REPO_ROOT / "catalogs"))
DATA_DIR = Path(os.environ.get("STARSYS_DATA", REPO_ROOT / "data"))
