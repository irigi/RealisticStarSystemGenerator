"""Readers and transforms for zipped SysSim physical catalogs.

The code intentionally uses only the Python standard library so catalog
inspection and tests can run before installing ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import math
from pathlib import Path
import re
from typing import Iterable
import zipfile

from .constants import (
    DEFAULT_MAX_PLANETS,
    SOLAR_MASSES_TO_EARTH_MASSES,
    SOLAR_RADII_TO_EARTH_RADII,
)


@dataclass(frozen=True)
class Planet:
    """One intrinsic planet in Earth/day units."""

    period_days: float
    radius_earth: float
    mass_earth: float
    eccentricity: float
    cluster_id: int | None = None


@dataclass(frozen=True)
class StarSystem:
    """One target star and its ordered intrinsic planet list."""

    target_id: int
    star_id: int
    star_mass_solar: float
    star_radius_solar: float
    planets: tuple[Planet, ...]

    @property
    def num_planets(self) -> int:
        return len(self.planets)


@dataclass(frozen=True)
class EncodedSystem:
    """Model-ready transformed representation."""

    star_features: tuple[float, float]
    host_flag: int
    multiplicity: int
    # Planet rows: log_p1_or_positive_delta, log_radius, log_mass, eccentricity.
    planet_features: tuple[tuple[float, float, float, float], ...]


def parse_catalog_ids(spec: str) -> list[int]:
    """Parse comma-separated catalog ids and inclusive ranges, e.g. ``1-3,10``."""

    ids: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending catalog range: {token}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(token))
    if not ids:
        raise ValueError("No catalog ids were provided")
    return ids


def _clean_zip_names(names: Iterable[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name
        and not name.startswith("__MACOSX")
        and "/._" not in name
        and not Path(name).name.startswith("._")
    )


def _read_csv_from_zip(zip_path: Path, member_name: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Read a CSV member with leading ``# key: value`` metadata comments."""

    metadata: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        try:
            raw = archive.read(member_name)
        except KeyError as exc:
            available = ", ".join(_clean_zip_names(archive.namelist())[:8])
            raise FileNotFoundError(
                f"Missing {member_name!r} in {zip_path}. First available entries: {available}"
            ) from exc

    data_lines: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            text = line[1:].strip()
            if ":" in text:
                key, value = text.split(":", 1)
                metadata[key.strip()] = value.strip()
            continue
        data_lines.append(line)

    rows = list(csv.DictReader(io.StringIO("\n".join(data_lines))))
    return metadata, rows


def list_catalog_ids(catalog_root: Path | str) -> list[int]:
    """Return physical catalog ids present in ``physical_catalogs.zip``."""

    root = Path(catalog_root)
    zip_path = root / "physical_catalogs.zip"
    pattern = re.compile(r"physical_catalog(\d+)\.csv$")
    with zipfile.ZipFile(zip_path) as archive:
        ids = []
        for name in _clean_zip_names(archive.namelist()):
            match = pattern.search(name)
            if match:
                ids.append(int(match.group(1)))
    return sorted(ids)


def read_physical_catalog(
    catalog_root: Path | str,
    catalog_id: int,
    *,
    max_systems: int | None = None,
) -> tuple[dict[str, str], list[StarSystem]]:
    """Read one 2020 SysSim physical catalog realization.

    The star file provides zero-planet systems; the planet file provides one
    row per intrinsic planet. Planets are sorted by period for each system.
    """

    root = Path(catalog_root)
    star_zip = root / "physical_catalogs_stars.zip"
    planet_zip = root / "physical_catalogs.zip"
    star_member = f"physical_catalog_stars{catalog_id}.csv"
    planet_member = f"physical_catalog{catalog_id}.csv"

    metadata, star_rows = _read_csv_from_zip(star_zip, star_member)
    _, planet_rows = _read_csv_from_zip(planet_zip, planet_member)

    planets_by_system: dict[tuple[int, int], list[Planet]] = {}
    for row in planet_rows:
        key = (int(float(row["target_id"])), int(float(row["star_id"])))
        planet = Planet(
            period_days=float(row["period"]),
            radius_earth=float(row["planet_radius"]) * SOLAR_RADII_TO_EARTH_RADII,
            mass_earth=float(row["planet_mass"]) * SOLAR_MASSES_TO_EARTH_MASSES,
            eccentricity=float(row["ecc"]),
            cluster_id=int(float(row["clusterid"])) if row.get("clusterid") else None,
        )
        planets_by_system.setdefault(key, []).append(planet)

    systems: list[StarSystem] = []
    for row in star_rows:
        target_id = int(float(row["target_id"]))
        star_id = int(float(row["star_id"]))
        key = (target_id, star_id)
        planets = tuple(sorted(planets_by_system.get(key, []), key=lambda planet: planet.period_days))
        declared_count = int(float(row["num_planets"]))
        if declared_count != len(planets):
            raise ValueError(
                f"Catalog {catalog_id} count mismatch for target_id={target_id}, "
                f"star_id={star_id}: star file says {declared_count}, planet rows say {len(planets)}"
            )
        systems.append(
            StarSystem(
                target_id=target_id,
                star_id=star_id,
                star_mass_solar=float(row["star_mass"]),
                star_radius_solar=float(row["star_radius"]),
                planets=planets,
            )
        )
        if max_systems is not None and len(systems) >= max_systems:
            break

    return metadata, systems


def encode_system(system: StarSystem, *, max_planets: int = DEFAULT_MAX_PLANETS) -> EncodedSystem:
    """Transform one system into features used by the baseline model."""

    if system.num_planets > max_planets:
        planets = system.planets[:max_planets]
    else:
        planets = system.planets

    star_features = (
        math.log(system.star_mass_solar),
        math.log(system.star_radius_solar),
    )

    encoded_planets: list[tuple[float, float, float, float]] = []
    previous_log_period: float | None = None
    for planet in planets:
        log_period = math.log(planet.period_days)
        if previous_log_period is None:
            period_feature = log_period
        else:
            period_feature = max(log_period - previous_log_period, 1.0e-8)
        previous_log_period = log_period
        encoded_planets.append(
            (
                period_feature,
                math.log(max(planet.radius_earth, 1.0e-12)),
                math.log(max(planet.mass_earth, 1.0e-12)),
                min(max(planet.eccentricity, 0.0), 0.999999),
            )
        )

    return EncodedSystem(
        star_features=star_features,
        host_flag=1 if encoded_planets else 0,
        multiplicity=len(encoded_planets),
        planet_features=tuple(encoded_planets),
    )


def load_encoded_systems(
    catalog_root: Path | str,
    catalog_ids: Iterable[int],
    *,
    max_planets: int = DEFAULT_MAX_PLANETS,
    max_systems_per_catalog: int | None = None,
) -> list[EncodedSystem]:
    """Load and encode multiple whole catalog realizations."""

    encoded: list[EncodedSystem] = []
    for catalog_id in catalog_ids:
        _, systems = read_physical_catalog(
            catalog_root,
            catalog_id,
            max_systems=max_systems_per_catalog,
        )
        encoded.extend(encode_system(system, max_planets=max_planets) for system in systems)
    return encoded


def summarize_systems(systems: Iterable[StarSystem]) -> dict[str, float]:
    """Compute basic catalog diagnostics."""

    systems_list = list(systems)
    total_stars = len(systems_list)
    host_count = sum(system.num_planets > 0 for system in systems_list)
    planet_count = sum(system.num_planets for system in systems_list)
    max_planets = max((system.num_planets for system in systems_list), default=0)
    return {
        "stars": float(total_stars),
        "planet_hosts": float(host_count),
        "host_fraction": host_count / total_stars if total_stars else 0.0,
        "planets": float(planet_count),
        "planets_per_star": planet_count / total_stars if total_stars else 0.0,
        "planets_per_host": planet_count / host_count if host_count else 0.0,
        "max_planets": float(max_planets),
    }

