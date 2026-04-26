from pathlib import Path

from star_system_generator.catalog import (
    encode_system,
    list_catalog_ids,
    parse_catalog_ids,
    read_physical_catalog,
    summarize_systems,
)


CATALOG_ROOT = Path("catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS")


def test_parse_catalog_ids():
    assert parse_catalog_ids("1-3,10") == [1, 2, 3, 10]


def test_read_local_catalog_sample():
    assert 1 in list_catalog_ids(CATALOG_ROOT)
    metadata, systems = read_physical_catalog(CATALOG_ROOT, 1, max_systems=100)
    assert metadata["num_targets_sim_pass_one"] == "88912"
    assert len(systems) == 100

    summary = summarize_systems(systems)
    assert summary["stars"] == 100.0
    assert summary["max_planets"] >= 0.0

    encoded = [encode_system(system) for system in systems]
    assert len(encoded) == 100
    assert all(len(item.star_features) == 2 for item in encoded)
    assert all(item.host_flag in (0, 1) for item in encoded)


if __name__ == "__main__":
    test_parse_catalog_ids()
    test_read_local_catalog_sample()
