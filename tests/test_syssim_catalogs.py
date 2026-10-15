import pytest

from starsys.syssim_catalogs import MODELS, parse_header


def test_parse_header_types_and_repeats():
    lines = [
        "# num_targets_sim_pass_one: 86760\n",
        "# power_law_P: -0.158\n",
        "# mr_model: Ning2018_table (above radius_switch)\n",
        "# mr_model: Lognormal mass around Earthlike rocky (below radius_switch)\n",
        "# period_ratios_mmr: [2.0, 1.5]\n",
        "#\n",
    ]
    h = parse_header(lines)
    assert h["num_targets_sim_pass_one"] == 86760
    assert h["power_law_P"] == pytest.approx(-0.158)
    assert h["period_ratios_mmr"] == [2.0, 1.5]
    assert isinstance(h["mr_model"], list) and len(h["mr_model"]) == 2


def test_local_catalogs_read():
    from starsys.syssim_catalogs import read_physical_catalog

    model = MODELS["hmc"]
    if not model.path.exists():
        pytest.skip("catalog not downloaded")
    params, df = read_physical_catalog(model, 1)
    assert df["planet_radius"].between(0.3, 10 + 1e-6).all()  # stripped cores go below 0.5
    assert df["period"].between(3, 300).all()
    assert params["min_period"] == 3.0
