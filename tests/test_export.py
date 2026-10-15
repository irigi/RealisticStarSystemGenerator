import json

import numpy as np
import pandas as pd
import pytest

from starsys import derived as dv
from starsys import describe as ds
from starsys import export as ex
from starsys.__main__ import main, select
from starsys.generate import generate


@pytest.fixture(scope="module")
def bundle():
    return generate(150, seed=11)


def _planets(t_eq, hz, mass, radius, lost):
    n = len(t_eq)
    return {"t_eq": np.array(t_eq, float), "hz": np.array(hz, np.int8), "planet_mass": np.array(mass, float),
            "planet_radius": np.array(radius, float), "lost_envelope": np.array(lost, bool)}


def test_surface_temperature_of_solar_system_analogues():
    # Earth (255 K, conservative HZ), Venus-like (too hot), Mars-mass, cold super-Earth, sub-Neptune.
    p = _planets([255.0, 299.0, 210.0, 150.0, 255.0],
                 [dv.HZ_CONSERVATIVE, dv.HZ_TOO_HOT, dv.HZ_OPT_OUTER, dv.HZ_TOO_COLD, dv.HZ_CONSERVATIVE],
                 [1.0, 0.8, 0.05, 3.0, 8.0], [1.0, 0.95, 0.4, 1.4, 2.5], [False, False, False, False, False])
    t, code = ds.surface_temperature(p)
    assert t[0] == pytest.approx(288.0)
    assert t[1] == pytest.approx(737.0, rel=1e-3)
    assert code[2] == ds.CLIMATE_AIRLESS and t[2] == pytest.approx(210.0)
    assert code[3] == ds.CLIMATE_FROZEN and t[3] < 273
    assert code[4] == ds.CLIMATE_NO_SURFACE and np.isnan(t[4])
    # Carbonate-silicate floor inside the habitable zone.
    p = _planets([200.0], [dv.HZ_CONSERVATIVE], [1.0], [1.0], [False])
    assert ds.surface_temperature(p)[0][0] == pytest.approx(273.0)


def test_planet_classes():
    p = _planets([1500.0, 255.0, 60.0, 900.0], [dv.HZ_TOO_HOT, dv.HZ_CONSERVATIVE, dv.HZ_TOO_COLD, dv.HZ_TOO_HOT],
                 [318.0, 1.0, 17.0, 5.0], [11.2, 1.0, 3.9, 2.4], [False, False, False, False])
    assert list(ds.planet_class(p)) == ["hot Jupiter", "temperate Earth-size", "cold ice giant", "warm sub-Neptune"]


def test_names_are_unique_and_ordered(bundle):
    for i in range(len(bundle)):
        cn, pn, mn = ds.names(bundle, i)
        assert len(set(pn.values())) == len(pn)
        assert len(set(mn.values())) == len(mn)
        _, pl, _ = bundle.rows(i)
        for host in np.unique(bundle.planets["component"][pl]):
            rows = [r for r in pl if bundle.planets["component"][r] == host and bundle.planets["host_kind"][r] == "S"]
            letters = [pn[r].split()[-1] for r in sorted(rows, key=lambda r: bundle.planets["period"][r])]
            assert letters == sorted(letters)


def test_text_summary(bundle):
    n_pl = np.bincount(bundle.planets["system"], minlength=len(bundle))
    i = int(np.argmax(n_pl))
    text = ds.system_text(bundle, i)
    _, pn, _ = ds.names(bundle, i)
    assert all(name in text for name in pn.values())
    assert "no greenhouse" in text and "illustrative" in text


def test_json_round_trip(bundle, tmp_path):
    path = ex.write_json(bundle, tmp_path / "s.json")
    d = json.loads(path.read_text())
    assert d["schema"] == ex.SCHEMA and len(d["systems"]) == len(bundle)
    n_pl = sum(len(s["planets"]) for sy in d["systems"] for s in sy["stars"]) + \
        sum(len(sy["circumbinary_planets"]) for sy in d["systems"])
    assert n_pl == len(bundle.planets["a_au"])
    n_moons = sum(len(p["moons"]) for sy in d["systems"] for s in sy["stars"] for p in s["planets"]) + \
        sum(len(p["moons"]) for sy in d["systems"] for p in sy["circumbinary_planets"])
    assert n_moons == len(bundle.moons["mass"])
    sy = next(s for s in d["systems"] if s["n_stars"] > 1)
    assert sy["stars"][1]["orbit"]["around"] == ["A"]
    p = next(p for sy in d["systems"] for s in sy["stars"] for p in s["planets"])
    assert {"name", "class", "t_eq_k", "t_surface_illustrative_k", "climate", "moons"} <= set(p)


def test_tables(bundle, tmp_path):
    paths = ex.write_tables(bundle, tmp_path, "csv")
    assert {p.name for p in paths} == {"systems.csv", "components.csv", "planets.csv", "moons.csv", "units.json"}
    pl = pd.read_csv(tmp_path / "planets.csv")
    assert len(pl) == len(bundle.planets["a_au"]) and pl["name"].notna().all()
    assert {"class", "climate", "t_surface_illustrative"} <= set(pl.columns)


def test_selection_and_cli(bundle, tmp_path, capsys):
    g = select(bundle, spectral="G")
    primary = np.searchsorted(bundle.components["system"], g)
    assert all(str(t).startswith("G") for t in bundle.components["spectral_type"][primary])
    hab = select(bundle, habitable=True)
    assert set(hab) <= set(range(len(bundle)))
    main(["-n", "20", "--seed", "3", "--show", "1", "--json", str(tmp_path / "x.json")])
    out = capsys.readouterr().out
    assert "System" in out and "wrote" in out
