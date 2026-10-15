import numpy as np
import pytest

from starsys.real import catalog as ct
from starsys.real import names, star
from starsys.stars import population as pop

FAST = star.StarFitModel(n_age=1500, n_mass=120)


def test_name_keys():
    assert names.key("alpha Centauri A") == names.key("alf Cen A") == "alfcena"
    assert names.key("Gliese 581") == names.key("GJ 581") == "gj581"
    assert names.key("Gaia DR3 4472832130942575872") == names.key("Gaia EDR3 4472832130942575872")
    assert names.key("Barnard's Star") == names.key("Barnard's star")
    assert names.key("omicron02 Eri") == names.key("omicron2 Eridani")
    assert names.bsc_name_key("Alp1Cen") == "alf1cen" and names.bsc_name_key("58Alp Lyr") == "alflyr"


def test_lookup_systems():
    r = ct.find("Proxima Centauri", use_gaia=False)
    assert [c.name for c in r.components] == ["alpha Cen A", "alpha Cen B", "Proxima Centauri"]
    assert r.star.name == "Proxima Centauri" and r.star.nasa_host == "Proxima Cen"
    r = ct.find("Sirius B", use_gaia=False)
    assert r.star.kind_hint == "WD" and r.separations[1][1] == "WDS"
    assert ct.find("Kepler-90", use_gaia=False).star.nasa_host == "KOI-351"  # alias through planet names
    # 40 Eri: the close B-C pair first, A orbiting it (census hierarchy 477.1, 477.2.1, 477.2.2).
    r = ct.find("40 Eri A", use_gaia=False)
    assert [c.name[-1] for c in r.components] == ["B", "C", "A"] and r.star.name.endswith("A")
    assert r.separations[1][0] == pytest.approx(7.7, abs=0.5)
    with pytest.raises(KeyError):
        ct.find("No Such Star 12345", use_gaia=False)


def test_age_feh_prior_normalised():
    model = pop.default_model()
    a = np.linspace(0.0, 13.0, 1301)
    f = np.linspace(-2.0, 1.0, 601)
    dens = star.age_feh_prior(a[:, None], f[None, :], model)
    assert dens.sum() * (a[1] - a[0]) * (f[1] - f[0]) == pytest.approx(1.0, abs=0.01)


def test_fit_sirius_and_alpha_cen():
    f = star.realise("Sirius B", n=400, seed=1, use_gaia=False, model=FAST)
    c = f.components
    assert (c["kind"][f.target] == "WD").mean() > 0.95
    assert 0.1 < np.median(f.systems["age_gyr"]) < 0.4  # Sirius A's isochrone and B's cooling agree
    assert np.median(c["mass"][f.target]) == pytest.approx(1.018, rel=0.05)
    f = star.realise("alpha Centauri A", n=400, seed=1, use_gaia=False, model=FAST)
    c = f.components
    assert np.median(c["mass"][f.target]) == pytest.approx(1.079, rel=0.03)
    assert c["role"][f.target].max() == 0 and len(c["mass"]) == 3 * 400
    b = f.target + 1
    assert np.median(c["a_au"][b]) == pytest.approx(16.5 * 1.332, rel=0.6)  # deprojected 16.5" at 1.33 pc


def test_fit_giant():
    f = star.realise("Arcturus", n=300, seed=2, use_gaia=False, model=FAST)
    r = f.components["radius"][f.target]
    assert np.median(r) == pytest.approx(25.4, rel=0.25)
    assert f.ess > 100
