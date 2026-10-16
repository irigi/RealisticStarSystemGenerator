import numpy as np
import pytest

from starsys import ids
from starsys import describe as ds


def _same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.array_equal(x, y, equal_nan=x.dtype.kind == "f")


def _equal(a, b):
    return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)


def test_same_id_same_system_regardless_of_history():
    a = ids.system("S7-70")
    ids._block.cache_clear()
    ids.system("S7-3")  # other block, other state
    b = ids.system(ids.SeqID(7, 70))
    for t in ("systems", "components", "planets", "moons"):
        assert _equal(getattr(a, t), getattr(b, t)), t
    assert "S7-70" in ds.system_text(a, 0)


def test_stars_only_match_full_system():
    s, c = ids.stars_of(3, 10)
    full = ids.system("S3-10")
    assert _equal(c, full.components) and _equal(s, full.systems)


def test_neighbours_differ_and_parse():
    a, b = ids.system("S1-0", moons=False), ids.system("S1-1", moons=False)
    assert a.systems["primary_initial_mass"][0] != b.systems["primary_initial_mass"][0]
    assert ids.parse(" S12-345 ") == ids.SeqID(12, 345)
    with pytest.raises(ValueError):
        ids.parse("12-345")
    assert ids.mix(1, 2) != ids.mix(2, 1) and ids.mix(-1) != ids.mix(1)
