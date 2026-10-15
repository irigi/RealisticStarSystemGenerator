"""Stable system identifiers: every system is a pure function of a short ID.

The pipeline is vectorised, so a system drawn inside a batch depends on the whole batch.
To make single systems addressable (the viewer's Previous / Next, the star map's click on a
star), stars are drawn in small fixed *blocks* whose seed comes from the block's address,
and the planets and moons of one star are drawn alone from a seed derived from the block
seed and the star's position in it. So

* the star map can draw a block's stars only (cheap, ~16 µs per star) and the planets of one
  star when it is clicked, with the same result as drawing the full system directly;
* the same ID always gives the identical system, whatever else was generated before.

IDs are strings:

``S<master>-<k>``
    System ``k`` (0, 1, 2, ...) of the sequence with master seed ``master``; blocks of
    :data:`BLOCK` systems, block seed = mix(master, k // BLOCK).
``G<ix>.<iy>.<iz>-<j>``
    Star ``j`` of galaxy sector ``(ix, iy, iz)`` (see :mod:`starsys.galaxy`), whose block is
    drawn by the galaxy model; this module only supplies the planets for it.

Seeds are mixed with :class:`numpy.random.SeedSequence` (hash of a tuple of non-negative
integers), stable across numpy versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import moons as mn
from . import planets as pt
from .generate import Systems
from .stars.population import draw_systems

BLOCK = 64
STREAM_PLANETS, STREAM_MOONS = 1, 2


def mix(*keys: int) -> int:
    """A 63-bit seed from a tuple of integers (negative ones are mapped to odd numbers)."""
    ent = [2 * k if k >= 0 else -2 * k - 1 for k in map(int, keys)]
    a, b = np.random.SeedSequence(ent).generate_state(2, np.uint32)
    return (int(a) << 31) ^ int(b)


@dataclass(frozen=True)
class SeqID:
    master: int
    k: int

    def __str__(self) -> str:
        return f"S{self.master}-{self.k}"


_SEQ = re.compile(r"^S(\d+)-(\d+)$")


def parse(text: str) -> SeqID:
    m = _SEQ.match(text.strip())
    if not m:
        raise ValueError(f"not a system ID: {text!r} (expected e.g. S1-42)")
    return SeqID(int(m.group(1)), int(m.group(2)))


@lru_cache(maxsize=64)
def _block(master: int, b: int):
    return draw_systems(BLOCK, seed=mix(master, b))


def stars_of(master: int, k: int) -> tuple[dict, dict]:
    """Stars of sequence system ``k`` (system tables with one row, components renumbered)."""
    s, c = _block(master, k // BLOCK)
    return subset(s, c, k % BLOCK)


def subset(systems: dict, components: dict, j: int) -> tuple[dict, dict]:
    """System ``j`` of a star batch as a one-system batch."""
    sel = np.flatnonzero(components["system"] == j)
    s1 = {k: np.asarray(v)[j:j + 1].copy() for k, v in systems.items()}
    c1 = {k: np.asarray(v)[sel].copy() for k, v in components.items()}
    c1["system"][:] = 0
    return s1, c1


def complete(systems: dict, components: dict, seed: int, *, moons: bool = True,
             planet_kwargs: dict | None = None, meta: dict | None = None) -> Systems:
    """Planets and moons for a one-system star batch, from ``seed`` alone."""
    planets = pt.attach_planets(systems, components, seed=mix(seed, STREAM_PLANETS), **(planet_kwargs or {}))
    moon_table = mn.attach_moons(planets, components, seed=mix(seed, STREAM_MOONS)) if moons else {}
    return Systems(systems, components, planets, moon_table, dict(meta or {}))


def system(sid: SeqID | str, *, moons: bool = True, planet_kwargs: dict | None = None) -> Systems:
    """The complete system with this ID."""
    sid = parse(sid) if isinstance(sid, str) else sid
    s, c = stars_of(sid.master, sid.k)
    return complete(s, c, mix(sid.master, sid.k // BLOCK, sid.k % BLOCK), moons=moons,
                    planet_kwargs=planet_kwargs, meta={"ids": [str(sid)]})


__all__ = ["BLOCK", "SeqID", "complete", "mix", "parse", "stars_of", "subset", "system"]
