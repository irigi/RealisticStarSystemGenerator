"""Complete star systems in one call: stars, planets (with derived properties) and moons.

:func:`generate` runs the whole pipeline (:func:`starsys.stars.population.draw_systems`,
:func:`starsys.planets.attach_planets`, :func:`starsys.moons.attach_moons`) and returns a
:class:`Systems` bundle, the common input of :mod:`starsys.describe` (readable summaries) and
:mod:`starsys.export` (JSON, CSV, Parquet). Systems conditioned on a real star
(:func:`starsys.real.condition.condition`) convert with :meth:`Systems.from_conditioned`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import moons as mn
from . import planets as pt
from .stars.population import draw_systems


@dataclass
class Systems:
    """The four tables of a set of systems (dicts of equal-length arrays, see the modules that
    make them) and how they were made.

    ``components`` rows are sorted by system and role; ``planets`` by system, component, host
    kind and period; ``moons`` reference their planet by row (``planet``).
    """

    systems: dict
    components: dict
    planets: dict
    moons: dict
    meta: dict = field(default_factory=dict)  # generator settings, seed, real-star information
    star_names: dict[int, str] = field(default_factory=dict)  # role -> name (real stars)
    target: np.ndarray | None = None  # component row of the real star in each system

    def __len__(self) -> int:
        return len(self.systems["age_gyr"])

    def rows(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Component, planet and moon rows of system ``i``."""
        comp = np.flatnonzero(np.asarray(self.components["system"]) == i)
        pl = np.flatnonzero(np.asarray(self.planets["system"]) == i)
        mo = (np.flatnonzero(np.asarray(self.moons["system"]) == i) if self.moons
              else np.zeros(0, dtype=np.int64))
        return comp, pl, mo

    @classmethod
    def from_conditioned(cls, c) -> "Systems":
        """Bundle of a :class:`starsys.real.condition.ConditionedSystems`."""
        rec = c.record
        meta = {"real_star": rec.name, "target": rec.star.name, "known_planets": [k.name for k in c.known],
                "surveys": list(c.surveys), "effective_sample_size": round(float(c.ess), 1),
                "inserted_fraction": {k: round(float(v), 3) for k, v in c.inserted_fraction.items()}}
        names = {j: comp.name for j, comp in enumerate(rec.components)}
        return cls(c.systems, c.components, c.planets, c.moons or {}, meta, names, np.asarray(c.target))


def generate(n: int, seed=None, *, planet_kwargs: dict | None = None, moons: bool = True,
             population=None) -> Systems:
    """``n`` random star systems with their planets and moons.

    ``planet_kwargs`` go to :func:`starsys.planets.attach_planets` (model settings, e.g.
    ``temperate_bare_fraction``); ``population`` is a
    :class:`starsys.stars.population.PopulationModel` (default: the calibrated local one).
    """
    rng = np.random.default_rng(seed)
    systems, components = draw_systems(n, seed=rng, model=population)
    planets = pt.attach_planets(systems, components, seed=rng, **(planet_kwargs or {}))
    moon_table = mn.attach_moons(planets, components, seed=rng) if moons else {}
    meta = {"seed": seed if isinstance(seed, (int, np.integer)) else None,
            "settings": {k: repr(v) for k, v in (planet_kwargs or {}).items()}}
    return Systems(systems, components, planets, moon_table, meta)


__all__ = ["Systems", "generate"]
