# Night run plan (phase 7: viewer, star map, two fixes)

Branch `phase7-viewer` (from `phase6` = `main` = origin/main at 0d3fad6). Commit after every
finished item with tests passing; never push. Tick items here and log every decision taken
without the user under "Decisions" with its reason. If a run is resumed, start at the first
unticked item.

## User decisions (2026-10-15)

- GUI in **PySide6** (installed in `.venv`, with pyqtgraph and PyOpenGL; optional extra
  `viewer` in pyproject). Not HTML.
- System viewer: Pioneer-style hierarchical system map + true-separation orbit view + click-for-info
  panel; **Next / Previous** buttons over a fixed pseudorandom seed sequence, so going back
  returns the identical system; "go to ID".
- **Procedurally generated planet artwork** in the spirit of Pioneer's system map
  (https://img.itch.zone/aW1nLzgwNzY5NDUucG5n/original/OJgdR3.png), improved where possible,
  with interesting variability within each planet class (deterministic from the body's seed).
- **One galaxy** (the Milky Way, Sol at its real place), clickable, rotatable, searchable,
  with spiral arms, real relative star abundances and real local density. Study how Frontier
  (Elite 2), Pioneer and Space Engine do it first (web search approved).
- **Real stars blended, not split by distance**: place the real catalogue stars, generate the
  procedural population everywhere, and delete each procedural star with the probability that
  the catalogue would contain such a star (selection function of apparent magnitude, and of
  distance for volume-complete censuses). Procedural stars thus only fill the catalogue's
  incompleteness: nothing within the 20-pc census, mostly faint M dwarfs around a bright distant
  star like those listed by Hipparcos. No minimum-separation rule: field stars are Poisson
  distributed (bound multiples already live inside one system).
- Fix A10 (known planets of non-target components) and review the extreme moon tidal heating.
- My defaults accepted: priority order below; Milky Way full model, but the neighbourhood must
  work first.

## Items (in order)

1. [ ] **Stable IDs.** `generate_one(seed)` / `generate_ids(...)`: a system depends only on its
   own seed (separate SeedSequence streams for stars, planets, moons), so the star map can draw
   the star alone and planets on click, identically. System k of a run = hash(master, k). Tests:
   same ID -> identical system regardless of batch; stars-only matches the full draw.
2. [ ] **System viewer** `python -m starsys.view` (and `python -m starsys.real NAME --view`):
   system map tree, orbit view (true separations, HZ shading, zoom/pan, lin/log, time slider),
   info panel, Prev/Next/go-to-ID, filters, export current system. Off-screen PNG checks.
3. [ ] **Procedural artwork**: textures per class (rocky: oceans/continents/ice caps/deserts/
   lava; giants: banded with storms, colour by temperature class (Sudarsky-like); ice giants;
   stars by Teff with limb darkening; moons; rings optional). Seeded per body.
4. [ ] **Galaxy model** `starsys.galaxy`: research notes in docs; density (thin/thick disc,
   bar/bulge, halo, log-spiral arms, Sun at ~8.2 kpc), position-dependent age/[Fe/H],
   arm-enhanced young stars; sectors seeded by coordinates, Poisson counts, system seed =
   hash(sector, j). Sources for every number in docs/ASSUMPTIONS.md.
5. [ ] **Real-star blend**: catalogue stars (20-pc census + a magnitude-limited bright-star
   catalogue, e.g. Hipparcos via VizieR) with their selection functions; thinning of procedural
   stars; real stars open through starsys.real conditioning.
6. [ ] **Star map window**: 3D rotatable (pyqtgraph OpenGL), level of detail (galaxy cloud far,
   individual stars near camera), search by name/ID, click -> system viewer.
7. [ ] **A10**: condition on known planets of all components (Alpha Cen -> Proxima b, d).
8. [ ] **Moon heating review**: check moon masses (Canup & Ward 1e-4 scaling), forced
   eccentricities, heating formula against literature (Heller & Pudritz on super-Jupiter moons);
   fix if a bug, otherwise label magma-ocean moons and document.
9. [ ] Docs (DESIGN, NEXT_STEPS, ASSUMPTIONS, README-style usage), memory, morning summary.

## Decisions

(Logged during the run.)
