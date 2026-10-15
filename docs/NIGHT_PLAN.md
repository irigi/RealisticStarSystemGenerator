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

1. [x] **Stable IDs.** (done: `starsys.ids`, `python -m starsys --id S1-42`) `generate_one(seed)` / `generate_ids(...)`: a system depends only on its
   own seed (separate SeedSequence streams for stars, planets, moons), so the star map can draw
   the star alone and planets on click, identically. System k of a run = hash(master, k). Tests:
   same ID -> identical system regardless of batch; stars-only matches the full draw.
2. [x] **System viewer** (done: `starsys.view`, `python -m starsys.view`) `python -m starsys.view` (and `python -m starsys.real NAME --view`):
   system map tree, orbit view (true separations, HZ shading, zoom/pan, lin/log, time slider),
   info panel, Prev/Next/go-to-ID, filters, export current system. Off-screen PNG checks.
3. [x] **Procedural artwork** (done: `starsys.view.art`; refine if time remains): textures per class (rocky: oceans/continents/ice caps/deserts/
   lava; giants: banded with storms, colour by temperature class (Sudarsky-like); ice giants;
   stars by Teff with limb darkening; moons; rings optional). Seeded per body.
4. [x] **Galaxy model** (done: `starsys.galaxy`; sources in its docstring, summary in ASSUMPTIONS) `starsys.galaxy`: research notes in docs; density (thin/thick disc,
   bar/bulge, halo, log-spiral arms, Sun at ~8.2 kpc), position-dependent age/[Fe/H],
   arm-enhanced young stars; sectors seeded by coordinates, Poisson counts, system seed =
   hash(sector, j). Sources for every number in docs/ASSUMPTIONS.md.
5. [x] **Real-star blend** (done: `starsys.realstars`): catalogue stars (20-pc census + a magnitude-limited bright-star
   catalogue, e.g. Hipparcos via VizieR) with their selection functions; thinning of procedural
   stars; real stars open through starsys.real conditioning.
6. [x] **Star map window** (done: `python -m starsys.view --galaxy`): 3D rotatable (pyqtgraph OpenGL), level of detail (galaxy cloud far,
   individual stars near camera), search by name/ID, click -> system viewer.
7. [x] **A10** (done): condition on known planets of all components (Alpha Cen -> Proxima b, d).
8. [x] **Moon heating review** (done: capped by tidal energy supply): check moon masses (Canup & Ward 1e-4 scaling), forced
   eccentricities, heating formula against literature (Heller & Pudritz on super-Jupiter moons);
   fix if a bug, otherwise label magma-ocean moons and document.
9. [x] Docs (DESIGN, NEXT_STEPS, ASSUMPTIONS, README-style usage), memory, morning summary.

## Decisions

(Logged during the run.)

- D1 (item 1): instead of rewriting the vectorised pipeline per system, stars are drawn in
  fixed blocks of 64 (block seed = mix(master, block)) and planets/moons of one system alone
  from mix(master, block, j). Stars-only draws stay batched (fast for the star map), planets
  ~5 ms per system on click. ID format `S<master>-<k>`; galaxy IDs `G<ix>.<iy>.<iz>-<j>` reuse
  `ids.complete`.

- D2 (item 2): Pioneer's system map is a vertical column; I used one horizontal row per star
  (planets left to right, moons hanging below), which suits a widescreen window and multiple
  stars (one row each, brackets show the hierarchy, a dashed bracket and a "AB" row for
  circumbinary planets). Body sizes ∝ R^0.5 (monotonic), stars capped.
- D3 (item 2): the orbit view is face-on in one plane (mutual inclinations ignored); companions
  lack an orbital phase in the generator, so it is taken from the body's art seed. Log scale
  maps distance from the view centre; it switches on automatically when the orbits span > 30x.
- D4 (item 3): no rings (no ring model in the generator; the viewer shows only generated
  physics). Giant palettes follow Sudarsky classes; star colours are blackbody, saturated x2.2
  for display. Real-star planets keep catalogue names but rows show the suffix ("b").
- D5 (item 2): `--real NAME` shows the conditioned realisations; Previous/Next walk them.
- D6 (item 4): spiral arms = Reid et al. (2019) Table 2 within measured azimuths, the four
  major arms extended at their mean 10° pitch from 3 to 15 kpc, Local arm only ±25° beyond
  its data. Old-star arm contrast ±15 %, young stars (< 300 Myr) confined to arms.
- D7 (item 4): halo normalised to its total mass 1.4e9 Msun (Deason+2019) rather than the 0.5 %
  local fraction (that would triple the halo mass with the Juric+2008 profile); thin-disc scale
  height 350 pc (upper end of BHG16's 300 ± 50) so the local surface density is closer to
  McKee+2015. Totals: 8.4e10 systems, ~3.9e10 Msun in stars+remnants (the real Galaxy has
  ~5e10: the disc comes out ~35 % light; local normalisation from the census was kept).
- D8 (item 4): sectors are 10-pc cubes (~80 systems near the Sun, ~5600 at the centre); IMF
  and multiplicity are the local ones everywhere; ages/[Fe/H] by component and radius.
- D9 (item 5): catalogues = 20-pc census (complete) + Hipparcos Survey part (complete to
  V <= 7.9/7.3 + 1.1|sin b|, ESA 1997) with parallax SNR > 5; procedural systems are removed
  exactly when they would be in those (selection function is 0/1, so no random thinning
  needed). Check: in 25-55 pc shells the procedural systems that would be catalogued match the
  real Hipparcos count within 10 %. Fainter Hipparcos stars are left out (they would duplicate
  the fill). IAU-named stars are kept down to parallax SNR 1.5 (Deneb, Betelgeuse).
  Downloads: Hipparcos I/239 via VizieR and the IAU-CSN name list (catalogs/, not committed);
  the merged table is cached in data/real/starmap.parquet (~25 s to build).
- D10 (item 6): star map = pyqtgraph OpenGL. Stars are drawn by apparent magnitude *from
  the camera* against a limit that follows the zoom, times a spatial window around the view
  centre (a map, not a sky; otherwise ~15k Hipparcos stars within 120 pc swamp the view).
  Beyond 3 kpc camera distance only the galaxy cloud (150k points sampled from the density
  model, arms young/blue) is shown. Elite-style depth cues: rings in the plane through the
  view centre and drop lines for the ~400 brightest nearby stars. Procedural sectors load
  12 per 60-ms tick within 12-45 pc of the view centre. Labels: IAU/Bayer/Flamsteed names.
- D11 (item 6): clicking a real star runs the real-star conditioning (12 realisations) in a
  worker thread and opens the system viewer on them; a procedural star opens the viewer with
  Previous/Next walking its loaded neighbours by distance.
- D12 (item 7): A10 as a product of per-component likelihoods (planets of different components
  are independent given the stars in the generator), each component with its own default surveys.
  Model planets skip letters taken by known planets even when the archive spells the host
  differently ("Proxima Cen b" vs "Proxima Centauri c").
- D13 (item 8): the extreme heating was not a bug in the formula but a missing physical limit:
  a resonance can only dissipate the power the planet's tide transfers to the moons' orbits
  (Yoder & Peale 1981). Without it the median heated moon got 5x that supply; super-Jupiter
  moons reached 1e4 W/m^2. With the cap the maximum in 20 000 systems is 142 W/m^2 and 35 moons
  exceed 30 W/m^2 (was 234). Kept the generic giant k2/Q (5e-6) rather than Jupiter's measured
  1.1e-5, so an Io analogue gets ~0.5x Io's observed heat (changing it would also move tidal
  locking times). Moons above 20 W/m^2 are labelled "super-Io".

## Morning summary (2026-10-15/16)

All nine items done on `phase7-viewer` (not pushed); 103 tests pass. Screenshots in
`docs/figures/phase7/`.

Try:

```bash
source .venv/bin/activate
python -m starsys.view                         # system viewer; Next/Previous (PageDown/PageUp)
python -m starsys.view --real "tau Ceti"       # a real star's conditioned systems
python -m starsys.view --galaxy                # 3D star map; search "Vega", click, "Open system"
python -m starsys --id S1-42                   # any system again by its ID
```

What to look at / decide:
- Art direction (`art_variety.png`): physics-guided palettes (Sudarsky giant classes,
  climates, blackbody stars); no rings by choice (D4).
- The galaxy comes out ~3.9e10 Msun vs ~5e10 real with the local census normalisation (D7).
- A10 makes Alpha Centauri's systems carry Proxima b and d (and Proxima's own surveys).
- Moon heating: capped by the planet's tidal energy supply; max 142 W/m^2 instead of 1.9e4 (D13).
- Proposals for next steps in `docs/NEXT_STEPS.md` ("Next (proposals)"): open clusters, Gaia,
  extinction maps, viewer legend.
