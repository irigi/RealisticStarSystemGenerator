# Next steps (as of 2026-10-15)

Status: phases 1–5 of `DESIGN.md` are built and validated (stars, planets of every
population, derived properties, moons). Phases 1–4 are merged into `main`; phase 5 and the
backlog session are on branch `phase5`. How each part was built and checked:
`DESIGN.md` §3, `docs/validation/` (`planets.md` for phase 4, `phase5.md` for phase 5).

The model is close to the limit of what the published data can test. Remaining questions that
no available data or paper settles are written down as assumptions in `docs/ASSUMPTIONS.md`,
not kept as open tasks.

## Plan

1. **Settle the backlog** — done (2026-10-15): backlog sorted into planned work (below),
   assumptions (`docs/ASSUMPTIONS.md`) and things we won't do; repository cleaned up.
2. **Worldbuilding-relevant fixes** — done (2026-10-15): A1–A4, A7, A8 below. A5–A6 left
   open (optional).
3. **Phase 6, real-star conditioning** (`DESIGN.md` §3.5): systems around named stars,
   conditioned on their known planets and on survey sensitivity. In progress; plan below.
4. **Usability:** readable per-system summaries and export formats, including A9.

Phase 7 (learned emulators) only if profiling in phase 6 shows it pays off.

## Phase 6 plan (real-star conditioning, branch `phase6`)

Goal: `python -m starsys.real "Barnard's Star"` returns systems that are drawn from the model,
contain the star's known planets, and are consistent with the surveys that did not find more.
Code in a new package `src/starsys/real/`. Each step ends with a commit, tests and a validation
note (`docs/validation/phase6.md`).

1. **The star** (`real/catalog.py`, `real/star.py`). Lookup by any common name (HD, HIP, GJ,
   TIC, Gaia, proper and Bayer names) in the 20-pc census (Kirkpatrick+2024; components and
   hierarchy) and the NASA Exoplanet Archive hosts; Gaia DR3 online (cached) as a fallback.
   Measured Teff, radius or luminosity, mass, log g, [Fe/H] and age with their errors give a
   posterior over (initial mass, age, [Fe/H]) by importance sampling the population prior
   through `evolve` (MIST, remnants, substellar). Known companions come from the census with
   their initial masses; separations from the components' positions, else drawn from the
   population model below the resolution limit. Output: `systems, components` for N
   realisations, so that `attach_planets` and `attach_moons` run unchanged.
   *Done (2026-10-15):* adds the Bright Star Catalogue, PASTEL and the WDS
   (`python -m starsys.real.fetch`); giants are fitted on the MIST isochrone nodes; a second,
   adapted round of draws where the first gives fewer than 300 effective samples.
2. **Known planets and conditioning** (`real/known.py`, `real/condition.py`). Known planets from
   the NASA archive. Prior systems are drawn for the star and weighted by how well they contain
   the known planets (kernels in log P and log R or log M sin i), times the probability that
   each of their other planets was missed. Transiting planets are matched by re-orienting the
   system so that the planet transits (weighted by the transit probability), so co-transiting
   neighbours come out right. Matched planets are snapped to the measured values. Known planets
   the prior cannot make are inserted with a penalty; the effective sample size is reported.
3. **Survey sensitivity** (`real/surveys.py`), per method:
   - transit: TESS sectors (tess-point) with a noise model by Tmag; Kepler DR25 for Kepler
     targets;
   - RV: the star's own archival time series where it exists (HARPS RVBank, HIRES Butler+2017,
     CARMENES DR1) with an expected-Delta-chi^2 detection test; otherwise a monitoring
     probability by magnitude and spectral type from those surveys' target lists;
   - astrometry: Gaia DR3 RUWE (no excess = no detectable wobble over ~3 yr);
   - imaging: coarse contrast limits, off unless the star is a known imaging target.
4. **Validation and report**: self-consistency on simulated stars (posterior coverage of
   unseen planets), benchmark stars (the Sun seen from 10 pc, alpha Cen A/B, Proxima,
   Barnard's Star, TRAPPIST-1, eps Eri, Vega, Sirius), runtime.

## Resuming

```bash
source .venv/bin/activate
pytest -q                                  # all tests should pass
python -m starsys.validation.census        # stellar population vs 20-pc census
python -m starsys.validation.hmc --catalogs 6-25 --criterion hill   # HM-C port check
python -m starsys.validation.planets --catalogs 1-30                # planet glue on Kepler stars
python -m starsys.validation.planets --valley                       # radius valley vs age (Berger+2020)
python -m starsys.validation.planets --metallicity [--fit]          # [Fe/H] dependence vs Petigura+2018
python -m starsys.validation.mdwarf                                 # M dwarfs vs DC15, Hsu+2020, Ment+2023, K2 UCDs
python -m starsys.validation.giants                                 # giants vs CLS, RV, direct imaging
python -m starsys.validation.small                                  # small planets > 300 d vs η⊕, microlensing
python -m starsys.validation.binaries                               # suppression, boost, hot Jupiters, circumbinary
python -m starsys.validation.evolved                                # giant stars, white dwarfs, engulfment
python -m starsys.validation.short                                  # planets inside 3 d vs Hsu+2019, Sanchis-Ojeda+2014
python -m starsys.validation.derived                                # insolation, HZ, spin, Kozai-Lidov
python -m starsys.validation.moons                                  # moons vs Solar System, Teachey+2018
python -m starsys.validation.report                                 # phase 4 report (writes docs)
python -m starsys.validation.report5                                # phase 5 report (writes docs)
```

`validation.mdwarf`, `.giants` and `.binaries` take `--fit` to redo their calibration.

`catalogs/` and `data/` are gitignored. Everything under `data/` is a cache rebuilt on first
use: the MIST grid and primary-mass weights (`data/stellar/`), the HM-C edge table
(`data/outer/`, ~2 min), the single-star boost table (~15 s) and the Kozai–Lidov cycle table
(~6 min, both `data/binaries/`).

## Backlog

Item numbers (B1–B30) are kept so that older commits and docs still resolve.

### A. Planned: improvable with the data and papers we have

Done in step 2 (2026-10-15; details in `docs/ASSUMPTIONS.md` and the validation docs):

- **A1 (B21) Multiplicity around K and M dwarfs** — confirmed as a tension and made a setting.
  Against HM-C's Kepler-fitted field model, the boost raises the share of systems with two
  transiting planets by +28% (G), +45% (K), +28% (M) (`validation.binaries`); putting less of it
  into the cluster count loses 14–29% of planets instead. Both Kepler observables cannot be kept
  without a re-fit; `BinaryModel(boost_cluster_share=1)` (default: planets per star) or 0
  (multiplicity mix).
- **A2 (B16) Temperate rocky planets** — setting `attach_planets(temperate_bare_fraction=0)`;
  η⊕ 0.10 (default) to 0.33.
- **A3 (B3) Metallicity** — `starsys.inner.metallicity` (Petigura+2018), on by default; three of
  six betas within 1σ, hot sub-Neptunes and sub-Saturns shallower than measured.
- **A4 (B5) Early-M degeneracy** — 40 equally good parameter sets drawn per host; DC15 score
  25.5 → 17.4.
- **A7 (B15, B22) Settings** — `SmallPlanetModel(wide_factor=1)`, `BinaryModel(cb_rate_factor=1)`.
- **A8 (B30) Moons slow their planet's spin** — `moons.attach_moons` updates `rotation_h`.

Open:

- **A5 (B6, optional) XUV of M dwarfs.** Constrain `d_log_alpha_pret` with the M-dwarf radius
  valley (Van Eylen+2021; Petigura+2022; Ho & Van Eylen 2023).
- **A6 (B28, B29, optional) Worldbuilding extras.** Small irregular moons (size
  distribution, Jewitt & Haghighipour 2007) and water lost during the runaway phase
  (Luger & Barnes 2015 escape rates).
- **A9 Temperature presentation** (step 4). `t_eq` is the no-greenhouse equilibrium
  temperature (Earth 255 K, real surface 288 K). Label it as such in summaries and add a rough
  surface-temperature estimate for rocky planets, marked as illustrative.

### B. Assumed (documented in `docs/ASSUMPTIONS.md`)

B4, B10, B13 (resonant chains), B15, B17, B18, B19, B20, B21, B22, B23, B24, B25, plus the choices
behind closed items (B1, B2, B3, B5, B9, B16, B30). B15, B16, B21 and B22 are user settings.

B18 is the only item more data would help: Zang+2025's KMTNet sensitivity functions and
planet table are public on Dryad (doi:10.5061/dryad.ksn02v7cg) but block scripted downloads.
Downloading `sensitivity.tar` and `table_S3.xlsx` by hand into
`catalogs/observational/zang2025/` would allow a joint refit of the microlensing component.

### C. Won't do

- **Re-fit HM-C with a SysSim forward model** (would settle B3, B16, B21, B24 inside the fit).
  It needs the whole Kepler detection pipeline rebuilt (DR25 completeness, CDPP, window
  functions, vetting) and an ABC fit that took the authors cluster time. Corrections are
  applied on top of HM-C instead, each preserving its Kepler averages
  (`inner.rv_masses`, `inner.short`, effective age, A1–A3).

### Closed

B1, B2, B3, B5, B7, B8, B9, B11, B12, B14, B16, B26, B27, B30. Resolutions: `docs/validation/planets.md`,
`docs/validation/phase5.md`, and git history of this file (commit 46936e9 has the full
entries).

## Housekeeping

- Nothing has been pushed: `origin/main` no longer exists on the remote. `gh` is not
  installed; use the GitHub API with curl or the web UI.
