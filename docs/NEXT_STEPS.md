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
2. **Worldbuilding-relevant fixes** (one session): A1–A4 below, A5–A6 if time allows.
3. **Phase 6, real-star conditioning** (`DESIGN.md` §3.5): systems around named stars,
   conditioned on their known planets and on survey sensitivity.
4. **Usability:** readable per-system summaries and export formats.

Phase 7 (learned emulators) only if profiling in phase 6 shows it pays off.

## Resuming

```bash
source .venv/bin/activate
pytest -q                                  # all tests should pass
python -m starsys.validation.census        # stellar population vs 20-pc census
python -m starsys.validation.hmc --catalogs 6-25 --criterion hill   # HM-C port check
python -m starsys.validation.planets --catalogs 1-30                # planet glue on Kepler stars
python -m starsys.validation.planets --valley                       # radius valley vs age (Berger+2020)
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

### A. Planned: improvable with the data and papers we have (step 2)

- **A1 (B21) Multiplicity around K and M dwarfs.** The single-star boost goes into HM-C's
  cluster count where f_swpa is already 1, so the share of hosts with ≥ 5 inner planets rises
  (M: 0.49 → 0.68, K: 0.15 → 0.32): too many TRAPPIST-1-like systems. Cap it with the
  published Kepler M-dwarf multiplicity (Ballard & Johnson 2016; Hsu+2020's multiplicity;
  DC15) while keeping the planet-per-star averages.
- **A2 (B16) Rocky or envelope-bearing temperate planets, as a setting.** HM-C gives every
  planet an initial H/He envelope, so most habitable-zone planets are 1.5–3 R⊕ and η⊕ (rocky,
  conservative HZ) is 0.09. Published η⊕ ranges from ~0.05 to ~0.6 (Neil & Rogers 2020;
  Bryson+2021); the data do not decide. Add a setting for the born-bare fraction outside
  Kepler's well-measured range, default unchanged, and document both ends.
- **A3 (B3) Metallicity dependence of small planets.** Petigura+2018: small planets at
  P < 10 d are more common at high [Fe/H]. Apply it as a mean-one reweighting over the
  Kepler field's [Fe/H] distribution, so Kepler averages are kept. Needed by phase 6, where
  stars have measured [Fe/H].
- **A4 (B5) Early-M degeneracy.** Several early-M fits score within noise. Draw
  (d_lrc, d_mu, f_bare, P_min) per system from the near-best fits, as HM-C's posterior is
  drawn per system.
- **A5 (B6, optional) XUV of M dwarfs.** Constrain `d_log_alpha_pret` with the M-dwarf radius
  valley (Van Eylen+2021; Petigura+2022; Ho & Van Eylen 2023).
- **A6 (B28, B29, optional) Worldbuilding extras.** Small irregular moons (size
  distribution, Jewitt & Haghighipour 2007) and water lost during the runaway phase
  (Luger & Barnes 2015 escape rates).

### B. Assumed (documented in `docs/ASSUMPTIONS.md`)

B4, B10, B13 (resonant chains), B15, B17, B18, B19, B20, B22, B23, B24, B25, plus the choices
behind closed items (B1, B2, B9, B30). B15 (wide-orbit rate) and B22 (circumbinary rate) are
candidates for settings alongside A2.

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

B1, B2, B7, B8, B9, B11, B12, B14, B26, B27, B30. Resolutions: `docs/validation/planets.md`,
`docs/validation/phase5.md`, and git history of this file (commit 46936e9 has the full
entries).

## Housekeeping

- Nothing has been pushed: `origin/main` no longer exists on the remote. `gh` is not
  installed; use the GitHub API with curl or the web UI.
