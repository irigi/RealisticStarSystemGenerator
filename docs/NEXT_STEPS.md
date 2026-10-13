# Next steps (as of 2026-10-13)

Status: phases 1–3 of `DESIGN.md` are done and committed on branch `rewrite-starsys`
(data layer, validated HM-C port, stellar population validated against the 20-pc
census). Phase 4 (attaching planets to stars) is in progress: 4.1 and 4.2 are done, 4.3 is next.

## Resuming

```bash
source .venv/bin/activate
pytest -q                                  # 24 tests, all should pass
python -m starsys.validation.census        # stellar population vs 20-pc census
python -m starsys.validation.hmc --catalogs 6-25 --criterion hill   # HM-C port check
python -m starsys.validation.planets --catalogs 1-30                # planet glue on Kepler stars
python -m starsys.validation.mdwarf                                 # M dwarfs vs DC15, Hsu+2020, Ment+2023, K2 UCDs
```

`catalogs/` and `data/` are gitignored. If `data/` is lost, rebuild the SysSim
parquet cache with `python -m starsys.cache <models>`; the MIST grid and primary-mass
weights under `data/stellar/` are rebuilt on first use.

## Phase 4, in recommended order

Each step gets a cited source, a valid range, an extrapolation rule (flagged in the
output), and a validation check, as `DESIGN.md` §3 requires.

### 4.1 Planet-attachment glue — done (2026-10-13)

`starsys.planets.attach_planets(systems, components, seed)` returns one planet table
keyed by (system, component), with a per-planet extrapolation bit mask (`flags`:
colour outside HM-C's 0.5–1.7, pre-main-sequence host (since 4.2: `young`, age < 100 Myr), host in a multiple). HM-C
parameters are drawn per system from the 100 posterior catalogs. Hosts are pre-MS and
MS hydrogen-burning stars. Everyone else gets nothing until 4.2/4.6. Colour comes from
Teff via Mamajek plus a Kepler-calibrated offset and scatter. Optional per-star age
for photoevaporation (`age="star"`, off by default). Validated on Kepler stars against
the published catalogs (`python -m starsys.validation.planets`); see
`docs/validation/planets.md`.

Left for later steps: M dwarfs and late M currently get HM-C extrapolated with
f_swp clipped to 1 (flagged), to be replaced in 4.2. Binary components get
independent HM-C systems (flagged) until 4.5 prunes them. No XUV/bolometric scaling
(needs an α_pret re-fit).

Follow-ups: B1–B3 in the backlog below.

### 4.2 M dwarfs — done (2026-10-13)

`starsys.inner.mdwarf`: HM-C with an M-dwarf calibration phased in below 4000 K.
More planets, a fraction born without H/He envelopes, and periods down to 1.5 d,
fitted to Dressing & Charbonneau 2015. Hsu+2020's integrated rates are matched
near their baseline prior. Below 0.5 M☉, planet masses scale ∝ M★, and the bare
fraction and period floor are set against Ment & Charbonneau 2023. Below 0.1 M☉,
host fraction ∝ M^1.9 (disk mass), flagged; checked against Sestovic & Demory 2020
(K2 ultracool dwarfs). `python -m starsys.validation.mdwarf [--fit]`; results in
`docs/validation/planets.md`. The pre-MS flag was replaced by `young` (age < 100 Myr).
The colour flag no longer fires on M dwarfs, and brown dwarfs are now hosts.

Follow-ups: B4–B8 in the backlog below.

### 4.3 Giant planets, 0.03–30 au

- Occurrence vs semi-major axis: Fulton+2021 (California Legacy Survey), with the
  peak near 1–3 au and the turnover beyond it.
- Scaling: ∝ M★ (Johnson+2010) and ∝ 10^(2·[Fe/H]) (Fischer & Valenti 2005);
  A stars and giants per Reffert+2015 / Wolthoff+2022 (drop above ~2.5 M☉).
- Mass function: dN/dlogM from CLS. Eccentricities: Kipping 2013 beta distribution
  for RV planets.
- Inner small / outer giant correlation (Bryan & Lee 2024, Weiss+2024, He & Weiss
  2023), applied by conditioning the giant draw on the HM-C inner system.
- Wide-orbit cap from direct imaging (Nielsen+2019, Vigan+2021).
- Hot Jupiters (~0.5–1% for FGK) as a separate component; they suppress inner
  companions.
- Stability: reject or prune with the existing Hill/AMD machinery.
- Validation: reproduce CLS occurrence vs a; hot-Jupiter rate vs Kepler.

### 4.4 Small planets beyond 300 d

- Microlensing mass-ratio function (Suzuki+2016; Zang+2025) beyond the snow line.
- Bridge 300 d to the snow line with η⊕ constraints (Bryson+2021), which are
  extrapolations themselves; carry the uncertainty.
- Packing with the same Hill/AMD machinery.

### 4.5 Binaries

- S-type planets: remove orbits beyond the Holman & Wiegert 1999 critical radius.
- Suppress planet occurrence in close binaries (a < ~100 au) per Moe & Kratter 2021.
- Circumbinary (P-type) planets: Kepler occurrence (Armstrong+2014), outside the
  Holman–Wiegert inner limit.
- The population already provides hierarchy, a, e and masses (`draw_systems`).

### 4.6 Evolved stars and remnants

- Giants: engulf planets inside the maximum past radius (from MIST) plus a tidal
  margin. Expand surviving orbits adiabatically by M_initial / M_now.
- White dwarfs: survivors of the above; pollution rates (25–50%) as a sanity check.
- NS/BH: pulsar planets are very rare; use a tiny flagged rate.
- Subgiants (MIST phase 2 with log g ≥ 3.5; ~0.3% of stars) currently get no planets.
  They should get their main-sequence inner system (HM-C on the progenitor's MS
  parameters), then engulfment and orbit expansion as above. The same path serves the
  "retired A star" giant-planet hosts of 4.3.

### 4.7 Phase 4 validation report

`docs/validation/planets.md`: occurrence vs P and R by stellar type compared with
Kepler (via the HM-C catalogs), Hsu+2020 and CLS. Include the giant planet
correlations and a "where it is extrapolating" map.

## Backlog: inner-system refinements (from 4.1/4.2)

Numbered so they can be referenced; ordered by expected impact within each group.
Each needs a cited source and a validation check like the main steps.

Inner occurrence, M dwarfs (from 4.2):

- **B4** Mid-to-late M radius distribution is too broad. HM-C's σ_ln M = 1.77 cannot
  make the narrow 1–1.5 R⊕ peak of Ment & Charbonneau (too many sub-Earths, ~2σ), and
  the total is 1.7× high. Try a narrower mass distribution for low-mass hosts, or an
  Earth-like (not pure-silicate) composition for bare planets. Revisit when more TESS
  M-dwarf statistics are available.
- **B5** The early-M parameters are degenerate (different fits score within noise).
  Carry that uncertainty like HM-C's posterior: sample (d_lrc, d_mu, f_bare, P_min)
  from near-best fits per system instead of one point.
- **B6** XUV/bolometric enhancement for M dwarfs (`d_log_alpha_pret`) is physically
  expected but unconstrained by occurrence data. Constrain it with the M-dwarf radius
  valley (Van Eylen+2021; Petigura+2022; Ho & Van Eylen 2023).

Short periods (from 4.2):

- **B7** Planets with P < 3 d around FGK stars (including USPs, Sanchis-Ojeda+2014)
  are absent: HM-C starts at 3 d. M dwarfs now go down to ~0.75–1.5 d.
- **B8** HM-C has no smooth roll-off at short periods. A hard inner limit makes the
  shortest bin too full when the limit is low (the 0.5–1.7 d excess for 2–2.5 R⊕ in
  DC15). Do together with B7.

Host mapping and age (from 4.1, low priority):

- **B1** Colour scatter is Gaussian, but the Kepler residuals have heavier tails, which
  leaves a +0.0009 bias in the host fraction. Resample the empirical residuals per
  Teff bin.
- **B2** Validate `age="star"` against the observed radius valley vs age (Berger+2020
  Gaia–Kepler ages; David+2021; Fernandes+2023) before considering it as the default.
- **B3** HM-C has no [Fe/H] dependence. Small close-in planets are mildly more common
  at high [Fe/H] for P < 10 d (Petigura+2018). Consider a flagged tilt now that 4.3
  has metallicity scaling for giants.

## After phase 4

- **Phase 5:** moons and derived properties (T_eq, insolation, tidal locking,
  habitable-zone flags). See `DESIGN.md` §3.4.
- **Phase 6:** real-star conditioning (SIMBAD/Gaia/TIC lookup, NASA archive,
  per-method survey sensitivity, importance sampling). See §3.5.
- **Phase 7:** LightGBM emulators only where profiling shows they pay off. No torch.

## Known gaps in the stellar population (low priority)

- Low-mass He-core white dwarfs from binary interaction (~10% of WDs) are missing.
- NS/BH are placeholders; none lie within 20 pc, so they barely matter locally.
- The [Fe/H]–age relation is approximate; GALAH/APOGEE could replace it, since
  metallicity drives giant-planet occurrence in 4.3.
- Subgiant fraction is low against the census (labelling difference, not physics).

## Housekeeping

- `main` has only the initial commit. Merge `rewrite-starsys` once phase 4 is in
  shape. `gh` is not installed; use the GitHub API with curl or the web UI.
