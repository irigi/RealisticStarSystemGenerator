# Next steps (as of 2026-10-14)

Status: phases 1–3 of `DESIGN.md` are done and committed on branch `rewrite-starsys`
(data layer, validated HM-C port, stellar population validated against the 20-pc
census). Phase 4 (attaching planets to stars) is in progress: 4.1–4.4 are done, 4.5 is next.

## Resuming

```bash
source .venv/bin/activate
pytest -q                                  # 29 tests, all should pass
python -m starsys.validation.census        # stellar population vs 20-pc census
python -m starsys.validation.hmc --catalogs 6-25 --criterion hill   # HM-C port check
python -m starsys.validation.planets --catalogs 1-30                # planet glue on Kepler stars
python -m starsys.validation.mdwarf                                 # M dwarfs vs DC15, Hsu+2020, Ment+2023, K2 UCDs
python -m starsys.validation.giants                                 # giants vs CLS, RV, direct imaging
python -m starsys.validation.small                                  # small planets > 300 d vs η⊕, microlensing
```

`catalogs/` and `data/` are gitignored. If `data/` is lost, rebuild the SysSim
parquet cache with `python -m starsys.cache <models>`; the MIST grid and primary-mass
weights under `data/stellar/` are rebuilt on first use, as is the HM-C edge table under
`data/outer/` (~2 min).

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

### 4.3 Giant planets — done (2026-10-13)

`starsys.outer.giants`, attached by `attach_planets` (`giants=` option; `model` =
"giant", "hot_jupiter", "wide_giant"). Three components:

- warm and cold giants to 30 au: Fulton+2021 broken power law in a, CLS mass function
  (flat to 900 M⊕, then ∝ M⁻¹, fitted to CLS counts), ∝ M★ (Johnson+2010) up to the
  1.68 M☉ peak with a Gaussian drop above (Wolthoff+2022), ∝ 10^(1.2 [Fe/H]);
  overdispersed (Gamma) numbers per star. Normalisation and dispersion fitted jointly
  to five CLS rates;
- hot Jupiters (1% of the CLS stellar sample; TESS stellar-mass trend), circular,
  clearing the inner system 88% of the time (Wu+2023), with 4–5× more outer giants
  (Knutson+2014);
- wide giants at 10–100 au from the GPIES all-star fit (Nielsen+2019), ∝ M★².

Stability: AMD-Hill or Petrovich (2015). Giants are placed with orbit redraws; inner
systems conflicting with them are redrawn alternately with the giants' eccentricities
(so giants with close inner companions are less eccentric, as observed), then pruned.
The eccentricity prior is fitted so that the conditioned result matches CLS. Inside
HM-C's window only giants ≥ 10 R⊕ are drawn. No inner-small/outer-giant correlation
beyond metallicity (the largest samples find none at average metallicity; Bonomo+2025).
`python -m starsys.validation.giants [--fit]`; results in `docs/validation/planets.md`.

Follow-ups: B9–B14 in the backlog below.

### 4.4 Small planets beyond 300 d — done (2026-10-14)

`starsys.outer.small`, attached by `attach_planets` (`small=` option; `model` =
"outer_small", `clusterid` −2). Planets of 0.1–30 M⊕ from HM-C's P_max to 10 snow-line
radii (a_sl = 2.7 au M★/M☉):

- beyond a_sl: Zang+2025's lower mass-ratio Gaussian (KMTNet), flat in ln a, masses ∝ M★;
- at P_max: HM-C's own edge density and mass distribution (per posterior catalog and Teff,
  measured with the port, cached in `data/outer`);
- in between: log density interpolated linearly in ln a (stays lognormal in mass),
  flagged `bridge`; beyond a_sl outside the measured q, a or host mass, `cold_extrapolated`;
- uncertainty: Zang's parameters drawn per system, HM-C's anchor follows the posterior
  catalog;
- packing: HM-C's 10 mutual Hill radii plus AMD-Hill-or-Petrovich against the outermost
  inner planet, giants and each other (98.9% kept). HM-C radii and photoevaporation.

Checks: reproduces Zang+2025 over the measured q range (and the 4.3 giants reproduce
Zang's giant peak independently); η⊕ = 0.09 (conservative), 1.3σ below Bryson+2021's
lower bound but matching photoevaporation-aware estimates (Neil & Rogers 2020) and
Hsu+2019's range. `python -m starsys.validation.small`; results in
`docs/validation/planets.md`.

Follow-ups: B15–B19 in the backlog below.

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

Giant planets (from 4.3):

- **B9** HM-C's masses for 3–6 R⊕ planets look too high for RV: 11 per 100 Sun-like stars
  get a ≥ 30 M⊕ planet inside 300 d, against ~1.9 from the CLS model at 30–106 M⊕.
  Compare HM-C's mass–radius relation with RV masses (e.g. Otegi+2020) and check the
  sub-Saturn continuity at the 300 d boundary.
- **B10** Fulton+2021's 2–8 au rate (14.1) sits 1.4σ above the joint fit, which the
  near-complete raw CLS counts pull down. Fulton's quoted 3.8–7.1 au (15.5) and
  3000–10000 d (12.6) rates exceed their own 2–8 au rate; the model gives ~5. Recheck
  with the CLS posterior samples or a newer CLS occurrence paper.
- **B11** Remaining inner-system deficit against Kepler-fitted HM-C: inner planets per
  star drop 4% for 0.9–1.2 M☉ hosts and 8% for 1.2–1.6 M☉ (redraws favour smaller
  systems; ~1% of FGK hosts end with no stable inner system; hot-Jupiter clearing). Compare with Kepler occurrence around RV giant hosts
  (Kepler Giant Planet Search, Weiss+2024).
- **B12** Giant radii use a simple Neptune–Saturn–Jupiter relation and Weiss+2013
  inflation. Replace with evolution models (Fortney+2007; Thorngren+2016: metallicity,
  age) in phase 5.
- **B13** The a-distribution is the same for all stellar masses (Fulton's assumption);
  evolved-star surveys peak at ~720 d (Wolthoff+2022). The wide component's M★² scaling
  rests on 4 detections and is capped at 2.5 M☉; no metallicity dependence is assumed.
  Giant architecture: no resonant chains, eccentricity independent of giant
  multiplicity (CLS IV finds differences).
- **B14** Giants' angles refer to the inner system's invariable plane; recompute the
  whole system's invariable plane in phase 5.

Small planets beyond 300 d (from 4.4):

- **B15** Wide orbits: Poleski+2021 find 1.4 +0.9/−0.6 planets per star with
  1e-4 < q < 0.033 at s = 2–6 (5 planets, occurrence rising as s^1.1); the model gives 0.04,
  consistent with Zang+2025 at s ~ 1 and a flat extrapolation (Suzuki+2016's s^0.49 gives
  0.41). Revisit with newer KMTNet wide-orbit statistics; if confirmed, it affects the giants
  (Fulton's s(a) falls beyond 3.6 au) as well as small planets.
- **B16** η⊕ is set by the radius model, not the occurrence: the model has 0.50 planets per
  Sun-like star in the conservative HZ, but HM-C gives every planet an initial H/He envelope,
  so most are 1.5–3 R⊕. An intrinsically rocky population (Neil & Rogers 2020 models 3–4)
  or water-rich planets beyond the snow line would change radii; any change inside 300 d
  needs an HM-C re-fit. Do with phase 5 radii.
- **B17** Bridge shape: log-linear in ln a, anchored at a_sl, is a choice, not a
  measurement. The density at a fixed mass dips mid-bridge (the anchors' mass distributions differ). Sample the
  anchor position (1–2 a_sl) or the interpolation as a model uncertainty, and check against
  long-baseline RV for Sun-like stars (cold Neptunes at 1–3 au).
- **B18** Microlensing hosts are mostly 0.1–1 M☉; FGK/A hosts assume a universal mass ratio
  and a/a_sl. No metallicity dependence. Zang's parameters are drawn independently (their
  Dryad MCMC chains would keep correlations). Projected-separation mapping in the check uses
  a single lens geometry.
- **B19** Orbits of outer small planets (Rayleigh e σ = 0.05, i σ = 1.4°) continue HM-C's
  multis; nothing measures them. No small planets beyond 10 a_sl.

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
