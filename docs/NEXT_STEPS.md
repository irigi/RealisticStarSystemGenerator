# Next steps (as of 2026-10-15)

Status: phases 1–5 of `DESIGN.md` are done. Phases 1–4 are merged into `main`; phase 5 is on
branch `phase5`. Done so far:
- phases 1–3: data layer, validated HM-C port, stellar population validated against the 20-pc
  census;
- phase 4: attaching planets to stars (4.1–4.7);
- phase 5: derived properties, Kozai–Lidov, moons, and tidal decay on the giant branches.

The backlog below holds the follow-ups. Next is phase 6 (real-star conditioning).

## Resuming

```bash
source .venv/bin/activate
pytest -q                                  # 55 tests, all should pass
python -m starsys.validation.census        # stellar population vs 20-pc census
python -m starsys.validation.hmc --catalogs 6-25 --criterion hill   # HM-C port check
python -m starsys.validation.planets --catalogs 1-30                # planet glue on Kepler stars
python -m starsys.validation.mdwarf                                 # M dwarfs vs DC15, Hsu+2020, Ment+2023, K2 UCDs
python -m starsys.validation.giants                                 # giants vs CLS, RV, direct imaging
python -m starsys.validation.small                                  # small planets > 300 d vs η⊕, microlensing
python -m starsys.validation.binaries                               # suppression, boost, hot Jupiters, circumbinary
python -m starsys.validation.evolved                                # giant stars, white dwarfs, engulfment
python -m starsys.validation.short                                  # planets inside 3 d vs Hsu+2019, Sanchis-Ojeda+2014
python -m starsys.validation.report                                 # whole-population phase 4 report (writes docs)
python -m starsys.validation.derived                                # insolation, HZ, spin, Kozai-Lidov
python -m starsys.validation.moons                                  # moons vs Solar System, Teachey+2018
python -m starsys.validation.report5                                # phase 5 report (writes docs)
```

`catalogs/` and `data/` are gitignored. If `data/` is lost, rebuild the SysSim
parquet cache with `python -m starsys.cache <models>`; the MIST grid and primary-mass
weights under `data/stellar/` are rebuilt on first use, as are the HM-C edge table under
`data/outer/` (~2 min), the single-star boost table under `data/binaries/` (~15 s) and the
Kozai–Lidov cycle table, also under `data/binaries/` (~6 min). The MIST grid gained a helium
core-mass column in phase 5; an older cached grid is rebuilt automatically (~10 s).

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

### 4.5 Binaries — done (2026-10-14)

`starsys.binaries`, with the host rows in `starsys.hosts` (`attach_planets(binaries=...)`;
`None` treats every component as single with field-average rates):

- suppression: a star keeps its planetary system with probability S_bin(a) of its nearest
  stellar companion (Moe & Kratter 2021: 0 at 1 au, 0.15 at 10 au, 1 beyond 200 au);
- S-type limit for every companion and circumbinary limits for the inner pair (Holman &
  Wiegert 1999);
- circumbinary planets (`host_kind` "P", flag `circumbinary`): the single-star models outside
  the limit, plus an empirical 4–10 R⊕ component ("circumbinary") calibrated to Armstrong+2014;
- single-star boost of the field-average rates (HM-C, M dwarfs, microlensing anchor) from a
  mock magnitude-limited Kepler sample, with a fitted Teff-dependent efficiency correction;
- planes aligned with the companion's orbit in half the systems inside 700 au
  (Christian+2022); binary orbits now have orientations in `draw_systems`.

Kepler-like averages are preserved to 1–2%; hot Jupiters come out at 0.62% in a transit-like
sample and 0.80% in an RV-like sample (MK21's explanation of the discrepancy). The giant
eccentricity prior was refitted. `python -m starsys.validation.binaries [--fit]`; results in
`docs/validation/planets.md` §4.5. Follow-ups: B20–B22.

### 4.6 Evolved stars and remnants — done (2026-10-14)

`starsys.evolved` (`attach_planets(evolved=...)`): subgiants, giants, post-AGB stars and white
dwarfs get the planets of their main-sequence progenitor (all models run on MIST's mid-MS
state). Planets inside the Mustill & Villaver (2012) engulfment radius, a multiple of the
maximum past radius from MIST, are removed; survivors move out by M_progenitor/M_now. Neutron
stars get rare flagged pulsar planets ("pulsar", 0.2%); black holes none. Systems whose evolved
component overfilled its Roche lobe are flagged `post_interaction`. 40% of white dwarfs keep
planets (25–50% polluted). Giant stars host giant planets at 5% at 89–4400 d (Wolthoff+2022:
10.7%; B23). Tidal decay on the giant branches added in phase 5.
`python -m starsys.validation.evolved`; results in §4.6. Follow-up: B23.

### 4.7 Phase 4 validation report — done (2026-10-14)

`python -m starsys.validation.report [--systems N]` draws the population with all models and
writes `docs/validation/phase4_report.md`, plus figures in `docs/validation/figures/`:
- occurrence by host type and in period–radius;
- Kepler (published HM-C catalogs), DC15/Hsu+2020 and CLS comparisons;
- giant correlations;
- the extrapolation map.

`docs/validation/planets.md` opens with a summary table of every phase-4 check and its verdict.
The whole population reproduces the published HM-C inner occurrence for a Kepler-like sample
(2.36 vs 2.38 planets per star, all P–R cells within 1σ). 35% of planets carry an
extrapolation flag. Attaching planets takes ~17–20 s per 100,000 systems.

## Phase 5 — done (2026-10-15)

Details and verdicts: `docs/validation/phase5.md`; generated tables:
`docs/validation/phase5_report.md`.

- **5.1 Tidal decay on the giant branches** (`starsys.evolved.tidal_decay`; B23).
  - The equilibrium tide (Villaver & Livio 2009) is integrated along the host's past, read off a
    MIST isochrone (helium core masses added to the grid). Planets are engulfed when the star
    fills its Roche lobe.
  - Matches MV12's engulfment radii to ~20%.
  - The giant-star check now counts only RV-detectable periods. That turned the earlier match
    into a factor-2 deficit (B23).
- **5.2 Derived properties** (`starsys.derived`, `derived=` option).
  - Orbit-averaged insolation from all stars, T_eq, Kopparapu+2014 habitable zone (stars
    combined as Haghighipour & Kaltenegger 2013), tidal-locking spin states.
  - Giants of evolved hosts are re-inflated by the present flux.
  - Recomputed insolation matches the archive's `pl_insol` (median 1.000).
- **5.3 Kozai–Lidov cycles** (`BinaryModel.kozai`).
  - Quadrupole test-particle cycles, quenched by GR and by the other planets' forced precession.
  - Eccentricities are drawn from a cached cycle table.
  - Planets driven into the star or out of the HW99 limit are removed.
  - Only 3% of giants cycle; the giant eccentricity prior did not need a refit.
- **5.4 Moons** (`starsys.moons.attach_moons`): regular satellites, giant-impact moons and
  captured moons, with tidal and Hill-sphere survival. Checked against the Solar System,
  Elser+2011 and Teachey+2018.

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

- **B7** ~~Planets with P < 3 d around FGK stars~~ — done (2026-10-15): `starsys.inner.short`
  (`attach_planets(short=...)`, model "short_period", flag `short_period`). It fills 0.5 d to the
  host's HM-C limit:
  - Hsu+2019's grid at 0.5–2 d;
  - at 2–3 d, the part of Hsu's 2–4 d rate HM-C's 3–4 d planets leave;
  - Sanchis-Ojeda+2014's stellar-type trend for USPs.

  Matches Hsu within ~10% below 2 d and Sanchis-Ojeda for F, G and K.
  `python -m starsys.validation.short`; `docs/validation/planets.md`. Not covered: periods
  below 0.5 d (Kepler-78 b at 8.5 h).
- **B8** HM-C has no smooth roll-off at short periods. A hard inner limit makes the
  shortest bin too full when the limit is low (the 0.5–1.7 d excess for 2–2.5 R⊕ in
  DC15). For FGK this is solved by B7. For M dwarfs the extension's floor remains: USPs
  around M dwarfs come out at 1.8% against Sanchis-Ojeda's 1.10 ± 0.40%.
- **B24** HM-C against Hsu+2019 at 2–8 d (same Kepler sample, different models): at 3–4 d HM-C
  alone exceeds Hsu's whole 2–4 d bin for 2–6 R⊕ (×1.9 at 2–4 R⊕), and at 4–8 d it has half of
  Hsu's sub-Earths (0.5–1 R⊕: 0.060 vs 0.124 +0.022/−0.029). HM-C's period scale is a
  power law per cluster with a hard 3 d edge, with no roll-off. A short-period roll-off in HM-C would need a
  re-fit (SysSim). Kepler's sub-Earth rates are uncertain by ~30%.

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
  measurement. The density at a fixed mass dips mid-bridge (the anchors' mass distributions
  differ). Checked 2026-10-14: the anchor cannot move outward (to 1–2 a_sl, as first planned).
  For a typical lens (0.5 M☉, R_E = 2.5 au, a_sl = 1.35 au), microlensing's measured
  s = 0.5–2 is ~0.9–3.7 a_sl in projection, so the microlensing anchor already starts where
  the data start. The open question is the bridge's shape inside a_sl. Constrain it with
  long-baseline RV of Sun-like stars (cold Neptunes and super-Earths at 1–3 au, e.g. the HARPS
  and CLS sensitivity maps) or Gaia DR4 astrometry, then sample the shape as a model uncertainty.
- **B18** Microlensing hosts are mostly 0.1–1 M☉; FGK/A hosts assume a universal mass ratio
  and a/a_sl. No metallicity dependence. Zang's parameters are drawn independently. Their Dryad
  record (doi:10.5061/dryad.ksn02v7cg) has no chains, but it has the KMTNet sensitivity
  functions and the planet sample (Table S3), enough to refit the double Gaussian and sample
  the joint posterior. Dryad refuses scripted downloads (403, bearer token): download
  `sensitivity.tar` and `table_S3.xlsx` by hand into `catalogs/observational/zang2025/` (left
  empty for them). Projected-separation mapping in the check uses a single lens geometry.
- **B19** Orbits of outer small planets (Rayleigh e σ = 0.05, i σ = 1.4°) continue HM-C's
  multis; nothing measures them. No small planets beyond 10 a_sl.

Binaries and evolved stars (from 4.5/4.6):

- **B20** The population has fewer close companions than MK21 assume (Moe & Di Stefano 2017):
  21% of G primaries are suppressed (volume-limited) against MK21's 33 ± 5%, so the single-star
  boost is 1.30, not ~1.8. The boost is consistent with our own population. If the population
  is moved to MDS17's period distribution (more companions at 0.1–10 au), B follows
  automatically (cached in `data/binaries`).
- **B21** Where f_swpa is already 1 (K and M dwarfs), the boost goes into HM-C's cluster count:
  Kepler-like averages are preserved, but the share of hosts with ≥ 5 inner planets rises
  (M: 0.49 → 0.68, K: 0.15 → 0.32). A proper fix is an HM-C re-fit with binaries (suppression
  and dilution) in the SysSim forward model. Transit dilution of planet radii in unresolved
  binaries is not modelled either.
- **B22** Circumbinary rates rest on ~10 Kepler planets (Armstrong+2014, factor ~3
  uncertain); the per-binary rates (0.78 for 4–10 R⊕ around P_bin ≥ 7 d binaries) depend on
  the placement (P/P_crit 1–2.5) and on the P_bin < 7 d shortfall. Wider binaries get the same
  planets near their limit (extrapolation). Revisit with TESS/BEBOP circumbinary statistics.
  Kozai–Lidov excitation of planets by inclined companions is not modelled (phase 5).
- **B23** Giant stars (checked 2026-10-15 after adding tidal decay on the giant branches,
  Villaver & Livio 2009, and counting only Wolthoff's detectable periods, 89–4400 d):
  - **Occurrence:** the model gives 5.2% of 1–3 M☉ giants a ≥ 0.8 MJ planet, against
    Wolthoff+2022's 10.7 +2.2/−1.6%, and about half of their mass–metallicity fit at every mass.
    Main-sequence stars in the same window match Cumming+2008.
  - **What the fix needs to decide:** either the giant occurrence rises faster with M★ than
    Johnson+2010's ∝ M★, or giant-star masses and RV planet samples are biased.
  - **Close-in giants:** giants with P < 90 d around giant stars stay at 1.8–1.9% (main
    sequence 3–4%), against almost none observed. The survivors are warm giants at 6–20 R★,
    beyond any tidal reach. A primordial cut for intermediate-mass stars (B13) or an
    observational explanation is needed; subgiant hosts (Bowler+2010) point to primordial.
  - **Engulfment radii:** MIST's AGB radii at 5 M☉ are ~40% smaller than MV12's models.

Host mapping and age (from 4.1, low priority):

- **B1** ~~Resample the empirical Kepler colour residuals instead of a Gaussian~~ — closed
  (2026-10-14): tried; resampling the empirical residuals per 200 K bin gives f_hosts 0.8673
  against 0.8680 published (−0.0007), the Gaussian +0.0007 (30 catalogs). The bias is at the
  0.001 level either way and is not set by the tails (residual–star correlations, e.g.
  metallicity, are in neither). The Gaussian is kept: it needs no Kepler table at run time.
- **B2** Validate `age="star"` against the observed radius valley vs age (Berger+2020
  Gaia–Kepler ages; David+2021; Fernandes+2023) before considering it as the default.
- **B3** HM-C has no [Fe/H] dependence. Small close-in planets are mildly more common
  at high [Fe/H] for P < 10 d (Petigura+2018). Consider a flagged tilt now that 4.3
  has metallicity scaling for giants.

From phase 5:

- **B25** Small-planet envelopes around evolved hosts are left as on the main sequence. Giant
  stars are bright, but their XUV output, which drives envelope loss, is unconstrained. Only
  giant-planet radii are recomputed (re-inflation).
- **B26** Kozai–Lidov: RV giants with a known stellar companion have a higher mean eccentricity
  than those without (NASA archive: 0.32 vs 0.25, 74 vs 332 planets at 0.1–5 au). The model
  gives none (0.217 vs 0.213), because GR and the planets' mutual precession quench the cycles
  unless the companion is within ~200 au.
  - Candidates: the octupole (eccentric) KL effect for eccentric companions; planet–planet
    scattering triggered by the companion; selection effects in the archive.
  - High-eccentricity migration into hot Jupiters is not modelled: such planets are removed,
    because hot-Jupiter rates are already calibrated.
- **B27** Moons of warm giants: 61% of giants larger than 6 R⊕ at 0.1–1 au carry a moon of
  ≥ 0.2 R⊕, while Teachey+2018 find η < 0.38 (95%) for Galilean analogues.
  - Teachey's sample was chosen for moon detectability, which favours large planets.
  - Next step: rebuild their sample's radius distribution.
  - If the tension holds, add moon loss during disk migration (evection resonance,
    Spalding+2016).
- **B28** Moons not modelled:
  - small irregular moons (each giant has ~100 above 1 km) and captured asteroids such as
    Phobos;
  - tidal heating (Io), and the obliquity of moons and planets;
  - moons of circumbinary planets beyond the single-star treatment.
  - Moon composition uses a fixed circumplanetary ice line.
- **B29** Habitable zone:
  - Kopparapu's fits are clamped outside 2600–7200 K (7% of planets flagged: brown dwarfs, late
    M dwarfs, white dwarfs, A stars).
  - The HZ is computed for present-day luminosity. Pre-main-sequence water loss for M-dwarf
    planets (Luger & Barnes 2015) and the evolution of the HZ are not modelled; this connects
    to `age="star"` (B2).
- **B30** Spin: tidal Q and k₂ are order-of-magnitude values by planet type. Planets not yet
  locked are given only "free", with no rotation period. Atmospheric thermal tides (Venus) and
  obliquity are not modelled.

## After phase 5

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

- Phases 1–4 were merged into `main` on 2026-10-15 as a fast-forward; phase 5 is on `phase5`.
  Nothing has been pushed: `origin/main` no longer exists on the remote. `gh` is not
  installed; use the GitHub API with curl or the web UI.
