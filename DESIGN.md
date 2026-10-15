# Realistic Star System Generator — Design

Goal: a worldbuilding generator with the highest scientific accuracy reasonably
achievable. It must

1. draw random star systems (stars, planets, moons) with realistic abundances, and
2. draw systems *conditioned* on a real star (looked up in public catalogs) and its
   known planets, accounting for which surveys looked at that star and how
   sensitive they were.

Constraints: CPU-first (numpy/scipy/numba; LightGBM where a learned model helps),
no PyTorch unless it becomes clearly necessary.

## 1. Guiding principle

The He–Ford–Ragozzine catalogs are *simulated* populations drawn from explicit,
published generative models fitted to Kepler. A black-box network trained on
them could at best rediscover those models. So the core of this project is a
**modular probabilistic simulator** built from published population models. The
catalogs are used to **validate** our reimplementation (they are exact samples of
the target distribution) and to carry the posterior uncertainty of the model
parameters (each catalog = one ABC posterior draw, recorded in its header).

Machine learning is used where it adds something:

- amortised conditioning when many planets are known (where rejection sampling fails),
- fast emulators of expensive pieces (survey detection probability),
- optionally, an empirical conditional model of the inner system learned straight
  from the catalogs, as a cross-check of the reimplementation.

## 2. Data inventory (`catalogs/`)

| Path | What | Use |
|---|---|---|
| `He_Ford_2026_HM-C/depth_min_0.29/` | Paper IV hybrid model, clustered masses (HM-C), the 100 posterior catalogs with the deepest radius valleys | **primary inner-system model** |
| `stellar_tables/` | Kepler target tables; `star_id` = 1-based row index | join stellar Teff/colour/CDPP to catalogs |
| `stellar_models/`, `papers/` | MIST, BHAC15, ATMO, white-dwarf cooling tables; PDFs of every cited paper | stellar evolution; sources |
| `mr_model/` | Ning+2018 M–R table, Zeng+2019 rocky/iron curves | mass–radius |
| `observational/` | Kepler DR25 KOI table, NASA Exoplanet Archive `ps` and `pscomppars` (date in `RETRIEVED_DATE`) | validation and real-star conditioning |

HM-C's `star_id` maps to `q1_q17_dr25_gaia_berger_fgk_HFR2020b_cleaned.csv` (86,760 stars;
checked by matching masses and radii). The earlier papers' catalogs (H20, Papers I–II) were
only cross-checks and are no longer read.

## 3. Architecture

```
StarSampler ──► Star(s) ──► PlanetSystemSampler ──► MoonSampler ──► System
   (pop. synth)      │        ├─ InnerKeplerModel (HM-C, 3–300 d)
   or RealStar       │        ├─ OuterGiantModel (RV / imaging)
   lookup            │        ├─ ColdSmallPlanetModel (microlensing)
                     │        └─ Assembly + stability (Hill/AMD)
                     └─ Multiplicity (binaries) → S-/P-type planet suppression
Conditioning: SurveySensitivity (transit/RV/astrometry/imaging) + importance
sampling / MCMC over latent structure, LightGBM-amortised for many-planet cases.
```

Each module has: (a) cited source(s), (b) explicit valid range, (c) an
extrapolation rule outside that range with the uncertainty flagged, and (d) tests
against data.

### 3.1 Stars (naturalistic abundance)

Implemented in `starsys.stars` (validated: `docs/validation/census.md`).

- Component initial-mass function taken directly from the 20-pc census
  (Kirkpatrick+2024, table 18, 0.01–8 M☉), Salpeter above 8 M☉. The primary-mass
  function is deconvolved so that primaries + companions reproduce it.
- Ages: thin disk p(age) ∝ exp(−age/5.5 Gyr) over 0–10 Gyr plus 4% thick disk
  (10–12.5 Gyr), fitted to the census white-dwarf count and Teff distribution.
  [Fe/H] from an approximate local age–metallicity relation.
- Evolution: MIST v1.2 isochrones (≥0.1 M☉), BHAC15 (0.075–0.1 M☉), ATMO 2020
  (<0.075 M☉); white dwarfs via the Cummings+2018 MIST IFMR and Montreal cooling
  tables (Bédard+2020); crude NS/BH placeholders.
- Multiplicity fraction from the census (0.1–1.3 M☉) and Duchêne & Kraus 2013 at
  the mass extremes; separations and mass ratios from Duchêne & Kraus 2013;
  hierarchical triples/quadruples with ≥3× nested separations.
- Spectral types from the Pecaut & Mamajek dwarf sequence plus luminosity class.

### 3.2 Inner planets, 3–300 d (FGK): HM-C reimplementation

Python/numba port of the Paper IV procedure (§2.2.4 there): f_swp(colour) →
ZTP clusters → ZTP planets per cluster → clustered initial masses → R(M) →
photoevaporation (Lopez+2012 / NR20) → Hill and MMR-overlap spacing → critical
AMD distributed among the planets → e, ω, i_m. Parameters are taken from the 100
catalog headers (posterior draws).

**Validation (done, `docs/validation/hmc_port.md`):** our port, run on the same
stars with each header's parameters, matches the published HM-C physical catalogs
at the noise level for multiplicity, P, R, M, period and radius ratios, e and i_m.
It also exposed that the published catalogs use the Hill-stability AMD criterion
(Julia method redefinition in `amd_stability.jl`), not the MMR-overlap one the
papers describe; the port defaults to the published behaviour.

Improvements the authors list as limitations, adopted where cheap: per-star ages
(instead of a fixed 5 Gyr; the default since 2026-10-15, `attach_planets(age="star")`, through
an effective age normalised to HM-C's Kepler fit and checked against Berger+2020's radius
valley vs age) and stellar XUV scaling with stellar type (not yet: bolometric flux already
scales per star, a separate XUV ratio would need α_pret re-fitted). HM-C's masses above ~4 R⊕
are rescaled to RV measurements (Otegi+2020; `starsys.inner.rv_masses`), as Kepler constrains
only radii there.

Attachment to model stars (`starsys.planets`): host colour from Teff via the Mamajek
dwarf sequence with a Kepler-calibrated offset and scatter; validated on Kepler stars
(`docs/validation/planets.md`).

### 3.3 Extensions beyond the fitted domain (each flagged as extrapolation)

- **M dwarfs:** f_swp → ~1, multiplicity and period distribution recalibrated to
  Kepler/TESS M-dwarf occurrence rates (Hsu, Ford & Terrien 2020, which also uses
  SysSim; Dressing & Charbonneau 2015; Ment & Charbonneau 2023 for mid-to-late M).
  Implemented in `starsys.inner.mdwarf` (validated: `docs/validation/planets.md` §4.2):
  more clusters, a born-bare fraction, a lower inner period limit, planet mass ∝ M★
  below 0.5 M☉, and disk-mass-scaled occurrence below 0.1 M☉ (flagged).
- **Short periods (0.5–3 d) for FGK stars:** Hsu+2019 occurrence below HM-C's 3 d limit and
  Sanchis-Ojeda+2014's USP trend with stellar type. Implemented in `starsys.inner.short`
  (validated: `docs/validation/planets.md`, backlog B7).
- **Hot stars (F0 and earlier, A):** f_swp extrapolated with a floor; giant planet
  occurrence vs stellar mass from RV surveys of evolved intermediate-mass stars
  (Reffert+2015; Wolthoff+2022).
- **Giant planets, 0.03–30 AU:** occurrence vs semi-major axis from Fulton+2021
  (California Legacy Survey); scaling with stellar mass and metallicity per
  Johnson+2010 and Fischer & Valenti 2005; wide orbits capped by direct-imaging rates
  (Nielsen+2019, Vigan+2021).
  Implemented in `starsys.outer.giants` (validated: `docs/validation/planets.md` §4.3):
  CLS warm/cold giants to 30 au, hot Jupiters and direct-imaging wide giants to
  100 au. The latest and largest samples find no inner-small/outer-giant correlation at
  average metallicity (Bonomo+2025), so none is imposed beyond the shared metallicity
  dependence. Giants and inner systems are reconciled by stability (AMD-Hill or
  Petrovich 2015), redrawing the inner system and the giants' eccentricities.
- **Small planets beyond 300 d:** microlensing mass-ratio function (Suzuki+2016;
  Zang+2025) near and beyond the snow line; η⊕ constraints (Bryson+2021). Packing
  enforced with the same Hill/AMD machinery.
  Implemented in `starsys.outer.small` (validated: `docs/validation/planets.md` §4.4):
  Zang+2025's lower mass-ratio Gaussian beyond the snow line, HM-C's own density at
  300 d, and a log-linear bridge between them (flagged). η⊕ comes out at photoevaporation-aware
  values (Neil & Rogers 2020), below Bryson+2021: it is set by HM-C's envelope model.
- **Binaries:** S-type stability limit (Holman & Wiegert 1999); planet suppression
  in close binaries (Moe & Kratter 2021); circumbinary (P-type) occurrence.
  Implemented in `starsys.binaries` with host rows in `starsys.hosts` (validated:
  `docs/validation/planets.md` §4.5). Transit- and microlensing-fitted rates are field
  averages over binaries, so they are boosted to single-star rates (B ≈ 1.2–1.4, from a mock
  magnitude-limited Kepler sample of the population). The inner pair hosts circumbinary
  planets: the single-star models outside the HW99 limit, plus an empirical 4–10 R⊕ component
  calibrated to Armstrong+2014.
- **Evolved stars and white dwarfs:** engulfment inside the maximum stellar radius,
  adiabatic orbit expansion by M_initial/M_final; WD planet occurrence tied to
  pollution rates.
  Implemented in `starsys.evolved` (validated: §4.6): planets drawn on the progenitor's
  main-sequence state, MIST maximum radii, Mustill & Villaver (2012) tidal engulfment
  ratios, adiabatic expansion; rare flagged pulsar planets; none for black holes.

### 3.4 Derived properties and moons (built in phase 5; `starsys.derived`, `starsys.moons`)

- **Derived planet properties:** orbit-averaged insolation from every star of the system,
  equilibrium temperature, Kopparapu+2014 habitable zone (several stars combined as
  Haghighipour & Kaltenegger 2013), the time spent in an early runaway greenhouse while the
  young star was brighter (Luger & Barnes 2015, from MIST histories), and spin: primordial
  spins (Kokubo & Genda 2010 for rocky planets) slowed by the stellar tide (Gladman+1996),
  giving tidal locking, rotation periods and obliquities. Kozai–Lidov cycles from stellar
  companions, quenched by GR and planet–planet precession. Tidal decay on the giant branches
  (Villaver & Livio 2009).
- **Moons** (theory-driven; no exomoon detections to fit):
  - giant planets: regular satellite systems with a total mass ratio ~2e-4 (Canup & Ward
    2006) and 0–5 moons (Cilibrasi+2018);
  - ice giants: captured Triton-like moons;
  - terrestrial planets: giant-impact moons in ~1 of 12 (Elser+2011), only below ~6 Earth
    masses (Nakajima+2022);
  - survival: Hill stability (Domingos+2006) and tidal migration, including the loss of moons
    around close-in planets (Barnes & O'Brien 2002);
  - tidal heating of moons held eccentric in 2:1 chains (Io-like);
  - Kepler exomoon non-detections (Teachey+2018) as an upper-limit check.

### 3.5 Conditioning on a real star

1. **Star lookup:** SIMBAD/Gaia DR3/TIC → mass, radius, Teff, age, [Fe/H],
   multiplicity, distance.
2. **Known planets:** NASA Exoplanet Archive (local copy, refreshable).
3. **Survey coverage and sensitivity, per method:**
   - Transit: TESS sectors observed (tess-point) + noise model by Tmag → SNR →
     detection probability; Kepler/K2 if targeted (DR25 completeness).
   - RV: public RV time series where available (HARPS RVBank, HIRES/Butler+2017,
     CARMENES DR1) → injection–recovery; otherwise a prior P(monitored |
     V mag, distance, spectral type) built from those surveys' target lists.
   - Astrometry: Gaia DR3 RUWE / non-single-star solutions (DR4 when released).
   - Imaging: archive of direct-imaging survey targets and contrast curves (coarse).
4. **Inference:** P(system | star, known planets, non-detections) by importance
   sampling from the prior simulator. Known planets are matched with kernels in
   (P, M sin i or R); unseen planets are weighted by (1 − p_det). For systems with
   several known planets, MCMC over latent cluster structure, plus a LightGBM-
   amortised proposal.

## 4. Phases

1. **Data layer:** catalog readers joined to stellar tables; loaders; tests (done).
2. **HM-C port + validation** against the HM-C catalogs (done).
3. **Star population synthesis** + 10/20-pc calibration (done).
4. **Extensions** (M dwarfs, giants, cold small planets, binaries, evolved stars) (done).
5. **Moons** and derived planet properties (T_eq, insolation, tidal locking,
   habitable-zone flags), plus Kozai–Lidov cycles and tidal decay around giant stars (done
   2026-10-15).
6. **Real-star conditioning** + survey-sensitivity models.
7. **Learned components** (amortised conditioning, detection emulators) — added
   where profiling shows they pay off.

Each phase ends with a validation report (plots + metrics against data).

Current status and the concrete plan for the next phase: `docs/NEXT_STEPS.md`. What the model
assumes where data run out, and the alternatives: `docs/ASSUMPTIONS.md`.
