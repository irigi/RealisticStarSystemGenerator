# Assumptions

Where the published data stop, the generator still has to produce something. This file lists
every such choice: what is assumed, why no available data or paper settles it, and what the
alternative would change. It is the counterpart of the validation docs, which list what *is*
tested (`docs/validation/`).

Each entry carries its backlog number (`docs/NEXT_STEPS.md`) where it has one. Entries marked
**setting** are user-selectable, with the data-based value as default, because the literature
spans a range that matters for worldbuilding:

| Setting | Default | Range | Effect |
|---|---|---|---|
| `attach_planets(temperate_bare_fraction=...)` | 0 | 0–1 | rocky η⊕ 0.10 → 0.22 (0.5) → 0.33 (1) |
| `SmallPlanetModel(wide_factor=...)` | 1 | 1–30 | small planets at 4–10 snow-line radii of Sun-like stars: 0.3 → 1.4 (10) → 1.9 (30) per star |
| `BinaryModel(cb_rate_factor=...)` | 1 | 1/3–3 | circumbinary planets (4–10 R⊕) per suitable binary, each bin capped at 1 |
| `BinaryModel(boost_cluster_share=...)` | 1 | 0–1 | 1: Kepler's planets per star, fuller systems; 0: Kepler's multiplicity mix, 14–29% fewer planets |

Where a planet comes from a model used outside its measured range, it carries an
extrapolation flag (`starsys.flags`), so generated systems show where they rest on
assumptions.

## Stars

- **Population gaps (low impact near the Sun).** Low-mass helium-core white dwarfs from
  binary interaction (~10% of white dwarfs) are missing; neutron stars and black holes are
  placeholders (none within 20 pc). The [Fe/H]–age relation is approximate (it drives
  giant-planet occurrence); GALAH/APOGEE could replace it. The census's subgiant fraction is
  higher than ours, a labelling difference.
- **Close binaries (B20).** The population follows the 20-pc census, which has fewer close
  companions than Moe & Di Stefano 2017 assume: 21% of G primaries suppressed vs Moe & Kratter
  2021's 33 ± 5%. The single-star boost (1.30 for G stars) is computed from our own
  population, so it is self-consistent. Moving to MDS17's periods would raise the boost
  towards ~1.8 and must still pass the census check. Kept: the census is the direct
  measurement.

## Inner planets (HM-C, 3–300 d)

- **HM-C as published.** The inner system is He & Ford's HM-C, fitted to Kepler DR25 with an
  ABC forward model. Re-fitting it is out of scope (`NEXT_STEPS.md`, "Won't do"); every
  correction below is applied on top of it so that its Kepler averages stay intact.
- **Host colour (B1).** Teff → Gaia bp−rp via Mamajek plus a Kepler offset (~−0.05 mag) and a
  Gaussian scatter. Resampling the empirical residuals changes f_hosts by 0.001; the Gaussian
  needs no Kepler table at run time.
- **Effective age for photoevaporation (B2).** HM-C was fitted with every star at 5 Gyr. With
  `age="star"` (default), a star's effective age is 1.35 × 5 Gyr × (age / 5 Gyr)^0.5: it keeps
  HM-C's Kepler field ratio of super-Earths to sub-Neptunes (0.618 vs 0.623) and gives
  Berger+2020's young/old trend (0.605 vs 0.61 ± 0.11). HM-C's own 1/age scaling breaks the
  field ratio. `age="fixed"` reproduces the published catalogs.
- **Masses of 4–10 R⊕ planets (B9).** Kepler measures radii only; above ~4 R⊕ HM-C's masses
  were 2–3× too heavy. They are scaled down to Otegi+2020's volatile-rich relation (to
  Saturn above 8 R⊕), keeping each planet's rank and HM-C's scatter (`inner.rv_masses`).
  Archive RV masses are biased heavy and bound the truth from above.
- **Every planet starts with an H/He envelope (B16).** HM-C's radius model; photoevaporation
  strips the close-in ones. Temperate small planets are therefore mostly 1.5–3 R⊕ sub-Neptunes
  and η⊕ (0.5–1.5 R⊕, conservative HZ, 4800–6300 K hosts) is 0.10. Published η⊕ ranges from
  ~0.05 (photoevaporation-aware, Neil & Rogers 2020) to ~0.6 (Bryson+2021). **Setting**
  `temperate_bare_fraction`: that fraction of small planets (< 10 M⊕ initially) receiving less
  than 2 Earth fluxes is born without an envelope (rocky). 1 gives η⊕ 0.33, Bryson's lower-bound
  model (0.37 +0.48/−0.21); higher values are out of reach, because heavier cores stay above
  1.5 R⊕ even when bare.
- **2–8 d periods (B24).** HM-C and Hsu+2019 analyse the same Kepler sample and disagree: at
  3–4 d HM-C exceeds Hsu's 2–4 d bin for 2–6 R⊕ (×1.9 at 2–4 R⊕); at 4–8 d it has half of
  Hsu's sub-Earths. We follow HM-C from 3 d and Hsu below (`inner.short`). Kepler's sub-Earth
  rates are uncertain by ~30%.
- **Below 0.5 d.** No planets (Kepler-78 b at 8.5 h is rare).
- **Metallicity (B3).** HM-C is a Kepler-field average. Petigura+2018's dependence (stronger
  for larger planets and at short periods) is put into HM-C's parameters, centred on the Kepler
  field's mean [Fe/H] (−0.01): planet masses ∝ 10^[Fe/H] (a prior: core mass ∝ solids) and the
  period index tilted by −1.5 per dex (fitted). It reproduces 3 of Petigura's 6 betas within
  1σ; hot sub-Neptunes (0.96 vs 1.6 ± 0.3) and hot sub-Saturns (1.5 vs 5.5 ± 1.6) stay shallower,
  warm super-Earths steeper (0.15 vs −0.3 ± 0.2). Planets inside 3 d take Petigura's betas
  directly. M dwarfs get the same dependence (extrapolated). Planets per star of the Kepler field
  are unchanged; hot sub-Neptunes rise 13% (averaging a nonlinear response).

## M dwarfs

- **Radius distribution of mid-to-late M dwarfs (B4).** HM-C's broad mass distribution
  (σ_ln M = 1.77) makes too many sub-Earths against Ment & Charbonneau 2023's narrow
  1–1.5 R⊕ peak (~2σ, 7 planets) and the total is 1.7× high. Kept: 7 planets cannot fix a
  shape; the M-dwarf calibration (DC15, Hsu+2020) is followed.
- **Early-M parameters (B5).** The four early-M calibration parameters are degenerate: many
  combinations fit Dressing & Charbonneau 2015 equally well. They are drawn per host from the
  40 sets within 68% of the best fit (`inner.mdwarf.EARLY_M_POSTERIOR`), as HM-C's own posterior
  is drawn per system: more varied M-dwarf systems, no preferred point.
- **Fuller K and M systems (B21).** Transit surveys measure field averages over binaries, so
  single stars get a boost; where every star already has planets (K and M dwarfs, HM-C's f_swpa
  = 1), the boost can only add planets to systems. That keeps Kepler's planets per star but
  raises the share of systems with two transiting planets by ~30–45% over HM-C's Kepler-fitted
  mix (`validation.binaries`). **Setting** `BinaryModel(boost_cluster_share=...)`: 1 (default)
  keeps planets per star; 0 keeps Kepler's multiplicity mix with 14–29% fewer planets.
- **Below 0.1 M☉:** occurrence ∝ M^1.9 (disk mass), flagged; checked loosely against K2
  ultracool dwarfs (Sestovic & Demory 2020).

## Giant planets

- **Architecture (B13).** No resonant chains among giants. Masses within a system are
  correlated (ρ = 0.7; CLS Spearman 0.63). The eccentricity prior does not depend on
  multiplicity (CLS difference not significant, KS p = 0.33). Wide giants (10–100 au) scale as
  M★², capped at 2.5 M☉, from 4 detections, with no metallicity dependence.
- **2–8 au rate (B10).** Fulton+2021's 2–8 au rate sits 1.4σ above our joint fit to five CLS
  rates, and Fulton's own sub-ranges are inconsistent with it. The joint fit is kept.
- **Giant stars (B23).** Red-giant-branch stars of ~1.3 M☉ get giant planets at 7.1% against
  Wolthoff+2022's 14.2 +4.1/−2.7% (2.6σ); horizontal-branch stars and close-in giants match.
  Wolthoff's own fit is low against the same subset, and known false positives sit there.
  Kept: the main-sequence (CLS) calibration carried through stellar evolution.

## Small planets beyond 300 d

- **Bridge from 300 d to the snow line (B17).** Log density interpolated linearly in ln a
  between HM-C's edge and Zang+2025's microlensing rate; no measurement constrains it
  (flagged `bridge`).
- **Universal microlensing rate (B18).** Zang+2025's mass-ratio function (hosts mostly
  0.1–1 M☉) is applied to all host masses at a/a_sl, with no metallicity dependence; its
  parameters are drawn independently. A joint refit needs the Dryad files (see
  `NEXT_STEPS.md`).
- **Wide orbits (B15).** Flat in ln a beyond the snow line out to 10 a_sl: 0.05 planets per
  star at s = 2–6, consistent with Zang+2025 and Suzuki+2016 near s ~ 1. Poleski+2021 (5
  planets) find 1.4 +0.9/−0.6, 30× more, mostly Neptune- to Jupiter-mass. **Setting**
  `wide_factor` multiplies the small-planet (< 30 M⊕) rate beyond 4 a_sl (the end of the
  microlensing-measured range). It does not raise the giants, so it cannot reach Poleski's
  total: it makes outer systems richer in Neptunes and smaller planets. Above ~10, the
  spacing rules cap the number.
- **Orbits (B19).** Eccentricities Rayleigh σ = 0.05, inclinations σ = 1.4°, continuing HM-C's
  multis; nothing measures them. No small planets beyond 10 a_sl.

## Binaries

- **Circumbinary planets (B22).** An empirical 4–10 R⊕ component calibrated to Armstrong+2014
  (~10 Kepler planets, rate uncertain by ~3×), placed at 1–2.5 × the stability limit. Wider
  binaries get the same planets near their limit (extrapolation). **Setting** `cb_rate_factor`
  (1/3–3).
- **Suppression** by a close companion follows Moe & Kratter 2021 (0 at 1 au, 1 beyond
  200 au); planets are aligned with the companion's orbit in half the systems inside 700 au
  (Christian+2022).

## Evolved hosts

- **Small-planet envelopes (B25).** Kept as on the main sequence. Red giants emit little XUV,
  so photoevaporation is weak; bolometric loss in the giant phase is not modelled. Giant
  planets are re-inflated by the present flux.
- **Engulfment.** MIST's AGB radii at 5 M☉ are ~40% smaller than Mustill & Villaver 2012's
  models, so intermediate-mass progenitors engulf slightly less.

## Derived properties

- **Climate.** Equilibrium temperatures use a Bond albedo of 0.3 for every planet and full
  heat redistribution (`t_eq0` is the zero-albedo value). It is not a surface temperature:
  no greenhouse effect (Earth: 255 K vs 288 K). Summaries and exports label it so.
- **Illustrative surface temperature (A9).** Rocky planets get a rough mean surface
  temperature (`describe.surface_temperature`, `t_surface_illustrative_k`), assuming an
  Earth-like atmosphere wherever one is plausible: Earth's greenhouse factor 288/255 inside the
  habitable zone (at least 273 K there, the carbonate–silicate cycle), and beyond it (a
  snowball); Venus-scaled (× 2.46, a lower bound) inside the recent-Venus limit; T_eq itself
  below 0.1 M⊕ (no lasting atmosphere). Planets between the recent-Venus and runaway limits are
  "marginal". Envelope planets and giants have no surface. Not a climate model: atmospheres are
  not drawn, and tidally locked planets hide large day–night contrasts. Planets that spent
  > 100 Myr in an early runaway greenhouse are noted as likely to have lost water.
  Habitable zones are Kopparapu+2014, clamped outside 2600–7200 K
  (flagged `hz_extrapolated`; brown dwarfs, late M dwarfs, white dwarfs, A stars). Several
  stars combine as in Haghighipour & Kaltenegger 2013.
- **Spin (B30).** Rocky planets start with Kokubo & Genda 2010's giant-impact spins (~3 h,
  isotropic axes), gas-rich ones near the Solar System giants' periods; all are slowed by the
  stellar tide with order-of-magnitude Q and k₂ (rocky Q = 100, k₂ = 0.3; gaseous Q = 10⁵,
  k₂ = 0.5). Moons then take angular momentum from the spin as they migrate outward
  (`moons.attach_moons`): a Moon-like moon gives a median 13 h day (the Earth's 24 h needs the
  Moon at 60 Earth radii; the model's tides put it near 50), heavier ones often lock day and
  month. Atmospheric thermal tides (Venus) are not modelled, so moonless rocky planets keep
  their primordial ~3.5 h days.
- **Early water loss (B29).** `t_runaway_myr` is the time a planet spent inside the runaway
  greenhouse limit while its young star was brighter (MIST histories). How much water it lost
  depends on unknown initial inventories and escape efficiencies and is not computed.

## Moons

All moons are theory-driven (no exomoon is confirmed); every moon carries the `moon` flag.
- **Regular satellites:** total mass ratio ~2 × 10⁻⁴ (Canup & Ward 2006), 0–5 moons
  (Cilibrasi+2018), fixed circumplanetary ice line. Tidal heating only for moons held
  eccentric in 2:1 chains (forced e 0.001–0.01).
- **Impact moons:** ~1 in 12 rocky planets below ~6 M⊕ (Elser+2011; Nakajima+2022).
- **Not modelled (B28):** small irregular moons and captured asteroids, moon obliquities,
  moons of circumbinary planets beyond the single-star treatment.

## Kozai–Lidov

Quadrupole, test-particle cycles from stellar companions, quenched by GR and by the other
planets' precession. Galactic-tide driving of very wide binaries (Kaib+2013) is not
modelled. Planets driven into the star are removed rather than turned into hot Jupiters,
because hot-Jupiter rates are already calibrated.

## Real stars (phase 6, `starsys.real`)

- **The star is the model's star.** A real star is represented by the generator's own stellar
  models at the posterior (age, [Fe/H], initial masses), not by the catalogue values, so that
  planets, habitable zones and evolution stay consistent. Error floors of 2% (Teff), 5%
  (luminosity), 4% (radius), 3% (mass), 0.1 dex (log g) and 0.05 dex ([Fe/H]) stand for the
  models' systematics. Known consequences: MIST M dwarfs come out ~3–5% hotter than measured
  (Barnard's Star 3360 K vs 3195 K); fast rotators look older (Vega 0.56 vs 0.455 Gyr);
  white dwarfs follow the IFMR without scatter, so a white dwarf whose progenitor does not fit
  the system's age comes out wrong where no temperature is measured (Procyon B ~19,000 K vs
  7,740 K).
- **Multiplicity.** Within 20 pc the census is taken as complete for stellar companions.
  Beyond, a star is single unless the NASA archive (`sy_snum`) or the Bright Star Catalogue
  lists companions, whose masses are then drawn from the population model; their projected
  separations come from the WDS system at the star's position where it lists pairs with
  component A (an "A,BC" pair, or AB and AC at one separation, becomes a 2+1 hierarchy with the
  inner pair unresolved), else any separation. Census
  hierarchies are mapped onto the generator's chain; 2+2 systems (Castor) become chains, with
  a flag. Companion orbits come from the projected separation (WDS or census positions) with
  a random phase, eccentricity and inclination; orbits from the visual-binary catalogue (ORB6)
  would fix them but are not on VizieR.
- **Matching known planets.** Kernels of 0.08 dex in period and radius and 0.15 dex in mass
  (`ConditionModel`) decide which model planet plays a known one; its values are then
  replaced by the measured ones, and model planets closer than 2√3 mutual Hill radii are
  removed. Known planets the model has no counterpart for are inserted at a penalty equal to a
  5σ mismatch, i.e. the prior probability that the model cannot make them is taken as ~4e-6.
  Where 80,000 prior systems give fewer than 200 effective samples, kernels are widened (up to
  ×2.5) and then the insertion penalty softened (up to ×4); both factors are reported.
  Controversial planets are left out unless asked for (`include_controversial`).
- **Survey coverage is a lower bound.** Only RV data public in VizieR are used (HARPS RVBank to
  2018, HIRES to 2014, CARMENES DR1 epochs at a nominal 2 m/s); ESPRESSO and later data are not.
  Stars without archival RVs are monitored with a probability calibrated on the 20-pc census,
  with a generic survey. TESS noise follows a simple photon, background and floor budget;
  Kepler uses the DR25 detection efficiency; Gaia only where RUWE < 1.4; direct imaging only
  for imaged systems, with one typical contrast curve. Microlensing and ground-based transit
  surveys are not modelled; `TransitCampaign` adds any campaign by hand. Each survey's
  sensitivity is a deterministic expectation (no per-star noise realisation).
- **Settings:** `condition(surveys=None)` conditions on the known planets only (prior
  everywhere else); `surveys=[...]` replaces the default set; `known=[...]` replaces the
  archive's planets (e.g. to add a candidate or drop one).

## Galaxy model (starsys.galaxy)

The procedural Milky Way (star map) uses an empirical density model; every parameter and
its source is listed in the `starsys.galaxy` module docstring. In short: R0 = 8.15 kpc and
spiral arms from maser parallaxes (Reid et al. 2019, arms extended beyond the data at 10°
pitch), thin/thick disc, bar and IMF-independent normalisations from Bland-Hawthorn &
Gerhard (2016), halo profile from Juric et al. (2008) normalised to 1.4e9 Msun (Deason et al.
2019), local density 0.081 systems/pc^3 from the 20-pc census. Assumed, not measured: the
local IMF and multiplicity everywhere; one [Fe/H] gradient (-0.06 dex/kpc) for all thin-disc
ages; arm contrast ±15 % for old stars and arm confinement of stars younger than 300 Myr.

Brightness tiers (for the star map; the galaxy's content is unchanged): systems whose primary
is a living star of 1.5-3, 3-8 and >= 8 Msun are drawn in 80, 160 and 1280-pc cells; the
10-pc sectors hold the rest. The mass edges are round numbers near F0, B9 and B2 main-sequence
stars; the cell sizes give a few hundred systems per cell near the Sun. Large cells place
stars on an 8^3 sub-grid of the density. The youngest thin-disc stars are 1 Myr old in the
galaxy (the local model's 10 Myr minimum would leave no living star above ~20 Msun). The arm
profile is read from a 25-pc raster of the exact one (error < 0.015). A random system ("Next"
from the map) is every galaxy system about equally likely: a point from the density, then a
uniform pick in its cell. Within ~400 pc the map shows more O/B-class real stars than the model
(74 within 220 pc against ~21 expected): the Gould Belt / Sco-Cen excess is real and kept.

## Real stars in the star map (starsys.realstars)

Real systems come from the 20-pc census (treated as complete) and the Hipparcos Survey (complete
to V = 7.9 + 1.1|sin b| before G5, 7.3 + 1.1|sin b| later; ESA 1997). A procedural system is
removed when these catalogues would list it. Assumed: Hipparcos parallax error 0.9 mas for all
Survey stars; extinction 0.7 mag/kpc in a 100-pc dust layer; bolometric corrections from the
Mamajek dwarf table also for giants. Census entries without a catalogue Teff (most of them)
take it from their spectral type. For the map's tiers a real star counts like a procedural
one of comparable mass: by Teff (7200, 10500, 20000 K for 1.5, 3, 8 Msun; Pecaut & Mamajek
2013), cool giants (class II-III) as 1.5-3 Msun, supergiants by absolute V (< -4: >= 8 Msun).
Not represented at large distances: red giants of < 1.5 Msun stars (they are in the 10-pc
sectors, loaded only near the view centre).

## Moon tidal heating cap (phase 7)

Forced eccentricities of resonant moons are drawn (0.001-0.01), and the Io-calibrated heating
formula is capped at the energy the planet's tide feeds into the resonant moons' orbits,
sum of T n with T = 3/2 (k2/Q)_p G m^2 R_p^5 / a^6 (Yoder & Peale 1981). With the generic giant
k2/Q = 5e-6 an Io analogue gets ~5e13 W (observed ~1e14 W; Jupiter's measured k2/Q is 1.1e-5,
Lainey et al. 2009). Above ~10x Io's flux (20 W/m^2) moons are labelled super-Ios.
