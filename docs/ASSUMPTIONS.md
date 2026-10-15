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
  no greenhouse effect (Earth: 255 K vs 288 K); see A9.
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
