# Phase 5 validation: derived properties, Kozai–Lidov, moons

Phase 5 turns the planet population into worlds:
- every planet gets an insolation, an equilibrium temperature, a habitable-zone class and a
  spin state (`starsys.derived`);
- planets in binaries undergo Kozai–Lidov cycles;
- planets get moons (`starsys.moons`);
- giant stars now drag planets in by tides (B23).

The generated report with all tables is [phase5_report.md](phase5_report.md)
(`python -m starsys.validation.report5`, 300,000 systems, 2026-10-15). The detailed checks are
`python -m starsys.validation.derived`, `python -m starsys.validation.moons` and
`python -m starsys.validation.evolved`.

## Summary (2026-10-15, updated after the backlog session the same day)

Verdicts as in [planets.md](planets.md): match, consistent, or tension (with a backlog item in
`docs/NEXT_STEPS.md`).

| Check | Model | Data | Verdict |
|---|---|---|---|
| Insolation from archive L and a vs `pl_insol` (5,589 planets) | median ratio 1.000 (16–84%: 0.92–1.10) | NASA Exoplanet Archive | match |
| T_eq (A = 0) vs `pl_eqt` (5,629 planets) | median ratio 1.03 | NASA Exoplanet Archive (mixed albedo choices) | match |
| Solar System HZ and spin | Venus too hot; Earth, Mars conservative HZ; Mercury 3:2 (rotation 1407 h); Venus, Earth free | as observed (Mercury 1407.5 h; Venus' slow spin is from atmospheric tides, not modelled; the Earth's 24 h day from the Moon) | match |
| Proxima b, TRAPPIST-1 e/f/g | conservative HZ, synchronous | conservative HZ (Kopparapu+2014); expected locked | match |
| Tidal engulfment radii, 1 M☉ white-dwarf progenitor | Jovian 2.4 au, Earth 1.2 au | ~3 and ~1.5 au (Mustill & Villaver 2012) | consistent (MIST radii) |
| Giant planets with P ≤ 89 d around giant stars, Wolthoff-like sample | 1.5 expected hosts among 482 | none (Wolthoff+2022) | consistent (p = 0.23) |
| Giant planets around giant stars, 89–4400 d, reweighted to Wolthoff's sample | HB 7.0%, RGB 7.1%, all 7.0% | HB 6.6 +2.0/−1.3%, RGB 14.2 +4.1/−2.7%, all 10.7 +2.2/−1.6% | HB match; RGB tension, 2.6σ (B23) |
| Kozai–Lidov: giants at 0.1–5 au, mean e with / without a companion | 0.211 / 0.222 | 0.317 / 0.250 (NASA archive RV, 74 / 332 planets; KS p = 0.12, Mann–Whitney p = 0.05) | consistent: the archive difference is ~2σ (B26 closed) |
| Moons, Jupiter / Saturn / Uranus analogues: total mass ratio | 1.9 / 2.2 / 1.5 × 10⁻⁴, median 3 moons | 2.1 / 2.5 / 1.0 × 10⁻⁴; 4, 7, 5 major moons | match (by construction, Canup & Ward 2006) |
| Earth analogues with a large impact moon | 9% (Moon-like ratio, ends at ~45 R⊕) | > 1 in 12, range 1/45–1/4 (Elser+2011); the Moon at 60 R⊕ | match |
| Galilean-analogue moons of warm planets, weighted by the radii of warm Kepler candidates: all / smaller / larger half | 0.08 / 0.01 / 0.15 | 0.16 +0.13/−0.10 (< 0.38) / [0.06, 0.35] / [0.07, 0.44] (Teachey+2018 Table 1) | consistent (B27 closed) |
| Giants inside 0.1 au with moons | 11% | Barnes & O'Brien 2002: only small moons survive | consistent |
| Early runaway greenhouse of conservative-HZ planets, median by host mass 0.1–0.2 / 0.35–0.5 / 0.65–0.8 / > 0.8 M☉ | 79 / 21 / 0 / 0 Myr (84th pct 180 / 64 / 3 / 0) | tens to hundreds of Myr in M-dwarf HZs, a few tens for K dwarfs, negligible above 0.8 M☉ (Luger & Barnes 2015, Fig. 4) | consistent |
| Io-like moons: tidal heat of Io itself | 9e13 W (2.2 W/m²) | ~1e14 W, ~2.5 W/m² | match |

**Runtime** per 100,000 systems: planets with derived properties and Kozai–Lidov 23–24 s
(18 s before phase 5), moons 0.1 s. First use builds the Kozai cycle table (`data/binaries`, ~6 min).

## 5.1 Tidal decay on the giant branches (B23)

Before this phase, stars still on the giant branches engulfed only what lay inside their radius.
Now the equilibrium tide is integrated along each host's past evolution (Villaver & Livio 2009),
read off a MIST isochrone (details in [planets.md §4.6](planets.md#46-evolved-stars-and-remnants-starsysevolved)).

The decay removes hot Jupiters once the star reaches a few times their orbit. It cannot touch the
warm giants at 6–20 stellar radii that make up most of the close-in planets left around giant
stars, so that tension (B23) stays.

Counting only Wolthoff's detectable periods also showed that the earlier apparent match in giant
occurrence (13.4% vs 10.7%) came from direct-imaging giants at 10–100 au. In like-for-like terms
the model was about half of the evolved-star surveys (B23).

**Backlog session (later the same day).** Two changes narrowed B23:
- Giants of intermediate-mass hosts now follow Wolthoff's log-normal period distribution (mode
  793 d), phased in over 1.1–1.6 M☉ (see [planets.md §4.3](planets.md#43-giant-planets-starsysoutergiants)). This moves warm giants out of
  the P < 90 d range and into the RV window.
- The check is reweighted to Wolthoff's sample: 171 RGB stars of ~1.0–1.6 M☉ and 311 HB
  stars spread over 1–3 M☉, with their [Fe/H] distribution (mean −0.05, SD 0.18; the model's
  RGB stars are older and 0.05 dex poorer).

| Stage | Model | Wolthoff+2022 | Expected hosts at P ≤ 89 d (they found none) |
|---|---|---|---|
| RGB | 7.1% | 14.2 +4.1/−2.7% | 1.5 |
| HB | 7.0% | 6.6 +2.0/−1.3% | 0.0 |
| all | 7.0% | 10.7 +2.2/−1.6% | 1.5 (Poisson p of none: 0.23) |

The horizontal branch and the close-in giants now agree. The RGB stars (~1.3 M☉) remain 2.6σ low
(the model's run-to-run scatter is ~1%).
There, the CLS-calibrated main-sequence rate is ~0.65 of Wolthoff's fit, and their own RGB
subset is anomalous in the same way (Jones+2021's common sample: 33% against 14% from their
fit, 3σ). False positives near 600–700 d and the giant-mass debate are the candidates (B23).

## 5.2 Derived properties (`starsys.derived`)

### Model

- **Insolation:** orbit-averaged L/(a²√(1−e²)) from the host plus every other star of the system,
  each at the separation of the wider of the two stars' orbits. Circumbinary planets see both
  inner stars from the barycentre.
- **Habitable zone:** Kopparapu+2014, including the planet-mass dependence of the runaway
  greenhouse. Several stars are combined as in Haghighipour & Kaltenegger (2013).
- **T_eq:** full redistribution, A = 0.3 (`t_eq`) and A = 0 (`t_eq0`).
- **Tidal locking:** Gladman+1996 despinning time, with Q, k₂, moment of inertia and initial spin
  by planet type (rocky, sub-Neptune, giant). Planets locked by the system's age are
  synchronous; with e ≥ 0.1 they are in a 3:2 resonance like Mercury.
- **Giant re-inflation:** giants of evolved hosts take the radius of the present flux
  (Lopez & Fortney 2016; Grunblatt+2016).

### Checks

- **Formulae:** recomputing insolation from the archive's L and a reproduces `pl_insol` with a
  median ratio of 1.000. The zero-albedo T_eq is 3% above `pl_eqt`, because some papers quote a
  non-zero albedo.

- **Habitable-zone planets per main-sequence star**, conservative HZ (S-type planets):

  | Host | 0.5–1.5 R⊕ | 1.5–2.5 R⊕ | all sizes | % locked | % with T_eq 200–300 K |
  |---|---|---|---|---|---|
  | M (< 3900 K) | 0.55 | 0.27 | 0.91 | 87 | 66 |
  | K | 0.16 | 0.37 | 0.72 | 13 | 76 |
  | G | 0.063 | 0.19 | 0.40 | 0.3 | 94 |
  | F | 0.023 | 0.12 | 0.26 | 0 | 100 |

  The G-dwarf value of 0.063 for 0.5–1.5 R⊕ differs from §4.4's η⊕ (0.085–0.094 over 4800–6300 K)
  because both the Teff range and the HZ definition differ. Here the runaway-greenhouse limit
  depends on planet mass, and eccentricity and companion light are included. The conclusion of
  B16 is unchanged.

  Most M-dwarf HZ planets are tidally locked; Sun-like HZ planets almost never are.

- **Spin states:** synchronous rotation dominates inside 100 d for every stellar type (54–95%).
  3:2 resonances make up 5% of locked planets around M dwarfs and 15–23% around FGK stars.

- **HZ extrapolation:** 7.2% of planets get `hz_extrapolated`. They are mostly planets of brown
  dwarfs, late M dwarfs below 2600 K, white dwarfs and A stars.

### Added in the backlog session (2026-10-15)

- **Rotation and obliquity** (`rotation_h`, `obliquity_deg`; B30):
  - **Primordial spins:** rocky planets start with Kokubo & Genda 2010's giant-impact spins.
    Each spin-vector component is Gaussian, with an RMS of 0.53 of the breakup rate (2.35 rad/h,
    a ~3 h day, for the Earth), and the axes are isotropic. Gas-rich planets start near the
    Solar System giants' periods (10 h and 16 h, 0.15 dex).
  - **Despin:** all are slowed linearly by the stellar tide (the same constant-Q model as
    t_lock), down to synchronous or 3:2 rotation. Locked planets have zero obliquity.
  - **Census of free planets:** rocky median 3.5 h (16–84%: 2.1–7.3 h), half of them
    retrograde; sub-Neptunes 17 h; giants 10 h.
  - **Not included:** spin-down by moons (the Earth's 24 h comes from the Moon) and
    atmospheric tides.
  - **Effect on locking:** faster primordial spins lengthen t_lock (∝ ω₀). The locked share
    of K-dwarf HZ planets falls from 17% to ~13%; M dwarfs stay at ~87%.
- **Early runaway greenhouse** (`t_runaway_myr`; B29):
  - **Method:** the time after formation (10 Myr) that a planet spent inside the
    runaway-greenhouse limit while its star was brighter. It uses the host's MIST L and Teff
    history at 40 log-spaced steps and the planet's present orbit.
  - **Result:** conservative-HZ planets below 10 M⊕ have medians of 79 Myr (0.1–0.2 M☉),
    40 (0.2–0.35), 21 (0.35–0.5), 4 (0.5–0.65) and 0 above 0.65 M☉. This follows Luger &
    Barnes 2015's Fig. 4: tens to hundreds of Myr through M-dwarf habitable zones, during
    which planets may lose several oceans.

## 5.3 Kozai–Lidov cycles (`starsys.binaries`, `planets._kozai`)

### Model

- **Which planets cycle:** S-type planets inclined 39.2–140.8° to their nearest companion's
  orbit. The quadrupole timescale (Antognini 2015) must be shorter than the age, the GR
  precession time, and the apsidal precession forced by the host's other planets
  (Laplace–Lagrange).
- **Effect:** e and i are drawn at a random phase of the test-particle cycle. The eccentricity
  quantiles come from integrating the secular equations on a grid of initial inclinations and
  eccentricities, pooled over the argument of pericentre. The Kozai constant √(1−e²) cos i is
  conserved.
- **Removal:** planets are removed if their pericentre enters 2 R★ or the Roche limit, their
  apocentre crosses the HW99 limit, or they become unstable with a neighbour. No new hot Jupiters
  are made.

### Results

- **How many cycle:** 0.9% of S-type planets and 3.2% of giants. The cycles remove 4.9 per 1000
  planets.
- **Why so few:** GR precession quenches the cycles for most giants at 0.1–5 au. At 1 au around a
  Sun-like star GR turns the orbit by a radian in ~5 Myr, so only companions within ~200 au are
  fast enough. Inner planets tied to giants or to each other are quenched as well.
- **Eccentricity fit:** the giant eccentricity prior was not refitted, because so few giants cycle.
  It was fitted on single stars anyway.
- **Against the archive:** RV giants with a known stellar companion have a higher mean
  eccentricity than those without (0.32 vs 0.25, 74 planets). The model shows no such excess.
  The archive difference is only ~2σ (KS p = 0.12, Mann–Whitney p = 0.05). It is not
  monotonic in planet mass (above 1500 M⊕ both samples have 0.33–0.34). The archive's
  companion sample is also inhomogeneous: hosts with trends or eccentric planets get more
  imaging follow-up. B26 is closed as consistent. Wide binaries perturbed by the Galactic tide
  (Kaib+2013) are a known mechanism the model does not include.

## 5.4 Moons (`starsys.moons`)

### Model

All moons are theory-driven and flagged `moon`.

- **Regular satellites** of planets with an envelope and at least 10 M⊕:
  - total mass log-normal around 2 × 10⁻⁴ of the planet (Canup & Ward 2006);
  - 0–5 moons per planet, from Cilibrasi+2018;
  - masses split by a Dirichlet distribution;
  - near-resonant chains from 4–8 R_p;
  - icy outside a circumplanetary ice line.
- **Giant-impact moons** of rocky planets of 0.1–6 M⊕, with a probability of 1/12 (Elser+2011;
  Nakajima+2022).
- **Triton-like captured moons** of ice giants, with a probability of 0.25.
- **Survival:**
  - moons migrate by planetary tides, outward, then inward once the star has locked the planet
    (Barnes & O'Brien 2002);
  - they must stay inside the Hill-stability limit (Domingos+2006) and outside the Roche limit.

### Checks

- **Solar System analogues:** these reproduce the mass ratios by construction. Earth analogues
  get a Moon-like moon 9% of the time; it migrates to ~45 R⊕ in 4.6 Gyr. Mercury analogues never
  keep one.
- **Survival vs orbit:**
  - giants beyond 0.3 au keep moons ~80% of the time, those inside 0.1 au 11%;
  - rocky planets around M dwarfs keep impact moons only beyond ~0.3 au, because tidal locking
    makes the moons spiral in.
- **Kepler limit (Teachey+2018):**
  - **Their sample:** Teachey's 284 planets are warm Kepler candidates (P < 397 d), and those
    are mostly sub-Neptunes. DR25 candidates at 0.1–1 au around FGK stars have a median radius
    of 2.5 R⊕, and 8% are above 6 R⊕.
  - **Model, weighted by those radii:** η = 0.08 for all of them, 0.01 for the smaller half
    and 0.15 for the larger half.
  - **Teachey's Table 1:** 0.16 +0.13/−0.10 (< 0.38) overall; [0.06, 0.35] for the smaller
    half (< 0.48); [0.07, 0.44] for the larger half (< 0.61).
  - **Giants alone:** the model gives 0.57 for warm giants above 6 R⊕. Their sample
    cannot isolate those, so B27 is closed as consistent.
  - **Effect of the B9 mass rescaling:** with lighter 4–10 R⊕ planets, 4–11 R⊕ planets now
    carry Galilean-size moons less often (0.26, from 0.45).
- **Tidal heating** (B28): regular moons in a 2:1 chain keep a forced eccentricity of
  0.001–0.01 (the Galilean range). They dissipate E = (21/2)(k₂/Q) G M_p² R⁵ n e²/a⁶, with
  Io's k₂/Q = 0.015 (Lainey+2009), which gives Io 9e13 W. About 26 moons per 1000 planets
  exceed the Earth's internal heat flow (0.09 W/m²).
- **Worldbuilding census:** 2.8 moons of ≥ 0.1 M⊕ in a conservative HZ per 1000 main-sequence stars.
