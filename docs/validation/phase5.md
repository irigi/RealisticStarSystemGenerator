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

## Summary (2026-10-15)

Verdicts as in [planets.md](planets.md): match, consistent, or tension (with a backlog item in
`docs/NEXT_STEPS.md`).

| Check | Model | Data | Verdict |
|---|---|---|---|
| Insolation from archive L and a vs `pl_insol` (5,589 planets) | median ratio 1.000 (16–84%: 0.92–1.10) | NASA Exoplanet Archive | match |
| T_eq (A = 0) vs `pl_eqt` (5,629 planets) | median ratio 1.03 | NASA Exoplanet Archive (mixed albedo choices) | match |
| Solar System HZ and spin | Venus too hot; Earth, Mars conservative HZ; Mercury 3:2; Venus, Earth free | as observed (Venus' slow spin is from atmospheric tides, not modelled) | match |
| Proxima b, TRAPPIST-1 e/f/g | conservative HZ, synchronous | conservative HZ (Kopparapu+2014); expected locked | match |
| Tidal engulfment radii, 1 M☉ white-dwarf progenitor | Jovian 2.4 au, Earth 1.2 au | ~3 and ~1.5 au (Mustill & Villaver 2012) | consistent (MIST radii) |
| Giant planets with P < 90 d around giant stars | 1.8–1.9% | almost none (Wolthoff+2022: 1 of 37 planets) | tension (B23) |
| Giant planets around giant stars, 89–4400 d | 5.2% | 10.7 +2.2/−1.6% (Wolthoff+2022) | tension, model ~2× low (B23) |
| Kozai–Lidov: giants at 0.1–5 au, mean e with / without a companion | 0.217 / 0.213 | 0.317 / 0.250 (NASA archive RV, 74 / 332 planets) | tension, weak (B26) |
| Moons, Jupiter / Saturn / Uranus analogues: total mass ratio | 1.9 / 2.2 / 1.5 × 10⁻⁴, median 3 moons | 2.1 / 2.5 / 1.0 × 10⁻⁴; 4, 7, 5 major moons | match (by construction, Canup & Ward 2006) |
| Earth analogues with a large impact moon | 9% (Moon-like ratio, ends at ~45 R⊕) | > 1 in 12, range 1/45–1/4 (Elser+2011); the Moon at 60 R⊕ | match |
| Galilean-analogue moons, warm planets 1–11 R⊕ at 0.1–1 au | 0.034 | η < 0.38 (95%), 0.16 +0.13/−0.10 (Teachey+2018) | consistent |
| Same, giants > 6 R⊕ | 0.61 | Teachey's sample favours large planets | tension (B27) |
| Giants inside 0.1 au with moons | 11% | Barnes & O'Brien 2002: only small moons survive | consistent |

**Runtime** per 100,000 systems: planets with derived properties and Kozai–Lidov 23 s
(18 s before), moons 0.1 s. First use builds the Kozai cycle table (`data/binaries`, ~6 min).

## 5.1 Tidal decay on the giant branches (B23)

Before this phase, stars still on the giant branches engulfed only what lay inside their radius.
Now the equilibrium tide is integrated along each host's past evolution (Villaver & Livio 2009),
read off a MIST isochrone (details in [planets.md §4.6](planets.md#46-evolved-stars-and-remnants-starsysevolved)).

The decay removes hot Jupiters once the star reaches a few times their orbit. It cannot touch the
warm giants at 6–20 stellar radii that make up most of the close-in planets left around giant
stars, so that tension (B23) stays.

Counting only Wolthoff's detectable periods also showed that the earlier apparent match in giant
occurrence (13.4% vs 10.7%) came from direct-imaging giants at 10–100 au. In like-for-like terms
the model is about half of the evolved-star surveys. Main-sequence stars in the same window agree
with Cumming+2008, so the gap concerns giant-star samples specifically (B23).

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
  | M (< 3900 K) | 0.54 | 0.27 | 0.91 | 89 | 66 |
  | K | 0.15 | 0.37 | 0.72 | 17 | 77 |
  | G | 0.058 | 0.21 | 0.40 | 0.2 | 94 |
  | F | 0.029 | 0.11 | 0.25 | 0 | 99 |

  The G-dwarf value of 0.058 for 0.5–1.5 R⊕ differs from §4.4's η⊕ (0.085–0.094 over 4800–6300 K)
  because both the Teff range and the HZ definition differ. Here the runaway-greenhouse limit
  depends on planet mass, and eccentricity and companion light are included. The conclusion of
  B16 is unchanged.

  Most M-dwarf HZ planets are tidally locked; Sun-like HZ planets almost never are.

- **Spin states:** synchronous rotation dominates inside 100 d for every stellar type (58–95%).
  3:2 resonances make up 5% of locked planets around M dwarfs and 15–23% around FGK stars.

- **HZ extrapolation:** 7.3% of planets get `hz_extrapolated`. They are mostly planets of brown
  dwarfs, late M dwarfs below 2600 K, white dwarfs and A stars.

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
  eccentricity than those without (0.32 vs 0.25, 74 planets). The model shows no such excess
  (B26).

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
- **Kepler limit:**
  - **All warm planets:** the model's Galilean-analogue occurrence for warm Kepler-like planets
    is 0.034, inside Teachey+2018's limit.
  - **Warm giants alone:** 0.61. Teachey's sample was selected for moon detectability, which
    favours large planets. If their 284 planets are mostly giants, the model has too many moons
    around warm giants (B27). One known mechanism is moon loss during disk migration (Spalding+2016).
- **Worldbuilding census:** 2.8 moons of ≥ 0.1 M⊕ in a conservative HZ per 1000 main-sequence stars.
