# Phase 6 validation: systems around real stars

Phase 6 builds systems around named stars (`python -m starsys.real NAME`, package
`src/starsys/real/`):
1. the star and its companions are looked up (20-pc census, NASA Exoplanet Archive, PASTEL,
   Bright Star Catalogue, WDS, Gaia DR3);
2. the generator's stars are fitted to them by importance sampling of age, [Fe/H] and initial
   masses;
3. the generator's planets are conditioned on the known planets and on what the surveys that
   observed the star could have found (RV archives, generic RV monitoring, TESS, Kepler, Gaia
   astrometry, direct imaging).

How it is built: `DESIGN.md` §3.5. Assumptions and their consequences: `docs/ASSUMPTIONS.md`
("Real stars"). The checks below are `python -m starsys.validation.real` (stellar
benchmarks), `--injection 100` and `--holdout 30 --no-gaia`, run on 2026-10-15.

## Summary (2026-10-15)

Verdicts as in [planets.md](planets.md): match, consistent, or tension.

| Check | Model | Data | Verdict |
|---|---|---|---|
| Stellar benchmarks: 13 quantities for 11 stars (ages, radii, masses, white-dwarf Teff) | 12 within 1.1σ | literature values below | match |
| Vega age | 0.56 [0.52–0.60] Gyr | 0.455 ± 0.013 Gyr (Yoon+2010, rotating models) | tension, 2.5σ: fast rotator seen pole-on, non-rotating MIST tracks |
| Procyon B Teff | 19,000 [13,000–45,000] K | 7,740 K | wrong but inside the wide interval: IFMR without scatter (assumption) |
| Injection: 100 synthetic Sun-like stars, synthetic RV and TESS-like surveys; undetected planets per star in 4 period × 2 mass bins, posterior vs truth | all 8 bins within 1.0σ | the injected truth | match |
| Injection: cold giants (P > 1000 d, > 0.3 M_J) not detected | posterior rate 0.046; Brier 0.031 | true rate 0.03; climatology Brier 0.029 | consistent: no skill beyond the base rate, as expected when 60 RV epochs over 8 yr do not constrain them |
| Hold-out: 30 Kepler systems with ≥ 3 transiting planets, middle planet hidden; probability of a planet within 0.1 dex in P and R | mean 0.081 posterior vs 0.046 prior; posterior / prior median 1.9, > 1 in 63% | the hidden planet exists | consistent: conditioning on the siblings helps on average |
| 51 Peg b (0.47 M_J at 4.2 d, a rare hot Jupiter) | matched by a model planet in 100% of systems (kernels ×2.3) | | match |
| TRAPPIST-1 (7), Kepler-90 (8 planets) | ESS 200; 7–32% of TRAPPIST-1's planets inserted; Kepler-90 g, h (giants at 211 and 332 d) inserted in 90–100% | | consistent: the model rarely makes Kepler-90's outer giants inside an 8-planet chain |
| β Pic, HR 8799 (young A stars with imaged giants) | 33–45% inserted | | tension: the model rarely makes wide giants around A stars (no planets.md backlog item; giants are calibrated on FGK) |
| Kepler-444 (five sub-Earths in 10 d around an 11-Gyr, [Fe/H] −0.45 K dwarf, M-dwarf pair at 66 au) | age 10.8 [9.4–11.9] Gyr; Teff 5282 K vs 5046 ± 74 K; 28–75% inserted | age 11 ± 1 Gyr (Campante+2015) | Teff tension (3σ, no α-enhanced tracks); sub-Earth chains are rare in the model |

**Runtime:** 5–50 s per star with the default 80,000 prior systems (51 Peg 32 s, TRAPPIST-1
27 s, Kepler-90 36 s, β Pic 23 s, Kepler-444 47 s). Almost all of it is spent drawing the
prior planet systems; the importance weights reach ESS 200 even for 8 planets. The stellar
benchmarks take 55 s, the 100-star injection test 2 min and the 30-system hold-out 3.5 min.

## Stellar benchmarks

`python -m starsys.validation.real --stars` (Gaia on). Pull = (median − literature) /
(half the 68% interval ⊕ literature error).

| Star | Quantity | Model (16–84%) | Literature | Pull | Reference |
|---|---|---|---|---|---|
| Sun | age (Gyr) | 3.2 [1.7–4.9] | 4.57 ± 0.02 | −0.8 | Bouvier & Wadhwa 2010 |
| Sirius B | age (Gyr) | 0.20 [0.15–0.31] | 0.24 ± 0.02 | −0.4 | Liebert+2005 |
| Sirius B | Teff (K) | 28,000 [22,500–35,200] | 25,000 ± 200 | +0.5 | Barstow+2005 |
| Procyon B | Teff (K) | 19,000 [13,300–45,400] | 7,740 ± 50 | +0.7 | Provencal+2002 |
| Procyon | age (Gyr) | 2.26 [2.05–2.57] | 2.7 ± 0.3 | −1.1 | Bond+2015 |
| α Cen A | radius (R☉) | 1.177 [1.128–1.236] | 1.218 ± 0.006 | −0.7 | Kervella+2017 |
| α Cen B | radius (R☉) | 0.855 [0.839–0.872] | 0.859 ± 0.004 | −0.3 | Kervella+2017 |
| Arcturus | radius (R☉) | 25.9 [24.5–27.6] | 25.4 ± 0.2 | +0.3 | Ramírez & Allende Prieto 2011 |
| Arcturus | age (Gyr) | 9.8 [6.6–11.7] | 7.1 ± 1.5 | +0.9 | Ramírez & Allende Prieto 2011 |
| Vega | age (Gyr) | 0.56 [0.52–0.60] | 0.455 ± 0.013 | +2.5 | Yoon+2010 |
| ε Eri | age (Gyr) | 0.61 [0.43–0.82] | 0.6 ± 0.2 | 0.0 | Mamajek & Hillenbrand 2008 |
| 40 Eri B | mass (M☉) | 0.580 [0.571–0.594] | 0.573 ± 0.018 | +0.3 | Mason+2017 |
| Pollux | mass (M☉) | 2.05 [2.01–2.11] | 2.0 ± 0.3 | +0.2 | Hatzes+2012 |

- The Sun's age is the population prior narrowed by Teff, L and R. Main-sequence G dwarfs
  carry little age information, so a 2-Gyr-wide interval is the honest answer.
- Sirius B was wrong (0.58 Gyr, 15,900 K) while the Gentile Fusillo+2021 solution was used.
  That solution is contaminated by Sirius A's glare. It is now used only if P_WD > 0.75 and
  its mass agrees with the dynamical mass.

## Injection test

100 Sun-like stars drawn from the model are the truth. Their planets are "observed" with a
generic RV survey (60 epochs over 8 yr at 2 m/s) and a TESS-like 100-day campaign at
100 ppm per hour. Each star is then conditioned on the planets detected around it, with the
same surveys, and the posterior's undetected planets are counted (50 systems per star; prior
of 8,000 systems, ESS 80). 0.31 planets per star are detected.

| Period (d) | Mass (M⊕) | True undetected / star | Posterior predicted |
|---|---|---|---|
| 0.5–10 | < 10 | 0.38 ± 0.08 | 0.35 |
| 0.5–10 | > 10 | 0.00 ± 0.00 | 0.01 |
| 10–100 | < 10 | 1.37 ± 0.13 | 1.48 |
| 10–100 | > 10 | 0.07 ± 0.03 | 0.09 |
| 100–1000 | < 10 | 1.52 ± 0.12 | 1.44 |
| 100–1000 | > 10 | 0.15 ± 0.04 | 0.17 |
| > 1000 | < 10 | 0.73 ± 0.10 | 0.83 |
| > 1000 | > 10 | 0.39 ± 0.06 | 0.37 |

This checks that the survey model and the conditioning are self-consistent: the model's own
non-detections are turned into the right counts of unseen planets. It cannot test the survey
model against real surveys. That test is the planet population's validation in phases 4–5,
which was done against those surveys.

## Hold-out test

The middle planet of 30 random Kepler systems with ≥ 3 transiting planets is hidden. The
system is conditioned on the other planets (no surveys), and the posterior probability of a
planet within 0.1 dex of the hidden one in both period and radius is compared with the
unconditioned prior for the same star.
- The posterior is higher in 19 of 30 systems: median ratio 1.9, mean 0.081 vs 0.046.
- It is zero in two:
  - Kepler-444 d, 0.53 R⊕: sub-Earths are rare in the model;
  - Kepler-130 c, 2.9 R⊕ between two smaller planets.

The gain comes from the model's intra-system correlations (similar sizes, regular spacing;
phase 4). A gain of ~2 is modest, as expected: the 0.1-dex box is a small target, and those
correlations are themselves only ~2σ features of the data.

## Known limits

- Assumed or wrong by construction: see `docs/ASSUMPTIONS.md` "Real stars". The main ones:
  - no rotation and no α-enhancement in the stellar tracks (Vega, Kepler-444);
  - IFMR without scatter (Procyon B);
  - archival RVs only from VizieR;
  - one generic contrast curve for direct imaging.
- Insertion fractions show where the generator rarely makes a known system:
  - wide giants around A stars (β Pic, HR 8799);
  - outer giants in compact multis (Kepler-90 g, h);
  - sub-Earth chains (Kepler-444).

  The known planets are still placed in every system. Their unseen neighbours then come from
  systems that are less like the real one.
- Companion orbits beyond 20 pc come from one projected separation. Kepler-444's BC pair comes
  out at a = 3 au (true ~0.3 au), and A's orbit around it at e ≈ 0.6 (true 0.86, Dupuy+2016).
  The model therefore keeps unseen planets beyond ~1 au that a periastron of 5 au would remove.
