# Planet validation

Phase 4 report: what the planet population is checked against, and how well it does. The
summary below collects every check. The sections after it give the details step by step
(4.1–4.6); numbers in a section are from the time of that step unless it says otherwise.
The generated whole-population report, with occurrence by host type, the Kepler comparison
and the extrapolation map, is [phase4_report.md](phase4_report.md)
(`python -m starsys.validation.report`). Phase 5 (derived properties, Kozai–Lidov, moons, tidal
decay on the giant branches) is validated in [phase5.md](phase5.md).

## Phase 4 summary (2026-10-14, updated 2026-10-15 after the backlog session and step 2)

All rates are intrinsic (planets per star, no detection model). Verdicts:
- **match:** within the quoted uncertainty;
- **consistent:** within ~1.5σ, or inside a range where the data only bound the rate;
- **tension:** with a backlog item in `docs/NEXT_STEPS.md`.

| Check | Model | Data | Verdict |
|---|---|---|---|
| HM-C port vs the 100 published catalogs (distributions, KS) | at the noise floor | He & Ford 2026 | match (§4.1, `hmc_port.md`) |
| Kepler-like inner planets, whole population with binaries (mock magnitude-limited sample) | 2.35 per star at 3–300 d | published HM-C 2.38 ± 0.11 | match (report) |
| Metallicity dependence of small planets, six P–R classes (Petigura+2018) | β 0.59 / 0.96 / 1.52 (1–10 d), 0.15 / 0.33 / 1.22 (10–100 d) | 0.6 / 1.6 / 5.5, −0.3 / 0.5 / 2.1 | 3 of 6 within 1σ; hot sub-Neptunes and sub-Saturns shallower (2.1σ, 2.6σ), warm super-Earths steeper (2.3σ) (B3, §4.1) |
| Early M, 1–4 R⊕, P < 200 d; DC15 grid | 2.61; score 17.4 over 33 cells (posterior drawn per host) | 2.5 ± 0.2 (DC15) | match (§4.2, B5) |
| Early M, Hsu+2020 integrated rates | 0.86; 3.76 | 0.9–1.6; 4.2–8.4 | consistent, at the Dirichlet (low) end |
| Mid-to-late M, 0.5–4 R⊕, 0.5–7 d | 0.97 | 0.61 +0.24/−0.19 (Ment & Charbonneau 2023) | tension, 1.4σ (B4) |
| Ultracool dwarfs, K2 | within 1σ | Sestovic & Demory 2020 | consistent |
| CLS fit targets (Fulton+2021, Rosenthal+2022, raw counts, multiplicity) | score 2.6 over 5 | — | match (§4.3) |
| Giants vs [Fe/H] (Bryan & Lee 2024), Bonomo+2025, Johnson+2010 | 14.0 / 4.7 / 9.5 / 7.5% | 14.3 / 5.0 / 9.8 / 7 ± 1% | match |
| Same giant model on the population's single Sun-like stars: 2–8 au; % with a giant at 0.23–10 au | 9.7; 18.3% | 14.1 +2.0/−1.8; 17.6 +2.4/−1.9 | expected: the population's [Fe/H] distribution (mean −0.05, SD 0.20) is narrower and lower than CLS's (−0.01, 0.28), which predicts 0.85× the CLS-sample rate; observed 0.83× (report) |
| Fulton+2021 3.8–7.1 au and 3000–10000 d rates | ~5 | 15.5, 12.6 (exceed their 2–8 au rate) | tension (B10) |
| HM-C masses of 4–10 R⊕ planets, rescaled to RV (`starsys.inner.rv_masses`) | median 20 / 38 / 82 M⊕ at 4–6 / 6–8 / 8–10 R⊕ (HM-C's own: 43 / 95 / 155) | 24 / 35 / 85 M⊕ (NASA archive RV masses, biased high); Otegi+2020 | match (B9, §4.3) |
| RV occurrence at P < 50 d around GK dwarfs, M sin i 10–30 / 30–100 / 100–1000 M⊕ | 8.9 / 2.1 / 1.3% | 4.8 / 3.0 / 2.6% (Howard+2010 fit, ±~60%) | consistent; 3–10 M⊕: 20% vs 9% (Howard) and 30–50% (HARPS) |
| Direct imaging (Nielsen+2019, Bowler 2016, Vigan+2021) | within ~1σ | — | consistent; GPIES < 6.9% vs 8.7% depends on the assumed shape |
| Microlensing mass-ratio function (Zang+2025) over the measured range | within ~6% | — | match (§4.4) |
| Wide orbits, s = 2–6 (Poleski+2021) | 0.05 | 1.4 +0.9/−0.6 | tension (B15; setting `wide_factor`) |
| η⊕, conservative HZ, 0.5–1.5 R⊕ | 0.10 (setting `temperate_bare_fraction` 1: 0.33) | 0.37–0.60 (Bryson+2021); 0.008–0.076 (Neil & Rogers 2020) | tension with Bryson, match with photoevaporation-aware estimates (B16; setting) |
| Small planets 237–500 d (Hsu+2019) | 0.053 | 0.03–0.40 | consistent |
| Planets at 0.5–2 d, FGK (Hsu+2019) | within 0.89–1.04 of each bin | — | match (calibrated; B7) |
| USPs, F / G / K / Kepler-like early M (Sanchis-Ojeda+2014) | 0.14 / 0.58 / 0.92 / 1.21% | 0.15 / 0.51 / 0.83 / 1.10% | match (B8; all M dwarfs 2.7%, set by late M dwarfs, where TESS agrees) |
| HM-C at 2–8 d vs Hsu+2019 | 1–4 R⊕ at 2–4 d ×1.2–1.9; sub-Earths at 4–8 d ×0.5 | — | tension (B24) |
| Suppressed fraction of G primaries (Moe & Kratter 2021) | 21% volume-, 26% magnitude-limited | 33 ± 5%, 43 ± 7% | tension: the population has fewer close companions (B20) (§4.5) |
| Hot Jupiters, transit-like vs RV-like samples | 0.62% / 0.80% | 0.4–0.6% / ~1% | match (not tuned) |
| Transit multiplicity with the single-star boost (simple detection model), vs HM-C's Kepler-fitted field model | systems with 2 / 1 transiting planets +28% (G), +45% (K), +28% (M) | unchanged expected | tension, ~3σ for G and K (B21; setting `boost_cluster_share`) |
| Circumbinary planets, 4–6 / 6–10 / > 10 R⊕ (Armstrong+2014) | 14.6 / 7.8 / 3.7% | 14 / 10 / < 5.9% | match (calibrated; B22; setting `cb_rate_factor`) |
| Giant planets around giant stars, 89–4400 d, reweighted to Wolthoff+2022's sample (stage, mass, [Fe/H]) | 6.0–7.0% (RGB 7.1, HB 5.4–7.0 between runs) | 10.7 +2.2/−1.6% (RGB 14.2, HB 6.6) | HB match; RGB 2.6σ low (B23) (§4.6) |
| Giant planets with P ≤ 89 d around giant stars, Wolthoff-like sample | 1.5 expected hosts among 482 | none found | consistent (Poisson p = 0.23) |
| White dwarfs with planets | 43% (14% at 1–10 au) | 25–50% metal-polluted (Koester+2014) | consistent |
| Innermost white-dwarf planets (Mustill & Villaver 2012) | ~3 au (1 M☉ progenitor), ~10 au (4–8 M☉) | ~1.5 and ~10 au | consistent |

**Where the model extrapolates** (generated report, 300,000 systems, 1.6 million planets): 35%
of planets carry a flag other than `young`.
- **By flag:**
  - `multiple` (companion inside 200 au): 15%;
  - `substellar` (brown-dwarf hosts): 9%;
  - `cold_extrapolated` (beyond the microlensing-measured range): 8%;
  - `bridge`: 4%;
  - `circumbinary`: 3%;
  - `colour` (hotter than F3): 1.4%;
  - `evolved`: 1.2%;
  - `short_period`: 0.2%.
- **Main-sequence M–F stars:** 16–26% of planets inside 1 au are flagged, almost all for
  companions; 60–85% beyond 1 au, where the small planets are bridged or extrapolated.
- **Always flagged:** everything around brown dwarfs, A stars, evolved stars, white dwarfs,
  neutron stars and binary pairs.

Attaching all planet models takes ~17–21 s per 100,000 systems (12 s without the binary and
evolved-star models; drawing the stars: 1.4 s).

## 4.1 Planet-attachment glue (`starsys.planets.attach_planets`)

### Host colour from Teff

HM-C's f_swp is linear in de-reddened Gaia bp−rp. Model stars only have Teff, so the
colour comes from the Pecaut & Mamajek dwarf sequence (`bp_rp_from_teff`). Checked
against the HFR2020b Kepler stellar table (`bp_rp - e_bp_min_rp_interp` vs `teff`;
`python -m starsys.validation.planets --colour`): the Kepler colours are 0.03–0.11 mag
bluer than Mamajek at the same (Berger+2020) Teff, with a Teff-dependent offset and a
robust scatter of 0.04–0.08 mag. The Teff-binned offset is added
(4200–6400 K; bins outside are biased by the table's 0.5 < bp−rp < 1.7 cut, so the
offset tapers to 0 at 3800 and 6800 K) and the scatter is drawn per star.

The scatter matters because f_swp is clipped at 1 (bp−rp ≳ 0.94): without it the host
fraction of the Kepler stars comes out 0.0026 too high. With a Gaussian scatter the
expected bias is +0.0009 (the real residuals have heavier tails), well below the
posterior uncertainty of f_swp.

### Kepler stars through the glue vs the published HM-C catalogs

Command: `python -m starsys.validation.planets --catalogs 1-30` (2026-10-13). The
stars of each catalog are passed through `attach_planets` as single main-sequence
stars, with colour from Teff (not the measured colour) and that catalog's header
parameters. KS_glue_vs_pub: mean two-sample KS distance to the published catalog;
KS_port_vs_port: noise floor (two runs of the bare port on the measured colours).

```
                          published    glue  KS_glue_vs_pub  KS_port_vs_port   ratio
statistic                                                                           
f_hosts                      0.8680  0.8687             NaN              NaN     NaN
n_planets|host               2.3000  2.3000          0.0033           0.0030  1.0802
log10 period                 1.8033  1.8033          0.0026           0.0026  0.9972
radius                       1.5498  1.5489          0.0034           0.0030  1.1487
log10 mass                   0.1726  0.1731          0.0035           0.0033  1.0384
period ratio                 1.8942  1.8944          0.0033           0.0035  0.9558
radius ratio                 1.0421  1.0425          0.0031           0.0032  0.9641
ecc (multis)                 0.0559  0.0559          0.0038           0.0042  0.8840
ecc (singles)                0.2945  0.2941          0.0096           0.0083  1.1544
incl_invariable (multis)     0.0275  0.0275          0.0031           0.0038  0.8138
```

All distributions are at the noise level. The glue and colour mapping reproduce the
published model on Kepler-like stars.

### Photoevaporation inputs

HM-C's mass-loss timescale (Lopez+2012 / Neil & Rogers 2020) already scales with each
star's bolometric luminosity (from its R and Teff), but there is no separate XUV
term, and retention uses a fixed age (`system_age` = 5 Gyr in all 100 headers).

Since 2026-10-15 (backlog B2) `attach_planets` uses each star's own age by default
(`age="star"`), through an effective age t_eff = 1.35 × 5 Gyr × (age / 5 Gyr)^0.5
(`planets.photoevaporation_age`). The two constants are fitted to two conditions
(`python -m starsys.validation.planets --valley`, 300,000 systems):

| Super-Earths (1–1.8 R⊕) / sub-Neptunes (1.8–3.5 R⊕), P < 100 d | fixed 5 Gyr | per-star age | Observed |
|---|---|---|---|
| FGK main-sequence stars (HM-C's Kepler fit) | 0.623 | 0.618 | (the fixed-age value) |
| 5300–7000 K hosts, young (< 1 Gyr) / old | 0.97 | 0.61 | 0.61 ± 0.11 (Berger+2020, property-matched Gaia–Kepler hosts) |

HM-C's own form (q = 1, c = 1) also reproduces Berger's trend (0.50) but lowers the
field ratio to 0.53, away from the Kepler fit, because young stars dominate ⟨1/age⟩.
`age="fixed"` keeps the published behaviour; the Kepler-glue check below uses it. A
stellar-type-dependent XUV/bolometric ratio is not implemented; it would need a re-fit of
α_pret to be meaningful.

### Metallicity (backlog B3, 2026-10-15)

HM-C is a Kepler-field average with no [Fe/H] dependence. `starsys.inner.metallicity` adds
Petigura+2018's (California-Kepler Survey IV), centred on the Kepler field's mean [Fe/H]
(−0.01, LAMOST): HM-C planet masses ∝ 10^[Fe/H] (a prior: core mass ∝ solid content) and the
period power-law index tilted by −1.5 per dex (Petigura's Table 6: above the break, metal-rich
sub-Neptunes fall off as P^−0.3, metal-poor ones rise as P^+0.1). A third term, f_swpa ∝
10^(c [Fe/H]), was fitted to c = 0. Planets inside 3 d (`inner.short`) take Petigura's hot
betas directly, as weights normalised over the Kepler field's [Fe/H] distribution (RMS 0.21).

`python -m starsys.validation.planets --metallicity [--fit]`: FGK single stars (0.75–1.3 M☉) with
[Fe/H] uniform over −0.4 to +0.4, occurrence ∝ 10^(β [Fe/H]) fitted in 0.1 dex bins (HM-C planets;
the hot bin is 3–10 d here); 120,000 stars, out of sample:

| P (d) | R (R⊕) | Petigura+2018 β | model β | masses only (no period tilt) |
|---|---|---|---|---|
| 1–10 | 1–1.7 | +0.6 ± 0.2 | +0.59 | +0.14 |
| 1–10 | 1.7–4 | +1.6 ± 0.3 | +0.96 | +0.55 |
| 1–10 | 4–8 | +5.5 +1.6/−1.5 | +1.52 | +1.02 |
| 10–100 | 1–1.7 | −0.3 ± 0.2 | +0.15 | 0.00 |
| 10–100 | 1.7–4 | +0.5 ± 0.2 | +0.33 | +0.27 |
| 10–100 | 4–8 | +2.1 ± 0.7 | +1.22 | +1.12 |

Score 18.9 over six betas (masses only: 31.9; no dependence at all: 68.6). Hot sub-Neptunes
(2.1σ) and hot sub-Saturns (2.6σ, 7 planets) stay shallower than measured, warm super-Earths
steeper (2.3σ): HM-C's parameters cannot make one size class at one period range steeper on
its own.

With the Kepler field's [Fe/H] distribution (mean −0.005, RMS 0.21) the planets per star are
unchanged (2.462 at 3–300 d either way; inside 3 d within 1%); hot sub-Neptunes (3–10 d,
1.7–4 R⊕) rise 13%, from averaging a nonlinear response.

## 4.2 M dwarfs and below (`starsys.inner.mdwarf`)

Command: `python -m starsys.validation.mdwarf` (refit with `--fit`; 2026-10-13).
Model stars come from the stellar population: single main-sequence stars, intrinsic
occurrence (planets per star, no detection model), separate samples for fitting and
for the numbers below. Score = sum of squared log-space z over cells (asymmetric
1-sigma errors; one-sided penalty above upper limits).

### What plain HM-C does to M dwarfs

HM-C's linear f_swp already clips at 1 for these colours, and it gives 1.97 planets
per early M dwarf (1–4 R⊕, P < 200 d) against DC15's 2.5 ± 0.2. The shape is wrong,
though:

- **Too few 1–1.5 R⊕ planets (3–4σ).** HM-C gives every planet an initial H/He envelope
  (a 1 M⊕ planet starts at 2.1 R⊕). Only photoevaporation turns planets into bare
  rocky cores, and M-dwarf irradiation is too weak for that.
- **No planets below 3 d,** where DC15 measures 2–10% per bin.
- **Too many large planets at 60–200 d.**

The score is 70 over 33 cells.

Stronger stripping (an XUV/bolometric fluence enhancement, which M dwarfs physically
have) leaves the score flat, so it is not fitted (`d_log_alpha_pret` = 0, available as
an option).

### Calibration

Early M (0.35–0.65 M☉, median 0.47) fitted to Dressing & Charbonneau (2015) Table 4
(revised stellar radii), four parameters phased in between 4000 and 3700 K:

| parameter | best value (posterior range) | meaning |
|---|---|---|
| `d_log_rate_clusters` | +1.77 (1.39–2.06) | ~6× the FGK cluster rate (stability then prunes many) |
| `d_mean_ln_mass` | +0.48 (−0.22–1.18) | planet masses ~1.6× larger before stripping |
| `f_bare` | 0.27 (0.07–0.48) | fraction of planets born without an H/He envelope |
| `min_period` | 1.84 d (1.67–2.13) | inner period limit (3 d for FGK; USPs below it come from `inner.short`) |

The four are degenerate: the data constrain the occurrence they produce, not each parameter.
Since 2026-10-15 (backlog B5) they are drawn per host from `EARLY_M_POSTERIOR`: a 512-point
Sobol scan of the box, the best 60 re-scored on two independent star samples, and the 40 sets
within Δχ² = 4.72 (68% for four parameters) of the best kept. With the posterior drawn per host
the score is 17.4 over 33 cells (single point before: 23.2, 25.5 after the USPs of B8). The
mid-to-late M period limit was refitted with the new anchor (`min_period_late` 0.75 → 0.9 d).

Mid-to-late M: below 0.5 M☉, planet masses scale ∝ M★ (`mass_slope` 1, a universal
planet-to-star mass ratio, Pascucci+2018). This is a prior: slopes 0–1.5 fit Ment &
Charbonneau (2023) about equally well. The bare fraction rises to `f_bare_late` 0.9
and the period limit falls to `min_period_late` 0.75 d at 0.15 M☉, both from a grid
search against Ment & Charbonneau. The grid caps the bare fraction at 0.9: it prefers
1.0 within noise, but sub-Neptunes such as GJ 1214 b exist around 0.18 M☉ stars.

### Results (out of sample)

Early M vs DC15: score 17.4 over 33 cells with the posterior drawn per host (2026-10-15, B5;
plain HM-C: 70; single best point before: 23.2, 25.5 with B8's USPs). Integrated:

| quantity | observed | model |
|---|---|---|
| DC15, 1–4 R⊕, P < 200 d | 2.5 ± 0.2 | 2.61 |
| Hsu+2020, 1–2.5 R⊕, 2–32 d | 0.9 (Dirichlet) – 1.6 (uniform) | 0.86 |
| Hsu+2020, 0.5–4 R⊕, 0.5–256 d | 4.2 (Dirichlet) – 8.4 (uniform) | 3.76 |

The model sits near the low (Dirichlet, the authors' baseline) end of Hsu+2020. The
largest remaining residuals are about 2–3σ:

- an excess of 2–2.5 R⊕ planets at 0.5–1.7 d, where DC15 has an upper limit;
- a deficit of 2–2.5 R⊕ at 5.5–18 d.

Mid-to-late M (0.1–0.3 M☉, median 0.18) vs Ment & Charbonneau (2023), TESS, 7 planets:

| P (d) | R (R⊕) | observed | model | z |
|---|---|---|---|---|
| 0.5–7 | 0.5–4 | 0.61 +0.24/−0.19 | 0.97 | +1.4 |
| 0.5–7 | 0.5–1 | 0.134 +0.123/−0.072 | 0.48 | +2.0 |
| 0.5–7 | 1–1.5 | 0.446 +0.194/−0.143 | 0.27 | −1.3 |
| 0.5–7 | 1.5–2 | < 0.073 | 0.14 | +1.0 |
| 0.5–7 | > 2 | < 0.072 | 0.080 | +0.2 |
| 0.5–1 | 0.5–4 | 0.047 +0.019/−0.015 | 0.036 | −0.7 |
| 1–2.7 | 0.5–4 | 0.159 +0.064/−0.050 | 0.34 | +2.3 |
| 2.7–7 | 0.5–4 | 0.402 +0.162/−0.127 | 0.59 | +1.1 |

Score 15.3 over 8 bins (18.6 before; 20.1 with B8's USPs), with the early-M posterior and
`min_period_late` 0.9 d (2026-10-15).

Known tension: the HM-C mass distribution (σ_ln M = 1.77) is too broad to produce the
narrow 1–1.5 R⊕ peak Ment & Charbonneau see (they find a downturn below 0.9 R⊕). The
model makes too many sub-Earths and too many planets at 1–2.7 d. Overall it has
about 1.6× too many planets around mid-to-late M dwarfs (1.4σ). The
terrestrial-to-sub-Neptune ratio is about 3.5:1 (observed ~14:1).

Ultracool dwarfs (Teff 2300–3000 K, median 0.10 M☉) vs Sestovic & Demory (2020), K2,
P = 1–20 d. These are a check only; nothing was fitted here:

- 1–2 R⊕: model 0.61, observed < 0.66 (84% upper limit).
- 2–4 R⊕: model 0.09, observed 0.20 +0.16/−0.11 (about 1σ low).
- 4–6 R⊕: model 0.00, observed < 0.21.
- 34% of hosts have ≥ 3 planets inside 20 d. K2 finding no TRAPPIST-1-like system
  only excludes this being near-universal (a 20% rate would give a 46% detection
  chance). Muirhead+2015 find 21% compact multis around mid-M dwarfs.

Below 0.1 M☉ the host fraction follows f = (M/0.1)^1.9 (disk-mass scaling, Pascucci+2016),
with planets flagged `substellar`. L/T dwarfs (Teff < 2300 K, median 0.060 M☉): 36% are
hosts, and 68% of their planets are smaller than 1 R⊕ (median 0.8 R⊕). There is no
measurement to compare with.

## 4.3 Giant planets (`starsys.outer.giants`)

Run: `python -m starsys.validation.giants [--fit]` (`--fit` reproduces the shipped
parameters in ~12 min). Unless noted, rates are evaluated on the California Legacy
Survey stellar sample (Rosenthal+2021, 719 stars, CDS J/ApJS/255/8, published masses and
[Fe/H]), replicated 200 times, in M sin i using each model planet's sky inclination.

### Model

| Component | Range | Source |
|---|---|---|
| Warm/cold giants | P ≥ 10 d to 30 au, 30–6000 M⊕ | Fulton+2021 broken power law in a; mass function flat in ln M to 900 M⊕ (Fulton+2021 Fig. 6), ∝ M⁻¹ above (fitted to CLS 1–5 au counts) |
| Hot Jupiters | 0.75–10 d, ≥ 95 M⊕ | 1.0% of CLS stars (RV and TESS single-star rates ~1%); TESS/Gan+2023 stellar-mass trend; lognormal period, median 4.4 d |
| Wide giants | 10–100 au, 2–13 MJ | Nielsen+2019 all-star fit, ∝ M★² (capped at 2.5 M☉) |

Host scaling: ∝ M★ (Johnson+2010) up to the 1.68 M☉ peak, Gaussian drop (σ 0.70 M☉)
above (Wolthoff+2022); ∝ 10^(1.2 [Fe/H]) (Johnson+2010). No giants below 0.1 M☉;
giant hosts outside 0.2–3 M☉ are flagged `giant_host`.

Added in the backlog session (2026-10-15):
- **Intermediate-mass hosts** (B13/B23): a share of the warm/cold giants, rising linearly from
  0 at 1.1 M☉ to 1 at 1.6 M☉, follows Wolthoff+2022's log-normal in period (mode 793 d,
  σ_lnP 0.89) instead of Fulton's s(a). The number of giants is unchanged. Retired A stars
  have no giants inside 0.6 au (Bowler+2010), and Wolthoff's ≥ 1.4 M☉ hosts differ from
  main-sequence period distributions. Main-sequence stars of 1.5–2 M☉ now have 13.0% with a
  ≥ 0.8 MJ (M sin i) giant at 89–4400 d (Wolthoff's fit: 15.8%; before: 9.1%).
- **Correlated masses** (B13): a host's warm/cold giants draw masses through a Gaussian copula
  (ρ = 0.7; the marginal stays q(M)). Detectable pairs: Spearman 0.60 (CLS: 0.63, 24 pairs;
  independent masses: 0.14). Rosenthal+2023 (CLS IV) also find multi-giant systems heavier
  and less eccentric than single giants. The model's multis are now heavier (median M sin i
  1.03 vs 0.91 MJ). The eccentricity difference is not significant in the CLS table (KS
  p = 0.33), so the eccentricity prior does not depend on multiplicity.
- **Radii** (B12): Fortney+2007 evolution models (age, insolation, core) with heavy-element
  masses from Thorngren+2016 (57.9 (M/MJ)^0.61 M⊕, factor 1.82 spread), pinned to the
  Neptune–Saturn relation below 120 M⊕; Weiss+2013 inflation for hot giants (see Checks).

Fitted parameters: normalisation 0.106 and per-star Gamma dispersion k = 1.0 (jointly,
five targets below); hot-Jupiter rate; 7.8× (5.2× before the mass correlation) more outer giants around hot-Jupiter hosts
(Knutson+2014: 51 ± 10% have a 1–13 MJ companion at 1–20 au); eccentricity prior
Beta(0.97, 2.36), refitted in 4.5 to Beta(0.98, 2.13) once single stars carried the
binary model's boosted inner systems.

### Stability and the inner system

Of the 62 adjacent planet pairs in CLS multi-planet systems, 95% are AMD-Hill stable
(Petit+2018), 84% pass Petrovich (2015), and 95% pass either. The failures are resonant
pairs (HD 82943, HD 128311). AMD-Hill alone is unusable next to small planets: its
critical AMD goes to zero with the mass ratio, so it rejects a 0.2 au planet next to a
Jupiter at 1 au with e = 0.2. The model accepts a pair if either criterion holds.

Three ways to reconcile giants with the Kepler-fitted inner systems were tried on
100,000 systems:

| Approach | FGK inner host fraction (no giants: 0.786) | Giants at 0.1–1 au |
|---|---|---|
| Remove unstable inner planets | 0.764 | as drawn |
| Redraw the giants' orbits around a fixed inner system | 0.782 | ~3× below CLS |
| Redraw the inner system, alternating with the giants' eccentricities (adopted) | 0.778 | as drawn |

With the adopted scheme, 62% of warm-Jupiter hosts (10–200 d) have inner planets
(Huang+2016: ~50%; Wu+2023: ≥ 70 ± 16%). 8% of hot-Jupiter hosts keep inner planets
(target 12%; Wu+2023). The conditioning lowers giant eccentricities, so the prior is
fitted to give the CLS values afterwards: mean 0.23 vs 0.238, median 0.19 vs 0.179
(113 CLS giants at 0.1–10 au).

### Fit targets

| Target | Model | Observed | z |
|---|---|---|---|
| 2–8 au, 30–6000 M⊕, per 100 stars (Fulton+2021) | 11.8 | 14.1 +2.0/−1.8 | −1.3 |
| 8–32 au (Fulton+2021) | 9.2 | 8.9 +3.0/−2.4 | +0.1 |
| % of stars with a giant at 0.23–10 au (Rosenthal+2022) | 18.8 | 17.6 +2.4/−1.9 | +0.5 |
| 0.1–5 au, 300–6000 M⊕, per 100 stars (raw CLS count, 58; near complete) | 8.9 | 8.1 ± 1.1 | +0.8 |
| % of detectable giant hosts with ≥ 2 giants (CLS 18/89) | 21.0 | 20.2 ± 4.3 | +0.2 |

Score 2.6 over 5 targets (2026-10-15, with the intermediate-mass period distribution and
correlated masses; not refitted). Plain Poisson numbers score 5.6, mostly from too few
multi-giant systems.

### Checks (not fitted)

| Check | Model | Observed |
|---|---|---|
| CLS raw counts > 300 M⊕, per 100 stars: 0.03–0.1 / 0.1–0.3 / 0.3–1 / 1–3 / 3–5 au | 0.60 / 1.01 / 2.12 / 3.60 / 2.05 | 0.83 / 0.70 / 1.95 / 3.48 / 1.95 (± 0.3–0.7) |
| CLS mass function at 1–5 au, counts 300–900–3000–6000 M⊕ | 24.6, 13.7, 2.4 | 23, 14, 2 |
| % with giant at 1–10 au, 0.5–20 MJ, [Fe/H] > 0 | 14.0 | 14.3 +2.0/−1.8 (Bryan & Lee 2024); 16.4 (CLS, Van Zandt & Petigura 2024) |
| same, [Fe/H] ≤ 0 | 4.7 | 5.0 +1.6/−1.3 (Bryan & Lee 2024) |
| same, sample at M = 0.99, [Fe/H] = −0.07 | 9.5 | 9.8 +0.9/−0.8 (Bonomo+2025) |
| 1 M☉, [Fe/H] = 0: % with giant at a < 2.5 au, K > 20 m/s | 7.5 | 7 ± 1 (Johnson+2010) |
| 1.5–3 M☉: 5–13 MJ at 10–100 au, % | 5.7 | 8.9 +5.0/−3.6 (Nielsen+2019) |
| 0.8–1.2 M☉: 5–13 MJ at 10–100 au, % | 1.5 | 1.1 [0.7, 1.8] (Nielsen+2019 fit, scaled to 1 M☉) |
| 0.8–1.2 M☉: 5–13 MJ at 30–100 au, % | 0.3 | 0.6 +0.7/−0.5 (Bowler 2016, to 300 au) |
| 0.7–1.4 M☉: 1–13 MJ at 5–100 au, % | 9.6 | 5.8 +4.7/−2.8 (Vigan+2021, incl. brown dwarfs, to 300 au) |
| Log-mass Spearman correlation of detectable giant pairs | 0.60 | 0.63 (CLS, 24 pairs) |
| Cool transiting giants (insolation < 150), median radius at 30–60 / 60–120 / 120–300 / 300–1000 / 1000–4000 M⊕ (R⊕) | 6.6 / 8.2 / 11.1 / 11.5 / 11.7 | 7.0 / 8.7 / 10.9 / 11.6 / 11.9 (NASA archive; the old relation gave 6.4 / 8.8 / 11.4 / 10.9 / 10.5) |
| RV occurrence at P < 50 d, GK dwarfs, M sin i 3–10 / 10–30 / 30–100 / 100–1000 M⊕, % | 20 / 8.9 / 2.1 / 1.3 | 9.1 / 4.8 / 3.0 / 2.6 (Howard+2010 fit) |
| f(cold Jupiter \| small inner planet) / f(cold Jupiter), single 0.6–1.2 M☉ stars | 0.95 | ~0.96 ± 0.2 (Bonomo+2025, metallicity difference removed) |

Tensions:

- Fulton+2021's 3.8–7.1 au rate (15.5) and their 3000–10000 d rate (12.6, vs
  Wittenmyer+2020's 6.9 +4.2/−2.1) exceed their own 2–8 au rate (14.1); the model gives
  ~5. Not used in the fit (backlog B10).
- Nielsen+2019's < 6.9% (95%) for 2–13 MJ at 3–100 au around < 1.5 M☉ assumes a
  log-flat distribution; the model (8.7%) puts most of these planets at 2–4 MJ and
  3–10 au, where GPIES is insensitive. Vigan+2021 (FGK) is +0.8σ.
- B9, resolved 2026-10-15: HM-C draws masses and fits only radii to Kepler. Above ~4 R⊕ its
  mass–radius relation is flat, and its masses came out 1.6–3× above RV-measured ones
  (median 43 vs 24 M⊕ at 4–6 R⊕, 95 vs 35 at 6–8 R⊕). `starsys.inner.rv_masses`
  rescales HM-C masses of envelope planets by a radius-dependent factor ≤ 1 to Otegi+2020's
  volatile-rich relation (to 8 R⊕, then towards Saturn). Ranks and scatter are kept, radii
  are unchanged, and stability can only improve. HM-C planets ≥ 30 M⊕ inside 300 d drop
  from 11 to 7 per 100 Sun-like stars; the rest is the heavy tail at 3–4 R⊕, which the
  archive shares (13% above 30 M⊕). The RV occurrence at P < 50 d agrees with Howard+2010
  with either set of masses; the "CLS mass function" rate (1.9) is an extrapolation of the
  giant model to 30 M⊕, not a measurement.
- Lone and multi-giant systems: see the Model notes above (CLS IV).
- The inner-planet deficit next to giants (B11): the model's f(CJ | small planet) /
  f(CJ) is 0.95, inside Bonomo+2025's ~0.96 ± 0.2.

### Stellar population (100,000 systems)

Before the binary and evolved-star models (4.5, 4.6); the current numbers by host type are in
[phase4_report.md](phase4_report.md).

| Host mass (M☉) | Hosts | % with giant | % hot Jupiter | % wide giant | Inner planets per star (without giants) |
|---|---|---|---|---|---|
| 0.08–0.3 | 52,307 | 4.2 | 0.07 | 0.20 | 5.09 (5.11) |
| 0.3–0.6 | 30,867 | 10.3 | 0.33 | 0.87 | 4.44 (4.50) |
| 0.6–0.9 | 11,134 | 16.2 | 0.74 | 2.59 | 2.77 (2.82) |
| 0.9–1.2 | 5,562 | 21.7 | 0.93 | 5.16 | 2.23 (2.33) |
| 1.2–1.6 | 1,675 | 28.9 | 0.78 | 8.84 | 1.75 (1.90) |
| 1.6–2.5 | 501 | 26.2 | 0.20 | 14.8 | 0.89 (0.96) |
| 2.5–8 | 118 | 10.2 | 0.85 | 28.0 | 0.24 (0.24) |

The inner system loses 4% of its planets per star around Sun-like stars and 8% at
1.2–1.6 M☉, against Kepler-fitted HM-C (backlog B11). Attaching planets with giants
takes ~10 s per 100,000 systems (~6.5 s without).

## 4.4 Small planets beyond 300 d (`starsys.outer.small`)

Command: `python -m starsys.validation.small` (200,000 systems from the stellar population,
all planet models attached; 2026-10-14). Rates are intrinsic, per main-sequence host. The
numbers below predate the binary model (4.5). With it, close binaries lose planets and the
microlensing anchor carries the single-star boost. Over all main-sequence hosts η⊕ becomes
0.085 (optimistic 0.151), the Hsu+2019 rate 0.053 and the Neil & Rogers rate 0.010. The
microlensing comparison is unchanged within 6%, and the wide-orbit rate is 0.048.

### Model

Planets of 0.1–30 M⊕ from HM-C's P_max (300 d) to 10 snow-line radii,
a_sl = 2.7 au (M★/M☉). Two anchors, with the log density interpolated linearly
in ln a between them:

- **Beyond a_sl:** the lower ("super-Earth") Gaussian of Zang+2025's mass-ratio function
  (KMTNet; log q_peak = −4.67, width 0.55 dex, 0.57 ± 0.13 planets per star per dex of
  separation over −5.2 < log q < −1.5), flat in ln a. Masses scale with M★ (universal
  mass ratio; Pascucci+2018).
- **At P_max:** HM-C's own output at 150–300 d (planets per host per ln P, mean and SD of
  ln initial mass), measured per posterior catalog and Teff with the port and scaled by
  the host's f_swp. Typical values: 0.93 per host per ln P for FGK, 1.3 for M dwarfs;
  ±13% across the posterior.

Both anchors are lognormal in mass, so the bridge stays lognormal, with a median that moves
from ~1.5 M⊕ (HM-C) to ~7 M★/M☉ M⊕ (microlensing). The microlensing parameters are drawn
per system from Zang+2025's posteriors; the HM-C anchor follows the system's posterior
catalog. Radii use HM-C's mass–radius relation and photoevaporation. Planets are packed
with HM-C's 10 mutual Hill radii and AMD-Hill-or-Petrovich stability against the
outermost inner planet, the giants and each other: 98.9% of the drawn planets are kept.

### Profile across 300 d (dN/d ln P, planets < 30 M⊕)

| P (d) | early M 0.35–0.65 M☉ | K 0.65–0.85 | G 0.85–1.15 | F 1.15–1.5 |
|---|---|---|---|---|
| 75–150 | 1.06 | 0.83 | 0.69 | 0.54 |
| 150–300 | 1.12 | 0.87 | 0.73 | 0.56 |
| 300–600 | 0.43 | 0.44 | 0.43 | 0.35 |
| 600–1200 | 0.26 | 0.29 | 0.30 | 0.26 |
| 1200–2400 | 0.24 | 0.22 | 0.22 | 0.20 |
| 2400–19200 | 0.24 | 0.21–0.23 | 0.21–0.22 | 0.18–0.21 |

The density is continuous at 300 d per unit mass, but it falls by ~3× over the bridge,
because the two anchors differ. HM-C's edge is dominated by sub-Earths and Earths
(median 1.5 M⊕, a quarter below 0.5 M⊕). Microlensing finds few planets below
q ~ 1e-5 and peaks at 7 M★/M☉ M⊕. Above 2 M⊕, the drop is only ~2×. For M dwarfs the
same contrast is in the data: Suzuki+2016 compare 3.0 planets per star at DC15's peak
with 0.43 from microlensing over an equal log range.

Per host, small outer planets per star: 0.76 (0.08–0.3 M☉, mostly bare: 80%
stripped or born bare), 1.16 (0.3–0.6), 1.26 (0.6–0.9), 1.31 (0.9–1.2; 0.56 of them in
the bridge), 1.26 (1.2–1.6), 1.09 (1.6–2.5). Median mass 1.1 M⊕ (late M) to 6 M⊕
(A stars). 22% of outer small planets are flagged `bridge` and 47% `cold_extrapolated`
(mostly log q < −5.2 or a > 4 a_sl).

### η⊕ and other long-period rates

| Check | Model | Published |
|---|---|---|
| η⊕, conservative HZ, 0.5–1.5 R⊕, 4800–6300 K | 0.094 (HM-C 0.054, outer 0.040) | 0.37 +0.48/−0.21 to 0.60 +0.90/−0.36 (Bryson+2021) |
| η⊕, optimistic HZ | 0.166 | 0.58 +0.73/−0.33 to 0.88 +1.28/−0.51 (Bryson+2021) |
| planets of any size in the conservative HZ | 0.50 | — |
| 0.75–1.5 R⊕, 237–500 d, FGK | 0.056 | < 0.27 (84%); robust range 0.03–0.40 (Hsu+2019) |
| 0.8–1.2 R⊕, 292–438 d, 4700–6500 K | 0.010 | 0.008–0.009 with envelope loss, 0.076 without (Neil & Rogers 2020); 0.1 +1.9/−0.1 (Burke+2015) |

The model has about as many HZ planets as Bryson+2021, but most are 1.5–3 R⊕: HM-C
starts every planet with an H/He envelope, and at HZ irradiation few lose it. Bryson's
η⊕ assumes the radius distribution measured at shorter periods carries on unchanged. With
envelope loss modelled, Neil & Rogers (whose retention law HM-C uses) find η⊕ an order
of magnitude lower, which the model matches. So the model sits 1.3σ below Bryson's lower
bound and inside the photoevaporation-aware estimates and Hsu+2019's range. η⊕ is
therefore set by the radius (composition) model, not by the bridge (backlog B16).

### Microlensing

Mass-ratio function at projected s = 0.5–2 Einstein radii (2.5 au at 0.5 M☉), around
main-sequence 0.1–1 M☉ hosts weighted by M^0.5, per dex of q per dex of s:

| log q | Model (all) | Small planets | Zang+2025 double Gaussian |
|---|---|---|---|
| −6.0 to −5.5 | 0.149 | 0.149 | 0.081 (not measured) |
| −5.5 to −5.0 | 0.375 | 0.375 | 0.297 (not measured) |
| −5.0 to −4.5 | 0.515 | 0.515 | 0.496 |
| −4.5 to −4.0 | 0.371 | 0.371 | 0.380 |
| −4.0 to −3.5 | 0.110 | 0.089 | 0.135 |
| −3.5 to −3.0 | 0.040 | 0.001 | 0.036 |
| −3.0 to −2.5 | 0.041 | — | 0.052 |
| −2.5 to −2.0 | 0.036 | — | 0.032 |
| −2.0 to −1.5 | 0.014 | — | 0.004 |

Small planets reproduce the lower Gaussian over the measured range, by construction. The
giants (the CLS-based model of 4.3, scaled to these hosts) reproduce Zang's upper Gaussian
independently. Below log q = −5.2 the model has more planets than the median
Gaussian: the per-system parameter draws and the bridge planets (a < a_sl) both add
low-q planets.

**Wide orbits:** Poleski+2021 (OGLE, 5 planets, s = 2–6, 1e-4 < q < 0.033) find 1.4
+0.9/−0.6 planets per star; the model gives 0.04 (0.01 of them below 30 M⊕),
consistent with Zang+2025 and Suzuki+2016 at s ~ 1 and a flat extrapolation. Suzuki+2016's
function with their s^0.49 extrapolates to 0.41. This 2–3σ tension is open (backlog
B15).

Attaching all planet models takes ~15 s per 100,000 systems (~8 s without small planets).
The first use builds the HM-C edge table (~2 min for the 100 posterior catalogs, cached in
`data/outer`).

## 4.5 Binaries (`starsys.binaries`)

Command: `python -m starsys.validation.binaries` (300,000 systems; `--fit` refits the boost
correction, ~3 min; 2026-10-14). Rates are intrinsic, `evolved=None`.

### Model

| Ingredient | Rule | Source |
|---|---|---|
| Suppression | a star keeps its planetary system with probability S_bin(a) of its nearest stellar companion: 0 at ≤ 1 au, 0.15 at 10 au, 1 at ≥ 200 au, linear in log a | Moe & Kratter 2021 (MK21) |
| S-type limit | planets inside a_crit = a_b (0.464 − 0.380μ − 0.631e + …), for every companion (inner subsystem and outer companions) | Holman & Wiegert 1999 eq. 1 |
| Circumbinary hosts | the inner pair (components 0+1, both MS/pre-MS stars or brown dwarfs) is a host of mass m₀+m₁ outside a_crit = a_b (1.60 + 5.10e + …) and inside the S-type limit of a tertiary | HW99 eq. 3 |
| Circumbinary planets | single-star models outside the limit, except HM-C planets ≥ 4 R⊕; plus an empirical 4–10 R⊕ component at P/P_crit 1–2.5, none for P_bin < 7 d | Armstrong+2014 |
| Single-star boost | HM-C, the M-dwarf extension and the microlensing anchor are field averages over binaries (MK21 R_AllStars): rates × B(M) = 1/⟨S_bin⟩ over a mock magnitude-limited Kepler sample, with a Teff-dependent efficiency correction for HM-C | MK21 |
| Planes | S-type systems with a companion inside 700 au: 50% aligned with its orbit (σ 10°), else isotropic; circumbinary planets within 2° of the binary plane | Christian+2022; Kepler circumbinary planets |

The RV- and imaging-based models (CLS giants, hot Jupiters, GPIES wide giants) are already
single-star rates and are only suppressed and truncated. In HM-C, B goes into f_swpa (clipped
at 1) and the remainder into λ_c. Extra clusters often fail HM-C's stability tests, so the
correction factor (1.00 for F and G stars, 1.06–1.08 for K, 1.16–1.18 for M) is fitted until
the mock-Kepler average equals the field average.

Companion orbits now carry an isotropic orientation (`incl_sky`, `node`, `omega`), drawn
after everything else in `draw_systems`.

### Checks

Fraction of primaries without close planets because of companions (1 − ⟨S_bin⟩):

| Primaries | Volume-limited | Magnitude-limited | Mag.-limited, L ≤ 1.75 L₁ | B | MK21 (mag.-limited) |
|---|---|---|---|---|---|
| M 0.1–0.5 M☉ | 0.17 | 0.28 | 0.19 | 1.24 | 28 ± 5% |
| G 0.8–1.2 M☉ | 0.21 | 0.26 | 0.23 | 1.30 | 43 ± 7% (volume-limited 33 ± 5%) |
| A 1.6–3 M☉ | 0.28 | 0.32 | 0.30 | 1.42 | 52 ± 8% |

The population (Duchêne & Kraus 2013 / Raghavan-like period distribution, validated against the
20-pc census) has fewer close companions than the Moe & Di Stefano (2017) statistics MK21 use.
For G dwarfs it suppresses 21% instead of 33 ± 5% (volume-limited), so B is 1.30 instead of
MK21's ~1.8. The boost is self-consistent with the population (backlog B20).

Kepler-like averages (inner HM-C planets per primary at 3–300 d, mock magnitude-limited sample):

| Teff | Binaries on | Field average (binaries off) | Ratio |
|---|---|---|---|
| M 3000–3900 K | 4.54 | 4.57 | 0.99 |
| K 3900–5300 K | 2.78 | 2.80 | 0.99 |
| G 5300–6000 K | 2.39 | 2.43 | 0.98 |
| F 6000–6800 K | 1.96 | 1.94 | 1.01 |

The averages are preserved, but the multiplicity mix changes. Among hosts, the share with ≥ 5
inner planets rises from 0.49 to 0.68 (M), 0.15 to 0.32 (K) and 0.16 to 0.22 (G), because the
boost has to go into the cluster count where f_swpa is already 1 (backlog B21).

Checked against Kepler's transit multiplicities (2026-10-15, `transit_multiplicity`, 1.2 million
systems): the inner planets of all stars, binaries included, are observed from random directions
with a simple detection model (SNR > 7.1 from depth, CDPPs of Kepler targets of the same Teff,
4 years). The model without the binary model is HM-C's field average, fitted to Kepler, so it
reproduces Kepler's observables; the boosted model should keep them. It keeps the planets per
star but not the multiplicity mix:

| Share of the boost put into clusters (`boost_cluster_share`) | N2/N1 vs field model, G / K / M | planets per star vs field model, G / K / M |
|---|---|---|
| 1 (default) | +28% / +45% / +28% | −2% / −2% / −11% |
| 0.5 | 0% / +36% / +17% | −8% / −14% / −20% |
| 0 | +2% / +19% / +3% | −14% / −26% / −29% |

(N3+/N1 rises by +58% / +119% / +53% with the default.) Kepler measures planets per star to
~5% and N2/N1 to 10% (G), 13% (K) and 32% (M), so the default is the better compromise (~3σ on
the multiplicity of G and K systems), but systems around K and M dwarfs are richer than Kepler
sees. Both cannot be matched: where every star already has planets, extra planets per single
star can only come as fuller systems; a consistent fix needs HM-C re-fitted with binaries in its
forward model (won't do). `BinaryModel(boost_cluster_share=0)` is a setting for Kepler's
multiplicity mix at the cost of fewer planets (B21, tension). Kepler's own ratios for reference:
N2/N1 G 0.18, K 0.22, M 0.23; N3+/N1 0.08, 0.13, 0.19.

| Check | Model | Observed |
|---|---|---|
| Hot Jupiters, FGK, transit-like sample (all primaries, magnitude-limited) | 0.62% | 0.4–0.6% (Kepler, TESS; MK21) |
| Hot Jupiters, FGK, RV-like sample (no companion inside 200 au) | 0.80% | ~1% (CLS, RV surveys) |
| Circumbinary planets, eclipse-weighted FGK pairs (P_bin 2–100 d, limit < 300 d), P < 300 d: 4–6 R⊕ | 14.6% | 14 (50%: 11.7–26) |
| 6–10 R⊕ | 7.8% | 10.0 (50%: 8.2–16.3; 95%: 3.5–28) |
| 8–10 R⊕ | 4.8% | 6.4 (50%: 5.7–11.6) |
| > 10 R⊕ | 3.7% | 0 (95% < 5.9) |

The transit/RV hot-Jupiter discrepancy comes out of the population without tuning: the ratio is
0.78 = 1/B. The circumbinary per-binary rates (0.46, 0.11 and 0.21 per binary with
P_bin ≥ 7 d for 4–6, 6–8 and 8–10 R⊕) are calibrated to Armstrong's maximum-likelihood values.
They are large because Armstrong's rates average over all eclipsing binaries, including
P_bin < 7 d, where no circumbinary planets are found, and because only part of the
1–2.5 P_crit range lies inside 300 d. The Kepler sample is ~10 planets: these rates are
uncertain by a factor ~3 (backlog B22).

Companions of S-type planet hosts (0.6–1.4 M☉) relative to the field recover S_bin in every
separation bin (0.02, 0.08, 0.29, 0.63, 0.94 at 1–3, 3–10, 10–30, 30–100 and 100–300 au,
against S_bin 0.04, 0.11, 0.31, 0.63, 0.96 at the bin centres). The model has no free
parameter here; this checks that suppression is applied as intended.

Effect on the population (300,000 systems): 1.57 million planets instead of 1.78 million
without the binary model. 49,000 are circumbinary, and 15% of planets are flagged `multiple`
(48% before, when any companion counted). The giant eccentricity prior was refitted because
single stars now carry boosted inner systems: Beta(0.98, 2.13), was Beta(0.97, 2.36).

## 4.6 Evolved stars and remnants (`starsys.evolved`)

Command: `python -m starsys.validation.evolved` (400,000 systems, all models; 2026-10-14,
rerun 2026-10-15 with tidal decay).

### Model

- **Hosts:** subgiants, RGB, core-He-burning and AGB stars, post-AGB stars and white dwarfs with
  progenitors up to 8 M☉. Their planets are drawn by all the planet models on the
  progenitor's main-sequence state (MIST at log age = log lifetime − 0.3, from initial mass and
  [Fe/H]), with the binary model applied to the present-day orbits.
- **Maximum past radius:** the RGB and AGB tip radii come from the last MIST isochrone on which
  the star is alive (stars a few per cent heavier are then on the giant branches, which
  resolves the short tip phases). A 1 M☉ star reaches 0.80 au (172 R☉) at the RGB tip and
  1.51 au on the AGB; a 3 M☉ star 0.17 and 2.66 au.
- **Engulfment (Mustill & Villaver 2012):** a planet is lost if its initial orbit is inside
  f × R_max. f runs from 1.0 to 0.53 for Earth-mass planets and from 2.0 to 0.95 for Jupiters,
  between 1 and 5 M☉ progenitors; it is interpolated in log M★ and as
  (ln M_p / ln 318)² in planet mass. These ratios come from MV12's critical initial orbits
  (Jupiters ~3 au at 1 M☉ and ~5 au at 5 M☉; Earths 1.5–3 au) divided by their maximum AGB radii.
  White dwarfs also passed their RGB tip.
- **Tidal decay (added 2026-10-15, phase 5 / B23):** the equilibrium tide in the convective
  envelope (Villaver & Livio 2009: da/dt = −(f/τ)(M_env/M★) q(1+q)(R★/a)⁸ a, τ = [M_env R²/3L]^⅓,
  f = 1 or (P/2τ)² when τ > P/2) is integrated along each host's past post-main-sequence
  evolution, together with the expansion from mass loss. The past is read off one MIST
  isochrone: stars slightly lighter than the host show its earlier states, timed by
  homology (lifetime ∝ M^−β), anchored at the row matching the host's present L and Teff.
  M_env = M − M_He-core (helium core masses added to the MIST grid), switched on between
  Teff 6000 and 5000 K. A planet is engulfed when the star fills its Roche lobe in the
  star–planet pair (R > ~0.8 a). Stars still on the giant branches engulf by this decay
  (replacing f ≥ 1); white dwarfs by the decay or the MV12 ratios, whichever reaches further.
  Integrated to the end of the AGB, Jovian planets around 1 / 2 / 5 M☉ stars are lost from
  2.4 / 2.6 / 3.5 au and Earths from 1.2 / 1.4 / 2.2 au (MV12: ~3 and ~1.5 au at 1 M☉). The
  dissipation is suppressed for hot Jupiters around subgiants (τ > P/2), so they survive
  until the star reaches a few times their orbit, consistent with the hot Jupiters found
  around low-luminosity giants (Grunblatt+2019).
- **Orbits:** survivors move out by M_progenitor / M_now (adiabatic mass loss), times the tidal
  shrinkage; periods use the present mass.
- **Neutron stars:** 0.2% have 1–3 planets of 0.01–10 M⊕ at 0.1–1 au (PSR B1257+12-like).
  Nitu+2022 find < 0.5% among 800 pulsars. Flagged `pulsar`. Black holes have none.
- **Flags:** `evolved` on all planets of these hosts. `post_interaction` marks systems where a
  component's maximum radius exceeded 0.3 of its nearest companion's separation (Roche-lobe
  overflow, not modelled). That is 0.25% of planets.

### Checks

Numbers from 2026-10-15, after the backlog session (intermediate-mass period distribution,
correlated giant masses, per-star photoevaporation ages):

| Phase | Stars | Planets per star | % with planets | % with giant | Innermost a (au) | Median a (au) |
|---|---|---|---|---|---|---|
| subgiant (log g ≥ 3.5) | 1,068 | 3.2 | 72 | 17 | 0.013 | 0.62 |
| RGB (log g < 3.5) | 879 | 2.8 | 69 | 18 | 0.024 | 0.78 |
| core-He burning | 636 | 1.4 | 59 | 23 | 0.16 | 3.7 |
| AGB | 52 | 1.4 | 62 | 17 | 0.36 | 8.4 |
| white dwarf | 23,371 | 0.78 | 43 | 12 | 1.8 | 29 |

Giant planets (M sin i ≥ 0.8 MJ) around giant stars of 1–3 M☉, counted at 89–4400 d: Wolthoff's
detections span 89–3168 d with baselines up to ~12 yr. Until 2026-10-15 the check counted every
period above 89 d, including direct-imaging giants at 10–100 au. That gave 13.4% and an
apparent match.

| Mass | Stage | Stars | Model | Wolthoff fit on the model stars |
|---|---|---|---|---|
| 1.0–1.5 M☉ | RGB / HB | 607 / 203 | 5.3 / 3.4% | 8.8 / 8.7% |
| 1.5–2.0 M☉ | RGB / HB | 160 / 142 | 9.4 / 8.5% | 13.4 / 12.8% |
| 2.0–3.0 M☉ | HB | 234 | 8.5% | 10.0% |
| all 1–3 M☉ | RGB / HB / all | 795 / 579 / 1,374 | 6.4 / 6.7 / 6.6% | 9.8 / 10.2 / 10.0% |

Wolthoff's fit is f = 12.8% exp(−(M − 1.68)²/(2 · 0.70²)) 10^(1.38 [Fe/H]), evaluated at each
model star's mass and [Fe/H]. Before the backlog session the model was about half of it at
all masses (5.2% overall).

Reweighted to Wolthoff's sample (171 RGB stars of 1.0–1.6 M☉; 311 HB stars spread evenly over
1–3 M☉, as their HB masses are near-uniform over 1–3.5 M☉; both with their [Fe/H] distribution,
mean −0.05, SD 0.18):

| Stage | Model | Wolthoff+2022 | Expected hosts at P ≤ 89 d |
|---|---|---|---|
| RGB | 7.1% | 14.2 +4.1/−2.7% | 1.5 |
| HB | 7.0% | 6.6 +2.0/−1.3% | 0.0 |
| all | 7.0% | 10.7 +2.2/−1.6% | 1.5 (they found none: Poisson p = 0.23) |

Close-in giants, giant stars vs main-sequence stars of the same mass: P < 90 d around 1.2% of
1.0–1.5 M☉ giants (main sequence 2.5%) and 0.2% of 1.5–2.5 M☉ giants (main sequence 1.4%;
3.0–4.1% before the intermediate-mass period distribution).

- **What changed:** the intermediate-mass period distribution (§4.3) moves warm giants of
  ≥ 1.1 M☉ hosts from P < 90 d into the RV window. Main-sequence 1.5–2 M☉ stars now get 13.0%
  at 89–4400 d (fit: 15.8%; before: 9.1%). Horizontal-branch stars match.
- **What remains (B23):** RGB stars of ~1.3 M☉, 2.6σ low. There the CLS-calibrated
  main-sequence rate is ~0.65 of Wolthoff's fit. Their own RGB subset is anomalous in the same
  way (Jones+2021's common sample: 33% vs 14% from their fit, 3σ). Candidates: RV signals of
  stellar origin near 600–700 d (γ Dra, Aldebaran), the giant-mass debate (Lloyd 2011;
  Schlaufman & Winn 2013; lower true masses would widen the gap), or a real excess.

The tidal decay removes hot Jupiters once the star reaches a few times their orbit, but warm
giants at 6–20 stellar radii of 4–15 R☉ giants are beyond its reach (decay timescale > 10¹⁰ yr
even with f = 1). Their scarcity around giant stars (Johnson+2007; Wolthoff+2022: one planet
at 89 d among 37) is therefore primordial: subgiants of the same mass are too small to have
engulfed them (Bowler+2010). That is what the intermediate-mass period distribution encodes.

White dwarfs (by progenitor mass):

| Progenitor (M☉) | % with planets | % with a planet at 1–10 au | % with giant | Innermost a, 5th percentile (au) | Median innermost a (au) |
|---|---|---|---|---|---|
| 0.8–1.5 | 50 | 29 | 10 | 3.1 | 8.4 |
| 1.5–2.5 | 46 | 14 | 14 | 5.4 | 16 |
| 2.5–4 | 37 | 1.4 | 14 | 11 | 46 |
| 4–8 | 29 | 0.1 | 11 | 18 | 83 |
| all | 43 | 14 | 12 | 3.8 | 17 |

43% of white dwarfs keep planets, against 25–50% showing metal pollution (Koester+2014). The
innermost survivors lie beyond ~3 au for Sun-like progenitors and ~10 au for 4–8 M☉ (MV12: ~1.5
and ~10 au; their closer 1 M☉ value is Jovian planets stranded by tides, which needs
fine-tuning). For the Sun, Earth (engulfed inside 1.51 au) and Neptune-mass planets at 1.5 au
(engulfed inside 1.88 au) are lost; Jupiter (engulfed only inside 3.0 au) survives at 9.6 au.
MIST's AGB radii are smaller than MV12's models at high mass (2.9 vs > 5 au at 5 M☉); the
engulfment ratios are applied to MIST, so for massive progenitors the model engulfs less.

Pulsar planets: 0–3 around 690 neutron stars in 400,000 systems (0.2% rate).

## Backlog B7: planets inside 3 d (`starsys.inner.short`)

Command: `python -m starsys.validation.short` (600,000 systems; mock magnitude-limited Kepler
sample, Teff reweighted to HM-C's Kepler targets; 2026-10-15).

HM-C starts at 3 d, so FGK stars had no planets on the shortest orbits. The new component
fills 0.5 d to each host's HM-C period limit:
- **Occurrence and sizes:** Hsu+2019's Kepler/Gaia grid (the HM-C stellar sample), 0.5–1 and
  1–2 d bins.
- **2–3 d:** only what HM-C's planets at 3–4 d leave of Hsu's 2–4 d bin.
- **USPs (P < 1 d):** scaled by Sanchis-Ojeda+2014's stellar-type trend, normalised to the Kepler
  sample.
- **Properties:** rocky or Neptunian masses, circular orbits, 7° mutual inclinations; spaced 10
  mutual Hill radii from the innermost HM-C planet.
- **M dwarfs:** phased out at 1–3 d with the M-dwarf extension, whose period floor (0.75–1.5 d)
  was fitted to DC15 and Ment+2023. The USP bin (0.5–1 d) applies to every host below its
  floor (since 2026-10-15, B8; below 0.1 M☉ scaled like the host fraction).
- **Flags and boost:** flagged `short_period`; carries the transit-rate single-star boost.

| P (d) | R (R⊕) | Model | Hsu+2019 | Ratio |
|---|---|---|---|---|
| 0.5–1 | 0.5–1 / 1–2 / 4–16 | 0.0046 / 0.0041 / 0.0003 | 0.0052 / 0.0044 / 0.0003 | 0.89 / 0.94 / 0.98 |
| 1–2 | 0.5–1 / 1–2 / 2–4 / 4–16 | 0.0054 / 0.0047 / 0.0005 / 0.0006 | 0.0055 / 0.0048 / 0.0005 / 0.0007 | 0.99 / 0.97 / 1.04 / 0.95 |
| 2–4 | 0.5–1 / 1–2 / 2–4 / 4–16 | 0.037 / 0.025 / 0.0084 / 0.0038 | 0.039 / 0.020 / 0.0045 / 0.0031 | 0.95 / 1.22 / 1.88 / 1.21 |
| 4–8 (HM-C only) | 0.5–1 / 1–2 / 2–4 / 4–16 | 0.060 / 0.064 / 0.028 / 0.0035 | 0.124 / 0.064 / 0.025 / 0.0050 | 0.48 / 1.00 / 1.13 / 0.70 |

The remaining 2–4 d excess at 1–4 R⊕ and the 4–8 d deficit of sub-Earths are HM-C's own (its
3–4 d planets alone exceed Hsu's whole 2–4 d bin for 2–6 R⊕). HM-C is the fitted model there,
so they are left alone (backlog B24).

USPs (P < 1 d, 0.84–4 R⊕), % of main-sequence stars, against Sanchis-Ojeda+2014 (2026-10-15,
after B8):

| Hosts | Model | Observed |
|---|---|---|
| F (6100–7100 K) | 0.14 | 0.15 ± 0.05 |
| G (5100–6100 K) | 0.58 | 0.51 ± 0.07 |
| K (4100–5100 K) | 0.92 | 0.83 ± 0.18 |
| M, early (≥ 0.35 M☉, Kepler-like) | 1.21 | 1.10 ± 0.40 |
| M, all (3100–4100 K) | 2.66 | — |

Kepler's M dwarfs are mostly early M (DC15: median 0.5 M☉). Before B8 those had almost no
USPs (~0.3%), because the M-dwarf extension's period floor is 1.5 d there and this
component was switched off. The all-M value is set by late M dwarfs (floor 0.75 d), where
TESS agrees (Ment & Charbonneau 0.5–1 d: 0.071 vs 0.047 +0.019/−0.015 planets per star).
Sanchis-Ojeda's planets reach 4.2 h, while the model starts at 0.5 d.
