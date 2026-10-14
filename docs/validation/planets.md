# Planet validation

Phase 4 report. Each step adds a section; 4.7 adds the occurrence comparison by
stellar type and the map of where the model extrapolates.

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
`attach_planets(..., age="star")` uses each star's own age instead. This is an
extrapolation of the fitted model, so it is off by default. For a 0.5 Gyr Sun-like
star it strips far fewer envelopes. Over the 20-pc population (mean age ~4 Gyr) the
stripped fraction drops from 13.7% to 10.4%. A stellar-type-dependent XUV/bolometric
ratio is not implemented; it would need a re-fit of α_pret to be meaningful.

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

| parameter | value | meaning |
|---|---|---|
| `d_log_rate_clusters` | +1.8 | ~6× the FGK cluster rate (stability then prunes many) |
| `d_mean_ln_mass` | +0.38 | planet masses ~1.5× larger before stripping |
| `f_bare` | 0.26 | fraction of planets born without an H/He envelope |
| `min_period` | 1.51 d | inner period limit (3 d for FGK) |

The four are degenerate. Powell runs from three starts end at different points
(e.g. +1.4/+0.6/0.30/1.4 d, +1.4/+0.2/0.24/1.1 d) whose scores are within noise
(20–26). The data constrain the occurrence they produce, not each parameter.

Mid-to-late M: below 0.5 M☉, planet masses scale ∝ M★ (`mass_slope` 1, a universal
planet-to-star mass ratio, Pascucci+2018). This is a prior: slopes 0–1.5 fit Ment &
Charbonneau (2023) about equally well. The bare fraction rises to `f_bare_late` 0.9
and the period limit falls to `min_period_late` 0.75 d at 0.15 M☉, both from a grid
search against Ment & Charbonneau. The grid caps the bare fraction at 0.9: it prefers
1.0 within noise, but sub-Neptunes such as GJ 1214 b exist around 0.18 M☉ stars.

### Results (out of sample)

Early M vs DC15: score 23.2 over 33 cells (plain HM-C: 70). Integrated:

| quantity | observed | model |
|---|---|---|
| DC15, 1–4 R⊕, P < 200 d | 2.5 ± 0.2 | 2.63 |
| Hsu+2020, 1–2.5 R⊕, 2–32 d | 0.9 (Dirichlet) – 1.6 (uniform) | 0.86 |
| Hsu+2020, 0.5–4 R⊕, 0.5–256 d | 4.2 (Dirichlet) – 8.4 (uniform) | 3.83 |

The model sits near the low (Dirichlet, the authors' baseline) end of Hsu+2020. The
largest remaining residuals are about 2–3σ:

- an excess of 2–2.5 R⊕ planets at 0.5–1.7 d, where DC15 has an upper limit;
- a deficit of 2–2.5 R⊕ at 5.5–18 d.

Mid-to-late M (0.1–0.3 M☉, median 0.18) vs Ment & Charbonneau (2023), TESS, 7 planets:

| P (d) | R (R⊕) | observed | model | z |
|---|---|---|---|---|
| 0.5–7 | 0.5–4 | 0.61 +0.24/−0.19 | 1.06 | +1.7 |
| 0.5–7 | 0.5–1 | 0.134 +0.123/−0.072 | 0.54 | +2.1 |
| 0.5–7 | 1–1.5 | 0.446 +0.194/−0.143 | 0.29 | −1.1 |
| 0.5–7 | 1.5–2 | < 0.073 | 0.15 | +1.0 |
| 0.5–7 | > 2 | < 0.072 | 0.086 | +0.3 |
| 0.5–1 | 0.5–4 | 0.047 +0.019/−0.015 | 0.056 | +0.5 |
| 1–2.7 | 0.5–4 | 0.159 +0.064/−0.050 | 0.39 | +2.6 |
| 2.7–7 | 0.5–4 | 0.402 +0.162/−0.127 | 0.62 | +1.3 |

Score 18.6 over 8 bins.

Known tension: the HM-C mass distribution (σ_ln M = 1.77) is too broad to produce the
narrow 1–1.5 R⊕ peak Ment & Charbonneau see (they find a downturn below 0.9 R⊕). The
model makes too many sub-Earths and too many planets at 1–2.7 d. Overall it has
about 1.7× too many planets around mid-to-late M dwarfs (1.7σ). The
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

Fitted parameters: normalisation 0.106 and per-star Gamma dispersion k = 1.0 (jointly,
five targets below); hot-Jupiter rate; 5.2× more outer giants around hot-Jupiter hosts
(Knutson+2014: 51 ± 10% have a 1–13 MJ companion at 1–20 au); eccentricity prior
Beta(0.97, 2.36).

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
| 2–8 au, 30–6000 M⊕, per 100 stars (Fulton+2021) | 11.6 | 14.1 +2.0/−1.8 | −1.4 |
| 8–32 au (Fulton+2021) | 9.6 | 8.9 +3.0/−2.4 | +0.2 |
| % of stars with a giant at 0.23–10 au (Rosenthal+2022) | 18.6 | 17.6 +2.4/−1.9 | +0.4 |
| 0.1–5 au, 300–6000 M⊕, per 100 stars (raw CLS count, 58; near complete) | 8.8 | 8.1 ± 1.1 | +0.7 |
| % of detectable giant hosts with ≥ 2 giants (CLS 18/89) | 20.9 | 20.2 ± 4.3 | +0.2 |

Score 2.6 over 5 targets. Plain Poisson numbers score 5.6, mostly from too few
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

Tensions:

- Fulton+2021's 3.8–7.1 au rate (15.5) and their 3000–10000 d rate (12.6, vs
  Wittenmyer+2020's 6.9 +4.2/−2.1) exceed their own 2–8 au rate (14.1); the model gives
  ~5. Not used in the fit (backlog B10).
- Nielsen+2019's < 6.9% (95%) for 2–13 MJ at 3–100 au around < 1.5 M☉ assumes a
  log-flat distribution; the model (8.7%) puts most of these planets at 2–4 MJ and
  3–10 au, where GPIES is insensitive. Vigan+2021 (FGK) is +0.8σ.
- Inside 3–300 d, HM-C gives 11 planets ≥ 30 M⊕ per 100 Sun-like stars (median
  4.4 R⊕), while the CLS mass function gives 1.9 at 30–106 M⊕: HM-C's masses for
  3–6 R⊕ planets look high for RV (backlog B9).

### Stellar population (100,000 systems)

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
all planet models attached; 2026-10-14). Rates are intrinsic, per main-sequence host.

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
