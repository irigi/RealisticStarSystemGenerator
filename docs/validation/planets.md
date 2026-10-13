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
