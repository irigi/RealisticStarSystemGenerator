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
