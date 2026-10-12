# Stellar population vs the 20-pc census

Command: `python -m starsys.validation.census` (200,000 model systems; run 2026-10-12).
Census: Kirkpatrick et al. (2024, ApJS 271, 55), table 4 (components) and table 18 (IMF).

The component initial-mass function matches the census by construction (the
primary-mass function is deconvolved so primaries + companions reproduce table 18).
The comparisons below test what is *not* fitted directly: stellar evolution, ages,
remnants and multiplicity.

```

== class_fractions
               census_N  census_frac  model_frac  census_unclassified
cls                                                                  
B                     2       0.0006      0.0009                    3
A                    15       0.0042      0.0038                    3
F                    84       0.0235      0.0303                    3
G                   108       0.0302      0.0313                    3
K                   335       0.0936      0.1107                    3
M                  2233       0.6239      0.5817                    3
L                   204       0.0570      0.0503                    3
T                   345       0.0964      0.1146                    3
Y                    56       0.0156      0.0269                    3
WD                  159       0.0444      0.0428                    3
giant (III)          12       0.0034      0.0030                    3
subgiant (IV)        26       0.0073      0.0019                    3

== wd_fraction_by_progenitor_mass
            census_N  census_wd_frac  model_wd_frac
mass                                               
(0.8, 1.0]       163          0.0736         0.0200
(1.0, 1.2]       105          0.0857         0.1549
(1.2, 1.5]        79          0.4177         0.4042
(1.5, 2.0]        58          0.5345         0.6513
(2.0, 3.0]        47          0.8298         0.8177
(3.0, 8.0]        35          1.0000         0.9613

== multiplicity_by_primary_mass
             census_N  census_mf  model_mf
m1                                        
(0.0, 0.1]        612     0.0768    0.2229
(0.1, 0.2]        568     0.2342    0.2411
(0.2, 0.35]       488     0.1742    0.2588
(0.35, 0.5]       338     0.2899    0.2991
(0.5, 0.7]        248     0.4073    0.3881
(0.7, 1.0]        200     0.4400    0.4378
(1.0, 1.3]        119     0.5126    0.4962
(1.3, 1.8]         84     0.4048    0.5428
(1.8, 3.0]         63     0.2857    0.5947
(3.0, 8.0]         35     0.2286    0.7052
```

## Calibration of the local age distribution

The only fitted population parameters are the decline time of the thin-disk age
distribution (tau) and the thick-disk fraction. A uniform 0-10 Gyr thin disk
over-produces white dwarfs by ~29% and makes them too cool (median Teff 5700 K vs
6545 K). Grid over tau and thick fraction:

| tau (Gyr) | thick | WD / MS(0.1-1 Msun) | median WD Teff | q25 | q75 |
|---|---|---|---|---|---|
| inf | 0.06 | 0.0808 | 5714 | 5153 | 7370 |
| 10 | 0.03 | 0.0690 | 6058 | 5305 | 8303 |
| 7 | 0.03 | 0.0648 | 6193 | 5349 | 8647 |
| 5 | 0.06 | 0.0631 | 6279 | 5320 | 8831 |
| 5 | 0.03 | 0.0601 | 6454 | 5430 | 9114 |
| **census** | | **0.0639** | **6545** | **5391** | **9000** |

Adopted: tau = 5.5 Gyr, thick-disk fraction 0.04 (mean local stellar age ~4 Gyr).
The giant (III) fraction is insensitive to this choice and matches throughout
(model 0.0042 vs census 0.0044 giants per main-sequence star).

## Known differences

- **Y and late-T dwarfs**: the model has more than the census; the census is known
  to be incomplete for the coldest brown dwarfs.
- **Subgiants (IV)**: observers assign class IV to slightly evolved main-sequence
  stars; the model only calls post-main-sequence (MIST phase >= 2) stars IV.
- **Multiplicity at the mass extremes**: brown-dwarf and A/B-star multiplicity follows
  Duchene & Kraus (2013), above the census values, because tight companions of
  brown dwarfs and spectroscopic companions of A/B stars (and companions of white
  dwarfs) are under-counted in the census.
- **WD fraction for 0.8-1.0 Msun progenitors**: the census assigns some low-mass
  white dwarfs (likely products of binary interaction) to low-mass progenitors;
  single-star evolution cannot make those within a Hubble time.
- **Low-mass (He-core) white dwarfs** (< 0.5 Msun, ~10% of census WDs) come from
  binary interaction and are not modelled.
