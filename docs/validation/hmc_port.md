# HM-C port validation

Command: `python -m starsys.validation.hmc --catalogs 6-25 --criterion {hill,mmr}`
(run on 2026-10-12; 20 published HM-C physical catalogs, same stars and header parameters).

KS_port_vs_pub: mean two-sample KS distance between the port and the published catalog.
KS_port_vs_port: same between two independent runs of the port (noise floor).
A faithful port has ratio ~ 1.

## amd_pair_criterion = "hill" (default; what the published catalogs actually used)

```
                          published    port  KS_port_vs_pub  KS_port_vs_port   ratio
statistic                                                                           
f_hosts                      0.8683  0.8685             NaN              NaN     NaN
n_planets|host               2.1000  2.1000          0.0028           0.0030  0.9132
log10 period                 1.7862  1.7865          0.0029           0.0026  1.1013
radius                       1.5433  1.5429          0.0032           0.0031  1.0284
log10 mass                   0.1601  0.1597          0.0035           0.0034  1.0363
period ratio                 1.8954  1.8966          0.0034           0.0036  0.9354
radius ratio                 1.0435  1.0432          0.0031           0.0033  0.9513
ecc (multis)                 0.0571  0.0571          0.0037           0.0041  0.8964
ecc (singles)                0.2949  0.2940          0.0094           0.0081  1.1656
incl_invariable (multis)     0.0280  0.0280          0.0036           0.0034  1.0683
```

## amd_pair_criterion = "mmr" (as described in Paper III/IV text)

Eccentricities and mutual inclinations of multis are off by ~17-19x the noise
(KS ~0.07 vs 0.004), everything else matches. Conclusion: the published catalogs
(and hence the ABC-fitted parameters) used min(C_coll, C_Hill), because
`relative_AMD_MMR_overlap` is redefined with the Hill-stability formula in
`src/AMD_stability/amd_stability.jl` of the SysSimExClusters He_Ford_2026 branch.
