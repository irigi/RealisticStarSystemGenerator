# Next steps (as of 2026-10-12)

Status: phases 1–3 of `DESIGN.md` are done and committed on branch `rewrite-starsys`
(data layer, validated HM-C port, stellar population validated against the 20-pc
census). Next is **phase 4: attaching planets to stars**.

## Resuming

```bash
source .venv/bin/activate
pytest -q                                  # 17 tests, all should pass
python -m starsys.validation.census        # stellar population vs 20-pc census
python -m starsys.validation.hmc --catalogs 6-25 --criterion hill   # HM-C port check
```

`catalogs/` and `data/` are gitignored. If `data/` is lost, rebuild the SysSim
parquet cache with `python -m starsys.cache <models>`; the MIST grid and primary-mass
weights under `data/stellar/` are rebuilt on first use.

## Phase 4, in recommended order

Each step gets a cited source, a valid range, an extrapolation rule (flagged in the
output), and a validation check, as `DESIGN.md` §3 requires.

### 4.1 Planet-attachment glue (do first)

- New module `starsys.planets` with `attach_planets(systems, components, seed)`:
  for each star, pick the applicable planet models, run them, and merge the results
  into one planet table keyed by (system, component).
- Map model stars to HM-C inputs: HM-C wants de-reddened Gaia `bp_rp`. Derive it
  from Teff with the Mamajek table (it has a Bp−Rp column). Check this mapping
  against the Kepler stellar table (`teff` vs `bp_rp - e_bp_min_rp_interp`).
- HM-C validity domain: main-sequence FGK, bp−rp 0.5–1.7, P 3–300 d, R 0.5–10 R⊕.
  Outside it, use the extensions below or extrapolate with a flag.
- Check how HM-C's photoevaporation uses age and stellar XUV. If it assumes a fixed
  age, add per-star age and luminosity scaling as an option (the authors list this
  as a limitation). Keep the published behaviour as the default.
- Validation: for Kepler-like stars, the planet table must match the HM-C catalogs
  (reuse `validation/hmc.py`).

### 4.2 M dwarfs

- Hsu, Ford & Terrien 2020 (a SysSim fit to Kepler M dwarfs) for occurrence vs P and
  R. Dressing & Charbonneau 2015 as a cross-check; Ment & Charbonneau 2023 for
  mid-to-late M (TESS, 0.1–0.3 M☉).
- Approach: keep HM-C's architecture (clusters, spacing, AMD) and rescale the
  number of planets and the period/radius distributions to the M-dwarf rates.
  f_swp should be close to 1 for early M.
- Brown dwarfs and late M (< 0.1 M☉): TRAPPIST-1-like compact systems are rare; use
  a low, flagged occurrence scaled from disk mass.
- Validation: reproduce the Hsu+2020 and Dressing & Charbonneau occurrence grids.

### 4.3 Giant planets, 0.03–30 au

- Occurrence vs semi-major axis: Fulton+2021 (California Legacy Survey), with the
  peak near 1–3 au and the turnover beyond it.
- Scaling: ∝ M★ (Johnson+2010) and ∝ 10^(2·[Fe/H]) (Fischer & Valenti 2005);
  A stars and giants per Reffert+2015 / Wolthoff+2022 (drop above ~2.5 M☉).
- Mass function: dN/dlogM from CLS. Eccentricities: Kipping 2013 beta distribution
  for RV planets.
- Inner small / outer giant correlation (Bryan & Lee 2024, Weiss+2024, He & Weiss
  2023), applied by conditioning the giant draw on the HM-C inner system.
- Wide-orbit cap from direct imaging (Nielsen+2019, Vigan+2021).
- Hot Jupiters (~0.5–1% for FGK) as a separate component; they suppress inner
  companions.
- Stability: reject or prune with the existing Hill/AMD machinery.
- Validation: reproduce CLS occurrence vs a; hot-Jupiter rate vs Kepler.

### 4.4 Small planets beyond 300 d

- Microlensing mass-ratio function (Suzuki+2016; Zang+2025) beyond the snow line.
- Bridge 300 d to the snow line with η⊕ constraints (Bryson+2021), which are
  extrapolations themselves; carry the uncertainty.
- Packing with the same Hill/AMD machinery.

### 4.5 Binaries

- S-type planets: remove orbits beyond the Holman & Wiegert 1999 critical radius.
- Suppress planet occurrence in close binaries (a < ~100 au) per Moe & Kratter 2021.
- Circumbinary (P-type) planets: Kepler occurrence (Armstrong+2014), outside the
  Holman–Wiegert inner limit.
- The population already provides hierarchy, a, e and masses (`draw_systems`).

### 4.6 Evolved stars and remnants

- Giants: engulf planets inside the maximum past radius (from MIST) plus a tidal
  margin. Expand surviving orbits adiabatically by M_initial / M_now.
- White dwarfs: survivors of the above; pollution rates (25–50%) as a sanity check.
- NS/BH: pulsar planets are very rare; use a tiny flagged rate.

### 4.7 Phase 4 validation report

`docs/validation/planets.md`: occurrence vs P and R by stellar type compared with
Kepler (via the HM-C catalogs), Hsu+2020 and CLS. Include the giant planet
correlations and a "where it is extrapolating" map.

## After phase 4

- **Phase 5:** moons and derived properties (T_eq, insolation, tidal locking,
  habitable-zone flags). See `DESIGN.md` §3.4.
- **Phase 6:** real-star conditioning (SIMBAD/Gaia/TIC lookup, NASA archive,
  per-method survey sensitivity, importance sampling). See §3.5.
- **Phase 7:** LightGBM emulators only where profiling shows they pay off. No torch.

## Known gaps in the stellar population (low priority)

- Low-mass He-core white dwarfs from binary interaction (~10% of WDs) are missing.
- NS/BH are placeholders; none lie within 20 pc, so they barely matter locally.
- The [Fe/H]–age relation is approximate; GALAH/APOGEE could replace it, since
  metallicity drives giant-planet occurrence in 4.3.
- Subgiant fraction is low against the census (labelling difference, not physics).

## Housekeeping

- `main` has only the initial commit. Merge `rewrite-starsys` once phase 4 is in
  shape. `gh` is not installed; use the GitHub API with curl or the web UI.
