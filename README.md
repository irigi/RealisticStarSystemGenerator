# Realistic Star System Generator

This project trains a hierarchical neural generator on SysSim physical exoplanet catalogs.

The first implementation targets the local 2020 `Linear_fswp/KS` physical catalogs:

- `catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS/physical_catalogs.zip`
- `catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS/physical_catalogs_stars.zip`

The generator follows the planned hurdle structure:

1. Predict whether a star hosts any planets.
2. Predict the planet multiplicity for hosted systems.
3. Generate an ordered planet sequence in ascending orbital period.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Inspect Catalogs

```bash
python -m star_system_generator.inspect_catalog \
  --catalog-root catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS \
  --catalog-id 1
```

## Train

Start with a small smoke run:

```bash
python -m star_system_generator.train \
  --catalog-root catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS \
  --train-catalogs 1,2 \
  --val-catalogs 86 \
  --epochs 1 \
  --max-systems-per-catalog 2000 \
  --batch-size 128 \
  --output-dir runs/smoke
```

Then scale to the realization split:

```bash
python -m star_system_generator.train \
  --catalog-root catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS \
  --train-catalogs 1-70 \
  --val-catalogs 71-85 \
  --epochs 50 \
  --batch-size 512 \
  --output-dir runs/linear_fswp_baseline
```

## Generate Samples

```bash
python -m star_system_generator.sample \
  --checkpoint runs/linear_fswp_baseline/model.pt \
  --catalog-root catalogs/He_Ford_Ragozzine_2020/Linear_fswp/KS \
  --catalog-id 86 \
  --num-stars 1000 \
  --output generated_catalog.csv
```

