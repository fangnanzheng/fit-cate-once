# CATE-Assisted Randomization Tests for Staggered Adoption Panels

This repository contains simulation code, semi-synthetic county employment experiments, consistency experiments, generated CSV results, and paper figures for CATE-assisted randomization tests in staggered adoption designs.

The main comparison methods are:

- **RT(CATE)**: randomization tests using CATE-assisted residual moment statistics (`RV` for static/lag-invariant effects and `RC` for lag-varying effects).
- **RT(DM), RT(cDM), RT(SS)**: alternative randomization-test baselines used in the synthetic experiments.
- **TWFE / event-study TWFE**: semi-synthetic MPDTA baselines using conventional clustered two-sided tests.

## Repository layout

```text
.
├── helpers.py                      # shared simulation, fitting, and randomization utilities
├── rc.py                           # RV/RC CATE-assisted estimators and test components
├── dm.py, cdm.py, ss.py             # baseline randomization-test methods
├── run_main_experiments.py          # main synthetic experiments
├── consistency_exp_utils.py         # consistency and warm-start utilities
├── run_consistency_experiments.py   # consistency and warm-start experiment driver
├── mpdta.csv                        # county employment panel used in MPDTA experiments
├── mpdta_exp_utils.py               # MPDTA DGPs and TWFE helpers
├── run_mpdta_experiments.py         # MPDTA experiment driver
├── plot_figures.py                  # regenerates figures from CSV outputs
├── results/                         # generated experiment outputs
├── figures/                         # generated paper figures
├── requirements.txt                 # Python package dependencies
└── README.md
```

## Installation

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

A quick import/compile check is:

```bash
python -m py_compile *.py
```

## Reproduce figures from included results

The repository includes generated CSV outputs under `results/`. To regenerate all figures from these existing outputs:

```bash
python plot_figures.py
```

The figures are written to `figures/`.

## Main synthetic experiments

The main synthetic experiments use a five-period staggered adoption design with starts `A=1,...,5` and `A=6` denoting never treated. The default run uses `n_reps=200`, `n_perms=500`, and `alpha=0.1`.

Example small smoke tests:

```bash
python run_main_experiments.py --assumption static --setting validity --N 60 --n_reps 2 --n_perms 5 --out_dir results/smoke/main_static_valid
python run_main_experiments.py --assumption lagged --setting power --N 60 --n_reps 2 --n_perms 5 --out_dir results/smoke/main_lag_power
```

Example full-style commands for one sample size:

```bash
python run_main_experiments.py --assumption static --setting validity --N 500 --out_dir results/main/sta_valid500
python run_main_experiments.py --assumption static --setting power    --N 500 --out_dir results/main/sta_power500
python run_main_experiments.py --assumption lagged --setting validity --N 500 --out_dir results/main/lag_valid500
python run_main_experiments.py --assumption lagged --setting power    --N 500 --out_dir results/main/lag_power500
```

The figure script expects the main-result folders for sample sizes `300, 400, 500, 600, 700` if you want to regenerate the full set of main synthetic figures.

## Consistency and warm-start experiments

Raw consistency experiments:

```bash
python run_consistency_experiments.py --assumption lagged --option nonparametric --out_dir results/consistency/nonparametric
python run_consistency_experiments.py --assumption lagged --option parametric    --out_dir results/consistency/parametric
```

Warm-start experiment:

```bash
python run_consistency_experiments.py --assumption lagged --option nonparametric --experiment start --out_dir results/consistency/start
```

For quick smoke tests, reduce `--n_reps` and use smaller grids, for example:

```bash
python run_consistency_experiments.py --assumption lagged --option parametric --n_reps 2 --n_grid 40,80 --out_dir results/smoke/consistency_parametric
```

## Semi-synthetic county employment experiments

The MPDTA experiments use `mpdta.csv`, standardized log population as the covariate, complete re-randomization of treatment timing under fixed cohort shares, and the semi-synthetic DGP in `mpdta_exp_utils.py`. The default run uses `n_reps=100`, `n_perms=300`, and `alpha=0.1`.

Validity and power experiments:

```bash
python run_mpdta_experiments.py --experiment validity --assumption static --out_dir results/mpdta/sta_valid
python run_mpdta_experiments.py --experiment power    --assumption static --out_dir results/mpdta/sta_power
python run_mpdta_experiments.py --experiment validity --assumption lagged --out_dir results/mpdta/lag_valid
python run_mpdta_experiments.py --experiment power    --assumption lagged --out_dir results/mpdta/lag_power
```

Subgroup experiments:

```bash
python run_mpdta_experiments.py --experiment subgroup --tau 0.10 --out_dir results/mpdta/subgroup0.1
python run_mpdta_experiments.py --experiment subgroup --tau 0.15 --out_dir results/mpdta/subgroup0.15
python run_mpdta_experiments.py --experiment subgroup --tau 0.20 --out_dir results/mpdta/subgroup0.2
python run_mpdta_experiments.py --experiment subgroup --tau 0.25 --out_dir results/mpdta/subgroup0.25
python run_mpdta_experiments.py --experiment subgroup --tau 0.30 --out_dir results/mpdta/subgroup0.3
```

Quick smoke test:

```bash
python run_mpdta_experiments.py --experiment validity --assumption static --n_reps 2 --n_perms 5 --out_dir results/smoke/mpdta_sta_valid
```

## Reproducibility notes

- All scripts expose `--seed`; the default is `12345`.
- Many full-grid runs are computationally expensive. Use the smoke-test commands before launching full runs.
- `results/` and `figures/` are generated artifacts. They are included here for reproducibility and to allow figure regeneration without rerunning all simulations.
