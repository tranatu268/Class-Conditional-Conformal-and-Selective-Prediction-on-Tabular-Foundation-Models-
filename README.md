# Class-conditional conformal and selective prediction on tabular foundation models

Code for the paper *Class-Conditional Conformal and Selective Prediction on Tabular
Foundation Models for Small, Imbalanced Clinical Cohorts* (Hoang Nhut Huynh, under review).

In small, class-imbalanced clinical cohorts, split conformal prediction with a single
(marginal) threshold meets its nominal coverage on average while under-covering the rare
diseased class. This repository implements and evaluates the remedy studied in the paper:

- **class-conditional (Mondrian) calibration**, with one threshold per class;
- **clustered calibration** over the taxonomy {no condition} vs. {condition subtypes},
  for the case where the rarest subtypes are too small for a finite Mondrian quantile;
- a **singleton-set selective rule**: answer when the prediction set holds one label,
  abstain when it is empty or holds two labels;
- **TabPFN** as the base model, compared with logistic regression, XGBoost and LightGBM
  (untuned, class-weighted with early stopping, and cross-fitted Platt or isotonic
  calibration).

Authors: **Hoang Nhut Huynh**, **Tran Anh Tu**

## Pipeline

1. **Splits.** Repeated stratified 5-fold cross-validation (10 repeats, 50 paired splits).
   Each training fold is split again into a proper training set (2/3) and a calibration
   set (1/3). The test fold is never used for tuning.
2. **Base model.** TabPFN uses the proper training set as its context (no gradient
   steps); baselines are fitted on it. The model outputs class probabilities
   $\hat p_y(x)$ and the LAC score $s(x,y) = 1-\hat p_y(x)$ (APS is also available).
3. **Conformal calibration.** The threshold is the
   $\lceil (m+1)(1-\alpha)\rceil$-th smallest of the $m$ calibration scores in a cell:
   one cell for all subjects (marginal), one per class (Mondrian) or one per group of
   subtypes (clustered). A finite threshold needs $m \geq \lceil 1/\alpha\rceil - 1$
   calibration points in the cell (9 at $\alpha = 0.1$).
4. **Selective prediction.** The prediction set is
   $\mathcal{C}(x) = \{y : s(x,y) \leq \hat q_y\}$; the subject is answered only
   if $|\mathcal{C}(x)| = 1$.
5. **Evaluation.** Per-class and worst-class coverage, coverage gap, abstention rate
   (and share of empty sets), selective risk, selective balanced accuracy, diseased
   subjects answered "no condition" per 100, AURC, balanced log loss, Brier score, ECE,
   sensitivity and specificity. Paired Wilcoxon tests at split and repeat level with
   Holm correction.

## Repository layout

```
configs/
  icr.yaml, hcv.yaml        dataset paths, CV protocol, conformal settings
src/icp/
  data.py                   loaders for ICR (train.csv + greeks.csv) and UCI HCV
  models.py                 TabPFN wrapper and baselines (incl. weighted/calibrated GBDTs)
  imbalance.py              threshold tuning, prior-shift correction, context downsampling
  conformal.py              split conformal (LAC/APS; marginal, Mondrian, clustered),
                            selective prediction, risk-coverage curve
  metrics.py                balanced log loss, ECE, Brier, AURC, ...
  evaluate.py               repeated stratified CV with an inner calibration split
  posthoc.py                rebuilds conformal sets from saved per-subject probabilities
experiments/
  run_experiment.py         main entry point (one model x dataset x target per call)
  risk_coverage.py          conformal rule vs. softmax-confidence rule at matched coverage
  subgroup_hcv.py           HCV coverage and abstention by sex and age tertile
  stats_tests.py            paired Wilcoxon tests with Holm correction
  make_tables.py            LaTeX table rows from the results files
scripts/
  reproduce_paper.sh/.bat   every run reported in the paper, in order
tests/
  test_conformal.py         coverage checks on synthetic data (no model needed)
results/                    per-fold metrics of the reported runs (see below)
```

## Installation

Tested with Python 3.12. A CUDA GPU is recommended for TabPFN; the baselines run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q          # quick check of the conformal layer
```

TabPFN downloads its weights on first use and asks for a license token. Accept the
license on the TabPFN website, then set the token before running:

```bash
export TABPFN_TOKEN=<your token>          # Windows: set TABPFN_TOKEN=<your token>
export TABPFN_NO_BROWSER=1                # optional, avoids opening a browser
```

The reported runs used the `tabpfn` package 9.2.0 with the TabPFN-3.5 checkpoint
`tabpfn-v3.5-20260909` (package defaults, 8 ensemble members) on an NVIDIA GeForce
RTX 4050 Laptop GPU.

## Data

The data are not redistributed here. Place them under `data/` (ignored by git):

```
data/ICR/train.csv
data/ICR/greeks.csv
data/HCV/hcvdat0.csv
```

- **ICR: Identifying Age-Related Conditions** (Kaggle competition, n = 617, 56 features,
  17.5% positive). Download `train.csv` and `greeks.csv` after accepting the competition
  rules: `kaggle competitions download -c icr-identify-age-related-conditions`.
  The auxiliary label `Alpha` in `greeks.csv` (A = no condition; B, D, G = conditions)
  is used for the subtype experiments.
- **UCI HCV data** (UCI Machine Learning Repository, ID 571,
  doi:[10.24432/C5D612](https://doi.org/10.24432/C5D612)). The 7 "suspect blood donor"
  records are dropped (`suspect: drop`), leaving n = 608 (12.3% positive). The loader
  reads the UCI CSV or an ARFF export of the same table (set `data.path` in
  `configs/hcv.yaml` to a `.arff` file); the reported runs read the ARFF export. Every
  run prints the cohort size, so `n=608` confirms that the right table was loaded.

Paths in the configs are relative to the repository root.

## Running

Single runs:

```bash
# TabPFN, binary target, all conformal variants
python experiments/run_experiment.py --config configs/icr.yaml --device cuda

# baselines
python experiments/run_experiment.py --config configs/icr.yaml --model xgb
python experiments/run_experiment.py --config configs/icr.yaml --model lgbm_cal
python experiments/run_experiment.py --config configs/hcv.yaml --model logreg

# subtype label: Mondrian vs. clustered calibration
python experiments/run_experiment.py --config configs/icr.yaml --device cuda --target alpha
python experiments/run_experiment.py --config configs/hcv.yaml --device cuda --target category
```

Useful options of `run_experiment.py`:

| option | effect | file suffix |
|---|---|---|
| `--model` | `tabpfn`, `logreg`, `rf`, `xgb`, `lgbm`, `xgb_w`, `lgbm_w`, `xgb_cal`, `lgbm_cal`, `xgb_iso`, `lgbm_iso` | |
| `--target` | `class` (binary), `alpha` (ICR subtypes), `category` (HCV stages) | |
| `--alpha` | miscoverage level, e.g. 0.05 or 0.2 | `_a<value>` |
| `--calib-fraction` | share of the training fold used for calibration | `_cf<value>` |
| `--downsample` | cap the majority:minority ratio of the TabPFN context, e.g. 2 | `_ds<value>` |
| `--repeats` | number of CV repeats (default 10) | |
| `--save-preds` | also write per-subject probabilities to `results/<stem>_preds.csv` | `_sp` |

All experiments of the paper, followed by the post-hoc analyses, tests and table rows:

```bash
bash scripts/reproduce_paper.sh                 # Linux / macOS
scripts\reproduce_paper.bat                     # Windows
```

## Results files

Each run writes two files to `results/`, named `<dataset>_<model>_<target><suffix>`:

- `.csv`: one row per (repeat, fold, method), where `method` is one of `point_raw`,
  `point_threshold`, `point_prior_shift`, `softmax_selective` or `cp_<score>_<variant>`
  (for example `cp_lac_mondrian`). Columns hold the metrics listed above, for example
  `coverage_class1`, `worst_class_coverage`, `abstention_rate`, `frac_empty`,
  `selective_balanced_accuracy` and `wrong_answer_frac_class1`.
- `.json`: the resolved configuration and, for every method and metric, the mean with
  a 95% t-interval over the 50 splits.

`risk_coverage.py` writes `rc_<stem>_summary.json` and `rc_<stem>_curve.csv`, and
`subgroup_hcv.py` writes `subgroup_<stem>.csv`. The `results/` folder of this
repository holds these files for every run reported in the paper. Per-subject
predictions (`*_preds.csv`) are not included.

The CV splitter is seeded by the repeat index, so all models and calibration regimes
are evaluated on identical splits, which is what makes the paired tests valid.

## Citation

The paper is under review; its reference will be added here. To cite the code, see
`CITATION.cff`.

## License

MIT, see `LICENSE`. The datasets and the TabPFN weights are subject to their own
licenses and terms of use.
