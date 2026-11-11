#!/usr/bin/env bash
# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
#
# Re-runs every experiment reported in the paper. Run from the repository root:
#     bash scripts/reproduce_paper.sh            # uses CUDA for TabPFN
#     DEVICE=cpu bash scripts/reproduce_paper.sh
# TabPFN needs TABPFN_TOKEN to be set (see README). On one laptop GPU the
# TabPFN runs take a few hours in total; the baselines take minutes.
set -euo pipefail
DEVICE="${DEVICE:-cuda}"
RUN="python experiments/run_experiment.py"

for DS in icr hcv; do
  CFG="configs/${DS}.yaml"
  # 1) main comparison: TabPFN and untuned baselines, binary target
  $RUN --config "$CFG" --device "$DEVICE"
  for M in xgb lgbm logreg; do $RUN --config "$CFG" --model "$M"; done
  # 2) weighted / early-stopped / calibrated tree ensembles (control)
  for M in xgb_w lgbm_w xgb_cal lgbm_cal xgb_iso lgbm_iso; do $RUN --config "$CFG" --model "$M"; done
  # 3) ablations: miscoverage level, calibration budget, context downsampling
  for A in 0.05 0.2;  do $RUN --config "$CFG" --device "$DEVICE" --alpha "$A"; done
  for F in 0.25 0.5;  do $RUN --config "$CFG" --device "$DEVICE" --calib-fraction "$F"; done
  $RUN --config "$CFG" --device "$DEVICE" --downsample 2
  # 4) TabPFN run that also saves per-subject probabilities (post-hoc analyses)
  $RUN --config "$CFG" --device "$DEVICE" --save-preds
  python experiments/risk_coverage.py "results/${DS}_tabpfn_class_preds.csv"
done

# 5) condition-subtype label: Mondrian versus clustered calibration
$RUN --config configs/icr.yaml --device "$DEVICE" --target alpha
for A in 0.05 0.2; do $RUN --config configs/icr.yaml --device "$DEVICE" --target alpha --alpha "$A"; done
$RUN --config configs/hcv.yaml --device "$DEVICE" --target category

# 6) HCV subgroups, paired tests and table rows
python experiments/subgroup_hcv.py results/hcv_tabpfn_class_preds.csv
python experiments/stats_tests.py > results/stats_tests.txt
python experiments/make_tables.py > results/tables.tex
echo "Done. Summaries are in results/."
