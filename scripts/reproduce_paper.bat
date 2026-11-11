@echo off
REM Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
REM SPDX-License-Identifier: MIT
REM Authors: Hoang Nhut Huynh, Tran Anh Tu
REM
REM Re-runs every experiment reported in the paper (Windows). From the repo root:
REM     scripts\reproduce_paper.bat            (TabPFN on CUDA)
REM     scripts\reproduce_paper.bat cpu
REM TabPFN needs TABPFN_TOKEN to be set (see README).
setlocal
set DEVICE=%1
if "%DEVICE%"=="" set DEVICE=cuda
set RUN=python experiments\run_experiment.py

for %%D in (icr hcv) do (
    REM 1) main comparison: TabPFN and untuned baselines, binary target
    %RUN% --config configs\%%D.yaml --device %DEVICE% || goto :error
    for %%M in (xgb lgbm logreg) do %RUN% --config configs\%%D.yaml --model %%M || goto :error
    REM 2) weighted / early-stopped / calibrated tree ensembles (control)
    for %%M in (xgb_w lgbm_w xgb_cal lgbm_cal xgb_iso lgbm_iso) do %RUN% --config configs\%%D.yaml --model %%M || goto :error
    REM 3) ablations: miscoverage level, calibration budget, context downsampling
    for %%A in (0.05 0.2) do %RUN% --config configs\%%D.yaml --device %DEVICE% --alpha %%A || goto :error
    for %%F in (0.25 0.5) do %RUN% --config configs\%%D.yaml --device %DEVICE% --calib-fraction %%F || goto :error
    %RUN% --config configs\%%D.yaml --device %DEVICE% --downsample 2 || goto :error
    REM 4) TabPFN run that also saves per-subject probabilities (post-hoc analyses)
    %RUN% --config configs\%%D.yaml --device %DEVICE% --save-preds || goto :error
    python experiments\risk_coverage.py results\%%D_tabpfn_class_preds.csv || goto :error
)

REM 5) condition-subtype label: Mondrian versus clustered calibration
%RUN% --config configs\icr.yaml --device %DEVICE% --target alpha || goto :error
for %%A in (0.05 0.2) do %RUN% --config configs\icr.yaml --device %DEVICE% --target alpha --alpha %%A || goto :error
%RUN% --config configs\hcv.yaml --device %DEVICE% --target category || goto :error

REM 6) HCV subgroups, paired tests and table rows
python experiments\subgroup_hcv.py results\hcv_tabpfn_class_preds.csv || goto :error
python experiments\stats_tests.py > results\stats_tests.txt || goto :error
python experiments\make_tables.py > results\tables.tex || goto :error
echo Done. Summaries are in results\.
exit /b 0

:error
echo A step failed; see the message above.
exit /b 1
