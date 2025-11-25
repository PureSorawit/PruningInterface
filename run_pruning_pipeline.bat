@echo off
setlocal ENABLEDELAYEDEXPANSION

:: ----------------------------------------------------
:: Disable TensorFlow oneDNN warnings
:: ----------------------------------------------------
set TF_ENABLE_ONEDNN_OPTS=0

echo ================================
echo   ViT Pruning Pipeline Launcher
echo ================================
echo.
echo Enter target prune level (0.0 - 1.0), e.g. 0.30:
set /p PRUNE_LEVEL=Prune level: 

echo.
echo You entered: %PRUNE_LEVEL%
echo.

:: ----------------------------------------------------
::  Add Python warning suppression automatically
::  (works for all scripts in this batch)
::  This injects "import warnings; warnings.filterwarnings('ignore')"
::  into PYTHONWARNINGS environment variable.
:: ----------------------------------------------------
set PYTHONWARNINGS=ignore

:: ----------------------------------------------------
:: 1) Averaging scores -> combined_masks.json
:: ----------------------------------------------------
echo [1/6] Running averaging.py ...
python averaging.py --prune %PRUNE_LEVEL%
echo.

:: ----------------------------------------------------
:: 2) Intersecting masks -> final_intersect_mask.json
:: ----------------------------------------------------
echo [2/6] Running intersecting.py ...
python intersecting.py --target %PRUNE_LEVEL%
echo.

:: ----------------------------------------------------
:: 3) Apply average mask
:: ----------------------------------------------------
echo [3/6] Applying average mask -> vit_b16_cifar100_average_pruned.pth ...
python apply_mask.py --mask output/combined_masks.json --save vit_b16_cifar100_average_pruned.pth
echo.

:: ----------------------------------------------------
:: 4) Apply intersected mask
:: ----------------------------------------------------
echo [4/6] Applying intersected mask -> vit_b16_cifar100_intersect_pruned.pth ...
python apply_mask.py --mask output/final_intersect_mask.json --save vit_b16_cifar100_intersect_pruned.pth
echo.

:: ----------------------------------------------------
:: 5) Eval average-pruned model
:: ----------------------------------------------------
echo [5/6] Evaluating AVERAGE pruned model ...
python eval.py --ckpt vit_b16_cifar100_average_pruned.pth
echo.

:: ----------------------------------------------------
:: 6) Eval intersect-pruned model
:: ----------------------------------------------------
echo [6/6] Evaluating INTERSECT pruned model ...
python eval.py --ckpt vit_b16_cifar100_intersect_pruned.pth
echo.

echo ============================================
echo   DONE. Scroll up to see all eval results.
echo   Press any key to close this window.
echo ============================================
pause >nul

endlocal
