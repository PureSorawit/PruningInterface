@echo off

:: ----------------------------------------
:: Suppress warnings
:: ----------------------------------------
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

:: ----------------------------------------
:: Log file setup
:: ----------------------------------------
set "LOGDIR=logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

:: Get timestamp for log filename
for /f %%i in ('powershell -command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set TIMESTAMP=%%i
set "LOGFILE=%LOGDIR%\eval_log_%TIMESTAMP%.txt"

echo ================================
echo   ViT pruning sweep 5%%–40%%
echo ================================
echo.

set /p USE_TEST=Use TEST dataset? (y/n, default n): 

if /I "%USE_TEST%"=="y" (
    set "EV_TEST_FLAG=--test"
) else (
    set "EV_TEST_FLAG="
)

echo.
echo Using TEST flag: %EV_TEST_FLAG%
echo Logging to: %LOGFILE%
echo.
echo Press any key to begin...
pause >nul

:: ----------------------------------------
:: Run each prune level
:: First level (0.05) logs everything
:: Others log only eval (steps 5 & 6)
:: ----------------------------------------
call :do_level 0.05 1
call :do_level 0.10 0
call :do_level 0.15 0
call :do_level 0.20 0
call :do_level 0.25 0
call :do_level 0.30 0
call :do_level 0.35 0
call :do_level 0.40 0

goto DONE

:: ----------------------------------------
:: Subroutine for one prune level
:: %1 = prune level (e.g. 0.30)
:: %2 = 1 if first level (log steps 1–4), else 0
:: ----------------------------------------
:do_level
set "PRUNE=%1"
set "FIRST=%2"

echo ===============================================
echo === Prune level: %PRUNE% =======================
echo ===============================================

if "%FIRST%"=="1" (
    echo [1/6] [LOG] averaging.py --prune %PRUNE%
    python averaging.py --prune %PRUNE% >> "%LOGFILE%" 2>&1 || goto ERROR
    echo.

    echo [2/6] [LOG] intersecting.py --target %PRUNE%
    python intersecting.py --target %PRUNE% >> "%LOGFILE%" 2>&1 || goto ERROR
    echo.

    echo [3/6] [LOG] apply_mask average
    python apply_mask.py --mask output/combined_masks.json --save vit_b16_cifar100_average_pruned.pth >> "%LOGFILE%" 2>&1 || goto ERROR
    echo.

    echo [4/6] [LOG] apply_mask intersect
    python apply_mask.py --mask output/final_intersect_mask.json --save vit_b16_cifar100_intersect_pruned.pth >> "%LOGFILE%" 2>&1 || goto ERROR
    echo.
) else (
    echo [1/6] averaging.py --prune %PRUNE%
    python averaging.py --prune %PRUNE% || goto ERROR
    echo.

    echo [2/6] intersecting.py --target %PRUNE%
    python intersecting.py --target %PRUNE% || goto ERROR
    echo.

    echo [3/6] apply_mask average
    python apply_mask.py --mask output/combined_masks.json --save vit_b16_cifar100_average_pruned.pth || goto ERROR
    echo.

    echo [4/6] apply_mask intersect
    python apply_mask.py --mask output/final_intersect_mask.json --save vit_b16_cifar100_intersect_pruned.pth || goto ERROR
    echo.
)

echo ---------------------------------------------->> "%LOGFILE%"
echo PRUNE LEVEL: %PRUNE% >> "%LOGFILE%"

echo [5/6] Evaluating AVERAGE
echo --- AVERAGE --- >> "%LOGFILE%"
python eval.py --ckpt vit_b16_cifar100_average_pruned.pth %EV_TEST_FLAG% >> "%LOGFILE%" 2>&1 || goto ERROR
echo.

echo [6/6] Evaluating INTERSECT
echo --- INTERSECT --- >> "%LOGFILE%"
python eval.py --ckpt vit_b16_cifar100_intersect_pruned.pth %EV_TEST_FLAG% >> "%LOGFILE%" 2>&1 || goto ERROR
echo.

goto :eof


:ERROR
echo.
echo ****************************************
echo *** ERROR OCCURRED — see message above
echo ****************************************
echo Log file:
echo   %LOGFILE%
echo.
pause
goto END

:DONE
echo ============================================
echo DONE — Results saved to:
echo   %LOGFILE%
echo Press any key to exit...
echo ============================================
pause

:END
endlocal
