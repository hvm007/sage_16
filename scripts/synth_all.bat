@echo off
REM =============================================================================
REM  synth_all.bat - run scripts/synth_top.tcl for every paper-relevant top
REM
REM  Usage (from anywhere):
REM    scripts\synth_all.bat                 -- all tops (OOC)
REM    scripts\synth_all.bat sage16_top        -- just one top
REM    scripts\synth_all.bat -deploy sage16_mmio sage16_mmio_top
REM          -- full (non-OOC) deploy flow on MMIO-family tops; requires IOB fit
REM
REM  Each run takes ~2-6 minutes on a modern laptop.
REM  Outputs land in reports\<top>\.
REM =============================================================================

setlocal ENABLEDELAYEDEXPANSION
pushd "%~dp0\.."

if not exist reports mkdir reports

set MODE=ooc
if "%1"=="-deploy" (
    set MODE=deploy
    shift
)

REM Default list: every paper-relevant top, in order.
REM matmul_sage16          : single-kernel baseline (unsigned matmul only)
REM conv3x3_sage16         : single-kernel baseline (conv only)
REM quat_sage16            : single-kernel baseline (quat only)
REM sage16_top             : SHARED FABRIC - the paper's headline module
REM sage16_mmio            : old MMIO wrapper around matmul_sage16 (backward compat)
REM sage16_mmio_top        : new MMIO wrapper around sage16_top (multi-kernel)
REM sage16_pcpi            : old PCPI wrapper around matmul_sage16
REM sage16_pcpi_top        : new PCPI wrapper around sage16_top (multi-kernel)
if "%~1"=="" (
    set TOPS=matmul_sage16 conv3x3_sage16 quat_sage16 sage16_top sage16_mmio sage16_mmio_top sage16_pcpi sage16_pcpi_top
) else (
    set TOPS=%*
)

set PERIOD=5.000

for %%T in (%TOPS%) do (
    if not exist reports\%%T mkdir reports\%%T
    echo.
    echo =====================================================================
    echo   Synthesizing %%T  ^(mode=%MODE%, period=%PERIOD% ns^)
    echo =====================================================================
    vivado -mode batch ^
           -source scripts\synth_top.tcl ^
           -tclargs %%T %PERIOD% %MODE% ^
           -log reports\%%T\vivado.log ^
           -journal reports\%%T\vivado.jou
    if errorlevel 1 (
        echo.
        echo !!! Synthesis FAILED for %%T -- see reports\%%T\vivado.log
        echo.
        popd
        exit /b 1
    )
)

echo.
echo =====================================================================
echo  All requested tops synthesized successfully.
echo  Run:  python scripts\parse_reports.py
echo  to regenerate the comparison markdown tables.
echo =====================================================================
popd
exit /b 0
