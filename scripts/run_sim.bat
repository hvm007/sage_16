@echo off
REM =============================================================================
REM  sage16 simulation runner  (iverilog + vvp)
REM
REM  Run from anywhere; the script cds to the project root automatically.
REM    run_sim.bat              -> all testbenches
REM    run_sim.bat matmul       -> bare matmul
REM    run_sim.bat pcpi         -> RISC-V PCPI (matmul only)
REM    run_sim.bat mmio         -> 32-bit MMIO (matmul only)
REM    run_sim.bat conv         -> 3x3 convolution (signed kernels)
REM    run_sim.bat quat         -> quaternion multiply (signed ops)
REM    run_sim.bat reconfig     -> matmul -> conv -> quat back-to-back demo
REM    run_sim.bat mmio_top     -> multi-kernel MMIO (all 3 kernels)
REM    run_sim.bat pcpi_top     -> multi-kernel PCPI (all 3 kernels)
REM    run_sim.bat tiled        -> 8x8 and 16x16 matmul via 4x4 tiling
REM    run_sim.bat b2b          -> 1-cycle back-to-back reconfig demo
REM    run_sim.bat minicnn      -> 3x3 small CNN end-to-end (Conv1+FC1+FC2)
REM    run_sim.bat minicnn-mnist -> same TB but on real MNIST trained INT4 weights
REM    run_sim.bat minicnn-v2   -> deeper MiniCNN v2 (Conv1+Pool+Conv2+FC1+FC2) synthetic
REM    run_sim.bat minicnn-v2-mnist -> v2 on real MNIST (data\v2\*.mem)
REM    run_sim.bat madgwick     -> quaternion attitude filter (100 IMU steps)
REM =============================================================================

setlocal
pushd "%~dp0\.."

if not exist build mkdir build

REM common RTL (base fabric)
set RTL=rtl\pe.v rtl\sage16_4x4_mac.v rtl\matmul_sage16.v

if "%1"=="pcpi"     goto pcpi
if "%1"=="mmio"     goto mmio
if "%1"=="conv"     goto conv
if "%1"=="quat"     goto quat
if "%1"=="reconfig" goto reconfig
if "%1"=="mmio_top" goto mmio_top
if "%1"=="pcpi_top" goto pcpi_top
if "%1"=="tiled"    goto tiled
if "%1"=="b2b"      goto b2b
if "%1"=="minicnn"  goto minicnn
if "%1"=="minicnn-mnist" goto minicnn-mnist
if "%1"=="minicnn-v2"       goto minicnn-v2
if "%1"=="minicnn-v2-mnist" goto minicnn-v2-mnist
if "%1"=="madgwick" goto madgwick
if "%1"=="matmul"   goto matmul

:matmul
echo ==================================================
echo   1/6  Bare sage16 matmul  (sim\tb_matmul.v)
echo        overflow stress, 304 cells checked
echo ==================================================
iverilog -g2012 -o build\sim_matmul.vvp %RTL% sim\tb_matmul.v || goto err
vvp build\sim_matmul.vvp
if "%1"=="matmul" goto done

:pcpi
echo.
echo ==================================================
echo   2/6  RISC-V PCPI  (sim\tb_pcpi.v)
echo        custom instructions: sage16.wrA wrB go rdC
echo ==================================================
iverilog -g2012 -o build\sim_pcpi.vvp %RTL% rtl\sage16_pcpi.v sim\tb_pcpi.v || goto err
vvp build\sim_pcpi.vvp
if "%1"=="pcpi" goto done

:mmio
echo.
echo ==================================================
echo   3/6  Memory-mapped MMIO  (sim\tb_mmio.v)  FPGA-deployable
echo        reg map: A=0x00 B=0x40 C=0x80 CTRL=0xC0 STATUS=0xC4
echo ==================================================
iverilog -g2012 -o build\sim_mmio.vvp %RTL% rtl\sage16_mmio.v sim\tb_mmio.v || goto err
vvp build\sim_mmio.vvp
if "%1"=="mmio" goto done

:conv
echo.
echo ==================================================
echo   4/6  conv3x3  (sim\tb_conv.v)
echo        identity / box-blur / Sobel-X / Sobel-Y /
echo        Laplacian / sharpen / negative kernel /
echo        20 randomised  -- 464 output cells
echo ==================================================
iverilog -g2012 -o build\sim_conv.vvp %RTL% rtl\conv3x3_sage16.v sim\tb_conv.v || goto err
vvp build\sim_conv.vvp
if "%1"=="conv" goto done

:quat
echo.
echo ==================================================
echo   5/6  quaternion multiply  (sim\tb_quat.v)
echo        Hamilton basis (i*j=k), Z/X rotation,
echo        10 random + adversarial -- 304 components
echo ==================================================
iverilog -g2012 -o build\sim_quat.vvp %RTL% rtl\quat_sage16.v sim\tb_quat.v || goto err
vvp build\sim_quat.vvp
if "%1"=="quat" goto done

:reconfig
echo.
echo ==================================================
echo   6/10 SHARED-FABRIC reconfigurability demo
echo        (sim\tb_reconfig.v driving rtl\sage16_top.v)
echo        ONE sage16_4x4_mac instance runs matmul, conv3x3
echo        and quaternion via 2-bit mode + OP_MACB_S bit.
echo ==================================================
iverilog -g2012 -o build\sim_reconfig.vvp %RTL% rtl\sage16_top.v sim\tb_reconfig.v || goto err
vvp build\sim_reconfig.vvp
if "%1"=="reconfig" goto done

:mmio_top
echo.
echo ==================================================
echo   7/10 Multi-kernel MMIO  (sim\tb_mmio_top.v)
echo        all 3 kernels via the 32-bit bus
echo ==================================================
iverilog -g2012 -o build\sim_mmio_top.vvp %RTL% rtl\sage16_top.v rtl\sage16_mmio_top.v sim\tb_mmio_top.v || goto err
vvp build\sim_mmio_top.vvp
if "%1"=="mmio_top" goto done

:pcpi_top
echo.
echo ==================================================
echo   8/10 Multi-kernel PCPI  (sim\tb_pcpi_top.v)
echo        sage16.setMode / wrOp / go / rdC on PicoRV32
echo ==================================================
iverilog -g2012 -o build\sim_pcpi_top.vvp %RTL% rtl\sage16_top.v rtl\sage16_pcpi_top.v sim\tb_pcpi_top.v || goto err
vvp build\sim_pcpi_top.vvp
if "%1"=="pcpi_top" goto done

:tiled
echo.
echo ==================================================
echo   9/10 Tiled matmul  (sim\tb_matmul_tiled.v)
echo        8x8 and 16x16 via 4x4 tiling on sage16_top
echo ==================================================
iverilog -g2012 -o build\sim_tiled.vvp %RTL% rtl\sage16_top.v sim\tb_matmul_tiled.v || goto err
vvp build\sim_tiled.vvp
if "%1"=="tiled" goto done

:b2b
echo.
echo ==================================================
echo   10/12 Back-to-back 1-cycle reconfig (sim\tb_reconfig_b2b.v)
echo        start held across done => S_DONE-^>S_CFG shortcut
echo ==================================================
iverilog -g2012 -o build\sim_b2b.vvp %RTL% rtl\sage16_top.v sim\tb_reconfig_b2b.v || goto err
vvp build\sim_b2b.vvp
if "%1"=="b2b" goto done

:minicnn
echo.
echo ==================================================
echo   11/12 MiniCNN-3x3 end-to-end (sim\tb_minicnn.v)
echo        Conv1 conv3x3 + AvgPool4 + FC1 + ReLU + FC2
echo        bit-exact vs in-TB golden reference
echo ==================================================
iverilog -g2012 -o build\sim_minicnn.vvp %RTL% rtl\sage16_top.v sim\tb_minicnn.v || goto err
vvp build\sim_minicnn.vvp
if "%1"=="minicnn" goto done
goto skip_minicnn_mnist

:minicnn-mnist
echo.
echo ==================================================
echo   11b/12 MiniCNN-3x3 on REAL MNIST (sim\tb_minicnn.v +DREAL_MNIST)
echo        loads trained INT4 weights + 200 test images from data\*.mem
echo        reports fabric-side top-1 MNIST accuracy
echo ==================================================
if not exist data\images.mem (
    echo data\images.mem missing -- run: python scripts\train_minicnn.py
    goto err
)
iverilog -g2012 -DREAL_MNIST -o build\sim_minicnn_mnist.vvp %RTL% rtl\sage16_top.v sim\tb_minicnn.v || goto err
vvp build\sim_minicnn_mnist.vvp
if "%1"=="minicnn-mnist" goto done

:skip_minicnn_mnist

:minicnn-v2
echo.
echo ==================================================
echo   11c/12 MiniCNN-v2 end-to-end (sim\tb_minicnn_v2.v)
echo        Conv1(1-^>8) + Pool4 + Conv2(8-^>16) + FC1(256-^>32) + FC2
echo        bit-exact vs in-TB golden, synthetic prand weights
echo ==================================================
iverilog -g2012 -o build\sim_minicnn_v2.vvp %RTL% rtl\sage16_top.v sim\tb_minicnn_v2.v || goto err
vvp build\sim_minicnn_v2.vvp
if "%1"=="minicnn-v2" goto done
goto skip_minicnn_v2_mnist

:minicnn-v2-mnist
echo.
echo ==================================================
echo   11d/12 MiniCNN-v2 on REAL MNIST (sim\tb_minicnn_v2.v +DREAL_MNIST)
echo        loads trained INT4 weights from data\v2\*.mem
echo        reports fabric-side top-1 MNIST accuracy
echo ==================================================
REM Reviewer package note: pre-exported data\v2\*.mem is sufficient; retraining is optional.
if not exist data\v2\images.mem (
    echo data\v2\images.mem missing -- run: python scripts\train_minicnn_v2.py
    goto err
)
iverilog -g2012 -DREAL_MNIST -o build\sim_minicnn_v2_mnist.vvp %RTL% rtl\sage16_top.v sim\tb_minicnn_v2.v || goto err
vvp build\sim_minicnn_v2_mnist.vvp
if "%1"=="minicnn-v2-mnist" goto done

:skip_minicnn_v2_mnist

:madgwick
echo.
echo ==================================================
echo   12/12 Madgwick attitude filter (sim\tb_madgwick.v)
echo        100 IMU steps; fabric = quat-mul, SW = add+norm
echo ==================================================
iverilog -g2012 -o build\sim_madgwick.vvp %RTL% rtl\sage16_top.v sim\tb_madgwick.v || goto err
vvp build\sim_madgwick.vvp

:done
echo.
echo ==================================================
echo   All requested testbenches finished.
echo ==================================================
popd
exit /b 0

:err
echo BUILD FAILED
popd
exit /b 1
