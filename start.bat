@echo off
rem Launch the hex-prep web UI and open it in your browser.
rem
rem First run creates .venv\ and installs hex-prep into it (pulls PyTorch,
rem several GB). Optional environment variables:
rem   HEXPREP_PYTHON       use this interpreter (an env that already has
rem                        hex-prep) instead of building .venv\
rem   HEXPREP_TORCH_INDEX  PyTorch wheel index for the first-run install.
rem                        NVIDIA default is CUDA 12.8 (RTX 20-50 series);
rem                        GTX 10-series: https://download.pytorch.org/whl/cu126
rem Extra arguments go to hex-prep-web, e.g.  start.bat --host 0.0.0.0
setlocal
cd /d "%~dp0"

set "PY=%HEXPREP_PYTHON%"
if defined PY goto run
set "PY=.venv\Scripts\python.exe"
rem Written once the install finishes, so a first run that was interrupted
rem (or failed) resumes the install instead of launching a half-built .venv.
if exist ".venv\.hexprep-installed" goto run
if exist "%PY%" goto install

rem 3.10-3.13: the versions the dependency stack ships Windows wheels for.
for %%V in (3.12 3.13 3.11 3.10) do (
    py -%%V -c "pass" >nul 2>&1 && (
        py -%%V -m venv .venv
        goto venv_made
    )
)
python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))" >nul 2>&1 && python -m venv .venv
:venv_made
if not exist "%PY%" (
    echo Need Python 3.10-3.13 ^(3.12 recommended^) from https://python.org
    pause & exit /b 1
)

:install
where ffmpeg >nul 2>&1 || echo WARNING: ffmpeg not in PATH - the music mix (.mp3) will be skipped.
echo First run: installing hex-prep into .venv (several GB)...
"%PY%" -m pip install --upgrade pip || goto failed
set "EXTRA=cpu"
where nvidia-smi >nul 2>&1 && set "EXTRA=gpu"
if "%EXTRA%"=="cpu" echo WARNING: no NVIDIA GPU found. hex-prep needs one - it will install, but runs far too slowly on CPU.
rem PyPI's Windows torch wheels are CPU-only, so NVIDIA cards need the CUDA
rem build from PyTorch's own index, installed first so the next step keeps it.
set "TORCH_INDEX=%HEXPREP_TORCH_INDEX%"
if "%EXTRA%"=="gpu" if not defined TORCH_INDEX set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
if defined TORCH_INDEX "%PY%" -m pip install torch --index-url "%TORCH_INDEX%" || goto failed
"%PY%" -m pip install -e ".[%EXTRA%]" || goto failed
type nul > ".venv\.hexprep-installed"
goto run

:failed
echo Install failed - see the errors above. Run start.bat again to retry.
pause & exit /b 1

:run
"%PY%" -m hex_prep.web --open %*
pause
