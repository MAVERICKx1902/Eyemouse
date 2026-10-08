@echo off
REM ---------------------------------------------------------------------
REM  Eyemouse launcher for Windows.
REM
REM  Creates .venv on first run, installs requirements, downloads the
REM  MediaPipe model bundles if they are missing, then starts the app.
REM  Double-click it, or run "run.bat --check" to diagnose your setup.
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)

if not exist ".venv\Scripts\python.exe" (
    echo [Eyemouse] Creating virtual environment in .venv ...
    %PY% -m venv .venv || goto :fail
    echo [Eyemouse] Installing dependencies ^(this downloads ~200 MB^) ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

if not exist "models\face_landmarker.task" (
    echo [Eyemouse] Downloading MediaPipe model bundles ...
    ".venv\Scripts\python.exe" tools\download_models.py || goto :fail
)

".venv\Scripts\python.exe" main.py %*
if %errorlevel%==0 goto :eof

:fail
echo.
echo [Eyemouse] Setup failed. Run ".venv\Scripts\python.exe main.py --check"
echo            for a full diagnosis, and see the Troubleshooting section of
echo            README.md.
pause
exit /b 1
