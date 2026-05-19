@echo off
title Media Sorter — Setup
color 0A
echo.
echo =====================================================
echo   Media Sorter — First-Time Setup
echo =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python is not installed on this computer.
    echo.
    echo  Please install Python 3.10 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: Check "Add Python to PATH" during install.
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  Python found:
python --version
echo.
echo  Installing required packages (this may take 5-15 minutes
echo  depending on your internet speed — please wait)...
echo.

pip install -r "%~dp0requirements.txt"
pip install facenet-pytorch --no-deps --quiet
pip install pillow-heif --quiet

if errorlevel 1 (
    echo.
    echo  Something went wrong during install.
    echo  Try running this file as Administrator (right-click, Run as administrator).
    pause
    exit /b 1
)

echo.
echo =====================================================
echo   Setup complete!
echo.
echo   The AI models (MegaDetector and CLIP) will
echo   download automatically the first time you run
echo   a sort — about 640 MB total, one time only.
echo.
echo   Double-click "Launch Media Sorter.bat" to start.
echo =====================================================
echo.
pause
