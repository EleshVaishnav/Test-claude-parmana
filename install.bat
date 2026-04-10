@echo off
setlocal enabledelayedexpansion
echo ^ Parmana 2.0 installer

:: Python check
where python >nul 2>&1
if errorlevel 1 (
    echo X python not found. Install Python 3.10+ from python.org and retry.
    exit /b 1
)

for /f "tokens=2 delims=." %%a in ('python --version 2^>^&1') do set MINOR=%%a
if !MINOR! LSS 10 (
    echo X Python 3.10+ required.
    exit /b 1
)

:: Clone or update
if exist "Parmana-2.0" (
    echo ^ Directory exists - pulling latest...
    cd Parmana-2.0
    git pull
) else (
    git clone https://github.com/EleshVaishnav/Test-claude-parmana.git
    cd Parmana-2.0
)

:: Virtual environment
if not exist ".venv" (
    echo ^ Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

:: Upgrade pip
pip install --upgrade pip -q

:: Install dependencies
echo ^ Installing dependencies (this may take a minute)...
pip install -r requirements.txt -q

:: .env setup
if not exist ".env" (
    copy .env.example .env >nul
    echo ^ .env created from .env.example - add your API keys.
) else (
    echo ^ .env already exists - skipping.
)

echo.
echo Done. To start:
echo   cd Parmana-2.0 ^&^& .venv\Scripts\activate ^&^& python main.py
