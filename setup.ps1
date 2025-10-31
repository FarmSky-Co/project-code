# Farmer Credit Scoring System - Setup Script
# Run this script to set up the environment and verify installation

Write-Host "========================================" -ForegroundColor Green
Write-Host "Farmer Credit Scoring System - Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
Write-Host ""
Write-Host "Checking virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    Write-Host "  [OK] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  [OK] Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "  [OK] Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "  [WARNING] Could not activate virtual environment" -ForegroundColor Yellow
    Write-Host "  You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
}

# Upgrade pip, setuptools, and wheel first
Write-Host ""
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Core tools upgraded" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Could not upgrade core tools, continuing anyway..." -ForegroundColor Yellow
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Error installing dependencies" -ForegroundColor Red
    Write-Host "  Try running: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Create directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow
$directories = @("data", "models", "results")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  [OK] Created $dir/" -ForegroundColor Green
    } else {
        Write-Host "  [OK] $dir/ already exists" -ForegroundColor Green
    }
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
$testScript = @"
import sys
try:
    import pandas
    import numpy
    import sklearn
    import xgboost
    import lightgbm
    import streamlit
    import plotly
    import pydantic
    import faker
    print('SUCCESS')
except ImportError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"@

$testResult = python -c $testScript
if ($testResult -eq 'SUCCESS') {
    Write-Host "  [OK] All required packages installed" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Package verification failed: $testResult" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate virtual environment:" -ForegroundColor White
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Launch web interface:" -ForegroundColor White
Write-Host "     streamlit run app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  OR use command line:" -ForegroundColor White
Write-Host "     python main.py --help" -ForegroundColor Yellow
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  - QUICKSTART.md  (quick start guide)" -ForegroundColor White
Write-Host "  - README.md      (comprehensive docs)" -ForegroundColor White
Write-Host "  - PROJECT_SUMMARY.md (overview)" -ForegroundColor White
Write-Host ""
Write-Host "Happy credit scoring!" -ForegroundColor Green
