@echo off
REM Windows Deployment Script for Farmer Credit Scoring System
REM Usage: deploy.bat [environment]

setlocal enabledelayedexpansion

REM Configuration
set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=production

set IMAGE_NAME=farmsky/credit-scoring
set TAG=%2
if "%TAG%"=="" set TAG=latest

set REGISTRY=%DOCKER_REGISTRY%
if "%REGISTRY%"=="" set REGISTRY=your-registry.com

echo.
echo ==========================================
echo   Farmer Credit Scoring System Deploy
echo ==========================================
echo Environment: %ENVIRONMENT%
echo Image: %IMAGE_NAME%:%TAG%
echo Registry: %REGISTRY%
echo.

REM Check prerequisites
echo [INFO] Checking prerequisites...
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not in PATH
    pause
    exit /b 1
)

where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Compose is not installed or not in PATH
    pause
    exit /b 1
)

echo [INFO] Prerequisites OK

REM Build Docker image
echo [INFO] Building Docker image...
docker build --tag %IMAGE_NAME%:%TAG% --tag %IMAGE_NAME%:latest --build-arg ENVIRONMENT=%ENVIRONMENT% .
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to build Docker image
    pause
    exit /b 1
)
echo [INFO] Image built successfully

REM Push to registry (optional)
if not "%REGISTRY%"=="your-registry.com" (
    echo [INFO] Pushing image to registry...
    docker tag %IMAGE_NAME%:%TAG% %REGISTRY%/%IMAGE_NAME%:%TAG%
    docker push %REGISTRY%/%IMAGE_NAME%:%TAG%
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Failed to push to registry
    ) else (
        echo [INFO] Image pushed successfully
    )
) else (
    echo [WARNING] Registry not configured, skipping push
)

REM Create directories
echo [INFO] Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "ssl" mkdir ssl

REM Copy environment file
if not exist ".env" (
    copy ".env.example" ".env"
    echo [WARNING] Created .env from .env.example - please review and update
)

REM Deploy
echo [INFO] Deploying application...
if "%ENVIRONMENT%"=="production" (
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
) else (
    docker-compose up -d
)

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to deploy application
    pause
    exit /b 1
)

echo [INFO] Application deployed successfully

REM Health check
echo [INFO] Performing health check...
set /a attempt=1
set /a max_attempts=30

:health_loop
curl -f http://localhost:8501/_stcore/health >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Health check passed
    goto :health_success
)

echo Attempt %attempt%/%max_attempts% - waiting for application...
timeout /t 10 >nul
set /a attempt+=1

if %attempt% LEQ %max_attempts% goto :health_loop

echo [ERROR] Health check failed after %max_attempts% attempts
echo [ERROR] Check logs with: docker-compose logs
pause
exit /b 1

:health_success

REM Show deployment info
echo.
echo ==========================================
echo       Deployment Completed Successfully!
echo ==========================================
echo.
echo Application URL: http://localhost:8501
echo Environment: %ENVIRONMENT%
echo Image: %IMAGE_NAME%:%TAG%
echo.
echo Useful Commands:
echo   View logs:     docker-compose logs -f farmsky-app
echo   Stop app:      docker-compose down
echo   Restart:       docker-compose restart
echo   Shell access:  docker-compose exec farmsky-app /bin/bash
echo.
echo ==========================================
echo.

echo [INFO] Deployment completed successfully! 🎉
pause