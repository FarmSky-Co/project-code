#!/bin/bash

# Docker Setup Test Script
# Verifies that all Docker files are properly configured

echo "🧪 Testing Docker Setup for Farmer Credit Scoring System"
echo "========================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_count=0
fail_count=0

# Test function
test_check() {
    if [ $1 -eq 0 ]; then
        echo -e "✅ ${GREEN}PASS${NC}: $2"
        ((pass_count++))
    else
        echo -e "❌ ${RED}FAIL${NC}: $2"
        ((fail_count++))
    fi
}

# Check if files exist
echo "📁 Checking Docker files..."
test_check $([ -f Dockerfile ] && echo 0 || echo 1) "Dockerfile exists"
test_check $([ -f docker-compose.yml ] && echo 0 || echo 1) "docker-compose.yml exists"
test_check $([ -f .dockerignore ] && echo 0 || echo 1) ".dockerignore exists"
test_check $([ -f .env.example ] && echo 0 || echo 1) ".env.example exists"

# Check Docker syntax
echo ""
echo "🔍 Validating Docker files..."
docker-compose config > /dev/null 2>&1
test_check $? "docker-compose.yml syntax is valid"

# Check if required directories exist or can be created
echo ""
echo "📂 Checking directory structure..."
mkdir -p data models results logs ssl
test_check $? "Required directories created"

# Check if Dockerfile builds (dry run)
echo ""
echo "🏗️  Testing Docker build (syntax only)..."
docker build --no-cache --dry-run . > /dev/null 2>&1
if [ $? -eq 0 ] || grep -q "dry-run" <<< $(docker build --help 2>/dev/null); then
    # Some Docker versions don't support --dry-run, so we'll do a quick validation
    grep -q "FROM python:3.10-slim" Dockerfile
    test_check $? "Dockerfile base image is correct"
    
    grep -q "EXPOSE 8501" Dockerfile
    test_check $? "Dockerfile exposes correct port"
    
    grep -q "streamlit run" Dockerfile
    test_check $? "Dockerfile has correct startup command"
else
    test_check 1 "Dockerfile build test (Docker version may not support dry-run)"
fi

# Check environment file
echo ""
echo "⚙️  Validating environment configuration..."
if [ -f .env.example ]; then
    grep -q "STREAMLIT_SERVER_PORT" .env.example
    test_check $? ".env.example has required Streamlit config"
    
    grep -q "REDIS_URL" .env.example
    test_check $? ".env.example has Redis configuration"
else
    test_check 1 ".env.example file missing"
fi

# Check requirements.txt
echo ""
echo "📦 Validating Python dependencies..."
if [ -f requirements.txt ]; then
    grep -q "streamlit" requirements.txt
    test_check $? "requirements.txt includes Streamlit"
    
    grep -q "pandas" requirements.txt
    test_check $? "requirements.txt includes pandas"
    
    grep -q "scikit-learn" requirements.txt
    test_check $? "requirements.txt includes scikit-learn"
else
    test_check 1 "requirements.txt file missing"
fi

# Summary
echo ""
echo "📊 Test Summary"
echo "==============="
echo -e "✅ Passed: ${GREEN}$pass_count${NC}"
echo -e "❌ Failed: ${RED}$fail_count${NC}"
echo -e "Total: $((pass_count + fail_count))"

if [ $fail_count -eq 0 ]; then
    echo ""
    echo -e "🎉 ${GREEN}All tests passed!${NC} Your Docker setup is ready for deployment."
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.example to .env and configure"
    echo "2. Run: docker-compose up -d"
    echo "3. Access application at http://localhost:8501"
    exit 0
else
    echo ""
    echo -e "⚠️  ${YELLOW}Some tests failed.${NC} Please fix the issues above before deploying."
    exit 1
fi