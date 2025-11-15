#!/bin/bash

# Coolify Setup Script for Farmer Credit Scoring System
# Prepares repository for Coolify deployment

set -e

echo "🚀 Setting up Farmer Credit Scoring System for Coolify"
echo "======================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    log_error "app.py not found. Are you in the right directory?"
    exit 1
fi

log_info "Repository detected ✓"

# Create necessary directories
log_info "Creating directories..."
mkdir -p data models results logs .github/workflows

# Copy Coolify-specific files
log_info "Setting up Coolify configuration..."

# Use Coolify-optimized Dockerfile
if [ -f "Dockerfile.coolify" ]; then
    log_info "Using Coolify-optimized Dockerfile"
else
    log_warning "Dockerfile.coolify not found"
fi

# Setup environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    log_info "Created .env from .env.example"
    log_warning "Please review and update .env with your settings"
else
    log_info ".env already exists"
fi

# Setup gitignore for Coolify
if [ -f ".gitignore.coolify" ]; then
    if [ ! -f ".gitignore" ]; then
        cp .gitignore.coolify .gitignore
        log_info "Created .gitignore from .gitignore.coolify"
    else
        log_info ".gitignore already exists"
    fi
fi

# Validate requirements.txt
if [ -f "requirements.txt" ]; then
    log_info "Checking Python dependencies..."
    
    # Check for essential packages
    if grep -q "streamlit" requirements.txt; then
        log_info "✓ Streamlit found in requirements.txt"
    else
        log_warning "Streamlit not found in requirements.txt"
    fi
    
    if grep -q "pandas" requirements.txt; then
        log_info "✓ Pandas found in requirements.txt"
    else
        log_warning "Pandas not found in requirements.txt"
    fi
else
    log_error "requirements.txt not found"
    exit 1
fi

# Test Docker build (syntax check)
log_info "Testing Dockerfile syntax..."
if command -v docker >/dev/null 2>&1; then
    if docker build -f Dockerfile.coolify -t test-build --dry-run . >/dev/null 2>&1 || \
       docker build -f Dockerfile.coolify -t test-build . >/dev/null 2>&1; then
        log_info "✓ Dockerfile syntax is valid"
        docker rmi test-build >/dev/null 2>&1 || true
    else
        log_warning "Dockerfile syntax check failed (but this might be normal)"
    fi
else
    log_warning "Docker not available for syntax check"
fi

# Create sample data if needed
if [ ! -f "data/sample_farmers.csv" ] && [ -f "data_generator.py" ]; then
    log_info "Generating sample data..."
    python data_generator.py || log_warning "Failed to generate sample data"
fi

# Git repository setup
if [ -d ".git" ]; then
    log_info "Git repository detected"
    
    # Add Coolify files to git
    git add .env.example coolify.yml Dockerfile.coolify package.json >/dev/null 2>&1 || true
    log_info "Added Coolify files to git staging"
    
    # Check if there are uncommitted changes
    if ! git diff --cached --quiet 2>/dev/null; then
        log_warning "You have uncommitted changes. Consider committing them before deploying."
    fi
else
    log_warning "Not a git repository. Consider initializing git for Coolify integration."
fi

# Display next steps
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Next steps for Coolify deployment:"
echo ""
echo "1. 📝 Review and update your .env file:"
echo "   nano .env"
echo ""
echo "2. 🏗️ Push to your git repository:"
echo "   git add ."
echo "   git commit -m 'Setup Coolify deployment'"
echo "   git push origin main"
echo ""
echo "3. 🌐 In Coolify dashboard:"
echo "   - Add new application"
echo "   - Connect your git repository"
echo "   - Set Dockerfile path: ./Dockerfile.coolify"
echo "   - Configure port: 8501"
echo "   - Add environment variables from .env"
echo "   - Deploy!"
echo ""
echo "4. 🔗 Optional - Add Redis service:"
echo "   - Add Redis 7-alpine service"
echo "   - Connect to your app"
echo "   - Update REDIS_URL in environment"
echo ""
echo "📚 For detailed instructions, see: COOLIFY_DEPLOYMENT_GUIDE.md"
echo ""
echo "🚀 Your app will be available at: https://your-domain.com"
echo ""

log_info "Setup script completed successfully!"