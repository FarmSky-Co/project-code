#!/bin/bash

# Production Deployment Script for Farmer Credit Scoring System
# Usage: ./deploy.sh [environment]

set -e

# Configuration
ENVIRONMENT=${1:-production}
IMAGE_NAME="farmsky/credit-scoring"
TAG=${2:-latest}
REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}

echo "🚀 Deploying Farmer Credit Scoring System"
echo "Environment: $ENVIRONMENT"
echo "Image: $IMAGE_NAME:$TAG"
echo "Registry: $REGISTRY"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_info "Prerequisites OK"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build \
        --tag $IMAGE_NAME:$TAG \
        --tag $IMAGE_NAME:latest \
        --build-arg ENVIRONMENT=$ENVIRONMENT \
        .
    
    log_info "Image built successfully"
}

# Push to registry
push_image() {
    if [ "$REGISTRY" != "your-registry.com" ]; then
        log_info "Pushing image to registry..."
        
        docker tag $IMAGE_NAME:$TAG $REGISTRY/$IMAGE_NAME:$TAG
        docker push $REGISTRY/$IMAGE_NAME:$TAG
        
        log_info "Image pushed successfully"
    else
        log_warning "Registry not configured, skipping push"
    fi
}

# Deploy with Docker Compose
deploy() {
    log_info "Deploying application..."
    
    # Create directories if they don't exist
    mkdir -p data models results logs ssl
    
    # Copy environment file if it doesn't exist
    if [ ! -f .env ]; then
        cp .env.example .env
        log_warning "Created .env from .env.example - please review and update"
    fi
    
    # Deploy based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    log_info "Application deployed successfully"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            log_info "Health check passed"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts - waiting for application..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Show application info
show_info() {
    log_info "Deployment Information"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Application URL: http://localhost:8501"
    echo "Environment: $ENVIRONMENT"
    echo "Image: $IMAGE_NAME:$TAG"
    echo ""
    echo "Useful Commands:"
    echo "  View logs:     docker-compose logs -f farmsky-app"
    echo "  Stop app:      docker-compose down"
    echo "  Restart:       docker-compose restart"
    echo "  Shell access:  docker-compose exec farmsky-app /bin/bash"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Main execution
main() {
    check_prerequisites
    build_image
    push_image
    deploy
    
    if health_check; then
        show_info
        log_info "Deployment completed successfully! 🎉"
    else
        log_error "Deployment failed - check logs with: docker-compose logs"
        exit 1
    fi
}

# Run main function
main