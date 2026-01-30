#!/bin/bash

# Docker Build and Run Script

set -e

echo "========================================"
echo "Demand Forecasting Docker Build"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Parse arguments
ACTION=${1:-"build"}

case $ACTION in
    "build")
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker-compose build
        echo -e "${GREEN}Build complete!${NC}"
        ;;
    
    "up")
        echo -e "${YELLOW}Starting services...${NC}"
        docker-compose up -d
        echo -e "${GREEN}Services started!${NC}"
        echo ""
        echo "API:       http://localhost:8000"
        echo "API Docs:  http://localhost:8000/docs"
        echo "Dashboard: http://localhost:8501"
        ;;
    
    "down")
        echo -e "${YELLOW}Stopping services...${NC}"
        docker-compose down
        echo -e "${GREEN}Services stopped!${NC}"
        ;;
    
    "logs")
        docker-compose logs -f
        ;;
    
    "restart")
        echo -e "${YELLOW}Restarting services...${NC}"
        docker-compose restart
        echo -e "${GREEN}Services restarted!${NC}"
        ;;
    
    "status")
        docker-compose ps
        ;;
    
    "clean")
        echo -e "${YELLOW}Cleaning up...${NC}"
        docker-compose down -v --rmi local
        echo -e "${GREEN}Cleanup complete!${NC}"
        ;;
    
    *)
        echo "Usage: $0 {build|up|down|logs|restart|status|clean}"
        exit 1
        ;;
esac
