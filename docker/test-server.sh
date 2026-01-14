#!/bin/bash
# Script to manage local OpenML test server for development and CI
# This script starts Docker services for local testing to avoid race conditions
# and server load issues with the remote test.openml.org server.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.test.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_usage() {
    echo "Usage: $0 [start|stop|restart|status|logs]"
    echo ""
    echo "Commands:"
    echo "  start    - Start local OpenML test server"
    echo "  stop     - Stop local OpenML test server"
    echo "  restart  - Restart local OpenML test server"
    echo "  status   - Check status of test server services"
    echo "  logs     - Show logs from test server services"
    echo ""
    echo "Example:"
    echo "  $0 start          # Start the test server"
    echo "  $0 status         # Check if services are running"
    echo "  pytest --local-server  # Run tests against local server"
}

function check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
}

function start_server() {
    echo -e "${GREEN}Starting local OpenML test server...${NC}"
    check_docker
    
    # Check if services are already running
    if docker ps | grep -q "openml-test-db\|openml-php-api"; then
        echo -e "${YELLOW}Warning: Some services are already running${NC}"
        echo "Use '$0 restart' to restart all services"
        return
    fi
    
    cd "$SCRIPT_DIR"
    
    # Note: We'll use placeholder images until official images are available
    echo -e "${YELLOW}Note: Using placeholder Docker configuration${NC}"
    echo -e "${YELLOW}In production, this will use official OpenML server images${NC}"
    
    docker-compose -f "$COMPOSE_FILE" up -d
    
    echo ""
    echo -e "${GREEN}Waiting for services to be healthy...${NC}"
    sleep 5
    
    # Check health status
    if docker ps | grep -q "openml-test-db.*healthy"; then
        echo -e "${GREEN}✓ Database is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Database is starting...${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}Local test server started!${NC}"
    echo "  - Database: localhost:3307"
    echo "  - PHP API v1: http://localhost:8080"
    echo "  - Python API v2: http://localhost:8000"
    echo ""
    echo "Run tests with: pytest --local-server"
    echo "View logs with: $0 logs"
}

function stop_server() {
    echo -e "${GREEN}Stopping local OpenML test server...${NC}"
    check_docker
    
    cd "$SCRIPT_DIR"
    docker-compose -f "$COMPOSE_FILE" down
    
    echo -e "${GREEN}Server stopped${NC}"
}

function restart_server() {
    stop_server
    echo ""
    start_server
}

function show_status() {
    echo -e "${GREEN}OpenML Test Server Status:${NC}"
    echo ""
    
    check_docker
    
    if ! docker ps | grep -q "openml-test-db\|openml-php-api\|openml-python-api"; then
        echo -e "${YELLOW}No services are running${NC}"
        echo "Use '$0 start' to start the test server"
        return
    fi
    
    echo "Running containers:"
    docker ps --filter "name=openml-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

function show_logs() {
    echo -e "${GREEN}OpenML Test Server Logs:${NC}"
    check_docker
    
    cd "$SCRIPT_DIR"
    docker-compose -f "$COMPOSE_FILE" logs -f --tail=100
}

# Main script logic
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
