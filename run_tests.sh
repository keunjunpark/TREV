#!/bin/bash
# Script to run TREV unit tests

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Running TREV Unit Tests ===${NC}"

# Parse command line arguments
RUN_MODE=${1:-"all"}

case $RUN_MODE in
    "all")
        echo -e "${GREEN}Running all tests...${NC}"
        pytest -v
        ;;
    "fast")
        echo -e "${GREEN}Running fast tests only...${NC}"
        pytest -v -m "not slow"
        ;;
    "slow")
        echo -e "${GREEN}Running slow tests only...${NC}"
        pytest -v -m "slow"
        ;;
    "cuda")
        echo -e "${GREEN}Running CUDA tests only...${NC}"
        pytest -v -m "cuda"
        ;;
    "no-cuda")
        echo -e "${GREEN}Running tests without CUDA...${NC}"
        pytest -v -m "not cuda"
        ;;
    "coverage")
        echo -e "${GREEN}Running tests with coverage report...${NC}"
        pytest --cov=src/TREV --cov-report=html --cov-report=term
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    "parallel")
        echo -e "${GREEN}Running tests in parallel...${NC}"
        pytest -n auto
        ;;
    "circuit")
        echo -e "${GREEN}Running circuit tests...${NC}"
        pytest -v tests/test_circuit.py
        ;;
    "hamiltonian")
        echo -e "${GREEN}Running hamiltonian tests...${NC}"
        pytest -v tests/test_hamiltonian.py
        ;;
    "gates")
        echo -e "${GREEN}Running gates tests...${NC}"
        pytest -v tests/test_gates.py
        ;;
    "measurement")
        echo -e "${GREEN}Running measurement tests...${NC}"
        pytest -v tests/test_measurement.py
        ;;
    "help")
        echo "Usage: ./run_tests.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all          - Run all tests (default)"
        echo "  fast         - Run only fast tests"
        echo "  slow         - Run only slow tests"
        echo "  cuda         - Run only CUDA tests"
        echo "  no-cuda      - Run tests without CUDA"
        echo "  coverage     - Run tests with coverage report"
        echo "  parallel     - Run tests in parallel"
        echo "  circuit      - Run circuit tests only"
        echo "  hamiltonian  - Run hamiltonian tests only"
        echo "  gates        - Run gates tests only"
        echo "  measurement  - Run measurement tests only"
        echo "  help         - Show this help message"
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown mode: $RUN_MODE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tests completed successfully${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
