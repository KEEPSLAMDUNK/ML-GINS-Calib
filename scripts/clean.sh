#!/bin/bash

# Clean script for ML-GINS-Calib

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Cleaning build directory...${NC}"

# Remove build directory
if [ -d "build" ]; then
    rm -rf build
    echo -e "${GREEN}Build directory removed.${NC}"
else
    echo -e "${YELLOW}Build directory not found.${NC}"
fi

echo -e "${GREEN}Clean completed!${NC}" 