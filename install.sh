#!/bin/bash
# ============================================
# DataWarrior-Agent - Quick Installation Script
# ============================================
# 
# This script installs Python dependencies only.
# For full installation guide, see README.md
#
# Prerequisites:
#   - DataWarrior installed
#   - macroagent.jar in ~/.datawarrior/plugin/
#   - Conda environment activated
#
# ============================================

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════╗"
echo "║     🔬 DataWarrior-Agent - Quick Install           ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda environment is active
echo -e "${YELLOW}[1/4]${NC} Checking environment..."
if [[ "$CONDA_DEFAULT_ENV" == "datawarrior-ai" ]]; then
    echo -e "${GREEN}✅ Conda environment 'datawarrior-ai' is active${NC}"
else
    echo -e "${YELLOW}⚠️  Conda environment 'datawarrior-ai' is not active${NC}"
    echo "   Please run: conda activate datawarrior-ai"
    echo "   Or create it: conda create -n datawarrior-ai python=3.11 -y"
    exit 1
fi

# Check Python version
echo ""
echo -e "${YELLOW}[2/4]${NC} Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}[3/4]${NC} Upgrading pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install Python dependencies
echo ""
echo -e "${YELLOW}[4/4]${NC} Installing Python dependencies..."
echo "   ⏳ This may take 5-10 minutes (downloading ~3 GB: PyTorch, Transformers, etc.)"
echo ""
pip install -r requirements.txt
echo ""
echo -e "${GREEN}✅ Python dependencies installed${NC}"

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data/input data/output logs .config
echo -e "${GREEN}✅ Directories created${NC}"

# Check DataWarrior plugin
echo ""
echo "Checking DataWarrior plugin..."
if [ -f "$HOME/.datawarrior/plugin/macroagent.jar" ]; then
    echo -e "${GREEN}✅ macroagent.jar found in ~/.datawarrior/plugin/${NC}"
else
    echo -e "${YELLOW}⚠️  macroagent.jar not found in ~/.datawarrior/plugin/${NC}"
    echo "   Please copy it: cp plugin/macroagent.jar ~/.datawarrior/plugin/"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║           ✅ Installation Complete!                ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║                                                    ║"
echo "║  To run DataWarrior-Agent:                            ║"
echo "║                                                    ║"
echo "║    python run.py                                   ║"
echo "║                                                    ║"
echo "║  Or manually:                                      ║"
echo "║    ./scripts/start_vnc.sh                          ║"
echo "║    streamlit run streamlit_app.py                  ║"
echo "║                                                    ║"
echo "║  Access: http://localhost:8501                     ║"
echo "║                                                    ║"
echo "╚════════════════════════════════════════════════════╝"
