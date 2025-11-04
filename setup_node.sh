#!/bin/bash
# Setup script for NIXL multi-node benchmarking
# Run this on both nodes to set up the environment

set -e  # Exit on error

echo "=========================================="
echo "NIXL Multi-Node Setup Script"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Check if PyTorch is installed
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch already installed"
    python3 -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
else
    echo "Installing PyTorch with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

# Check if NIXL is installed
if python3 -c "import nixl" 2>/dev/null; then
    echo "✓ NIXL already installed"
    python3 -c "import nixl; print(f'  Version: {nixl.__version__}')"
else
    echo "Installing NIXL..."
    pip install nixl
fi

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
python3 verify_nixl_setup.py

echo ""
echo "=========================================="
echo "Network Information"
echo "=========================================="
echo "IP Address: $(hostname -I | awk '{print $1}')"
echo "Hostname: $(hostname)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Note your IP address above"
echo "2. Test connectivity to the other node: ping <other_node_ip>"
echo "3. Run the benchmark (see QUICKSTART.md)"
echo ""

