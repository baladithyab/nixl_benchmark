#!/bin/bash

sudo apt install python3.10-venv -y

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch using official command from pytorch.org
# This is the command for Linux with CUDA support
echo "Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install NIXL from PyPI (as mentioned in the official NIXL documentation)
echo "Installing NIXL..."
pip install nixl

# Install other requirements if they exist
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
fi

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate" 