#!/bin/bash

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Try different package names
echo "Trying to install NVIDIA GDS..."

# Try the original package name
if sudo apt-get install -y nvidia-gds-12-1 2>/dev/null; then
    echo "Successfully installed nvidia-gds-12-1"
    exit 0
fi

# Try alternative package names
for package in nvidia-gds nvidia-gds-12 nvidia-gds-11-8 nvidia-gds-11-9; do
    echo "Trying $package..."
    if sudo apt-get install -y $package 2>/dev/null; then
        echo "Successfully installed $package"
        exit 0
    fi
done

# If all fail, check what's available
echo "No GDS package found. Available NVIDIA packages:"
apt search nvidia-gds 2>/dev/null | head -20

# Clean up
rm -f cuda-keyring_1.1-1_all.deb 