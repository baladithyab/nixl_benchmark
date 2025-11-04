#!/bin/bash

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NVIDIA GDS
sudo apt-get install -y nvidia-gds-12-1

# Clean up
rm cuda-keyring_1.1-1_all.deb