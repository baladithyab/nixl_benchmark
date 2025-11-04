#!/bin/bash

# Simple NIXL Rebuild Script with GDS Support
# This script provides a simplified approach to rebuilding NIXL with GDS

set -e

echo "=== NIXL Rebuild with GDS Support ==="
echo

# Check if we have the necessary tools
echo "1. Checking prerequisites..."
if ! command -v cmake >/dev/null 2>&1; then
    echo "Installing build tools..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git
fi

# Check for Meson and Ninja
MESON_OK=false
if command -v meson >/dev/null 2>&1; then
    MESON_VERSION=$(meson --version | awk -F. '{printf "%d%02d", $1, $2}')
    if [ "$MESON_VERSION" -ge 64 ]; then
        MESON_OK=true
    fi
fi
if ! $MESON_OK; then
    echo "Installing/upgrading Meson >= 0.64.0 via pip..."
    pip install --upgrade pip
    pip install 'meson>=0.64.0'
    export PATH=~/.local/bin:$PATH
fi
if ! command -v ninja >/dev/null 2>&1; then
    echo "Installing Ninja..."
    sudo apt-get update
    sudo apt-get install -y ninja-build
fi

# Check for pybind11 (needed for Python bindings)
if ! pkg-config --exists pybind11 2>/dev/null; then
    echo "Installing pybind11..."
    pip install pybind11
fi

# Find CUDA installation
CUDA_HOME=""
for path in "/usr/local/cuda-12.4" "/usr/local/cuda-12.1" "/usr/local/cuda"; do
    if [ -d "$path" ]; then
        CUDA_HOME="$path"
        echo "Found CUDA at: $CUDA_HOME"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: CUDA not found. Please install CUDA first."
    exit 1
fi

# Set environment variables
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

echo "2. Setting up environment..."
echo "   CUDA_HOME: $CUDA_HOME"
echo "   LD_LIBRARY_PATH includes: $CUDA_HOME/targets/x86_64-linux/lib"

# Check for NIXL source
echo "3. Looking for NIXL source..."
NIXL_SOURCE=""
for path in "../nixl" "../../nixl" "./nixl"; do
    if [ -d "$path" ]; then
        NIXL_SOURCE="$path"
        echo "   Found NIXL source at: $NIXL_SOURCE"
        break
    fi
done

if [ -z "$NIXL_SOURCE" ]; then
    echo "ERROR: NIXL source not found."
    echo "Please either:"
    echo "  - Set NIXL_SOURCE_URL environment variable"
    echo "  - Place NIXL source in ../nixl, ../../nixl, or ./nixl"
    echo "  - Example: export NIXL_SOURCE_URL=https://github.com/nvidia/nixl.git"
    exit 1
fi

# Create build directory
BUILD_DIR="./nixl_build"
echo "4. Creating build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Copy or clone NIXL source
if [ "$NIXL_SOURCE" != "" ]; then
    echo "5. Copying NIXL source..."
    cp -r "../$NIXL_SOURCE" ./nixl_source
    cd nixl_source
    echo "   Source copied successfully"
else
    echo "5. Cloning NIXL source..."
    git clone "$NIXL_SOURCE_URL" nixl_source
    cd nixl_source
fi

# Detect build system and build
echo "6. Detecting build system..."

if [ -f "CMakeLists.txt" ]; then
    echo "   Using CMake build system..."
    mkdir -p build
    cd build
    
    echo "   Configuring with GDS support..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_GDS=ON \
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
        -DCMAKE_PREFIX_PATH="$CUDA_HOME"
    
    echo "   Building..."
    make -j$(nproc)
    
    echo "   Installing..."
    sudo make install
    
elif [ -f "meson.build" ]; then
    echo "   Using Meson build system..."
    
    echo "   Configuring with GDS support..."
    meson setup build \
        --prefix=/usr/local \
        --buildtype=release \
        -Ddisable_gds_backend=false \
        -Dgds_path="$CUDA_HOME" \
        -Dcuda_root="$CUDA_HOME"
    
    echo "   Building..."
    ninja -C build
    
    echo "   Installing..."
    sudo ninja -C build install
    
elif [ -f "setup.py" ]; then
    echo "   Using Python setup.py build system..."
    
    export NIXL_ENABLE_GDS=1
    pip install -e . --no-deps --force-reinstall
    
else
    echo "ERROR: Could not detect build system."
    echo "Please check that NIXL source contains CMakeLists.txt, meson.build, or setup.py"
    exit 1
fi

# Return to original directory
cd ../..

echo "7. Verifying installation..."
if [ -f "check_nixl_plugins.py" ]; then
    echo "   Running plugin check..."
    python check_nixl_plugins.py
else
    echo "   Please run 'python check_nixl_plugins.py' to verify GDS plugin availability"
fi

echo
echo "=== Build Complete ==="
echo "If GDS plugin is now available, you can test it with:"
echo "  python nixl_gds_example.py <file_path>"
echo
echo "If you still don't see GDS plugin, check:"
echo "  1. Build logs in $BUILD_DIR"
echo "  2. That NIXL source supports GDS"
echo "  3. That all GDS libraries are properly installed" 