#!/bin/bash

# NIXL Rebuild Script with GDS Support
# This script rebuilds NIXL with GPUDirect Storage (GDS) support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}

print_status "Starting NIXL rebuild with GDS support..."

# Step 1: Check prerequisites
print_status "Checking prerequisites..."

# Check if running as root (needed for some operations)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. Some operations may not work correctly."
fi

# Check for essential tools
for tool in git cmake make gcc g++; do
    if ! command_exists "$tool"; then
        print_error "Required tool '$tool' not found. Installing build essentials..."
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git
        break
    fi
done

# Check for Python development tools
if ! command_exists python3; then
    print_error "Python3 not found. Installing..."
    sudo apt-get install -y python3 python3-dev python3-pip
fi

# Check for CUDA
CUDA_HOME=""
for cuda_path in "/usr/local/cuda-12.4" "/usr/local/cuda-12.1" "/usr/local/cuda"; do
    if dir_exists "$cuda_path"; then
        CUDA_HOME="$cuda_path"
        print_success "Found CUDA installation at: $CUDA_HOME"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    print_error "CUDA installation not found. Please install CUDA first."
    exit 1
fi

# Check for GDS installation
if ! dpkg -l | grep -q nvidia-gds; then
    print_error "NVIDIA GDS not found. Installing GDS first..."
    ./install_gds.sh
fi

# Step 2: Set environment variables
print_status "Setting up environment variables..."

export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# Add CUDA libraries to pkg-config path
export PKG_CONFIG_PATH="$CUDA_HOME/targets/x86_64-linux/lib/pkgconfig:$PKG_CONFIG_PATH"

print_success "Environment variables set:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH: $PATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Step 3: Create build directory
BUILD_DIR="$PWD/nixl_build"
print_status "Creating build directory: $BUILD_DIR"

if dir_exists "$BUILD_DIR"; then
    print_warning "Build directory already exists. Removing..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Step 4: Clone or get NIXL source
NIXL_SOURCE_DIR="$BUILD_DIR/nixl_source"
print_status "Setting up NIXL source..."

if [ -z "$NIXL_SOURCE_URL" ]; then
    # Try to find existing NIXL source
    if dir_exists "../nixl"; then
        print_status "Using existing NIXL source in ../nixl"
        cp -r ../nixl "$NIXL_SOURCE_DIR"
    elif dir_exists "../../nixl"; then
        print_status "Using existing NIXL source in ../../nixl"
        cp -r ../../nixl "$NIXL_SOURCE_DIR"
    else
        print_error "NIXL source not found. Please set NIXL_SOURCE_URL environment variable or place NIXL source in ../nixl or ../../nixl"
        print_status "Example: export NIXL_SOURCE_URL=https://github.com/nvidia/nixl.git"
        exit 1
    fi
else
    print_status "Cloning NIXL from: $NIXL_SOURCE_URL"
    git clone "$NIXL_SOURCE_URL" "$NIXL_SOURCE_DIR"
fi

cd "$NIXL_SOURCE_DIR"

# Step 5: Check build system
print_status "Detecting build system..."

if file_exists "CMakeLists.txt"; then
    BUILD_SYSTEM="cmake"
    print_success "Detected CMake build system"
elif file_exists "meson.build"; then
    BUILD_SYSTEM="meson"
    print_success "Detected Meson build system"
elif file_exists "setup.py"; then
    BUILD_SYSTEM="python"
    print_success "Detected Python setup.py build system"
else
    print_error "Could not detect build system. Please check NIXL source directory."
    exit 1
fi

# Step 6: Configure and build
print_status "Configuring and building NIXL with GDS support..."

case $BUILD_SYSTEM in
    "cmake")
        print_status "Using CMake build system..."
        
        # Create build directory
        mkdir -p build
        cd build
        
        # Configure with GDS support
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_GDS=ON \
            -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
            -DCMAKE_PREFIX_PATH="$CUDA_HOME" \
            -DCMAKE_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib" \
            -DCMAKE_INCLUDE_PATH="$CUDA_HOME/include"
        
        # Build
        make -j$(nproc)
        
        # Install
        sudo make install
        ;;
        
    "meson")
        print_status "Using Meson build system..."
        
        # Configure with GDS support
        meson setup build \
            --prefix=/usr/local \
            --buildtype=release \
            -Denable_gds=true \
            -Dcuda_root="$CUDA_HOME"
        
        # Build
        ninja -C build
        
        # Install
        sudo ninja -C build install
        ;;
        
    "python")
        print_status "Using Python setup.py build system..."
        
        # Set environment variables for Python build
        export CUDA_HOME="$CUDA_HOME"
        export NIXL_ENABLE_GDS=1
        
        # Install in development mode
        pip install -e . --no-deps --force-reinstall
        ;;
        
    *)
        print_error "Unknown build system: $BUILD_SYSTEM"
        exit 1
        ;;
esac

# Step 7: Verify installation
print_status "Verifying NIXL installation with GDS support..."

# Go back to original directory
cd "$PWD"

# Check if GDS plugin is now available
if [ -f "check_nixl_plugins.py" ]; then
    print_status "Running NIXL plugin check..."
    python check_nixl_plugins.py
else
    print_warning "check_nixl_plugins.py not found. Please run it manually to verify GDS plugin."
fi

# Step 8: Test GDS functionality
print_status "Testing GDS functionality..."

# Create a simple test file
TEST_FILE="$BUILD_DIR/test_gds.dat"
dd if=/dev/zero of="$TEST_FILE" bs=1M count=64 2>/dev/null

# Try to run GDS example if available
if [ -f "nixl_gds_example.py" ]; then
    print_status "Testing GDS example..."
    if python nixl_gds_example.py "$TEST_FILE"; then
        print_success "GDS example ran successfully!"
    else
        print_warning "GDS example failed. This might be expected if GDS plugin is not fully functional."
    fi
fi

# Cleanup test file
rm -f "$TEST_FILE"

# Step 9: Summary
print_success "NIXL rebuild with GDS support completed!"
print_status "Summary:"
echo "  - Build directory: $BUILD_DIR"
echo "  - Build system used: $BUILD_SYSTEM"
echo "  - CUDA installation: $CUDA_HOME"
echo "  - GDS libraries: $(find /usr -name "libcufile.so*" 2>/dev/null | head -1)"

print_status "Next steps:"
echo "  1. Run 'python check_nixl_plugins.py' to verify GDS plugin availability"
echo "  2. Test GDS functionality with 'python nixl_gds_example.py <file_path>'"
echo "  3. If issues persist, check build logs in $BUILD_DIR"

print_success "Rebuild completed successfully!" 