#!/bin/bash

# Setup NIXL Environment Script
# This script sets up the environment to use the system NIXL installation with GDS support

echo "=== Setting up NIXL Environment ==="

# Remove the problematic editable installation
echo "1. Removing editable NIXL installation..."
pip uninstall nixl -y

# Create a simple nixl package that points to the system installation
echo "2. Creating NIXL package link..."

NIXL_DIR="/home/ubuntu/nixl_benchmark/venv/lib/python3.12/site-packages/nixl"
mkdir -p "$NIXL_DIR"

# Create __init__.py
cat > "$NIXL_DIR/__init__.py" << 'EOF'
# NIXL package - links to system installation
import sys
import os

# Add system NIXL to path
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')

# Import from system installation
try:
    from nixl import *
except ImportError as e:
    print(f"Warning: Could not import NIXL from system installation: {e}")
    print("Make sure NIXL is properly installed in /usr/local/lib/python3.12/site-packages/")
EOF

# Create _api.py link
cat > "$NIXL_DIR/_api.py" << 'EOF'
# Link to system NIXL API
import sys
import os

# Add system NIXL to path
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')

# Import from system installation
try:
    from nixl._api import *
except ImportError as e:
    print(f"Warning: Could not import NIXL API from system installation: {e}")
EOF

# Create _utils.py link
cat > "$NIXL_DIR/_utils.py" << 'EOF'
# Link to system NIXL utils
import sys
import os

# Add system NIXL to path
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')

# Import from system installation
try:
    from nixl._utils import *
except ImportError as e:
    print(f"Warning: Could not import NIXL utils from system installation: {e}")
EOF

# Set environment variable to point to system plugins
export NIXL_PLUGIN_DIR="/usr/local/lib/x86_64-linux-gnu/plugins"

echo "3. Setting NIXL_PLUGIN_DIR to: $NIXL_PLUGIN_DIR"

# Add to shell profile
echo "export NIXL_PLUGIN_DIR=\"$NIXL_PLUGIN_DIR\"" >> ~/.bashrc

echo "4. Testing NIXL installation..."
python -c "import nixl; print('NIXL imported successfully')"

echo "5. Running plugin check..."
python check_nixl_plugins.py

echo "=== Setup Complete ==="
echo "NIXL environment is now configured to use the system installation with GDS support."
echo "You may need to restart your shell or run 'source ~/.bashrc' for changes to take effect." 