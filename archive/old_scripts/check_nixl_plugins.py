#!/usr/bin/env python3

import os
import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

def check_nixl_installation():
    """Check NIXL installation and available plugins"""
    
    print("=== NIXL Installation Diagnostic ===\n")
    
    # Check environment
    print("1. Environment Variables:")
    nixl_plugin_dir = os.environ.get("NIXL_PLUGIN_DIR", "NOT SET")
    print(f"   NIXL_PLUGIN_DIR: {nixl_plugin_dir}")
    
    if nixl_plugin_dir != "NOT SET":
        if os.path.exists(nixl_plugin_dir):
            print(f"   ✓ Plugin directory exists")
            plugins_in_dir = [f for f in os.listdir(nixl_plugin_dir) if f.endswith('.so')]
            print(f"   Files in plugin directory: {plugins_in_dir}")
        else:
            print(f"   ✗ Plugin directory does not exist")
    
    print("\n2. Available NIXL Plugins:")
    try:
        # Try with empty backend list first
        agent_config = nixl_agent_config(backends=[])
        agent = nixl_agent("DiagnosticAgent", agent_config)
        
        plugin_list = agent.get_plugin_list()
        print(f"   Available plugins: {plugin_list}")
        
        if not plugin_list:
            print("   ⚠ No plugins found!")
        else:
            print("\n3. Plugin Details:")
            for plugin in plugin_list:
                print(f"\n   Plugin: {plugin}")
                try:
                    mem_types = agent.get_plugin_mem_types(plugin)
                    print(f"     Memory types: {mem_types}")
                except Exception as e:
                    print(f"     Error getting memory types: {e}")
                
                try:
                    params = agent.get_plugin_params(plugin)
                    print(f"     Parameters: {params}")
                except Exception as e:
                    print(f"     Error getting parameters: {e}")
        
        # Check if GDS is specifically missing
        if "GDS" not in plugin_list:
            print("\n4. GDS Plugin Status:")
            print("   ✗ GDS plugin not found in available plugins")
            print("\n   Possible solutions:")
            print("   1. Install NVIDIA GDS: ./install_gds.sh")
            print("   2. Check if NIXL was built with GDS support")
            print("   3. Verify CUDA and driver versions are compatible")
            print("   4. Check if GDS kernel modules are loaded")
            
            # Check for GDS-related files
            print("\n5. System GDS Check:")
            gds_files = [
                "/usr/lib/x86_64-linux-gnu/libnvidia-gds.so*",
                "/usr/lib/x86_64-linux-gnu/libcufile.so*",
                "/usr/lib/x86_64-linux-gnu/libcufile_rdma.so*"
            ]
            
            for pattern in gds_files:
                import glob
                files = glob.glob(pattern)
                if files:
                    print(f"   Found: {files}")
                else:
                    print(f"   Missing: {pattern}")
        
    except Exception as e:
        print(f"   Error initializing NIXL agent: {e}")
        print("   This suggests NIXL may not be properly installed")

if __name__ == "__main__":
    check_nixl_installation() 