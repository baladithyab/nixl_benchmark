#!/usr/bin/env python3
"""
Quick verification script to check NIXL installation and setup.
Run this before using the benchmarks.
"""

import sys
import os


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_result(test_name, passed, message=""):
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {test_name}")
    if message:
        print(f"  → {message}")


def test_python_version():
    """Check Python version."""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 7
    print_result(
        "Python version",
        passed,
        f"Python {version.major}.{version.minor}.{version.micro}" +
        ("" if passed else " (need Python 3.7+)")
    )
    return passed


def test_imports():
    """Test required imports."""
    all_passed = True
    
    # Test torch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        print_result(
            "PyTorch",
            True,
            f"CUDA: {'Yes' if cuda_available else 'No'}" +
            (f", {cuda_count} GPU(s)" if cuda_available else "")
        )
    except ImportError:
        print_result("PyTorch", False, "pip install torch")
        all_passed = False
    
    # Test NIXL
    try:
        from nixl._api import nixl_agent, nixl_agent_config
        print_result("NIXL", True, "Imported successfully")
    except ImportError as e:
        print_result("NIXL", False, f"pip install nixl (Error: {e})")
        all_passed = False
    
    return all_passed


def test_environment():
    """Check environment variables."""
    all_passed = True
    
    plugin_dir = os.environ.get("NIXL_PLUGIN_DIR")
    if plugin_dir:
        exists = os.path.exists(plugin_dir)
        print_result(
            "NIXL_PLUGIN_DIR",
            exists,
            f"{plugin_dir}" + ("" if exists else " (directory not found)")
        )
        all_passed &= exists
    else:
        print_result(
            "NIXL_PLUGIN_DIR",
            False,
            "Not set. Set with: export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins"
        )
        all_passed = False
    
    ld_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_path:
        print_result("LD_LIBRARY_PATH", True, f"Set ({len(ld_path)} chars)")
    else:
        print_result(
            "LD_LIBRARY_PATH",
            False,
            "Not set. May need: export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
        )
        all_passed = False
    
    return all_passed


def test_nixl_agent():
    """Test creating a NIXL agent."""
    try:
        from nixl._api import nixl_agent, nixl_agent_config
        
        config = nixl_agent_config(backends=["UCX"])
        agent = nixl_agent("test_agent", config)
        
        plugins = agent.get_plugin_list()
        print_result("NIXL Agent Creation", True, f"Available plugins: {plugins}")
        
        # Check specific plugins
        has_ucx = "UCX" in plugins
        has_gds = "GDS" in plugins
        
        print_result("  UCX Plugin", has_ucx, "Required for network transfers")
        print_result("  GDS Plugin", has_gds, "Optional, for GPU-to-storage" if not has_gds else "For GPU-to-storage")
        
        del agent
        return True
        
    except Exception as e:
        print_result("NIXL Agent Creation", False, f"Error: {e}")
        return False


def test_simple_transfer():
    """Test a simple loopback transfer."""
    try:
        import torch
        from nixl._api import nixl_agent, nixl_agent_config
        
        # Create agent
        config = nixl_agent_config(backends=["UCX"])
        agent = nixl_agent("test_transfer", config)
        
        # Add self as remote
        meta = agent.get_agent_metadata()
        agent.add_remote_agent(meta)
        
        # Allocate small buffers
        src = torch.ones(64, dtype=torch.float32)
        dst = torch.zeros(64, dtype=torch.float32)
        
        # Register memory
        src_reg = agent.register_memory([src])
        dst_reg = agent.register_memory([dst])
        
        # Create transfer
        xfer_handle = agent.initialize_xfer(
            "WRITE",
            src_reg.trim(),
            dst_reg.trim(),
            "test_transfer",
            b"test"
        )
        
        # Execute
        state = agent.transfer(xfer_handle)
        
        # Wait for completion
        import time
        timeout = 5
        start = time.time()
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                raise RuntimeError("Transfer failed")
            if time.time() - start > timeout:
                raise RuntimeError("Transfer timeout")
            time.sleep(0.001)
        
        # Verify
        verified = torch.allclose(dst, src)
        
        # Cleanup
        agent.release_xfer_handle(xfer_handle)
        agent.deregister_memory(src_reg)
        agent.deregister_memory(dst_reg)
        del agent
        
        print_result(
            "Simple Transfer Test",
            verified,
            "256 bytes transferred and verified" if verified else "Data verification failed"
        )
        return verified
        
    except Exception as e:
        print_result("Simple Transfer Test", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps(all_passed):
    """Print next steps based on results."""
    print_header("Next Steps")
    
    if all_passed:
        print("✅ All tests passed! Your NIXL setup is ready.")
        print("\nYou can now run the benchmarks:")
        print("\n  1. Single-node test:")
        print("     python3 benchmarks/simple_ucx_loopback.py --buffer_sizes 4KB,1MB")
        print("\n  2. Multi-node test (requires 2 nodes):")
        print("     Node 1: python3 benchmarks/simple_ucx_p2p.py --mode target --ip <node1_ip>")
        print("     Node 2: python3 benchmarks/simple_ucx_p2p.py --mode initiator --ip <node1_ip>")
        print("\n  3. Storage test (requires GDS):")
        print("     python3 benchmarks/simple_gds_storage.py --file /tmp/test.dat")
        print("\nSee benchmarks/README.md for more details.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("\n  1. Install NIXL:")
        print("     pip install nixl")
        print("\n  2. Set environment variables:")
        print("     export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins")
        print("     export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
        print("\n  3. Check NIXL installation:")
        print("     See README.md for build instructions")


def main():
    print_header("NIXL Setup Verification")
    print("This script checks if NIXL is properly installed and configured.")
    
    results = []
    
    print_header("1. Python Environment")
    results.append(test_python_version())
    results.append(test_imports())
    
    print_header("2. Environment Variables")
    results.append(test_environment())
    
    print_header("3. NIXL Functionality")
    results.append(test_nixl_agent())
    results.append(test_simple_transfer())
    
    all_passed = all(results)
    
    print_next_steps(all_passed)
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ VERIFICATION PASSED")
    else:
        print("  ❌ VERIFICATION FAILED")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

