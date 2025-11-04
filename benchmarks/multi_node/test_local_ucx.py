#!/usr/bin/env python3
"""
Test UCX transfers between two local agents (same process).
This helps isolate whether the issue is with UCX transfers in general
or specifically with multi-node communication.
"""

import os
import sys
import time

# Set UCX environment variables BEFORE importing nixl
os.environ.setdefault('UCX_TLS', 'tcp')
os.environ.setdefault('UCX_TCP_PREFER_DEFAULT', 'n')

import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)

def test_local_ucx_transfer():
    """Test UCX transfer between two local agents."""
    
    print("=" * 80)
    print("Local UCX Transfer Test")
    print("=" * 80)
    
    # Create two agents - target and initiator
    print("\n1. Creating agents...")
    
    # Target agent
    target_config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        backends=['UCX'],
        listen_port=5557
    )
    target_agent = nixl_agent("target", target_config)
    print("✓ Target agent created")
    
    # Initiator agent
    initiator_config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        backends=['UCX'],
        listen_port=0
    )
    initiator_agent = nixl_agent("initiator", initiator_config)
    print("✓ Initiator agent created")
    
    # Create tensors
    print("\n2. Creating tensors...")
    torch.set_default_device("cpu")
    
    target_tensor = torch.ones(1024 * 256, dtype=torch.float32)  # 1 MB
    initiator_tensor = torch.zeros(1024 * 256, dtype=torch.float32)  # 1 MB
    
    print(f"✓ Target tensor: {target_tensor.shape}, sum={target_tensor.sum()}")
    print(f"✓ Initiator tensor: {initiator_tensor.shape}, sum={initiator_tensor.sum()}")
    
    # Register memory
    print("\n3. Registering memory...")
    
    target_reg = target_agent.register_memory([target_tensor])
    if not target_reg:
        print("✗ Target memory registration failed")
        return False
    print("✓ Target memory registered")
    
    initiator_reg = initiator_agent.register_memory([initiator_tensor])
    if not initiator_reg:
        print("✗ Initiator memory registration failed")
        return False
    print("✓ Initiator memory registered")
    
    # Exchange metadata
    print("\n4. Exchanging metadata...")
    
    # Initiator fetches target metadata
    initiator_agent.fetch_remote_metadata("target", "127.0.0.1", 5557)
    initiator_agent.send_local_metadata("127.0.0.1", 5557)
    
    # Wait for metadata exchange
    timeout = 10
    start_time = time.time()
    while not initiator_agent.check_remote_metadata("target"):
        time.sleep(0.1)
        if time.time() - start_time > timeout:
            print("✗ Metadata exchange timed out")
            return False
    
    # Wait for target to receive initiator metadata
    while not target_agent.check_remote_metadata("initiator"):
        time.sleep(0.1)
        if time.time() - start_time > timeout:
            print("✗ Target didn't receive initiator metadata")
            return False
    
    print("✓ Metadata exchange complete")
    
    # Get descriptors
    print("\n5. Preparing transfer descriptors...")
    
    target_descs = target_reg.trim()
    initiator_descs = initiator_reg.trim()
    
    # Target sends its descriptor to initiator
    target_desc_str = target_agent.get_serialized_descs(target_descs)
    target_agent.send_notif("initiator", target_desc_str)
    
    # Initiator waits for target descriptor
    notifs = []
    start_time = time.time()
    while len(notifs) == 0:
        notifs = initiator_agent.get_new_notifs()
        if "target" in notifs and len(notifs["target"]) > 0:
            break
        time.sleep(0.001)
        if time.time() - start_time > timeout:
            print("✗ Didn't receive target descriptor")
            return False
    
    remote_target_descs = initiator_agent.deserialize_descs(notifs["target"][0])
    print("✓ Descriptors exchanged")
    
    # Initialize transfer
    print("\n6. Initializing transfer...")
    print(f"   Initiator descs type: {type(initiator_descs)}")
    print(f"   Target descs type: {type(remote_target_descs)}")
    
    try:
        transfer_id = b"test_transfer"
        xfer_handle = initiator_agent.initialize_xfer(
            "READ",  # Initiator reads from target
            initiator_descs,
            remote_target_descs,
            "target",
            transfer_id,
            backends=["UCX"]
        )
        print("✓ Transfer initialized")
    except Exception as e:
        print(f"✗ Transfer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Execute transfer
    print("\n7. Executing transfer...")
    
    try:
        initiator_agent.transfer(xfer_handle)
        print("✓ Transfer started")
    except Exception as e:
        print(f"✗ Transfer failed: {e}")
        return False
    
    # Wait for transfer to complete
    timeout = 10
    start_time = time.time()
    while True:
        state = initiator_agent.check_xfer_state(xfer_handle)
        if state == "DONE":
            print("✓ Transfer complete")
            break
        elif state == "ERROR":
            print("✗ Transfer error")
            return False
        
        time.sleep(0.001)
        if time.time() - start_time > timeout:
            print(f"✗ Transfer timed out (state: {state})")
            return False
    
    # Verify data
    print("\n8. Verifying data...")
    print(f"   Initiator tensor sum: {initiator_tensor.sum()} (expected: {target_tensor.sum()})")
    
    if torch.allclose(initiator_tensor, target_tensor):
        print("✓ Data verification passed!")
        return True
    else:
        print("✗ Data verification failed!")
        return False

if __name__ == "__main__":
    success = test_local_ucx_transfer()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ LOCAL UCX TRANSFER TEST PASSED!")
    else:
        print("❌ LOCAL UCX TRANSFER TEST FAILED!")
    print("=" * 80)
    
    sys.exit(0 if success else 1)

