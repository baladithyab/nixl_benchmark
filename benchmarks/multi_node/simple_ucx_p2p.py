#!/usr/bin/env python3
"""
Simple UCX Point-to-Point Benchmark
Tests inter-node transfers with UCX backend.
Based on working patterns from nixl blocking_send_recv_example.py
"""

import argparse
import time
import torch
import socket
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


def check_tcp_connectivity(ip, port, timeout=5):
    """Check if we can connect to the target IP:port via TCP."""
    try:
        logger.info(f"Checking TCP connectivity to {ip}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        if result == 0:
            logger.info(f"✓ TCP connection to {ip}:{port} successful")
            return True
        else:
            logger.error(f"✗ TCP connection to {ip}:{port} failed (error code: {result})")
            return False
    except socket.timeout:
        logger.error(f"✗ TCP connection to {ip}:{port} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"✗ TCP connection to {ip}:{port} failed: {e}")
        return False


def format_size(bytes_val):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def parse_size(size_str):
    """Parse size string like '4KB', '1MB' to bytes."""
    size_str = size_str.upper().strip()
    # Check units in order from longest to shortest to avoid partial matches
    units = [('GB', 1024**3), ('MB', 1024**2), ('KB', 1024), ('B', 1)]
    for unit, multiplier in units:
        if size_str.endswith(unit):
            num_str = size_str[:-len(unit)].strip()
            return int(float(num_str) * multiplier)
    return int(size_str)


def run_target(ip, port, buffer_sizes, use_cuda, gpu_id):
    """Run as target (receiver)."""
    
    # Set device
    if use_cuda:
        device = f"cuda:{gpu_id}"
        torch.set_default_device(device)
    else:
        device = "cpu"
        torch.set_default_device(device)
    
    # Create agent with listening enabled
    config = nixl_agent_config(enable_listen_thread=True, backends=["UCX"], listen_port=port)
    agent = nixl_agent("target", config)
    
    logger.info(f"Target listening on {ip}:{port}")
    logger.info(f"Waiting for initiator to connect...")
    logger.info(f"Target metadata available at {ip}:{port}")

    # The listen thread will automatically respond to fetch_remote_metadata calls
    # We need to wait for the initiator to send us their metadata
    # Check for new notifications which indicate the initiator has connected
    timeout_seconds = 60
    start_time = time.time()
    connected = False

    while not connected:
        if time.time() - start_time > timeout_seconds:
            logger.error(f"Timeout waiting for initiator after {timeout_seconds}s")
            logger.error("Initiator may not be connecting properly")
            return

        # Update and check for notifications from initiator
        agent.update_notifs()
        notifs = agent.get_new_notifs()

        # Check if we have metadata from initiator
        if agent.check_remote_metadata("initiator"):
            connected = True
            logger.info("✓ Connected to initiator")
            break

        time.sleep(0.1)

    if not connected:
        logger.error("Failed to connect to initiator")
        return
    
    # Process each buffer size
    for buffer_size in buffer_sizes:
        num_elements = buffer_size // 4
        tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
        
        # Register memory
        reg_descs = agent.register_memory([tensor])
        if not reg_descs:
            logger.error("Memory registration failed")
            return
        
        target_descs = reg_descs.trim()
        target_desc_str = agent.get_serialized_descs(target_descs)
        
        # Send descriptor to initiator
        agent.send_notif("initiator", target_desc_str)
        
        # Wait for transfer completion
        transfer_id = f"transfer_{buffer_size}".encode()
        while not agent.check_remote_xfer_done("initiator", transfer_id):
            time.sleep(0.001)
        
        # Verify data (should be all ones from initiator)
        if torch.allclose(tensor, torch.ones_like(tensor)):
            logger.info(f"✓ Received {format_size(buffer_size)} - data verified")
        else:
            logger.error(f"✗ Data verification failed for {format_size(buffer_size)}")
        
        # Cleanup
        agent.deregister_memory(reg_descs)
        del tensor
    
    logger.info("Target completed all transfers")


def run_initiator(target_ip, port, buffer_sizes, iterations, warmup, use_cuda, gpu_id):
    """Run as initiator (sender)."""
    
    # Set device
    if use_cuda:
        device = f"cuda:{gpu_id}"
        torch.set_default_device(device)
    else:
        device = "cpu"
        torch.set_default_device(device)
    
    # Create agent
    config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent("initiator", config)

    logger.info(f"Connecting to target at {target_ip}:{port}")

    # First check basic TCP connectivity
    if not check_tcp_connectivity(target_ip, port, timeout=5):
        logger.error(f"Cannot reach target at {target_ip}:{port}")
        logger.error("Please ensure:")
        logger.error("  1. Target is running and listening on the correct port")
        logger.error("  2. Firewall allows connections on this port")
        logger.error("  3. IP address is correct")
        return

    # Fetch target metadata and send local metadata
    logger.info("Fetching target metadata...")
    try:
        agent.fetch_remote_metadata("target", target_ip, port)
        agent.send_local_metadata(target_ip, port)
    except Exception as e:
        logger.error(f"Metadata exchange failed: {e}")
        return

    # Wait for metadata exchange with timeout
    logger.info("Waiting for metadata exchange...")
    timeout_seconds = 30
    start_time = time.time()
    while not agent.check_remote_metadata("target"):
        if time.time() - start_time > timeout_seconds:
            logger.error(f"Metadata exchange timed out after {timeout_seconds}s")
            logger.error("Target may not be responding properly")
            return
        time.sleep(0.1)

    logger.info("✓ Connected to target")
    
    print("\n" + "="*80)
    print("UCX Point-to-Point Benchmark (Initiator)")
    print("="*80)
    print(f"Memory: {'CUDA (GPU)' if use_cuda else 'CPU (DRAM)'}")
    print(f"Target: {target_ip}:{port}")
    print(f"Iterations: {iterations} (warmup: {warmup})")
    print("="*80)
    
    results = []
    
    for buffer_size in buffer_sizes:
        print(f"\nTesting {format_size(buffer_size)}...", flush=True)
        
        num_elements = buffer_size // 4
        tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
        
        # Register memory
        reg_descs = agent.register_memory([tensor])
        if not reg_descs:
            logger.error("Memory registration failed")
            continue
        
        initiator_descs = reg_descs.trim()
        
        # Wait for target descriptor
        notifs = []
        while len(notifs) == 0:
            notifs = agent.get_new_notifs()
            if "target" in notifs and len(notifs["target"]) > 0:
                break
            time.sleep(0.001)
        
        target_descs = agent.deserialize_descs(notifs["target"][0])
        
        # Create transfer handle
        transfer_id = f"transfer_{buffer_size}".encode()
        xfer_handle = agent.initialize_xfer(
            "WRITE",
            initiator_descs,
            target_descs,
            "target",
            transfer_id
        )
        
        if not xfer_handle:
            logger.error("Failed to create transfer handle")
            continue
        
        # Warmup
        for _ in range(warmup):
            state = agent.transfer(xfer_handle)
            if state == "ERR":
                logger.error("Warmup transfer failed")
                break
            
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "DONE":
                    break
                elif state == "ERR":
                    logger.error("Warmup transfer error")
                    break
        
        # Measurement
        latencies = []
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            iter_start = time.perf_counter()
            
            state = agent.transfer(xfer_handle)
            if state == "ERR":
                logger.error("Transfer failed")
                break
            
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "DONE":
                    break
                elif state == "ERR":
                    logger.error("Transfer error")
                    break
            
            iter_end = time.perf_counter()
            latencies.append((iter_end - iter_start) * 1e6)  # microseconds
        
        total_time = time.perf_counter() - start_time
        
        # Calculate metrics
        if latencies:
            total_bytes = buffer_size * iterations
            bandwidth_gbps = (total_bytes / total_time) / 1e9
            mean_latency = sum(latencies) / len(latencies)
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies) // 2]
            p95 = latencies_sorted[int(len(latencies) * 0.95)]
            p99 = latencies_sorted[int(len(latencies) * 0.99)]
            
            result = {
                'buffer_size': buffer_size,
                'bandwidth_gbps': bandwidth_gbps,
                'mean_latency_us': mean_latency,
                'p50_latency_us': p50,
                'p95_latency_us': p95,
                'p99_latency_us': p99,
            }
            results.append(result)
            
            print(f"  ✓ {bandwidth_gbps:.2f} GB/s, {mean_latency:.1f} μs (mean)")
        
        # Cleanup
        agent.release_xfer_handle(xfer_handle)
        agent.deregister_memory(reg_descs)
        del tensor
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"{'Size':<10} | {'Bandwidth':<12} | {'Mean (μs)':<10} | "
              f"{'P50 (μs)':<10} | {'P95 (μs)':<10} | {'P99 (μs)':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{format_size(r['buffer_size']):<10} | "
                  f"{r['bandwidth_gbps']:>10.2f} GB/s | "
                  f"{r['mean_latency_us']:>10.1f} | "
                  f"{r['p50_latency_us']:>10.1f} | "
                  f"{r['p95_latency_us']:>10.1f} | "
                  f"{r['p99_latency_us']:>10.1f}")
        print("="*80)
    
    # Cleanup
    agent.remove_remote_agent("target")
    agent.invalidate_local_metadata(target_ip, port)


def main():
    parser = argparse.ArgumentParser(description='Simple UCX Point-to-Point Benchmark')
    parser.add_argument('--mode', type=str, required=True, choices=['target', 'initiator'],
                       help='Run as target (receiver) or initiator (sender)')
    parser.add_argument('--ip', type=str, required=True,
                       help='IP address (local IP for target, target IP for initiator)')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port number')
    parser.add_argument('--buffer_sizes', type=str, default='1MB,16MB,256MB',
                       help='Comma-separated buffer sizes (initiator only)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per buffer size (initiator only)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (initiator only)')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA (GPU) memory instead of CPU')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Parse buffer sizes
    buffer_sizes = [parse_size(s.strip()) for s in args.buffer_sizes.split(',')]
    
    if args.mode == 'target':
        run_target(args.ip, args.port, buffer_sizes, args.cuda, args.gpu_id)
    else:
        run_initiator(args.ip, args.port, buffer_sizes, args.iterations, 
                     args.warmup, args.cuda, args.gpu_id)


if __name__ == "__main__":
    main()

