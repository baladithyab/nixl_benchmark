#!/usr/bin/env python3
"""
Simple UCX Loopback Benchmark
Tests local (same-agent) transfers with UCX backend.
Based on working patterns from nixl examples.
"""

import argparse
import time
import torch
from nixl._api import nixl_agent, nixl_agent_config


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
    # Check units from longest to shortest to avoid matching 'B' in 'KB'
    units = [('GB', 1024**3), ('MB', 1024**2), ('KB', 1024), ('B', 1)]
    for unit, multiplier in units:
        if size_str.endswith(unit):
            return int(float(size_str[:-len(unit)]) * multiplier)
    return int(size_str)


def run_loopback_test(buffer_size, iterations, warmup, use_cuda=False, gpu_id=0):
    """Run loopback transfer test for a specific buffer size."""
    
    # Set device
    if use_cuda:
        device = f"cuda:{gpu_id}"
        torch.set_default_device(device)
    else:
        device = "cpu"
        torch.set_default_device(device)
    
    # Create two agents for loopback (NIXL doesn't support single agent loopback)
    config = nixl_agent_config(backends=["UCX"])
    agent_target = nixl_agent("target", config)
    agent_initiator = nixl_agent("initiator", config)

    # Exchange metadata (initiator needs target's metadata)
    meta_target = agent_target.get_agent_metadata()
    remote_name = agent_initiator.add_remote_agent(meta_target)

    # Allocate buffers (use float32 for simplicity)
    num_elements = buffer_size // 4  # 4 bytes per float32
    target_tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
    initiator_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)

    # Register memory with respective agents
    target_reg = agent_target.register_memory([target_tensor])
    initiator_reg = agent_initiator.register_memory([initiator_tensor])

    if not target_reg or not initiator_reg:
        print("Memory registration failed")
        return None

    # Get transfer descriptors
    target_descs = target_reg.trim()
    initiator_descs = initiator_reg.trim()

    # Create transfer handle on initiator agent (READ from target)
    # Note: We use the descriptors directly, not serialized/deserialized
    xfer_handle = agent_initiator.initialize_xfer(
        "READ",
        initiator_descs,
        target_descs,
        remote_name,  # Remote agent name
        b"loopback"
    )
    
    if not xfer_handle:
        print("Failed to create transfer handle")
        return None
    
    # Warmup iterations
    for _ in range(warmup):
        state = agent_initiator.transfer(xfer_handle)
        if state == "ERR":
            print("Warmup transfer failed")
            return None

        while True:
            state = agent_initiator.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Warmup transfer error")
                return None

    # Measurement iterations
    latencies = []
    start_time = time.perf_counter()

    for _ in range(iterations):
        iter_start = time.perf_counter()

        state = agent_initiator.transfer(xfer_handle)
        if state == "ERR":
            print("Transfer failed")
            return None

        while True:
            state = agent_initiator.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Transfer error")
                return None

        iter_end = time.perf_counter()
        latencies.append((iter_end - iter_start) * 1e6)  # microseconds
    
    total_time = time.perf_counter() - start_time
    
    # Verify data
    if not torch.allclose(dst_tensor, src_tensor):
        print("Data verification failed!")
        return None
    
    # Cleanup
    agent_initiator.release_xfer_handle(xfer_handle)
    agent_initiator.deregister_memory(initiator_reg)
    agent_target.deregister_memory(target_reg)
    del agent
    
    # Calculate metrics
    total_bytes = buffer_size * iterations
    bandwidth_gbps = (total_bytes / total_time) / 1e9
    mean_latency = sum(latencies) / len(latencies)
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies) // 2]
    p95 = latencies_sorted[int(len(latencies) * 0.95)]
    p99 = latencies_sorted[int(len(latencies) * 0.99)]
    
    return {
        'buffer_size': buffer_size,
        'iterations': iterations,
        'total_time': total_time,
        'bandwidth_gbps': bandwidth_gbps,
        'mean_latency_us': mean_latency,
        'p50_latency_us': p50,
        'p95_latency_us': p95,
        'p99_latency_us': p99,
    }


def main():
    parser = argparse.ArgumentParser(description='Simple UCX Loopback Benchmark')
    parser.add_argument('--buffer_sizes', type=str, default='4KB,64KB,1MB,16MB',
                       help='Comma-separated buffer sizes (e.g., 4KB,1MB,16MB)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per buffer size')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA (GPU) memory instead of CPU')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Parse buffer sizes
    buffer_sizes = [parse_size(s.strip()) for s in args.buffer_sizes.split(',')]
    
    print("="*80)
    print("UCX Loopback Benchmark")
    print("="*80)
    print(f"Memory: {'CUDA (GPU)' if args.cuda else 'CPU (DRAM)'}")
    print(f"Buffer sizes: {', '.join(format_size(s) for s in buffer_sizes)}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print("="*80)
    
    results = []
    for buffer_size in buffer_sizes:
        print(f"\nTesting {format_size(buffer_size)}...", end=' ', flush=True)
        
        result = run_loopback_test(
            buffer_size,
            args.iterations,
            args.warmup,
            args.cuda,
            args.gpu_id
        )
        
        if result:
            results.append(result)
            print(f"✓ {result['bandwidth_gbps']:.2f} GB/s, "
                  f"{result['mean_latency_us']:.1f} μs (mean)")
        else:
            print("✗ Failed")
    
    # Print summary table
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


if __name__ == "__main__":
    main()

