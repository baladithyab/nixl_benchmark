#!/usr/bin/env python3
"""
Simple GDS Storage Benchmark
Tests GPU-to-Storage transfers using GDS backend.
Based on working patterns from nixl_gds_example.py
"""

import argparse
import os
import time
import nixl._utils as nixl_utils
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
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            return int(float(size_str[:-len(unit)]) * multiplier)
    return int(size_str)


def run_gds_test(agent, file_path, buffer_size, iterations, warmup):
    """Run GDS read/write test for a specific buffer size."""
    
    # Allocate buffers
    write_addr = nixl_utils.malloc_passthru(buffer_size)
    read_addr = nixl_utils.malloc_passthru(buffer_size)
    
    # Initialize write buffer with test pattern
    nixl_utils.ba_buf(write_addr, buffer_size)
    
    # Register memory
    write_strings = [(write_addr, buffer_size, 0, "write")]
    read_strings = [(read_addr, buffer_size, 0, "read")]
    
    write_reg = agent.get_reg_descs(write_strings, "DRAM")
    read_reg = agent.get_reg_descs(read_strings, "DRAM")
    
    agent.register_memory(write_reg)
    agent.register_memory(read_reg)
    
    write_xfer = agent.get_xfer_descs([(write_addr, buffer_size, 0)], "DRAM")
    read_xfer = agent.get_xfer_descs([(read_addr, buffer_size, 0)], "DRAM")
    
    # Open file
    fd = os.open(file_path, os.O_RDWR | os.O_CREAT, 0o644)
    if fd < 0:
        print(f"Failed to open file: {file_path}")
        return None
    
    # Register file
    file_list = [(0, buffer_size, fd, "file")]
    file_descs = agent.register_memory(file_list, "FILE")
    if not file_descs:
        print("Failed to register file")
        os.close(fd)
        return None
    
    file_xfer = file_descs.trim()
    
    # Create transfer handles
    write_handle = agent.initialize_xfer("WRITE", write_xfer, file_xfer, "gds_test")
    read_handle = agent.initialize_xfer("READ", read_xfer, file_xfer, "gds_test")
    
    if not write_handle or not read_handle:
        print("Failed to create transfer handles")
        os.close(fd)
        return None
    
    # Warmup - WRITE
    for _ in range(warmup):
        state = agent.transfer(write_handle)
        if state == "ERR":
            print("Warmup write failed")
            os.close(fd)
            return None
        
        while True:
            state = agent.check_xfer_state(write_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Warmup write error")
                os.close(fd)
                return None
    
    # Warmup - READ
    for _ in range(warmup):
        state = agent.transfer(read_handle)
        if state == "ERR":
            print("Warmup read failed")
            os.close(fd)
            return None
        
        while True:
            state = agent.check_xfer_state(read_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Warmup read error")
                os.close(fd)
                return None
    
    # Measure WRITE
    write_latencies = []
    write_start = time.perf_counter()
    
    for _ in range(iterations):
        iter_start = time.perf_counter()
        
        state = agent.transfer(write_handle)
        if state == "ERR":
            print("Write transfer failed")
            break
        
        while True:
            state = agent.check_xfer_state(write_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Write transfer error")
                break
        
        iter_end = time.perf_counter()
        write_latencies.append((iter_end - iter_start) * 1e6)  # microseconds
    
    write_total_time = time.perf_counter() - write_start
    
    # Measure READ
    read_latencies = []
    read_start = time.perf_counter()
    
    for _ in range(iterations):
        iter_start = time.perf_counter()
        
        state = agent.transfer(read_handle)
        if state == "ERR":
            print("Read transfer failed")
            break
        
        while True:
            state = agent.check_xfer_state(read_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Read transfer error")
                break
        
        iter_end = time.perf_counter()
        read_latencies.append((iter_end - iter_start) * 1e6)  # microseconds
    
    read_total_time = time.perf_counter() - read_start
    
    # Cleanup
    agent.release_xfer_handle(write_handle)
    agent.release_xfer_handle(read_handle)
    agent.deregister_memory(write_reg)
    agent.deregister_memory(read_reg)
    agent.deregister_memory(file_descs)
    os.close(fd)
    
    nixl_utils.free_passthru(write_addr)
    nixl_utils.free_passthru(read_addr)
    
    # Calculate metrics
    if not write_latencies or not read_latencies:
        return None
    
    total_bytes = buffer_size * iterations
    
    write_bandwidth = (total_bytes / write_total_time) / 1e9
    write_mean = sum(write_latencies) / len(write_latencies)
    write_sorted = sorted(write_latencies)
    write_p50 = write_sorted[len(write_sorted) // 2]
    write_p99 = write_sorted[int(len(write_sorted) * 0.99)]
    
    read_bandwidth = (total_bytes / read_total_time) / 1e9
    read_mean = sum(read_latencies) / len(read_latencies)
    read_sorted = sorted(read_latencies)
    read_p50 = read_sorted[len(read_sorted) // 2]
    read_p99 = read_sorted[int(len(read_sorted) * 0.99)]
    
    return {
        'buffer_size': buffer_size,
        'write_bandwidth_gbps': write_bandwidth,
        'write_mean_us': write_mean,
        'write_p50_us': write_p50,
        'write_p99_us': write_p99,
        'read_bandwidth_gbps': read_bandwidth,
        'read_mean_us': read_mean,
        'read_p50_us': read_p50,
        'read_p99_us': read_p99,
    }


def main():
    parser = argparse.ArgumentParser(description='Simple GDS Storage Benchmark')
    parser.add_argument('--file', type=str, required=True,
                       help='File path for testing')
    parser.add_argument('--buffer_sizes', type=str, default='1MB,16MB,256MB',
                       help='Comma-separated buffer sizes')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations per buffer size')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    # Parse buffer sizes
    buffer_sizes = [parse_size(s.strip()) for s in args.buffer_sizes.split(',')]
    
    # Create agent with GDS backend
    agent_config = nixl_agent_config(backends=[])
    agent = nixl_agent("gds_test", agent_config)
    
    # Check for GDS plugin
    plugins = agent.get_plugin_list()
    if "GDS" not in plugins:
        print("ERROR: GDS plugin not available")
        print(f"Available plugins: {plugins}")
        return
    
    # Create GDS backend
    agent.create_backend("GDS")
    
    print("="*80)
    print("GDS Storage Benchmark")
    print("="*80)
    print(f"File: {args.file}")
    print(f"Buffer sizes: {', '.join(format_size(s) for s in buffer_sizes)}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print("="*80)
    
    results = []
    for buffer_size in buffer_sizes:
        print(f"\nTesting {format_size(buffer_size)}...", flush=True)
        
        result = run_gds_test(agent, args.file, buffer_size, args.iterations, args.warmup)
        
        if result:
            results.append(result)
            print(f"  WRITE: {result['write_bandwidth_gbps']:.2f} GB/s, "
                  f"{result['write_mean_us']:.1f} μs (mean)")
            print(f"  READ:  {result['read_bandwidth_gbps']:.2f} GB/s, "
                  f"{result['read_mean_us']:.1f} μs (mean)")
        else:
            print("  ✗ Failed")
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("WRITE RESULTS")
        print("="*80)
        print(f"{'Size':<10} | {'Bandwidth':<12} | {'Mean (μs)':<10} | "
              f"{'P50 (μs)':<10} | {'P99 (μs)':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{format_size(r['buffer_size']):<10} | "
                  f"{r['write_bandwidth_gbps']:>10.2f} GB/s | "
                  f"{r['write_mean_us']:>10.1f} | "
                  f"{r['write_p50_us']:>10.1f} | "
                  f"{r['write_p99_us']:>10.1f}")
        
        print("\n" + "="*80)
        print("READ RESULTS")
        print("="*80)
        print(f"{'Size':<10} | {'Bandwidth':<12} | {'Mean (μs)':<10} | "
              f"{'P50 (μs)':<10} | {'P99 (μs)':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{format_size(r['buffer_size']):<10} | "
                  f"{r['read_bandwidth_gbps']:>10.2f} GB/s | "
                  f"{r['read_mean_us']:>10.1f} | "
                  f"{r['read_p50_us']:>10.1f} | "
                  f"{r['read_p99_us']:>10.1f}")
        print("="*80)


if __name__ == "__main__":
    main()

