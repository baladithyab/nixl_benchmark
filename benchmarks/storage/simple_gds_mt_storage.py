#!/usr/bin/env python3
"""
Simple GDS_MT (Multi-Threaded) Storage Benchmark
Tests GPU-to-Storage transfers using multi-threaded GDS backend for better performance.
"""

import argparse
import os
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
    # Check from longest to shortest to avoid matching 'B' in 'KB'
    units = [('GB', 1024**3), ('MB', 1024**2), ('KB', 1024), ('B', 1)]
    for unit, multiplier in units:
        if size_str.endswith(unit):
            return int(float(size_str[:-len(unit)]) * multiplier)
    return int(size_str)


def run_gds_mt_test(agent, agent_name, file_path, buffer_size, iterations, warmup, use_cuda=True):
    """Run GDS_MT read/write test for a specific buffer size."""
    
    # Allocate GPU buffers
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_elements = buffer_size // 4  # float32
        write_tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
        read_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
        mem_type = "VRAM"
    else:
        device = torch.device("cpu")
        num_elements = buffer_size // 4
        write_tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
        read_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
        mem_type = "DRAM"
    
    # Register GPU memory with NIXL
    write_reg = agent.register_memory([write_tensor])
    read_reg = agent.register_memory([read_tensor])
    
    if not write_reg or not read_reg:
        print("Failed to register GPU memory")
        return None
    
    write_xfer = write_reg.trim()
    read_xfer = read_reg.trim()
    
    # Open file for storage
    fd = os.open(file_path, os.O_RDWR | os.O_CREAT, 0o644)
    if fd < 0:
        print(f"Failed to open file: {file_path}")
        return None
    
    # Register file with NIXL
    file_reg_descs = agent.get_reg_descs([(0, buffer_size, fd, "file")], "FILE")
    file_reg = agent.register_memory(file_reg_descs)
    if not file_reg:
        print("Failed to register file")
        os.close(fd)
        return None
    
    file_xfer = file_reg.trim()
    
    # Create transfer handles (GPU <-> Storage)
    # For local storage transfers, use the agent's own name
    # GDS_MT doesn't support notifications, so use empty notif_msg
    write_handle = agent.initialize_xfer("WRITE", write_xfer, file_xfer, agent_name)
    read_handle = agent.initialize_xfer("READ", read_xfer, file_xfer, agent_name)
    
    if not write_handle or not read_handle:
        print("Failed to create transfer handles")
        agent.deregister_memory(file_reg)
        os.close(fd)
        return None
    
    # Warmup
    for _ in range(warmup):
        state = agent.transfer(write_handle)
        if state == "ERR":
            print("Warmup write failed")
            break
        
        while True:
            state = agent.check_xfer_state(write_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Warmup write error")
                break
        
        state = agent.transfer(read_handle)
        if state == "ERR":
            print("Warmup read failed")
            break
        
        while True:
            state = agent.check_xfer_state(read_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Warmup read error")
                break
    
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
    agent.deregister_memory(file_reg)
    os.close(fd)
    
    # Calculate statistics
    import statistics
    
    write_mean = statistics.mean(write_latencies)
    write_p50 = statistics.median(write_latencies)
    write_p99 = sorted(write_latencies)[int(len(write_latencies) * 0.99)]
    write_bandwidth = (buffer_size * iterations) / write_total_time / (1024**3)  # GB/s
    
    read_mean = statistics.mean(read_latencies)
    read_p50 = statistics.median(read_latencies)
    read_p99 = sorted(read_latencies)[int(len(read_latencies) * 0.99)]
    read_bandwidth = (buffer_size * iterations) / read_total_time / (1024**3)  # GB/s
    
    return {
        'buffer_size': buffer_size,
        'write_mean_us': write_mean,
        'write_p50_us': write_p50,
        'write_p99_us': write_p99,
        'write_bandwidth_gbps': write_bandwidth,
        'read_mean_us': read_mean,
        'read_p50_us': read_p50,
        'read_p99_us': read_p99,
        'read_bandwidth_gbps': read_bandwidth,
    }


def main():
    parser = argparse.ArgumentParser(description='Simple GDS_MT Storage Benchmark')
    parser.add_argument('--file', type=str, required=True,
                       help='File path for testing')
    parser.add_argument('--buffer_sizes', type=str, default='1MB,16MB,64MB',
                       help='Comma-separated buffer sizes')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations per buffer size')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA (GPU) memory instead of CPU')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Set GPU device if using CUDA
    if args.cuda:
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # Parse buffer sizes
    buffer_sizes = [parse_size(s.strip()) for s in args.buffer_sizes.split(',')]
    
    # Create agent with GDS_MT backend
    agent_config = nixl_agent_config(backends=[])
    agent = nixl_agent("gds_mt_test", agent_config)
    
    # Check for GDS_MT plugin
    plugins = agent.get_plugin_list()
    if "GDS_MT" not in plugins:
        print("ERROR: GDS_MT plugin not available")
        print(f"Available plugins: {plugins}")
        return
    
    # Create GDS_MT backend
    agent.create_backend("GDS_MT")
    
    print("="*80)
    print("GDS_MT (Multi-Threaded) Storage Benchmark")
    print("="*80)
    print(f"Memory: {'GPU (VRAM)' if args.cuda else 'CPU (DRAM)'}")
    print(f"File: {args.file}")
    print(f"Buffer sizes: {', '.join(format_size(s) for s in buffer_sizes)}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print("="*80)
    
    results = []
    for buffer_size in buffer_sizes:
        print(f"\nTesting {format_size(buffer_size)}...", flush=True)
        
        result = run_gds_mt_test(agent, "gds_mt_test", args.file, buffer_size, args.iterations, args.warmup, args.cuda)
        
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

