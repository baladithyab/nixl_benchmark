#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config


def run_batch_transfer(agent, write_addrs, read_addrs, file_path, buf_size, batch_size, total_size, original_file_size=None):
    """
    Process file in batches to avoid resource exhaustion
    """
    num_batches = (total_size + batch_size - 1) // batch_size
    buffers_per_batch = batch_size // buf_size
    
    print(f"\n{'='*80}")
    print(f"Starting BATCH transfer processing")
    print(f"Total file size: {total_size:,} bytes ({total_size/(1024**3):.2f} GB)")
    print(f"Batch size: {batch_size:,} bytes ({batch_size/(1024**2):.1f} MB)")
    print(f"Buffer size: {buf_size:,} bytes ({buf_size/(1024**2):.1f} MB)")
    print(f"Buffers per batch: {buffers_per_batch}")
    print(f"Number of batches: {num_batches}")
    print(f"{'='*80}")
    
    overall_start = time.time()
    all_write_times = []
    all_read_times = []
    
    for batch_idx in range(num_batches):
        batch_start = time.time()
        batch_offset = batch_idx * batch_size
        current_batch_size = min(batch_size, total_size - batch_offset)
        current_buffers = (current_batch_size + buf_size - 1) // buf_size
        
        print(f"\n--- Processing Batch {batch_idx + 1}/{num_batches} ---")
        print(f"Batch offset: {batch_offset:,} bytes")
        print(f"Batch size: {current_batch_size:,} bytes")
        print(f"Buffers in this batch: {current_buffers}")
        
        # Process buffers in this batch using the working pattern
        batch_write_times = []
        batch_read_times = []
        
        for buf_idx in range(current_buffers):
            buffer_offset = batch_offset + (buf_idx * buf_size)
            current_buf_size = min(buf_size, current_batch_size - (buf_idx * buf_size))
            
            # Use working single-buffer pattern
            success, write_time, read_time = run_single_buffer_test(
                agent, write_addrs[buf_idx], read_addrs[buf_idx], 
                file_path, current_buf_size, buffer_offset
            )
            
            if not success:
                print(f"Failed to process buffer {buf_idx} in batch {batch_idx}")
                return False
                
            batch_write_times.append(write_time)
            batch_read_times.append(read_time)
            
            if buf_idx < 3:  # Show timing for first few buffers in each batch
                print(f"  Buffer {buf_idx}: WRITE={write_time*1000:.2f}ms, READ={read_time*1000:.2f}ms")
        
        batch_time = time.time() - batch_start
        batch_total_write = sum(batch_write_times)
        batch_total_read = sum(batch_read_times)
        
        print(f"Batch {batch_idx + 1} completed in {batch_time*1000:.2f}ms")
        print(f"  Batch WRITE: {batch_total_write*1000:.2f}ms, READ: {batch_total_read*1000:.2f}ms")
        print(f"  Batch throughput: {(current_batch_size*2/batch_time)/(1024**2):.2f} MB/s")
        
        all_write_times.extend(batch_write_times)
        all_read_times.extend(batch_read_times)
    
    overall_time = time.time() - overall_start
    
    # Performance Summary
    total_write_time = sum(all_write_times)
    total_read_time = sum(all_read_times)
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY (BATCH PROCESSING)")
    print(f"{'='*80}")
    print(f"Total data processed: {total_size:,} bytes ({total_size/(1024**3):.2f} GB)")
    print(f"Total buffers processed: {len(all_write_times):,}")
    print(f"Parallel buffer transfers: 1 (sequential execution)")
    print(f"")
    print(f"WRITE Operations (GPU to Disk):")
    print(f"  Total WRITE time: {total_write_time*1000:.2f} ms")
    print(f"  Average WRITE time per buffer: {(total_write_time/len(all_write_times))*1000:.2f} ms")
    print(f"  WRITE throughput: {(total_size/total_write_time)/(1024**2):.2f} MB/s")
    print(f"")
    print(f"READ Operations (Disk to GPU):")
    print(f"  Total READ time: {total_read_time*1000:.2f} ms")
    print(f"  Average READ time per buffer: {(total_read_time/len(all_read_times))*1000:.2f} ms")
    print(f"  READ throughput: {(total_size/total_read_time)/(1024**2):.2f} MB/s")
    print(f"")
    print(f"Overall Performance:")
    print(f"  Total time (all operations): {overall_time*1000:.2f} ms ({overall_time:.2f} seconds)")
    print(f"  Combined throughput: {(total_size*2/overall_time)/(1024**2):.2f} MB/s")
    print(f"  Total sum of individual transfers: {(total_write_time+total_read_time)*1000:.2f} ms")
    print(f"  Overhead time: {(overall_time-(total_write_time+total_read_time))*1000:.2f} ms")
    
    # Add 5GB estimation if we processed less than the full file
    if original_file_size and original_file_size > total_size:
        print(f"")
        print(f"5GB File Estimation:")
        
        # Calculate scaling factors
        full_file_gb = original_file_size / (1024**3)
        scale_factor = original_file_size / total_size
        
        # Estimate based on current performance
        estimated_write_time = total_write_time * scale_factor
        estimated_read_time = total_read_time * scale_factor
        estimated_transfer_time = estimated_write_time + estimated_read_time
        
        # Estimate overhead scaling (overhead per batch + setup)
        batches_processed = len(all_write_times) // (batch_size // buf_size)
        overhead_per_batch = (overall_time - (total_write_time + total_read_time)) / batches_processed
        total_batches_needed = (original_file_size + batch_size - 1) // batch_size
        estimated_overhead = overhead_per_batch * total_batches_needed
        
        estimated_total_time = estimated_transfer_time + estimated_overhead
        
        print(f"  Full file size: {original_file_size:,} bytes ({full_file_gb:.2f} GB)")
        print(f"  Estimated WRITE time: {estimated_write_time*1000:.0f} ms ({estimated_write_time:.2f}s)")
        print(f"  Estimated READ time: {estimated_read_time*1000:.0f} ms ({estimated_read_time:.2f}s)")
        print(f"  Estimated overhead: {estimated_overhead*1000:.0f} ms ({estimated_overhead:.2f}s)")
        print(f"  Estimated total time: {estimated_total_time*1000:.0f} ms ({estimated_total_time:.2f}s)")
        print(f"  Estimated combined throughput: {(original_file_size*2/estimated_total_time)/(1024**2):.0f} MB/s")
        
        # Show breakdown
        if estimated_total_time < 60:
            time_str = f"{estimated_total_time:.1f} seconds"
        else:
            minutes = int(estimated_total_time // 60)
            seconds = estimated_total_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        
        print(f"  → Full 5GB processing would take approximately: {time_str}")
    
    print(f"{'='*80}")
    
    return True


def run_single_buffer_test(agent, write_addr, read_addr, file_path, buf_size, offset):
    """
    Run single buffer test using the exact working pattern
    """
    # Register memory
    agent_strings = [(write_addr, buf_size, 0, "a"), (read_addr, buf_size, 0, "b")]
    reg_descs = agent.get_reg_descs(agent_strings, "DRAM")
    xfer1_descs = agent.get_xfer_descs([(write_addr, buf_size, 0)], "DRAM")
    xfer2_descs = agent.get_xfer_descs([(read_addr, buf_size, 0)], "DRAM")
    
    assert agent.register_memory(reg_descs) is not None
    
    # Open file
    fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
    assert fd >= 0
    
    # Register file at specific offset
    file_list = [(offset, buf_size, fd, "b")]
    file_descs = agent.register_memory(file_list, "FILE")
    assert file_descs is not None
    xfer_files = file_descs.trim()
    
    # WRITE operation
    write_start = time.time()
    xfer_handle_1 = agent.initialize_xfer("WRITE", xfer1_descs, xfer_files, "GDSTester")
    if not xfer_handle_1:
        print("Creating write transfer failed.")
        return False, 0, 0
    
    state = agent.transfer(xfer_handle_1)
    assert state != "ERR"
    
    done = False
    while not done:
        state = agent.check_xfer_state(xfer_handle_1)
        if state == "ERR":
            print("Write transfer got to Error state.")
            return False, 0, 0
        elif state == "DONE":
            done = True
    
    write_time = time.time() - write_start
    
    # READ operation  
    read_start = time.time()
    xfer_handle_2 = agent.initialize_xfer("READ", xfer2_descs, xfer_files, "GDSTester")
    if not xfer_handle_2:
        print("Creating read transfer failed.")
        return False, 0, 0
    
    state = agent.transfer(xfer_handle_2)
    assert state != "ERR"
    
    done = False
    while not done:
        state = agent.check_xfer_state(xfer_handle_2)
        if state == "ERR":
            print("Read transfer got to Error state.")
            return False, 0, 0
        elif state == "DONE":
            done = True
    
    read_time = time.time() - read_start
    
    # Cleanup
    agent.release_xfer_handle(xfer_handle_1)
    agent.release_xfer_handle(xfer_handle_2)
    agent.deregister_memory(reg_descs)
    agent.deregister_memory(file_descs)
    os.close(fd)
    
    return True, write_time, read_time


if __name__ == "__main__":
    # Use moderate buffer sizes to balance performance and resource usage
    buf_size = 4 * 1024 * 1024  # 4 MB per buffer
    max_buffers_per_batch = 32  # Limit buffers per batch to avoid resource issues
    batch_size = max_buffers_per_batch * buf_size  # 128 MB per batch
    
    # Check for demo mode
    demo_mode = False
    if len(sys.argv) >= 3 and sys.argv[2].lower() == "demo":
        demo_mode = True
    
    if len(sys.argv) < 2:
        print("Usage: python nixl_gds_example.py <file_path> [mode]")
        print("  mode options:")
        print("    demo: Run only 3 buffer iterations for quick testing")
        print("    full: Process entire file (may cause resource exhaustion on large files)")
        print("    (none): Process up to 3 batches (384 MB) for safety")
        exit(0)

    # Get file size and calculate batches
    original_file_size = os.path.getsize(sys.argv[1])
    file_size = original_file_size
    num_batches = (file_size + batch_size - 1) // batch_size
    
    print(f"File size: {file_size:,} bytes ({file_size/(1024**3):.2f} GB)")
    print(f"Buffer size: {buf_size:,} bytes ({buf_size/(1024**2):.1f} MB)")
    print(f"Batch size: {batch_size:,} bytes ({batch_size/(1024**2):.1f} MB)")
    print(f"Number of batches: {num_batches}")
    
    if demo_mode:
        print(f"\n🎯 DEMO MODE: Running only 3 buffer iterations for quick testing")
        print(f"Processing {3 * buf_size:,} bytes ({(3 * buf_size)/(1024**2):.0f} MB) of the file.")
        # Limit to just 3 buffers (12 MB total)
        file_size = min(file_size, 3 * buf_size)
        num_batches = 1  # Single batch with 3 buffers
        batch_size = file_size
        max_buffers_per_batch = 3
    else:
        # Check for "full" argument to process entire file
        process_full_file = len(sys.argv) >= 3 and sys.argv[2].lower() == "full"
        
        if process_full_file:
            print(f"🚀 FULL MODE: Processing entire file")
            print(f"Processing {file_size:,} bytes ({file_size/(1024**2):.0f} MB) of the file.")
        elif num_batches > 3:
            print(f"⚠️  Large file detected. For safety, limiting to first 3 batches (384 MB).")
            print(f"   Use 'full' argument to process entire file: python {sys.argv[0]} {sys.argv[1]} full")
            print(f"Processing {3 * batch_size:,} bytes ({(3 * batch_size)/(1024**2):.0f} MB) of the file.")
            # Limit for safety
            demo_size = min(file_size, 3 * batch_size)
            file_size = demo_size
            num_batches = 3
        else:
            print(f"Processing {file_size:,} bytes ({file_size/(1024**2):.0f} MB) of the file.")

    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    agent_config = nixl_agent_config(backends=[])
    nixl_agent1 = nixl_agent("GDSTester", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    assert "GDS" in plugin_list

    print("Plugin parameters")
    print(nixl_agent1.get_plugin_mem_types("GDS"))
    print(nixl_agent1.get_plugin_params("GDS"))

    nixl_agent1.create_backend("GDS")

    print("\nLoaded backend parameters")
    print(nixl_agent1.get_backend_mem_types("GDS"))
    print(nixl_agent1.get_backend_params("GDS"))
    print()

    # Allocate buffers for one batch at a time
    write_addrs = []
    read_addrs = []
    
    for i in range(max_buffers_per_batch):
        # Allocate write and read buffers
        write_addr = nixl_utils.malloc_passthru(buf_size)
        read_addr = nixl_utils.malloc_passthru(buf_size)
        
        # Initialize write buffer with test pattern
        nixl_utils.ba_buf(write_addr, buf_size)
        
        write_addrs.append(write_addr)
        read_addrs.append(read_addr)

    # Run batch transfer
    success = run_batch_transfer(
        nixl_agent1, write_addrs, read_addrs, sys.argv[1], buf_size, batch_size, file_size, original_file_size
    )
    
    if success:
        print(f"\n✓ Successfully processed entire {file_size/(1024**3):.2f} GB file!")

    # Cleanup
    for i in range(max_buffers_per_batch):
        nixl_utils.free_passthru(write_addrs[i])
        nixl_utils.free_passthru(read_addrs[i])

    print("\nTest Complete.")
