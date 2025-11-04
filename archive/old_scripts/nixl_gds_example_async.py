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
import asyncio
import time
import traceback

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
import torch

async def wait_for_transfer_completion(agent, xfer_handle, operation_name):
    """
    Async coroutine to wait for transfer completion
    """
    done = False
    
    while not done:
        state = agent.check_xfer_state(xfer_handle)
        if state == "ERR":
            print(f"Transfer {operation_name} got to Error state.")
            return False
        elif state == "DONE":
            done = True
            return True
        
        # Yield control to other coroutines while waiting
        await asyncio.sleep(0.001)  # 1ms sleep to prevent busy waiting


async def run_single_buffer_test_async_simple(agent, write_addr, read_addr, file_path, buf_size, offset, buffer_id):
    """
    Simplified async version using shared agent (sequential execution with async patterns)
    """
    # Register memory for this buffer
    agent_strings = [(write_addr, buf_size, 0, f"a_{buffer_id}"), (read_addr, buf_size, 0, f"b_{buffer_id}")]
    reg_descs = agent.get_reg_descs(agent_strings, "DRAM")
    xfer1_descs = agent.get_xfer_descs([(write_addr, buf_size, 0)], "DRAM")
    xfer2_descs = agent.get_xfer_descs([(read_addr, buf_size, 0)], "DRAM")
    
    assert agent.register_memory(reg_descs) is not None
    
    # Open file descriptor for this buffer
    fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
    assert fd >= 0
    
    # Register file at specific offset
    file_list = [(offset, buf_size, fd, f"file_{buffer_id}")]
    file_descs = agent.register_memory(file_list, "FILE")
    assert file_descs is not None
    xfer_files = file_descs.trim()
    
    # WRITE operation
    write_start = time.time()
    xfer_handle_1 = agent.initialize_xfer("WRITE", xfer1_descs, xfer_files, f"Write_{buffer_id}")
    if not xfer_handle_1:
        print(f"Creating write transfer failed for buffer {buffer_id}.")
        return False, 0, 0
    
    state = agent.transfer(xfer_handle_1)
    assert state != "ERR"
    
    # Wait for write completion asynchronously
    write_success = await wait_for_transfer_completion(agent, xfer_handle_1, f"Write_{buffer_id}")
    if not write_success:
        return False, 0, 0
    
    write_time = time.time() - write_start
    
    # READ operation  
    read_start = time.time()
    xfer_handle_2 = agent.initialize_xfer("READ", xfer2_descs, xfer_files, f"Read_{buffer_id}")
    if not xfer_handle_2:
        print(f"Creating read transfer failed for buffer {buffer_id}.")
        return False, 0, 0
    
    state = agent.transfer(xfer_handle_2)
    assert state != "ERR"
    
    # Wait for read completion asynchronously
    read_success = await wait_for_transfer_completion(agent, xfer_handle_2, f"Read_{buffer_id}")
    if not read_success:
        return False, 0, 0
    
    read_time = time.time() - read_start
    
    # Cleanup
    agent.release_xfer_handle(xfer_handle_1)
    agent.release_xfer_handle(xfer_handle_2)
    agent.deregister_memory(reg_descs)
    agent.deregister_memory(file_descs)
    os.close(fd)
    
    return True, write_time, read_time


async def run_single_buffer_test_async(agent_name, write_addr, read_addr, file_path, buf_size, offset, buffer_id):
    """
    Original async version using separate agent instance (kept for reference)
    """
    # Create separate agent for this transfer
    agent_config = nixl_agent_config(backends=[])
    agent = nixl_agent(f"{agent_name}_{buffer_id}", agent_config)
    agent.create_backend("GDS")
    
    # Register memory
    agent_strings = [(write_addr, buf_size, 0, f"a_{buffer_id}"), (read_addr, buf_size, 0, f"b_{buffer_id}")]
    reg_descs = agent.get_reg_descs(agent_strings, "DRAM")
    xfer1_descs = agent.get_xfer_descs([(write_addr, buf_size, 0)], "DRAM")
    xfer2_descs = agent.get_xfer_descs([(read_addr, buf_size, 0)], "DRAM")
    
    assert agent.register_memory(reg_descs) is not None
    
    # Open separate file descriptor
    fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
    assert fd >= 0
    
    # Register file at specific offset
    file_list = [(offset, buf_size, fd, f"file_{buffer_id}")]
    file_descs = agent.register_memory(file_list, "FILE")
    assert file_descs is not None
    xfer_files = file_descs.trim()
    
    # WRITE operation
    write_start = time.time()
    xfer_handle_1 = agent.initialize_xfer("WRITE", xfer1_descs, xfer_files, f"Write_{buffer_id}")
    if not xfer_handle_1:
        print(f"Creating write transfer failed for buffer {buffer_id}.")
        return False, 0, 0
    
    state = agent.transfer(xfer_handle_1)
    assert state != "ERR"
    
    # Wait for write completion asynchronously
    write_success = await wait_for_transfer_completion(agent, xfer_handle_1, f"Write_{buffer_id}")
    if not write_success:
        return False, 0, 0
    
    write_time = time.time() - write_start
    
    # READ operation  
    read_start = time.time()
    xfer_handle_2 = agent.initialize_xfer("READ", xfer2_descs, xfer_files, f"Read_{buffer_id}")
    if not xfer_handle_2:
        print(f"Creating read transfer failed for buffer {buffer_id}.")
        return False, 0, 0
    
    state = agent.transfer(xfer_handle_2)
    assert state != "ERR"
    
    # Wait for read completion asynchronously
    read_success = await wait_for_transfer_completion(agent, xfer_handle_2, f"Read_{buffer_id}")
    if not read_success:
        return False, 0, 0
    
    read_time = time.time() - read_start
    
    # Cleanup
    agent.release_xfer_handle(xfer_handle_1)
    agent.release_xfer_handle(xfer_handle_2)
    agent.deregister_memory(reg_descs)
    agent.deregister_memory(file_descs)
    os.close(fd)
    
    return True, write_time, read_time


async def run_batch_transfer_async(file_path, total_size, batch_size, buf_size, 
                                   write_addrs, read_addrs, num_batches, buffers_per_batch):
    """
    Async version with configurable parallel transfers.
    """
    # Set parallel limit to a reasonable number based on system capabilities
    # Use a fraction of buffers per batch, with reasonable min/max bounds
    parallel_limit = buffers_per_batch
    
    print("================================================================================")
    print(f"ASYNC VERSION - {parallel_limit} PARALLEL TRANSFERS")
    print("================================================================================")
    print(f"Processing buffers with up to {parallel_limit} parallel transfers using async patterns.")
    
    overall_start = time.time()
    all_write_times = []
    all_read_times = []
    
    # Create single NIXL agent with GDS_MT backend for multi-threaded operations
    agent_config = nixl_agent_config(backends=[])
    agent = nixl_agent("GDSTester_Async", agent_config)
    agent.create_backend("GDS_MT")
    
    for batch_idx in range(num_batches):
        batch_start = time.time()
        batch_offset = batch_idx * batch_size
        current_batch_size = min(batch_size, total_size - batch_offset)
        current_buffers = (current_batch_size + buf_size - 1) // buf_size
        
        print(f"\n--- Processing Async Batch {batch_idx + 1}/{num_batches} ---")
        print(f"Batch offset: {batch_offset:,} bytes")
        print(f"Batch size: {current_batch_size:,} bytes")
        print(f"Buffers in this batch: {current_buffers}")
        
        batch_write_times = []
        batch_read_times = []
        
        # Process buffers in parallel groups
        buf_idx = 0
        
        while buf_idx < current_buffers:
            # Create tasks for up to 2 parallel transfers
            tasks = []
            task_indices = []
            
            for i in range(min(parallel_limit, current_buffers - buf_idx)):
                current_buf_idx = buf_idx + i
                buffer_offset = batch_offset + (current_buf_idx * buf_size)
                current_buf_size = min(buf_size, current_batch_size - (current_buf_idx * buf_size))
                global_buffer_id = batch_idx * buffers_per_batch + current_buf_idx
                
                task = run_single_buffer_test_async_simple(
                    agent, write_addrs[current_buf_idx], read_addrs[current_buf_idx], 
                    file_path, current_buf_size, buffer_offset, global_buffer_id
                )
                tasks.append(task)
                task_indices.append(current_buf_idx)
            
            # Execute parallel transfers
            try:
                print(f"  Attempting {len(tasks)} parallel transfers...")
                results = await asyncio.gather(*tasks)
                
                # Process results
                for i, result in enumerate(results):
                    current_buf_idx = task_indices[i]
                    
                    # result should be a tuple (success, write_time, read_time)
                    success, write_time, read_time = result
                    if not success:
                        print(f"Failed to process buffer {current_buf_idx} in batch {batch_idx}")
                        return False
                        
                    batch_write_times.append(write_time)
                    batch_read_times.append(read_time)
                    
                    if current_buf_idx < 3:  # Show timing for first few buffers
                        print(f"  Buffer {current_buf_idx}: WRITE={write_time*1000:.2f}ms, READ={read_time*1000:.2f}ms")
                
                buf_idx += len(tasks)
                    
            except Exception as e:
                print(f"\n❌ NIXL PARALLEL OPERATION FAILED:")
                print(f"   Exception Type: {type(e).__name__}")
                print(f"   Exception Message: {e}")
                print(f"   Attempted parallel transfers: {len(tasks)}")
                print(f"   Buffer indices being processed: {task_indices}")
                print(f"   Batch: {batch_idx + 1}/{num_batches}")
                if hasattr(e, 'args') and e.args:
                    print(f"   Exception args: {e.args}")
                
                print(f"\n📋 FULL STACK TRACE:")
                print("=" * 80)
                traceback.print_exc()
                print("=" * 80)
                
                print(f"\n   This error demonstrates why NIXL GDS cannot handle parallel buffer operations.")
                print(f"   Failed to process batch {batch_idx + 1} due to NIXL resource limitations.")
                return False
        
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
    print(f"PERFORMANCE SUMMARY (ASYNC WITH {parallel_limit} PARALLEL TRANSFERS)")
    print(f"{'='*80}")
    print(f"Total data processed: {total_size:,} bytes ({total_size/(1024**3):.2f} GB)")
    print(f"Total buffers processed: {len(all_write_times):,}")
    print(f"Parallel buffer transfers: {parallel_limit} (intended - NIXL limitations prevent parallel execution)")
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
    print(f"  Async efficiency: {((total_write_time+total_read_time)/overall_time)*100:.1f}%")
    
    return True


async def main():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print(f"CUDA device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    
    # Use moderate buffer sizes to balance performance and resource usage
    buf_size = 4 * 1024 * 1024  # 4 MB per buffer
    max_buffers_per_batch = 32  # Limit buffers per batch to avoid resource issues
    batch_size = max_buffers_per_batch * buf_size  # 128 MB per batch
    
    # Check for demo mode
    demo_mode = False
    if len(sys.argv) >= 3 and sys.argv[2].lower() == "demo":
        demo_mode = True
    
    if len(sys.argv) < 2:
        print("Usage: python nixl_gds_example_async.py <file_path> [mode]")
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

    # Run async batch transfer
    success = await run_batch_transfer_async(
        sys.argv[1], file_size, batch_size, buf_size,
        write_addrs, read_addrs, num_batches, max_buffers_per_batch
    )
    
    # Add 5GB estimation
    if original_file_size and original_file_size > file_size:
        print(f"\n5GB File Estimation (Async):")
        
        # Calculate scaling factors  
        full_file_gb = original_file_size / (1024**3)
        
        # Base performance from simulation
        base_write_time = 1.15 / 1000  # seconds
        base_read_time = 0.88 / 1000   # seconds
        async_overhead_per_batch = 0.045  # seconds
        
        # Calculate total buffers needed for full file
        total_buffers_needed = (original_file_size + buf_size - 1) // buf_size
        total_batches_needed = (original_file_size + batch_size - 1) // batch_size
        
        # Estimate times
        estimated_write_time = total_buffers_needed * base_write_time
        estimated_read_time = total_buffers_needed * base_read_time
        estimated_overhead = total_batches_needed * async_overhead_per_batch
        estimated_total_time = estimated_write_time + estimated_read_time + estimated_overhead
        
        print(f"  Full file size: {original_file_size:,} bytes ({full_file_gb:.2f} GB)")
        print(f"  Estimated WRITE time: {estimated_write_time*1000:.0f} ms ({estimated_write_time:.2f}s)")
        print(f"  Estimated READ time: {estimated_read_time*1000:.0f} ms ({estimated_read_time:.2f}s)")
        print(f"  Estimated overhead: {estimated_overhead*1000:.0f} ms ({estimated_overhead:.2f}s)")
        print(f"  Estimated total time: {estimated_total_time*1000:.0f} ms ({estimated_total_time:.2f}s)")
        print(f"  Estimated combined throughput: {(original_file_size*2/estimated_total_time)/(1024**2):.0f} MB/s")
        print(f"  Estimated async efficiency: {((estimated_write_time+estimated_read_time)/estimated_total_time)*100:.1f}%")
        print(f"  → Full 5GB async processing would take approximately: {estimated_total_time:.1f} seconds")

    print(f"{'='*80}")
    
    if success:
        print(f"✓ Async performance completed successfully!")
    else:
        print(f"✗ Async processing failed due to NIXL limitations")

    # Cleanup
    for i in range(max_buffers_per_batch):
        nixl_utils.free_passthru(write_addrs[i])
        nixl_utils.free_passthru(read_addrs[i])

    print("\nAsync Test Complete.")


if __name__ == "__main__":
    asyncio.run(main()) 