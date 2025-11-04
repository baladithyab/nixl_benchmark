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

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

def run_posix_example():
    """Run NIXL example using POSIX plugin for file operations"""
    
    buf_size = 16 * 4096
    
    if len(sys.argv) < 2:
        print("Please specify file path in argv")
        print("Usage: python nixl_posix_example.py <file_path>")
        exit(1)

    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    agent_config = nixl_agent_config(backends=[])
    nixl_agent1 = nixl_agent("POSIXTester", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    print(f"Available plugins: {plugin_list}")
    
    # Check if POSIX is available
    if "POSIX" not in plugin_list:
        print("✗ POSIX plugin not available!")
        print("Available plugins:", plugin_list)
        exit(1)

    print("✓ POSIX plugin found!")
    
    # Show POSIX plugin details
    print("POSIX Plugin parameters")
    print(nixl_agent1.get_plugin_mem_types("POSIX"))
    print(nixl_agent1.get_plugin_params("POSIX"))

    nixl_agent1.create_backend("POSIX")

    print("\nLoaded POSIX backend parameters")
    print(nixl_agent1.get_backend_mem_types("POSIX"))
    print(nixl_agent1.get_backend_params("POSIX"))
    print()

    # Allocate DRAM buffers and initialize one with test pattern
    addr1 = nixl_utils.malloc_passthru(buf_size)
    addr2 = nixl_utils.malloc_passthru(buf_size)
    nixl_utils.ba_buf(addr1, buf_size)  # Initialize with 0xba pattern

    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM")
    agent1_xfer1_descs = nixl_agent1.get_xfer_descs([(addr1, buf_size, 0)], "DRAM")
    agent1_xfer2_descs = nixl_agent1.get_xfer_descs([(addr2, buf_size, 0)], "DRAM")

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    # Open file for testing
    file_path = sys.argv[1]
    agent1_fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
    assert agent1_fd >= 0

    agent1_file_list = [(0, buf_size, agent1_fd, "b")]

    agent1_file_descs = nixl_agent1.register_memory(agent1_file_list, "FILE")
    assert agent1_file_descs is not None

    agent1_xfer_files = agent1_file_descs.trim()

    print("Starting POSIX file operations...")

    # Write data from DRAM to file
    xfer_handle_1 = nixl_agent1.initialize_xfer(
        "WRITE", agent1_xfer1_descs, agent1_xfer_files, "POSIXTester"
    )
    if not xfer_handle_1:
        print("Creating write transfer failed.")
        exit(1)

    state = nixl_agent1.transfer(xfer_handle_1)
    assert state != "ERR"

    done = False
    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_1)
        if state == "ERR":
            print("Write transfer got to Error state.")
            exit(1)
        elif state == "DONE":
            done = True
            print("Write transfer completed")

    # Read data from file back to second DRAM buffer
    xfer_handle_2 = nixl_agent1.initialize_xfer(
        "READ", agent1_xfer2_descs, agent1_xfer_files, "POSIXTester"
    )
    if not xfer_handle_2:
        print("Creating read transfer failed.")
        exit(1)

    state = nixl_agent1.transfer(xfer_handle_2)
    assert state != "ERR"

    done = False
    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_2)
        if state == "ERR":
            print("Read transfer got to Error state.")
            exit(1)
        elif state == "DONE":
            done = True
            print("Read transfer completed")

    # Verify the transfer
    print("Verifying transfer...")
    nixl_utils.verify_transfer(addr1, addr2, buf_size)
    print("✓ Transfer verification successful!")

    # Cleanup
    nixl_agent1.release_xfer_handle(xfer_handle_1)
    nixl_agent1.release_xfer_handle(xfer_handle_2)
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent1.deregister_memory(agent1_file_descs)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr2)

    os.close(agent1_fd)

    print("POSIX Test Complete.")

if __name__ == "__main__":
    run_posix_example() 