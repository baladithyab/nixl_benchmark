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
import time

import numpy as np
import torch

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

if __name__ == "__main__":
    buf_size = 256
    # Allocate memory and register with NIXL

    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    # Example using nixl_agent_config
    agent_config = nixl_agent_config(backends=["UCX"])
    nixl_agent1 = nixl_agent("target", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    assert "UCX" in plugin_list

    print("Plugin parameters")
    print(nixl_agent1.get_plugin_mem_types("UCX"))
    print(nixl_agent1.get_plugin_params("UCX"))

    print("\nLoaded backend parameters")
    print(nixl_agent1.get_backend_mem_types("UCX"))
    print(nixl_agent1.get_backend_params("UCX"))
    print()

    addr1 = nixl_utils.malloc_passthru(buf_size * 2)
    addr2 = addr1 + buf_size

    agent1_addrs = [(addr1, buf_size, 0), (addr2, buf_size, 0)]
    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM", is_sorted=True)
    agent1_xfer_descs = nixl_agent1.get_xfer_descs(agent1_addrs, "DRAM", is_sorted=True)

    # Prefer numpy arrays for performance
    agent1_addrs_np = np.array(agent1_addrs)
    agent1_xfer_descs_np = nixl_agent1.get_xfer_descs(
        agent1_addrs_np, "DRAM", is_sorted=True
    )
    agent1_reg_descs_np = nixl_agent1.get_reg_descs(
        agent1_addrs_np, "DRAM", is_sorted=True
    )

    assert agent1_xfer_descs == agent1_xfer_descs_np
    assert agent1_reg_descs == agent1_reg_descs_np
    print(agent1_reg_descs, agent1_reg_descs_np)

    # Just for tensor test
    tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]
    agent1_tensor_reg_descs = nixl_agent1.get_reg_descs(tensors)
    agent1_tensor_xfer_descs = nixl_agent1.get_xfer_descs(tensors)

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    # Example using default configs, which is UCX backend only
    nixl_agent2 = nixl_agent("initiator", None)
    addr3 = nixl_utils.malloc_passthru(buf_size * 2)
    addr4 = addr3 + buf_size

    agent2_addrs = [(addr3, buf_size, 0), (addr4, buf_size, 0)]
    agent2_strings = [(addr3, buf_size, 0, "a"), (addr4, buf_size, 0, "b")]

    agent2_reg_descs = nixl_agent2.get_reg_descs(agent2_strings, "DRAM", is_sorted=True)
    agent2_xfer_descs = nixl_agent2.get_xfer_descs(agent2_addrs, "DRAM", is_sorted=True)

    agent2_descs = nixl_agent2.register_memory(agent2_reg_descs, is_sorted=True)
    assert agent2_descs is not None

    # Exchange metadata
    meta = nixl_agent1.get_agent_metadata()
    remote_name = nixl_agent2.add_remote_agent(meta)
    print("Loaded name from metadata:", remote_name, flush=True)

    serdes = nixl_agent1.get_serialized_descs(agent1_reg_descs)
    src_descs_recvd = nixl_agent2.deserialize_descs(serdes)
    assert src_descs_recvd == agent1_reg_descs

    # initialize transfer mode
    xfer_handle_1 = nixl_agent2.initialize_xfer(
        "READ", agent2_xfer_descs, agent1_xfer_descs, remote_name, b"UUID1"
    )
    if not xfer_handle_1:
        print("Creating transfer failed.")
        exit()

    # test multiple postings
    transfer_start_time = time.time()
    for i in range(2):
        transfer_iter_start = time.time()
        state = nixl_agent2.transfer(xfer_handle_1)
        assert state != "ERR"

        target_done = False
        init_done = False

        while (not init_done) or (not target_done):
            if not init_done:
                state = nixl_agent2.check_xfer_state(xfer_handle_1)
                if state == "ERR":
                    print("Transfer got to Error state.")
                    exit()
                elif state == "DONE":
                    init_done = True
                    print("Initiator done")

            if not target_done:
                if nixl_agent1.check_remote_xfer_done("initiator", b"UUID1"):
                    target_done = True
                    print("Target done")
        
        transfer_iter_end = time.time()
        transfer_time = transfer_iter_end - transfer_iter_start
        data_size = 512  # 2 buffers * 256 bytes each
        bandwidth = (data_size / transfer_time) / (1024 * 1024)  # MB/s
        print(f"Transfer 1 iteration {i+1}: {transfer_time*1000000:.1f}μs, {bandwidth:.2f} MB/s")
    
    transfer_total_time = time.time() - transfer_start_time
    total_data = 2 * 512  # 2 iterations * 512 bytes
    total_bandwidth = (total_data / transfer_total_time) / (1024 * 1024)  # MB/s
    print(f"Transfer 1 total: {transfer_total_time*1000000:.1f}μs, {total_bandwidth:.2f} MB/s")

    # prep transfer mode
    local_prep_handle = nixl_agent2.prep_xfer_dlist(
        "NIXL_INIT_AGENT", [(addr3, buf_size, 0), (addr4, buf_size, 0)], "DRAM", True
    )
    remote_prep_handle = nixl_agent2.prep_xfer_dlist(
        remote_name, agent1_xfer_descs, "DRAM"
    )

    assert local_prep_handle != 0
    assert remote_prep_handle != 0

    # test send_notif

    test_notif = str.encode("DESCS: ") + serdes
    nixl_agent2.send_notif(remote_name, test_notif)

    print("sent notif ")
    print(test_notif)

    notif_recv = False

    while not notif_recv:
        notif_map = nixl_agent1.get_new_notifs()
        if "initiator" in notif_map:
            print("received message from initiator")
            for msg in notif_map["initiator"]:
                if msg == test_notif:
                    notif_recv = True

    print("notif test complete, doing transfer 2\n")

    xfer_handle_2 = nixl_agent2.make_prepped_xfer(
        "WRITE", local_prep_handle, [0, 1], remote_prep_handle, [1, 0], b"UUID2"
    )
    if not local_prep_handle or not remote_prep_handle:
        print("Preparing transfer side handles failed.")
        exit()

    if not xfer_handle_2:
        print("Make prepped transfer failed.")
        exit()

    transfer2_start = time.time()
    state = nixl_agent2.transfer(xfer_handle_2)
    assert state != "ERR"

    target_done = False
    init_done = False

    print("Transfer 2 started")

    while (not init_done) or (not target_done):
        if not init_done:
            state = nixl_agent2.check_xfer_state(xfer_handle_2)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                init_done = True
                print("Initiator done")

        if not target_done:
            if nixl_agent1.check_remote_xfer_done("initiator", b"UUID2"):
                target_done = True
                print("Target done")
    
    transfer2_end = time.time()
    transfer2_time = transfer2_end - transfer2_start
    data_size = 512  # 2 buffers * 256 bytes each
    bandwidth = (data_size / transfer2_time) / (1024 * 1024)  # MB/s
    print(f"Transfer 2: {transfer2_time*1000000:.1f}μs, {bandwidth:.2f} MB/s")

    nixl_agent2.release_xfer_handle(xfer_handle_1)
    nixl_agent2.release_xfer_handle(xfer_handle_2)
    nixl_agent2.release_dlist_handle(local_prep_handle)
    nixl_agent2.release_dlist_handle(remote_prep_handle)
    nixl_agent2.remove_remote_agent("target")
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent2.deregister_memory(agent2_reg_descs)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr3)

    # Summary of all transfers
    total_transfers = 3  # 2 from transfer 1 + 1 from transfer 2
    total_data_transferred = total_transfers * 512  # bytes
    print(f"\n=== TRANSFER SUMMARY ===")
    print(f"Total transfers: {total_transfers}")
    print(f"Total data transferred: {total_data_transferred} bytes ({total_data_transferred/1024:.2f} KB)")
    print(f"Average data per transfer: 512 bytes")
    print("Test Complete.")