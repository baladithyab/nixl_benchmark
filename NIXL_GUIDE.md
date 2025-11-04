# NIXL (NVIDIA Inference Xfer Library) - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Available Backends](#available-backends)
4. [Python API Usage](#python-api-usage)
5. [C++ API Usage](#cpp-api-usage)
6. [Multi-Node Communication](#multi-node-communication)
7. [Benchmark Suite](#benchmark-suite)

## Overview

NIXL (NVIDIA Inference Xfer Library) is a high-performance communication library designed for accelerating point-to-point data transfers in AI inference frameworks. It provides:

- **Unified API** for heterogeneous memory (CPU DRAM, GPU VRAM) and storage (File, Block, Object Store)
- **Modular plugin architecture** supporting multiple communication backends
- **High bandwidth, low-latency** transfers optimized for distributed inference workloads
- **Asynchronous operations** with non-blocking transfers
- **Metadata exchange** for distributed coordination

### Key Features
- **Memory Types**: DRAM (CPU), VRAM (GPU), Block Storage, File Storage, Object Storage
- **Transfer Operations**: READ and WRITE with one-sided RDMA semantics
- **Notification System**: Inter-agent coordination and completion signaling
- **ETCD Integration**: Distributed metadata exchange and coordination

## Architecture

NIXL uses a two-layer architecture:

### North Bound API (NB API)
User-facing API for expressing transfer requests through simple buffer list primitives:
- Agent creation and configuration
- Memory registration
- Transfer request creation and execution
- Metadata exchange

### South Bound API (SB API)
Backend plugin interface that delegates transfers to optimal transport:
- Connection management
- Memory registration with backend-specific metadata
- Transfer operations (prep, post, check, release)
- Notification handling

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│    (Python/C++ User Code)               │
└─────────────────┬───────────────────────┘
                  │ North Bound API
┌─────────────────▼───────────────────────┐
│         NIXL Transfer Agent             │
│  - Memory Management                    │
│  - Metadata Bookkeeping                 │
│  - Backend Selection                    │
└─────────────────┬───────────────────────┘
                  │ South Bound API
┌─────────────────▼───────────────────────┐
│         Backend Plugins                 │
│  UCX | GDS | POSIX | Mooncake | ...     │
└─────────────────────────────────────────┘
```

## Available Backends

### Network Backends (Multi-Node Communication)

| Backend | Description | Memory Types | Use Case |
|---------|-------------|--------------|----------|
| **UCX** | Unified Communication X - High-performance networking | DRAM, VRAM | General-purpose multi-node GPU/CPU transfers |
| **UCX_MO** | UCX Memory-Only variant | DRAM, VRAM | Memory-focused transfers |
| **GPUNETIO** | NVIDIA DOCA GPUNetIO - GPU-initiated networking | VRAM | GPU-to-GPU direct transfers |
| **Mooncake** | Mooncake Transfer Engine | DRAM, VRAM | KV-cache transfers in LLM serving |
| **Libfabric** | OpenFabrics libfabric | DRAM, VRAM | AWS EFA and general fabric support |

### Storage Backends (Local/Remote Storage)

| Backend | Description | Memory Types | Use Case |
|---------|-------------|--------------|----------|
| **GDS** | GPU Direct Storage | VRAM, File | Direct GPU-to-storage transfers |
| **GDS_MT** | Multi-threaded GDS | VRAM, File | Parallel GPU-to-storage transfers |
| **POSIX** | POSIX file I/O (AIO/io_uring) | DRAM, File | CPU-to-storage transfers |
| **HF3FS** | Hierarchical File System | DRAM, VRAM, File | Tiered storage access |
| **OBJ** | S3 Object Storage | DRAM, Object Store | Cloud object storage |
| **GUSLI** | G3+ User Space Library | VRAM, Block | Direct block device access |

### Backend Capabilities

Each backend implements different capabilities:
- `supportsLocal()`: Within-node transfers
- `supportsRemote()`: Cross-node transfers  
- `supportsNotif()`: Notification support
- `getSupportedMems()`: Supported memory types

**Network backends** typically support all three (local, remote, notifications).
**Storage backends** typically support only local transfers (storage client is local).

## Python API Usage

### Installation

```bash
# From PyPI (recommended)
pip install nixl

# From source
cd nixl/
pip install .
```

### Basic Workflow

```python
from nixl._api import nixl_agent, nixl_agent_config
import torch

# 1. Create agent with backend configuration
config = nixl_agent_config(backends=["UCX"])
agent = nixl_agent("my_agent", config)

# 2. Allocate and register memory
tensors = [torch.zeros(1024, dtype=torch.float32) for _ in range(2)]
reg_descs = agent.register_memory(tensors)

# 3. Exchange metadata with remote agent
local_metadata = agent.get_agent_metadata()
# ... send to remote agent via network/ETCD ...
agent.add_remote_agent(remote_metadata)

# 4. Create and execute transfer
xfer_handle = agent.initialize_xfer(
    "WRITE",              # Operation: READ or WRITE
    local_descs,          # Source descriptors
    remote_descs,         # Target descriptors  
    "remote_agent_name",  # Remote agent name
    b"transfer_id"        # Optional notification payload
)

# 5. Post transfer (non-blocking)
state = agent.transfer(xfer_handle)

# 6. Check completion
while True:
    state = agent.check_xfer_state(xfer_handle)
    if state == "DONE":
        break
    elif state == "ERR":
        raise RuntimeError("Transfer failed")

# 7. Cleanup
agent.release_xfer_handle(xfer_handle)
agent.deregister_memory(reg_descs)
```

### Memory Descriptor Creation

```python
# From PyTorch tensors (automatic)
tensors = [torch.randn(100) for _ in range(5)]
reg_descs = agent.register_memory(tensors)

# From NumPy arrays
import numpy as np
arrays = [np.zeros(100) for _ in range(5)]
reg_descs = agent.register_memory(arrays)

# Manual descriptor creation
import nixl._utils as nixl_utils
addr = nixl_utils.malloc_passthru(1024)
descs = [(addr, 1024, 0)]  # (address, length, device_id)
reg_descs = agent.get_reg_descs(descs, "DRAM")
agent.register_memory(reg_descs)
```

### Backend-Specific Configuration

```python
# UCX backend with custom parameters
config = nixl_agent_config(
    backends=["UCX"],
    backend_params={
        "UCX": {
            "UCX_TLS": "rc,cuda_copy,cuda_ipc",
            "UCX_NET_DEVICES": "mlx5_0:1"
        }
    }
)

# GDS backend for storage
config = nixl_agent_config(backends=["GDS"])
agent = nixl_agent("storage_agent", config)

# Register file for GDS
file_path = "/mnt/storage/data.bin"
file_descs = agent.get_reg_descs(
    [(0, file_size, fd, file_path.encode())],
    "FILE"
)
agent.register_memory(file_descs)
```

### Notification Handling

```python
# Send notification
agent.send_notif("remote_agent", b"custom_payload")

# Receive notifications
notifs = agent.get_new_notifs()
for agent_name, notif_list in notifs.items():
    for notif in notif_list:
        print(f"Received from {agent_name}: {notif}")

# Check for specific notification
if agent.check_remote_xfer_done("remote_agent", b"transfer_id"):
    print("Transfer completed")
```

## C++ API Usage

### Basic Example

```cpp
#include <nixl/agent.h>
#include <nixl/backend.h>

// Create agent
nixl::AgentConfig config;
config.backends = {"UCX"};
auto agent = std::make_unique<nixl::Agent>("my_agent", config);

// Register memory
std::vector<nixl::MemDescriptor> descs = {
    {addr, size, 0, nixl::MemType::DRAM}
};
auto reg_handle = agent->registerMemory(descs, "UCX");

// Create transfer
auto xfer_handle = agent->prepareTransfer(
    nixl::OpType::WRITE,
    local_descs,
    remote_descs,
    "remote_agent",
    "UCX"
);

// Execute transfer
agent->postTransfer(xfer_handle);

// Poll for completion
while (agent->checkTransfer(xfer_handle) != nixl::XferState::DONE) {
    // Progress
}

// Cleanup
agent->releaseTransfer(xfer_handle);
agent->deregisterMemory(reg_handle);
```

## Multi-Node Communication

### Using ETCD for Coordination

NIXL supports ETCD for distributed metadata exchange:

```bash
# Start ETCD server
docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379
```

```python
# Set environment variables
import os
os.environ["NIXL_ETCD_ENDPOINTS"] = "http://etcd-server:2379"
os.environ["NIXL_ETCD_NAMESPACE"] = "/nixl/agents"

# Agents automatically discover each other via ETCD
config = nixl_agent_config(
    backends=["UCX"],
    use_etcd=True
)
agent = nixl_agent("agent_1", config)
```

### Direct IP-based Communication

```python
# Target agent (listens on port)
config = nixl_agent_config(
    backends=["UCX"],
    enable_listen=True,
    listen_port=5555
)
target = nixl_agent("target", config)

# Initiator agent (connects to target)
config = nixl_agent_config(backends=["UCX"])
initiator = nixl_agent("initiator", config)

# Fetch remote metadata
initiator.fetch_remote_metadata("target", "192.168.1.100", 5555)
initiator.send_local_metadata("192.168.1.100", 5555)

# Wait for metadata exchange
while not initiator.check_remote_metadata("target"):
    pass
```

### Hierarchical Communication Patterns

NIXL supports various communication patterns:

1. **Pairwise**: Point-to-point between agent pairs
2. **Many-to-One**: Multiple initiators → single target (aggregation)
3. **One-to-Many**: Single initiator → multiple targets (broadcast)
4. **Tensor Parallel (TP)**: Optimized for distributed training

```python
# Example: Many-to-one aggregation
# Multiple worker agents send to aggregator
for worker_id in range(num_workers):
    xfer_handle = worker_agents[worker_id].initialize_xfer(
        "WRITE",
        worker_descs[worker_id],
        aggregator_descs,
        "aggregator",
        f"worker_{worker_id}".encode()
    )
    worker_agents[worker_id].transfer(xfer_handle)
```

## Benchmark Suite

This repository includes a comprehensive benchmark suite for testing all NIXL backends in various configurations.

### Benchmark Organization

```
benchmarks/
├── single_node/          # Single-node benchmarks
│   ├── ucx_loopback.py      # UCX local transfers
│   ├── gds_storage.py       # GDS GPU-to-storage
│   ├── posix_storage.py     # POSIX file I/O
│   └── memory_copy.py       # Memory-to-memory baseline
├── multi_node/           # Multi-node benchmarks
│   ├── ucx_p2p.py          # UCX point-to-point
│   ├── gpunetio_p2p.py     # GPUNetIO transfers
│   ├── all_reduce.py       # Collective operations
│   └── hierarchical.py     # Multi-level communication
├── storage/              # Storage backend benchmarks
│   ├── gds_benchmark.py    # GDS comprehensive test
│   ├── obj_s3.py           # S3 object storage
│   └── gusli_block.py      # Block device access
└── utils/                # Shared utilities
    ├── benchmark_base.py   # Base benchmark class
    ├── metrics.py          # Performance metrics
    └── config.py           # Configuration management
```

### Running Benchmarks

#### Single-Node Benchmarks

```bash
# UCX loopback (CPU-to-CPU)
python benchmarks/single_node/ucx_loopback.py \
  --buffer_sizes 4KB,64KB,1MB,64MB \
  --iterations 1000

# GDS storage (GPU-to-File)
python benchmarks/single_node/gds_storage.py \
  --file_path /mnt/nvme/test.dat \
  --buffer_size 1GB \
  --direct_io

# POSIX storage with io_uring
python benchmarks/single_node/posix_storage.py \
  --file_path /mnt/storage/test.dat \
  --api_type URING \
  --buffer_sizes 4KB,64KB,1MB
```

#### Multi-Node Benchmarks

```bash
# On Node 1 (target)
python benchmarks/multi_node/ucx_p2p.py \
  --mode target \
  --ip 192.168.1.100 \
  --port 5555 \
  --memory_type VRAM \
  --gpu_id 0

# On Node 2 (initiator)
python benchmarks/multi_node/ucx_p2p.py \
  --mode initiator \
  --target_ip 192.168.1.100 \
  --port 5555 \
  --memory_type VRAM \
  --gpu_id 0 \
  --buffer_sizes 1MB,16MB,256MB
```

#### Using ETCD for Multi-Node Coordination

```bash
# Start ETCD (on coordination node)
docker run -d -p 2379:2379 --name etcd \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379

# Run benchmark on all nodes (automatically coordinate)
# Node 1
python benchmarks/multi_node/ucx_p2p.py \
  --etcd_endpoints http://etcd-server:2379 \
  --benchmark_group test1 \
  --memory_type VRAM

# Node 2
python benchmarks/multi_node/ucx_p2p.py \
  --etcd_endpoints http://etcd-server:2379 \
  --benchmark_group test1 \
  --memory_type VRAM
```

### Backend-Specific Benchmarks

#### UCX Backend
```bash
# CPU-to-CPU (DRAM)
python benchmarks/multi_node/ucx_p2p.py \
  --memory_type DRAM \
  --buffer_sizes 4KB,64KB,1MB,16MB,256MB

# GPU-to-GPU (VRAM)
python benchmarks/multi_node/ucx_p2p.py \
  --memory_type VRAM \
  --gpu_id 0 \
  --buffer_sizes 1MB,16MB,256MB,1GB

# Mixed CPU-GPU
python benchmarks/multi_node/ucx_p2p.py \
  --initiator_mem DRAM \
  --target_mem VRAM \
  --buffer_sizes 1MB,16MB,256MB
```

#### GDS Backend
```bash
# GPU-to-Storage write
python benchmarks/storage/gds_benchmark.py \
  --operation WRITE \
  --file_path /mnt/nvme/test.dat \
  --buffer_size 1GB \
  --batch_size 128

# GPU-to-Storage read
python benchmarks/storage/gds_benchmark.py \
  --operation READ \
  --file_path /mnt/nvme/test.dat \
  --buffer_size 1GB \
  --verify_data
```

#### GPUNETIO Backend
```bash
# GPU-to-GPU direct (requires DOCA)
python benchmarks/multi_node/gpunetio_p2p.py \
  --gpu_devices 0,1 \
  --buffer_sizes 1MB,16MB,256MB \
  --etcd_endpoints http://etcd-server:2379
```

#### Object Storage (S3)
```bash
# S3 benchmark
python benchmarks/storage/obj_s3.py \
  --bucket_name my-bucket \
  --region us-west-2 \
  --buffer_sizes 1MB,16MB,256MB \
  --num_objects 100
```

### Benchmark Metrics

All benchmarks report:
- **Bandwidth**: GB/s or MB/s
- **Latency**: Mean, P50, P95, P99 percentiles
- **IOPS**: Operations per second (for small transfers)
- **CPU Utilization**: Percentage during transfer
- **GPU Utilization**: For VRAM transfers
- **Consistency**: Data verification results

### Performance Tuning Tips

#### UCX Backend
```bash
# Enable GPU Direct RDMA
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy

# Specify network device
export UCX_NET_DEVICES=mlx5_0:1

# Increase buffer sizes
export UCX_RC_TX_QUEUE_LEN=4096
export UCX_RC_RX_QUEUE_LEN=4096
```

#### GDS Backend
```bash
# Increase batch sizes for better throughput
--gds_batch_pool_size 256 \
--gds_batch_limit 512

# Use direct I/O
--storage_enable_direct
```

#### General Tuning
```bash
# Enable progress threads
--enable_pt --progress_threads 2

# Increase iterations for stable measurements
--warmup_iter 100 --num_iter 1000

# Use VMM for large allocations
--enable_vmm
```

### Example Benchmark Results Format

```
=== NIXL Benchmark Results ===
Backend: UCX
Memory Type: VRAM (GPU 0) -> VRAM (GPU 0, Remote)
Operation: WRITE

Buffer Size | Bandwidth | Latency (μs) | P95 (μs) | P99 (μs)
------------|-----------|--------------|----------|----------
4 KB        | 0.12 GB/s | 32.5         | 35.2     | 38.1
64 KB       | 1.85 GB/s | 34.2         | 37.8     | 41.3
1 MB        | 22.3 GB/s | 44.8         | 48.5     | 52.7
16 MB       | 89.7 GB/s | 178.4        | 185.2    | 192.8
256 MB      | 95.2 GB/s | 2689.3       | 2712.5   | 2745.1
```

## Common Use Cases

### 1. LLM KV-Cache Transfer
```python
# Transfer KV-cache between GPU nodes
config = nixl_agent_config(backends=["UCX"])
agent = nixl_agent("kv_cache_node", config)

# Register KV-cache tensors
kv_cache = [torch.randn(batch, heads, seq_len, dim).cuda()
            for _ in range(num_layers)]
reg_descs = agent.register_memory(kv_cache)

# Transfer to remote node
xfer_handle = agent.initialize_xfer(
    "WRITE", local_descs, remote_descs, "remote_node", b"kv_cache"
)
agent.transfer(xfer_handle)
```

### 2. Checkpoint to Storage
```python
# Save model checkpoint to storage using GDS
config = nixl_agent_config(backends=["GDS"])
agent = nixl_agent("checkpoint_agent", config)

# Register model parameters
model_params = [p.data for p in model.parameters()]
reg_descs = agent.register_memory(model_params)

# Write to file
file_descs = agent.get_reg_descs(
    [(0, total_size, fd, "/mnt/storage/checkpoint.pt".encode())],
    "FILE"
)
agent.register_memory(file_descs)

xfer_handle = agent.initialize_xfer(
    "WRITE", reg_descs, file_descs, "checkpoint_agent", b"save"
)
agent.transfer(xfer_handle)
```

### 3. Multi-GPU Aggregation
```python
# Aggregate gradients from multiple GPUs
config = nixl_agent_config(backends=["UCX"])
aggregator = nixl_agent("aggregator", config)

# Receive from all workers
for worker_id in range(num_workers):
    while not aggregator.check_remote_xfer_done(
        f"worker_{worker_id}", b"gradient"
    ):
        pass
    # Process gradient
```

## Troubleshooting

### Common Issues

1. **Plugin not found**
   ```bash
   export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins
   export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   ```

2. **UCX connection failures**
   ```bash
   export UCX_LOG_LEVEL=DEBUG
   # Check available devices
   ucx_info -d
   ```

3. **GDS not available**
   ```bash
   # Verify GDS installation
   ls /usr/local/cuda/gds/
   # Check plugin
   python check_nixl_plugins.py
   ```

4. **ETCD connection issues**
   ```bash
   # Test ETCD connectivity
   curl http://etcd-server:2379/version
   ```

## References

- **NIXL Repository**: https://github.com/ai-dynamo/nixl
- **Documentation**: https://github.com/ai-dynamo/nixl/tree/main/docs
- **Backend Guide**: https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md
- **Python API**: https://github.com/ai-dynamo/nixl/blob/main/docs/python_api.md
- **Examples**: https://github.com/ai-dynamo/nixl/tree/main/examples


