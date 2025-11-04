# Multi-Node Agent Setup Guide

This guide explains how to set up NIXL agents on multiple nodes for distributed testing and benchmarking.

## Overview

NIXL supports multi-node communication through network backends like UCX, GPUNetIO, and Libfabric. This guide focuses on setting up a two-node configuration for testing point-to-point transfers.

## Prerequisites

### Hardware Requirements
- **Minimum**: 2 nodes with network connectivity (1 GbE or better)
- **Recommended**: 2 nodes with NVIDIA GPUs and high-speed network (InfiniBand, RoCE, or 10+ GbE)

### Software Requirements
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.8+
- NVIDIA GPU drivers (for GPU transfers)
- CUDA Toolkit 12.0+ (for GPU transfers)
- Network connectivity between nodes

## Node Setup

### Step 1: Clone Repository on Both Nodes

```bash
# On both Node 1 and Node 2
git clone https://github.com/baladithyab/nixl_benchmark.git
cd nixl_benchmark
```

### Step 2: Set Up Python Environment

```bash
# On both nodes
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124  # For GPU support
pip install nixl
```

### Step 3: Verify NIXL Installation

```bash
# On both nodes
source venv/bin/activate
python3 verify_nixl_setup.py
```

Expected output:
```
✓ NIXL version: 0.7.0
✓ Available plugins: ['UCX', 'GDS', 'GDS_MT', 'POSIX', ...]
✓ PyTorch version: 2.6.0+cu124
✓ CUDA available: True
✓ GPU count: 8
```

### Step 4: Network Configuration

#### Get IP Addresses

```bash
# On Node 1
hostname -I
# Example output: 192.168.1.100 ...

# On Node 2
hostname -I
# Example output: 192.168.1.101 ...
```

#### Test Network Connectivity

```bash
# From Node 1 to Node 2
ping -c 3 192.168.1.101

# From Node 2 to Node 1
ping -c 3 192.168.1.100
```

#### Open Firewall Ports (if needed)

```bash
# On both nodes - allow TCP/UDP on port 5555 (default benchmark port)
sudo ufw allow 5555/tcp
sudo ufw allow 5555/udp

# Or disable firewall for testing (not recommended for production)
sudo ufw disable
```

## Multi-Node Benchmark Testing

### UCX Point-to-Point Benchmark

This is the primary benchmark for testing network transfers between nodes.

#### Configuration

**Node 1 (Target)**: Receives data
**Node 2 (Initiator)**: Sends data

#### Basic CPU-to-CPU Transfer

```bash
# On Node 1 (Target) - IP: 192.168.1.100
cd /home/ubuntu/nixl_benchmark
source venv/bin/activate
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 192.168.1.100 \
  --port 5555

# On Node 2 (Initiator) - IP: 192.168.1.101
cd /home/ubuntu/nixl_benchmark
source venv/bin/activate
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator \
  --ip 192.168.1.100 \
  --port 5555 \
  --buffer_sizes 1MB,16MB,64MB \
  --iterations 100
```

#### GPU-to-GPU Transfer

```bash
# On Node 1 (Target)
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 192.168.1.100 \
  --port 5555 \
  --cuda \
  --gpu_id 0

# On Node 2 (Initiator)
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator \
  --ip 192.168.1.100 \
  --port 5555 \
  --buffer_sizes 1MB,16MB,256MB,1GB \
  --iterations 100 \
  --cuda \
  --gpu_id 0
```

### Expected Output

**Target Node**:
```
================================================================================
UCX P2P Benchmark - TARGET MODE
================================================================================
Listening on: 192.168.1.100:5555
Memory: CPU (DRAM)
Waiting for initiator connection...
Connected to initiator!
Ready to receive transfers...
```

**Initiator Node**:
```
================================================================================
UCX P2P Benchmark - INITIATOR MODE
================================================================================
Target: 192.168.1.100:5555
Memory: CPU (DRAM)
Buffer sizes: 1.0 MB, 16.0 MB, 64.0 MB
Iterations: 100 (warmup: 10)
================================================================================

Testing 1.0 MB...
  Bandwidth: 8.45 GB/s, Latency: 118.3 μs (mean)

Testing 16.0 MB...
  Bandwidth: 9.23 GB/s, Latency: 1734.2 μs (mean)

...
```

## Troubleshooting

### Connection Issues

#### "Connection refused" or "Connection timeout"

**Causes**:
- Target node not running or not listening
- Firewall blocking connection
- Wrong IP address or port
- Network routing issues

**Solutions**:
```bash
# 1. Verify target is running and listening
# On target node, check if process is running:
ps aux | grep simple_ucx_p2p

# 2. Check if port is open
sudo netstat -tulpn | grep 5555

# 3. Test basic connectivity
ping <target_ip>
telnet <target_ip> 5555

# 4. Check firewall
sudo ufw status
sudo iptables -L

# 5. Temporarily disable firewall for testing
sudo ufw disable
```

#### "No potential backend found to be able to do the transfer"

**Cause**: UCX backend not properly configured or network not detected

**Solutions**:
```bash
# Check UCX installation
python3 -c "from nixl._api import nixl_agent, nixl_agent_config; \
  agent = nixl_agent('test', nixl_agent_config(backends=[])); \
  print('UCX' in agent.get_plugin_list())"

# Set UCX environment variables
export UCX_TLS=tcp,self
export UCX_NET_DEVICES=all

# For InfiniBand/RoCE
export UCX_TLS=rc,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_0:1
```

### Performance Issues

#### Low Bandwidth

**Possible Causes**:
- Network congestion
- Wrong network interface selected
- CPU/GPU not on same NUMA node as NIC
- Small buffer sizes

**Solutions**:
```bash
# 1. Check network interface speed
ethtool <interface_name>  # e.g., ethtool eth0

# 2. Monitor network utilization
iftop -i <interface_name>
nload <interface_name>

# 3. Use larger buffer sizes
--buffer_sizes 64MB,256MB,1GB

# 4. Increase iterations for stable measurements
--iterations 1000 --warmup 100

# 5. For InfiniBand, verify RDMA is working
ibstat
ibv_devinfo
```

#### High Latency

**Possible Causes**:
- Network distance/hops
- CPU frequency scaling
- Interrupt handling

**Solutions**:
```bash
# 1. Set CPU governor to performance
sudo cpupower frequency-set -g performance

# 2. Pin to specific CPU cores (modify benchmark script)
taskset -c 0-7 python3 benchmarks/multi_node/simple_ucx_p2p.py ...

# 3. Enable GPU Direct RDMA (if supported)
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
```

### GPU-Specific Issues

#### "CUDA not available"

**Solutions**:
```bash
# 1. Verify CUDA installation
nvidia-smi
nvcc --version

# 2. Verify PyTorch CUDA support
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

#### "GPU X not found"

**Solutions**:
```bash
# 1. List available GPUs
nvidia-smi -L

# 2. Use correct GPU ID (0-based)
--gpu_id 0  # First GPU
--gpu_id 1  # Second GPU

# 3. Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Advanced Configuration

### Using ETCD for Metadata Exchange

For automatic agent discovery across nodes:

```bash
# Start ETCD server (on one node or separate server)
docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379

# Set environment variables on both nodes
export NIXL_ETCD_ENDPOINTS="http://<etcd_server_ip>:2379"
export NIXL_ETCD_NAMESPACE="/nixl/agents"

# Agents will automatically discover each other
```

### UCX Environment Variables

For optimal performance, configure UCX based on your network:

```bash
# TCP/IP (basic)
export UCX_TLS=tcp,self
export UCX_NET_DEVICES=all

# InfiniBand with GPU Direct RDMA
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
export UCX_NET_DEVICES=mlx5_0:1
export UCX_RC_TX_QUEUE_LEN=4096
export UCX_RC_RX_QUEUE_LEN=4096

# RoCE (RDMA over Converged Ethernet)
export UCX_TLS=rc,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_0:1

# Debug logging
export UCX_LOG_LEVEL=INFO  # or DEBUG for verbose output
```

## Benchmark Scenarios

### Scenario 1: Network Bandwidth Test

Test maximum network throughput:

```bash
# Target
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target --ip <target_ip> --cuda

# Initiator
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator --ip <target_ip> \
  --buffer_sizes 256MB,1GB,4GB \
  --iterations 50 \
  --cuda
```

### Scenario 2: Latency Test

Test minimum latency with small messages:

```bash
# Target
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target --ip <target_ip>

# Initiator
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator --ip <target_ip> \
  --buffer_sizes 4KB,64KB,1MB \
  --iterations 10000 \
  --warmup 1000
```

### Scenario 3: Mixed CPU-GPU Transfer

Test heterogeneous memory transfers:

```bash
# Target (GPU memory)
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target --ip <target_ip> --cuda --gpu_id 0

# Initiator (CPU memory)
python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator --ip <target_ip> \
  --buffer_sizes 16MB,64MB,256MB \
  --iterations 100
```

## Node Information Template

Use this template to document your node configuration:

```
Node 1:
- Hostname: node1.example.com
- IP Address: 192.168.1.100
- GPUs: 8x NVIDIA H200
- Network: InfiniBand HDR (200 Gbps)
- OS: Ubuntu 22.04
- CUDA: 12.4
- Role: Target

Node 2:
- Hostname: node2.example.com
- IP Address: 192.168.1.101
- GPUs: 8x NVIDIA H200
- Network: InfiniBand HDR (200 Gbps)
- OS: Ubuntu 22.04
- CUDA: 12.4
- Role: Initiator
```

## Next Steps

After successful multi-node setup:

1. **Run comprehensive benchmarks** with various buffer sizes
2. **Test different memory types** (DRAM, VRAM, mixed)
3. **Explore other backends** (GPUNetIO, Libfabric)
4. **Implement hierarchical transfers** (Storage → GPU → Network → GPU → Storage)
5. **Optimize for your workload** (adjust buffer sizes, iterations, backends)

## References

- [BENCHMARKS.md](BENCHMARKS.md) - Detailed benchmark documentation
- [NIXL_GUIDE.md](NIXL_GUIDE.md) - NIXL library overview
- [BackendGuide.md](BackendGuide.md) - Backend-specific documentation
- [UCX Documentation](https://openucx.readthedocs.io/) - UCX configuration guide

