# NIXL Benchmark Suite

Simple, robust benchmarking suite for testing NIXL backends.
Based on working patterns from official NIXL examples.

## Available Benchmarks

```
nixl_benchmark/
├── simple_ucx_loopback.py   # Single-node UCX transfers (loopback)
├── simple_ucx_p2p.py        # Multi-node UCX point-to-point
├── simple_gds_storage.py    # GDS GPU-to-storage transfers
├── verify_nixl_setup.py     # Setup verification script
└── BENCHMARKS.md            # This file
```

## Philosophy

These benchmarks are intentionally **simple and focused**:
- ✅ Based on proven working patterns from NIXL examples
- ✅ Minimal dependencies and complexity
- ✅ Easy to understand and modify
- ✅ Robust error handling
- ✅ Clear performance metrics

Unlike complex benchmark frameworks, these scripts prioritize **reliability** and **clarity**.

## Prerequisites

```bash
# 1. Install NIXL (from PyPI or source)
pip install nixl

# 2. Set environment variables
export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 3. Verify installation
python3 -c "from nixl._api import nixl_agent, nixl_agent_config; print('NIXL OK')"
```

## Quick Start

### 1. Single-Node UCX Loopback

Test local transfers (agent to itself):

```bash
# CPU memory (DRAM)
python3 simple_ucx_loopback.py \
  --buffer_sizes 4KB,64KB,1MB,16MB \
  --iterations 100

# GPU memory (VRAM) - requires CUDA
python3 simple_ucx_loopback.py \
  --buffer_sizes 1MB,16MB,256MB \
  --iterations 100 \
  --cuda \
  --gpu_id 0
```

**Expected output:**
```
================================================================================
UCX Loopback Benchmark
================================================================================
Memory: CPU (DRAM)
Buffer sizes: 4.0 KB, 64.0 KB, 1.0 MB, 16.0 MB
Iterations: 100 (warmup: 10)
================================================================================

Testing 4.0 KB... ✓ 0.15 GB/s, 27.3 μs (mean)
Testing 64.0 KB... ✓ 2.1 GB/s, 30.5 μs (mean)
Testing 1.0 MB... ✓ 18.5 GB/s, 54.2 μs (mean)
Testing 16.0 MB... ✓ 45.3 GB/s, 353.1 μs (mean)
```

### 2. Multi-Node UCX Point-to-Point

Test transfers between two nodes:

**On Node 1 (Target/Receiver):**
```bash
python3 simple_ucx_p2p.py \
  --mode target \
  --ip 192.168.1.100 \
  --port 5555 \
  --buffer_sizes 1MB,16MB,256MB
```

**On Node 2 (Initiator/Sender):**
```bash
python3 simple_ucx_p2p.py \
  --mode initiator \
  --ip 192.168.1.100 \
  --port 5555 \
  --buffer_sizes 1MB,16MB,256MB \
  --iterations 100
```

**With CUDA (GPU memory):**
```bash
# Add --cuda flag on both nodes
python3 simple_ucx_p2p.py \
  --mode target \
  --ip 192.168.1.100 \
  --cuda \
  --gpu_id 0
```

**Expected output (initiator):**
```
================================================================================
UCX Point-to-Point Benchmark (Initiator)
================================================================================
Memory: CPU (DRAM)
Target: 192.168.1.100:5555
Iterations: 100 (warmup: 10)
================================================================================

Testing 1.0 MB...
  ✓ 8.5 GB/s, 117.6 μs (mean)
Testing 16.0 MB...
  ✓ 45.2 GB/s, 354.1 μs (mean)
Testing 256.0 MB...
  ✓ 92.3 GB/s, 2773.5 μs (mean)
```

### 3. GDS Storage Benchmark

Test GPU-to-Storage transfers (requires GDS plugin):

```bash
# Create test file
dd if=/dev/zero of=/mnt/nvme/test.dat bs=1M count=1024

# Run benchmark
python3 simple_gds_storage.py \
  --file /mnt/nvme/test.dat \
  --buffer_sizes 1MB,16MB,256MB \
  --iterations 50
```

**Expected output:**
```
================================================================================
GDS Storage Benchmark
================================================================================
File: /mnt/nvme/test.dat
Buffer sizes: 1.0 MB, 16.0 MB, 256.0 MB
Iterations: 50 (warmup: 5)
================================================================================

Testing 1.0 MB...
  WRITE: 3.2 GB/s, 312.5 μs (mean)
  READ:  4.1 GB/s, 243.9 μs (mean)
Testing 16.0 MB...
  WRITE: 8.5 GB/s, 1882.4 μs (mean)
  READ:  10.2 GB/s, 1568.6 μs (mean)
Testing 256.0 MB...
  WRITE: 11.3 GB/s, 22654.9 μs (mean)
  READ:  12.8 GB/s, 20000.0 μs (mean)
```

## Command-Line Arguments

### simple_ucx_loopback.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--buffer_sizes` | Comma-separated sizes (e.g., 4KB,1MB) | 4KB,64KB,1MB,16MB |
| `--iterations` | Number of iterations per size | 100 |
| `--warmup` | Number of warmup iterations | 10 |
| `--cuda` | Use GPU memory instead of CPU | False |
| `--gpu_id` | GPU device ID | 0 |

### simple_ucx_p2p.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `target` or `initiator` | Required |
| `--ip` | IP address (local for target, remote for initiator) | Required |
| `--port` | Port number | 5555 |
| `--buffer_sizes` | Comma-separated sizes | 1MB,16MB,256MB |
| `--iterations` | Number of iterations (initiator only) | 100 |
| `--warmup` | Number of warmup iterations | 10 |
| `--cuda` | Use GPU memory | False |
| `--gpu_id` | GPU device ID | 0 |

### simple_gds_storage.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--file` | File path for testing | Required |
| `--buffer_sizes` | Comma-separated sizes | 1MB,16MB,256MB |
| `--iterations` | Number of iterations per size | 50 |
| `--warmup` | Number of warmup iterations | 5 |

## Test Scenarios

### Scenario 1: Baseline Single-Node Performance

Establish baseline for local transfers:

```bash
# CPU baseline
python3 simple_ucx_loopback.py --buffer_sizes 4KB,1MB,16MB,256MB

# GPU baseline (if available)
python3 simple_ucx_loopback.py --buffer_sizes 1MB,16MB,256MB --cuda
```

### Scenario 2: Multi-Node Network Performance

Test network bandwidth between nodes:

```bash
# Node 1
python3 simple_ucx_p2p.py --mode target --ip 192.168.1.100

# Node 2
python3 simple_ucx_p2p.py --mode initiator --ip 192.168.1.100 \
  --buffer_sizes 1MB,16MB,256MB,1GB --iterations 100
```

### Scenario 3: Storage Performance

Test GPU-to-storage bandwidth:

```bash
python3 simple_gds_storage.py \
  --file /mnt/nvme/test.dat \
  --buffer_sizes 1MB,16MB,256MB \
  --iterations 50
```

### Scenario 4: Hierarchical Transfer (Manual)

For complex patterns (e.g., Storage → GPU → Network → GPU → Storage), you can chain the benchmarks or modify them to suit your needs.

## Performance Tuning

### UCX Backend

```bash
# Enable GPU Direct RDMA
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy

# Specify network device
export UCX_NET_DEVICES=mlx5_0:1

# Increase buffer sizes
export UCX_RC_TX_QUEUE_LEN=4096
export UCX_RC_RX_QUEUE_LEN=4096

# Enable debug logging
export UCX_LOG_LEVEL=DEBUG
```

### GDS Tuning

```bash
# Check GDS installation
ls /usr/local/cuda/gds/

# Verify cuFile configuration
cat /etc/cufile.json

# Use appropriate file system (XFS or ext4 recommended)
# Mount with direct I/O support
```

## Troubleshooting

### Plugin Not Found

```bash
export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Verify plugins
python -c "from nixl._api import nixl_agent, nixl_agent_config; \
           a = nixl_agent('test', nixl_agent_config(backends=['UCX'])); \
           print(a.get_plugin_list())"
```

### UCX Connection Issues

```bash
# Check available devices
ucx_info -d

# Enable debug logging
export UCX_LOG_LEVEL=DEBUG

# Check transport selection
export UCX_PROTO_INFO=y
```

### GPU Not Available

```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Understanding the Results

### Bandwidth
- **GB/s**: Gigabytes per second of data transferred
- Higher is better
- Limited by: network speed, PCIe bandwidth, storage speed, memory bandwidth

### Latency
- **Mean**: Average time per transfer
- **P50**: Median (50th percentile)
- **P95/P99**: Tail latencies (95th/99th percentile)
- Lower is better
- Important for understanding consistency

### Typical Performance Ranges

| Transfer Type | Expected Bandwidth | Expected Latency |
|---------------|-------------------|------------------|
| CPU-to-CPU (local) | 10-20 GB/s | 5-30 μs |
| GPU-to-GPU (local) | 50-100 GB/s | 10-50 μs |
| GPU-to-GPU (IB network) | 80-100 GB/s | 20-100 μs |
| GPU-to-Storage (GDS) | 8-15 GB/s | 100-1000 μs |
| CPU-to-Storage (POSIX) | 2-5 GB/s | 200-2000 μs |

## Next Steps

### For Single-Node Testing
1. Run `simple_ucx_loopback.py` to establish baseline
2. Try different buffer sizes to find optimal transfer size
3. Compare CPU vs GPU memory performance

### For Multi-Node Testing
1. Ensure network connectivity between nodes
2. Run `simple_ucx_p2p.py` on both nodes
3. Test with increasing buffer sizes
4. Monitor network utilization with `iftop` or `nload`

### For Storage Testing
1. Ensure GDS is properly installed
2. Use fast storage (NVMe SSD recommended)
3. Run `simple_gds_storage.py` with various buffer sizes
4. Compare with POSIX I/O for baseline

## Extending the Benchmarks

These benchmarks are intentionally simple to make them easy to modify:

1. **Add new backends**: Copy a script and change the backend name
2. **Add verification**: Uncomment or add data verification code
3. **Add metrics**: Add timing points and calculations
4. **Chain operations**: Combine scripts for hierarchical patterns

Example: To test Mooncake backend, copy `simple_ucx_p2p.py` and change:
```python
config = nixl_agent_config(backends=["Mooncake"])
```

## References

- **NIXL Guide**: See `../NIXL_GUIDE.md` for comprehensive NIXL documentation
- **Benchmark Plan**: See `../BENCHMARK_PLAN.md` for detailed testing strategy
- **NIXL Examples**: See `../nixl/examples/python/` for more examples
- **NIXL Docs**: See `../nixl/docs/` for API documentation
