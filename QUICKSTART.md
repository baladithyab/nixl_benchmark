# Multi-Node Benchmark Quick Start Guide

This guide will help you quickly set up and run multi-node benchmarks between two nodes.

## Current Node Information

**Node 1 (This Node)**:
- IP Address: `172.31.42.78`
- Location: `/home/ubuntu/nixl_benchmark`

**Node 2 (Second Node)**:
- IP Address: `<TO BE DETERMINED>`
- Location: `/home/ubuntu/nixl_benchmark`

---

## Setup Instructions

### On Node 1 (This Node - 172.31.42.78)

```bash
cd /home/ubuntu/nixl_benchmark
git pull  # Get latest changes
chmod +x setup_node.sh
./setup_node.sh
```

### On Node 2 (Second Node)

```bash
# Clone the repository
git clone https://github.com/baladithyab/nixl_benchmark.git
cd nixl_benchmark

# Run setup
chmod +x setup_node.sh
./setup_node.sh

# Note the IP address shown at the end
```

---

## Test Network Connectivity

After setup, test connectivity between nodes:

```bash
# From Node 1 to Node 2
ping -c 3 <NODE2_IP>

# From Node 2 to Node 1
ping -c 3 172.31.42.78
```

If ping fails, check firewall settings:

```bash
# Allow port 5555 (benchmark port)
sudo ufw allow 5555/tcp
sudo ufw allow 5555/udp

# Or temporarily disable firewall for testing
sudo ufw disable
```

---

## Running Benchmarks

### Test 1: CPU-to-CPU Transfer (Basic)

**On Node 1 (Target - Receiver)**:
```bash
cd /home/ubuntu/nixl_benchmark
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555
```

**On Node 2 (Initiator - Sender)**:
```bash
cd /home/ubuntu/nixl_benchmark
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator \
  --ip 172.31.42.78 \
  --port 5555 \
  --buffer_sizes 1MB,16MB,64MB,256MB \
  --iterations 100
```

### Test 2: GPU-to-GPU Transfer

**On Node 1 (Target)**:
```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555 \
  --cuda \
  --gpu_id 0
```

**On Node 2 (Initiator)**:
```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator \
  --ip 172.31.42.78 \
  --port 5555 \
  --buffer_sizes 1MB,16MB,256MB,1GB \
  --iterations 100 \
  --cuda \
  --gpu_id 0
```

### Test 3: Large Buffer Bandwidth Test

**On Node 1 (Target)**:
```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555 \
  --cuda
```

**On Node 2 (Initiator)**:
```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator \
  --ip 172.31.42.78 \
  --port 5555 \
  --buffer_sizes 256MB,1GB,4GB \
  --iterations 50 \
  --cuda
```

### Test 4: Latency Test (Small Buffers)

**On Node 1 (Target)**:
```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555
```

**On Node 2 (Initiator)**:
```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode initiator \
  --ip 172.31.42.78 \
  --port 5555 \
  --buffer_sizes 4KB,64KB,1MB \
  --iterations 1000 \
  --warmup 100
```

---

## Expected Output

### Target Node (Receiver)
```
INFO:__main__:Target listening on 172.31.42.78:5555
INFO:__main__:Waiting for initiator to connect...
INFO:__main__:Connected to initiator
INFO:__main__:✓ Received 1.0 MB - data verified
INFO:__main__:✓ Received 16.0 MB - data verified
INFO:__main__:✓ Received 64.0 MB - data verified
INFO:__main__:Target completed all transfers
```

### Initiator Node (Sender)
```
================================================================================
UCX Point-to-Point Benchmark (Initiator)
================================================================================
Memory: CPU (DRAM)
Target: 172.31.42.78:5555
Iterations: 100 (warmup: 10)
================================================================================

Testing 1.0 MB...
  ✓ 8.45 GB/s, 118.3 μs (mean)

Testing 16.0 MB...
  ✓ 9.23 GB/s, 1734.2 μs (mean)

...

================================================================================
RESULTS SUMMARY
================================================================================
Size       | Bandwidth    | Mean (μs)  | P50 (μs)   | P95 (μs)   | P99 (μs)
--------------------------------------------------------------------------------
1.0 MB     |    8.45 GB/s |      118.3 |      115.2 |      125.8 |      132.1
16.0 MB    |    9.23 GB/s |     1734.2 |     1720.5 |     1850.3 |     1920.7
...
================================================================================
```

---

## Troubleshooting

### "Connection refused"
- Make sure target node is running first
- Check firewall: `sudo ufw status`
- Verify IP address is correct

### "No potential backend found"
- Set UCX environment variables:
  ```bash
  export UCX_TLS=tcp,self
  export UCX_NET_DEVICES=all
  ```

### "CUDA not available"
- Verify GPU: `nvidia-smi`
- Check PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### Low bandwidth
- Use larger buffers: `--buffer_sizes 256MB,1GB,4GB`
- Check network speed: `ethtool <interface>`
- For InfiniBand, set: `export UCX_TLS=rc,cuda_copy,cuda_ipc`

---

## Advanced Options

### Custom Buffer Sizes
```bash
--buffer_sizes 4KB,1MB,16MB,256MB,1GB,4GB
```

### More Iterations for Stable Results
```bash
--iterations 1000 --warmup 100
```

### Different GPU
```bash
--cuda --gpu_id 1  # Use GPU 1 instead of GPU 0
```

### Different Port
```bash
--port 6000  # Use port 6000 instead of 5555
```

---

## Next Steps

After successful testing:

1. **Document your results** - Save bandwidth and latency measurements
2. **Test different scenarios** - Try CPU→GPU, GPU→CPU transfers
3. **Optimize UCX settings** - See AGENTS.md for UCX environment variables
4. **Run comprehensive benchmarks** - See BENCHMARKS.md for detailed tests

---

## Files Reference

- **AGENTS.md** - Comprehensive multi-node setup guide
- **BENCHMARKS.md** - Detailed benchmark documentation
- **NIXL_GUIDE.md** - NIXL library overview
- **setup_node.sh** - Automated setup script
- **verify_nixl_setup.py** - Installation verification script

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review AGENTS.md for detailed solutions
3. Verify setup with: `python3 verify_nixl_setup.py`
4. Check NIXL logs for detailed error messages

