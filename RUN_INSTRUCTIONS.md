# 🚀 Ready-to-Run Multi-Node Benchmark Instructions

## ✅ Code Committed and Pushed!

All code has been committed to the repository. You can now pull and run on both nodes.

---

## 📍 Node Information

- **Node 1 (This Node)**: `172.31.42.78`
- **Node 2 (Second Node)**: `<Get IP after setup>`

---

## 🔧 Setup Instructions

### On Node 1 (This Node - 172.31.42.78)

```bash
cd /home/ubuntu/nixl_benchmark
git pull
./setup_node.sh
```

### On Node 2 (Second Node)

```bash
# Clone the repo
git clone https://github.com/baladithyab/nixl_benchmark.git
cd nixl_benchmark

# Run setup
./setup_node.sh

# Note the IP address displayed at the end!
```

---

## 🧪 Test 1: CPU-to-CPU Transfer (Start Here!)

This is the simplest test to verify connectivity.

### Step 1: Start Target on Node 1

```bash
cd /home/ubuntu/nixl_benchmark
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555
```

**Expected output:**
```
INFO:__main__:Target listening on 172.31.42.78:5555
INFO:__main__:Waiting for initiator to connect...
```

### Step 2: Start Initiator on Node 2

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

**Expected output:**
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
```

---

## 🎮 Test 2: GPU-to-GPU Transfer

After CPU test succeeds, try GPU transfer.

### Step 1: Start Target on Node 1

```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555 \
  --cuda \
  --gpu_id 0
```

### Step 2: Start Initiator on Node 2

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

---

## 🔥 Test 3: High Bandwidth Test (Large Buffers)

Test maximum throughput with large buffers.

### Step 1: Target on Node 1

```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555 \
  --cuda
```

### Step 2: Initiator on Node 2

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

---

## ⚡ Test 4: Low Latency Test (Small Buffers)

Test minimum latency with small messages.

### Step 1: Target on Node 1

```bash
source venv/bin/activate

python3 benchmarks/multi_node/simple_ucx_p2p.py \
  --mode target \
  --ip 172.31.42.78 \
  --port 5555
```

### Step 2: Initiator on Node 2

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

## 🛠️ Troubleshooting

### Connection Issues

If you get "Connection refused":

```bash
# On both nodes - allow port 5555
sudo ufw allow 5555/tcp
sudo ufw allow 5555/udp

# Or disable firewall temporarily
sudo ufw disable

# Test connectivity
ping -c 3 172.31.42.78  # From Node 2 to Node 1
```

### UCX Backend Issues

If you get "No potential backend found":

```bash
# Set UCX environment variables
export UCX_TLS=tcp,self
export UCX_NET_DEVICES=all

# Then run the benchmark again
```

### PyTorch CUDA Issues

If CUDA is not available:

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA if needed
source venv/bin/activate
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 📊 What to Expect

### Good Results
- **CPU-to-CPU**: 5-10 GB/s (depends on network)
- **GPU-to-GPU**: 10-100+ GB/s (depends on network and GPU Direct RDMA)
- **Latency**: 50-200 μs for small buffers

### If Results Are Low
1. Check network speed: `ethtool <interface>`
2. Use larger buffers for bandwidth tests
3. For InfiniBand/RoCE, set UCX for RDMA:
   ```bash
   export UCX_TLS=rc,cuda_copy,cuda_ipc
   export UCX_NET_DEVICES=mlx5_0:1
   ```

---

## 📁 Files Created

- ✅ `setup_node.sh` - Automated setup script
- ✅ `QUICKSTART.md` - Detailed quick start guide
- ✅ `AGENTS.md` - Comprehensive multi-node guide
- ✅ `benchmarks/multi_node/simple_ucx_p2p.py` - Benchmark script
- ✅ `verify_nixl_setup.py` - Verification script

---

## 🎯 Quick Command Reference

### Setup
```bash
./setup_node.sh
```

### Verify Installation
```bash
source venv/bin/activate
python3 verify_nixl_setup.py
```

### Get IP Address
```bash
hostname -I | awk '{print $1}'
```

### Test Connectivity
```bash
ping -c 3 <other_node_ip>
```

### Run Target (Receiver)
```bash
source venv/bin/activate
python3 benchmarks/multi_node/simple_ucx_p2p.py --mode target --ip 172.31.42.78 --port 5555
```

### Run Initiator (Sender)
```bash
source venv/bin/activate
python3 benchmarks/multi_node/simple_ucx_p2p.py --mode initiator --ip 172.31.42.78 --port 5555 --buffer_sizes 1MB,16MB,64MB --iterations 100
```

---

## 🚀 Next Steps

1. **Run Test 1** (CPU-to-CPU) to verify basic connectivity
2. **Run Test 2** (GPU-to-GPU) to test GPU transfers
3. **Run Test 3** (Large buffers) to measure peak bandwidth
4. **Run Test 4** (Small buffers) to measure latency
5. **Document results** and compare with expected performance

---

## 📚 Additional Documentation

- **QUICKSTART.md** - More detailed quick start guide
- **AGENTS.md** - Comprehensive setup and troubleshooting
- **BENCHMARKS.md** - All available benchmarks
- **NIXL_GUIDE.md** - NIXL library documentation

---

## ✨ Summary

**What's Ready:**
- ✅ Multi-node benchmark code committed and pushed
- ✅ Setup script for easy installation
- ✅ Ready-to-run commands for 4 test scenarios
- ✅ Troubleshooting guide for common issues

**What You Need to Do:**
1. Pull code on Node 1: `git pull`
2. Clone repo on Node 2: `git clone https://github.com/baladithyab/nixl_benchmark.git`
3. Run `./setup_node.sh` on both nodes
4. Start with Test 1 (CPU-to-CPU)

**Good luck! 🎉**

