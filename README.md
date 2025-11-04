# NIXL Benchmark Suite

Benchmarks for testing [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Inference Xfer Library) - a high-performance communication library for AI inference workloads.

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Verify NIXL setup
python3 verify_nixl_setup.py

# Run storage benchmarks (GDS or POSIX)
python3 benchmarks/storage/simple_gds_storage.py --file /tmp/test.dat
```

## 📁 Repository Structure

```
nixl_benchmark/
├── benchmarks/
│   ├── single_node/           # Single-node benchmarks
│   ├── multi_node/            # Multi-node benchmarks
│   ├── storage/               # Storage benchmarks (GDS, POSIX)
│   └── utils/                 # Helper functions
│
├── verify_nixl_setup.py       # Setup verification
├── BENCHMARKS.md              # Detailed benchmark guide
├── NIXL_GUIDE.md              # Comprehensive NIXL documentation
│
├── nixl/                      # NIXL submodule (v0.7.0)
├── archive/                   # Original example scripts
└── venv/                      # Python virtual environment
```

## 📊 Available Benchmarks

### Storage Benchmarks (Recommended for Local Testing)

**GDS (GPU Direct Storage)** - GPU-to-storage transfers
```bash
python3 benchmarks/storage/simple_gds_storage.py --file /tmp/test.dat --buffer_sizes 1MB,16MB
```

**POSIX** - CPU-to-storage transfers
```bash
python3 benchmarks/storage/simple_posix_storage.py --file /tmp/test.dat --buffer_sizes 4KB,1MB
```

### Multi-Node Benchmarks

**UCX Point-to-Point** - Network transfers between nodes
```bash
# On Node 1 (target)
python3 benchmarks/multi_node/simple_ucx_p2p.py --mode target --ip 192.168.1.100

# On Node 2 (initiator)
python3 benchmarks/multi_node/simple_ucx_p2p.py --mode initiator --ip 192.168.1.100
```

See **[BENCHMARKS.md](BENCHMARKS.md)** for detailed usage.

## 📚 Documentation

- **[BENCHMARKS.md](BENCHMARKS.md)** - How to run benchmarks, command-line arguments, test scenarios
- **[NIXL_GUIDE.md](NIXL_GUIDE.md)** - Complete NIXL architecture, backends, API reference

## 🔧 Setup

### Prerequisites
- Python 3.7+
- PyTorch with CUDA support (for GPU benchmarks)
- NIXL library
- NVIDIA GDS (for GDS benchmarks)

### Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install NIXL
pip install nixl

# Verify installation
python3 verify_nixl_setup.py
```

## 🎯 What is NIXL?

NIXL (NVIDIA Inference Xfer Library) is a high-performance communication library designed for AI inference workloads. It provides:

- **Unified API** for different memory types (CPU, GPU, Storage)
- **Multiple backends** (UCX, GDS, POSIX, GPUNetIO, Mooncake, etc.)
- **High performance** with RDMA, GPU Direct, and optimized transfers
- **Flexibility** through modular plugin architecture

### Common Use Cases
- LLM KV-cache sharing across nodes
- Model checkpoint loading/saving
- Distributed inference pipelines
- GPU-to-storage offloading

## 🏗️ Building NIXL from Source

The `nixl/` directory contains the NIXL source as a git submodule (tag v0.7.0).

For build instructions, see the [original setup notes](archive/old_setup/) or the [official NIXL repository](https://github.com/ai-dynamo/nixl).

## 📦 What's in the Archive?

The `archive/` directory contains original example scripts and setup files from before the repository reorganization:

- **archive/old_scripts/** - Original NIXL examples (with_nxl_ucx.py, nixl_gds_example.py, etc.)
- **archive/old_setup/** - Installation and build scripts

These are kept for reference. The new simplified benchmarks in the root directory should be used instead.

## 🤝 Contributing

Contributions are welcome! When adding new benchmarks:

1. Keep them simple and focused
2. Base them on working NIXL patterns
3. Include clear documentation
4. Test with multiple buffer sizes
5. Add proper error handling

## 📄 License

Apache License 2.0 - See individual file headers for details.

## 🔗 Links

- **NIXL Repository**: https://github.com/ai-dynamo/nixl
- **This Fork**: https://github.com/baladithyab/nixl_benchmark

