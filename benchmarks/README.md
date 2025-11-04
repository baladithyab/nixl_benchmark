# NIXL Benchmarks

Organized benchmark suite for testing NIXL backends.

## Structure

```
benchmarks/
├── single_node/           # Single-node benchmarks
│   └── simple_ucx_loopback.py
├── multi_node/            # Multi-node benchmarks
│   └── simple_ucx_p2p.py
├── storage/               # Storage benchmarks
│   └── simple_gds_storage.py
└── utils/                 # Shared utilities
    └── helpers.py
```

## Quick Start

```bash
# Single-node test
python3 -m benchmarks.single_node.simple_ucx_loopback --buffer_sizes 4KB,1MB,16MB

# Multi-node test (2 nodes required)
# Node 1:
python3 -m benchmarks.multi_node.simple_ucx_p2p --mode target --ip 192.168.1.100

# Node 2:
python3 -m benchmarks.multi_node.simple_ucx_p2p --mode initiator --ip 192.168.1.100

# Storage test
python3 -m benchmarks.storage.simple_gds_storage --file /tmp/test.dat
```

Or run directly:

```bash
cd benchmarks
python3 single_node/simple_ucx_loopback.py --buffer_sizes 4KB,1MB,16MB
python3 multi_node/simple_ucx_p2p.py --mode target --ip 192.168.1.100
python3 storage/simple_gds_storage.py --file /tmp/test.dat
```

See [../BENCHMARKS.md](../BENCHMARKS.md) for detailed documentation.

