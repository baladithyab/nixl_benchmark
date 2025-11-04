"""Helper functions for NIXL benchmarks"""


def format_size(bytes_val):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def parse_size(size_str):
    """Parse size string like '4KB', '1MB' to bytes."""
    size_str = size_str.upper().strip()
    # Check units from longest to shortest to avoid matching 'B' in 'KB'
    units = [
        ('TB', 1024**4),
        ('GB', 1024**3),
        ('MB', 1024**2),
        ('KB', 1024),
        ('B', 1),
    ]
    for unit, multiplier in units:
        if size_str.endswith(unit):
            return int(float(size_str[:-len(unit)]) * multiplier)
    return int(size_str)


def format_bandwidth(gbps):
    """Format bandwidth in GB/s."""
    if gbps < 0.01:
        return f"{gbps * 1000:.2f} MB/s"
    return f"{gbps:.2f} GB/s"


def format_latency(us):
    """Format latency in microseconds."""
    if us < 1:
        return f"{us * 1000:.1f} ns"
    elif us < 1000:
        return f"{us:.1f} μs"
    else:
        return f"{us / 1000:.2f} ms"


def print_header(text, width=80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f"{text:^{width}}")
    print("=" * width)


def print_results_table(results, columns):
    """
    Print results in a formatted table.
    
    Args:
        results: List of result dictionaries
        columns: List of (key, header, format_fn) tuples
    """
    if not results:
        return
    
    # Calculate column widths
    widths = []
    headers = []
    for key, header, _ in columns:
        headers.append(header)
        widths.append(max(len(header), 12))
    
    # Print header
    header_row = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
    for result in results:
        row_values = []
        for key, _, format_fn in columns:
            value = result.get(key, 0)
            if format_fn:
                formatted = format_fn(value)
            else:
                formatted = str(value)
            row_values.append(formatted)
        
        row = " | ".join(f"{v:>{w}}" for v, w in zip(row_values, widths))
        print(row)


def calculate_statistics(latencies):
    """Calculate statistics from a list of latencies."""
    if not latencies:
        return {}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        'mean': sum(sorted_latencies) / n,
        'min': sorted_latencies[0],
        'max': sorted_latencies[-1],
        'p50': sorted_latencies[n // 2],
        'p95': sorted_latencies[int(n * 0.95)],
        'p99': sorted_latencies[int(n * 0.99)],
    }

