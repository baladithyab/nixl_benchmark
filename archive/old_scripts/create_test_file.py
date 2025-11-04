#!/usr/bin/env python3

import os
import random
import time

def create_test_file(filename, size_gb=5.0):
    """
    Create a test file of specified size in GB with random data.
    
    Args:
        filename (str): Name of the output file
        size_gb (float): Size in gigabytes
    """
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    chunk_size = 64 * 1024 * 1024  # 64 MB chunks for efficient writing
    
    print(f"Creating {size_gb} GB test file: {filename}")
    print(f"Total size: {size_bytes / (1024*1024*1024):.2f} GB ({size_bytes:,} bytes)")
    print(f"Writing in {chunk_size / (1024*1024):.0f} MB chunks...")
    
    start_time = time.time()
    
    with open(filename, 'wb') as f:
        bytes_written = 0
        chunk_count = 0
        
        while bytes_written < size_bytes:
            # Calculate remaining bytes to write
            remaining = size_bytes - bytes_written
            current_chunk_size = min(chunk_size, remaining)
            
            # Generate random data for this chunk
            chunk_data = random.randbytes(current_chunk_size)
            f.write(chunk_data)
            
            bytes_written += current_chunk_size
            chunk_count += 1
            
            # Progress update every 10 chunks
            if chunk_count % 10 == 0:
                progress = (bytes_written / size_bytes) * 100
                elapsed = time.time() - start_time
                rate = bytes_written / elapsed / (1024*1024) if elapsed > 0 else 0
                print(f"Progress: {progress:.1f}% ({bytes_written / (1024*1024*1024):.2f} GB) - Rate: {rate:.1f} MB/s")
    
    total_time = time.time() - start_time
    actual_size = os.path.getsize(filename)
    
    print(f"\nFile creation completed!")
    print(f"Actual file size: {actual_size / (1024*1024*1024):.2f} GB ({actual_size:,} bytes)")
    print(f"Creation time: {total_time:.2f} seconds")
    print(f"Average write rate: {actual_size / total_time / (1024*1024):.1f} MB/s")
    print(f"File location: {os.path.abspath(filename)}")

if __name__ == "__main__":
    file_size_gb = 5.0
    test_filename = f"test_file_{file_size_gb}gb.dat"
    
    # Check if file already exists
    if os.path.exists(test_filename):
        file_size = os.path.getsize(test_filename)
        file_size_gb = file_size / (1024*1024*1024)
        print(f"File {test_filename} already exists ({file_size_gb:.2f} GB)")
        response = input("Do you want to recreate it? (y/N): ")
        if response.lower() != 'y':
            print("Using existing file.")
            exit(0)
    
    create_test_file(test_filename, file_size_gb) 