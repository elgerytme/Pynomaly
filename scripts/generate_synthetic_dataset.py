#!/usr/bin/env python3
"""
Generate a 10 GB synthetic dataset for testing purposes.
This script creates a large dataset with various data types and patterns.
"""

import csv
import json
import os
import random
import string
import sys
from datetime import datetime, timedelta


def generate_random_string(length=10):
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_timestamp():
    """Generate a random timestamp within the last year."""
    start_date = datetime.now() - timedelta(days=365)
    random_days = random.randint(0, 365)
    return start_date + timedelta(days=random_days)


def generate_row():
    """Generate a single row of synthetic data."""
    return {
        'id': random.randint(1, 10000000),
        'timestamp': generate_timestamp().isoformat(),
        'user_id': random.randint(1, 100000),
        'session_id': generate_random_string(16),
        'event_type': random.choice(['click', 'view', 'purchase', 'login', 'logout']),
        'category': random.choice(['electronics', 'clothing', 'books', 'food', 'sports']),
        'value': round(random.uniform(0.01, 999.99), 2),
        'quantity': random.randint(1, 10),
        'status': random.choice(['active', 'inactive', 'pending', 'completed']),
        'metadata': json.dumps({
            'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
            'os': random.choice(['Windows', 'macOS', 'Linux']),
            'device': random.choice(['desktop', 'mobile', 'tablet']),
            'location': random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU'])
        }),
        'description': generate_random_string(50),
        'tags': ','.join([generate_random_string(8) for _ in range(random.randint(1, 5))])
    }


def estimate_row_size():
    """Estimate the size of a single row in bytes."""
    sample_row = generate_row()
    # Convert to JSON string to estimate size
    json_str = json.dumps(sample_row)
    return len(json_str.encode('utf-8'))


def generate_synthetic_dataset(target_size_gb=10, output_file='data/synthetic_10gb.csv'):
    """Generate a synthetic dataset of approximately the target size."""
    target_size_bytes = target_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    
    # Estimate row size
    row_size = estimate_row_size()
    estimated_rows = target_size_bytes // row_size
    
    print(f"Target size: {target_size_gb} GB ({target_size_bytes:,} bytes)")
    print(f"Estimated row size: {row_size} bytes")
    print(f"Estimated rows needed: {estimated_rows:,}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate the dataset
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'id', 'timestamp', 'user_id', 'session_id', 'event_type', 
            'category', 'value', 'quantity', 'status', 'metadata', 
            'description', 'tags'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        rows_written = 0
        bytes_written = 0
        
        try:
            while bytes_written < target_size_bytes:
                row = generate_row()
                writer.writerow(row)
                rows_written += 1
                
                # Update progress every 10,000 rows
                if rows_written % 10000 == 0:
                    current_size = os.path.getsize(output_file)
                    progress = (current_size / target_size_bytes) * 100
                    print(f"Progress: {progress:.1f}% ({rows_written:,} rows, {current_size / (1024*1024*1024):.2f} GB)")
                    bytes_written = current_size
                    
                    # Check if we've reached the target
                    if current_size >= target_size_bytes:
                        break
                        
        except KeyboardInterrupt:
            print(f"\nInterrupted by user. Generated {rows_written:,} rows.")
            
    final_size = os.path.getsize(output_file)
    print(f"\nDataset generation complete!")
    print(f"Final size: {final_size / (1024*1024*1024):.2f} GB")
    print(f"Total rows: {rows_written:,}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    # Allow command line arguments for target size and output file
    target_size = 10
    output_file = 'data/synthetic_10gb.csv'
    
    if len(sys.argv) > 1:
        target_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    generate_synthetic_dataset(target_size, output_file)
