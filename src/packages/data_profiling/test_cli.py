#!/usr/bin/env python3
"""Test script for Data Profiling CLI."""

import tempfile
import json
from pathlib import Path
import pandas as pd
from typer.testing import CliRunner
from presentation.cli.data_profiling_cli import app

def test_data_profiling_cli():
    """Test the data profiling CLI commands."""
    runner = CliRunner()
    
    # Create test data
    test_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis', 'Eva Wilson', 'Frank Miller', 'Grace Lee'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com', 'eva@example.com', 'frank@example.com', 'grace@example.com'],
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'salary': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering'],
        'join_date': pd.date_range('2020-01-01', periods=8, freq='3M')
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        data_file = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(data_file, index=False)
        
        # Test health check
        result = runner.invoke(app, ["health"])
        print("Health check result:", result.exit_code)
        assert result.exit_code == 0
        
        # Test profile command
        result = runner.invoke(app, [
            "profile",
            str(data_file),
            "--strategy", "full",
            "--patterns",
            "--statistics",
            "--quality",
            "--verbose"
        ])
        print("Profile result:", result.exit_code)
        print("Profile output:", result.stdout)
        
        # Test profile with sampling
        result = runner.invoke(app, [
            "profile",
            str(data_file),
            "--strategy", "sample",
            "--sample-size", "5",
            "--verbose"
        ])
        print("Sample profile result:", result.exit_code)
        print("Sample profile output:", result.stdout)
        
        # Test list command
        result = runner.invoke(app, ["list", "--verbose"])
        print("List result:", result.exit_code)
        print("List output:", result.stdout)
        
        print("✅ All Data Profiling CLI tests passed!")

if __name__ == "__main__":
    test_data_profiling_cli()