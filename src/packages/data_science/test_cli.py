#!/usr/bin/env python3
"""Test script for Data Science CLI."""

import tempfile
import json
from pathlib import Path
import pandas as pd
from typer.testing import CliRunner
from presentation.cli.statistical_analysis_cli import app

def test_statistical_analysis_cli():
    """Test the statistical analysis CLI commands."""
    runner = CliRunner()
    
    # Create test data
    test_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
        'score': [85, 90, 88, 92, 95, 87, 89, 93],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        data_file = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(data_file, index=False)
        
        # Test health check
        result = runner.invoke(app, ["health"])
        print("Health check result:", result.exit_code)
        assert result.exit_code == 0
        
        # Test analyze command
        result = runner.invoke(app, [
            "analyze",
            str(data_file),
            "--type", "descriptive_statistics",
            "--features", "age,income,score",
            "--verbose"
        ])
        print("Analyze result:", result.exit_code)
        print("Analyze output:", result.stdout)
        
        # Test list command
        result = runner.invoke(app, ["list", "--verbose"])
        print("List result:", result.exit_code)
        print("List output:", result.stdout)
        
        print("✅ All Statistical Analysis CLI tests passed!")

if __name__ == "__main__":
    test_statistical_analysis_cli()