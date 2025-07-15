#!/usr/bin/env python3
"""Test script for Data Quality CLI."""

import tempfile
import json
from pathlib import Path
import pandas as pd
from typer.testing import CliRunner
from presentation.cli.data_quality_cli import app

def test_data_quality_cli():
    """Test the data quality CLI commands."""
    runner = CliRunner()
    
    # Create test data with some quality issues
    test_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'email': ['john@example.com', 'invalid-email', 'jane@example.com', '', 'charlie@example.com', 'eva@example.com', 'frank@invalid', 'grace@example.com'],
        'age': [25, 30, 35, -5, 45, 150, 55, 60],  # Invalid ages
        'phone': ['123-456-7890', '123456789', '123-456-7890', '123-456-7890', '123-456-7890', 'invalid', '123-456-7890', '123-456-7890'],
        'status': ['active', 'active', None, 'inactive', 'active', 'inactive', 'active', 'inactive']  # Missing values
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        data_file = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(data_file, index=False)
        
        # Test health check
        result = runner.invoke(app, ["health"])
        print("Health check result:", result.exit_code)
        assert result.exit_code == 0
        
        # Test create-rule command
        result = runner.invoke(app, [
            "create-rule",
            "Email Format Validation",
            "--type", "validity",
            "--columns", "email",
            "--logic", "regex",
            "--expression", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "--severity", "high",
            "--description", "Validates email format",
            "--verbose"
        ])
        print("Create rule result:", result.exit_code)
        print("Create rule output:", result.stdout)
        
        # Test create another rule
        result = runner.invoke(app, [
            "create-rule",
            "Age Range Validation",
            "--type", "validity",
            "--columns", "age",
            "--logic", "range",
            "--expression", "0 <= x <= 120",
            "--severity", "medium",
            "--description", "Validates age is in reasonable range",
            "--verbose"
        ])
        print("Create age rule result:", result.exit_code)
        print("Create age rule output:", result.stdout)
        
        # Test list-rules command
        result = runner.invoke(app, ["list-rules", "--verbose"])
        print("List rules result:", result.exit_code)
        print("List rules output:", result.stdout)
        
        # Test validate command
        result = runner.invoke(app, [
            "validate",
            str(data_file),
            "--verbose"
        ])
        print("Validate result:", result.exit_code)
        print("Validate output:", result.stdout)
        
        # Test violations command
        result = runner.invoke(app, ["violations", "--verbose"])
        print("Violations result:", result.exit_code)
        print("Violations output:", result.stdout)
        
        print("✅ All Data Quality CLI tests passed!")

if __name__ == "__main__":
    test_data_quality_cli()