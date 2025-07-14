#!/usr/bin/env python3
"""Data Science CLI entry point."""

import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from presentation.cli.statistical_analysis_cli import app

if __name__ == "__main__":
    app()