#!/usr/bin/env python3
"""Data Profiling CLI entry point."""

import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from presentation.cli.data_profiling_cli import app

if __name__ == "__main__":
    app()