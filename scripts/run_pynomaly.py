#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pynomaly.presentation.cli.app import app

if __name__ == "__main__":
    app()
