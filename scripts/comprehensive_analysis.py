#!/usr/bin/env python3
"""
Comprehensive Static Analysis Tool

Entry point for the comprehensive static analysis system that provides
compiler-level analysis for Python code.
"""

import sys
import os
from pathlib import Path

# Add the comprehensive_analysis package to the path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_analysis.main import main

if __name__ == "__main__":
    main()