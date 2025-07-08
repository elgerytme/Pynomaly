#!/usr/bin/env python3
"""Simple test script to isolate schema import issues."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Try importing just the schema versioning
    from pynomaly.schemas.versioning import SchemaVersion
    print("✓ Versioning import successful")
    
    # Try importing validation
    from pynomaly.schemas.validation import validate_schema_compatibility
    print("✓ Validation import successful")
    
    # Try importing analytics base
    from pynomaly.schemas.analytics.base import MetricFrame
    print("✓ Analytics base import successful")
    
    # Try importing schema version directly
    sys.path.insert(0, str(Path(__file__).parent / "src" / "pynomaly" / "schemas"))
    from pynomaly.schemas import SCHEMA_VERSION
    print(f"✓ Schema version import successful: {SCHEMA_VERSION}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
