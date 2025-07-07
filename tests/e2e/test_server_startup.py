#!/usr/bin/env python3
"""Test server startup"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    # Test if the main API module can be imported
    from pynomaly.presentation.api.app import app
    print("✅ API app import successful")
    
    # Test if FastAPI is available
    from fastapi import FastAPI
    print("✅ FastAPI import successful")
    
    # Test if the app is properly initialized
    if hasattr(app, 'routes'):
        print(f"✅ App has {len(app.routes)} routes configured")
    else:
        print("❌ App routes not found")
        
except Exception as e:
    print(f"❌ Error testing server startup: {e}")
    import traceback
    traceback.print_exc()