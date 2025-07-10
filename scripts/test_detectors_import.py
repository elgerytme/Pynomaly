#!/usr/bin/env python3

import sys
import traceback

try:
    print("Testing detectors endpoint import...")
    from pynomaly.presentation.api.endpoints.detectors import router
    print("✅ Detectors endpoint imports successfully")
    print(f"Router has {len(router.routes)} routes")
    
    # List the routes
    for route in router.routes:
        print(f"  - {route.methods}: {route.path}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)