#!/usr/bin/env python3
"""Test script for hierarchical dependency validation"""

# Test legitimate imports (should be allowed)
# Layer 4 -> Layer 3: anomaly_detection -> machine_learning
try:
    from ai.machine_learning.domain.services import AutoMLService
    print("✅ VALID: anomaly_detection -> ai.machine_learning (Layer 4 -> Layer 3)")
except ImportError:
    print("❌ Could not import ai.machine_learning.domain.services.AutoMLService")

# Layer 4 -> Layer 2: anomaly_detection -> data_platform 
try:
    from data.data_platform.profiling.services import ProfilingService
    print("✅ VALID: anomaly_detection -> data.data_platform (Layer 4 -> Layer 2)")
except ImportError:
    print("❌ Could not import data.data_platform.profiling.services.ProfilingService")

print("\nThis file tests legitimate cross-layer imports that should be allowed")
print("according to the hierarchical dependency architecture.")