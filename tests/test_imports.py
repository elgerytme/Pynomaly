#!/usr/bin/env python3
"""Test imports to debug conftest issues."""

try:
    from pynomaly.domain.entities import Dataset, Detector, DetectionResult
    print('✓ Successfully imported domain entities')
except Exception as e:
    print(f'✗ Import error: {e}')
    import traceback
    traceback.print_exc()

try:
    from pynomaly.application.services.performance_benchmarking_service import PerformanceBenchmarkingService
    print('✓ Successfully imported PerformanceBenchmarkingService')
except Exception as e:
    print(f'✗ Import error: {e}')
    import traceback
    traceback.print_exc()
