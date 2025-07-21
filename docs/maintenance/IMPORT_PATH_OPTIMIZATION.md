# Import Path Optimization Guide

## Overview

After the comprehensive domain reorganization, import paths need to be updated to reflect the new clean architecture structure. This document provides guidance for maintaining and optimizing imports.

## New Import Structure

### Core Packages
```python
# Domain layer - pure business logic
from src.packages.core.domain.entities import Anomaly, Dataset, Detector
from src.packages.core.domain.value_objects import AnomalyScore, ContaminationRate
from src.packages.core.application.services import DetectionService

# Anomaly detection domain - consolidated algorithms
from src.packages.anomaly_detection.core.domain import AnomalyDetector
from src.packages.anomaly_detection.algorithms import IsolationForestAdapter
from src.packages.anomaly_detection.services import EnhancedDetectionService

# Infrastructure adapters
from src.packages.infrastructure.adapters import PyODAdapter, DatabaseAdapter
from src.packages.infrastructure.config import Settings, create_container

# User interfaces
from src.packages.interfaces.api import create_app
from src.packages.interfaces.cli import PynomaCLI
from src.packages.interfaces.sdk.python import AnomalyDetectionClient
```

### Application Services
```python
# Machine learning workflows
from src.packages.machine_learning.application.services import (
    TrainingService,
    ModelLifecycleService,
    AutoMLService
)

# Data platform services
from src.packages.data_platform.application.services import (
    DataProcessingService,
    DataQualityService
)

# Enterprise features
from src.packages.enterprise.application.services import (
    MultiTenantService,
    GovernanceService
)
```

## Migration Strategy

### Phase 1: Critical Imports (Immediate)
Focus on imports that break core functionality:

1. **Domain Entity Imports**
   ```python
   # Old (broken)
   from anomaly_detection.domain.entities.anomaly import Anomaly
   
   # New (correct)
   from src.packages.core.domain.entities.anomaly import Anomaly
   ```

2. **Algorithm Adapter Imports**
   ```python
   # Old (scattered)
   from anomaly_detection.algorithms.adapters.pyod_adapter import PyODAdapter
   
   # New (consolidated)
   from src.packages.anomaly_detection.adapters.pyod_adapter import PyODAdapter
   ```

### Phase 2: Service Layer Imports (Secondary)
Update application service imports:

1. **Detection Services**
   ```python
   # Old
   from anomaly_detection.application.services.detection_service import DetectionService
   
   # New
   from src.packages.anomaly_detection.application.services.detection_service import DetectionService
   ```

2. **Infrastructure Services**
   ```python
   # Old
   from anomaly_detection.infrastructure.persistence import ModelRepository
   
   # New
   from src.packages.infrastructure.persistence.repositories import ModelRepository
   ```

### Phase 3: Test Imports (Final)
Update test file imports systematically:

1. **Unit Tests**
   ```python
   # Update test imports to match new structure
   from src.packages.core.domain.entities.anomaly import Anomaly
   from src.packages.testing.fixtures import sample_dataset
   ```

2. **Integration Tests**
   ```python
   # Update integration test imports
   from src.packages.anomaly_detection.core.application.services import DetectionService
   from src.packages.infrastructure.adapters.pyod_adapter import PyODAdapter
   ```

## Automated Import Fixing

### Using sed (Linux/macOS)
```bash
# Fix core domain imports
find . -name "*.py" -exec sed -i 's/from anomaly_detection\.domain\.entities/from src.packages.core.domain.entities/g' {} \;

# Fix anomaly detection imports
find . -name "*.py" -exec sed -i 's/from anomaly_detection\.algorithms/from src.packages.anomaly_detection.algorithms/g' {} \;

# Fix infrastructure imports
find . -name "*.py" -exec sed -i 's/from anomaly_detection\.infrastructure/from src.packages.infrastructure/g' {} \;
```

### Using Python Script
```python
#!/usr/bin/env python3
"""Automated import path updater."""

import os
import re
from pathlib import Path

def update_imports(file_path):
    """Update imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define import mappings
    mappings = {
        r'from anomaly_detection\.domain\.entities': 'from src.packages.core.domain.entities',
        r'from anomaly_detection\.domain\.value_objects': 'from src.packages.core.domain.value_objects',
        r'from anomaly_detection\.algorithms': 'from src.packages.anomaly_detection.algorithms',
        r'from anomaly_detection\.infrastructure': 'from src.packages.infrastructure',
        r'from anomaly_detection\.application\.services': 'from src.packages.services.application.services',
    }
    
    # Apply mappings
    for old_pattern, new_import in mappings.items():
        content = re.sub(old_pattern, new_import, content)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Update all Python files in the project."""
    for py_file in Path('.').rglob('*.py'):
        if 'site-packages' not in str(py_file) and '.venv' not in str(py_file):
            try:
                update_imports(py_file)
                print(f"Updated: {py_file}")
            except Exception as e:
                print(f"Failed to update {py_file}: {e}")

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Use Absolute Imports
```python
# Good - absolute import
from src.packages.core.domain.entities.anomaly import Anomaly

# Avoid - relative imports can break with reorganization
from ...domain.entities.anomaly import Anomaly
```

### 2. Import at Package Level
```python
# Good - clean package-level imports
from src.packages.anomaly_detection import AnomalyDetector

# Avoid - deep module imports
from src.packages.anomaly_detection.core.domain.entities.detector import AnomalyDetector
```

### 3. Use Type Checking Imports
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.packages.core.domain.entities import Dataset
    from src.packages.infrastructure.adapters import PyODAdapter
```

### 4. Lazy Imports for Optional Dependencies
```python
def get_pytorch_adapter():
    """Lazy import PyTorch adapter."""
    try:
        from src.packages.anomaly_detection.adapters.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter
    except ImportError:
        raise ImportError("PyTorch adapter requires torch to be installed")
```

## Validation

### Check Import Health
```bash
# Validate imports in a file
python -m py_compile src/packages/core/domain/entities/anomaly.py

# Check all Python files
find src/ -name "*.py" -exec python -m py_compile {} \;

# Use mypy for import checking
mypy src/packages/core/
```

### Test Import Performance
```python
import time
import importlib

def test_import_time(module_name):
    """Test import performance."""
    start = time.time()
    importlib.import_module(module_name)
    end = time.time()
    print(f"{module_name}: {(end - start) * 1000:.2f}ms")

# Test key packages
test_import_time('src.packages.core')
test_import_time('src.packages.anomaly_detection')
test_import_time('src.packages.infrastructure')
```

## Common Issues and Solutions

### 1. Circular Import Errors
```python
# Problem: Circular imports between domain and infrastructure
# Solution: Use dependency inversion and interfaces

# In domain layer - define interface
from typing import Protocol

class DetectorRepository(Protocol):
    def save(self, detector: Detector) -> None: ...

# In infrastructure layer - implement interface  
from src.packages.core.domain.interfaces import DetectorRepository

class SqlDetectorRepository(DetectorRepository):
    def save(self, detector: Detector) -> None:
        # Implementation
        pass
```

### 2. Missing Dependencies
```python
# Problem: Missing optional dependencies
# Solution: Graceful degradation

try:
    from src.packages.anomaly_detection.adapters.pytorch_adapter import PyTorchAdapter
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchAdapter = None

def create_detector(algorithm_type: str):
    if algorithm_type == "pytorch" and not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch adapter requires 'torch' to be installed")
    # ... rest of implementation
```

### 3. Test Import Failures
```python
# Problem: Test imports fail after reorganization
# Solution: Update test configuration and fixtures

# In conftest.py
import sys
from pathlib import Path

# Add src to Python path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Update test imports
from packages.core.domain.entities import Anomaly
from packages.testing.fixtures import sample_dataset
```

## Monitoring Import Health

### CI Integration
Add import validation to your CI pipeline:

```yaml
# In .github/workflows/main-ci.yml
- name: Validate imports
  run: |
    echo "Checking import syntax..."
    find src/ -name "*.py" -exec python -m py_compile {} \;
    
    echo "Checking import performance..."
    python scripts/performance_validation.py
```

### Pre-commit Hook
```yaml
# In .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-imports
        name: Validate Python imports
        entry: python -m py_compile
        language: system
        files: \.py$
```

## Summary

The domain reorganization has created a clean, maintainable architecture. Import path updates are a one-time migration cost that will pay dividends in:

1. **Clarity**: Clear domain boundaries make code easier to understand
2. **Maintainability**: Changes are isolated to specific domains
3. **Testability**: Domain separation enables better unit testing
4. **Scalability**: New features can be added without affecting existing domains

Follow this guide to systematically update imports while maintaining the benefits of our clean architecture.