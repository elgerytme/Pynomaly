# Test Domain Isolation Guidelines

This document outlines the rules and best practices for maintaining test domain isolation in our monorepo to prevent test domain leakage.

## Overview

Test domain isolation ensures that:
- Package tests only import from their own package
- System tests don't directly import from domain packages  
- Repository-level tests don't import from specific packages

This isolation prevents tightly coupled tests, reduces test flakiness, and maintains clean architectural boundaries.

## Rules and Enforcement

### 1. Package Test Rules

**Rule**: Package tests must only import from their own package using relative imports.

**Location**: `src/packages/*/tests/**/*.py`

#### ✅ Allowed Patterns

```python
# Relative imports from the same package
from ..models import UserModel
from ...services import AuthService
from . import fixtures

# Standard library imports
import os, sys, json, logging
from pathlib import Path
from datetime import datetime

# Testing framework imports
import pytest
from unittest import TestCase
from unittest.mock import Mock, patch
from faker import Faker
from hypothesis import given
```

#### ❌ Prohibited Patterns

```python
# Absolute imports from other packages
from ai.mlops import MLModel
from data.analytics import DataProcessor
from finance.billing import BillingService

# Direct package imports
import ai.mlops.models
import data.quality.validators
```

#### 🔧 How to Fix

```python
# Before (❌)
from data.analytics import DataProcessor

# After (✅)  
from ..analytics import DataProcessor
# or if analytics is in the same package
from .analytics import DataProcessor
```

### 2. System Test Rules

**Rule**: System tests must not import directly from domain packages.

**Location**: `src/packages/system_tests/**/*.py`

#### ✅ Allowed Patterns

```python
# Public API imports (if available)
from api.client import APIClient
from shared.test_fixtures import create_test_data

# Repository-level utilities
from tests.fixtures import system_fixtures
from tests.utils import integration_helpers

# Standard library and testing frameworks
import requests, json, pytest
```

#### ❌ Prohibited Patterns

```python
# Direct domain package imports
from ai.mlops import MLModel
from data.analytics import DataProcessor
from finance.billing.services import BillingProcessor
```

#### 🔧 How to Fix

```python
# Before (❌)
from ai.mlops import MLModel

# After (✅)
# Use public APIs, test fixtures, or mock objects
response = api_client.post('/api/ml/models', data=model_data)
# or
mock_model = Mock(spec=MLModel)
```

### 3. Repository Test Rules

**Rule**: Repository-level tests must not import from `src.packages`.

**Location**: `tests/**/*.py` (repository root)

#### ✅ Allowed Patterns

```python
# Repository-level utilities
from tests.fixtures import repo_fixtures
from tests.utils import test_helpers

# Scripts and tools
from scripts.validation import validate_structure
from tools.build import BuildSystem

# Standard library and testing frameworks
import subprocess, os, json, pytest
```

#### ❌ Prohibited Patterns

```python
# Direct package imports
from src.packages.ai.mlops import something
import src.packages.data.analytics

# sys.path manipulation to access packages
sys.path.insert(0, 'src/packages/ai')
from mlops import models
```

#### 🔧 How to Fix

```python
# Before (❌)
from src.packages.ai.mlops import MLModel

# After (✅)
# Test the repository structure, not package internals
def test_mlops_package_exists():
    assert Path('src/packages/ai/mlops').exists()

# Or use subprocess to test CLI interfaces
result = subprocess.run(['python', '-m', 'ai.mlops.cli', '--help'])
assert result.returncode == 0
```

## Exceptions and Special Cases

### Allowed Exceptions

The following imports are always allowed in all test files:

#### Standard Library
```python
import os, sys, re, json, yaml, logging
import tempfile, shutil, subprocess
import asyncio, threading, multiprocessing
import uuid, hashlib, base64, pickle
import urllib, http, socket
import time, random, math, statistics
import itertools, functools, operator
import contextlib, warnings, traceback
import inspect, types, copy, weakref
```

#### Type Annotations
```python
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, date
```

#### Testing Frameworks
```python
# pytest
import pytest
from pytest import fixture, mark, raises

# unittest
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

# Testing utilities
from faker import Faker
from hypothesis import given, strategies
from freezegun import freeze_time
from testcontainers import DockerContainer
```

### Configuration-Based Exceptions

Specific exceptions can be configured in `.domain-boundaries.yaml`:

```yaml
testing:
  exceptions:
    - file: "src/packages/*/tests/conftest.py"
      pattern: "from\\s+\\.\\."
      reason: "Test configuration files may need to import package modules for fixtures"
      approved_by: "testing-team"
    
    - file: "src/packages/shared/*/tests/**/*.py"
      pattern: "from\\s+(infrastructure|interfaces)\\."
      reason: "Shared package tests may import from infrastructure and interfaces"
      approved_by: "architecture-team"
```

## Tools and Automation

### Detection Tool

Use the test domain leakage detector to find violations:

```bash
# Scan for violations
python src/packages/tools/test_domain_leakage_detector/cli.py scan

# Show suggested fixes
python src/packages/tools/test_domain_leakage_detector/cli.py scan --show-fixes

# Generate JSON report
python src/packages/tools/test_domain_leakage_detector/cli.py scan --format json --output report.json
```

### Automatic Fixing

Use the built-in fixer for common patterns:

```bash
# Preview fixes (dry run)
python src/packages/tools/test_domain_leakage_detector/cli.py fix --dry-run

# Apply fixes
python src/packages/tools/test_domain_leakage_detector/cli.py fix

# Apply fixes with verbose output
python src/packages/tools/test_domain_leakage_detector/cli.py fix --verbose
```

### Pre-commit Hook

The repository includes a pre-commit hook that automatically checks for test domain leakage:

```yaml
- id: test-domain-leakage-validator
  name: Test Domain Leakage Validator
  entry: python src/packages/tools/test_domain_leakage_detector/cli.py
  args: [scan, --strict, --format, console]
```

### CI/CD Integration

Test domain leakage validation is integrated into the CI/CD pipeline and will:
- Run on all pull requests
- Block merging if critical violations are found
- Generate detailed reports with suggested fixes
- Comment on PRs with violation details

## Best Practices

### 1. Design Tests for Isolation

```python
# ✅ Good: Test focuses on the current package
def test_user_service_creates_user():
    from ..services import UserService
    from ..models import User
    
    service = UserService()
    user = service.create_user("test@example.com")
    assert isinstance(user, User)

# ❌ Bad: Test depends on other packages
def test_user_service_with_billing():
    from ..services import UserService  
    from finance.billing import BillingService  # Cross-package import
    
    # This test is too broad and couples packages
```

### 2. Use Dependency Injection

```python
# ✅ Good: Inject dependencies for testing
class UserService:
    def __init__(self, billing_service=None):
        self.billing_service = billing_service or get_default_billing()
    
# Test with mock dependency
def test_user_service():
    mock_billing = Mock()
    service = UserService(billing_service=mock_billing)
    # Test behavior without coupling to billing package
```

### 3. Create Shared Test Fixtures

```python
# ✅ Good: Shared fixtures in the same package
# src/packages/ai/mlops/tests/fixtures.py
@pytest.fixture
def sample_model():
    from ..models import MLModel
    return MLModel(name="test-model")

# ❌ Bad: Importing fixtures from other packages  
# src/packages/ai/mlops/tests/test_training.py
from data.analytics.tests.fixtures import sample_data  # Cross-package
```

### 4. Mock External Dependencies

```python
# ✅ Good: Mock external package dependencies
@patch('external_package.ExternalService')
def test_service_integration(mock_external):
    mock_external.return_value.process.return_value = "result"
    # Test your service without depending on external_package
```

## Troubleshooting

### Common Issues

#### "Package tests importing from other packages"

**Problem**: Test file contains `from other_package import something`

**Solution**: 
1. Use relative imports: `from ..module import something`
2. If the dependency is necessary, consider if the test belongs in an integration test suite
3. Use mocking to avoid the dependency

#### "System tests importing domain packages directly"

**Problem**: System test contains `from ai.mlops import MLModel`

**Solution**:
1. Use public APIs or CLI interfaces instead
2. Create test fixtures that don't depend on package internals
3. Use HTTP requests to test API endpoints

#### "Repository tests importing from src.packages"

**Problem**: Repository test contains `from src.packages.ai.mlops import something`

**Solution**:
1. Test repository structure rather than package internals
2. Use subprocess to test CLI interfaces
3. Move the test to the appropriate package's test suite

### Getting Help

1. **Automatic Fixing**: Try the built-in fixer first
2. **Manual Review**: Check if the test should be moved to a different location
3. **Architecture Review**: Consider if the violation indicates a design issue
4. **Team Discussion**: Reach out to the architecture team for complex cases

## Integration with Existing Tools

This test domain isolation system works alongside:

- **Domain Boundary Detector**: Validates general package boundaries
- **Buck2 Build System**: Enforces boundaries at build time
- **Import Consolidation Validator**: Ensures clean import patterns
- **Pre-commit Hooks**: Prevents violations from entering the codebase
- **CI/CD Pipeline**: Validates boundaries on every change

## Monitoring and Metrics

The system tracks:
- Number of test files scanned
- Violations found by type
- Fixes applied automatically
- Exception usage and approval status
- Trends over time to measure improvement

## Future Enhancements

Planned improvements include:
- Integration with IDE plugins for real-time feedback
- Automatic suggestion of test refactoring opportunities
- Enhanced fixture sharing mechanisms
- Performance impact analysis of test isolation
- Advanced pattern recognition for complex violations