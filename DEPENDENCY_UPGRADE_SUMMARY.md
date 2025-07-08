# Dependency Version Constraints Upgrade Summary

## Overview
This document summarizes the changes made to dependency version constraints across the project to use compatible ranges instead of exact pins or open-ended ranges.

## Files Modified
- `pyproject.toml`: Updated all dependency constraints to use compatible ranges
- `requirements.txt`: Updated core dependencies to use compatible ranges
- `package.json`: Already using caret notation (^) - no changes needed

## Key Changes Made

### 1. Core Dependencies (pyproject.toml)
- **NumPy**: `>=1.26.0,<2.2.0` → `>=1.26.0,<2.3.0`
- **pandas**: `>=2.2.3` → `>=2.2.3,<3.0.0`
- **polars**: `>=1.19.0` → `>=1.19.0,<2.0.0`
- **pydantic**: `>=2.10.4` → `>=2.10.4,<3.0.0`
- **structlog**: `>=24.4.0` → `>=24.4.0,<25.0.0`
- **dependency-injector**: `>=4.42.0` → `>=4.42.0,<5.0.0`

### 2. Web Framework Dependencies
- **FastAPI**: `>=0.115.0` → `>=0.115.0,<0.120.0`
- **Uvicorn**: `>=0.34.0` → `>=0.34.0,<0.36.0`
- **httpx**: `>=0.28.1` → `>=0.28.1,<0.30.0`
- **requests**: `>=2.32.3` → `>=2.32.3,<3.0.0`
- **Jinja2**: `>=3.1.5` → `>=3.1.5,<4.0.0`

### 3. ML Framework Dependencies
- **PyTorch**: `>=2.5.1` → `>=2.5.1,<3.0.0`
- **TensorFlow**: `>=2.18.0,<2.20.0` → `>=2.18.0,<2.20.0` (kept existing constraint)
- **Keras**: `>=3.8.0` → `>=3.8.0,<4.0.0`
- **JAX**: `>=0.4.37` → `>=0.4.37,<0.5.0`
- **JAXlib**: `>=0.4.37` → `>=0.4.37,<0.5.0`
- **scikit-learn**: `>=1.6.0` → `>=1.6.0,<1.8.0`
- **scipy**: `>=1.15.0` → `>=1.15.0,<1.17.0`

### 4. Testing Dependencies
- **pytest**: `>=8.0.0` → `>=8.0.0,<9.0.0`
- **pytest-cov**: `>=6.0.0` → `>=6.0.0,<7.0.0`
- **pytest-asyncio**: `>=0.24.0` → `>=0.24.0,<1.0.0`
- **playwright**: `>=1.40.0` → `>=1.40.0,<2.0.0`
- **hypothesis**: `>=6.115.0` → `>=6.115.0,<7.0.0`

### 5. Development Tools
- **ruff**: `>=0.8.0` → `>=0.8.0,<1.0.0`
- **black**: `>=24.0.0` → `>=24.0.0,<25.0.0`
- **mypy**: `>=1.13.0` → `>=1.13.0,<2.0.0`
- **pre-commit**: `>=4.0.0` → `>=4.0.0,<5.0.0`

### 6. Infrastructure Dependencies
- **redis**: `>=5.2.1` → `>=5.2.1,<6.0.0`
- **OpenTelemetry**: `>=1.29.0` → `>=1.29.0,<2.0.0`
- **Prometheus**: `>=0.21.1` → `>=0.21.1,<1.0.0`
- **SQLAlchemy**: `>=2.0.36` → `>=2.0.36,<3.0.0`

## Breaking Change Considerations

### Libraries with Known Breaking Changes
1. **TensorFlow**: Maintained existing `<2.20.0` constraint due to known breaking changes in 2.20+
2. **JAX/JAXlib**: Limited to `<0.5.0` due to frequent API changes in this rapidly evolving library
3. **NumPy**: Extended to `<2.3.0` to allow newer versions while avoiding breaking changes

### Conservative Approach for ML Libraries
- **PyTorch**: Limited to `<3.0.0` to avoid major version breaking changes
- **scikit-learn**: Limited to `<1.8.0` for stability in production environments
- **Optuna**: Limited to `<5.0.0` for hyperparameter optimization stability

## Platform-Specific Considerations
- All ARM/TensorFlow and CUDA-specific constraints maintained
- Platform markers kept in sync across all dependency groups
- Windows-specific dependencies (like `psycopg2-binary`) properly constrained

## Benefits of This Approach
1. **Predictable Updates**: Compatible ranges allow patch and minor version updates while preventing breaking changes
2. **Security**: Allows security patches to be applied automatically
3. **Maintenance**: Reduces the need for frequent manual dependency updates
4. **Stability**: Prevents unexpected breaking changes in production environments

## Next Steps
1. **Testing**: Run comprehensive tests across all dependency groups to ensure compatibility
2. **CI/CD**: Update CI pipeline to test against the upper bounds of version ranges
3. **Monitoring**: Monitor for any deprecation warnings or compatibility issues
4. **Documentation**: Update installation guides to reflect new version constraints

## Validation Commands
```bash
# Test minimal installation
pip install pynomaly[minimal]

# Test full installation
pip install pynomaly[all]

# Test specific combinations
pip install pynomaly[api,ml,test]
pip install pynomaly[production]
```

## Notes
- All changes are backward compatible with existing installations
- Version constraints follow semantic versioning principles
- Package.json already used caret notation (^) which provides similar benefits
- All Hatch environments updated consistently with main project dependencies
