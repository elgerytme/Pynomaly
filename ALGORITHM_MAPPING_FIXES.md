# Algorithm Mapping Fixes for PyOD 2.0.5

## Summary

This document outlines the fixes implemented to handle missing algorithm identifiers in the PyOD adapter for version 2.0.5.

## Problem Analysis

Using a static check script, we identified 6 algorithms in `ALGORITHM_MAPPING` that point to modules/classes that do not exist in the pinned PyOD version 2.0.5:

### Missing Class Algorithms
- **FastABOD**: Class doesn't exist in `pyod.models.abod` (only `ABOD` exists)
- **Beta-VAE**: Class doesn't exist in `pyod.models.vae` (only `VAE` exists)

### Missing Module Algorithms  
- **CLF**: Module `pyod.models.clf` doesn't exist

### Missing Dependency Algorithms
- **FeatureBagging**: Requires `combo` package (not installed)
- **XGBOD**: Requires `xgboost` package (not installed) 
- **SUOD**: Requires `suod` package (not installed)

## Solution Implemented

We chose solution (a) from the task description: **Add conditional try/except inside `_load_model_class` to raise clear `InvalidAlgorithmError` that tests can expect**.

### Changes Made

1. **Updated `InvalidAlgorithmError` exception** (in `src/pynomaly/domain/exceptions/detector_exceptions.py`):
   - Modified constructor to accept either a custom message or algorithm name
   - Maintains backward compatibility with existing code

2. **Enhanced `_load_model_class` method** (in `src/pynomaly/infrastructure/adapters/pyod_adapter.py`):
   - Added specific error handling for different failure types
   - Provides clear, actionable error messages
   - Includes installation instructions for missing dependencies

3. **Added comprehensive tests** (in `tests/infrastructure/test_adapters.py`):
   - `test_missing_class_algorithms()`: Tests algorithms where class doesn't exist
   - `test_missing_module_algorithms()`: Tests algorithms where module doesn't exist  
   - `test_missing_dependency_algorithms()`: Tests algorithms with missing dependencies
   - `test_algorithm_validation_comprehensive()`: Comprehensive validation test

## Error Messages

The enhanced error handling provides clear, specific messages:

### Missing Class Errors
```
Algorithm 'FastABOD' is not available in PyOD version 2.0.5. Class 'FastABOD' not found in module 'pyod.models.abod'.
```

### Missing Module Errors  
```
Algorithm 'CLF' is not available in PyOD version 2.0.5. Module 'pyod.models.clf' does not exist.
```

### Missing Dependency Errors
```
Algorithm 'FeatureBagging' requires 'combo' package. Install with: pip install combo
Algorithm 'XGBOD' requires 'xgboost' package. Install with: pip install xgboost
Algorithm 'SUOD' requires 'suod' package. Install with: pip install suod
```

## Benefits

1. **Clear Error Messages**: Users get specific, actionable error messages instead of generic ones
2. **Proactive Guidance**: Installation instructions are provided for missing dependencies
3. **Version Awareness**: Error messages include PyOD version information
4. **Maintained Compatibility**: Existing working algorithms continue to work unchanged
5. **Comprehensive Testing**: All failure scenarios are covered by tests

## Testing

All tests pass, covering:
- ✅ Missing class algorithms (FastABOD, Beta-VAE)
- ✅ Missing module algorithms (CLF) 
- ✅ Missing dependency algorithms (FeatureBagging, XGBOD, SUOD)
- ✅ Working algorithms (PCA, LOF, IsolationForest, etc.)
- ✅ Comprehensive validation of all problematic algorithms

## Implementation Details

The solution is low-risk because:
- It only affects error handling paths
- Working algorithms continue to function normally
- The mapping remains unchanged (no entries removed)
- Tests validate both success and failure cases
- Error messages are backward compatible

## Conclusion

The implementation successfully addresses the task requirements by providing clear, actionable error messages for missing algorithms while maintaining full backward compatibility with existing functionality.
