# Anomaly Entity Refactoring Summary

## Task Completed: Step 3 - Refactor Anomaly Entity

### Changes Made

#### 1. Created New Enums
- **AnomalyType** (`src/pynomaly/domain/value_objects/anomaly_type.py`)
  - Added enum with values: UNKNOWN, POINT, CONTEXTUAL, COLLECTIVE, SEASONAL, TREND, OUTLIER, DRIFT, CONCEPT_DRIFT, DATA_QUALITY, PERFORMANCE, SECURITY, BUSINESS_RULE, STATISTICAL, PATTERN, BEHAVIORAL, TEMPORAL, SPATIAL, NETWORK, SYSTEM
  - Default value: `AnomalyType.UNKNOWN`
  - Includes `get_default()` class method

- **SeverityLevel** (`src/pynomaly/domain/value_objects/severity_level.py`)
  - Added enum with values: LOW, MEDIUM, HIGH, CRITICAL
  - Includes `from_score()` class method for converting numeric scores to severity levels
  - String representation support

#### 2. Extended Anomaly Dataclass
- **New Fields Added**:
  - `anomaly_type: AnomalyType = AnomalyType.UNKNOWN`
  - `severity_level: SeverityLevel | None = None`

#### 3. Deprecated Severity Property
- **Backwards Compatibility**: The old `severity` property is still functional
- **Deprecation Warning**: Shows `DeprecationWarning` when accessed
- **Redirect**: Internally redirects to `severity_level` property

#### 4. Enhanced Validation in `__post_init__`
- **Type Validation**: Validates that `anomaly_type` is `AnomalyType` instance
- **Type Validation**: Validates that `severity_level` is `SeverityLevel` instance or None
- **Auto-derivation**: Automatically derives `severity_level` from score if None provided
- **Consistency Check**: Validates consistency between `score`, `anomaly_type`, and `severity_level`
- **Warning System**: Shows warnings for inconsistent severity levels

#### 5. Updated Methods
- **`to_dict()`**: Now includes `anomaly_type` and `severity_level` in serialization
- **`_validate_consistency()`**: New private method for validation logic

### Key Features Implemented

✅ **New Fields**: Added `anomaly_type` and `severity_level` with proper defaults
✅ **Backwards Compatibility**: Old `severity` property still works but deprecated
✅ **Validation**: Enhanced validation in `__post_init__` for consistency
✅ **Auto-derivation**: Automatically derives severity level from score when not provided
✅ **Warning System**: Warns users about inconsistencies and deprecation
✅ **Type Safety**: Proper type annotations and runtime validation

### Usage Examples

```python
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType, SeverityLevel
from pynomaly.domain.entities.anomaly import Anomaly

# Create anomaly with new fields
anomaly = Anomaly(
    score=AnomalyScore(0.8),
    data_point={"value": 100},
    detector_name="test_detector",
    anomaly_type=AnomalyType.CONTEXTUAL,
    severity_level=SeverityLevel.HIGH
)

# Auto-derive severity level
anomaly_auto = Anomaly(
    score=AnomalyScore(0.3),
    data_point={"value": 100},
    detector_name="test_detector",
    anomaly_type=AnomalyType.OUTLIER,
    severity_level=None  # Will auto-derive to LOW
)

# Deprecated usage (still works but shows warning)
severity = anomaly.severity  # Shows DeprecationWarning
```

### Files Modified/Created

1. **Created**: `src/pynomaly/domain/value_objects/anomaly_type.py`
2. **Created**: `src/pynomaly/domain/value_objects/severity_level.py`
3. **Modified**: `src/pynomaly/domain/value_objects/__init__.py` (added exports)
4. **Modified**: `src/pynomaly/domain/entities/anomaly.py` (complete refactoring)

### Testing Verified

- ✅ Basic anomaly creation with new fields
- ✅ Deprecated severity property shows warning
- ✅ Auto-derivation of severity level
- ✅ Validation of field types
- ✅ Consistency validation with warnings
- ✅ Serialization includes new fields
- ✅ Backwards compatibility maintained

The refactoring successfully extends the Anomaly entity while maintaining backwards compatibility and adding robust validation and consistency checking.
