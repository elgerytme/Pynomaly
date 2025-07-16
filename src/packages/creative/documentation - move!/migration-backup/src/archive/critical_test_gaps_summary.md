# Critical Test Gaps Summary

Based on the comprehensive test coverage analysis, here are the specific files that don't have corresponding tests, focusing on the most critical areas:

## HIGH PRIORITY - Protocol Definitions (5 files)
These are critical interfaces that define contracts between components:

- **src/pynomaly/shared/protocols/__init__.py**
  - Expected test: `tests/unit/shared/protocols/test___init__.py`
  - Test type: Unit test for protocol imports/exports

- **src/pynomaly/shared/protocols/data_loader_protocol.py**
  - Expected test: `tests/unit/shared/protocols/test_data_loader_protocol.py`
  - Test type: Protocol contract validation tests

- **src/pynomaly/shared/protocols/detector_protocol.py**
  - Expected test: `tests/unit/shared/protocols/test_detector_protocol.py`
  - Test type: Protocol contract validation tests

- **src/pynomaly/shared/protocols/export_protocol.py**
  - Expected test: `tests/unit/shared/protocols/test_export_protocol.py`
  - Test type: Protocol contract validation tests

- **src/pynomaly/shared/protocols/import_protocol.py**
  - Expected test: `tests/unit/shared/protocols/test_import_protocol.py`
  - Test type: Protocol contract validation tests

- **src/pynomaly/shared/protocols/repository_protocol.py**
  - Expected test: `tests/unit/shared/protocols/test_repository_protocol.py`
  - Test type: Protocol contract validation tests

## HIGH PRIORITY - Shared Modules (4 files)
These modules are used across the entire application:

- **src/pynomaly/shared/__init__.py**
  - Expected test: `tests/unit/shared/test___init__.py`
  - Test type: Module structure and imports

- **src/pynomaly/shared/error_handling.py**
  - Expected test: `tests/unit/shared/test_error_handling.py`
  - Test type: Error handling behavior and edge cases

- **src/pynomaly/shared/exceptions.py**
  - Expected test: `tests/unit/shared/test_exceptions.py`
  - Test type: Exception class behavior and hierarchy

- **src/pynomaly/shared/types.py**
  - Expected test: `tests/unit/shared/test_types.py`
  - Test type: Type definitions and validation

## HIGH PRIORITY - Domain Value Objects (9 files)
These contain core business logic and validation:

- **src/pynomaly/domain/value_objects/anomaly_category.py**
  - Expected test: `tests/unit/domain/value_objects/test_anomaly_category.py`
  - Test type: Value object validation and immutability

- **src/pynomaly/domain/value_objects/anomaly_type.py**
  - Expected test: `tests/unit/domain/value_objects/test_anomaly_type.py`
  - Test type: Value object validation and immutability

- **src/pynomaly/domain/value_objects/confidence_interval.py**
  - Expected test: `tests/unit/domain/value_objects/test_confidence_interval.py`
  - Test type: Mathematical validation and edge cases

- **src/pynomaly/domain/value_objects/contamination_rate.py**
  - Expected test: `tests/unit/domain/value_objects/test_contamination_rate.py`
  - Test type: Range validation and business rules

- **src/pynomaly/domain/value_objects/hyperparameters.py**
  - Expected test: `tests/unit/domain/value_objects/test_hyperparameters.py`
  - Test type: Parameter validation and serialization

- **src/pynomaly/domain/value_objects/model_storage_info.py**
  - Expected test: `tests/unit/domain/value_objects/test_model_storage_info.py`
  - Test type: Storage path validation and metadata

- **src/pynomaly/domain/value_objects/performance_metrics.py**
  - Expected test: `tests/unit/domain/value_objects/test_performance_metrics.py`
  - Test type: Metric calculations and validation

- **src/pynomaly/domain/value_objects/semantic_version.py**
  - Expected test: `tests/unit/domain/value_objects/test_semantic_version.py`
  - Test type: Version comparison and validation

- **src/pynomaly/domain/value_objects/threshold_config.py**
  - Expected test: `tests/unit/domain/value_objects/test_threshold_config.py`
  - Test type: Threshold validation and business rules

## HIGH PRIORITY - Domain Exceptions (6 files)
Critical for error handling reliability:

- **src/pynomaly/domain/exceptions/base.py**
  - Expected test: `tests/unit/domain/exceptions/test_base.py`
  - Test type: Base exception behavior and inheritance

- **src/pynomaly/domain/exceptions/dataset_exceptions.py**
  - Expected test: `tests/unit/domain/exceptions/test_dataset_exceptions.py`
  - Test type: Dataset-specific exception scenarios

- **src/pynomaly/domain/exceptions/detector_exceptions.py**
  - Expected test: `tests/unit/domain/exceptions/test_detector_exceptions.py`
  - Test type: Detector-specific exception scenarios

- **src/pynomaly/domain/exceptions/entity_exceptions.py**
  - Expected test: `tests/unit/domain/exceptions/test_entity_exceptions.py`
  - Test type: Entity validation exception scenarios

- **src/pynomaly/domain/exceptions/result_exceptions.py**
  - Expected test: `tests/unit/domain/exceptions/test_result_exceptions.py`
  - Test type: Result processing exception scenarios

## HIGH PRIORITY - Data Transfer Objects (16 files)
These define critical data contracts:

### Core DTOs:
- **src/pynomaly/application/dto/dataset_dto.py**
  - Expected test: `tests/unit/application/dto/test_dataset_dto.py`
  - Test type: Data validation and serialization

- **src/pynomaly/application/dto/detector_dto.py**
  - Expected test: `tests/unit/application/dto/test_detector_dto.py`
  - Test type: Data validation and serialization

- **src/pynomaly/application/dto/detection_dto.py**
  - Expected test: `tests/unit/application/dto/test_detection_dto.py`
  - Test type: Data validation and serialization

- **src/pynomaly/application/dto/result_dto.py**
  - Expected test: `tests/unit/application/dto/test_result_dto.py`
  - Test type: Data validation and serialization

- **src/pynomaly/application/dto/training_dto.py**
  - Expected test: `tests/unit/application/dto/test_training_dto.py`
  - Test type: Data validation and serialization

### Advanced DTOs:
- **src/pynomaly/application/dto/automl_dto.py**
  - Expected test: `tests/unit/application/dto/test_automl_dto.py`
  - Test type: AutoML configuration validation

- **src/pynomaly/application/dto/experiment_dto.py**
  - Expected test: `tests/unit/application/dto/test_experiment_dto.py`
  - Test type: Experiment configuration validation

- **src/pynomaly/application/dto/explainability_dto.py**
  - Expected test: `tests/unit/application/dto/test_explainability_dto.py`
  - Test type: Explainability configuration validation

## Test Creation Strategy

### For Protocol Files:
- Test protocol conformance
- Test type hints and method signatures
- Test that implementations satisfy the protocol
- Mock implementations for testing

### For Shared Modules:
- Test error handling paths
- Test exception hierarchy and inheritance
- Test utility functions
- Test type validation

### For Value Objects:
- Test immutability
- Test validation rules
- Test equality and comparison
- Test serialization/deserialization
- Test edge cases and boundary conditions

### For DTOs:
- Test data validation
- Test serialization/deserialization
- Test field constraints
- Test nested object handling
- Test error scenarios

### For Domain Exceptions:
- Test exception creation
- Test error message formatting
- Test exception inheritance
- Test context preservation
- Test error codes and categorization

## Recommended Test Structure

```
tests/unit/
├── shared/
│   ├── protocols/
│   │   ├── test_data_loader_protocol.py
│   │   ├── test_detector_protocol.py
│   │   ├── test_export_protocol.py
│   │   ├── test_import_protocol.py
│   │   └── test_repository_protocol.py
│   ├── test_error_handling.py
│   ├── test_exceptions.py
│   └── test_types.py
├── domain/
│   ├── value_objects/
│   │   ├── test_anomaly_category.py
│   │   ├── test_anomaly_type.py
│   │   ├── test_confidence_interval.py
│   │   ├── test_contamination_rate.py
│   │   ├── test_hyperparameters.py
│   │   ├── test_model_storage_info.py
│   │   ├── test_performance_metrics.py
│   │   ├── test_semantic_version.py
│   │   └── test_threshold_config.py
│   └── exceptions/
│       ├── test_base.py
│       ├── test_dataset_exceptions.py
│       ├── test_detector_exceptions.py
│       ├── test_entity_exceptions.py
│       └── test_result_exceptions.py
└── application/
    └── dto/
        ├── test_dataset_dto.py
        ├── test_detector_dto.py
        ├── test_detection_dto.py
        ├── test_result_dto.py
        ├── test_training_dto.py
        ├── test_automl_dto.py
        ├── test_experiment_dto.py
        └── test_explainability_dto.py
```

This represents the most critical 40 files that need tests to achieve good coverage of the core framework components.