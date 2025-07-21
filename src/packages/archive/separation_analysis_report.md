# Repository Separation Analysis Report
==================================================

## Summary
- Total packages: 530
- Ready for separation (score ≥ 8): 23
- Partially ready (score 5-7): 503
- Not ready (score < 5): 4

## Dependency Graph
- data.data_platform → software.interfaces, software.interfaces, software.interfaces, software.interfaces
- data.data_platform.integration → software.interfaces, software.interfaces
- data.data_platform.quality.application.services → software.interfaces, software.interfaces

## Package Analysis

### data.anomaly_detection - ✅ READY (Score: 10/10)
- Files: 896
- Internal dependencies: 0
- External dependencies: 571
- Has pyproject.toml: True
- Has tests: True

### data.anomaly_detection.src.anomaly_detection.core - ✅ READY (Score: 10/10)
- Files: 146
- Internal dependencies: 0
- External dependencies: 182
- Has pyproject.toml: True
- Has tests: True

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder - ✅ READY (Score: 10/10)
- Files: 435
- Internal dependencies: 0
- External dependencies: 320
- Has pyproject.toml: True
- Has tests: True

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core - ✅ READY (Score: 10/10)
- Files: 150
- Internal dependencies: 0
- External dependencies: 182
- Has pyproject.toml: True
- Has tests: True

### data.data_platform.transformation - ✅ READY (Score: 10/10)
- Files: 30
- Internal dependencies: 0
- External dependencies: 29
- Has pyproject.toml: True
- Has tests: True

### ops.infrastructure - move! - ✅ READY (Score: 10/10)
- Files: 345
- Internal dependencies: 0
- External dependencies: 575
- Has pyproject.toml: True
- Has tests: True

### software.core - ✅ READY (Score: 10/10)
- Files: 124
- Internal dependencies: 0
- External dependencies: 141
- Has pyproject.toml: True
- Has tests: True

### software.interfaces.python_sdk - ✅ READY (Score: 10/10)
- Files: 77
- Internal dependencies: 0
- External dependencies: 74
- Has pyproject.toml: True
- Has tests: True

### data.anomaly_detection.src.anomaly_detection - ✅ READY (Score: 9/10)
- Files: 893
- Internal dependencies: 0
- External dependencies: 569
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection - ✅ READY (Score: 9/10)
- Files: 145
- Internal dependencies: 0
- External dependencies: 45
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### data.data_observability - ✅ READY (Score: 9/10)
- Files: 15
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### data.data_platform.profiling - ✅ READY (Score: 9/10)
- Files: 52
- Internal dependencies: 0
- External dependencies: 58
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### data.data_platform.transformation.tests - ✅ READY (Score: 9/10)
- Files: 10
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.unit - ✅ READY (Score: 9/10)
- Files: 107
- Internal dependencies: 0
- External dependencies: 142
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### software.interfaces - ✅ READY (Score: 9/10)
- Files: 274
- Internal dependencies: 0
- External dependencies: 437
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - no_pyproject: Missing pyproject.toml

### ai.mlops - ✅ READY (Score: 8/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: True
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder - ✅ READY (Score: 8/10)
- Files: 42
- Internal dependencies: 0
- External dependencies: 77
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform - ✅ READY (Score: 8/10)
- Files: 243
- Internal dependencies: 4
- External dependencies: 305
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - low_internal_deps: 4 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - software.interfaces.shared.error_handling
  - software.interfaces.data_quality_interface
  - software.interfaces.shared.base_entity
  - software.interfaces.data_profiling_interface

### data.data_platform.integration - ✅ READY (Score: 8/10)
- Files: 9
- Internal dependencies: 2
- External dependencies: 28
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - low_internal_deps: 2 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - software.interfaces.shared.error_handling
  - software.interfaces.shared.base_entity

### data.data_platform.science - ✅ READY (Score: 8/10)
- Files: 43
- Internal dependencies: 0
- External dependencies: 100
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### formal_sciences.mathematics - ✅ READY (Score: 8/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: True
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers

### ops.tools - move! - ✅ READY (Score: 8/10)
- Files: 28
- Internal dependencies: 0
- External dependencies: 35
- Has pyproject.toml: True
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers

### software.domain_library - ✅ READY (Score: 8/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: True
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers

### ai.machine_learning - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: True
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests

### ai.mlops.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### creative.documentation - move! - ⚠️ PARTIAL (Score: 7/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: True
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.algorithms - ⚠️ PARTIAL (Score: 7/10)
- Files: 79
- Internal dependencies: 0
- External dependencies: 129
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder - ⚠️ PARTIAL (Score: 7/10)
- Files: 42
- Internal dependencies: 0
- External dependencies: 125
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.core.domain - ⚠️ PARTIAL (Score: 7/10)
- Files: 93
- Internal dependencies: 0
- External dependencies: 126
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.core.domain.entities - ⚠️ PARTIAL (Score: 7/10)
- Files: 42
- Internal dependencies: 0
- External dependencies: 34
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain - ⚠️ PARTIAL (Score: 7/10)
- Files: 95
- Internal dependencies: 0
- External dependencies: 126
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.entities - ⚠️ PARTIAL (Score: 7/10)
- Files: 42
- Internal dependencies: 0
- External dependencies: 34
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure - ⚠️ PARTIAL (Score: 7/10)
- Files: 61
- Internal dependencies: 0
- External dependencies: 37
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.storage - ⚠️ PARTIAL (Score: 7/10)
- Files: 9
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.testing - ⚠️ PARTIAL (Score: 7/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.testing.orchestration - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.storage - ⚠️ PARTIAL (Score: 7/10)
- Files: 9
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.src.anomaly_detection.services - ⚠️ PARTIAL (Score: 7/10)
- Files: 132
- Internal dependencies: 0
- External dependencies: 222
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.anomaly_detection.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_observability.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_platform.integration.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_platform.profiling.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_platform.profiling.tests.{__init__.py} - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_platform.quality.application.services - ⚠️ PARTIAL (Score: 7/10)
- Files: 51
- Internal dependencies: 2
- External dependencies: 119
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - low_internal_deps: 2 internal dependencies
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests
- **Internal dependencies:**
  - software.interfaces.data_quality_interface
  - software.interfaces.data_profiling_interface

### data.data_platform.quality.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_platform.quality.tests.{__init__.py} - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### data.data_platform.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### formal_sciences.mathematics.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.config - move! - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.config - move!.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.infrastructure - move!.infrastructure - ⚠️ PARTIAL (Score: 7/10)
- Files: 338
- Internal dependencies: 0
- External dependencies: 567
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.infrastructure - move!.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.people_ops - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: True
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests

### ops.testing - move!.testing - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.application - ⚠️ PARTIAL (Score: 7/10)
- Files: 63
- Internal dependencies: 0
- External dependencies: 97
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.automl - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.bdd - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.common - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.contract - ⚠️ PARTIAL (Score: 7/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 14
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.functional - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.infrastructure.data_loaders - ⚠️ PARTIAL (Score: 7/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.infrastructure.data_processing - ⚠️ PARTIAL (Score: 7/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.integration - ⚠️ PARTIAL (Score: 7/10)
- Files: 100
- Internal dependencies: 0
- External dependencies: 215
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.integration.framework - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.mutation - ⚠️ PARTIAL (Score: 7/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.performance - ⚠️ PARTIAL (Score: 7/10)
- Files: 33
- Internal dependencies: 0
- External dependencies: 76
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.plugins - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.presentation - ⚠️ PARTIAL (Score: 7/10)
- Files: 51
- Internal dependencies: 0
- External dependencies: 78
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.property - ⚠️ PARTIAL (Score: 7/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.security - ⚠️ PARTIAL (Score: 7/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 43
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.ui.bdd.step_definitions - ⚠️ PARTIAL (Score: 7/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.ui.enhanced_page_objects - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.unit.shared - ⚠️ PARTIAL (Score: 7/10)
- Files: 11
- Internal dependencies: 0
- External dependencies: 18
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.unit.shared.protocols - ⚠️ PARTIAL (Score: 7/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 13
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests.utils - ⚠️ PARTIAL (Score: 7/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.testing - move!.tests.tests._stability - ⚠️ PARTIAL (Score: 7/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ops.tools - move!.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.core.domain - ⚠️ PARTIAL (Score: 7/10)
- Files: 76
- Internal dependencies: 0
- External dependencies: 98
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.core.domain.entities - move many of these to the correct domain - ⚠️ PARTIAL (Score: 7/10)
- Files: 32
- Internal dependencies: 0
- External dependencies: 24
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.core.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.domain_library.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.enterprise - ⚠️ PARTIAL (Score: 7/10)
- Files: 17
- Internal dependencies: 0
- External dependencies: 54
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.enterprise.adapters - ⚠️ PARTIAL (Score: 7/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 23
- Has pyproject.toml: True
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests

### software.enterprise.core - ⚠️ PARTIAL (Score: 7/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: True
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests

### software.enterprise.infrastructure - ⚠️ PARTIAL (Score: 7/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: True
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests

### software.enterprise.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.interfaces.python_sdk.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.interfaces.python_sdk.tests.e2e - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.interfaces.python_sdk.tests.fixtures - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.interfaces.python_sdk.tests.integration - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.interfaces.python_sdk.tests.unit - ⚠️ PARTIAL (Score: 7/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.interfaces.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.mobile.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.services - ⚠️ PARTIAL (Score: 7/10)
- Files: 126
- Internal dependencies: 0
- External dependencies: 211
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.services.src.services - ⚠️ PARTIAL (Score: 7/10)
- Files: 124
- Internal dependencies: 0
- External dependencies: 208
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.services.src.services.anomaly_detection - ⚠️ PARTIAL (Score: 7/10)
- Files: 65
- Internal dependencies: 0
- External dependencies: 133
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.services.src.services.infrastructure - ⚠️ PARTIAL (Score: 7/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 27
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.services.src.services.machine_learning - ⚠️ PARTIAL (Score: 7/10)
- Files: 27
- Internal dependencies: 0
- External dependencies: 89
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### software.services.tests - ⚠️ PARTIAL (Score: 7/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### ai.mlops.mlops - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.algorithms.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 36
- Internal dependencies: 0
- External dependencies: 115
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.algorithms.adapters.deep_learning - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 31
- Internal dependencies: 0
- External dependencies: 106
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.adapters.deep_learning - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.algorithms - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 20
- Internal dependencies: 0
- External dependencies: 61
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.application.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 38
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.application.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 22
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.abstractions - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.common - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.exceptions - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 17
- Internal dependencies: 0
- External dependencies: 73
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.validation - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 17
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 20
- Internal dependencies: 0
- External dependencies: 20
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.shared - ⚠️ PARTIAL (Score: 6/10)
- Files: 17
- Internal dependencies: 0
- External dependencies: 32
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.shared.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 22
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.shared.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.shared.utils - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.core.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 39
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 28
- Internal dependencies: 0
- External dependencies: 37
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.domain.abstractions - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 15
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.domain.exceptions - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 16
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 10
- Internal dependencies: 0
- External dependencies: 13
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 18
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.infrastructure.explainers - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.infrastructure.preprocessing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 36
- Internal dependencies: 0
- External dependencies: 115
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.adapters.deep_learning - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.alerting - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.algorithms - ⚠️ PARTIAL (Score: 6/10)
- Files: 37
- Internal dependencies: 0
- External dependencies: 115
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.algorithms.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 36
- Internal dependencies: 0
- External dependencies: 115
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.algorithms.adapters.deep_learning - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.auth - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.backup - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.batch - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.business_intelligence - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.cache - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.caching - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.cicd - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.compliance - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.abstractions - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.common - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.exceptions - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 17
- Internal dependencies: 0
- External dependencies: 73
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.validation - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 17
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 20
- Internal dependencies: 0
- External dependencies: 20
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared - ⚠️ PARTIAL (Score: 6/10)
- Files: 17
- Internal dependencies: 0
- External dependencies: 32
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 22
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared.utils - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 39
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 15
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data_loaders - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data_processing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data_quality - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.distributed - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.explainers - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.federated - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.feedback - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.global_scale - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 24
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces.python_sdk - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.lifecycle - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.logging - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.messaging - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.middleware - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.ml_governance - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.monitoring - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.multitenancy - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.notifications - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.optimization - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.performance - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.performance_v2 - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.preprocessing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.production - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.di - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.services.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.services.interfaces - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.docs_validation - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.docs_validation.checkers - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.docs_validation.core - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 26
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.abstractions - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.common - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.exceptions - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.models - ⚠️ PARTIAL (Score: 6/10)
- Files: 15
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.validation - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.enterprise - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.features - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.adapters.deep_learning - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.alerting - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.auth - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.backup - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.batch - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.business_intelligence - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.cache - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.caching - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.cicd - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.compliance - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 15
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data_loaders - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data_processing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data_quality - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.distributed - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.explainers - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.federated - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.feedback - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.global_scale - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.lifecycle - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.logging - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.messaging - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.middleware - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.ml_governance - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.monitoring - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.multitenancy - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.notifications - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.optimization - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.performance - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.performance_v2 - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.preprocessing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.production - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.quality - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.resilience - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.scheduler - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.serving - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.streaming - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.tdd - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.tracing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.websocket - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.mlops - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation - ⚠️ PARTIAL (Score: 6/10)
- Files: 27
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 10
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.dependencies - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.docs - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.endpoints - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.graphql - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.graphql.resolvers - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.middleware - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.routers - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.websocket - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.cli._click_backup - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.graphql - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.sdk - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web - ⚠️ PARTIAL (Score: 6/10)
- Files: 12
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.models - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.routes - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.data - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.js - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.js.src - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.js.src.components - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.edge - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.explainability - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.quantum - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.synthetic - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.schemas - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.schemas.analytics - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.scripts - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared.utils - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.quality - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.resilience - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.scheduler - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.serving - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.streaming - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.tdd - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.tracing - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.websocket - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to anomaly_detection folder.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 18
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 16
- Internal dependencies: 0
- External dependencies: 29
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 19
- Internal dependencies: 0
- External dependencies: 51
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.di - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.execution - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 9
- Internal dependencies: 0
- External dependencies: 26
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.storage - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.anomaly_detection.src.anomaly_detection.services.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_observability.application.facades - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_observability.data_observability - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.data-platform - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 13
- Internal dependencies: 0
- External dependencies: 23
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.application.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.application.dto.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 23
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.application.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.application.use_cases.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.docs - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.docs.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 17
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.entities.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.repositories.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.services.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.domain.value_objects.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.infrastructure.adapters.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.infrastructure.persistence.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation.api.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation.cli.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation.web - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.profiling.presentation.web.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.application.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.application.dto.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.application.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.application.use_cases.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.application.__init__.py - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.docs - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.docs.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 13
- Internal dependencies: 0
- External dependencies: 14
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.entities.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.repositories.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.services.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.value_objects.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.domain.__init__.py - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.adapters.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.logging - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.persistence.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.infrastructure.__init__.py - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.api.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.cli.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.web - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.web.{__init__.py} - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.presentation.__init__.py - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.quality.__init__.py - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 32
- Internal dependencies: 0
- External dependencies: 60
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 10
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.domain.interfaces - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 16
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 29
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 14
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.infrastructure.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.science.infrastructure.di - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.application.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.application.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 8
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 5
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### data.data_platform.transformation.infrastructure.processors - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### formal_sciences.mathematics.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### formal_sciences.mathematics.mathematics - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.config - move!.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 35
- Internal dependencies: 0
- External dependencies: 114
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.adapters.deep_learning - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.alerting - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 21
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.auth - ⚠️ PARTIAL (Score: 6/10)
- Files: 11
- Internal dependencies: 0
- External dependencies: 35
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.batch - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 14
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.business_intelligence - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.cache - ⚠️ PARTIAL (Score: 6/10)
- Files: 12
- Internal dependencies: 0
- External dependencies: 30
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 26
- Internal dependencies: 0
- External dependencies: 74
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.data_loaders - ⚠️ PARTIAL (Score: 6/10)
- Files: 15
- Internal dependencies: 0
- External dependencies: 44
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.data_processing - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 23
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.distributed - ⚠️ PARTIAL (Score: 6/10)
- Files: 9
- Internal dependencies: 0
- External dependencies: 24
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 19
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.explainers - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 18
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.federated - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 13
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.lifecycle - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.logging - ⚠️ PARTIAL (Score: 6/10)
- Files: 9
- Internal dependencies: 0
- External dependencies: 22
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.messaging - ⚠️ PARTIAL (Score: 6/10)
- Files: 15
- Internal dependencies: 0
- External dependencies: 21
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.messaging.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.messaging.config - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.messaging.models - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 5
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.messaging.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.messaging.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.middleware - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.ml_governance - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.monitoring - ⚠️ PARTIAL (Score: 6/10)
- Files: 29
- Internal dependencies: 0
- External dependencies: 110
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.multitenancy - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.performance - ⚠️ PARTIAL (Score: 6/10)
- Files: 15
- Internal dependencies: 0
- External dependencies: 42
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 23
- Internal dependencies: 0
- External dependencies: 55
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.preprocessing - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 23
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.production - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 17
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.quality - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 5
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 36
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.resilience - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 16
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 31
- Internal dependencies: 0
- External dependencies: 60
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.streaming - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 13
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.infrastructure.tdd - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.infrastructure - move!.quality - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.testing - move!.tests.tests.packages.data_profiling - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 8
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - medium_internal_deps: 8 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_profiling.application.services.performance_optimizer
  - src.packages.data_profiling.application.services.advanced_profiling_orchestrator
  - src.packages.data_profiling.application.services.statistical_profiling_service
  - src.packages.data_profiling.domain.entities.data_profile
  - src.packages.data_profiling.application.services.profiling_engine
  - ... and 3 more

### ops.testing - move!.tests.tests.packages.data_profiling.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 1
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - low_internal_deps: 1 internal dependencies
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_profiling.domain.entities.data_profile

### ops.testing - move!.tests.tests.packages.data_profiling.infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 1
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - low_internal_deps: 1 internal dependencies
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_profiling.infrastructure.adapters.file_adapter

### ops.tools - move!.tools - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.core - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.abstractions - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.common - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.exceptions - move these - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.protocols - move these - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.services - move these - ⚠️ PARTIAL (Score: 6/10)
- Files: 14
- Internal dependencies: 0
- External dependencies: 54
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.validation - move these - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 11
- Internal dependencies: 0
- External dependencies: 17
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 19
- Internal dependencies: 0
- External dependencies: 20
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.shared - ⚠️ PARTIAL (Score: 6/10)
- Files: 19
- Internal dependencies: 0
- External dependencies: 32
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.shared.error_handling - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 22
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.shared.protocols - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.shared.utils - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.core.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 20
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.domain_library.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.domain_library.domain.exceptions - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.domain_library.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 6
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.domain_library.domain_library - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.enterprise.adapters.src.enterprise_adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 23
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.enterprise.core.src.enterprise_core - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.enterprise.enterprise - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.enterprise.infrastructure.src.enterprise_infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 25
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 91
- Internal dependencies: 0
- External dependencies: 202
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 88
- Internal dependencies: 0
- External dependencies: 201
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api.api.dependencies - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api.api.docs - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api.api.endpoints - ⚠️ PARTIAL (Score: 6/10)
- Files: 43
- Internal dependencies: 0
- External dependencies: 87
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api.api.middleware - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.api.api.security - ⚠️ PARTIAL (Score: 6/10)
- Files: 10
- Internal dependencies: 0
- External dependencies: 30
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 66
- Internal dependencies: 0
- External dependencies: 181
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.cli.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 58
- Internal dependencies: 0
- External dependencies: 126
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.cli.commands - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 65
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.interfaces - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.application - ⚠️ PARTIAL (Score: 6/10)
- Files: 6
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.application.dto - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.application.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.application.use_cases - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.docs - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.docs.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.docs.examples - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.docs.guides - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.domain - ⚠️ PARTIAL (Score: 6/10)
- Files: 12
- Internal dependencies: 0
- External dependencies: 8
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.domain.entities - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.domain.exceptions - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.domain.repositories - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.domain.services - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 4
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.domain.value_objects - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 1
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.infrastructure - ⚠️ PARTIAL (Score: 6/10)
- Files: 7
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.infrastructure.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 11
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.infrastructure.external - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.infrastructure.logging - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.infrastructure.persistence - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.presentation - ⚠️ PARTIAL (Score: 6/10)
- Files: 5
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.presentation.api - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.presentation.cli - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 7
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.python_sdk.presentation.web - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.sdks - ⚠️ PARTIAL (Score: 6/10)
- Files: 4
- Internal dependencies: 0
- External dependencies: 10
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.sdks.sdks.python.monorepo_client - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 9
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.web - ⚠️ PARTIAL (Score: 6/10)
- Files: 27
- Internal dependencies: 0
- External dependencies: 62
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.interfaces.web.web - ⚠️ PARTIAL (Score: 6/10)
- Files: 26
- Internal dependencies: 0
- External dependencies: 61
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.mobile.adapters - ⚠️ PARTIAL (Score: 6/10)
- Files: 2
- Internal dependencies: 0
- External dependencies: 2
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.mobile.mobile - ⚠️ PARTIAL (Score: 6/10)
- Files: 1
- Internal dependencies: 0
- External dependencies: 0
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.services.src.services.anomaly_detection.automl - ⚠️ PARTIAL (Score: 6/10)
- Files: 3
- Internal dependencies: 0
- External dependencies: 12
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.services.src.services.data_platform - ⚠️ PARTIAL (Score: 6/10)
- Files: 13
- Internal dependencies: 0
- External dependencies: 31
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### software.services.src.services.enterprise - ⚠️ PARTIAL (Score: 6/10)
- Files: 10
- Internal dependencies: 0
- External dependencies: 26
- Has pyproject.toml: False
- Has tests: False
- **Blockers:**
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
  - no_tests: Missing tests

### ops.testing - move! - ⚠️ PARTIAL (Score: 5/10)
- Files: 746
- Internal dependencies: 38
- External dependencies: 712
- Has pyproject.toml: True
- Has tests: True
- **Blockers:**
  - high_internal_deps: 38 internal dependencies
- **Internal dependencies:**
  - src.packages.data_quality.application.services.predictive_quality_service
  - src.packages.data_quality.application.services.advanced_quality_metrics_service
  - src.packages.data_quality.application.services.comprehensive_quality_scoring_engine
  - src.packages.data_quality.application.services.quality_lineage_service
  - src.packages.python_sdk.domain.exceptions.validation_exceptions
  - ... and 33 more

### ops.testing - move!.tests - ❌ NOT READY (Score: 4/10)
- Files: 743
- Internal dependencies: 38
- External dependencies: 712
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - high_internal_deps: 38 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_quality.application.services.predictive_quality_service
  - src.packages.data_quality.application.services.advanced_quality_metrics_service
  - src.packages.data_quality.application.services.comprehensive_quality_scoring_engine
  - src.packages.data_quality.application.services.quality_lineage_service
  - src.packages.python_sdk.domain.exceptions.validation_exceptions
  - ... and 33 more

### ops.testing - move!.tests.tests - ❌ NOT READY (Score: 4/10)
- Files: 742
- Internal dependencies: 38
- External dependencies: 712
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - high_internal_deps: 38 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_quality.application.services.predictive_quality_service
  - src.packages.data_quality.application.services.advanced_quality_metrics_service
  - src.packages.data_quality.application.services.comprehensive_quality_scoring_engine
  - src.packages.data_quality.application.services.quality_lineage_service
  - src.packages.python_sdk.domain.exceptions.validation_exceptions
  - ... and 33 more

### ops.testing - move!.tests.tests.packages.data_profiling.application - ❌ NOT READY (Score: 4/10)
- Files: 7
- Internal dependencies: 7
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - medium_internal_deps: 7 internal dependencies
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_profiling.application.services.performance_optimizer
  - src.packages.data_profiling.application.services.advanced_profiling_orchestrator
  - src.packages.data_profiling.application.services.statistical_profiling_service
  - src.packages.data_profiling.domain.entities.data_profile
  - src.packages.data_profiling.application.services.profiling_engine
  - ... and 2 more

### ops.testing - move!.tests.tests.packages.data_quality.integration - ❌ NOT READY (Score: 4/10)
- Files: 2
- Internal dependencies: 9
- External dependencies: 3
- Has pyproject.toml: False
- Has tests: True
- **Blockers:**
  - medium_internal_deps: 9 internal dependencies
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **Internal dependencies:**
  - src.packages.data_quality.application.services.predictive_quality_service
  - src.packages.data_quality.application.services.comprehensive_quality_scoring_engine
  - src.packages.data_quality.application.services.validation_engine
  - src.packages.data_quality.domain.entities.quality_scores
  - src.packages.data_quality.application.services.data_cleansing_service
  - ... and 4 more

## Recommendations for Repository Separation

### High Priority (Ready packages - score ≥ 8)
- **ai.mlops**: Can be separated immediately
- **data.anomaly_detection**: Can be separated immediately
- **data.anomaly_detection.src.anomaly_detection**: Can be separated immediately
- **data.anomaly_detection.src.anomaly_detection.core**: Can be separated immediately
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder**: Can be separated immediately
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core**: Can be separated immediately
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection**: Can be separated immediately
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder**: Can be separated immediately
- **data.data_observability**: Can be separated immediately
- **data.data_platform**: Can be separated immediately
- **data.data_platform.integration**: Can be separated immediately
- **data.data_platform.profiling**: Can be separated immediately
- **data.data_platform.science**: Can be separated immediately
- **data.data_platform.transformation**: Can be separated immediately
- **data.data_platform.transformation.tests**: Can be separated immediately
- **formal_sciences.mathematics**: Can be separated immediately
- **ops.infrastructure - move!**: Can be separated immediately
- **ops.testing - move!.tests.tests.unit**: Can be separated immediately
- **ops.tools - move!**: Can be separated immediately
- **software.core**: Can be separated immediately
- **software.domain_library**: Can be separated immediately
- **software.interfaces**: Can be separated immediately
- **software.interfaces.python_sdk**: Can be separated immediately

### Medium Priority (Partially ready - score 5-7)
- **ai.machine_learning**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests
- **ai.mlops.mlops**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ai.mlops.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **creative.documentation - move!**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests
- **data.anomaly_detection.src.anomaly_detection.algorithms**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.adapters.deep_learning**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.adapters.deep_learning**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.algorithms**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.algorithms.algorithms - move to data anomaly_detection folder.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.application.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.application.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.abstractions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.common**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.entities**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.exceptions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.validation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.shared**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.shared.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.shared.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.shared.utils**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.core.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.domain.abstractions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.domain.exceptions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.infrastructure**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.infrastructure.explainers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.infrastructure.preprocessing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.adapters.deep_learning**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.alerting**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.algorithms**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.algorithms.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.algorithms.adapters.deep_learning**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.auth**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.backup**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.batch**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.business_intelligence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.cache**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.caching**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.cicd**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.compliance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.abstractions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.common**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.entities**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.exceptions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.validation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.shared.utils**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.core.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data_loaders**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data_processing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.data_quality**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.distributed**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.explainers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.federated**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.feedback**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.global_scale**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.interfaces.python_sdk**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.lifecycle**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.logging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.messaging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.middleware**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.ml_governance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.monitoring**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.multitenancy**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.notifications**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.optimization**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.performance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.performance_v2**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.preprocessing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.production**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.di**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.services.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.services.interfaces**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.application.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.docs_validation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.docs_validation.checkers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.docs_validation.core**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.abstractions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.common**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.exceptions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.models**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.validation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.enterprise**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.features**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.adapters.deep_learning**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.alerting**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.auth**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.backup**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.batch**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.business_intelligence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.cache**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.caching**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.cicd**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.compliance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data_loaders**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data_processing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.data_quality**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.distributed**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.explainers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.federated**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.feedback**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.global_scale**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.lifecycle**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.logging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.messaging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.middleware**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.ml_governance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.monitoring**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.multitenancy**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.notifications**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.optimization**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.performance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.performance_v2**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.preprocessing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.production**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.quality**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.resilience**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.scheduler**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.serving**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.storage**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.streaming**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.tdd**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.tracing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.infrastructure.websocket**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.mlops**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.dependencies**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.docs**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.endpoints**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.graphql**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.graphql.resolvers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.middleware**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.routers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.api.websocket**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.cli._click_backup**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.graphql**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.sdk**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.models**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.routes**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.data**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.js**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.js.src**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.presentation.web.static.js.src.components**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.edge**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.explainability**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.quantum**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.research.synthetic**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.schemas**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.schemas.analytics**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.scripts**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.shared.utils**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.testing**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.testing.orchestration**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.anomaly_detection.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.quality**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.resilience**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.scheduler**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.serving**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.storage**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.streaming**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.tdd**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.tracing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection - move to data anomaly_detection folder.websocket**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to anomaly_detection folder.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.di**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.execution**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.ml.anomaly_detection_mlops - move to data anomaly_detection folder.infrastructure.storage**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.services**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.src.anomaly_detection.services.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.anomaly_detection.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_observability.application.facades**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_observability.data_observability**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_observability.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.data-platform**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.integration.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.application.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.application.dto.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.application.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.application.use_cases.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.docs**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.docs.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.entities.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.repositories.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.services.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.domain.value_objects.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.infrastructure**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.infrastructure.adapters.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.infrastructure.persistence.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation.api.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation.cli.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation.web**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.presentation.web.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.profiling.tests.{__init__.py}**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.application.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.application.dto.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.application.services**: Address 3 blockers
  - low_internal_deps: 2 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.application.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.application.use_cases.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.application.__init__.py**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.docs**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.docs.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.entities.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.repositories.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.services.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.value_objects.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.domain.__init__.py**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.adapters.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.logging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.persistence.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.infrastructure.__init__.py**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.api.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.cli.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.web**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.web.{__init__.py}**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.presentation.__init__.py**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.tests.{__init__.py}**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.quality.__init__.py**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.domain.interfaces**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.infrastructure.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.science.infrastructure.di**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.application.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.application.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.infrastructure**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **data.data_platform.transformation.infrastructure.processors**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **formal_sciences.mathematics.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **formal_sciences.mathematics.mathematics**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **formal_sciences.mathematics.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.config - move!**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.config - move!.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.config - move!.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.adapters.deep_learning**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.alerting**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.auth**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.batch**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.business_intelligence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.cache**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.data_loaders**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.data_processing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.distributed**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.explainers**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.federated**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.lifecycle**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.logging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.messaging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.messaging.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.messaging.config**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.messaging.models**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.messaging.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.messaging.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.middleware**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.ml_governance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.monitoring**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.multitenancy**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.performance**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.preprocessing**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.production**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.quality**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.resilience**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.streaming**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.infrastructure.tdd**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.quality**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.infrastructure - move!.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.people_ops**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests
- **ops.testing - move!**: Address 1 blockers
  - high_internal_deps: 38 internal dependencies
- **ops.testing - move!.testing**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.application**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.automl**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.bdd**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.common**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.contract**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.functional**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.infrastructure.data_loaders**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.infrastructure.data_processing**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.integration**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.integration.framework**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.mutation**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.packages.data_profiling**: Address 2 blockers
  - medium_internal_deps: 8 internal dependencies
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.packages.data_profiling.domain**: Address 3 blockers
  - low_internal_deps: 1 internal dependencies
  - incomplete_layers: Missing domain/application layers
- **ops.testing - move!.tests.tests.packages.data_profiling.infrastructure**: Address 3 blockers
  - low_internal_deps: 1 internal dependencies
  - incomplete_layers: Missing domain/application layers
- **ops.testing - move!.tests.tests.performance**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.plugins**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.presentation**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.property**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.security**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.ui.bdd.step_definitions**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.ui.enhanced_page_objects**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.unit.shared**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.unit.shared.protocols**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests.utils**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.testing - move!.tests.tests._stability**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.tools - move!.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **ops.tools - move!.tools**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.core**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.abstractions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.common**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.entities - move many of these to the correct domain**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.exceptions - move these**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.protocols - move these**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.services - move these**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.validation - move these**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.shared**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.shared.error_handling**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.shared.protocols**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.shared.utils**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.core.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.domain_library.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.domain_library.domain.exceptions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.domain_library.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.domain_library.domain_library**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.domain_library.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.enterprise**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.enterprise.adapters**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests
- **software.enterprise.adapters.src.enterprise_adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.enterprise.core**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests
- **software.enterprise.core.src.enterprise_core**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.enterprise.enterprise**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.enterprise.infrastructure**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_tests: Missing tests
- **software.enterprise.infrastructure.src.enterprise_infrastructure**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.enterprise.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api.api.dependencies**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api.api.docs**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api.api.endpoints**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api.api.middleware**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.api.api.security**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.cli.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.cli.commands**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.interfaces**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.application**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.application.dto**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.application.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.application.use_cases**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.docs**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.docs.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.docs.examples**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.docs.guides**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.domain**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.domain.entities**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.domain.exceptions**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.domain.repositories**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.domain.services**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.domain.value_objects**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.infrastructure**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.infrastructure.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.infrastructure.external**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.infrastructure.logging**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.infrastructure.persistence**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.presentation**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.presentation.api**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.presentation.cli**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.presentation.web**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.tests.e2e**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.tests.fixtures**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.tests.integration**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.python_sdk.tests.unit**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.sdks**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.sdks.sdks.python.monorepo_client**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.web**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.interfaces.web.web**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.mobile.adapters**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.mobile.mobile**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.mobile.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services.anomaly_detection**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services.anomaly_detection.automl**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services.data_platform**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services.enterprise**: Address 3 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services.infrastructure**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.src.services.machine_learning**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml
- **software.services.tests**: Address 2 blockers
  - incomplete_layers: Missing domain/application layers
  - no_pyproject: Missing pyproject.toml

### Low Priority (Not ready - score < 5)
- **ops.testing - move!.tests**: Requires significant refactoring
- **ops.testing - move!.tests.tests**: Requires significant refactoring
- **ops.testing - move!.tests.tests.packages.data_profiling.application**: Requires significant refactoring
- **ops.testing - move!.tests.tests.packages.data_quality.integration**: Requires significant refactoring

## Shared Infrastructure Analysis

### Most Common External Dependencies
- __future__: used by 184 packages
- uuid: used by 168 packages
- numpy: used by 137 packages
- enum: used by 136 packages
- asyncio: used by 128 packages
- pandas: used by 126 packages
- abc: used by 95 packages
- pydantic: used by 89 packages
- hashlib: used by 69 packages
- sklearn.preprocessing: used by 67 packages