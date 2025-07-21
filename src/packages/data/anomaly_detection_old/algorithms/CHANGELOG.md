# Changelog - Algorithms Package

All notable changes to the Pynomaly algorithms package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Support for custom algorithm implementations
- Enhanced hyperparameter optimization with Optuna
- Batch processing capabilities for large datasets
- Algorithm performance benchmarking utilities
- Support for streaming anomaly detection

### Changed
- Improved algorithm adapter performance and memory usage
- Enhanced error handling for algorithm failures
- Optimized hyperparameter search spaces

### Fixed
- Memory leaks in long-running algorithm sessions
- Edge cases in ensemble voting mechanisms
- Compatibility issues with latest PyOD versions

## [1.0.0] - 2025-07-14

### Added
- **PyOD Integration**: Full adapter for PyOD library
  - 40+ anomaly detection algorithms supported
  - Isolation Forest, Local Outlier Factor, One-Class SVM
  - COPOD, ECOD, ABOD, and neural network methods
  - Ensemble methods with feature bagging and model combination
- **Scikit-learn Adapters**: Native scikit-learn algorithm support
  - One-Class SVM with various kernels
  - Isolation Forest with optimized parameters
  - Local Outlier Factor with distance metrics
  - Elliptic Envelope for Gaussian data
- **Neural Network Algorithms**: Deep learning approaches
  - AutoEncoder for dimensionality reduction
  - Variational AutoEncoder (VAE) for probabilistic detection
  - Deep Support Vector Data Description (Deep SVDD)
  - LSTM-based time series anomaly detection
- **Ensemble Methods**: Multiple algorithm combination
  - Weighted voting ensemble
  - Stacking ensemble with meta-learners
  - Dynamic ensemble selection
  - Adaptive ensemble with online learning
- **Time Series Algorithms**: Temporal anomaly detection
  - Seasonal decomposition methods
  - Change point detection algorithms
  - ARIMA-based anomaly detection
  - Prophet integration for trend analysis

### Algorithm Categories
- **Statistical Methods**: 12 algorithms including IForest, LOF, OCSVM
- **Probabilistic Methods**: 8 algorithms including GMM, COPOD, ECOD
- **Neural Networks**: 6 deep learning approaches
- **Ensemble Methods**: 4 combination strategies
- **Time Series**: 5 temporal detection methods
- **Graph-based**: 3 network anomaly detection algorithms

### Performance Features
- **Async Operations**: Non-blocking algorithm execution
- **Batch Processing**: Efficient processing of large datasets
- **Memory Optimization**: Streaming processing for memory-constrained environments
- **GPU Support**: CUDA acceleration for neural network algorithms
- **Parallel Execution**: Multi-core algorithm execution

### Extensibility
- **Plugin Architecture**: Easy custom algorithm integration
- **Adapter Pattern**: Consistent interface across all algorithms
- **Configuration System**: Flexible parameter management
- **Factory Methods**: Simplified algorithm instantiation

## [0.9.0] - 2025-06-01

### Added
- Initial PyOD adapter implementation
- Basic ensemble methods
- Core algorithm factory patterns
- Configuration management system

### Changed
- Refined adapter interfaces for consistency
- Improved algorithm parameter validation
- Enhanced error handling and logging

### Fixed
- Initial performance optimizations
- Memory usage improvements
- Algorithm compatibility issues

## [0.1.0] - 2025-01-15

### Added
- Project structure for algorithm adapters
- Basic adapter interface definitions
- Foundation for extensible algorithm system

---

## Algorithm Support Matrix

| Algorithm | Library | Status | GPU Support | Streaming |
|-----------|---------|--------|-------------|-----------|
| Isolation Forest | PyOD/sklearn | ✅ Stable | ❌ | ✅ |
| Local Outlier Factor | PyOD/sklearn | ✅ Stable | ❌ | ✅ |
| One-Class SVM | sklearn | ✅ Stable | ❌ | ❌ |
| AutoEncoder | PyTorch | ✅ Stable | ✅ | ✅ |
| COPOD | PyOD | ✅ Stable | ❌ | ✅ |
| ECOD | PyOD | ✅ Stable | ❌ | ✅ |
| Deep SVDD | PyTorch | ✅ Stable | ✅ | ❌ |
| VAE | PyTorch | ✅ Stable | ✅ | ❌ |
| LSTM | PyTorch | ✅ Stable | ✅ | ✅ |

## Performance Benchmarks

### Dataset Size Scalability
- **Small (< 1K samples)**: All algorithms < 1s execution time
- **Medium (1K-100K samples)**: Most algorithms < 10s execution time
- **Large (100K-1M samples)**: Optimized algorithms < 60s execution time
- **Extra Large (> 1M samples)**: Streaming algorithms with batching

### Memory Usage
- **Statistical Methods**: O(n) memory complexity
- **Neural Networks**: O(n + m) where m is model size
- **Ensemble Methods**: O(k*n) where k is number of base algorithms

## Migration Guide

### Upgrading to 1.0.0

```python
# Before (0.9.x)
from pynomaly.algorithms import IsolationForestAdapter
detector = IsolationForestAdapter(contamination=0.1)

# After (1.0.0)
from pynomaly.algorithms.adapters.pyod import PyODAdapter
from pynomaly.core.domain.value_objects import ContaminationRate

detector = PyODAdapter(
    algorithm_name="IsolationForest",
    contamination_rate=ContaminationRate(0.1),
    n_estimators=100,
    random_state=42
)
```

## Adding Custom Algorithms

```python
from pynomaly.algorithms.base import BaseAlgorithmAdapter

class CustomAlgorithmAdapter(BaseAlgorithmAdapter):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = YourCustomAlgorithm(**params)
    
    async def fit(self, dataset: Dataset) -> None:
        data = dataset.to_numpy()
        await self._run_async(self.model.fit, data)
    
    async def predict(self, dataset: Dataset) -> DetectionResult:
        data = dataset.to_numpy()
        scores = await self._run_async(self.model.decision_function, data)
        return self._create_detection_result(dataset, scores)
```

## Dependencies

### Runtime Dependencies
- `pyod>=1.0.0`: Outlier detection algorithms
- `scikit-learn>=1.3.0`: Machine learning algorithms
- `torch>=2.0.0`: Neural network algorithms (optional)
- `numpy>=1.24.0`: Numerical computations

### Optional Dependencies
- `cupy>=12.0.0`: GPU acceleration
- `optuna>=3.0.0`: Hyperparameter optimization
- `prophet>=1.1.0`: Time series forecasting

## Contributing

When contributing algorithms:

1. **Follow Adapter Pattern**: Implement the base adapter interface
2. **Add Comprehensive Tests**: Unit tests with various dataset types
3. **Document Parameters**: Clear parameter descriptions and ranges
4. **Performance Testing**: Benchmark with different dataset sizes
5. **GPU Support**: Consider GPU acceleration where applicable

For detailed contribution guidelines, see [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Support

- **Package Documentation**: [docs/](docs/)
- **Algorithm Guide**: [docs/algorithms.md](docs/algorithms.md)
- **Performance Guide**: [docs/performance.md](docs/performance.md)
- **Issues**: [GitHub Issues](../../../issues)

[Unreleased]: https://github.com/elgerytme/Pynomaly/compare/algorithms-v1.0.0...HEAD
[1.0.0]: https://github.com/elgerytme/Pynomaly/releases/tag/algorithms-v1.0.0
[0.9.0]: https://github.com/elgerytme/Pynomaly/releases/tag/algorithms-v0.9.0
[0.1.0]: https://github.com/elgerytme/Pynomaly/releases/tag/algorithms-v0.1.0