# Changelog

All notable changes to the `monorepo-detection` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-07-15

### Added

#### Core Features
- **AnomalyDetector** main class for anomaly detection
- Clean architecture with domain-driven design
- Support for scikit-learn's IsolationForest as fallback algorithm
- Type-safe implementation with full type annotations

#### Package Structure
- **Core Package**: Domain entities, services, and use cases
- **Algorithms Package**: Algorithm adapters for various ML libraries
- **Services Package**: High-level application services
- Modular design allowing for easy extension

#### Dependencies
- Core dependencies: numpy, pandas, scikit-learn, pydantic, structlog
- Optional dependencies for ML, AutoML, deep learning, and more
- Support for Python 3.11, 3.12, and 3.13

#### Testing
- Comprehensive test suite with pytest
- Tests for basic functionality, configuration, and edge cases
- All tests passing with proper error handling

#### Build System
- Modern build system using Hatch
- Proper package metadata and classifiers
- Source and wheel distributions

#### Documentation
- Comprehensive README with installation and usage examples
- API documentation in docstrings
- Publishing guide for maintainers

#### CI/CD
- GitHub Actions workflow for automated testing
- Multi-platform testing (Ubuntu, Windows, macOS)
- Multi-version Python testing (3.11, 3.12, 3.13)
- Automated publishing to PyPI and TestPyPI
- Security scanning with Bandit and Safety
- Performance testing
- Code quality checks (linting, type checking)

### Technical Details

#### Algorithms Supported
- **Statistical Methods**: Isolation Forest (via scikit-learn)
- **Framework Ready**: Prepared for PyOD, TensorFlow, PyTorch, JAX integration
- **Extensible**: Algorithm adapter pattern for easy addition of new methods

#### Architecture
```
monorepo_detection/
├── core/           # Domain logic and business rules
├── algorithms/     # Algorithm implementations and adapters  
├── services/       # Application services
└── __init__.py     # Main package interface
```

#### Key Classes
- `AnomalyDetector`: Main entry point for anomaly detection
- `get_default_detector()`: Convenience function for quick setup
- Graceful fallback to sklearn when advanced services unavailable

#### Installation Options
```bash
pip install monorepo-detection           # Basic installation
pip install monorepo-detection[ml]      # With ML features
pip install monorepo-detection[automl]  # With AutoML
pip install monorepo-detection[all]     # Full installation
```

#### Usage Example
```python
from monorepo_detection import AnomalyDetector
import numpy as np

# Create detector and detect anomalies
detector = AnomalyDetector()
X = np.random.randn(1000, 10)
detector.fit(X)
anomalies = detector.predict(X)
```

### Known Limitations

- Advanced services integration pending (AutoML, explainability)
- Limited to sklearn algorithms in current fallback implementation
- Some copied modules may need refactoring for optimal integration

### Future Plans

- Integration with PyOD library for 40+ algorithms
- AutoML capabilities for automatic algorithm selection
- Explainable AI features using SHAP and LIME
- Deep learning algorithm support
- Real-time streaming detection
- Enhanced documentation and tutorials

[0.1.0]: https://github.com/elgerytme/Monorepo/releases/tag/v0.1.0