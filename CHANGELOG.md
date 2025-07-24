# Changelog

All notable changes to the monorepo data intelligence monorepo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive monorepo organization and governance system
- Automated repository structure enforcement with pre-commit hooks
- Standardized package documentation across all components
- CHANGELOG.md files for repository and all packages
- Enhanced developer experience with consistent tooling
- Complete AutoML service implementation with Optuna optimization
- Ray Tune distributed optimization support
- Comprehensive database persistence for AutoML experiments
- Background job management for long-running optimizations
- Enhanced ensemble methods with weighted voting and stacking
- MLOps integration with experiment tracking
- Container integration fixes for dependency injection

### Changed
- Reorganized repository structure following monorepo best practices
- Standardized package structure with consistent templates
- Improved documentation quality and completeness
- Enhanced governance automation and validation
- Upgraded to Development Status Beta (from Alpha)
- Enhanced package metadata and keywords for PyPI
- Updated project URLs to correct GitHub repository

### Fixed
- Removed 26,097+ build artifacts and cache files
- Consolidated scattered configuration files
- Standardized inconsistent package layouts
- Improved repository organization and cleanliness
- AutoML container integration and DI registration issues
- Circular dependencies in domain layer
- Package structure consolidation and duplicates removal
- OpenAPI schema generation and Pydantic forward references

## [0.1.0] - 2025-07-14

### Added
- Initial release of data intelligence monorepo
- Clean Architecture implementation with Domain-Driven Design
- PyOD integration with 40+ detection algorithms
- FastAPI REST API with 65+ endpoints
- Progressive Web Application with HTMX and Tailwind CSS
- Command-line interface with Typer and Rich formatting
- Comprehensive testing framework with 85%+ coverage
- Production-ready infrastructure with monitoring and security
- Authentication system with JWT support
- Export functionality for CSV, JSON, and Excel formats
- Enterprise features including multi-tenancy and RBAC
- Docker containerization and deployment configurations
- Comprehensive documentation and developer guides

### Core Features
- Domain entities: Detector, Dataset, Anomaly, DetectionResult
- Value objects: ContaminationRate, ConfidenceInterval, AnomalyScore
- Algorithm adapters: PyOD, scikit-learn, PyTorch, TensorFlow, JAX
- Repository pattern for clean data access
- Dependency injection container for modular architecture
- Event-driven architecture with observer patterns

### Supported Algorithms
- Statistical: Isolation Forest, Local Outlier Factor, One-Class SVM
- Probabilistic: Gaussian Mixture Model, COPOD, ECOD
- Neural Networks: AutoEncoder, Variational AutoEncoder, Deep SVDD
- Ensemble Methods: Feature Bagging, Model Combination
- Time Series: Seasonal Decomposition, Change Point Detection
- Graph-based: PyGOD integration for network detection

### Infrastructure
- Clean monorepo structure with package organization
- Hatch build system with modern Python tooling
- Pre-commit hooks with Ruff, Black, MyPy, Bandit
- GitHub Actions CI/CD with automated testing
- Security scanning with dependency vulnerability checks
- Performance benchmarking and monitoring
- Documentation generation with MkDocs Material

---

## Package Changelogs

For detailed changes specific to each package, see:

- [Core Package](src/packages/core/CHANGELOG.md)
- [Algorithms Package](src/packages/algorithms/CHANGELOG.md)
- [Infrastructure Package](src/packages/infrastructure/CHANGELOG.md)
- [Interfaces Package](src/packages/interfaces/CHANGELOG.md)
- [MLOps Package](src/packages/mlops/CHANGELOG.md)

## Version History Summary

| Version | Release Date | Major Changes |
|---------|--------------|---------------|
| 0.1.0   | 2025-07-14   | Initial production release with full feature set |

## Contributing

When making changes to this repository:

1. **Update Relevant Changelogs**: Update both this main changelog and package-specific changelogs
2. **Follow Format**: Use Keep a Changelog format with proper categorization
3. **Semantic Versioning**: Follow semantic versioning principles for version numbers
4. **Clear Descriptions**: Write clear, user-focused change descriptions
5. **Link Issues**: Reference GitHub issues and pull requests where applicable

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md).

[Unreleased]: https://github.com/elgerytme/data_intelligence/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/elgerytme/data_intelligence/releases/tag/v0.1.0