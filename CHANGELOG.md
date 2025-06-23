# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical**: Fixed PyGOD adapter inheritance to properly extend `Detector` base class
- **Critical**: Fixed TODS adapter inheritance to properly extend `Detector` base class
- **Core**: Resolved adapter initialization issues preventing proper algorithm instantiation

### Added
- **Database Persistence**: Complete SQLAlchemy-based repository implementation
  - `DatabaseDetectorRepository` with CRUD operations
  - `DatabaseDatasetRepository` with metadata persistence
  - `DatabaseDetectionResultRepository` with score serialization
  - Cross-database JSON type handling (PostgreSQL JSONB, SQLite TEXT)
  - Cross-database UUID type handling with proper serialization
  - Database session factory with connection pooling
  - Support for both PostgreSQL and SQLite backends
- **Infrastructure**: Database models for all core entities
  - `DatasetModel` with feature and metadata support
  - `DetectorModel` with model serialization capability
  - `DetectionResultModel` with anomaly score persistence

### Changed
- **Architecture**: Enhanced infrastructure layer to 95% completion
- **Adapters**: Standardized adapter initialization patterns across all implementations
- **Dependencies**: All optional dependencies properly configured in pyproject.toml

### Technical Details
- Database repositories implement proper async/await patterns
- SQLAlchemy models use declarative base with type decorators
- Session management with proper transaction handling and rollback
- Repository pattern maintains clean separation from infrastructure details

## [0.1.0] - 2024-01-15

### Added
- Initial release of Pynomaly
- Clean architecture implementation with Domain-Driven Design
- Support for 40+ anomaly detection algorithms via PyOD and scikit-learn
- REST API with FastAPI
- Command-line interface with Typer
- Progressive Web App with HTMX, Tailwind CSS, D3.js, and Apache ECharts
- Comprehensive test suite
- Docker support for deployment
- Documentation with MkDocs

### Features
- **Domain Layer**
  - Entities: Anomaly, Detector, Dataset, DetectionResult
  - Value Objects: AnomalyScore, ContaminationRate, ConfidenceInterval
  - Domain Services: AnomalyScorer, ThresholdCalculator, FeatureValidator
  
- **Application Layer**
  - Use Cases: DetectAnomalies, TrainDetector, EvaluateModel, ExplainAnomaly
  - Services: DetectionService, EnsembleService, ModelPersistenceService
  - DTOs for data transfer
  
- **Infrastructure Layer**
  - PyOD adapter supporting 40+ algorithms
  - Scikit-learn adapter for additional algorithms
  - Data loaders for CSV and Parquet files
  - In-memory repositories with planned database support
  
- **Presentation Layer**
  - REST API with comprehensive endpoints
  - CLI with intuitive commands
  - PWA with real-time updates and visualizations

[Unreleased]: https://github.com/pynomaly/pynomaly/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pynomaly/pynomaly/releases/tag/v0.1.0