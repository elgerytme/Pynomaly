# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Comprehensive Template System**: Production-ready template framework for anomaly detection workflows
  - Complete template architecture with 2,500+ lines across 9 core template categories
  - Domain-specific preprocessing pipelines (Financial, Healthcare, IoT, Text data)
  - Algorithm comparison and ensemble selection frameworks with statistical validation
  - Executive and technical reporting templates with business intelligence integration
  - Experiment configuration system with YAML-based reproducible research standards
  - Interactive Jupyter notebook templates for end-to-end anomaly detection workflows
  - Template validation framework with comprehensive quality assurance and automated testing
  - Enterprise features: Regulatory compliance (HIPAA, AML/KYC), security, audit trails
  - Complete documentation system with usage guides, best practices, and troubleshooting

### Added
- **Advanced Algorithm Comparison Framework**: Multi-algorithm benchmarking with statistical rigor
  - Cross-validation evaluation with stratified sampling and performance metrics tracking
  - Statistical significance testing (Wilcoxon, Friedman) for algorithm comparison validation
  - Hyperparameter tuning with grid search and automated optimization capabilities
  - Algorithm characteristics analysis including complexity, interpretability, and scalability ratings
  - Performance benchmarking with execution time, memory usage, and throughput measurements
  - Comprehensive reporting with algorithm rankings, recommendations, and use-case guidance

### Added
- **Intelligent Ensemble Selection System**: Automated ensemble composition and optimization
  - Multiple ensemble strategies (voting, averaging, stacking, dynamic selection)
  - Diversity analysis with disagreement, correlation, and kappa statistics measurement
  - Automated ensemble optimization using exhaustive search, greedy, and genetic algorithms
  - Performance-based selection with multi-criteria scoring (performance, diversity, efficiency)
  - Cross-validation ensemble validation with statistical performance assessment
  - Meta-learning capabilities for optimal ensemble weight determination

### Added
- **Domain-Specific Preprocessing Pipelines**: Specialized data preparation for different industries
  - **Financial Data Processing**: AML/KYC compliance, transaction validation, risk feature engineering
  - **Healthcare Data Processing**: HIPAA anonymization, medical code standardization, clinical validation
  - **IoT Sensor Processing**: Time series resampling, sensor fusion, calibration, temporal analysis
  - **Text Data Processing**: NLP preprocessing, TF-IDF extraction, semantic similarity, sentiment analysis
  - Advanced feature engineering with domain-specific knowledge and business rule integration
  - Comprehensive data quality assessment with automated issue detection and remediation

### Added
- **Template Validation Framework**: Comprehensive quality assurance for template ecosystem
  - Multi-category validation: syntax, functionality, performance, documentation, best practices
  - Mock data testing with automated functionality verification across different data scenarios
  - Performance benchmarking with memory usage analysis and execution time measurement
  - Documentation quality assessment with docstring coverage and parameter documentation analysis
  - HTML report generation with detailed validation results and improvement recommendations
  - Automated template discovery and batch validation capabilities

### Added
- **Enhanced Model Persistence Framework**: Production-ready model lifecycle management infrastructure
  - Multi-format serialization support (Pickle, Joblib, ONNX, TensorFlow SavedModel, PyTorch State Dict)
  - Semantic versioning system with automatic version management and compatibility checking
  - Comprehensive model registry with centralized cataloging and metadata management
  - Performance metrics tracking with automated comparison and benchmarking capabilities
  - Model storage optimization with compression, encryption, and integrity verification
  - Cross-platform deployment bundles with dependency management and example generation

### Added
- **Model Registry System**: Centralized model discovery and access control
  - Role-based access control with user/group permissions (read, write, admin)
  - Advanced search and filtering with domain-specific model recommendations
  - Model lifecycle tracking across development, staging, and production environments
  - Performance-based model comparison with automated quality assessment
  - Model archival and version management with complete audit trails
  - Multi-registry support with configurable access policies

### Added
- **Autonomous Mode Preprocessing Integration**: Revolutionary intelligent data preparation in autonomous detection
  - Automatic data quality assessment with 10+ quality metrics and issue detection
  - Context-aware preprocessing strategy selection (auto, aggressive, conservative, minimal)
  - Seamless integration within existing autonomous workflow without breaking changes
  - Comprehensive quality reporting with preprocessing metadata and improvement tracking
  - Time-budgeted processing with configurable limits and early termination
  - Enhanced CLI options: `--preprocess/--no-preprocess`, `--quality-threshold`, `--max-preprocess-time`, `--preprocessing-strategy`

### Added
- **Autonomous Quality Analyzer**: Comprehensive data quality assessment component
  - Missing values, outliers, duplicates, constant features, infinite values detection
  - Poor scaling, high cardinality, imbalanced categories analysis
  - Quality scoring (0.0-1.0) with severity assessment and improvement potential estimation
  - Processing time and memory impact predictions for informed decision making
  - Intelligent pipeline recommendation generation based on detected issues

### Added
- **Autonomous Preprocessing Orchestrator**: Intelligent preprocessing application manager
  - Quality threshold evaluation and preprocessing decision logic
  - Processing time budget enforcement with graceful degradation
  - Error handling and fallback mechanisms for robust operation
  - Applied step tracking and metadata generation for comprehensive reporting
  - Strategy-based preprocessing with data characteristic consideration

### Changed
- **Enhanced Autonomous Detection Service**: Extended workflow with preprocessing integration
  - New workflow step: Quality assessment and intelligent preprocessing between data loading and profiling
  - Enhanced DataProfile with quality metrics, preprocessing status, and metadata
  - Backward-compatible AutonomousConfig with new preprocessing configuration options
  - Improved algorithm recommendations based on preprocessing context

### Changed  
- **Enhanced CLI Output**: Comprehensive preprocessing information display
  - Data Quality & Preprocessing section with quality scores and applied steps
  - Preprocessing metadata table showing operations, actions, and affected columns
  - Quality issue detection and severity visualization with color coding
  - Shape transformation reporting (before/after preprocessing)

### Documentation
- **Comprehensive Preprocessing Integration Guide**: Complete autonomous preprocessing documentation
  - `docs/autonomous/preprocessing-integration.md` - 600+ line comprehensive guide
  - Usage examples, configuration options, strategy selection guidance
  - Performance considerations, troubleshooting, and best practices
  - Integration points, output formats, and programmatic usage examples

### Testing
- **Extensive Autonomous Preprocessing Tests**: Comprehensive test coverage for integration
  - `tests/test_autonomous_preprocessing.py` - 400+ lines of integration testing
  - Quality analyzer component testing with specific issue detection validation
  - Preprocessing orchestrator testing with workflow and error handling validation
  - End-to-end integration testing with autonomous service workflow validation

### Added
- **CLI Configuration Generation**: New `generate-config` command for creating test/experiment configuration files
  - Support for test, experiment, and autonomous configuration types
  - JSON and YAML output formats with comprehensive CLI options
  - Includes usage examples and workflow descriptions for each configuration type
  - Configurable parameters: detector algorithm, dataset path, contamination rate, cross-validation settings
  - Rich console output with colored formatting and validation

## [0.4.0] - 2025-06-24

### Added
- **Comprehensive Data Preprocessing CLI**: Complete command-line interface for data preprocessing operations
  - `pynomaly data clean` - Advanced data cleaning with 10+ missing value strategies, 5+ outlier handling methods
  - `pynomaly data transform` - Feature transformation with 5+ scaling methods, 6+ encoding strategies, feature engineering
  - `pynomaly data pipeline` - Pipeline management for creating, saving, loading, and applying preprocessing workflows
  - Dry-run mode for previewing changes without applying them
  - Save-as functionality to preserve original datasets while creating cleaned versions

### Added
- **Enhanced Data Quality Integration**: Seamless integration between quality analysis and preprocessing
  - Extended `pynomaly dataset quality` command with specific preprocessing command suggestions
  - Context-aware recommendations based on detected data quality issues
  - Intelligent strategy selection for missing values, outliers, and duplicates
  - Direct command generation for immediate preprocessing application

### Added
- **Production-Ready Preprocessing Infrastructure**: Enterprise-grade data preparation capabilities
  - Memory-efficient processing with data type optimization
  - Cross-platform compatibility with error handling and validation
  - Performance monitoring and progress tracking for large datasets
  - Comprehensive logging and debugging support

### Added
- **Reusable Preprocessing Pipelines**: Systematic approach to data preparation
  - JSON-based pipeline configuration for reproducible preprocessing
  - Pipeline versioning and management with save/load functionality
  - Step-by-step pipeline execution with enable/disable controls
  - Template pipelines for common data types (financial, IoT, e-commerce)

### Added
- **Advanced Feature Engineering**: Sophisticated transformation capabilities
  - Polynomial feature generation for interaction modeling
  - Feature selection with variance, correlation, and statistical methods
  - Categorical encoding with target, frequency, and binary encoding
  - Column name normalization and data type optimization

### Changed
- **Enhanced CLI User Experience**: Improved usability and workflow integration
  - Updated quickstart guide to include preprocessing steps
  - Extended help documentation with comprehensive examples
  - Integration with existing detector training and anomaly detection workflows
  - Consistent command structure and option naming across all preprocessing operations

### Documentation
- **Comprehensive Preprocessing Documentation**: Complete reference and examples
  - `docs/cli/preprocessing.md` - 400+ line comprehensive CLI reference guide
  - `examples/preprocessing_cli_examples.py` - Complete demonstration script with real-world scenarios
  - Best practices guide for data cleaning, transformation, and pipeline management
  - Performance considerations and troubleshooting guidance

### Testing
- **Extensive CLI Testing**: Comprehensive test coverage for preprocessing functionality
  - `tests/test_preprocessing_cli.py` - 200+ lines of CLI command testing
  - Unit tests for data cleaning, transformation, and pipeline management
  - Integration tests for complete preprocessing workflows
  - Error handling and edge case validation

### Infrastructure
- **Modular CLI Architecture**: Clean separation of preprocessing concerns
  - New `preprocessing.py` CLI module with organized command structure
  - Integration with existing CLI container and dependency injection
  - Consistent error handling and user feedback patterns
  - Extensible design for future preprocessing enhancements

## [0.3.0] - 2025-06-24

### Added
- **Comprehensive Dataset Collection**: 69,000+ samples across 8 diverse anomaly detection datasets
  - Financial Fraud Detection (10K samples, 9 features, 2% anomalies) - Transaction fraud with timing/amount patterns
  - Network Intrusion Detection (8K samples, 11 features, 5% anomalies) - DDoS, port scanning, traffic anomalies
  - IoT Sensor Monitoring (12K samples, 10 features, 3% anomalies) - Environmental monitoring with sensor failures
  - Manufacturing Quality Control (6K samples, 11 features, 8% anomalies) - Process control and defect detection
  - E-commerce Behavior Analysis (15K samples, 12 features, 4% anomalies) - Bot detection and user behavior
  - Time Series Anomalies (5K samples, 10 features, 6% anomalies) - Temporal patterns with trend changes
  - High-Dimensional Data (3K samples, 54 features, 10% anomalies) - Curse of dimensionality challenges
  - KDD Cup 1999 (10K samples, real-world network intrusion benchmark)

### Added
- **Analysis Tools & Scripts**: Complete dataset analysis infrastructure
  - `scripts/generate_comprehensive_datasets.py` - Automated dataset generation with realistic patterns
  - `scripts/analyze_dataset_comprehensive.py` - Multi-dataset analysis with algorithm recommendations
  - `examples/analyze_financial_fraud.py` - Domain-specific fraud detection analysis
  - `examples/analyze_network_intrusion.py` - Network security analysis with attack pattern detection
  - Individual metadata files with detailed dataset characteristics

### Added
- **Documentation & Guides**: Comprehensive analysis documentation
  - `docs/guides/dataset-analysis-guide.md` - 200+ line comprehensive analysis guide
  - Algorithm selection matrix for each dataset type
  - Domain-specific feature engineering approaches
  - Implementation strategies and best practices
  - Production deployment considerations

### Added
- **Algorithm Recommendations**: Domain-specific algorithm guidance
  - Financial Fraud: IsolationForest (primary), LocalOutlierFactor, OneClassSVM
  - Network Intrusion: IsolationForest (primary), EllipticEnvelope, PyOD.ABOD
  - IoT Sensors: LocalOutlierFactor (primary), EllipticEnvelope, PyOD.KNN
  - Manufacturing: IsolationForest (primary), EllipticEnvelope, PyOD.OCSVM
  - E-commerce: LocalOutlierFactor (primary), IsolationForest, PyOD.COPOD
  - Time Series: LocalOutlierFactor (primary), IsolationForest, EllipticEnvelope
  - High-Dimensional: IsolationForest (primary), PyOD.PCA, PyOD.ABOD

### Documentation
- Updated `examples/README.md` with comprehensive dataset collection information
- Added detailed dataset characteristics and usage instructions
- Provided analysis scripts and visualization examples for each domain
- Included implementation guidelines and performance considerations

### Infrastructure
- Organized datasets in structured directory with synthetic and real-world subdirectories
- Added metadata JSON files for each dataset with characteristics and recommendations
- Created master dataset metadata with comprehensive overview

## [0.2.0] - 2025-06-23

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