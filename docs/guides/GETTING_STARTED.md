# Getting Started with Pynomaly

Welcome to Pynomaly, a production-ready open source anomaly detection platform built with clean architecture principles and enterprise-grade features.

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Git for repository management
- Virtual environment tool (venv, conda, etc.)

### Installation

#### Option 1: Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv environments/.venv
source environments/.venv/bin/activate  # Linux/macOS
# environments\.venv\Scripts\activate   # Windows

# Install core packages
pip install -e .
```

#### Option 2: Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,test,lint]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import pynomaly; print('Installation successful')"
```

#### Option 3: Full Installation

```bash
# Install all features
pip install -e ".[all]"
```

## Basic Usage

### Python API

```python
from pynomaly import AnomalyDetector
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000)
})

# Add some anomalies
data.iloc[::100] = data.iloc[::100] * 3

# Initialize detector
detector = AnomalyDetector(algorithm='isolation_forest')

# Fit and predict
detector.fit(data)
anomalies = detector.predict(data)

# Results
print(f"Found {anomalies.sum()} anomalies out of {len(data)} samples")
```

### CLI Interface

```bash
# Show help
pynomaly --help

# Run anomaly detection on a CSV file
pynomaly detect --input data.csv --output results.csv --algorithm isolation_forest

# List available algorithms
pynomaly algorithms list

# Get algorithm information
pynomaly algorithms info --name isolation_forest
```

### API Server

```bash
# Start the API server
uvicorn pynomaly.api:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# - Swagger docs: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
```

## Configuration

### Environment Variables

```bash
# Core configuration
export PYNOMALY_ENV=development
export PYNOMALY_LOG_LEVEL=INFO
export PYNOMALY_DEBUG=true

# API configuration
export PYNOMALY_API_HOST=0.0.0.0
export PYNOMALY_API_PORT=8000

# Database configuration (optional)
export PYNOMALY_DATABASE_URL=postgresql://user:pass@localhost/pynomaly

# Redis configuration (optional)
export PYNOMALY_REDIS_URL=redis://localhost:6379/0
```

### Configuration File

Create `pynomaly.yaml` in your project root:

```yaml
# Core settings
debug: true
log_level: INFO
environment: development

# Algorithm defaults
default_algorithm: isolation_forest
contamination: 0.1
random_state: 42

# API settings
api:
  host: 0.0.0.0
  port: 8000
  workers: 1
  reload: true

# Data processing
data:
  max_samples: 100000
  chunk_size: 1000
  cache_results: true

# Security
security:
  enable_auth: false
  rate_limiting: true
  cors_origins: ["*"]
```

## Architecture Overview

Pynomaly follows Clean Architecture principles:

```
pynomaly/
├── domain/              # Business logic and entities
│   ├── entities/        # Core business objects
│   ├── value_objects/   # Immutable value objects
│   └── services/        # Domain services
├── application/         # Use cases and orchestration
│   ├── use_cases/       # Application use cases
│   ├── services/        # Application services
│   └── dto/             # Data transfer objects
├── infrastructure/      # External integrations
│   ├── adapters/        # External service adapters
│   ├── persistence/     # Data storage
│   └── config/          # Configuration
└── presentation/        # User interfaces
    ├── api/             # REST API
    ├── cli/             # Command line interface
    └── web/             # Web interface
```

## Available Algorithms

Pynomaly supports 40+ anomaly detection algorithms:

### Statistical Methods
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Elliptic Envelope
- Z-Score
- Modified Z-Score

### Machine Learning Methods
- AutoEncoder
- Variational AutoEncoder
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- k-Nearest Neighbors (kNN)
- Histogram-based Outlier Score (HBOS)

### Deep Learning Methods
- Deep SVDD
- LSTM AutoEncoder
- Convolutional AutoEncoder
- Adversarial AutoEncoder

### Ensemble Methods
- Feature Bagging
- Isolation Forest Ensemble
- LSCP (Locally Selective Combination)
- XGBOD (Extreme Gradient Boosting Outlier Detection)

## Examples

### Time Series Anomaly Detection

```python
import pandas as pd
from pynomaly import TimeSeriesDetector

# Load time series data
df = pd.read_csv('timeseries.csv', parse_dates=['timestamp'])

# Initialize time series detector
detector = TimeSeriesDetector(
    algorithm='lstm_autoencoder',
    window_size=50,
    contamination=0.05
)

# Fit and detect
detector.fit(df)
anomalies = detector.detect_anomalies(df)

# Visualize results
detector.plot_results()
```

### Multivariate Anomaly Detection

```python
from pynomaly import MultivariateDetector
import pandas as pd

# Load data
data = pd.read_csv('multivariate_data.csv')

# Initialize detector with ensemble method
detector = MultivariateDetector(
    algorithms=['isolation_forest', 'lof', 'ocsvm'],
    ensemble_method='average',
    contamination=0.1
)

# Fit and predict
detector.fit(data)
scores = detector.decision_scores(data)
predictions = detector.predict(data)

# Get feature importance
importance = detector.feature_importance()
print(importance)
```

### Streaming Anomaly Detection

```python
from pynomaly import StreamingDetector

# Initialize streaming detector
detector = StreamingDetector(
    algorithm='half_space_trees',
    window_size=1000,
    drift_detection=True
)

# Process streaming data
for batch in data_stream:
    anomalies = detector.predict(batch)
    
    # Handle concept drift
    if detector.drift_detected:
        detector.update_model(batch)
        
    print(f"Batch anomalies: {anomalies.sum()}")
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only

# Run tests with coverage
pytest --cov=pynomaly --cov-report=html

# Run tests in parallel
pytest -n auto
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,test,lint]"

# Run code quality checks
ruff check src/
black src/
mypy src/

# Run security checks
bandit -r src/
safety check

# Pre-commit hooks
pre-commit run --all-files
```

## Docker

### Development

```bash
# Build development image
docker build -t pynomaly:dev .

# Run development container
docker run -it --rm -p 8000:8000 pynomaly:dev
```

### Production

```bash
# Use docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Services will be available:
# - API: http://localhost:8000
# - Monitoring: http://localhost:3000 (Grafana)
# - Metrics: http://localhost:9090 (Prometheus)
```

## Next Steps

1. **Read the Architecture Guide**: Learn about clean architecture implementation
2. **Explore Examples**: Check the `examples/` directory for detailed use cases
3. **API Documentation**: Visit the interactive API docs at `/docs` endpoint
4. **Performance Tuning**: Read the performance optimization guide
5. **Contributing**: See `CONTRIBUTING.md` for development guidelines

## Support

- **Documentation**: Full documentation at `/docs`
- **Issues**: Report issues on GitHub
- **Discussions**: Join community discussions
- **Security**: Report security issues to security@pynomaly.org

## License

Pynomaly is licensed under the MIT License. See `LICENSE` file for details.