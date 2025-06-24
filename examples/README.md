# Pynomaly Examples

This directory contains comprehensive examples demonstrating various aspects of the Pynomaly anomaly detection framework.

## Quick Start

The fastest way to get started is with the basic usage example:

```bash
python basic_usage.py
```

## Available Examples

### 1. Basic Usage (`basic_usage.py`)
**Perfect for beginners** - Learn the fundamentals of Pynomaly in under 5 minutes.

- Create and configure an anomaly detector
- Load and prepare sample data  
- Train the detector on normal data
- Detect anomalies in new data
- Understand detection results and confidence scores

**Use this example if:** You're new to Pynomaly or anomaly detection in general.

### 2. Algorithm Comparison (`algorithm_comparison.py`)  
**For method selection** - Compare different anomaly detection algorithms on the same dataset.

- Test 6 different algorithms: IsolationForest, LOF, OCSVM, COPOD, ABOD, KNN
- Evaluate performance with precision, recall, and F1-score
- Understand when to use each algorithm
- See execution time comparisons

**Use this example if:** You want to choose the best algorithm for your specific use case.

### 3. Ensemble Detection (`ensemble_detection.py`)
**For robust detection** - Combine multiple algorithms for more reliable anomaly detection.

- Create an ensemble of 5 different detectors
- Compare voting strategies: majority, unanimous, any
- Understand ensemble benefits and trade-offs
- Handle disagreements between detectors

**Use this example if:** You need highly reliable anomaly detection or want to reduce false positives.

### 4. Streaming Detection (`streaming_detection.py`)
**For real-time applications** - Process continuous data streams with real-time anomaly detection.

- Simulate real-time sensor data streams
- Handle streaming data with backpressure
- Real-time alerting and monitoring
- Ensemble streaming detection
- Performance tracking and metrics

**Use this example if:** You need to monitor data streams in real-time or build IoT monitoring systems.

### 5. Time Series Detection (`time_series_detection.py`)
**For temporal data** - Specialized anomaly detection for time series data with seasonal patterns.

- Extract temporal features (hourly, daily, weekly patterns)
- Handle seasonal decomposition
- Multi-algorithm comparison on time series
- Advanced feature engineering for temporal data
- Detect seasonal anomalies vs. trend anomalies

**Use this example if:** You work with time series data, logs, or any data with temporal patterns.

### 6. Custom Algorithm Integration (`custom_algorithm_integration.py`)
**For advanced users** - Integrate your own custom anomaly detection algorithms.

- Implement custom statistical detector
- Create algorithm adapters following the framework protocols
- Build custom ensemble methods
- Provide explanations and metadata
- Register algorithms in the framework

**Use this example if:** You want to extend Pynomaly with proprietary algorithms or research methods.

### 7. Web UI Integration (`web_ui_integration.py`)
**For web applications** - Integrate with the Progressive Web App and create dashboards.

- Multi-domain monitoring dashboards
- Real-time web UI simulation
- REST API endpoint examples
- HTMX integration patterns
- WebSocket real-time updates

**Use this example if:** You need web-based monitoring, dashboards, or user interfaces.

## CLI Examples

### Basic Workflow (`cli_basic_workflow.sh`)
Step-by-step CLI usage demonstration:
```bash
./cli_basic_workflow.sh
```

### Batch Detection (`cli_batch_detection.sh`)  
Process multiple datasets with multiple algorithms:
```bash
./cli_batch_detection.sh
```

## Sample Data

The `sample_data/` directory contains realistic datasets for different domains:

- **`normal_2d.csv`** - 2D gaussian data with labeled anomalies
- **`credit_transactions.csv`** - Financial transaction data for fraud detection  
- **`sensor_readings.csv`** - IoT sensor data with equipment failures

### Comprehensive Dataset Collection

The `sample_datasets/` directory contains an extensive collection of tabular datasets covering various anomaly detection scenarios:

#### Synthetic Datasets
- **Financial Fraud** (10K samples) - Transaction fraud with timing and amount patterns
- **Network Intrusion** (8K samples) - DDoS, port scanning, traffic anomalies  
- **IoT Sensors** (12K samples) - Environmental monitoring with sensor failures
- **Manufacturing Quality** (6K samples) - Process control and defect detection
- **E-commerce Behavior** (15K samples) - Bot detection and user behavior analysis
- **Time Series Anomalies** (5K samples) - Temporal patterns with trend changes
- **High-Dimensional** (3K samples) - Curse of dimensionality challenges

#### Real-World Datasets
- **KDD Cup 1999** (10K sample) - Network intrusion detection benchmark

Each dataset includes:
- ✅ Detailed metadata and characteristics
- 📊 Analysis scripts with domain-specific approaches
- 🧠 Algorithm recommendations
- 📋 Implementation guidelines
- 📈 Visualization examples

**Getting Started with Sample Datasets:**
```bash
# Generate all datasets
python scripts/generate_comprehensive_datasets.py

# Analyze all datasets
python scripts/analyze_dataset_comprehensive.py

# Analyze specific domains
python examples/analyze_financial_fraud.py
python examples/analyze_network_intrusion.py
```

## Running the Examples

### Prerequisites
1. Ensure Pynomaly is installed (see main README)
2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

### Python Examples
```bash
# Basic examples
python basic_usage.py
python algorithm_comparison.py  
python ensemble_detection.py

# Advanced examples
python streaming_detection.py
python time_series_detection.py
python custom_algorithm_integration.py
python web_ui_integration.py

# For verbose output with detailed explanations
python basic_usage.py --verbose
```

### CLI Examples
```bash
# Make scripts executable (Linux/Mac)
chmod +x *.sh

# Run CLI examples  
./cli_basic_workflow.sh
./cli_batch_detection.sh
```

## Example Domains

These examples cover common anomaly detection use cases:

- **Financial Fraud Detection** - Credit card transaction monitoring
- **IoT Monitoring** - Sensor anomaly detection for predictive maintenance
- **Quality Control** - Manufacturing defect detection
- **Network Security** - Intrusion detection systems
- **Time Series Analysis** - Log analysis, metrics monitoring
- **Real-time Systems** - Streaming data processing
- **Web Applications** - Dashboard and UI integration

## Learning Path

**Recommended learning sequence:**

1. 📚 **Start with `basic_usage.py`** - Learn core concepts
2. 🔬 **Try `algorithm_comparison.py`** - Understand algorithm differences  
3. 🎭 **Explore `ensemble_detection.py`** - Advanced detection techniques
4. 📡 **Run `streaming_detection.py`** - Real-time processing
5. 📈 **Test `time_series_detection.py`** - Temporal patterns
6. 💻 **Use CLI examples** - Production workflows
7. 🔧 **Advanced: `custom_algorithm_integration.py`** - Extensibility
8. 🌐 **Finally: `web_ui_integration.py`** - User interfaces

## Performance Benchmarks

Each example includes performance metrics where relevant:

- **Algorithm Comparison**: Execution time and memory usage
- **Streaming Detection**: Throughput and latency measurements
- **Time Series**: Feature extraction overhead
- **Ensemble Methods**: Voting strategy performance impact

## Advanced Usage

For production deployments, consider:

- **Streaming Detection** - Process data in real-time with backpressure handling
- **Model Persistence** - Save and load trained detectors
- **Custom Algorithms** - Integrate your own detection methods
- **Web Interface** - Use the Progressive Web App for visualization
- **API Integration** - RESTful endpoints for system integration
- **Horizontal Scaling** - Distribute detection across multiple workers

## Getting Help

- Check the main documentation in `docs/`
- Review algorithm-specific guides
- See the API reference for detailed parameter information
- Join our community for support and discussions

## Contributing Examples

We welcome additional examples! Please ensure:

- Clear documentation and comments
- Realistic sample data  
- Performance benchmarks where relevant
- Cross-platform compatibility
- Following the established patterns and structure