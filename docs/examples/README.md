# anomaly_detection Examples

This directory contains comprehensive examples demonstrating anomaly_detection's capabilities across different domains and use cases.

## Getting Started

All examples are self-contained and can be run independently. Each example includes detailed explanations, visualizations, and practical insights.

### Prerequisites

```bash
# Install anomaly_detection with examples dependencies
pip install -e ".[all]"

# Or install minimal dependencies for basic examples
pip install -e ".[examples]"

# Additional dependencies for visualization
pip install matplotlib seaborn plotly
```

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

**What it covers:**
- Fundamental anomaly_detection operations
- Loading and preparing data
- Initializing and fitting detectors
- Making predictions and interpreting results
- Performance evaluation metrics
- Algorithm comparison

**Key concepts:**
- Data preprocessing
- Model fitting and prediction
- Evaluation metrics (precision, recall, F1)
- Visualization of results

```bash
python examples/basic_usage.py
```

### 2. Time Series Detection (`time_series_detection.py`)

**What it covers:**
- Time series anomaly detection
- Seasonal pattern recognition
- Point anomalies, collective anomalies, trend changes
- Temporal context and windowing
- Performance evaluation for time series

**Key concepts:**
- Seasonal decomposition
- Change point detection
- Collective anomaly detection
- Time-aware evaluation metrics

```bash
python examples/time_series_detection.py
```

### 3. Multivariate Detection (`multivariate_detection.py`)

**What it covers:**
- High-dimensional anomaly detection
- Feature correlation analysis
- Dimensionality reduction techniques
- Ensemble methods for multivariate data
- Feature importance and selection

**Key concepts:**
- Curse of dimensionality
- Feature correlation
- Principal component analysis
- Ensemble voting strategies

```bash
python examples/multivariate_detection.py
```

### 4. Streaming Detection (`streaming_detection.py`)

**What it covers:**
- Real-time anomaly detection
- Online learning and model updates
- Concept drift detection and adaptation
- Buffer management and sliding windows
- Performance monitoring in streaming scenarios

**Key concepts:**
- Online algorithms
- Concept drift
- Adaptive thresholds
- Stream processing patterns

```bash
python examples/streaming_detection.py
```

### 5. Deep Learning Methods (`deep_learning_detection.py`)

**What it covers:**
- Neural network-based anomaly detection
- Autoencoders and variational autoencoders
- Deep SVDD and other deep methods
- GPU acceleration and optimization
- Transfer learning for anomaly detection

**Key concepts:**
- Reconstruction error
- Latent space analysis
- Deep feature learning
- Model interpretability

```bash
python examples/deep_learning_detection.py
```

### 6. Production Deployment (`production_example.py`)

**What it covers:**
- Production-ready implementations
- API integration and service architecture
- Monitoring and alerting systems
- Scalability and performance optimization
- Error handling and logging

**Key concepts:**
- Service architecture
- API design patterns
- Production monitoring
- Scalability considerations

```bash
python examples/production_example.py
```

### 7. Custom Algorithm Development (`custom_algorithm.py`)

**What it covers:**
- Creating custom anomaly detection algorithms
- Integrating with anomaly detection framework
- Algorithm validation and testing
- Performance benchmarking
- Documentation and best practices

**Key concepts:**
- Algorithm interface design
- Validation frameworks
- Performance testing
- Code organization

```bash
python examples/custom_algorithm.py
```

### 8. Industry-Specific Examples

#### Financial Fraud Detection (`finance/fraud_detection.py`)
- Credit card transaction monitoring
- Account behavior analysis
- Real-time fraud scoring
- Regulatory compliance considerations

#### IoT Sensor Monitoring (`iot/sensor_monitoring.py`)
- Industrial equipment monitoring
- Predictive maintenance
- Multi-sensor correlation
- Edge computing deployment

#### Network Security (`security/network_anomalies.py`)
- Network traffic analysis
- Intrusion detection
- Behavioral analysis
- Threat intelligence integration

#### Healthcare Monitoring (`healthcare/patient_monitoring.py`)
- Patient vital signs monitoring
- Medical imaging anomalies
- Drug interaction detection
- Privacy-preserving techniques

## Example Structure

Each example follows a consistent structure:

```python
#!/usr/bin/env python3
"""
Example Title

Brief description of what this example demonstrates.
"""

# 1. Imports and setup
import pandas as pd
import numpy as np
from anomaly_detection import AnomalyDetector

# 2. Data generation or loading
def create_sample_data():
    """Create or load sample data for the example."""
    pass

# 3. Main demonstration
def main():
    """Main example execution with step-by-step explanations."""
    print("🔍 Example Title")
    print("=" * 50)
    
    # Step 1: Data preparation
    print("1. Preparing data...")
    
    # Step 2: Model initialization
    print("2. Initializing detector...")
    
    # Step 3: Training
    print("3. Training model...")
    
    # Step 4: Prediction
    print("4. Making predictions...")
    
    # Step 5: Evaluation
    print("5. Evaluating results...")
    
    # Step 6: Visualization
    print("6. Visualizing results...")
    
    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

## Running Examples

### Individual Examples

```bash
# Run a specific example
python examples/basic_usage.py

# Run with specific parameters
python examples/time_series_detection.py --algorithm lstm_autoencoder --contamination 0.05
```

### All Examples

```bash
# Run all examples (demo mode)
python examples/run_all_examples.py

# Run all examples with full output
python examples/run_all_examples.py --verbose
```

### Jupyter Notebooks

Interactive versions of examples are available as Jupyter notebooks:

```bash
# Start Jupyter server
jupyter notebook examples/

# Available notebooks:
# - basic_usage.ipynb
# - time_series_detection.ipynb
# - multivariate_detection.ipynb
# - streaming_detection.ipynb
```

## Data Requirements

### Synthetic Data
Most examples generate synthetic data and don't require external datasets.

### Real-World Data
Some examples can optionally use real-world datasets:

- **UCI ML Repository datasets**: Available through `sklearn.datasets`
- **Time series data**: Can use your own CSV files with timestamp and value columns
- **Custom data**: Examples show how to adapt to your specific data format

### Data Format Requirements

**Tabular Data:**
```csv
timestamp,feature1,feature2,feature3,target
2023-01-01 00:00:00,1.2,3.4,5.6,normal
2023-01-01 01:00:00,1.1,3.2,5.8,normal
2023-01-01 02:00:00,5.0,1.1,2.3,anomaly
```

**Time Series Data:**
```csv
timestamp,value
2023-01-01 00:00:00,100.5
2023-01-01 01:00:00,101.2
2023-01-01 02:00:00,150.0
```

## Customization

### Modifying Examples

Examples are designed to be easily customizable:

1. **Change algorithms**: Modify the `algorithm` parameter
2. **Adjust parameters**: Change `contamination`, `window_size`, etc.
3. **Use your data**: Replace data generation with your data loading
4. **Add evaluation**: Include additional metrics or visualizations

### Creating New Examples

To create a new example:

1. Copy the structure from `basic_usage.py`
2. Focus on a specific use case or algorithm
3. Include comprehensive documentation
4. Add visualization and evaluation
5. Test with different parameters

## Performance Considerations

### Memory Usage
- Large datasets may require chunked processing
- Use `batch_size` parameter for streaming examples
- Monitor memory usage with `psutil` or similar tools

### Computation Time
- Deep learning examples may take longer to run
- Use GPU acceleration when available
- Consider reducing dataset size for quick testing

### Visualization
- Large datasets may slow down plotting
- Use sampling for visualization of big datasets
- Interactive plots work better for exploration

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -e ".[all]"
   ```

2. **Memory Errors**
   ```python
   # Reduce dataset size
   data = data.sample(n=10000)
   ```

3. **Slow Performance**
   ```python
   # Use faster algorithms for large datasets
   detector = AnomalyDetector(algorithm='isolation_forest')
   ```

4. **Visualization Issues**
   ```bash
   # For headless environments
   export MPLBACKEND=Agg
   ```

### Getting Help

- Check the main documentation
- Review example comments and docstrings
- Open issues on GitHub for bugs
- Join community discussions for questions

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the established structure and style
2. Include comprehensive documentation
3. Test with different scenarios
4. Add appropriate error handling
5. Include performance considerations

See `CONTRIBUTING.md` for detailed guidelines.

---

**Next Steps:**
1. Start with `basic_usage.py` to understand fundamentals
2. Explore domain-specific examples relevant to your use case
3. Experiment with different algorithms and parameters
4. Try the examples with your own data
5. Build upon the examples for your specific needs