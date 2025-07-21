# Platform Examples

This directory contains comprehensive examples demonstrating the platform's capabilities across different domains and use cases including AI/ML, data engineering, and enterprise services.

## Getting Started

All examples are self-contained and can be run independently. Each example includes detailed explanations, visualizations, and practical insights.

### Prerequisites

```bash
# Install core dependencies
pip install -r requirements-prod.txt

# Install specific domain packages as needed
cd src/packages/ai/anomaly_detection && pip install -e .
cd src/packages/data/data_quality && pip install -e .

# Additional dependencies for visualization
pip install matplotlib seaborn plotly
```

## Available Examples

### AI/ML Domain Examples

#### 1. Anomaly Detection (`ai/anomaly_detection_basic.py`)
- Fundamental anomaly detection operations
- Algorithm comparison and evaluation
- Visualization of results

#### 2. Time Series Analysis (`ai/time_series_detection.py`)
- Time series anomaly detection
- Seasonal pattern recognition
- Temporal context and windowing

#### 3. ML Operations (`ai/mlops_workflow.py`)
- Model lifecycle management
- Experiment tracking
- Model deployment and monitoring

### Data Domain Examples

#### 4. Data Quality Assessment (`data/data_quality_example.py`)
- Data validation and profiling
- Quality metrics and reporting
- Data quality monitoring

#### 5. Data Engineering Pipeline (`data/etl_pipeline_example.py`)
- ETL process implementation
- Data transformation workflows
- Pipeline orchestration

#### 6. Data Observability (`data/lineage_tracking.py`)
- Data lineage tracking
- Impact analysis
- Operational monitoring

### Enterprise Examples

#### 7. Governance and Compliance (`enterprise/governance_example.py`)
- Audit logging implementation
- Compliance framework usage
- SLA monitoring

#### 8. Authentication and Authorization (`enterprise/auth_example.py`)
- User authentication flows
- Role-based access control
- Enterprise integration patterns

## Example Structure

Each example follows a consistent structure:

```python
#!/usr/bin/env python3
"""
Domain Example Template

Brief description of what this example demonstrates within its domain.
"""

# 1. Imports and setup
import sys
sys.path.append('../../src')  # Adjust path to domain packages

from packages.{domain}.{package}.application.services import SomeService

# 2. Data generation or loading
def create_sample_data():
    """Create or load sample data for the example."""
    pass

# 3. Main demonstration
def main():
    """Main example execution with step-by-step explanations."""
    print("🔍 Domain Example Title")
    print("=" * 50)
    
    # Step 1: Data preparation
    print("1. Preparing data...")
    
    # Step 2: Service initialization
    print("2. Initializing service...")
    
    # Step 3: Process execution
    print("3. Executing process...")
    
    # Step 4: Result analysis
    print("4. Analyzing results...")
    
    # Step 5: Visualization (if applicable)
    print("5. Visualizing results...")
    
    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

## Running Examples

### Running Examples

```bash
# AI/ML examples
python examples/ai/anomaly_detection_basic.py
python examples/ai/mlops_workflow.py

# Data examples  
python examples/data/data_quality_example.py
python examples/data/etl_pipeline_example.py

# Enterprise examples
python examples/enterprise/governance_example.py
python examples/enterprise/auth_example.py

# Run all examples in a domain
python examples/run_domain_examples.py --domain ai
python examples/run_domain_examples.py --domain data
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