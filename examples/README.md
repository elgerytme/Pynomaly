# Pynomaly Examples =

This directory contains practical examples demonstrating how to use Pynomaly for various anomaly detection tasks.

## Prerequisites

Before running the examples, make sure you have Pynomaly installed:

```bash
# From the root directory
python setup_simple.py

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

## Available Examples

### Python Scripts

#### 1. **basic_usage.py** - Getting Started
The simplest example showing the core workflow:
- Creating sample data with anomalies
- Setting up a detector (Isolation Forest)
- Training and detecting anomalies
- Viewing results

```bash
python examples/basic_usage.py
```

**Expected Output:**
- Creates 104 samples with 4 known anomalies
- Detects anomalies using Isolation Forest
- Shows detected anomaly indices and values

#### 2. **algorithm_comparison.py** - Compare Multiple Algorithms
Compares different algorithms on the same dataset:
- Tests 6 different algorithms (IsolationForest, LOF, OCSVM, COPOD, ECOD, KNN)
- Evaluates using precision, recall, F1-score, and accuracy
- Shows timing information for each algorithm
- Identifies the best performer

```bash
python examples/algorithm_comparison.py
```

**Expected Output:**
- Performance metrics table for all algorithms
- Best algorithm recommendation based on F1-score
- Detailed confusion matrix for the winner

#### 3. **ensemble_detection.py** - Ensemble Methods
Demonstrates combining multiple detectors:
- Creates ensemble with 5 different algorithms
- Tests different voting strategies (majority, unanimous, any)
- Handles different types of anomalies (global, local, cluster, dependency)
- Compares ensemble vs individual detector performance

```bash
python examples/ensemble_detection.py
```

**Expected Output:**
- Individual detector performance
- Ensemble results for each voting method
- Best ensemble strategy recommendation

### More Examples Coming Soon

#### Python Scripts (Planned)
- `streaming_detection.py` - Real-time anomaly detection
- `custom_preprocessing.py` - Feature engineering for anomaly detection
- `model_persistence.py` - Saving and loading trained models
- `api_client_example.py` - Using the REST API
- `batch_processing.py` - Processing multiple datasets

#### Jupyter Notebooks (Planned)
- `01_getting_started.ipynb` - Interactive introduction
- `02_data_exploration.ipynb` - EDA for anomaly detection
- `03_algorithm_selection.ipynb` - Choosing the right algorithm
- `04_visualization.ipynb` - Visualizing anomalies
- `05_time_series_anomalies.ipynb` - Time-series specific
- `06_graph_anomalies.ipynb` - Graph-based detection
- `07_deep_learning_models.ipynb` - Neural network approaches

#### Real-World Use Cases (Planned)
- `credit_card_fraud/` - Financial fraud detection
- `network_intrusion/` - Cybersecurity anomalies
- `sensor_monitoring/` - IoT anomaly detection
- `manufacturing_defects/` - Quality control
- `healthcare_anomalies/` - Medical data analysis

## Running the Examples

### Option 1: Direct Python Execution
```bash
cd examples
python basic_usage.py
```

### Option 2: From Project Root
```bash
python examples/basic_usage.py
```

### Option 3: Using the CLI
Some examples can be replicated using the CLI:

```bash
# Create detector
python cli.py detector create --name "Example Detector" --algorithm IsolationForest

# Load data
python cli.py dataset load sample_data.csv --name "Example Data"

# Run detection
python cli.py detect run <detector_id> <dataset_id>
```

## Understanding the Output

Each example provides:
1. **Progress messages** - Shows what the example is doing
2. **Dataset information** - Size, features, anomaly statistics
3. **Results** - Detected anomalies, performance metrics
4. **Recommendations** - Best algorithms or settings

## Common Patterns

All examples follow similar patterns that you can use in your own code:

```python
# 1. Initialize container
container = create_container()

# 2. Create/load data
dataset = Dataset(name="My Data", data=df)

# 3. Create detector
detector = Detector(
    name="My Detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)

# 4. Train and detect
detection_service = container.detection_service()
await detection_service.train_detector(detector_id, dataset)
result = await detection_service.detect_anomalies(detector_id, dataset)
```

## Troubleshooting

### Import Errors
Make sure you're in the virtual environment and have installed Pynomaly:
```bash
pip install -e .
```

### Async Errors
Examples use `asyncio.run()` to handle async functions. For Jupyter notebooks, use:
```python
await main()  # Instead of asyncio.run(main())
```

### Missing Algorithms
Some algorithms require additional dependencies:
```bash
pip install pyod scikit-learn
```

## Contributing Examples

We welcome new examples! Please:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Add error handling
4. Update this README
5. Test on Windows and Linux

## Sample Datasets

Sample datasets for testing are available in `examples/sample_data/`:
- `normal_2d.csv` - Simple 2D normal distribution
- `credit_transactions.csv` - Financial transaction data
- `sensor_readings.csv` - Time-series sensor data
- (More coming soon)

## Further Resources

- [Documentation](../docs/index.md)
- [API Reference](../docs/api/index.md)
- [Algorithm Guide](../docs/guides/algorithms.md)
- [CLI Guide](../docs/guides/cli.md)