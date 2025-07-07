# Quick Start Tutorial

This tutorial will get you up and running with Pynomaly in 10 minutes. You'll learn how to detect anomalies using both the Python API and CLI interface.

## Prerequisites

- Python 3.11 or higher
- 5-10 minutes of your time

## Installation

### Step 1: Clone and Install
```bash
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv environments/.venv
source environments/.venv/bin/activate  # Linux/macOS
# OR
environments\.venv\Scripts\activate     # Windows

# Install with basic features
pip install -e ".[server]"
```

### Step 2: Verify Installation
```bash
python -c "import pynomaly; print('✓ Installation successful')"
pynomaly --version
```

## Method 1: Python API (Recommended)

### Simple Anomaly Detection Example
```python
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

# Step 1: Create sample data with outliers
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 3))
outliers = np.random.uniform(-4, 4, (50, 3))
data = np.vstack([normal_data, outliers])

# Step 2: Create dataset
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
dataset = Dataset(name="sample_data", data=df)

print(f"Dataset: {len(dataset.data)} samples, {len(dataset.data.columns)} features")

# Step 3: Create and configure detector
adapter = PyODAdapter(
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.05,  # Expected 5% outliers
        "n_estimators": 100,
        "random_state": 42
    }
)

# Step 4: Train the detector
print("Training detector...")
detector = await adapter.train(dataset)
print(f"✓ Detector trained: {detector.name}")

# Step 5: Detect anomalies
print("Detecting anomalies...")
result = await adapter.detect(dataset)

print(f"\n🎯 Results:")
print(f"   Total samples: {len(dataset.data)}")
print(f"   Anomalies found: {len(result.anomalies)}")
print(f"   Detection rate: {len(result.anomalies)/len(dataset.data)*100:.1f}%")

# Step 6: Examine anomalies
print(f"\n📊 Top 5 anomalies:")
for i, anomaly in enumerate(result.anomalies[:5]):
    print(f"   {i+1}. Row {anomaly.index}: Score {anomaly.score.value:.3f}")
```

### Run the example:
```bash
python -c "
import asyncio
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

async def main():
    # Sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 3))
    outliers = np.random.uniform(-4, 4, (50, 3))
    data = np.vstack([normal_data, outliers])
    
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    dataset = Dataset(name='sample_data', data=df)
    
    # Detect anomalies
    adapter = PyODAdapter('IsolationForest', {'contamination': 0.05})
    detector = await adapter.train(dataset)
    result = await adapter.detect(dataset)
    
    print(f'✓ Found {len(result.anomalies)} anomalies out of {len(dataset.data)} samples')

asyncio.run(main())
"
```

## Method 2: CLI Interface

### Step 1: Prepare Data
Create a sample CSV file:
```bash
python -c "
import numpy as np
import pandas as pd

np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 3))
outliers = np.random.uniform(-4, 4, (50, 3))
data = np.vstack([normal_data, outliers])

df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
df.to_csv('sample_data.csv', index=False)
print('✓ Created sample_data.csv')
"
```

### Step 2: Detect Anomalies via CLI
```bash
# Quick detection with default settings
pynomaly detect sample_data.csv --algorithm IsolationForest --contamination 0.05

# With detailed output
pynomaly detect sample_data.csv \
    --algorithm IsolationForest \
    --contamination 0.05 \
    --output results.json \
    --format json \
    --verbose
```

### Step 3: View Results
```bash
# View results summary
cat results.json | head -20

# Export to Excel (if installed)
pynomaly export results.json report.xlsx --format excel --include-charts
```

## Method 3: Web Interface

### Step 1: Start the Server
```bash
pynomaly server start --port 8000
```

### Step 2: Access Web Interface
Open your browser and go to: http://localhost:8000

### Step 3: Upload and Analyze
1. Click "Upload Dataset" 
2. Select your `sample_data.csv` file
3. Choose "Isolation Forest" algorithm
4. Set contamination to 0.05
5. Click "Detect Anomalies"
6. View results in the dashboard

## Method 4: REST API

### Step 1: Start API Server
```bash
pynomaly server start --port 8000
```

### Step 2: Upload Dataset
```bash
curl -X POST "http://localhost:8000/api/datasets/upload" \
     -F "file=@sample_data.csv" \
     -F "name=sample_dataset"
```

### Step 3: Create Detector
```bash
curl -X POST "http://localhost:8000/api/detectors" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "isolation_forest",
       "algorithm": "IsolationForest", 
       "parameters": {"contamination": 0.05}
     }'
```

### Step 4: Run Detection
```bash
curl -X POST "http://localhost:8000/api/detection/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "detector_id": "DETECTOR_ID_FROM_STEP_3",
       "dataset_id": "DATASET_ID_FROM_STEP_2"
     }'
```

## Exploring Different Algorithms

### Compare Multiple Algorithms
```python
import asyncio
from pynomaly.infrastructure.adapters.optimized_pyod_adapter import AsyncAlgorithmExecutor

async def compare_algorithms():
    # Your dataset here
    dataset = create_your_dataset()
    
    # Compare algorithms
    executor = AsyncAlgorithmExecutor(max_concurrent=3)
    algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
    
    results = await executor.execute_multiple_algorithms(algorithms, dataset)
    
    for algo_name, result in results:
        if result:
            print(f"{algo_name}: {len(result.anomalies)} anomalies")

asyncio.run(compare_algorithms())
```

### Algorithm Recommendations by Data Type

#### Numerical Data (most common)
```python
# Isolation Forest - Fast, good for large datasets
adapter = PyODAdapter("IsolationForest", {"contamination": 0.1})

# Local Outlier Factor - Good for local anomalies  
adapter = PyODAdapter("LocalOutlierFactor", {"n_neighbors": 20})
```

#### High-dimensional Data
```python
# PCA-based detection
adapter = PyODAdapter("PCA", {"contamination": 0.1})

# Feature Bagging for ensemble approach
adapter = PyODAdapter("FeatureBagging", {"contamination": 0.1})
```

#### Time Series Data
```python
# Histogram-based Outlier Score
adapter = PyODAdapter("HBOS", {"contamination": 0.1})
```

## Performance Tips

### For Large Datasets (100K+ rows)
```python
from pynomaly.infrastructure.adapters.optimized_pyod_adapter import OptimizedPyODAdapter

# Use optimized adapter
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_batch_processing=True,
    batch_size=10000,
    enable_feature_selection=True,
    memory_optimization=True
)
```

### For Memory-Constrained Environments
```python
from pynomaly.infrastructure.performance.memory_manager import AdaptiveMemoryManager

# Enable automatic memory management
memory_manager = AdaptiveMemoryManager(
    target_memory_percent=80.0,
    enable_automatic_optimization=True
)
await memory_manager.start_monitoring()
```

## Common Use Cases

### 1. Fraud Detection (Financial Data)
```python
# Optimized for financial transactions
adapter = PyODAdapter(
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.001,  # Very low fraud rate
        "n_estimators": 200,     # Higher accuracy
        "random_state": 42
    }
)
```

### 2. Network Security (Log Analysis)
```python
# Good for network intrusion detection
adapter = PyODAdapter(
    algorithm="LocalOutlierFactor", 
    parameters={
        "n_neighbors": 50,       # Larger neighborhood
        "contamination": 0.01    # Low intrusion rate
    }
)
```

### 3. Quality Control (Manufacturing)
```python
# Sensitive to process variations
adapter = PyODAdapter(
    algorithm="OneClassSVM",
    parameters={
        "nu": 0.05,            # Expected defect rate
        "gamma": "scale"       # Automatic scaling
    }
)
```

### 4. IoT Sensor Data
```python
# Real-time anomaly detection
adapter = PyODAdapter(
    algorithm="LODA",          # Lightweight Online Detector
    parameters={
        "contamination": 0.02,
        "n_bins": 10           # Fast binning approach
    }
)
```

## Next Steps

### Learn More
1. **Advanced Features**: Check out [advanced tutorials](../tutorials/)
2. **API Documentation**: Full API reference at [docs/api/](../api/)
3. **Performance**: Optimization guide at [performance guide](../guides/PERFORMANCE_OPTIMIZATION.md)

### Real Projects
1. **Load your own data**: Replace sample data with your CSV files
2. **Tune parameters**: Experiment with different contamination rates
3. **Try algorithms**: Compare results across different algorithms
4. **Production deployment**: Use the REST API for applications

### Get Help
- **Documentation**: https://pynomaly.readthedocs.io
- **Examples**: Check the `examples/` directory
- **Issues**: https://github.com/pynomaly/pynomaly/issues
- **Discussions**: https://github.com/pynomaly/pynomaly/discussions

## Common Issues & Solutions

### Import Errors
```bash
# Ensure you're in the virtual environment
source environments/.venv/bin/activate

# Reinstall if needed
pip install -e ".[server]"
```

### Memory Issues
```python
# Use chunked processing for large files
from pynomaly.infrastructure.data_loaders.optimized_csv_loader import OptimizedCSVLoader

loader = OptimizedCSVLoader(chunk_size=10000, memory_optimization=True)
dataset = await loader.load("large_file.csv")
```

### Performance Issues
```bash
# Run performance diagnostics
pynomaly perf benchmark --suite quick

# Monitor resource usage
pynomaly perf monitor
```

### Algorithm Selection
```python
# List all available algorithms
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

print("Available algorithms:")
for algo in PyODAdapter.get_available_algorithms():
    print(f"  - {algo}")
```

Congratulations! 🎉 You've successfully detected anomalies with Pynomaly. You're now ready to apply these techniques to your own datasets and explore the advanced features.

## Quick Reference Card

```python
# Basic Pattern
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

# 1. Create dataset
dataset = Dataset(name="my_data", data=df)

# 2. Create detector  
adapter = PyODAdapter("IsolationForest", {"contamination": 0.1})

# 3. Train
detector = await adapter.train(dataset)

# 4. Detect
result = await adapter.detect(dataset)

# 5. Analyze
print(f"Found {len(result.anomalies)} anomalies")
```

Save this pattern - it works for 90% of anomaly detection tasks!
