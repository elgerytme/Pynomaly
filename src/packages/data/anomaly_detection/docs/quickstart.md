# Quickstart Template

Get started with anomaly detection in under 5 minutes! This page provides copy-paste templates and interactive examples to help you immediately detect anomalies in your data.

## 🚀 5-Minute Quickstart

Choose your scenario and copy the complete working example:

=== "I have CSV data"
    ```python title="detect_from_csv.py"
    import pandas as pd
    import numpy as np
    from anomaly_detection import DetectionService
    
    # Load your data
    data = pd.read_csv('your_data.csv')
    
    # Select numeric columns (modify as needed)
    numeric_data = data.select_dtypes(include=[np.number]).values
    
    # Detect anomalies
    service = DetectionService()
    result = service.detect(
        data=numeric_data,
        algorithm='isolation_forest',
        contamination=0.1  # Expect 10% anomalies
    )
    
    # Show results
    print(f"✅ Detection complete!")
    print(f"📊 Analyzed {result.total_samples} samples")
    print(f"🚨 Found {result.anomaly_count} anomalies")
    print(f"⏱️ Processing time: {result.processing_time:.2f}s")
    
    # Get anomaly indices
    anomaly_indices = np.where(result.predictions == -1)[0]
    print(f"🔍 Anomaly rows: {anomaly_indices.tolist()}")
    
    # Save results
    data['anomaly_score'] = result.scores
    data['is_anomaly'] = result.predictions == -1
    data.to_csv('results_with_anomalies.csv', index=False)
    print("💾 Results saved to 'results_with_anomalies.csv'")
    ```

=== "I have NumPy arrays"
    ```python title="detect_from_numpy.py"
    import numpy as np
    from anomaly_detection import DetectionService
    
    # Your data (replace with your actual data)
    # data = np.load('your_data.npy')  # Load from file
    # OR create sample data for testing:
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 5))
    anomalies = np.random.normal(3, 0.5, (50, 5))
    data = np.vstack([normal_data, anomalies])
    np.random.shuffle(data)
    
    # Detect anomalies
    service = DetectionService()
    result = service.detect(
        data=data,
        algorithm='isolation_forest',
        contamination=0.05  # Expect 5% anomalies
    )
    
    # Analyze results
    print(f"✅ Detection complete!")
    print(f"📊 Data shape: {data.shape}")
    print(f"🚨 Anomalies found: {result.anomaly_count}")
    print(f"📈 Anomaly rate: {result.anomaly_count/len(data)*100:.1f}%")
    
    # Get top 10 most anomalous samples
    top_anomalies = np.argsort(result.scores)[-10:]
    print(f"🔝 Top anomaly indices: {top_anomalies.tolist()}")
    print(f"🔥 Top anomaly scores: {result.scores[top_anomalies]}")
    ```

=== "I want real-time detection"
    ```python title="streaming_detection.py"
    import numpy as np
    from anomaly_detection import StreamingService
    import time
    
    # Initialize streaming service
    streaming = StreamingService()
    
    def on_anomaly_detected(sample, score, timestamp):
        """Callback when anomaly is detected"""
        print(f"🚨 ANOMALY DETECTED at {timestamp}")
        print(f"   Sample: {sample}")
        print(f"   Score: {score:.3f}")
        print(f"   Action: Alert sent!")
    
    # Configure detection
    streaming.configure(
        algorithm='isolation_forest',
        window_size=100,
        contamination=0.05,
        callback=on_anomaly_detected
    )
    
    # Simulate real-time data stream
    print("🔄 Starting real-time anomaly detection...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        for i in range(1000):
            # Generate sample (replace with your data source)
            if np.random.random() < 0.95:
                # Normal sample
                sample = np.random.normal(0, 1, 5)
            else:
                # Anomalous sample
                sample = np.random.normal(4, 0.5, 5)
            
            # Process sample
            result = streaming.process_sample(sample)
            
            if not result.is_anomaly:
                print(f"✅ Sample {i+1}: Normal (score: {result.score:.3f})")
            
            time.sleep(0.1)  # Simulate real-time delay
            
    except KeyboardInterrupt:
        print("\n⏹️ Streaming stopped by user")
        streaming.stop()
    ```

=== "I want to compare algorithms"
    ```python title="algorithm_comparison.py"
    import numpy as np
    from anomaly_detection import DetectionService
    import matplotlib.pyplot as plt
    
    # Create test data
    np.random.seed(42)
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 200)
    anomalies = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 20)
    data = np.vstack([normal_data, anomalies])
    true_labels = np.hstack([np.ones(200), -np.ones(20)])
    
    # Test different algorithms
    algorithms = [
        'isolation_forest',
        'lof',
        'one_class_svm'
    ]
    
    service = DetectionService()
    results = {}
    
    print("🔬 Comparing Anomaly Detection Algorithms")
    print("=" * 50)
    
    for algorithm in algorithms:
        print(f"\n🧪 Testing {algorithm.upper()}...")
        
        result = service.detect(
            data=data,
            algorithm=algorithm,
            contamination=0.1
        )
        
        # Calculate accuracy
        accuracy = np.mean(result.predictions == true_labels)
        
        results[algorithm] = {
            'result': result,
            'accuracy': accuracy
        }
        
        print(f"   ✅ Accuracy: {accuracy*100:.1f}%")
        print(f"   🚨 Detected: {result.anomaly_count} anomalies")
        print(f"   ⏱️ Time: {result.processing_time:.3f}s")
    
    # Find best algorithm
    best_algo = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\n🏆 Best algorithm: {best_algo.upper()}")
    print(f"   Accuracy: {results[best_algo]['accuracy']*100:.1f}%")
    ```

## 📋 Copy-Paste Templates

### Basic Detection Template
```python title="basic_template.py"
# STEP 1: Import and setup
from anomaly_detection import DetectionService
import numpy as np

# STEP 2: Prepare your data
# Replace this with your actual data loading
data = your_data_here  # Shape: (n_samples, n_features)

# STEP 3: Detect anomalies
service = DetectionService()
result = service.detect(
    data=data,
    algorithm='isolation_forest',  # or 'lof', 'one_class_svm'
    contamination=0.1  # Expected percentage of anomalies
)

# STEP 4: Get results
anomaly_indices = np.where(result.predictions == -1)[0]
print(f"Found {len(anomaly_indices)} anomalies")
```

### Ensemble Detection Template
```python title="ensemble_template.py"
# STEP 1: Import ensemble service
from anomaly_detection import EnsembleService

# STEP 2: Your data
data = your_data_here

# STEP 3: Use multiple algorithms for better accuracy
ensemble = EnsembleService()
result = ensemble.detect(
    data=data,
    algorithms=['isolation_forest', 'lof', 'one_class_svm'],
    method='voting',  # or 'averaging', 'stacking'
    contamination=0.1
)

# STEP 4: More reliable results
print(f"Ensemble detected {result.anomaly_count} anomalies")
```

### Batch Processing Template
```python title="batch_template.py"
import glob
from anomaly_detection import DetectionService

# Process multiple files
service = DetectionService()
data_files = glob.glob('data/*.csv')

for file_path in data_files:
    print(f"Processing {file_path}...")
    
    # Load data (adjust loading method as needed)
    data = pd.read_csv(file_path).select_dtypes(include=[np.number]).values
    
    # Detect anomalies
    result = service.detect(data, algorithm='isolation_forest')
    
    # Save results
    output_file = file_path.replace('.csv', '_anomalies.csv')
    # ... save logic here ...
    
    print(f"✅ Found {result.anomaly_count} anomalies, saved to {output_file}")
```

## 🎯 Interactive Quickstart Wizard

<div class="interactive-demo">
    <div class="demo-controls">
        <h3>📝 Quickstart Configuration</h3>
        <label>
            <strong>1. Data Source:</strong>
            <select id="data-source">
                <option value="csv">CSV File</option>
                <option value="numpy">NumPy Array</option>
                <option value="pandas">Pandas DataFrame</option>
                <option value="streaming">Real-time Stream</option>
            </select>
        </label>
        <br><br>
        <label>
            <strong>2. Algorithm:</strong>
            <select id="algorithm">
                <option value="isolation_forest">Isolation Forest (Recommended)</option>
                <option value="lof">Local Outlier Factor</option>
                <option value="one_class_svm">One-Class SVM</option>
                <option value="ensemble">Ensemble (Multiple algorithms)</option>
            </select>
        </label>
        <br><br>
        <label>
            <strong>3. Expected Anomaly Rate:</strong>
            <select id="contamination">
                <option value="0.01">1% (Very rare anomalies)</option>
                <option value="0.05">5% (Uncommon anomalies)</option>
                <option value="0.1">10% (Common anomalies)</option>
                <option value="0.2">20% (Many anomalies)</option>
            </select>
        </label>
        <br><br>
        <button class="demo-button" onclick="generateQuickstartCode()">Generate My Code</button>
    </div>
    <div class="demo-output" id="quickstart-output">
        <p>Configure your settings above and click "Generate My Code" to get a personalized quickstart template!</p>
    </div>
</div>

<script>
function generateQuickstartCode() {
    const dataSource = document.getElementById('data-source').value;
    const algorithm = document.getElementById('algorithm').value;
    const contamination = document.getElementById('contamination').value;
    
    let code = '';
    let description = '';
    
    // Base imports
    const imports = algorithm === 'ensemble' 
        ? 'from anomaly_detection import EnsembleService'
        : 'from anomaly_detection import DetectionService';
    
    // Data loading section
    let dataLoading = '';
    switch(dataSource) {
        case 'csv':
            dataLoading = `import pandas as pd
import numpy as np

# Load your CSV file
data = pd.read_csv('your_file.csv')
# Select numeric columns for analysis
numeric_data = data.select_dtypes(include=[np.number]).values`;
            description = 'Perfect for analyzing CSV files with mixed data types.';
            break;
        case 'numpy':
            dataLoading = `import numpy as np

# Load your NumPy data
data = np.load('your_data.npy')  # or however you load your data
# data = your_numpy_array  # if already in memory`;
            description = 'Ideal for scientific data or pre-processed numeric arrays.';
            break;
        case 'pandas':
            dataLoading = `import pandas as pd
import numpy as np

# Load your DataFrame
df = pd.read_pickle('your_data.pkl')  # or however you load it
# Select features for anomaly detection
data = df[['feature1', 'feature2', 'feature3']].values  # adjust column names`;
            description = 'Great for structured data analysis with feature selection.';
            break;
        case 'streaming':
            dataLoading = `import numpy as np
from anomaly_detection import StreamingService

# Your streaming data source (modify as needed)
def get_next_sample():
    # Replace with your actual data source
    return np.random.random(5)  # Example: 5-dimensional sample`;
            description = 'Perfect for real-time anomaly detection on streaming data.';
            break;
    }
    
    // Detection section
    let detection = '';
    if (dataSource === 'streaming') {
        detection = `# Initialize streaming service
streaming = StreamingService()

# Process samples in real-time
for i in range(100):  # or while True for continuous processing
    sample = get_next_sample()
    result = streaming.process_sample(sample)
    
    if result.is_anomaly:
        print(f"🚨 ANOMALY DETECTED: Sample {i}")
        print(f"   Score: {result.score:.3f}")
        # Add your alerting logic here
    else:
        print(f"✅ Sample {i}: Normal")`;
    } else if (algorithm === 'ensemble') {
        detection = `# Use ensemble for better accuracy
ensemble = EnsembleService()
result = ensemble.detect(
    data=${dataSource === 'csv' ? 'numeric_data' : 'data'},
    algorithms=['isolation_forest', 'lof', 'one_class_svm'],
    method='voting',
    contamination=${contamination}
)`;
    } else {
        detection = `# Detect anomalies
service = DetectionService()
result = service.detect(
    data=${dataSource === 'csv' ? 'numeric_data' : 'data'},
    algorithm='${algorithm}',
    contamination=${contamination}
)`;
    }
    
    // Results section
    let results = '';
    if (dataSource !== 'streaming') {
        results = `
# Analyze results
print(f"✅ Detection complete!")
print(f"📊 Total samples: {result.total_samples}")
print(f"🚨 Anomalies found: {result.anomaly_count}")
print(f"📈 Anomaly rate: {result.anomaly_count/result.total_samples*100:.1f}%")
print(f"⏱️ Processing time: {result.processing_time:.2f}s")

# Get anomaly details
anomaly_indices = np.where(result.predictions == -1)[0]
print(f"🔍 Anomaly indices: {anomaly_indices}")

# Top 5 most anomalous samples
top_anomalies = np.argsort(result.scores)[-5:]
print(f"🔥 Top anomaly scores: {result.scores[top_anomalies]}")`;
    }
    
    // Combine all parts
    code = `# ${description}
${imports}
${dataLoading}

${detection}${results}`;
    
    // Display the generated code
    document.getElementById('quickstart-output').innerHTML = `
        <h4>🎉 Your Personalized Quickstart Code:</h4>
        <p><em>${description}</em></p>
        <pre><code class="language-python">${code}</code></pre>
        <button class="demo-button" onclick="copyToClipboard()">📋 Copy Code</button>
        <p><small>💡 <strong>Next steps:</strong> Replace the data loading section with your actual data source, then run the code!</small></p>
    `;
}

function copyToClipboard() {
    const code = document.querySelector('#quickstart-output pre code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        alert('✅ Code copied to clipboard!');
    });
}
</script>

## 🛠️ Common Data Preparation Patterns

### CSV Files
```python
import pandas as pd
import numpy as np

# Option 1: All numeric columns
data = pd.read_csv('data.csv').select_dtypes(include=[np.number]).values

# Option 2: Specific columns
data = pd.read_csv('data.csv')[['col1', 'col2', 'col3']].values

# Option 3: Handle missing values
df = pd.read_csv('data.csv')
df = df.dropna()  # or df.fillna(0)
data = df.select_dtypes(include=[np.number]).values
```

### Time Series Data
```python
import pandas as pd

# Load time series
df = pd.read_csv('timeseries.csv', parse_dates=['timestamp'])

# Option 1: Rolling window features
df['rolling_mean'] = df['value'].rolling(10).mean()
df['rolling_std'] = df['value'].rolling(10).std()
data = df[['rolling_mean', 'rolling_std']].dropna().values

# Option 2: Time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
data = df[['value', 'hour', 'day_of_week']].values
```

### Text/Categorical Data
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Scale features
scaler = StandardScaler()
data = scaler.fit_transform(df[['numeric_col', 'category_encoded']])
```

## 🔧 Common Configuration Patterns

### Conservative Detection (Few False Positives)
```python
result = service.detect(
    data=data,
    algorithm='isolation_forest',
    contamination=0.01,  # Very low threshold
    n_estimators=200,    # More trees for stability
    random_state=42      # Reproducible results
)
```

### Aggressive Detection (Catch More Anomalies)
```python
result = service.detect(
    data=data,
    algorithm='lof',
    contamination=0.2,   # Higher threshold
    n_neighbors=10       # Smaller neighborhood
)
```

### Balanced Detection (Good Starting Point)
```python
result = service.detect(
    data=data,
    algorithm='isolation_forest',
    contamination=0.1,   # 10% anomalies expected
    n_estimators=100,    # Default trees
    max_samples='auto'   # Automatic sample size
)
```

## 🚨 Troubleshooting Common Issues

### "No anomalies detected"
```python
# Try these solutions:

# 1. Lower the contamination threshold
result = service.detect(data, contamination=0.05)  # Instead of 0.1

# 2. Try a different algorithm
result = service.detect(data, algorithm='lof')  # Instead of isolation_forest

# 3. Check your data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
result = service.detect(scaled_data, algorithm='isolation_forest')
```

### "Too many anomalies detected"
```python
# Try these solutions:

# 1. Increase the contamination threshold
result = service.detect(data, contamination=0.15)  # Instead of 0.1

# 2. Use ensemble for more stable results
from anomaly_detection import EnsembleService
ensemble = EnsembleService()
result = ensemble.detect(data, algorithms=['isolation_forest', 'lof'])

# 3. Filter by score threshold
high_confidence_anomalies = np.where(
    (result.predictions == -1) & (result.scores > 0.7)
)[0]
```

### "Import errors"
```bash
# Install the package
pip install anomaly-detection

# Or if using conda
conda install -c conda-forge anomaly-detection

# Verify installation
python -c "from anomaly_detection import DetectionService; print('✅ Import successful')"
```

## 📚 What's Next?

After your quickstart success, explore these advanced topics:

1. **[Algorithm Selection](algorithms.md)** - Choose the best algorithm for your data
2. **[Ensemble Methods](ensemble.md)** - Combine multiple algorithms for better accuracy  
3. **[Real-time Processing](streaming.md)** - Set up streaming anomaly detection
4. **[Production Deployment](deployment.md)** - Deploy your models at scale
5. **[Model Explainability](explainability.md)** - Understand why data points are anomalous

## 💡 Pro Tips

!!! tip "Getting Better Results"
    - **Start simple**: Use Isolation Forest with default parameters
    - **Scale your data**: Many algorithms work better with standardized features
    - **Validate results**: Always check if detected anomalies make business sense
    - **Use ensemble methods**: Combine algorithms for more robust detection
    - **Tune contamination**: Adjust based on your domain knowledge

!!! success "Performance Tips"
    - **For large datasets**: Use Isolation Forest (scales best)
    - **For real-time**: Use streaming mode with appropriate window sizes
    - **For accuracy**: Use ensemble methods with multiple algorithms
    - **For interpretability**: Use LOF or explainability features

Ready to detect your first anomaly? Copy one of the templates above and start experimenting! 🚀