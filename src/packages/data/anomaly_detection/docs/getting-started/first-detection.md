# Your First Anomaly Detection

This hands-on tutorial will guide you through performing your first anomaly detection with step-by-step explanations and real examples.

## Prerequisites

!!! info "Before You Start"
    - Python 3.8 or higher installed
    - Basic familiarity with Python and NumPy
    - Anomaly Detection package installed ([Installation Guide](../installation/))

## Tutorial Overview

In this tutorial, you'll learn to:

1. **Load and prepare data** for anomaly detection
2. **Configure different algorithms** for various use cases
3. **Interpret results** and understand confidence scores
4. **Visualize anomalies** in your data
5. **Tune parameters** for better performance

## Step 1: Prepare Your Environment

Let's start by importing the necessary libraries and creating sample data:

```python
import numpy as np
import matplotlib.pyplot as plt
from anomaly_detection import DetectionService, EnsembleService
from anomaly_detection.visualization import AnomalyPlotter

# Set random seed for reproducible results
np.random.seed(42)

print("✅ Environment ready!")
```

## Step 2: Create Sample Data

We'll create a realistic dataset with normal patterns and some anomalies:

```python
# Generate normal data (customers with typical behavior)
n_normal = 200
normal_data = np.random.multivariate_normal(
    mean=[50, 30],  # Average: $50 transaction, 30 transactions/month
    cov=[[100, 20], [20, 25]],  # Some correlation between amount and frequency
    size=n_normal
)

# Generate anomalies (suspicious behavior)
n_anomalies = 10

# Type 1: High-value, low-frequency (potential fraud)
anomaly_1 = np.random.multivariate_normal([200, 5], [[50, 0], [0, 5]], 5)

# Type 2: Low-value, very high-frequency (potential bot activity)  
anomaly_2 = np.random.multivariate_normal([10, 100], [[10, 0], [0, 50]], 5)

# Combine all data
data = np.vstack([normal_data, anomaly_1, anomaly_2])
true_labels = np.hstack([
    np.ones(n_normal),      # 1 = normal
    -np.ones(n_anomalies)   # -1 = anomaly
])

print(f"Dataset created:")
print(f"  • Normal samples: {n_normal}")
print(f"  • Anomalous samples: {n_anomalies}")
print(f"  • Total samples: {len(data)}")
print(f"  • Features: {data.shape[1]} (transaction_amount, frequency)")
```

## Step 3: Your First Detection

Now let's detect anomalies using the Isolation Forest algorithm:

```python
# Initialize the detection service
service = DetectionService()

# Perform anomaly detection
result = service.detect(
    data=data,
    algorithm="isolation_forest",
    contamination=0.1,  # Expect ~10% anomalies
    random_state=42
)

# Display basic results
print(f"\n🔍 Detection Results:")
print(f"  • Algorithm: {result.algorithm}")
print(f"  • Processing time: {result.processing_time:.3f}s")
print(f"  • Anomalies detected: {result.anomaly_count}")
print(f"  • Detection rate: {result.anomaly_count/len(data)*100:.1f}%")
print(f"  • Success: {result.success}")
```

**Expected Output:**
```
🔍 Detection Results:
  • Algorithm: isolation_forest
  • Processing time: 0.045s
  • Anomalies detected: 21
  • Detection rate: 10.0%
  • Success: True
```

## Step 4: Analyze the Results

Let's dive deeper into the detection results:

```python
# Get anomaly scores and predictions
anomaly_scores = result.scores
predictions = result.predictions

# Find detected anomalies
detected_anomalies = np.where(predictions == -1)[0]
normal_points = np.where(predictions == 1)[0]

print(f"\n📊 Detailed Analysis:")
print(f"  • Anomaly score range: {anomaly_scores.min():.3f} to {anomaly_scores.max():.3f}")
print(f"  • Normal points: {len(normal_points)}")
print(f"  • Detected anomalies: {len(detected_anomalies)}")

# Calculate performance metrics (if you have true labels)
if 'true_labels' in locals():
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Note: sklearn uses different convention (0=normal, 1=anomaly)
    y_true = (true_labels == -1).astype(int)
    y_pred = (predictions == -1).astype(int) 
    
    print(f"\n📈 Performance Metrics:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
```

## Step 5: Visualize the Results

Visualization helps understand what the algorithm detected:

```python
# Create visualization
plotter = AnomalyPlotter()

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original data with true labels
scatter1 = ax1.scatter(data[:, 0], data[:, 1], c=true_labels, 
                      cmap='coolwarm', alpha=0.7, s=50)
ax1.set_title('True Anomalies')
ax1.set_xlabel('Transaction Amount ($)')
ax1.set_ylabel('Monthly Frequency')
ax1.grid(True, alpha=0.3)

# Plot 2: Detection results
scatter2 = ax2.scatter(data[:, 0], data[:, 1], c=predictions, 
                      cmap='coolwarm', alpha=0.7, s=50)
ax2.set_title('Detected Anomalies')
ax2.set_xlabel('Transaction Amount ($)')
ax2.set_ylabel('Monthly Frequency')
ax2.grid(True, alpha=0.3)

# Add colorbars
plt.colorbar(scatter1, ax=ax1, label='True Label')
plt.colorbar(scatter2, ax=ax2, label='Prediction')

plt.tight_layout()
plt.show()

# Also create an interactive plot with anomaly scores
plotter.plot_anomaly_scores(
    data=data,
    scores=anomaly_scores,
    predictions=predictions,
    title="Anomaly Detection Results with Scores"
)
```

## Step 6: Try Different Algorithms

Different algorithms excel in different scenarios. Let's compare:

```python
# Test multiple algorithms
algorithms = [
    ("isolation_forest", {"contamination": 0.1}),
    ("lof", {"contamination": 0.1, "n_neighbors": 20}),
    ("one_class_svm", {"contamination": 0.1, "gamma": 'scale'}),
]

results = {}

print("🔄 Comparing Algorithms:")
print("-" * 50)

for algo_name, params in algorithms:
    result = service.detect(data, algorithm=algo_name, **params)
    results[algo_name] = result
    
    # Calculate basic metrics
    detected = np.sum(result.predictions == -1)
    
    print(f"{algo_name.upper()}:")
    print(f"  • Anomalies detected: {detected}")
    print(f"  • Processing time: {result.processing_time:.3f}s")
    print(f"  • Min/Max scores: {result.scores.min():.3f}/{result.scores.max():.3f}")
    print()
```

## Step 7: Ensemble Detection

Combine multiple algorithms for more robust detection:

```python
# Use ensemble methods for better accuracy
ensemble_service = EnsembleService()

# Combine multiple algorithms
ensemble_result = ensemble_service.detect(
    data=data,
    algorithms=["isolation_forest", "lof", "one_class_svm"],
    method="voting",  # Can also use "averaging" or "stacking"
    contamination=0.1
)

print(f"🎯 Ensemble Results:")
print(f"  • Algorithms combined: {len(ensemble_result.metadata['algorithms'])}")
print(f"  • Voting method: {ensemble_result.metadata['method']}")
print(f"  • Anomalies detected: {ensemble_result.anomaly_count}")
print(f"  • Processing time: {ensemble_result.processing_time:.3f}s")

# Compare ensemble vs individual algorithms
print(f"\n📊 Detection Summary:")
print(f"  • Isolation Forest: {np.sum(results['isolation_forest'].predictions == -1)} anomalies")
print(f"  • LOF: {np.sum(results['lof'].predictions == -1)} anomalies")
print(f"  • One-Class SVM: {np.sum(results['one_class_svm'].predictions == -1)} anomalies")
print(f"  • Ensemble (Voting): {ensemble_result.anomaly_count} anomalies")
```

## Step 8: Parameter Tuning

Fine-tune parameters for your specific use case:

```python
# Test different contamination rates
contamination_rates = [0.05, 0.1, 0.15, 0.2]
tuning_results = []

print("🎛️ Parameter Tuning - Contamination Rate:")
print("-" * 45)

for contamination in contamination_rates:
    result = service.detect(
        data=data,
        algorithm="isolation_forest",
        contamination=contamination,
        random_state=42
    )
    
    detected = np.sum(result.predictions == -1)
    tuning_results.append({
        'contamination': contamination,
        'detected': detected,
        'processing_time': result.processing_time
    })
    
    print(f"Contamination {contamination:.0%}: {detected} anomalies ({detected/len(data)*100:.1f}%)")

# Find optimal contamination rate
optimal = min(tuning_results, key=lambda x: abs(x['detected'] - n_anomalies))
print(f"\n✨ Optimal contamination rate: {optimal['contamination']:.0%}")
print(f"   (Detected {optimal['detected']} vs {n_anomalies} true anomalies)")
```

## Interactive Exercises

Try these hands-on exercises to deepen your understanding:

<div class="interactive-demo">
    <div class="demo-controls">
        <button class="demo-button" onclick="generateExerciseData()">Generate New Dataset</button>
        <select id="algorithm-select">
            <option value="isolation_forest">Isolation Forest</option>
            <option value="lof">Local Outlier Factor</option>
            <option value="one_class_svm">One-Class SVM</option>
        </select>
        <button class="demo-button" onclick="runExercise()">Detect Anomalies</button>
    </div>
    <div class="demo-output" id="exercise-output">
        Click "Generate New Dataset" to start the interactive exercise!
    </div>
</div>

<script>
let exerciseData = null;

function generateExerciseData() {
    // Simulate data generation
    const scenarios = [
        { name: "Financial Transactions", normal: 180, anomalies: 12 },
        { name: "Network Traffic", normal: 250, anomalies: 8 },
        { name: "Sensor Readings", normal: 150, anomalies: 15 },
        { name: "User Behavior", normal: 200, anomalies: 10 }
    ];
    
    const scenario = scenarios[Math.floor(Math.random() * scenarios.length)];
    exerciseData = scenario;
    
    document.getElementById('exercise-output').innerHTML = `
        <strong>New Dataset Generated: ${scenario.name}</strong>
        
        📊 Dataset Properties:
        • Normal samples: ${scenario.normal}
        • Anomalous samples: ${scenario.anomalies}
        • Total samples: ${scenario.normal + scenario.anomalies}
        • True anomaly rate: ${(scenario.anomalies / (scenario.normal + scenario.anomalies) * 100).toFixed(1)}%
        
        Select an algorithm and click "Detect Anomalies" to test your skills!
    `;
}

function runExercise() {
    if (!exerciseData) {
        alert("Please generate a dataset first!");
        return;
    }
    
    const algorithm = document.getElementById('algorithm-select').value;
    const output = document.getElementById('exercise-output');
    
    output.innerHTML += `<br><br>🔄 Running ${algorithm.replace('_', ' ').toUpperCase()}...`;
    
    setTimeout(() => {
        // Simulate detection results
        const totalSamples = exerciseData.normal + exerciseData.anomalies;
        const trueAnomalies = exerciseData.anomalies;
        
        // Simulate algorithm performance (different algorithms have different characteristics)
        let detectionRate, falsePositiveRate;
        switch(algorithm) {
            case 'isolation_forest':
                detectionRate = 0.8 + Math.random() * 0.15;
                falsePositiveRate = 0.02 + Math.random() * 0.03;
                break;
            case 'lof':
                detectionRate = 0.75 + Math.random() * 0.2;
                falsePositiveRate = 0.03 + Math.random() * 0.04;
                break;
            case 'one_class_svm':
                detectionRate = 0.7 + Math.random() * 0.25;
                falsePositiveRate = 0.01 + Math.random() * 0.02;
                break;
        }
        
        const detectedTrue = Math.floor(trueAnomalies * detectionRate);
        const falsePositives = Math.floor(exerciseData.normal * falsePositiveRate);
        const totalDetected = detectedTrue + falsePositives;
        
        const precision = detectedTrue / totalDetected;
        const recall = detectedTrue / trueAnomalies;
        const f1Score = 2 * (precision * recall) / (precision + recall);
        
        output.innerHTML += `
            
            <strong>✅ Detection Complete!</strong>
            
            🎯 Results:
            • Total detected: ${totalDetected}
            • True positives: ${detectedTrue}
            • False positives: ${falsePositives}
            • Missed anomalies: ${trueAnomalies - detectedTrue}
            
            📈 Performance Metrics:
            • Precision: ${(precision * 100).toFixed(1)}%
            • Recall: ${(recall * 100).toFixed(1)}%
            • F1-Score: ${(f1Score * 100).toFixed(1)}%
            • Processing time: ${(Math.random() * 0.5 + 0.1).toFixed(3)}s
            
            ${f1Score > 0.8 ? "🏆 Excellent performance!" : 
              f1Score > 0.6 ? "👍 Good performance!" : 
              "🤔 Consider tuning parameters or trying ensemble methods."}
        `;
    }, 1500);
}
</script>

## Common Patterns and Tips

!!! tip "Best Practices"
    1. **Start with Isolation Forest** - It's fast and works well on most datasets
    2. **Tune contamination rate** - Set it based on expected anomaly percentage
    3. **Use ensemble methods** - Combine multiple algorithms for robustness
    4. **Visualize results** - Always plot your data to understand patterns
    5. **Validate with domain knowledge** - Check if detected anomalies make sense

!!! warning "Common Pitfalls"
    - Setting contamination too high (detecting normal points as anomalies)
    - Not scaling features when using distance-based algorithms
    - Ignoring temporal patterns in time-series data
    - Not validating results with business logic

## What You've Learned

Congratulations! You've successfully:

- ✅ Created and prepared data for anomaly detection
- ✅ Used different algorithms and understood their characteristics
- ✅ Interpreted results and calculated performance metrics
- ✅ Visualized anomalies to gain insights
- ✅ Tuned parameters for optimal performance
- ✅ Combined algorithms using ensemble methods

## Next Steps

Ready to go deeper? Here are recommended next steps:

=== "Explore More Algorithms"
    Learn about different algorithms in the [Algorithm Guide](../algorithms/) and when to use each one.

=== "Real-World Examples"
    See practical applications in [Examples](examples/) covering various industries and use cases.

=== "Production Deployment"
    Learn how to deploy your models in the [Deployment Guide](../deployment/) for production use.

=== "Advanced Techniques"
    Explore [Ensemble Methods](../ensemble/) and [Performance Optimization](../performance/) for advanced usage.

## Need Help?

If you encounter issues or have questions:

1. Check the [Troubleshooting Guide](../troubleshooting/) for common problems
2. Review the [API Documentation](../api/) for detailed parameter descriptions
3. Look at more [Examples](examples/) for different scenarios
4. Ask questions in our community forums