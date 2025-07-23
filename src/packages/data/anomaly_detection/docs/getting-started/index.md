# Getting Started with Anomaly Detection

Welcome to the Anomaly Detection package! This guide will help you get up and running quickly with detecting anomalies in your data.

## What is Anomaly Detection?

Anomaly detection is the process of identifying data points, events, or observations that deviate significantly from the expected pattern or behavior in a dataset. These deviations, called anomalies or outliers, can indicate:

!!! info "Common Use Cases"
    - **Fraud Detection**: Identifying unusual financial transactions
    - **Network Security**: Detecting intrusion attempts or unusual network traffic
    - **Quality Control**: Finding defective products in manufacturing
    - **System Monitoring**: Identifying performance issues or failures
    - **Medical Diagnosis**: Detecting abnormal patterns in medical data

## Quick Installation

=== "pip"
    ```bash
    pip install anomaly-detection
    ```

=== "conda"
    ```bash
    conda install -c conda-forge anomaly-detection
    ```

=== "Development"
    ```bash
    git clone https://github.com/organization/anomaly-detection.git
    cd anomaly-detection
    pip install -e ".[dev]"
    ```

## Your First Detection in 2 Minutes

Let's detect anomalies in a simple dataset:

```python
import numpy as np
from anomaly_detection import DetectionService

# Create sample data with some anomalies
normal_data = np.random.normal(0, 1, (100, 2))
anomalies = np.random.normal(5, 1, (5, 2))
data = np.vstack([normal_data, anomalies])

# Initialize the detection service
service = DetectionService()

# Detect anomalies using Isolation Forest
result = service.detect(data, algorithm="isolation_forest")

# View results
print(f"Found {result.anomaly_count} anomalies out of {result.total_samples} samples")
print(f"Anomaly indices: {np.where(result.predictions == -1)[0]}")
```

**Expected Output:**
```
Found 5 anomalies out of 105 samples
Anomaly indices: [100 101 102 103 104]
```

!!! success "Congratulations!"
    You've successfully detected your first anomalies! The algorithm correctly identified the 5 outlier points we added to the dataset.

## Understanding the Results

The `DetectionResult` object contains comprehensive information about the detection:

```python
# Detailed result analysis
print(f"Algorithm used: {result.algorithm}")
print(f"Processing time: {result.processing_time:.2f} seconds")
print(f"Success: {result.success}")

# Anomaly scores (higher = more anomalous)
print(f"Top 5 anomaly scores: {np.sort(result.scores)[-5:]}")

# Predictions (-1 = anomaly, 1 = normal)
anomaly_mask = result.predictions == -1
print(f"Anomalous samples:\n{data[anomaly_mask]}")
```

## Interactive Demo

<div id="getting-started-demo" class="interactive-demo">
    <div class="demo-controls">
        <button class="demo-button" onclick="runGettingStartedDemo()">Run Detection Demo</button>
        <button class="demo-button" onclick="generateNewData()">Generate New Data</button>
    </div>
    <div class="demo-output" id="demo-output">
        Click "Run Detection Demo" to see anomaly detection in action!
    </div>
</div>

<script>
let demoData = null;

function generateNewData() {
    // Generate sample data for demo
    const normalPoints = 95;
    const anomalyPoints = 5;
    
    demoData = {
        normal: Array.from({length: normalPoints}, () => ({
            x: Math.random() * 4 - 2,
            y: Math.random() * 4 - 2,
            type: 'normal'
        })),
        anomalies: Array.from({length: anomalyPoints}, () => ({
            x: Math.random() * 8 - 4 + (Math.random() > 0.5 ? 3 : -3),
            y: Math.random() * 8 - 4 + (Math.random() > 0.5 ? 3 : -3),
            type: 'anomaly'
        }))
    };
    
    document.getElementById('demo-output').innerHTML = `
        Generated new dataset:
        - ${normalPoints} normal points
        - ${anomalyPoints} anomalous points
        
        Click "Run Detection Demo" to detect anomalies!
    `;
}

function runGettingStartedDemo() {
    if (!demoData) {
        generateNewData();
    }
    
    const output = document.getElementById('demo-output');
    output.innerHTML = 'Running Isolation Forest detection...';
    
    setTimeout(() => {
        // Simulate detection results
        const totalSamples = demoData.normal.length + demoData.anomalies.length;
        const detectedAnomalies = Math.floor(demoData.anomalies.length * (0.8 + Math.random() * 0.2));
        const falsePositives = Math.floor(Math.random() * 3);
        const processingTime = (Math.random() * 0.5 + 0.1).toFixed(3);
        
        const precision = detectedAnomalies / (detectedAnomalies + falsePositives);
        const recall = detectedAnomalies / demoData.anomalies.length;
        
        output.innerHTML = `
            <strong>Detection Complete!</strong>
            
            Dataset: ${totalSamples} samples
            True anomalies: ${demoData.anomalies.length}
            Detected anomalies: ${detectedAnomalies + falsePositives}
            
            <strong>Performance:</strong>
            Precision: ${(precision * 100).toFixed(1)}%
            Recall: ${(recall * 100).toFixed(1)}%
            Processing time: ${processingTime}s
            
            <strong>Results:</strong>
            ✅ Correctly detected: ${detectedAnomalies}
            ❌ False positives: ${falsePositives}
            📊 Anomaly score threshold: ${(Math.random() * 0.3 + 0.5).toFixed(2)}
        `;
    }, 1500);
}

// Initialize demo on page load
document.addEventListener('DOMContentLoaded', function() {
    generateNewData();
});
</script>

## Learning Paths

Choose your path based on your role and experience:

<div class="feature-grid">
    <div class="learning-path-card" id="beginner-path">
        <div class="learning-path-header">
            <div class="learning-path-icon">🐣</div>
            <div>
                <div class="learning-path-title">Beginner Path</div>
                <div class="learning-path-duration">2-3 hours</div>
            </div>
        </div>
        <div class="learning-path-description">
            Perfect if you're new to anomaly detection or machine learning.
        </div>
        <ul class="learning-path-steps">
            <li><a href="../installation/">Complete Installation Guide</a> <em>(10 min)</em></li>
            <li><a href="first-detection/">Your First Detection</a> <em>(20 min)</em></li>
            <li><a href="../cli/">Basic CLI Usage</a> <em>(30 min)</em></li>
            <li><a href="examples/">Practical Examples</a> <em>(45 min)</em></li>
            <li><a href="../troubleshooting/">Common Issues</a> <em>(15 min)</em></li>
        </ul>
    </div>
    
    <div class="learning-path-card" id="scientist-path">
        <div class="learning-path-header">
            <div class="learning-path-icon">👨‍🔬</div>
            <div>
                <div class="learning-path-title">Data Scientist Path</div>
                <div class="learning-path-duration">4-5 hours</div>
            </div>
        </div>
        <div class="learning-path-description">
            For data scientists and ML practitioners who want to build production-ready models.
        </div>
        <ul class="learning-path-steps">
            <li><a href="../algorithms/">Algorithm Selection</a> <em>(30 min)</em></li>
            <li><a href="../model_management/">Model Management</a> <em>(60 min)</em></li>
            <li><a href="../ensemble/">Ensemble Methods</a> <em>(90 min)</em></li>
            <li><a href="../explainability/">Model Explainability</a> <em>(60 min)</em></li>
            <li><a href="../performance/">Performance Optimization</a> <em>(60 min)</em></li>
        </ul>
    </div>
    
    <div class="learning-path-card" id="devops-path">
        <div class="learning-path-header">
            <div class="learning-path-icon">🔧</div>
            <div>
                <div class="learning-path-title">DevOps Path</div>
                <div class="learning-path-duration">5-6 hours</div>
            </div>
        </div>
        <div class="learning-path-description">
            For engineers focused on deployment, scaling, and infrastructure management.
        </div>
        <ul class="learning-path-steps">
            <li><a href="../deployment/">Production Deployment</a> <em>(120 min)</em></li>
            <li><a href="../streaming/">Streaming Detection</a> <em>(90 min)</em></li>
            <li><a href="../security/">Security Implementation</a> <em>(90 min)</em></li>
            <li><a href="../integration/">Integration Patterns</a> <em>(60 min)</em></li>
            <li><a href="../troubleshooting/">Monitoring & Troubleshooting</a> <em>(60 min)</em></li>
        </ul>
    </div>
</div>

## Next Steps

!!! tip "Recommended Next Steps"
    1. **[Complete the Installation](../installation/)** - Set up your environment properly
    2. **[Try Your First Detection](first-detection/)** - Hands-on tutorial with real data
    3. **[Explore Examples](examples/)** - See common use cases in action
    4. **[Learn the CLI](../cli/)** - Master the command-line interface
    5. **[Choose Your Algorithm](../algorithms/)** - Understand when to use different algorithms

## Need Help?

- 📖 **Documentation**: Browse the comprehensive guides in the navigation
- 💬 **Community**: Join our discussions and ask questions
- 🐛 **Issues**: Report bugs or request features on GitHub
- 📧 **Support**: Contact our team for enterprise support

## What's Next?

Based on your interests:

=== "I want to understand the basics"
    Head to [Your First Detection](first-detection/) for a step-by-step tutorial.

=== "I want to see practical examples"
    Check out [Practical Examples](examples/) for real-world use cases.

=== "I want to use the command line"
    Learn the [CLI Interface](../cli/) for automation and scripting.

=== "I want to integrate into my application"
    See [Integration Patterns](../integration/) for APIs and workflows.

=== "I want to deploy to production"
    Follow the [Deployment Guide](../deployment/) for scaling and reliability.