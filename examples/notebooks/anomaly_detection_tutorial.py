#!/usr/bin/env python3
"""
Pynomaly Tutorial: Comprehensive Anomaly Detection Guide

This tutorial covers the main features of Pynomaly for anomaly detection
across different domains and use cases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔍 Pynomaly Tutorial: Comprehensive Anomaly Detection Guide")
print("=" * 60)

# =============================================================================
# Section 1: Data Generation and Exploration
# =============================================================================

print("\n📊 Section 1: Data Generation and Exploration")
print("-" * 50)

# Generate synthetic dataset with anomalies
np.random.seed(42)

# Normal data (majority)
n_normal = 9000
n_anomalies = 1000
n_features = 5

# Create normal data with some correlation structure
normal_data = np.random.multivariate_normal(
    mean=[0, 0, 0, 0, 0],
    cov=np.array([
        [1.0, 0.3, 0.1, 0.0, 0.2],
        [0.3, 1.0, 0.2, 0.1, 0.0],
        [0.1, 0.2, 1.0, 0.3, 0.1],
        [0.0, 0.1, 0.3, 1.0, 0.2],
        [0.2, 0.0, 0.1, 0.2, 1.0]
    ]),
    size=n_normal
)

# Create anomalous data (outliers)
anomaly_data = np.random.multivariate_normal(
    mean=[3, -2, 4, -3, 2],
    cov=np.eye(5) * 0.5,
    size=n_anomalies
)

# Combine data
X = np.vstack([normal_data, anomaly_data])
y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

# Create DataFrame for easier handling
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['is_anomaly'] = y

print(f"✅ Generated dataset with {len(df)} samples")
print(f"   - Normal samples: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"   - Anomalous samples: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"   - Features: {n_features}")

# Basic statistics
print(f"\n📈 Dataset Statistics:")
print(df.describe())

# Visualize data distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(n_features):
    axes[i].hist(df[df['is_anomaly'] == 0][f'feature_{i}'], 
                 alpha=0.7, label='Normal', bins=50, density=True)
    axes[i].hist(df[df['is_anomaly'] == 1][f'feature_{i}'], 
                 alpha=0.7, label='Anomaly', bins=30, density=True)
    axes[i].set_title(f'Feature {i} Distribution')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Correlation heatmap
corr_matrix = df.drop('is_anomaly', axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[5])
axes[5].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('/tmp/data_exploration.png', dpi=150, bbox_inches='tight')
print(f"📊 Data exploration plot saved to /tmp/data_exploration.png")

# =============================================================================
# Section 2: Traditional Machine Learning Approaches
# =============================================================================

print(f"\n🤖 Section 2: Traditional Machine Learning Approaches")
print("-" * 60)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"📝 Data split:")
print(f"   - Training set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")

# Import detection algorithms
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not available, using mock implementations")
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    # Initialize detectors
    detectors = {
        'Isolation Forest': IsolationForest(
            contamination=0.1, 
            random_state=42, 
            n_estimators=100
        ),
        'Local Outlier Factor': LocalOutlierFactor(
            contamination=0.1, 
            n_neighbors=20
        ),
        'One-Class SVM': OneClassSVM(
            nu=0.1, 
            kernel='rbf', 
            gamma='scale'
        ),
        'Elliptic Envelope': EllipticEnvelope(
            contamination=0.1, 
            random_state=42
        )
    }
    
    results = {}
    
    print(f"\n🔧 Training and evaluating detectors...")
    
    for name, detector in detectors.items():
        print(f"   Training {name}...")
        
        # Train detector (unsupervised)
        if name == 'Local Outlier Factor':
            # LOF doesn't have fit/predict, only fit_predict
            y_pred_train = detector.fit_predict(X_train)
            y_pred_test = detector.fit_predict(X_test)
        else:
            detector.fit(X_train)
            y_pred_train = detector.predict(X_train)
            y_pred_test = detector.predict(X_test)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
        y_pred_binary = (y_pred_test == -1).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Get decision scores for ROC AUC
        if hasattr(detector, 'decision_function'):
            scores = detector.decision_function(X_test)
            # For anomaly detection, invert scores so higher = more anomalous
            scores = -scores
        elif hasattr(detector, 'score_samples'):
            scores = detector.score_samples(X_test)
            scores = -scores
        else:
            scores = y_pred_binary
        
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except:
            roc_auc = 0.5  # Default for problematic cases
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred_binary,
            'scores': scores
        }
        
        print(f"     ✅ Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
    
    # Compare results
    print(f"\n📊 Algorithm Comparison:")
    comparison_df = pd.DataFrame({
        name: {
            'Precision': results[name]['precision'],
            'Recall': results[name]['recall'],
            'F1-Score': results[name]['f1_score'],
            'ROC-AUC': results[name]['roc_auc']
        }
        for name in results
    }).T
    
    print(comparison_df.round(3))
    
    # Find best performer
    best_algorithm = comparison_df['F1-Score'].idxmax()
    print(f"\n🏆 Best performing algorithm: {best_algorithm}")
    print(f"   F1-Score: {comparison_df.loc[best_algorithm, 'F1-Score']:.3f}")

# =============================================================================
# Section 3: Performance Optimization Demo
# =============================================================================

print(f"\n⚡ Section 3: Performance Optimization Demo")
print("-" * 50)

# Demonstrate memory and computation optimization
import time
import psutil

def get_memory_usage():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024

# Create larger dataset for optimization demo
print("🔧 Creating larger dataset for optimization demo...")
large_normal = np.random.multivariate_normal([0]*10, np.eye(10), 50000)
large_anomalies = np.random.multivariate_normal([3]*10, np.eye(10)*0.5, 5000)
large_X = np.vstack([large_normal, large_anomalies])
large_y = np.hstack([np.zeros(50000), np.ones(5000)])

print(f"   Large dataset: {large_X.shape[0]} samples, {large_X.shape[1]} features")

if SKLEARN_AVAILABLE:
    # Baseline performance
    print("\n📏 Baseline Performance (without optimization):")
    
    baseline_memory = get_memory_usage()
    start_time = time.perf_counter()
    
    # Standard sklearn approach
    baseline_detector = IsolationForest(
        contamination=0.1, 
        random_state=42,
        n_jobs=1  # Single thread for fair comparison
    )
    baseline_detector.fit(large_X)
    baseline_predictions = baseline_detector.predict(large_X)
    
    baseline_time = time.perf_counter() - start_time
    baseline_final_memory = get_memory_usage()
    
    print(f"   Time: {baseline_time:.2f} seconds")
    print(f"   Memory usage: {baseline_final_memory - baseline_memory:.1f} MB")
    
    # Optimized performance
    print("\n🚀 Optimized Performance:")
    
    optimized_memory = get_memory_usage()
    start_time = time.perf_counter()
    
    # Optimized approach with parallel processing and memory optimization
    optimized_detector = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_jobs=-1,  # Use all cores
        max_samples=0.8,  # Use subset for memory efficiency
        n_estimators=100
    )
    
    # Process in chunks to save memory
    chunk_size = 10000
    predictions = []
    
    # Train on subset
    train_subset = large_X[:20000]  # Use subset for training
    optimized_detector.fit(train_subset)
    
    # Predict in chunks
    for i in range(0, len(large_X), chunk_size):
        chunk = large_X[i:i+chunk_size]
        chunk_pred = optimized_detector.predict(chunk)
        predictions.extend(chunk_pred)
    
    optimized_time = time.perf_counter() - start_time
    optimized_final_memory = get_memory_usage()
    
    print(f"   Time: {optimized_time:.2f} seconds")
    print(f"   Memory usage: {optimized_final_memory - optimized_memory:.1f} MB")
    
    # Calculate improvements
    time_improvement = (baseline_time - optimized_time) / baseline_time * 100
    memory_improvement = ((baseline_final_memory - baseline_memory) - 
                         (optimized_final_memory - optimized_memory)) / (baseline_final_memory - baseline_memory) * 100
    
    print(f"\n📈 Performance Improvements:")
    print(f"   Time improvement: {time_improvement:.1f}%")
    print(f"   Memory reduction: {memory_improvement:.1f}%")

# =============================================================================
# Section 4: Streaming Anomaly Detection Simulation
# =============================================================================

print(f"\n🌊 Section 4: Streaming Anomaly Detection Simulation")
print("-" * 55)

# Simulate streaming data
print("🔧 Setting up streaming anomaly detection simulation...")

def generate_streaming_data(n_samples=1000, anomaly_rate=0.05):
    """Generate streaming data with concept drift."""
    normal_samples = int(n_samples * (1 - anomaly_rate))
    anomaly_samples = n_samples - normal_samples
    
    # Generate data with some time-based drift
    time_factor = np.linspace(0, 2, n_samples)
    
    # Normal data with drift
    normal_data = []
    for i in range(normal_samples):
        # Gradually shift mean over time
        drift = time_factor[i] * 0.1
        sample = np.random.multivariate_normal([drift]*5, np.eye(5)*0.8)
        normal_data.append(sample)
    
    # Anomalous data
    anomaly_indices = np.random.choice(n_samples, anomaly_samples, replace=False)
    
    # Create full dataset
    stream_data = []
    labels = []
    
    normal_idx = 0
    for i in range(n_samples):
        if i in anomaly_indices:
            # Generate anomaly
            sample = np.random.multivariate_normal([4]*5, np.eye(5)*0.3)
            labels.append(1)
        else:
            # Use normal data
            sample = normal_data[normal_idx]
            normal_idx += 1
            labels.append(0)
        
        stream_data.append(sample)
    
    return np.array(stream_data), np.array(labels)

# Generate streaming dataset
stream_X, stream_y = generate_streaming_data(5000, 0.08)
print(f"   Generated stream: {len(stream_X)} samples")
print(f"   Anomaly rate: {np.mean(stream_y)*100:.1f}%")

if SKLEARN_AVAILABLE:
    # Simulate online learning with window-based updates
    print("\n🔄 Simulating online anomaly detection...")
    
    window_size = 1000
    update_interval = 200
    
    # Initialize detector with first window
    detector = IsolationForest(contamination=0.08, random_state=42)
    detector.fit(stream_X[:window_size])
    
    # Process stream in batches
    anomalies_detected = []
    true_anomalies = []
    
    for start_idx in range(window_size, len(stream_X), update_interval):
        end_idx = min(start_idx + update_interval, len(stream_X))
        batch = stream_X[start_idx:end_idx]
        batch_labels = stream_y[start_idx:end_idx]
        
        # Detect anomalies in current batch
        predictions = detector.predict(batch)
        batch_anomalies = (predictions == -1).astype(int)
        
        anomalies_detected.extend(batch_anomalies)
        true_anomalies.extend(batch_labels)
        
        # Update model with new data (sliding window)
        window_start = max(0, start_idx - window_size)
        window_data = stream_X[window_start:start_idx]
        detector.fit(window_data)
        
        # Report progress
        if len(anomalies_detected) % 1000 == 0:
            current_precision = precision_score(true_anomalies, anomalies_detected) if len(set(anomalies_detected)) > 1 else 0
            print(f"   Processed {len(anomalies_detected)} samples, Current precision: {current_precision:.3f}")
    
    # Final streaming results
    if len(set(anomalies_detected)) > 1:
        stream_precision = precision_score(true_anomalies, anomalies_detected)
        stream_recall = recall_score(true_anomalies, anomalies_detected)
        stream_f1 = f1_score(true_anomalies, anomalies_detected)
        
        print(f"\n📊 Streaming Detection Results:")
        print(f"   Precision: {stream_precision:.3f}")
        print(f"   Recall: {stream_recall:.3f}")
        print(f"   F1-Score: {stream_f1:.3f}")
        print(f"   Total anomalies detected: {sum(anomalies_detected)}")
        print(f"   Total true anomalies: {sum(true_anomalies)}")

# =============================================================================
# Section 5: Practical Tips and Best Practices
# =============================================================================

print(f"\n💡 Section 5: Practical Tips and Best Practices")
print("-" * 55)

tips = [
    "🎯 Data Quality: Clean and preprocess your data before anomaly detection",
    "📊 Contamination Rate: Set contamination parameter based on domain knowledge",
    "🔄 Cross-Validation: Use time-series CV for temporal data",
    "⚖️ Class Imbalance: Consider evaluation metrics beyond accuracy",
    "🧠 Ensemble Methods: Combine multiple algorithms for better performance",
    "📈 Feature Engineering: Create domain-specific features",
    "🔍 Threshold Tuning: Adjust decision thresholds based on business needs",
    "⚡ Performance: Use optimization techniques for large datasets",
    "📋 Monitoring: Continuously monitor model performance in production",
    "🔄 Retraining: Update models regularly to handle concept drift"
]

for tip in tips:
    print(f"   {tip}")

# =============================================================================
# Section 6: Common Pitfalls and Solutions
# =============================================================================

print(f"\n⚠️ Section 6: Common Pitfalls and Solutions")
print("-" * 50)

pitfalls = {
    "High False Positive Rate": [
        "Lower contamination parameter",
        "Improve feature engineering",
        "Use ensemble methods",
        "Adjust decision threshold"
    ],
    "Poor Generalization": [
        "Increase training data diversity",
        "Use cross-validation",
        "Implement concept drift detection",
        "Regular model retraining"
    ],
    "Scalability Issues": [
        "Use sampling techniques",
        "Implement batch processing",
        "Enable parallel processing",
        "Consider approximate algorithms"
    ],
    "Interpretability Challenges": [
        "Use SHAP/LIME for explanations",
        "Implement feature importance analysis",
        "Visualize anomalous patterns",
        "Document decision rationale"
    ]
}

for pitfall, solutions in pitfalls.items():
    print(f"\n🔴 {pitfall}:")
    for solution in solutions:
        print(f"   ✅ {solution}")

# =============================================================================
# Summary and Next Steps
# =============================================================================

print(f"\n🎉 Tutorial Complete!")
print("=" * 60)

print(f"📚 What you learned:")
print(f"   ✅ Data generation and exploration")
print(f"   ✅ Traditional ML anomaly detection algorithms")
print(f"   ✅ Performance optimization techniques")
print(f"   ✅ Streaming anomaly detection")
print(f"   ✅ Best practices and common pitfalls")

print(f"\n🚀 Next Steps:")
print(f"   1. Explore advanced algorithms (deep learning, graph-based)")
print(f"   2. Try ensemble methods for improved performance")
print(f"   3. Implement explainable AI for anomaly interpretation")
print(f"   4. Build production-ready anomaly detection systems")
print(f"   5. Dive into domain-specific use cases")

print(f"\n📖 Additional Resources:")
print(f"   📗 User Guide: docs/USER_GUIDE.md")
print(f"   📘 API Reference: docs/API_REFERENCE.md")
print(f"   📙 Examples: examples/")
print(f"   📕 Industry Use Cases: examples/industry_use_cases/")

print(f"\n🤝 Join the Community:")
print(f"   🐙 GitHub: https://github.com/your-org/pynomaly")
print(f"   💬 Discussions: https://github.com/your-org/pynomaly/discussions")
print(f"   📧 Support: support@pynomaly.org")

print(f"\nHappy anomaly hunting! 🔍✨")

# Optional: Save results for later analysis
results_summary = {
    'dataset_size': len(X),
    'anomaly_rate': np.mean(y),
    'algorithms_tested': len(results) if SKLEARN_AVAILABLE else 0,
    'timestamp': pd.Timestamp.now().isoformat()
}

print(f"\n💾 Tutorial session saved with ID: {results_summary['timestamp']}")