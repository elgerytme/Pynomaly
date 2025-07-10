# Your First Anomaly Detection

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 📚 [User Guides](../README.md) > 🚀 [Onboarding](README.md) > 🎯 First Detection

---

Ready to detect your first anomalies? This tutorial will guide you through the complete process, from loading data to interpreting results. By the end, you'll have successfully identified unusual patterns in a real dataset!

## 🎯 What You'll Learn

- How to load and explore data for anomaly detection
- Choosing the right algorithm for your use case  
- Running anomaly detection with different parameters
- Interpreting and visualizing results
- Best practices for getting started

## 📊 Sample Dataset: E-commerce Transactions

We'll use a realistic e-commerce transaction dataset that contains:

- **Normal transactions**: Regular customer purchases
- **Anomalous transactions**: Unusual patterns that might indicate fraud, errors, or interesting behaviors

### Dataset Features

- `transaction_id`: Unique identifier
- `amount`: Transaction amount in USD
- `customer_id`: Customer identifier
- `merchant_category`: Type of merchant
- `transaction_time`: When the transaction occurred
- `location`: Geographic location
- `payment_method`: How the payment was made

## 🚀 Method 1: Interactive Web Interface

### Step 1: Start the Web Interface

```bash
# Start Pynomaly server
pynomaly server start

# Or with Hatch
hatch env run dev:serve

# Access at http://localhost:8000
```

### Step 2: Upload Your Data

1. Click **"Upload Dataset"** or navigate to `/datasets`
2. Choose your CSV file or use our sample data
3. Preview the data and confirm column types
4. Name your dataset (e.g., "E-commerce Transactions")

### Step 3: Create a Detector

1. Go to **"Detectors"** section
2. Click **"Create New Detector"**
3. Choose **"Isolation Forest"** (great for beginners)
4. Set contamination to `0.05` (expect 5% anomalies)
5. Name your detector (e.g., "Fraud Detector")

### Step 4: Run Detection

1. Go to **"Detection"** section
2. Select your dataset and detector
3. Click **"Run Detection"**
4. Watch real-time progress and results

### Step 5: Explore Results

1. View anomaly summary and statistics
2. Explore interactive visualizations
3. Examine individual anomalous transactions
4. Download results for further analysis

## 💻 Method 2: Python Code (Recommended)

### Complete Example

```python
"""
Your First Anomaly Detection with Pynomaly
A comprehensive example for beginners
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Pynomaly
from pynomaly import detect_anomalies
from pynomaly.algorithms import IsolationForest, LocalOutlierFactor
from pynomaly.preprocessing import StandardScaler, prepare_data
from pynomaly.visualization import plot_anomalies, plot_distribution

print("🔍 Welcome to Your First Anomaly Detection!")
print("=" * 50)

# Step 1: Create Sample E-commerce Data
print("\n📊 Step 1: Creating sample e-commerce transaction data...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate normal transactions
n_normal = 1000
normal_transactions = {
    'transaction_id': [f'TXN_{i:05d}' for i in range(n_normal)],
    'amount': np.random.lognormal(mean=3, sigma=1, size=n_normal),  # Log-normal distribution for amounts
    'customer_id': np.random.randint(1, 200, n_normal),
    'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_normal),
    'hour_of_day': np.random.choice(range(6, 23), n_normal),  # Normal business hours
    'day_of_week': np.random.choice(range(1, 8), n_normal),
    'payment_method': np.random.choice(['credit', 'debit', 'cash'], n_normal, p=[0.6, 0.3, 0.1])
}

# Generate anomalous transactions (fraud/errors)
n_anomalies = 50
anomalous_transactions = {
    'transaction_id': [f'TXN_{i:05d}' for i in range(n_normal, n_normal + n_anomalies)],
    'amount': np.concatenate([
        np.random.lognormal(mean=7, sigma=0.5, size=20),  # Very high amounts
        np.random.uniform(0.01, 1, size=15),  # Very low amounts
        np.random.lognormal(mean=3, sigma=1, size=15)  # Normal amounts but other features anomalous
    ]),
    'customer_id': np.concatenate([
        np.random.randint(1, 200, 35),  # Normal customers
        np.random.randint(9000, 9999, 15)  # New/suspicious customer IDs
    ]),
    'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_anomalies),
    'hour_of_day': np.concatenate([
        np.random.choice(range(0, 6), 20),  # Late night transactions
        np.random.choice(range(6, 23), 30)  # Normal hours
    ]),
    'day_of_week': np.random.choice(range(1, 8), n_anomalies),
    'payment_method': np.random.choice(['credit', 'debit', 'cash'], n_anomalies, p=[0.8, 0.15, 0.05])
}

# Combine normal and anomalous data
all_data = {}
for key in normal_transactions.keys():
    all_data[key] = np.concatenate([normal_transactions[key], anomalous_transactions[key]])

# Create DataFrame
df = pd.DataFrame(all_data)

# Add derived features
df['is_weekend'] = (df['day_of_week'] >= 6).astype(int)
df['is_night'] = (df['hour_of_day'] < 6).astype(int)
df['amount_log'] = np.log1p(df['amount'])

print(f"✅ Created dataset with {len(df)} transactions")
print(f"   - Normal transactions: {n_normal}")
print(f"   - Anomalous transactions: {n_anomalies}")
print(f"   - Features: {list(df.columns)}")

# Display sample data
print("\n📋 Sample transactions:")
print(df.head())

# Step 2: Explore the Data
print("\n📈 Step 2: Exploring the data...")

print("\nBasic statistics:")
print(df.describe())

print(f"\nTransaction amounts:")
print(f"   - Mean: ${df['amount'].mean():.2f}")
print(f"   - Median: ${df['amount'].median():.2f}")
print(f"   - Min: ${df['amount'].min():.2f}")
print(f"   - Max: ${df['amount'].max():.2f}")

print(f"\nMerchant categories:")
print(df['merchant_category'].value_counts())

# Step 3: Prepare Data for Anomaly Detection
print("\n🛠️ Step 3: Preparing data for anomaly detection...")

# Select numerical features for detection
feature_columns = ['amount', 'customer_id', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_night']

# Prepare features
X = df[feature_columns].copy()

# Handle categorical variables (encode payment_method)
payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')
X = pd.concat([X, payment_dummies], axis=1)

print(f"✅ Prepared {X.shape[1]} features for detection:")
print(f"   Features: {list(X.columns)}")

# Step 4: Run Anomaly Detection
print("\n🎯 Step 4: Running anomaly detection...")

# Method 1: Simple detection with default settings
print("\n4a) Simple Detection:")
anomalies_simple = detect_anomalies(X, contamination=0.05)
print(f"   Detected {anomalies_simple.sum()} anomalies ({anomalies_simple.mean():.1%})")

# Method 2: Using specific algorithm
print("\n4b) Isolation Forest Detection:")
from sklearn.ensemble import IsolationForest

# Create and train detector
iso_forest = IsolationForest(
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,
    n_estimators=100
)

# Fit and predict
anomaly_scores = iso_forest.fit_predict(X)
anomalies_iso = (anomaly_scores == -1)

print(f"   Detected {anomalies_iso.sum()} anomalies ({anomalies_iso.mean():.1%})")

# Method 3: Local Outlier Factor
print("\n4c) Local Outlier Factor Detection:")
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

anomaly_scores_lof = lof.fit_predict(X)
anomalies_lof = (anomaly_scores_lof == -1)

print(f"   Detected {anomalies_lof.sum()} anomalies ({anomalies_lof.mean():.1%})")

# Step 5: Analyze Results
print("\n📊 Step 5: Analyzing results...")

# Add anomaly flags to dataframe
df['anomaly_simple'] = anomalies_simple
df['anomaly_iso'] = anomalies_iso
df['anomaly_lof'] = anomalies_lof

# True anomalies (we know which ones we created)
df['true_anomaly'] = df.index >= n_normal

# Calculate accuracy metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\n🎯 Detection Performance:")
print("\nIsolation Forest Results:")
print(classification_report(df['true_anomaly'], df['anomaly_iso'], 
                          target_names=['Normal', 'Anomaly']))

print("\nLocal Outlier Factor Results:")
print(classification_report(df['true_anomaly'], df['anomaly_lof'], 
                          target_names=['Normal', 'Anomaly']))

# Step 6: Examine Anomalous Transactions
print("\n🔍 Step 6: Examining anomalous transactions...")

# Get transactions detected as anomalies by Isolation Forest
detected_anomalies = df[df['anomaly_iso']]

print(f"\nTop 10 detected anomalies:")
anomaly_display = detected_anomalies[['transaction_id', 'amount', 'customer_id', 
                                    'merchant_category', 'hour_of_day', 'payment_method']].head(10)
print(anomaly_display.to_string(index=False))

# Analyze anomaly characteristics
print(f"\n📋 Anomaly characteristics:")
print(f"   Average amount: ${detected_anomalies['amount'].mean():.2f} vs ${df[~df['anomaly_iso']]['amount'].mean():.2f} (normal)")
print(f"   Night transactions: {detected_anomalies['is_night'].mean():.1%} vs {df[~df['anomaly_iso']]['is_night'].mean():.1%} (normal)")
print(f"   Weekend transactions: {detected_anomalies['is_weekend'].mean():.1%} vs {df[~df['anomaly_iso']]['is_weekend'].mean():.1%} (normal)")

# Step 7: Visualize Results
print("\n📈 Step 7: Creating visualizations...")

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Amount distribution
plt.subplot(2, 3, 1)
normal_amounts = df[~df['anomaly_iso']]['amount']
anomaly_amounts = df[df['anomaly_iso']]['amount']

plt.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='blue')
plt.hist(anomaly_amounts, bins=20, alpha=0.7, label='Anomalous', color='red')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.title('Transaction Amount Distribution')
plt.legend()
plt.yscale('log')

# Plot 2: Time of day
plt.subplot(2, 3, 2)
normal_hours = df[~df['anomaly_iso']]['hour_of_day']
anomaly_hours = df[df['anomaly_iso']]['hour_of_day']

plt.hist(normal_hours, bins=24, alpha=0.7, label='Normal', color='blue')
plt.hist(anomaly_hours, bins=24, alpha=0.7, label='Anomalous', color='red')
plt.xlabel('Hour of Day')
plt.ylabel('Frequency')
plt.title('Transaction Time Distribution')
plt.legend()

# Plot 3: Scatter plot - Amount vs Customer ID
plt.subplot(2, 3, 3)
normal_data = df[~df['anomaly_iso']]
anomaly_data = df[df['anomaly_iso']]

plt.scatter(normal_data['customer_id'], normal_data['amount'], 
           alpha=0.6, s=20, color='blue', label='Normal')
plt.scatter(anomaly_data['customer_id'], anomaly_data['amount'], 
           alpha=0.8, s=40, color='red', label='Anomalous')
plt.xlabel('Customer ID')
plt.ylabel('Transaction Amount ($)')
plt.title('Customer ID vs Amount')
plt.legend()
plt.yscale('log')

# Plot 4: Merchant category distribution
plt.subplot(2, 3, 4)
normal_merchants = df[~df['anomaly_iso']]['merchant_category'].value_counts()
anomaly_merchants = df[df['anomaly_iso']]['merchant_category'].value_counts()

x = range(len(normal_merchants))
width = 0.35

plt.bar([i - width/2 for i in x], normal_merchants.values, width, 
        label='Normal', alpha=0.7, color='blue')
plt.bar([i + width/2 for i in x], anomaly_merchants.values, width, 
        label='Anomalous', alpha=0.7, color='red')

plt.xlabel('Merchant Category')
plt.ylabel('Count')
plt.title('Merchant Category Distribution')
plt.xticks(x, normal_merchants.index, rotation=45)
plt.legend()

# Plot 5: Confusion Matrix
plt.subplot(2, 3, 5)
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(df['true_anomaly'], df['anomaly_iso'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Normal', 'Anomaly'], 
           yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Plot 6: ROC Curve (if we have scores)
plt.subplot(2, 3, 6)
try:
    # Get anomaly scores
    scores = iso_forest.decision_function(X)
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(df['true_anomaly'], -scores)  # Negative because lower scores = more anomalous
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
except Exception as e:
    plt.text(0.5, 0.5, f'ROC plot unavailable:\n{str(e)}', 
             horizontalalignment='center', verticalalignment='center')
    plt.title('ROC Curve')

plt.tight_layout()
plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
print("📊 Visualizations saved as 'anomaly_detection_results.png'")

# Step 8: Save Results
print("\n💾 Step 8: Saving results...")

# Save detected anomalies
anomalies_to_save = df[df['anomaly_iso']].copy()
anomalies_to_save.to_csv('detected_anomalies.csv', index=False)
print(f"✅ Saved {len(anomalies_to_save)} detected anomalies to 'detected_anomalies.csv'")

# Save full results
df.to_csv('full_results_with_anomalies.csv', index=False)
print(f"✅ Saved full dataset with anomaly flags to 'full_results_with_anomalies.csv'")

# Create summary report
summary = {
    'detection_timestamp': datetime.now().isoformat(),
    'total_transactions': len(df),
    'detected_anomalies': int(anomalies_iso.sum()),
    'detection_rate': float(anomalies_iso.mean()),
    'algorithm': 'Isolation Forest',
    'parameters': {
        'contamination': 0.05,
        'n_estimators': 100,
        'random_state': 42
    },
    'performance': {
        'precision': float(classification_report(df['true_anomaly'], df['anomaly_iso'], output_dict=True)['True']['precision']),
        'recall': float(classification_report(df['true_anomaly'], df['anomaly_iso'], output_dict=True)['True']['recall']),
        'f1_score': float(classification_report(df['true_anomaly'], df['anomaly_iso'], output_dict=True)['True']['f1-score'])
    }
}

with open('detection_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✅ Saved detection summary to 'detection_summary.json'")

print("\n🎉 First Anomaly Detection Complete!")
print("=" * 50)
print(f"✅ Successfully detected {anomalies_iso.sum()} anomalies out of {len(df)} transactions")
print(f"✅ Detection rate: {anomalies_iso.mean():.1%}")
print(f"✅ Files created:")
print(f"   - detected_anomalies.csv")
print(f"   - full_results_with_anomalies.csv") 
print(f"   - detection_summary.json")
print(f"   - anomaly_detection_results.png")

print(f"\n🎯 Next steps:")
print(f"   - Review the detected anomalies in the CSV file")
print(f"   - Examine the visualizations")
print(f"   - Try different algorithms and parameters")
print(f"   - Apply to your own datasets!")
```

## 🚀 Method 3: Command Line Interface

### Step-by-Step CLI Commands

```bash
# Step 1: Create sample data
pynomaly dataset create-sample \
  --name "ecommerce_transactions" \
  --type "ecommerce" \
  --size 1000 \
  --anomaly-rate 0.05

# Step 2: Explore the data
pynomaly dataset info ecommerce_transactions
pynomaly dataset preview ecommerce_transactions --rows 10

# Step 3: Create detector
pynomaly detector create \
  --name "fraud_detector" \
  --algorithm IsolationForest \
  --contamination 0.05 \
  --description "E-commerce fraud detection"

# Step 4: Train and detect
pynomaly detect run \
  --detector fraud_detector \
  --dataset ecommerce_transactions \
  --output results.csv

# Step 5: View results
pynomaly detect results --latest
pynomaly detect visualize --input results.csv --output charts.png
```

## 📊 Understanding Your Results

### Key Metrics to Look For

1. **Detection Rate**: Percentage of data flagged as anomalous
   - Should match your expected contamination rate
   - Too high = algorithm too sensitive
   - Too low = algorithm not sensitive enough

2. **Anomaly Score Distribution**: How confident the algorithm is
   - Higher absolute scores = more confident anomalies
   - Use for prioritizing investigation

3. **Feature Importance**: Which features drive anomaly detection
   - Helps understand why something is anomalous
   - Guides feature engineering

### Common Patterns in Results

#### Financial Anomalies

- **Very high amounts**: Potential fraud or data errors
- **Very low amounts**: Testing transactions or errors
- **Unusual timing**: Off-hours or weekend activity
- **New customers**: First-time users with unusual patterns

#### Behavioral Anomalies

- **Rapid sequential transactions**: Possible automation
- **Geographic inconsistencies**: Impossible travel times
- **Payment method changes**: Switching between methods rapidly

## 🔧 Tuning Your Detection

### Adjusting Sensitivity

```python
# More sensitive (catches more anomalies)
detector = IsolationForest(contamination=0.1)  # 10% anomalies

# Less sensitive (catches fewer anomalies)  
detector = IsolationForest(contamination=0.01) # 1% anomalies

# Let the algorithm decide
detector = IsolationForest(contamination='auto')
```

### Different Algorithms for Different Use Cases

```python
# Isolation Forest: Good for high-dimensional data
from sklearn.ensemble import IsolationForest
detector = IsolationForest(contamination=0.05)

# Local Outlier Factor: Good for local anomalies  
from sklearn.neighbors import LocalOutlierFactor
detector = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

# One-Class SVM: Good for complex boundaries
from sklearn.svm import OneClassSVM
detector = OneClassSVM(nu=0.05)  # nu ≈ contamination

# Ensemble approach: Combine multiple algorithms
from pynomaly.ensemble import VotingAnomalyDetector
detector = VotingAnomalyDetector([
    IsolationForest(contamination=0.05),
    LocalOutlierFactor(contamination=0.05),
    OneClassSVM(nu=0.05)
])
```

## 🚨 Common Pitfalls and Solutions

### Issue 1: Too Many False Positives

**Symptoms**: Everything looks anomalous
**Solutions**:

- Reduce contamination parameter
- Remove noisy features
- Standardize/normalize features
- Check for data quality issues

### Issue 2: Missing Known Anomalies

**Symptoms**: Obvious anomalies not detected
**Solutions**:

- Increase contamination parameter
- Try different algorithms
- Engineer better features
- Check feature scaling

### Issue 3: Inconsistent Results

**Symptoms**: Different results each run
**Solutions**:

- Set random_state parameter
- Use more data for training
- Ensemble multiple runs
- Check for data leakage

## 📈 Advanced Tips

### Feature Engineering

```python
# Create time-based features
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

# Create ratio features
df['amount_to_avg_ratio'] = df['amount'] / df.groupby('customer_id')['amount'].transform('mean')

# Create frequency features
df['customer_transaction_count'] = df.groupby('customer_id').cumcount() + 1
```

### Validation Strategies

```python
# Time-based validation (for time series data)
train_data = df[df['date'] < '2023-10-01']
test_data = df[df['date'] >= '2023-10-01']

# Cross-validation for anomaly detection
from sklearn.model_selection import cross_val_score
scores = cross_val_score(detector, X, scoring='roc_auc', cv=5)
print(f"Average AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## 🎯 Next Steps

### Immediate Actions

1. **Investigate detected anomalies** - Are they actually problematic?
2. **Adjust parameters** - Fine-tune based on your domain knowledge
3. **Try different algorithms** - See which works best for your data
4. **Create validation process** - How will you verify results?

### Advanced Learning

1. **[Algorithm Deep Dive](../../reference/algorithms/README.md)** - Learn about different algorithms
2. **[Feature Engineering Guide](../advanced-features/feature-engineering.md)** - Create better features
3. **[Production Deployment](../../deployment/README.md)** - Deploy your detector
4. **[Model Evaluation](../advanced-features/model-evaluation.md)** - Measure performance

### Apply to Your Data

1. **Load your own dataset** - Replace sample data with real data
2. **Domain-specific features** - Add features relevant to your use case
3. **Business logic** - Incorporate domain knowledge
4. **Monitoring setup** - Track performance over time

---

**Congratulations!** 🎉 You've successfully completed your first anomaly detection. You now have the foundation to apply these techniques to your own data and use cases.

**Ready for more?** Continue with the **[Algorithm Selection Guide](../advanced-features/algorithm-selection.md)** →
