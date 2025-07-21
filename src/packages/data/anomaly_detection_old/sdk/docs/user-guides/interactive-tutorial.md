# Interactive Tutorial: Hands-On Anomaly Detection

## Welcome to Your Hands-On Journey!

This interactive tutorial takes you through real anomaly detection scenarios with actual code you can run. By the end, you'll have practical experience with the most common anomaly detection tasks.

## What You'll Learn

- ✅ **Hands-on practice** with real datasets
- ✅ **Step-by-step examples** you can copy and run
- ✅ **Common patterns** used in production
- ✅ **Troubleshooting tips** for real-world problems
- ✅ **Best practices** from industry experience

## Prerequisites

### Installation
```bash
# Basic installation
pip install pynomaly

# Full installation with all features
pip install pynomaly[all]

# For this tutorial, also install:
pip install jupyter matplotlib seaborn plotly
```

### Download Tutorial Data
```bash
# Download sample datasets for this tutorial
curl -O https://datasets.pynomaly.com/tutorial/credit_card_sample.csv
curl -O https://datasets.pynomaly.com/tutorial/sensor_data_sample.csv
curl -O https://datasets.pynomaly.com/tutorial/network_logs_sample.csv
```

## Table of Contents

1. [Scenario 1: Credit Card Fraud Detection](#scenario-1)
2. [Scenario 2: Industrial Sensor Monitoring](#scenario-2)
3. [Scenario 3: Network Security Monitoring](#scenario-3)
4. [Advanced Topics](#advanced-topics)
5. [Building Your Own Detector](#custom-detector)

---

## Scenario 1: Credit Card Fraud Detection {#scenario-1}

**Your Role**: Data analyst at a bank  
**Your Task**: Identify potentially fraudulent credit card transactions  
**Time to Complete**: 20-30 minutes

### Step 1: Understanding the Problem

You have a dataset of credit card transactions and need to identify which ones might be fraudulent. Let's start by exploring the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pynomaly import PynomalyClient

# Load the credit card transaction data
df = pd.read_csv('credit_card_sample.csv')

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())
```

**Expected Output:**
```
Dataset Overview:
Shape: (10000, 8)
Columns: ['transaction_id', 'amount', 'merchant_category', 'hour_of_day', 'day_of_week', 'time_since_last', 'customer_id', 'is_weekend']

Basic statistics:
              amount  hour_of_day  day_of_week  time_since_last
count   10000.000000 10000.000000  10000.000000      10000.000000
mean       87.234500    12.456700      3.501200         18.234000
std       156.789000     6.789000      2.123000         24.567000
min         0.010000     0.000000      0.000000          0.100000
max      8999.990000    23.000000      6.000000        168.000000
```

### Step 2: Data Exploration and Visualization

Let's visualize the data to understand normal vs potentially fraudulent patterns:

```python
# Create visualizations to understand the data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Transaction amounts distribution
axes[0, 0].hist(df['amount'], bins=50, alpha=0.7)
axes[0, 0].set_title('Transaction Amount Distribution')
axes[0, 0].set_xlabel('Amount ($)')
axes[0, 0].set_yscale('log')  # Log scale due to wide range

# Transactions by hour of day
hour_counts = df.groupby('hour_of_day').size()
axes[0, 1].bar(hour_counts.index, hour_counts.values)
axes[0, 1].set_title('Transactions by Hour of Day')
axes[0, 1].set_xlabel('Hour')

# Time since last transaction
axes[1, 0].hist(df['time_since_last'], bins=30, alpha=0.7)
axes[1, 0].set_title('Time Since Last Transaction (hours)')
axes[1, 0].set_xlabel('Hours')

# Merchant category distribution
category_counts = df['merchant_category'].value_counts()
axes[1, 1].pie(category_counts.values[:5], labels=category_counts.index[:5], autopct='%1.1f%%')
axes[1, 1].set_title('Top 5 Merchant Categories')

plt.tight_layout()
plt.show()

# Look for obvious outliers
print("\nPotential outliers:")
print(f"High amount transactions (>$1000): {len(df[df['amount'] > 1000])}")
print(f"Very quick transactions (<1 min since last): {len(df[df['time_since_last'] < 0.017])}")
print(f"Late night transactions (11PM-5AM): {len(df[df['hour_of_day'].isin([23, 0, 1, 2, 3, 4, 5])])}")
```

### Step 3: Basic Anomaly Detection

Now let's detect anomalies using Pynomaly's default settings:

```python
# Initialize Pynomaly client
client = PynomalyClient()

# Prepare features for anomaly detection
features = df[['amount', 'hour_of_day', 'day_of_week', 'time_since_last']].copy()

# Handle categorical data - convert merchant_category to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['merchant_category_encoded'] = le.fit_transform(df['merchant_category'])

# Add derived features that might indicate fraud
features['is_weekend'] = df['is_weekend']
features['is_night'] = df['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
features['amount_log'] = np.log1p(features['amount'])  # Log transform for better distribution

print("Features prepared:")
print(features.head())
print(f"Feature shape: {features.shape}")

# Detect anomalies
print("\nRunning anomaly detection...")
results = client.detect_anomalies(
    data=features,
    algorithm='isolation_forest',
    contamination=0.02,  # Expect 2% fraud rate
    return_explanations=True
)

print("Detection complete!")
print(f"Anomalies found: {sum(results['predictions'])}")
print(f"Anomaly rate: {sum(results['predictions'])/len(results['predictions'])*100:.2f}%")
```

### Step 4: Analyzing Results

Let's examine which transactions were flagged and why:

```python
# Add results to dataframe for analysis
df['is_anomaly'] = results['predictions']
df['anomaly_score'] = results['scores']

# Look at the most suspicious transactions
anomalies = df[df['is_anomaly'] == True].copy()
anomalies = anomalies.sort_values('anomaly_score', ascending=False)

print("Top 10 Most Suspicious Transactions:")
print(anomalies[['transaction_id', 'amount', 'merchant_category', 'hour_of_day', 'anomaly_score']].head(10))

# Analyze patterns in anomalies
print("\nAnomaly Analysis:")
print(f"Average amount - Normal: ${df[df['is_anomaly']==False]['amount'].mean():.2f}")
print(f"Average amount - Anomalies: ${df[df['is_anomaly']==True]['amount'].mean():.2f}")

print(f"\nNight transactions - Normal: {df[df['is_anomaly']==False]['hour_of_day'].isin([22,23,0,1,2,3,4,5]).mean()*100:.1f}%")
print(f"Night transactions - Anomalies: {df[df['is_anomaly']==True]['hour_of_day'].isin([22,23,0,1,2,3,4,5]).mean()*100:.1f}%")

# Visualize anomalies
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Amount distribution
axes[0].hist(df[df['is_anomaly']==False]['amount'], bins=30, alpha=0.7, label='Normal', density=True)
axes[0].hist(df[df['is_anomaly']==True]['amount'], bins=30, alpha=0.7, label='Anomalies', density=True)
axes[0].set_xlabel('Transaction Amount')
axes[0].set_ylabel('Density')
axes[0].set_title('Amount Distribution: Normal vs Anomalies')
axes[0].legend()
axes[0].set_yscale('log')

# Scatter plot: Amount vs Time since last transaction
normal = df[df['is_anomaly']==False]
anomaly = df[df['is_anomaly']==True]

axes[1].scatter(normal['time_since_last'], normal['amount'], alpha=0.5, label='Normal', s=20)
axes[1].scatter(anomaly['time_since_last'], anomaly['amount'], alpha=0.8, label='Anomalies', s=30, color='red')
axes[1].set_xlabel('Time Since Last Transaction (hours)')
axes[1].set_ylabel('Amount ($)')
axes[1].set_title('Amount vs Time Pattern')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Step 5: Understanding Feature Importance

Let's see which features contributed most to the anomaly detection:

```python
# Get feature importance/explanations
if 'explanations' in results:
    explanations = results['explanations']
    
    # Calculate average feature importance for anomalies
    feature_importance = {}
    anomaly_indices = [i for i, pred in enumerate(results['predictions']) if pred]
    
    for idx in anomaly_indices[:10]:  # Top 10 anomalies
        if idx < len(explanations):
            explanation = explanations[idx]
            for feature, importance in explanation.get('feature_importance', {}).items():
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(abs(importance))
    
    # Average importance
    avg_importance = {k: np.mean(v) for k, v in feature_importance.items()}
    
    # Plot feature importance
    features_sorted = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names, importance_values = zip(*features_sorted)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_values)
    plt.title('Average Feature Importance for Anomaly Detection')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("Feature Importance (higher = more important for detecting anomalies):")
    for feature, importance in features_sorted:
        print(f"{feature}: {importance:.3f}")
```

### Step 6: Practical Business Application

Now let's implement a practical fraud detection system:

```python
def fraud_detector(transaction_data):
    """
    Production-ready fraud detection function
    """
    # Prepare features
    features = {
        'amount': transaction_data['amount'],
        'hour_of_day': transaction_data['hour_of_day'],
        'day_of_week': transaction_data['day_of_week'],
        'time_since_last': transaction_data['time_since_last'],
        'is_weekend': transaction_data['is_weekend'],
        'is_night': 1 if transaction_data['hour_of_day'] in [22,23,0,1,2,3,4,5] else 0,
        'amount_log': np.log1p(transaction_data['amount'])
    }
    
    # Run detection
    result = client.detect_anomalies(
        data=[features],
        algorithm='isolation_forest',
        contamination=0.02
    )
    
    anomaly_score = result['scores'][0]
    is_suspicious = result['predictions'][0]
    
    # Business logic for handling results
    if is_suspicious:
        if anomaly_score > 0.8:
            action = "BLOCK"
            message = "Transaction blocked - high fraud risk"
        elif anomaly_score > 0.6:
            action = "REVIEW"
            message = "Transaction flagged for manual review"
        else:
            action = "MONITOR"
            message = "Transaction approved but monitoring recommended"
    else:
        action = "APPROVE"
        message = "Transaction approved"
    
    return {
        'action': action,
        'message': message,
        'fraud_score': anomaly_score,
        'confidence': 'high' if anomaly_score > 0.7 else 'medium' if anomaly_score > 0.4 else 'low'
    }

# Test the function
test_transaction = {
    'amount': 2500.00,  # High amount
    'hour_of_day': 2,   # Late night
    'day_of_week': 6,   # Weekend
    'time_since_last': 0.25,  # Very recent
    'is_weekend': 1
}

result = fraud_detector(test_transaction)
print("Fraud Detection Result:")
print(f"Action: {result['action']}")
print(f"Message: {result['message']}")
print(f"Fraud Score: {result['fraud_score']:.3f}")
print(f"Confidence: {result['confidence']}")
```

**🎉 Congratulations!** You've successfully:
- Explored transaction data
- Detected anomalies using machine learning
- Analyzed which features indicate fraud
- Built a practical fraud detection system

---

## Scenario 2: Industrial Sensor Monitoring {#scenario-2}

**Your Role**: Manufacturing engineer  
**Your Task**: Monitor industrial sensors to detect equipment problems  
**Time to Complete**: 25-35 minutes

### Step 1: Understanding Industrial Data

Industrial sensors generate continuous data streams. Let's simulate monitoring a manufacturing line:

```python
# Load industrial sensor data
sensor_df = pd.read_csv('sensor_data_sample.csv')

print("Sensor Data Overview:")
print(f"Shape: {sensor_df.shape}")
print(f"Columns: {list(sensor_df.columns)}")
print(f"Time range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")

# Convert timestamp to datetime
sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
sensor_df = sensor_df.sort_values('timestamp')

print("\nSensor readings summary:")
sensor_cols = ['temperature', 'pressure', 'vibration', 'power_consumption', 'production_rate']
print(sensor_df[sensor_cols].describe())
```

### Step 2: Time Series Visualization

Industrial data is time-dependent, so let's visualize trends:

```python
# Create time series plots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

sensor_metrics = ['temperature', 'pressure', 'vibration', 'power_consumption', 'production_rate']

for i, metric in enumerate(sensor_metrics):
    axes[i].plot(sensor_df['timestamp'], sensor_df[metric], linewidth=0.8)
    axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)

# Correlation matrix
correlation = sensor_df[sensor_metrics].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[5])
axes[5].set_title('Sensor Correlation Matrix')

plt.tight_layout()
plt.show()

# Look for obvious anomalies
print("Potential equipment issues:")
print(f"High temperature readings (>100°C): {len(sensor_df[sensor_df['temperature'] > 100])}")
print(f"Low production rate (<80%): {len(sensor_df[sensor_df['production_rate'] < 80])}")
print(f"High vibration (>0.5): {len(sensor_df[sensor_df['vibration'] > 0.5])}")
```

### Step 3: Time-Series Anomaly Detection

For industrial sensors, we need time-aware anomaly detection:

```python
# Prepare time-series features
def create_time_features(df):
    """Create time-based features for better anomaly detection"""
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Rolling statistics (sliding window features)
    window = 10  # 10-point rolling window
    for col in sensor_metrics:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window).std()
        df[f'{col}_deviation'] = (df[col] - df[f'{col}_rolling_mean']).abs()
    
    # Rate of change features
    for col in sensor_metrics:
        df[f'{col}_rate_change'] = df[col].diff().abs()
    
    return df

# Create enhanced features
sensor_enhanced = create_time_features(sensor_df)

# Select features for anomaly detection
feature_cols = sensor_metrics + [col for col in sensor_enhanced.columns if 'rolling' in col or 'rate_change' in col]
feature_cols = [col for col in feature_cols if col in sensor_enhanced.columns]

# Remove rows with NaN (from rolling calculations)
sensor_features = sensor_enhanced[feature_cols].dropna()
print(f"Features for detection: {len(feature_cols)}")
print(f"Data points after preprocessing: {len(sensor_features)}")

# Time-series aware anomaly detection
print("Running time-series anomaly detection...")
results = client.detect_anomalies(
    data=sensor_features,
    algorithm='lstm_autoencoder',  # Good for time series
    contamination=0.05,  # 5% of readings might be anomalous
    time_series=True
)

print(f"Anomalies detected: {sum(results['predictions'])}")
print(f"Anomaly rate: {sum(results['predictions'])/len(results['predictions'])*100:.2f}%")
```

### Step 4: Anomaly Analysis and Alerting

Let's analyze the detected anomalies and implement an alerting system:

```python
# Add results back to dataframe (adjusting for dropped NaN rows)
sensor_enhanced_clean = sensor_enhanced.dropna()
sensor_enhanced_clean['is_anomaly'] = results['predictions']
sensor_enhanced_clean['anomaly_score'] = results['scores']

# Analyze anomalies by severity
high_severity = sensor_enhanced_clean[
    (sensor_enhanced_clean['is_anomaly'] == True) & 
    (sensor_enhanced_clean['anomaly_score'] > 0.8)
]
medium_severity = sensor_enhanced_clean[
    (sensor_enhanced_clean['is_anomaly'] == True) & 
    (sensor_enhanced_clean['anomaly_score'] <= 0.8) &
    (sensor_enhanced_clean['anomaly_score'] > 0.6)
]

print(f"High severity anomalies (score > 0.8): {len(high_severity)}")
print(f"Medium severity anomalies (0.6-0.8): {len(medium_severity)}")

# Show the most critical anomalies
if len(high_severity) > 0:
    print("\nMost Critical Equipment Issues:")
    critical_cols = ['timestamp'] + sensor_metrics + ['anomaly_score']
    print(high_severity.nlargest(5, 'anomaly_score')[critical_cols])

# Visualize anomalies on time series
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Temperature with anomalies highlighted
normal_data = sensor_enhanced_clean[sensor_enhanced_clean['is_anomaly'] == False]
anomaly_data = sensor_enhanced_clean[sensor_enhanced_clean['is_anomaly'] == True]

axes[0].plot(normal_data['timestamp'], normal_data['temperature'], 'b-', alpha=0.7, label='Normal')
axes[0].scatter(anomaly_data['timestamp'], anomaly_data['temperature'], c='red', s=50, label='Anomalies', zorder=5)
axes[0].set_title('Temperature Monitoring with Anomaly Detection')
axes[0].set_ylabel('Temperature (°C)')
axes[0].legend()

# Production rate with anomalies
axes[1].plot(normal_data['timestamp'], normal_data['production_rate'], 'g-', alpha=0.7, label='Normal')
axes[1].scatter(anomaly_data['timestamp'], anomaly_data['production_rate'], c='red', s=50, label='Anomalies', zorder=5)
axes[1].set_title('Production Rate Monitoring with Anomaly Detection')
axes[1].set_ylabel('Production Rate (%)')
axes[1].set_xlabel('Time')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Step 5: Implementing Real-Time Monitoring

Let's create a real-time monitoring system:

```python
class IndustrialMonitor:
    def __init__(self, alert_thresholds=None):
        self.client = PynomalyClient()
        self.alert_thresholds = alert_thresholds or {
            'critical': 0.8,
            'warning': 0.6,
            'info': 0.4
        }
        self.alert_history = []
    
    def process_sensor_reading(self, sensor_data):
        """Process a single sensor reading in real-time"""
        
        # Prepare features (in production, you'd maintain rolling windows)
        features = {
            'temperature': sensor_data['temperature'],
            'pressure': sensor_data['pressure'],
            'vibration': sensor_data['vibration'],
            'power_consumption': sensor_data['power_consumption'],
            'production_rate': sensor_data['production_rate']
        }
        
        # Detect anomaly
        result = self.client.detect_anomalies(
            data=[features],
            algorithm='isolation_forest',  # Fast for real-time
            contamination=0.05
        )
        
        anomaly_score = result['scores'][0]
        is_anomaly = result['predictions'][0]
        
        # Generate alerts based on severity
        alert_level = self._determine_alert_level(anomaly_score)
        
        if alert_level:
            alert = {
                'timestamp': sensor_data['timestamp'],
                'alert_level': alert_level,
                'anomaly_score': anomaly_score,
                'sensor_readings': features,
                'message': self._generate_alert_message(alert_level, features, anomaly_score)
            }
            
            self.alert_history.append(alert)
            self._send_alert(alert)
            
            return alert
        
        return None
    
    def _determine_alert_level(self, score):
        """Determine alert level based on anomaly score"""
        if score >= self.alert_thresholds['critical']:
            return 'CRITICAL'
        elif score >= self.alert_thresholds['warning']:
            return 'WARNING'
        elif score >= self.alert_thresholds['info']:
            return 'INFO'
        return None
    
    def _generate_alert_message(self, level, features, score):
        """Generate human-readable alert message"""
        messages = {
            'CRITICAL': "CRITICAL: Immediate equipment inspection required",
            'WARNING': "WARNING: Equipment anomaly detected - monitor closely",
            'INFO': "INFO: Minor deviation detected in equipment behavior"
        }
        
        # Identify problematic sensors
        problems = []
        if features['temperature'] > 95:
            problems.append("high temperature")
        if features['vibration'] > 0.4:
            problems.append("excessive vibration")
        if features['production_rate'] < 85:
            problems.append("low production rate")
        
        message = messages[level]
        if problems:
            message += f" - Issues: {', '.join(problems)}"
        
        return message
    
    def _send_alert(self, alert):
        """Send alert (in production, this would send emails, SMS, etc.)"""
        print(f"🚨 {alert['alert_level']} ALERT 🚨")
        print(f"Time: {alert['timestamp']}")
        print(f"Score: {alert['anomaly_score']:.3f}")
        print(f"Message: {alert['message']}")
        print("-" * 50)

# Test the real-time monitor
monitor = IndustrialMonitor()

# Simulate some sensor readings
test_readings = [
    {
        'timestamp': '2024-01-15 10:00:00',
        'temperature': 85.2,
        'pressure': 14.7,
        'vibration': 0.15,
        'power_consumption': 450,
        'production_rate': 98.5
    },
    {
        'timestamp': '2024-01-15 10:01:00',
        'temperature': 105.8,  # High temperature!
        'pressure': 14.9,
        'vibration': 0.45,     # High vibration!
        'power_consumption': 480,
        'production_rate': 75.2  # Low production!
    }
]

print("Testing Real-Time Industrial Monitor:")
print("="*50)

for reading in test_readings:
    alert = monitor.process_sensor_reading(reading)
    if not alert:
        print(f"✅ Normal operation at {reading['timestamp']}")
        print("-" * 50)
```

**🎉 Excellent work!** You've created:
- Time-series anomaly detection for industrial sensors
- Real-time monitoring system with alerts
- Severity-based alerting with business logic

---

## Scenario 3: Network Security Monitoring {#scenario-3}

**Your Role**: Cybersecurity analyst  
**Your Task**: Detect network intrusions and security threats  
**Time to Complete**: 20-30 minutes

### Step 1: Network Log Analysis

Network security involves analyzing connection patterns and identifying suspicious activity:

```python
# Load network log data
network_df = pd.read_csv('network_logs_sample.csv')

print("Network Logs Overview:")
print(f"Shape: {network_df.shape}")
print(f"Time range: {network_df['timestamp'].min()} to {network_df['timestamp'].max()}")
print("\nColumns:", list(network_df.columns))

# Convert timestamp and examine data
network_df['timestamp'] = pd.to_datetime(network_df['timestamp'])
print("\nSample network connections:")
print(network_df.head())

# Analyze connection patterns
print("\nNetwork Activity Summary:")
print(f"Unique source IPs: {network_df['src_ip'].nunique()}")
print(f"Unique destination IPs: {network_df['dst_ip'].nunique()}")
print(f"Unique protocols: {network_df['protocol'].unique()}")
print(f"Average bytes transferred: {network_df['bytes'].mean():.0f}")
```

### Step 2: Feature Engineering for Security

Security analysis requires specific network-based features:

```python
def create_network_features(df):
    """Create security-relevant features from network logs"""
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                              (df['day_of_week'] < 5)).astype(int)
    
    # Connection features
    df['duration_log'] = np.log1p(df['duration'])
    df['bytes_log'] = np.log1p(df['bytes'])
    df['packets_log'] = np.log1p(df['packets'])
    
    # Rate-based features (connections per minute for each IP)
    df['timestamp_minute'] = df['timestamp'].dt.floor('min')
    
    # Source IP connection frequency
    src_counts = df.groupby(['src_ip', 'timestamp_minute']).size().reset_index(name='src_connections_per_minute')
    df = df.merge(src_counts, on=['src_ip', 'timestamp_minute'], how='left')
    
    # Destination IP connection frequency  
    dst_counts = df.groupby(['dst_ip', 'timestamp_minute']).size().reset_index(name='dst_connections_per_minute')
    df = df.merge(dst_counts, on=['dst_ip', 'timestamp_minute'], how='left')
    
    # Port analysis
    df['is_common_port'] = df['dst_port'].isin([80, 443, 22, 21, 25, 53, 993, 995]).astype(int)
    df['is_high_port'] = (df['dst_port'] > 1024).astype(int)
    
    # Protocol encoding
    protocol_encoder = LabelEncoder()
    df['protocol_encoded'] = protocol_encoder.fit_transform(df['protocol'])
    
    # Suspicious patterns
    df['very_short_connection'] = (df['duration'] < 1).astype(int)
    df['very_long_connection'] = (df['duration'] > 3600).astype(int)  # > 1 hour
    df['large_transfer'] = (df['bytes'] > df['bytes'].quantile(0.95)).astype(int)
    df['many_packets'] = (df['packets'] > df['packets'].quantile(0.95)).astype(int)
    
    return df

# Create security features
network_enhanced = create_network_features(network_df)

# Select features for anomaly detection
security_features = [
    'duration_log', 'bytes_log', 'packets_log', 'dst_port',
    'hour', 'day_of_week', 'is_business_hours',
    'src_connections_per_minute', 'dst_connections_per_minute',
    'is_common_port', 'is_high_port', 'protocol_encoded',
    'very_short_connection', 'very_long_connection', 
    'large_transfer', 'many_packets'
]

network_features = network_enhanced[security_features].fillna(0)
print(f"Security features created: {len(security_features)}")
print(f"Data points: {len(network_features)}")
```

### Step 3: Security Anomaly Detection

Run anomaly detection optimized for security scenarios:

```python
# Network security anomaly detection
print("Running network security anomaly detection...")

results = client.detect_anomalies(
    data=network_features,
    algorithm='isolation_forest',  # Good for detecting various attack patterns
    contamination=0.01,  # 1% of network traffic might be malicious
    n_estimators=200,  # More trees for better accuracy
    random_state=42
)

print(f"Potential security threats detected: {sum(results['predictions'])}")
print(f"Threat rate: {sum(results['predictions'])/len(results['predictions'])*100:.3f}%")

# Add results to dataframe
network_enhanced['is_threat'] = results['predictions']
network_enhanced['threat_score'] = results['scores']

# Analyze detected threats
threats = network_enhanced[network_enhanced['is_threat'] == True].copy()
threats = threats.sort_values('threat_score', ascending=False)

print("\nTop 10 Security Threats:")
threat_cols = ['timestamp', 'src_ip', 'dst_ip', 'dst_port', 'protocol', 'bytes', 'threat_score']
print(threats[threat_cols].head(10))
```

### Step 4: Threat Analysis and Classification

Let's analyze the types of threats detected:

```python
def classify_threat_type(row):
    """Classify the type of security threat based on connection characteristics"""
    
    threat_indicators = []
    
    # Port scan detection
    if row['src_connections_per_minute'] > 20:
        threat_indicators.append("Port Scan")
    
    # DDoS detection
    if row['dst_connections_per_minute'] > 50:
        threat_indicators.append("DDoS")
    
    # Data exfiltration
    if row['bytes'] > network_enhanced['bytes'].quantile(0.99) and row['duration'] > 300:
        threat_indicators.append("Data Exfiltration")
    
    # Brute force attack
    if row['very_short_connection'] and row['src_connections_per_minute'] > 10 and row['dst_port'] in [22, 3389, 21]:
        threat_indicators.append("Brute Force")
    
    # Unusual protocol/port
    if not row['is_common_port'] and row['is_high_port']:
        threat_indicators.append("Unusual Service")
    
    # After hours activity
    if not row['is_business_hours'] and row['bytes'] > network_enhanced['bytes'].median():
        threat_indicators.append("After Hours Activity")
    
    return threat_indicators if threat_indicators else ["Unknown"]

# Classify threats
threats['threat_types'] = threats.apply(classify_threat_type, axis=1)

# Analyze threat distribution
all_threat_types = [threat for threats_list in threats['threat_types'] for threat in threats_list]
threat_counts = pd.Series(all_threat_types).value_counts()

print("Threat Type Distribution:")
for threat_type, count in threat_counts.items():
    print(f"{threat_type}: {count} incidents")

# Visualize threats
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Threat scores distribution
axes[0, 0].hist(network_enhanced[network_enhanced['is_threat']==False]['threat_score'], 
                bins=30, alpha=0.7, label='Normal', density=True)
axes[0, 0].hist(network_enhanced[network_enhanced['is_threat']==True]['threat_score'], 
                bins=30, alpha=0.7, label='Threats', density=True)
axes[0, 0].set_title('Threat Score Distribution')
axes[0, 0].set_xlabel('Threat Score')
axes[0, 0].legend()

# Threats by hour
threat_by_hour = threats.groupby('hour').size()
normal_by_hour = network_enhanced[network_enhanced['is_threat']==False].groupby('hour').size()

x = range(24)
axes[0, 1].plot(x, normal_by_hour.reindex(x, fill_value=0), label='Normal Traffic', marker='o')
axes[0, 1].plot(x, threat_by_hour.reindex(x, fill_value=0), label='Threats', marker='s', color='red')
axes[0, 1].set_title('Network Activity by Hour')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Number of Connections')
axes[0, 1].legend()

# Threat types pie chart
axes[1, 0].pie(threat_counts.values, labels=threat_counts.index, autopct='%1.1f%%')
axes[1, 0].set_title('Distribution of Threat Types')

# Bytes vs Duration for threats
normal = network_enhanced[network_enhanced['is_threat']==False]
threats_plot = network_enhanced[network_enhanced['is_threat']==True]

axes[1, 1].scatter(normal['duration'], normal['bytes'], alpha=0.3, label='Normal', s=10)
axes[1, 1].scatter(threats_plot['duration'], threats_plot['bytes'], alpha=0.8, label='Threats', s=20, color='red')
axes[1, 1].set_xlabel('Duration (seconds)')
axes[1, 1].set_ylabel('Bytes')
axes[1, 1].set_title('Connection Duration vs Bytes Transferred')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### Step 5: Security Incident Response System

Create an automated incident response system:

```python
class SecurityMonitor:
    def __init__(self):
        self.client = PynomalyClient()
        self.incident_counter = 0
        self.blocked_ips = set()
        self.alert_history = []
    
    def analyze_connection(self, connection_data):
        """Analyze a single network connection for threats"""
        
        # Prepare features (simplified for demo)
        features = {
            'duration_log': np.log1p(connection_data['duration']),
            'bytes_log': np.log1p(connection_data['bytes']),
            'packets_log': np.log1p(connection_data['packets']),
            'dst_port': connection_data['dst_port'],
            'hour': pd.to_datetime(connection_data['timestamp']).hour,
            'is_common_port': 1 if connection_data['dst_port'] in [80, 443, 22] else 0
        }
        
        # Detect anomaly
        result = self.client.detect_anomalies(
            data=[features],
            algorithm='isolation_forest',
            contamination=0.01
        )
        
        threat_score = result['scores'][0]
        is_threat = result['predictions'][0]
        
        if is_threat:
            incident = self._create_incident(connection_data, threat_score)
            self._respond_to_incident(incident)
            return incident
        
        return None
    
    def _create_incident(self, connection_data, threat_score):
        """Create security incident record"""
        self.incident_counter += 1
        
        # Classify threat type
        threat_type = self._classify_threat(connection_data, threat_score)
        severity = self._determine_severity(threat_score, threat_type)
        
        incident = {
            'incident_id': f"INC_{self.incident_counter:05d}",
            'timestamp': connection_data['timestamp'],
            'src_ip': connection_data['src_ip'],
            'dst_ip': connection_data['dst_ip'],
            'dst_port': connection_data['dst_port'],
            'protocol': connection_data['protocol'],
            'threat_score': threat_score,
            'threat_type': threat_type,
            'severity': severity,
            'status': 'ACTIVE'
        }
        
        self.alert_history.append(incident)
        return incident
    
    def _classify_threat(self, connection_data, threat_score):
        """Classify the type of threat"""
        
        # Simple classification logic
        if connection_data['dst_port'] in [22, 3389] and threat_score > 0.8:
            return "Brute Force Attack"
        elif connection_data['bytes'] > 10000000:  # > 10MB
            return "Data Exfiltration"
        elif connection_data['dst_port'] > 8000:
            return "Unusual Service Access"
        else:
            return "Suspicious Activity"
    
    def _determine_severity(self, threat_score, threat_type):
        """Determine incident severity"""
        
        if threat_score > 0.9 or threat_type in ["Brute Force Attack", "Data Exfiltration"]:
            return "HIGH"
        elif threat_score > 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _respond_to_incident(self, incident):
        """Automated incident response"""
        
        print(f"🔒 SECURITY INCIDENT DETECTED 🔒")
        print(f"Incident ID: {incident['incident_id']}")
        print(f"Severity: {incident['severity']}")
        print(f"Type: {incident['threat_type']}")
        print(f"Source IP: {incident['src_ip']}")
        print(f"Target: {incident['dst_ip']}:{incident['dst_port']}")
        print(f"Threat Score: {incident['threat_score']:.3f}")
        
        # Automated response based on severity
        if incident['severity'] == "HIGH":
            print("🚨 AUTOMATIC RESPONSE: Blocking source IP")
            self.blocked_ips.add(incident['src_ip'])
            print("📧 ALERT: Notifying SOC team immediately")
            
        elif incident['severity'] == "MEDIUM":
            print("⚠️  RESPONSE: Flagging for investigation")
            print("📧 ALERT: Adding to SOC queue")
            
        else:
            print("ℹ️  RESPONSE: Logging for analysis")
        
        print("-" * 60)

# Test the security monitor
security_monitor = SecurityMonitor()

# Simulate some network connections
test_connections = [
    {
        'timestamp': '2024-01-15 14:30:00',
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.50',
        'dst_port': 80,
        'protocol': 'TCP',
        'duration': 45,
        'bytes': 1500,
        'packets': 10
    },
    {
        'timestamp': '2024-01-15 02:15:00',  # Suspicious time
        'src_ip': '203.0.113.45',
        'dst_ip': '10.0.0.50',
        'dst_port': 22,  # SSH
        'protocol': 'TCP',
        'duration': 2,   # Very short
        'bytes': 100,
        'packets': 5
    },
    {
        'timestamp': '2024-01-15 03:45:00',
        'src_ip': '198.51.100.25',
        'dst_ip': '10.0.0.100',
        'dst_port': 443,
        'protocol': 'TCP',
        'duration': 3600,    # 1 hour
        'bytes': 50000000,   # 50MB - potential data exfiltration
        'packets': 35000
    }
]

print("Testing Security Monitor:")
print("=" * 60)

for connection in test_connections:
    incident = security_monitor.analyze_connection(connection)
    if not incident:
        print(f"✅ Normal connection from {connection['src_ip']} at {connection['timestamp']}")
        print("-" * 60)

print(f"\nBlocked IPs: {list(security_monitor.blocked_ips)}")
print(f"Total incidents: {len(security_monitor.alert_history)}")
```

**🎉 Outstanding!** You've built:
- Network security anomaly detection
- Threat classification system
- Automated incident response
- IP blocking and alerting system

---

## Advanced Topics {#advanced-topics}

### Ensemble Methods for Better Accuracy

Combine multiple algorithms for more robust detection:

```python
# Ensemble approach - combine multiple algorithms
def ensemble_detection(data):
    """Use multiple algorithms and combine their results"""
    
    algorithms = ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
    results = {}
    
    for algorithm in algorithms:
        result = client.detect_anomalies(
            data=data,
            algorithm=algorithm,
            contamination=0.05
        )
        results[algorithm] = result
    
    # Combine results using voting
    ensemble_predictions = []
    ensemble_scores = []
    
    for i in range(len(data)):
        # Count votes for anomaly
        votes = sum([results[alg]['predictions'][i] for alg in algorithms])
        # Average scores
        avg_score = np.mean([results[alg]['scores'][i] for alg in algorithms])
        
        # Majority vote (at least 2 out of 3 algorithms agree)
        is_anomaly = votes >= 2
        
        ensemble_predictions.append(is_anomaly)
        ensemble_scores.append(avg_score)
    
    return {
        'predictions': ensemble_predictions,
        'scores': ensemble_scores,
        'individual_results': results
    }

# Test ensemble method
print("Testing Ensemble Method:")
sample_data = features.head(100)  # Use credit card data from earlier
ensemble_result = ensemble_detection(sample_data)

print(f"Ensemble anomalies: {sum(ensemble_result['predictions'])}")
print(f"Individual algorithm results:")
for alg, result in ensemble_result['individual_results'].items():
    print(f"  {alg}: {sum(result['predictions'])} anomalies")
```

### Handling Imbalanced Data

Real-world data is often imbalanced. Here's how to handle it:

```python
# Techniques for imbalanced datasets
def handle_imbalanced_data(data, known_anomalies=None):
    """Handle imbalanced datasets with various techniques"""
    
    if known_anomalies is not None:
        # If you have labeled data, use semi-supervised approaches
        
        # SMOTE for oversampling anomalies
        from imblearn.over_sampling import SMOTE
        
        X = data.values
        y = known_anomalies
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"Original dataset: {len(data)} samples")
        print(f"Resampled dataset: {len(X_resampled)} samples")
        print(f"Original anomaly ratio: {sum(y)/len(y)*100:.2f}%")
        print(f"Resampled anomaly ratio: {sum(y_resampled)/len(y_resampled)*100:.2f}%")
        
        return pd.DataFrame(X_resampled, columns=data.columns), y_resampled
    
    else:
        # Unsupervised techniques
        
        # Adjust contamination based on domain knowledge
        contamination_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        results = {}
        for contamination in contamination_rates:
            result = client.detect_anomalies(
                data=data,
                algorithm='isolation_forest',
                contamination=contamination
            )
            results[contamination] = {
                'anomaly_count': sum(result['predictions']),
                'mean_score': np.mean(result['scores'])
            }
        
        print("Contamination Rate Analysis:")
        for rate, info in results.items():
            print(f"Rate {rate}: {info['anomaly_count']} anomalies, avg score: {info['mean_score']:.3f}")
        
        return results

# Test with credit card data
print("Analyzing contamination rates:")
contamination_analysis = handle_imbalanced_data(features.head(1000))
```

### Model Validation and Testing

Properly validate your anomaly detection models:

```python
def validate_anomaly_model(data, test_size=0.2):
    """Comprehensive model validation"""
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Train model on training data
    train_result = client.detect_anomalies(
        data=train_data,
        algorithm='isolation_forest',
        contamination=0.05
    )
    
    # Test on test data
    test_result = client.detect_anomalies(
        data=test_data,
        algorithm='isolation_forest',
        contamination=0.05
    )
    
    # If you have ground truth labels (synthetic for demo)
    # In practice, you'd have actual labeled data
    synthetic_labels_train = (np.random.random(len(train_data)) < 0.05).astype(int)
    synthetic_labels_test = (np.random.random(len(test_data)) < 0.05).astype(int)
    
    print("Model Validation Results:")
    print("=" * 40)
    
    # Training performance
    print("Training Set Performance:")
    print(classification_report(synthetic_labels_train, train_result['predictions']))
    
    # Test performance
    print("\nTest Set Performance:")
    print(classification_report(synthetic_labels_test, test_result['predictions']))
    
    # Stability check - run multiple times
    stability_scores = []
    for i in range(5):
        result = client.detect_anomalies(
            data=test_data,
            algorithm='isolation_forest',
            contamination=0.05,
            random_state=i
        )
        stability_scores.append(np.mean(result['scores']))
    
    print(f"\nModel Stability (5 runs):")
    print(f"Mean score: {np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")
    
    return {
        'train_performance': train_result,
        'test_performance': test_result,
        'stability': stability_scores
    }

# Validate the model
validation_results = validate_anomaly_model(features.head(1000))
```

---

## Building Your Own Custom Detector {#custom-detector}

### Creating a Domain-Specific Detector

Let's build a custom detector for a specific use case:

```python
class CustomFraudDetector:
    """Custom fraud detector with domain-specific rules and ML"""
    
    def __init__(self):
        self.client = PynomalyClient()
        self.business_rules = {
            'max_amount_single': 5000,
            'max_amount_daily': 10000,
            'velocity_threshold': 5,  # transactions per hour
            'high_risk_merchants': ['gambling', 'crypto', 'adult'],
            'suspicious_hours': [0, 1, 2, 3, 4, 5]
        }
        self.customer_profiles = {}
    
    def detect_fraud(self, transaction):
        """Multi-layered fraud detection"""
        
        # Layer 1: Business Rules
        rule_score = self._check_business_rules(transaction)
        
        # Layer 2: ML Anomaly Detection
        ml_features = self._prepare_ml_features(transaction)
        ml_result = self.client.detect_anomalies(
            data=[ml_features],
            algorithm='isolation_forest',
            contamination=0.02
        )
        ml_score = ml_result['scores'][0]
        
        # Layer 3: Behavioral Analysis
        behavioral_score = self._check_behavioral_patterns(transaction)
        
        # Combine scores with weights
        combined_score = (
            0.4 * rule_score +      # Business rules
            0.4 * ml_score +        # ML anomaly detection
            0.2 * behavioral_score  # Behavioral patterns
        )
        
        # Determine fraud decision
        if combined_score > 0.8:
            decision = "BLOCK"
        elif combined_score > 0.6:
            decision = "REVIEW"
        elif combined_score > 0.4:
            decision = "MONITOR"
        else:
            decision = "APPROVE"
        
        return {
            'decision': decision,
            'combined_score': combined_score,
            'rule_score': rule_score,
            'ml_score': ml_score,
            'behavioral_score': behavioral_score,
            'explanation': self._generate_explanation(transaction, combined_score)
        }
    
    def _check_business_rules(self, transaction):
        """Check against business rules"""
        score = 0.0
        
        # High amount
        if transaction['amount'] > self.business_rules['max_amount_single']:
            score += 0.3
        
        # Suspicious time
        if transaction['hour'] in self.business_rules['suspicious_hours']:
            score += 0.2
        
        # High-risk merchant
        if transaction.get('merchant_category') in self.business_rules['high_risk_merchants']:
            score += 0.3
        
        # Weekend transaction
        if transaction.get('is_weekend') and transaction['amount'] > 1000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _prepare_ml_features(self, transaction):
        """Prepare features for ML model"""
        return {
            'amount_log': np.log1p(transaction['amount']),
            'hour': transaction['hour'],
            'day_of_week': transaction.get('day_of_week', 0),
            'time_since_last': transaction.get('time_since_last', 24),
            'is_weekend': transaction.get('is_weekend', 0),
            'merchant_risk': 1 if transaction.get('merchant_category') in self.business_rules['high_risk_merchants'] else 0
        }
    
    def _check_behavioral_patterns(self, transaction):
        """Check against customer behavioral patterns"""
        customer_id = transaction.get('customer_id')
        
        if customer_id not in self.customer_profiles:
            # New customer - higher risk
            return 0.3
        
        profile = self.customer_profiles[customer_id]
        score = 0.0
        
        # Amount deviation from customer average
        avg_amount = profile.get('avg_amount', transaction['amount'])
        if transaction['amount'] > avg_amount * 3:
            score += 0.4
        
        # Location deviation (simplified)
        if transaction.get('location') != profile.get('usual_location'):
            score += 0.3
        
        # Time pattern deviation
        usual_hours = profile.get('usual_hours', [])
        if usual_hours and transaction['hour'] not in usual_hours:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_explanation(self, transaction, score):
        """Generate human-readable explanation"""
        reasons = []
        
        if transaction['amount'] > self.business_rules['max_amount_single']:
            reasons.append(f"High amount (${transaction['amount']:,})")
        
        if transaction['hour'] in self.business_rules['suspicious_hours']:
            reasons.append(f"Unusual time ({transaction['hour']}:00)")
        
        if transaction.get('merchant_category') in self.business_rules['high_risk_merchants']:
            reasons.append(f"High-risk merchant category")
        
        if score > 0.8:
            return f"High fraud risk: {', '.join(reasons)}"
        elif score > 0.6:
            return f"Moderate fraud risk: {', '.join(reasons)}"
        else:
            return "Low fraud risk detected"
    
    def update_customer_profile(self, customer_id, transaction):
        """Update customer behavioral profile"""
        if customer_id not in self.customer_profiles:
            self.customer_profiles[customer_id] = {
                'transaction_count': 0,
                'total_amount': 0,
                'usual_hours': [],
                'usual_location': None
            }
        
        profile = self.customer_profiles[customer_id]
        profile['transaction_count'] += 1
        profile['total_amount'] += transaction['amount']
        profile['avg_amount'] = profile['total_amount'] / profile['transaction_count']
        
        # Update usual hours (simplified)
        if transaction['hour'] not in profile['usual_hours']:
            profile['usual_hours'].append(transaction['hour'])

# Test the custom detector
custom_detector = CustomFraudDetector()

test_transactions = [
    {
        'customer_id': 'CUST_001',
        'amount': 45.67,
        'hour': 14,
        'day_of_week': 2,
        'merchant_category': 'grocery',
        'is_weekend': 0,
        'time_since_last': 2.5,
        'location': 'home_city'
    },
    {
        'customer_id': 'CUST_001',
        'amount': 2500.00,  # High amount
        'hour': 2,          # Suspicious time
        'day_of_week': 6,   # Weekend
        'merchant_category': 'gambling',  # High risk
        'is_weekend': 1,
        'time_since_last': 0.1,  # Very quick
        'location': 'foreign_city'
    }
]

print("Testing Custom Fraud Detector:")
print("=" * 50)

for i, transaction in enumerate(test_transactions):
    print(f"\nTransaction {i+1}:")
    result = custom_detector.detect_fraud(transaction)
    
    print(f"Decision: {result['decision']}")
    print(f"Combined Score: {result['combined_score']:.3f}")
    print(f"  - Rule Score: {result['rule_score']:.3f}")
    print(f"  - ML Score: {result['ml_score']:.3f}")
    print(f"  - Behavioral Score: {result['behavioral_score']:.3f}")
    print(f"Explanation: {result['explanation']}")
    
    # Update customer profile
    custom_detector.update_customer_profile(transaction['customer_id'], transaction)
```

**🎉 Congratulations!** You've successfully:
- Completed three real-world anomaly detection scenarios
- Learned advanced techniques like ensemble methods
- Built a custom, multi-layered fraud detection system
- Gained hands-on experience with production patterns

## What's Next?

### Immediate Actions
1. **Try with your own data** - Apply these techniques to your specific datasets
2. **Experiment with parameters** - Try different algorithms and contamination rates
3. **Build monitoring** - Set up alerts and dashboards for your use case

### Advanced Learning
1. **[Production Deployment Guide](../deployment/production-deployment.md)** - Deploy your models
2. **[Algorithm Deep Dive](../reference/algorithm-comparison.md)** - Understand algorithms better
3. **[Custom Algorithm Development](advanced-customization.md)** - Build your own detectors

### Get Help
- **Questions?** [GitHub Discussions](https://github.com/pynomaly/pynomaly/discussions)
- **Issues?** [GitHub Issues](https://github.com/pynomaly/pynomaly/issues)
- **Enterprise Support?** [Contact Sales](mailto:sales@pynomaly.com)

### Share Your Success
We'd love to hear about your anomaly detection projects! Share your experience:
- **Blog posts** about your use cases
- **Case studies** on GitHub Discussions
- **Contributions** to improve Pynomaly

---

**🚀 You're now ready to tackle real-world anomaly detection challenges!** Start applying these techniques to your own data and don't hesitate to reach out if you need help.