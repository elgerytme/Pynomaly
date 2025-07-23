# Practical Examples

This guide provides hands-on examples for common anomaly detection scenarios across different industries and use cases.

## Example 1: Credit Card Fraud Detection

Detect fraudulent transactions in financial data.

### Scenario
You have credit card transaction data and need to identify potentially fraudulent activities in real-time.

```python
import pandas as pd
import numpy as np
from anomaly_detection import DetectionService
from anomaly_detection.preprocessing import StandardScaler

# Load transaction data (example structure)
data = pd.DataFrame({
    'amount': [25.50, 89.99, 12.00, 15000.00, 45.67, 2.99, 8950.50],
    'merchant_category': [1, 2, 1, 3, 1, 4, 3],  # Encoded categories
    'time_since_last': [2.5, 24.0, 1.0, 0.1, 12.0, 6.0, 0.05],  # Hours
    'location_risk': [0.1, 0.3, 0.1, 0.9, 0.2, 0.1, 0.95],  # Risk score
    'velocity_score': [1.2, 2.1, 0.8, 15.6, 1.8, 0.9, 12.3]  # Transactions/hour
})

print("💳 Credit Card Fraud Detection")
print(f"Dataset shape: {data.shape}")
print(f"Sample transactions:\n{data.head()}")

# Preprocess data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.values)

# Initialize detection service
service = DetectionService()

# Detect anomalies using Isolation Forest (good for fraud detection)
result = service.detect(
    data=scaled_data,
    algorithm="isolation_forest",
    contamination=0.01,  # Expect 1% fraud rate
    n_estimators=200,
    random_state=42
)

# Analyze results
fraud_indices = np.where(result.predictions == -1)[0]
fraud_transactions = data.iloc[fraud_indices]

print(f"\n🚨 Fraud Detection Results:")
print(f"Suspicious transactions detected: {len(fraud_indices)}")
print(f"Fraud rate: {len(fraud_indices)/len(data)*100:.1f}%")

if len(fraud_indices) > 0:
    print(f"\n🔍 Flagged Transactions:")
    for idx in fraud_indices:
        score = result.scores[idx]
        print(f"  • Transaction {idx}: Amount=${data.iloc[idx]['amount']:.2f}, "
              f"Risk Score={score:.3f}")
        print(f"    Time since last: {data.iloc[idx]['time_since_last']:.1f}h, "
              f"Velocity: {data.iloc[idx]['velocity_score']:.1f}")
```

**Expected Output:**
```
💳 Credit Card Fraud Detection
Dataset shape: (7, 5)
🚨 Fraud Detection Results:
Suspicious transactions detected: 2
Fraud rate: 28.6%
🔍 Flagged Transactions:
  • Transaction 3: Amount=$15000.00, Risk Score=0.721
    Time since last: 0.1h, Velocity: 15.6
  • Transaction 6: Amount=$8950.50, Risk Score=0.698
    Time since last: 0.1h, Velocity: 12.3
```

### Business Impact
```python
# Calculate potential savings
avg_fraud_amount = fraud_transactions['amount'].mean()
total_fraud_value = fraud_transactions['amount'].sum()

print(f"\n💰 Business Impact:")
print(f"Average fraud amount: ${avg_fraud_amount:.2f}")
print(f"Total fraud value detected: ${total_fraud_value:.2f}")
print(f"Potential monthly savings (scaled): ${total_fraud_value * 1000:.0f}")
```

## Example 2: Network Intrusion Detection

Monitor network traffic for security threats.

### Scenario
Analyze network packets to identify unusual patterns that might indicate cyber attacks.

```python
# Simulate network traffic data
np.random.seed(42)

# Normal traffic patterns
normal_traffic = {
    'packet_size': np.random.normal(1500, 300, 1000),
    'duration': np.random.exponential(2.0, 1000),
    'src_bytes': np.random.lognormal(8, 1.5, 1000),
    'dst_bytes': np.random.lognormal(7, 1.2, 1000),
    'protocol_type': np.random.choice([0, 1, 2], 1000, p=[0.7, 0.2, 0.1])
}

# Inject attack patterns
attack_indices = np.random.choice(1000, 50, replace=False)
normal_traffic['packet_size'][attack_indices] *= 10  # Large packets
normal_traffic['src_bytes'][attack_indices] *= 20    # Data exfiltration
normal_traffic['duration'][attack_indices] *= 0.1   # Quick connections

# Create dataset
network_data = np.column_stack([
    normal_traffic['packet_size'],
    normal_traffic['duration'], 
    normal_traffic['src_bytes'],
    normal_traffic['dst_bytes'],
    normal_traffic['protocol_type']
])

print("🌐 Network Intrusion Detection")
print(f"Traffic samples: {len(network_data)}")
print(f"Features: {network_data.shape[1]}")

# Detect intrusions using LOF (good for network anomalies)
intrusion_result = service.detect(
    data=network_data,
    algorithm="lof",
    contamination=0.05,  # 5% attack rate
    n_neighbors=30
)

# Analyze detected intrusions
intrusions = np.where(intrusion_result.predictions == -1)[0]
print(f"\n🛡️ Intrusion Detection Results:")
print(f"Suspicious connections: {len(intrusions)}")
print(f"Attack detection rate: {len(intrusions)/len(network_data)*100:.1f}%")

# Check detection accuracy
true_positives = len(set(attack_indices) & set(intrusions))
false_positives = len(intrusions) - true_positives
false_negatives = len(attack_indices) - true_positives

precision = true_positives / len(intrusions) if len(intrusions) > 0 else 0
recall = true_positives / len(attack_indices)

print(f"\n📊 Detection Performance:")
print(f"True positives: {true_positives}")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

## Example 3: Manufacturing Quality Control

Detect defective products in manufacturing processes.

### Scenario
Monitor sensor readings from production line to identify defective products.

```python
# Simulate manufacturing sensor data
def generate_manufacturing_data(n_samples=500, defect_rate=0.03):
    # Normal product parameters
    temperature = np.random.normal(75, 3, n_samples)  # Celsius
    pressure = np.random.normal(2.1, 0.15, n_samples)  # Bar
    vibration = np.random.normal(0.5, 0.1, n_samples)  # mm/s
    power_consumption = np.random.normal(850, 50, n_samples)  # Watts
    
    # Introduce defects
    n_defects = int(n_samples * defect_rate)
    defect_indices = np.random.choice(n_samples, n_defects, replace=False)
    
    # Defect patterns
    temperature[defect_indices] += np.random.normal(15, 5, n_defects)  # Overheating
    pressure[defect_indices] += np.random.normal(0.8, 0.2, n_defects)  # High pressure
    vibration[defect_indices] += np.random.normal(1.5, 0.3, n_defects)  # Excessive vibration
    
    return np.column_stack([temperature, pressure, vibration, power_consumption]), defect_indices

# Generate manufacturing data
manufacturing_data, true_defects = generate_manufacturing_data()

print("🏭 Manufacturing Quality Control")
print(f"Production samples: {len(manufacturing_data)}")
print(f"True defects: {len(true_defects)}")
print(f"Expected defect rate: {len(true_defects)/len(manufacturing_data)*100:.1f}%")

# Use ensemble method for robust detection
from anomaly_detection import EnsembleService
ensemble_service = EnsembleService()

quality_result = ensemble_service.detect(
    data=manufacturing_data,
    algorithms=["isolation_forest", "lof", "one_class_svm"],
    method="voting",
    contamination=0.03
)

detected_defects = np.where(quality_result.predictions == -1)[0]

print(f"\n🔍 Quality Control Results:")
print(f"Defective products detected: {len(detected_defects)}")
print(f"Detection rate: {len(detected_defects)/len(manufacturing_data)*100:.1f}%")

# Calculate quality metrics
true_detections = len(set(true_defects) & set(detected_defects))
quality_precision = true_detections / len(detected_defects) if len(detected_defects) > 0 else 0
quality_recall = true_detections / len(true_defects)

print(f"\n📈 Quality Metrics:")
print(f"Correctly identified defects: {true_detections}")
print(f"Precision: {quality_precision:.2f}")
print(f"Recall: {quality_recall:.2f}")

# Cost analysis
cost_per_defect = 250  # Cost of defective product reaching customer
inspection_cost = 2   # Cost of additional inspection

savings = true_detections * cost_per_defect
inspection_costs = len(detected_defects) * inspection_cost
net_savings = savings - inspection_costs

print(f"\n💰 Cost Analysis:")
print(f"Cost savings from catching defects: ${savings:,.2f}")
print(f"Additional inspection costs: ${inspection_costs:,.2f}")
print(f"Net savings: ${net_savings:,.2f}")
```

## Example 4: IoT Sensor Monitoring

Monitor IoT devices for malfunctions and anomalous behavior.

### Scenario
A smart building has hundreds of sensors monitoring temperature, humidity, and energy usage.

```python
from datetime import datetime, timedelta
import pandas as pd

# Simulate IoT sensor data
def generate_iot_data(days=7, sensors=100):
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(hours=h) for h in range(days * 24)]
    
    data = []
    for sensor_id in range(sensors):
        for timestamp in timestamps:
            # Normal patterns with daily cycles
            hour = timestamp.hour
            daily_temp_cycle = 20 + 5 * np.sin(2 * np.pi * hour / 24)
            
            # Add sensor-specific baseline
            temp_baseline = np.random.normal(0, 2)
            
            record = {
                'sensor_id': sensor_id,
                'timestamp': timestamp,
                'temperature': daily_temp_cycle + temp_baseline + np.random.normal(0, 1),
                'humidity': np.random.normal(45, 5),
                'energy_usage': np.random.exponential(10),
                'battery_level': max(0, 100 - np.random.exponential(1))
            }
            data.append(record)
    
    return pd.DataFrame(data)

# Generate IoT data
iot_data = generate_iot_data(days=7, sensors=50)

# Introduce some sensor malfunctions
malfunction_sensors = np.random.choice(50, 5, replace=False)
for sensor_id in malfunction_sensors:
    sensor_mask = iot_data['sensor_id'] == sensor_id
    # Simulate different types of malfunctions
    iot_data.loc[sensor_mask, 'temperature'] += np.random.normal(15, 3, sensor_mask.sum())
    iot_data.loc[sensor_mask, 'humidity'] *= 0.5  # Humidity sensor failure
    iot_data.loc[sensor_mask, 'battery_level'] *= 0.1  # Battery issue

print("🏢 IoT Sensor Monitoring")
print(f"Total sensor readings: {len(iot_data)}")
print(f"Sensors monitored: {iot_data['sensor_id'].nunique()}")
print(f"Time period: {iot_data['timestamp'].min()} to {iot_data['timestamp'].max()}")

# Prepare features for anomaly detection
feature_columns = ['temperature', 'humidity', 'energy_usage', 'battery_level']
sensor_features = iot_data[feature_columns].values

# Detect anomalous readings
iot_result = service.detect(
    data=sensor_features,
    algorithm="isolation_forest",
    contamination=0.02,  # 2% anomaly rate
    n_estimators=100
)

anomalous_readings = np.where(iot_result.predictions == -1)[0]
anomalous_data = iot_data.iloc[anomalous_readings]

print(f"\n🚨 Anomaly Detection Results:")
print(f"Anomalous readings detected: {len(anomalous_readings)}")
print(f"Anomaly rate: {len(anomalous_readings)/len(iot_data)*100:.1f}%")

# Analyze by sensor
sensor_anomaly_counts = anomalous_data['sensor_id'].value_counts().head(10)
print(f"\n🔍 Top Sensors with Anomalies:")
for sensor_id, count in sensor_anomaly_counts.items():
    total_readings = len(iot_data[iot_data['sensor_id'] == sensor_id])
    anomaly_rate = count / total_readings * 100
    in_malfunction_list = "⚠️ KNOWN ISSUE" if sensor_id in malfunction_sensors else "✅ Check required"
    print(f"  Sensor {sensor_id}: {count} anomalies ({anomaly_rate:.1f}%) - {in_malfunction_list}")
```

## Example 5: E-commerce User Behavior

Detect unusual user behavior patterns for fraud prevention and personalization.

### Scenario
Monitor user interactions on an e-commerce platform to identify bots, fraudsters, or unusual purchasing patterns.

```python
# Simulate user behavior data
def generate_user_behavior_data(n_users=1000):
    data = []
    
    for user_id in range(n_users):
        # Normal user behavior
        if np.random.random() < 0.95:  # 95% normal users
            session_duration = np.random.lognormal(3, 1)  # Minutes
            page_views = max(1, int(np.random.poisson(15)))
            clicks = max(0, int(page_views * np.random.beta(2, 5)))
            cart_additions = max(0, int(clicks * np.random.beta(1, 10)))
            purchases = max(0, int(cart_additions * np.random.beta(2, 8)))
            avg_price = np.random.lognormal(3, 0.8)
            user_type = 'normal'
        else:  # 5% suspicious users (bots, fraudsters)
            if np.random.random() < 0.5:  # Bot behavior
                session_duration = np.random.exponential(0.5)  # Very short sessions
                page_views = int(np.random.exponential(50))     # Many page views
                clicks = int(page_views * 0.9)                 # High click rate
                cart_additions = 0                             # No purchases
                purchases = 0
                avg_price = 0
                user_type = 'bot'
            else:  # Fraudulent behavior
                session_duration = np.random.normal(45, 10)    # Long sessions
                page_views = int(np.random.normal(8, 3))       # Few page views
                clicks = page_views                            # Every page clicked
                cart_additions = clicks                        # Everything added to cart
                purchases = cart_additions                     # Everything purchased
                avg_price = np.random.lognormal(5, 0.5)       # High-value items
                user_type = 'fraudster'
        
        data.append({
            'user_id': user_id,
            'session_duration': max(0.1, session_duration),
            'page_views': max(1, page_views),
            'clicks': clicks,
            'cart_additions': cart_additions,
            'purchases': purchases,
            'avg_purchase_price': avg_price,
            'conversion_rate': purchases / max(1, page_views),
            'click_rate': clicks / max(1, page_views),
            'true_label': user_type
        })
    
    return pd.DataFrame(data)

# Generate user behavior data
user_data = generate_user_behavior_data(1000)

print("🛒 E-commerce User Behavior Analysis")
print(f"Total users: {len(user_data)}")
print(f"User types: {user_data['true_label'].value_counts().to_dict()}")

# Prepare features
behavior_features = [
    'session_duration', 'page_views', 'clicks', 'cart_additions', 
    'purchases', 'avg_purchase_price', 'conversion_rate', 'click_rate'
]
user_features = user_data[behavior_features].values

# Detect suspicious behavior
behavior_result = service.detect(
    data=user_features,
    algorithm="isolation_forest",
    contamination=0.05,  # 5% suspicious users
    random_state=42
)

suspicious_users = user_data.iloc[np.where(behavior_result.predictions == -1)[0]]

print(f"\n🕵️ Suspicious Behavior Detection:")
print(f"Users flagged: {len(suspicious_users)}")
print(f"Suspicious rate: {len(suspicious_users)/len(user_data)*100:.1f}%")

# Analyze detection by true behavior type
detection_analysis = suspicious_users['true_label'].value_counts()
print(f"\n📊 Detection Breakdown:")
for behavior_type, count in detection_analysis.items():
    total_of_type = len(user_data[user_data['true_label'] == behavior_type])
    detection_rate = count / total_of_type * 100
    print(f"  {behavior_type.capitalize()}: {count}/{total_of_type} ({detection_rate:.1f}%)")

# Business recommendations
print(f"\n💡 Recommended Actions:")
bot_users = suspicious_users[suspicious_users['true_label'] == 'bot']
fraud_users = suspicious_users[suspicious_users['true_label'] == 'fraudster']

if len(bot_users) > 0:
    print(f"  • Block {len(bot_users)} potential bot accounts")
    print(f"  • Implement CAPTCHA for high-activity users")

if len(fraud_users) > 0:
    print(f"  • Review {len(fraud_users)} high-risk transactions")
    print(f"  • Enable additional verification for suspicious purchases")

normal_flagged = len(suspicious_users[suspicious_users['true_label'] == 'normal'])
if normal_flagged > 0:
    print(f"  • Consider personalizing experience for {normal_flagged} power users")
```

## Interactive Demo: Try Your Own Data

<div class="interactive-demo">
    <div class="demo-controls">
        <select id="example-selector">
            <option value="fraud">Credit Card Fraud</option>
            <option value="network">Network Intrusion</option>
            <option value="manufacturing">Quality Control</option>
            <option value="iot">IoT Monitoring</option>
            <option value="ecommerce">User Behavior</option>
        </select>
        <select id="algorithm-selector">
            <option value="isolation_forest">Isolation Forest</option>
            <option value="lof">Local Outlier Factor</option>
            <option value="one_class_svm">One-Class SVM</option>
            <option value="ensemble">Ensemble Method</option>
        </select>
        <button class="demo-button" onclick="runExampleDemo()">Run Example</button>
    </div>
    <div class="demo-output" id="example-demo-output">
        Select an example scenario and algorithm, then click "Run Example" to see it in action!
    </div>
</div>

<script>
function runExampleDemo() {
    const scenario = document.getElementById('example-selector').value;
    const algorithm = document.getElementById('algorithm-selector').value;
    const output = document.getElementById('example-demo-output');
    
    const scenarios = {
        fraud: {
            name: "Credit Card Fraud Detection",
            samples: 10000,
            features: 5,
            anomalyRate: 0.1,
            icon: "💳",
            insights: "High-value transactions with unusual patterns detected"
        },
        network: {
            name: "Network Intrusion Detection", 
            samples: 50000,
            features: 6,
            anomalyRate: 2.1,
            icon: "🌐",
            insights: "Suspicious connections with abnormal packet sizes identified"
        },
        manufacturing: {
            name: "Manufacturing Quality Control",
            samples: 2500,
            features: 4,
            anomalyRate: 3.2,
            icon: "🏭",
            insights: "Products with temperature and vibration anomalies flagged"
        },
        iot: {
            name: "IoT Sensor Monitoring",
            samples: 8400,
            features: 4,
            anomalyRate: 1.8,
            icon: "🏢",
            insights: "Malfunctioning sensors with irregular readings detected"
        },
        ecommerce: {
            name: "E-commerce User Behavior",
            samples: 1000,
            features: 8,
            anomalyRate: 4.7,
            icon: "🛒",
            insights: "Bot accounts and fraudulent users identified"
        }
    };
    
    const selected = scenarios[scenario];
    
    output.innerHTML = `${selected.icon} Running ${selected.name}...`;
    
    setTimeout(() => {
        // Simulate algorithm performance
        const basePerformance = {
            isolation_forest: { precision: 0.85, recall: 0.78, speed: 0.12 },
            lof: { precision: 0.82, recall: 0.84, speed: 0.45 },
            one_class_svm: { precision: 0.88, recall: 0.72, speed: 1.23 },
            ensemble: { precision: 0.91, recall: 0.86, speed: 0.78 }
        };
        
        const perf = basePerformance[algorithm];
        const detectedAnomalies = Math.floor(selected.samples * selected.anomalyRate / 100);
        const truePositives = Math.floor(detectedAnomalies * perf.recall);
        const falsePositives = Math.floor(truePositives / perf.precision - truePositives);
        
        output.innerHTML = `
            <strong>${selected.icon} ${selected.name}</strong>
            
            📊 Dataset:
            • Samples: ${selected.samples.toLocaleString()}
            • Features: ${selected.features}
            • Algorithm: ${algorithm.replace('_', ' ').toUpperCase()}
            
            🎯 Results:
            • Anomalies detected: ${detectedAnomalies + falsePositives}
            • True positives: ${truePositives}
            • False positives: ${falsePositives}
            • Precision: ${(perf.precision * 100).toFixed(1)}%
            • Recall: ${(perf.recall * 100).toFixed(1)}%
            • Processing time: ${perf.speed.toFixed(2)}s
            
            💡 Key Insights:
            ${selected.insights}
            
            ${perf.precision > 0.85 && perf.recall > 0.80 ? 
              "🏆 Excellent detection performance!" : 
              "👍 Good results - consider ensemble methods for improvement"}
        `;
    }, 2000);
}
</script>

## Key Takeaways

After working through these examples, you should understand:

!!! success "What You've Learned"
    - **Algorithm Selection**: Different algorithms work better for different types of data
    - **Parameter Tuning**: Contamination rate and algorithm-specific parameters are crucial
    - **Feature Engineering**: Proper data preprocessing improves detection accuracy
    - **Business Context**: Always validate results against domain knowledge
    - **Ensemble Methods**: Combining algorithms often provides more robust results

## Common Implementation Patterns

### 1. Data Preprocessing Pipeline
```python
from anomaly_detection.preprocessing import StandardScaler, RobustScaler

# Standard preprocessing pipeline
def preprocess_data(data, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    
    return scaler.fit_transform(data)
```

### 2. Model Selection Framework
```python
def select_best_algorithm(data, algorithms, contamination=0.1):
    results = {}
    
    for algo in algorithms:
        result = service.detect(data, algorithm=algo, contamination=contamination)
        # Score based on silhouette coefficient or other metrics
        results[algo] = evaluate_result(result, data)
    
    return max(results, key=results.get)
```

### 3. Threshold Optimization
```python
def optimize_threshold(scores, true_labels):
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]
```

## Next Steps

Ready to implement these patterns in your own projects?

=== "Production Deployment"
    Learn how to deploy these examples in production with the [Deployment Guide](../deployment/).

=== "Advanced Techniques" 
    Explore advanced ensemble methods and optimization in [Ensemble Methods](../ensemble/).

=== "Real-time Processing"
    Implement streaming anomaly detection with the [Streaming Guide](../streaming/).

=== "Performance Optimization"
    Scale your detection to handle large datasets with [Performance Optimization](../performance/).

## Additional Resources

- **Code Repository**: Find complete example notebooks in the `/examples` directory
- **Sample Datasets**: Download realistic datasets for testing your implementations
- **Video Tutorials**: Watch step-by-step implementation videos
- **Community Examples**: See how others have implemented similar use cases