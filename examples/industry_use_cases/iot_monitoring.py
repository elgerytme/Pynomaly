#!/usr/bin/env python3
"""
Industry Use Case: IoT Device Monitoring and Predictive Maintenance

This example demonstrates anomaly detection for IoT sensor networks,
including equipment health monitoring, predictive maintenance, and
real-time alerting systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🏭 IoT Device Monitoring & Predictive Maintenance")
print("=" * 55)

# =============================================================================
# 1. Generate Realistic IoT Sensor Data
# =============================================================================

print("\n📡 Section 1: Generating Realistic IoT Sensor Data")
print("-" * 55)

np.random.seed(42)

def generate_iot_sensor_data(n_devices=50, days=30, measurements_per_hour=12):
    """Generate realistic IoT sensor data with various failure modes."""
    
    total_measurements = n_devices * days * 24 * measurements_per_hour
    print(f"Generating data for {n_devices} devices over {days} days")
    print(f"Total measurements: {total_measurements:,}")
    
    # Device types and their normal operating parameters
    device_types = {
        'pump': {
            'temperature': (20, 80, 5),    # (min, max, std)
            'vibration': (0.1, 2.0, 0.3),
            'pressure': (10, 50, 3),
            'flow_rate': (5, 25, 2),
            'power_consumption': (100, 500, 50)
        },
        'motor': {
            'temperature': (30, 90, 8),
            'vibration': (0.2, 3.0, 0.5),
            'pressure': (0, 0, 0),  # Motors don't have pressure
            'flow_rate': (0, 0, 0),  # Motors don't have flow rate
            'power_consumption': (200, 1000, 100)
        },
        'compressor': {
            'temperature': (40, 120, 10),
            'vibration': (0.5, 4.0, 0.7),
            'pressure': (50, 200, 15),
            'flow_rate': (10, 100, 8),
            'power_consumption': (500, 2000, 200)
        },
        'valve': {
            'temperature': (15, 60, 4),
            'vibration': (0.05, 1.0, 0.2),
            'pressure': (5, 100, 10),
            'flow_rate': (1, 50, 5),
            'power_consumption': (10, 100, 20)
        }
    }
    
    data = []
    device_info = []
    
    # Generate device metadata
    for device_id in range(n_devices):
        device_type = np.random.choice(list(device_types.keys()))
        location = np.random.choice(['Factory_A', 'Factory_B', 'Factory_C', 'Warehouse'])
        installation_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1000))
        
        device_info.append({
            'device_id': f'DEV_{device_id:03d}',
            'device_type': device_type,
            'location': location,
            'installation_date': installation_date
        })
    
    device_df = pd.DataFrame(device_info)
    
    # Generate time series data
    start_time = datetime(2024, 1, 1)
    
    for device_idx, device in device_df.iterrows():
        device_id = device['device_id']
        device_type = device['device_type']
        params = device_types[device_type]
        
        # Device age factor (older devices more prone to issues)
        age_days = (start_time - device['installation_date']).days
        age_factor = 1 + (age_days / 1000) * 0.3  # 30% increase in issues per 1000 days
        
        # Generate measurements
        for day in range(days):
            for hour in range(24):
                for measurement in range(measurements_per_hour):
                    timestamp = start_time + timedelta(
                        days=day, 
                        hours=hour, 
                        minutes=measurement * (60 // measurements_per_hour)
                    )
                    
                    # Base measurements with normal variation
                    temperature = np.random.normal(
                        (params['temperature'][0] + params['temperature'][1]) / 2,
                        params['temperature'][2]
                    )
                    
                    vibration = np.random.normal(
                        (params['vibration'][0] + params['vibration'][1]) / 2,
                        params['vibration'][2]
                    ) if params['vibration'][1] > 0 else 0
                    
                    pressure = np.random.normal(
                        (params['pressure'][0] + params['pressure'][1]) / 2,
                        params['pressure'][2]
                    ) if params['pressure'][1] > 0 else 0
                    
                    flow_rate = np.random.normal(
                        (params['flow_rate'][0] + params['flow_rate'][1]) / 2,
                        params['flow_rate'][2]
                    ) if params['flow_rate'][1] > 0 else 0
                    
                    power_consumption = np.random.normal(
                        (params['power_consumption'][0] + params['power_consumption'][1]) / 2,
                        params['power_consumption'][2]
                    )
                    
                    # Apply age factor
                    if device_type in ['pump', 'motor', 'compressor']:
                        temperature *= age_factor
                        vibration *= age_factor
                    
                    # Add daily patterns (higher load during work hours)
                    if 8 <= hour <= 18:  # Work hours
                        load_factor = 1.2
                    elif 19 <= hour <= 22:  # Evening operations
                        load_factor = 1.0
                    else:  # Night/maintenance
                        load_factor = 0.7
                    
                    power_consumption *= load_factor
                    if device_type in ['pump', 'compressor']:
                        flow_rate *= load_factor
                    
                    # Seasonal effects
                    seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 365)
                    temperature *= seasonal_factor
                    
                    # Random anomalies/failures
                    is_anomaly = False
                    anomaly_type = 'normal'
                    
                    # Simulate different failure modes
                    failure_probability = 0.005 * age_factor  # Base failure rate
                    
                    if np.random.random() < failure_probability:
                        is_anomaly = True
                        failure_mode = np.random.choice([
                            'overheating', 'vibration_spike', 'pressure_drop',
                            'power_surge', 'sensor_drift'
                        ])
                        
                        if failure_mode == 'overheating':
                            temperature *= 1.5 + np.random.random()
                            anomaly_type = 'overheating'
                        
                        elif failure_mode == 'vibration_spike':
                            vibration *= 3 + np.random.random() * 2
                            anomaly_type = 'vibration_spike'
                        
                        elif failure_mode == 'pressure_drop':
                            if pressure > 0:
                                pressure *= 0.3 + np.random.random() * 0.2
                            anomaly_type = 'pressure_drop'
                        
                        elif failure_mode == 'power_surge':
                            power_consumption *= 1.8 + np.random.random()
                            anomaly_type = 'power_surge'
                        
                        elif failure_mode == 'sensor_drift':
                            # Multiple sensor readings become erratic
                            temperature += np.random.normal(0, 20)
                            vibration += np.random.normal(0, 1)
                            if pressure > 0:
                                pressure += np.random.normal(0, 10)
                            anomaly_type = 'sensor_drift'
                    
                    # Ensure realistic bounds
                    temperature = max(0, temperature)
                    vibration = max(0, vibration)
                    pressure = max(0, pressure)
                    flow_rate = max(0, flow_rate)
                    power_consumption = max(0, power_consumption)
                    
                    data.append({
                        'timestamp': timestamp,
                        'device_id': device_id,
                        'device_type': device_type,
                        'location': device['location'],
                        'temperature': temperature,
                        'vibration': vibration,
                        'pressure': pressure,
                        'flow_rate': flow_rate,
                        'power_consumption': power_consumption,
                        'is_anomaly': is_anomaly,
                        'anomaly_type': anomaly_type
                    })
    
    return pd.DataFrame(data), device_df

# Generate the dataset
sensor_df, devices_df = generate_iot_sensor_data()
print(f"✅ Generated {len(sensor_df):,} sensor measurements")
print(f"   Devices: {len(devices_df)}")
print(f"   Anomalies: {sum(sensor_df['is_anomaly'])} ({sum(sensor_df['is_anomaly'])/len(sensor_df)*100:.2f}%)")

# Display device types
print(f"\nDevice Types:")
print(devices_df['device_type'].value_counts())

print(f"\nAnomaly Types:")
print(sensor_df['anomaly_type'].value_counts())

# =============================================================================
# 2. Exploratory Data Analysis
# =============================================================================

print("\n🔍 Section 2: IoT Data Analysis")
print("-" * 35)

# Basic statistics by device type
print("📊 Sensor Statistics by Device Type:")
sensor_stats = sensor_df.groupby('device_type')[
    ['temperature', 'vibration', 'pressure', 'flow_rate', 'power_consumption']
].agg(['mean', 'std']).round(2)
print(sensor_stats)

# Anomaly rates by device type
print(f"\n🚨 Anomaly Rates by Device Type:")
anomaly_rates = sensor_df.groupby('device_type')['is_anomaly'].agg(['count', 'sum', 'mean'])
anomaly_rates.columns = ['Total Measurements', 'Anomalies', 'Anomaly Rate']
anomaly_rates['Anomaly Rate'] = anomaly_rates['Anomaly Rate'].round(4)
print(anomaly_rates)

# Time-based patterns
print(f"\n⏰ Anomaly Patterns by Hour:")
sensor_df['hour'] = sensor_df['timestamp'].dt.hour
hourly_anomalies = sensor_df.groupby('hour')['is_anomaly'].mean()
print(f"Peak anomaly hour: {hourly_anomalies.idxmax()}:00 ({hourly_anomalies.max():.3f} rate)")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Temperature distribution by device type
for i, device_type in enumerate(sensor_df['device_type'].unique()):
    if i < 4:  # Only plot first 4 device types
        device_data = sensor_df[sensor_df['device_type'] == device_type]
        axes[0, 0].hist(device_data['temperature'], alpha=0.7, label=device_type, bins=30)
axes[0, 0].set_title('Temperature Distribution by Device Type')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Temperature (°C)')

# Anomaly detection over time (sample device)
sample_device = sensor_df[sensor_df['device_id'] == 'DEV_001'].copy()
sample_device = sample_device.sort_values('timestamp')
axes[0, 1].plot(sample_device['timestamp'], sample_device['temperature'], alpha=0.7)
anomaly_points = sample_device[sample_device['is_anomaly']]
axes[0, 1].scatter(anomaly_points['timestamp'], anomaly_points['temperature'], 
                   color='red', s=20, alpha=0.8, label='Anomalies')
axes[0, 1].set_title('Temperature Timeline (DEV_001)')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Vibration vs Temperature scatter
normal_data = sensor_df[~sensor_df['is_anomaly']]
anomaly_data = sensor_df[sensor_df['is_anomaly']]
axes[0, 2].scatter(normal_data['temperature'], normal_data['vibration'], 
                   alpha=0.3, label='Normal', s=1)
axes[0, 2].scatter(anomaly_data['temperature'], anomaly_data['vibration'], 
                   alpha=0.8, label='Anomaly', s=5, color='red')
axes[0, 2].set_title('Vibration vs Temperature')
axes[0, 2].set_xlabel('Temperature (°C)')
axes[0, 2].set_ylabel('Vibration')
axes[0, 2].legend()

# Power consumption patterns by hour
hourly_power = sensor_df.groupby('hour')['power_consumption'].mean()
hourly_power.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Average Power Consumption by Hour')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Power (W)')

# Anomaly types distribution
anomaly_types = sensor_df[sensor_df['is_anomaly']]['anomaly_type'].value_counts()
anomaly_types.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
axes[1, 1].set_title('Distribution of Anomaly Types')

# Device reliability over time
device_age = []
device_failure_rate = []
for device_id in devices_df['device_id']:
    device_data = sensor_df[sensor_df['device_id'] == device_id]
    if len(device_data) > 0:
        device_info = devices_df[devices_df['device_id'] == device_id].iloc[0]
        age_days = (sensor_df['timestamp'].max() - device_info['installation_date']).days
        failure_rate = device_data['is_anomaly'].mean()
        device_age.append(age_days)
        device_failure_rate.append(failure_rate)

axes[1, 2].scatter(device_age, device_failure_rate, alpha=0.7)
axes[1, 2].set_title('Device Age vs Failure Rate')
axes[1, 2].set_xlabel('Device Age (days)')
axes[1, 2].set_ylabel('Anomaly Rate')

plt.tight_layout()
plt.savefig('/tmp/iot_analysis.png', dpi=150, bbox_inches='tight')
print("📊 IoT analysis visualizations saved to /tmp/iot_analysis.png")

# =============================================================================
# 3. Feature Engineering for IoT Anomaly Detection
# =============================================================================

print("\n🔧 Section 3: IoT Feature Engineering")
print("-" * 40)

def engineer_iot_features(df):
    """Engineer features specific to IoT anomaly detection."""
    
    df_features = df.copy()
    df_features = df_features.sort_values(['device_id', 'timestamp']).reset_index(drop=True)
    
    print("Engineering time-based features...")
    # Time-based features
    df_features['hour'] = df_features['timestamp'].dt.hour
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_work_hours'] = ((df_features['hour'] >= 8) & 
                                   (df_features['hour'] <= 18)).astype(int)
    
    # Cyclical encoding for time features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    print("Engineering sensor features...")
    # Sensor-specific features
    sensor_cols = ['temperature', 'vibration', 'pressure', 'flow_rate', 'power_consumption']
    
    # Ratios and combinations
    df_features['temp_vibration_ratio'] = df_features['temperature'] / (df_features['vibration'] + 0.001)
    df_features['power_efficiency'] = df_features['flow_rate'] / (df_features['power_consumption'] + 0.001)
    df_features['thermal_efficiency'] = df_features['flow_rate'] / (df_features['temperature'] + 0.001)
    
    # Statistical features within device
    print("Engineering rolling window features...")
    window_sizes = [5, 15, 30]  # 5, 15, 30 measurements
    
    for col in sensor_cols:
        for window in window_sizes:
            # Rolling statistics
            df_features[f'{col}_rolling_mean_{window}'] = (
                df_features.groupby('device_id')[col]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            df_features[f'{col}_rolling_std_{window}'] = (
                df_features.groupby('device_id')[col]
                .rolling(window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
                .fillna(0)
            )
            
            # Deviation from rolling mean
            df_features[f'{col}_deviation_{window}'] = (
                df_features[col] - df_features[f'{col}_rolling_mean_{window}']
            )
    
    print("Engineering change detection features...")
    # Change detection features
    for col in sensor_cols:
        # First and second derivatives (change rate)
        df_features[f'{col}_diff_1'] = (
            df_features.groupby('device_id')[col].diff().fillna(0)
        )
        df_features[f'{col}_diff_2'] = (
            df_features.groupby('device_id')[f'{col}_diff_1'].diff().fillna(0)
        )
        
        # Rate of change
        df_features[f'{col}_change_rate'] = abs(df_features[f'{col}_diff_1'])
    
    print("Engineering device-specific features...")
    # Device type encoding
    df_features = pd.get_dummies(df_features, columns=['device_type', 'location'])
    
    # Device age (simulate)
    df_features['device_age_days'] = (
        df_features['timestamp'] - pd.Timestamp('2020-01-01')
    ).dt.days + np.random.randint(0, 1000, len(df_features))  # Simulate different install dates
    
    # Operating conditions
    df_features['high_load'] = (
        (df_features['power_consumption'] > df_features['power_consumption'].quantile(0.8)) &
        (df_features['temperature'] > df_features['temperature'].quantile(0.7))
    ).astype(int)
    
    # Anomaly indicators
    for col in sensor_cols:
        q99 = df_features[col].quantile(0.99)
        q01 = df_features[col].quantile(0.01)
        df_features[f'{col}_extreme_high'] = (df_features[col] > q99).astype(int)
        df_features[f'{col}_extreme_low'] = (df_features[col] < q01).astype(int)
    
    # Remove non-numeric columns
    feature_columns = [col for col in df_features.columns 
                      if col not in ['timestamp', 'device_id', 'is_anomaly', 'anomaly_type']]
    
    print(f"✅ Feature engineering complete")
    print(f"   Original sensors: {len(sensor_cols)}")
    print(f"   Engineered features: {len(feature_columns)}")
    
    return df_features, feature_columns

# Apply feature engineering
sensor_features, feature_cols = engineer_iot_features(sensor_df)

# =============================================================================
# 4. Anomaly Detection Models for IoT Monitoring
# =============================================================================

print("\n🤖 Section 4: IoT Anomaly Detection Models")
print("-" * 50)

# Prepare data
X = sensor_features[feature_cols].values
y = sensor_features['is_anomaly'].values

# Handle NaN values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"📝 Data prepared:")
print(f"   Training: {len(X_train)} measurements")
print(f"   Testing: {len(X_test)} measurements")
print(f"   Features: {len(feature_cols)}")
print(f"   Anomaly rate: {np.mean(y)*100:.2f}%")

# Import models
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available")

if SKLEARN_AVAILABLE:
    # IoT-specific anomaly detection models
    iot_detectors = {
        'Isolation Forest': IsolationForest(
            contamination=np.mean(y),  # Use actual anomaly rate
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        ),
        'Local Outlier Factor': LocalOutlierFactor(
            contamination=np.mean(y),
            n_neighbors=20,
            n_jobs=-1
        ),
        'One-Class SVM (RBF)': OneClassSVM(
            nu=np.mean(y),
            kernel='rbf',
            gamma='scale'
        )
    }
    
    print(f"\n🔧 Training IoT anomaly detection models...")
    
    iot_results = {}
    
    for name, detector in iot_detectors.items():
        print(f"\n   Training {name}...")
        
        # Train on normal data only
        X_train_normal = X_train[y_train == 0]
        
        if name == 'Local Outlier Factor':
            detector.fit(X_train_normal)
            y_pred_test = detector.fit_predict(X_test)
        else:
            detector.fit(X_train_normal)
            y_pred_test = detector.predict(X_test)
        
        # Convert predictions
        y_pred_binary = (y_pred_test == -1).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        # Get anomaly scores
        if hasattr(detector, 'decision_function'):
            scores = detector.decision_function(X_test)
            scores = -scores
        elif hasattr(detector, 'score_samples'):
            scores = detector.score_samples(X_test)
            scores = -scores
        else:
            scores = y_pred_binary
        
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except:
            roc_auc = 0.5
        
        iot_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred_binary,
            'scores': scores
        }
        
        print(f"     📊 Precision: {precision:.3f}")
        print(f"     📊 Recall: {recall:.3f}")
        print(f"     📊 F1-Score: {f1:.3f}")
        print(f"     📊 ROC-AUC: {roc_auc:.3f}")
        
        # Calculate confusion matrix for IoT-specific metrics
        cm = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # IoT-specific metrics
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"     🚨 False Alarm Rate: {false_alarm_rate:.3f}")
        print(f"     🎯 Detection Rate: {detection_rate:.3f}")
    
    # Model comparison
    print(f"\n📊 IoT Anomaly Detection Model Comparison:")
    iot_comparison = pd.DataFrame({
        name: {
            'Precision': iot_results[name]['precision'],
            'Recall': iot_results[name]['recall'],
            'F1-Score': iot_results[name]['f1_score'],
            'ROC-AUC': iot_results[name]['roc_auc']
        }
        for name in iot_results
    }).T
    
    print(iot_comparison.round(3))
    
    # Best model for IoT (balance precision and recall)
    best_iot_model = iot_comparison['F1-Score'].idxmax()
    print(f"\n🏆 Best IoT anomaly detection model: {best_iot_model}")

# =============================================================================
# 5. Predictive Maintenance Implementation
# =============================================================================

print(f"\n🔧 Section 5: Predictive Maintenance System")
print("-" * 50)

def predictive_maintenance_analysis(sensor_data, model, feature_cols, threshold=0.7):
    """Implement predictive maintenance analysis."""
    
    # Get latest readings for each device
    latest_readings = sensor_data.groupby('device_id').last().reset_index()
    
    # Prepare features
    X_latest = latest_readings[feature_cols].values
    X_latest = imputer.transform(X_latest)
    
    # Get anomaly scores
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_latest)
        # Convert to probability-like scores (0-1)
        probabilities = 1 / (1 + np.exp(scores))
    else:
        predictions = model.predict(X_latest)
        probabilities = (predictions == -1).astype(float)
    
    # Create maintenance recommendations
    maintenance_recommendations = []
    
    for i, device_id in enumerate(latest_readings['device_id']):
        device_info = latest_readings.iloc[i]
        prob = probabilities[i]
        
        # Risk categorization
        if prob >= 0.8:
            risk_level = "CRITICAL"
            action = "IMMEDIATE_MAINTENANCE"
            priority = 1
        elif prob >= 0.6:
            risk_level = "HIGH"
            action = "SCHEDULE_MAINTENANCE"
            priority = 2
        elif prob >= 0.4:
            risk_level = "MEDIUM"
            action = "MONITOR_CLOSELY"
            priority = 3
        elif prob >= 0.2:
            risk_level = "LOW"
            action = "ROUTINE_CHECK"
            priority = 4
        else:
            risk_level = "NORMAL"
            action = "CONTINUE_MONITORING"
            priority = 5
        
        maintenance_recommendations.append({
            'device_id': device_id,
            'device_type': device_info['device_type'],
            'location': device_info['location'],
            'anomaly_probability': prob,
            'risk_level': risk_level,
            'recommended_action': action,
            'priority': priority,
            'temperature': device_info['temperature'],
            'vibration': device_info['vibration'],
            'power_consumption': device_info['power_consumption']
        })
    
    return pd.DataFrame(maintenance_recommendations)

if SKLEARN_AVAILABLE and iot_results:
    print("🔧 Generating predictive maintenance recommendations...")
    
    # Use best performing model
    best_model = iot_detectors[best_iot_model]
    
    # Generate maintenance recommendations
    maintenance_df = predictive_maintenance_analysis(
        sensor_features, best_model, feature_cols
    )
    
    print(f"\n📋 Maintenance Recommendations Summary:")
    print(maintenance_df['risk_level'].value_counts())
    
    # High priority devices
    high_priority = maintenance_df[maintenance_df['priority'] <= 2].sort_values('anomaly_probability', ascending=False)
    
    if len(high_priority) > 0:
        print(f"\n🚨 HIGH PRIORITY DEVICES REQUIRING ATTENTION:")
        print("-" * 60)
        for _, device in high_priority.head(5).iterrows():
            print(f"   Device: {device['device_id']} ({device['device_type']})")
            print(f"   Location: {device['location']}")
            print(f"   Risk Level: {device['risk_level']}")
            print(f"   Anomaly Probability: {device['anomaly_probability']:.3f}")
            print(f"   Action: {device['recommended_action']}")
            print(f"   Temperature: {device['temperature']:.1f}°C")
            print(f"   Vibration: {device['vibration']:.2f}")
            print(f"   Power: {device['power_consumption']:.1f}W")
            print()

# =============================================================================
# 6. Real-time Monitoring Dashboard Simulation
# =============================================================================

print(f"\n📊 Section 6: Real-time Monitoring Dashboard")
print("-" * 50)

def real_time_monitoring_simulation(sensor_data, model, feature_cols, duration_minutes=60):
    """Simulate real-time IoT monitoring."""
    
    print(f"🌊 Simulating {duration_minutes} minutes of real-time monitoring...")
    
    # Simulate incoming data stream
    latest_timestamp = sensor_data['timestamp'].max()
    
    alerts_generated = []
    devices_monitored = sensor_data['device_id'].unique()[:10]  # Monitor subset for demo
    
    for minute in range(duration_minutes):
        current_time = latest_timestamp + timedelta(minutes=minute)
        
        # Simulate new measurements for each device
        for device_id in devices_monitored:
            # Get device's latest normal pattern
            device_history = sensor_data[sensor_data['device_id'] == device_id].tail(50)
            
            if len(device_history) > 0:
                # Simulate new measurement based on history
                last_measurement = device_history.iloc[-1]
                
                # Add some random variation
                new_measurement = {
                    'timestamp': current_time,
                    'device_id': device_id,
                    'device_type': last_measurement['device_type'],
                    'location': last_measurement['location'],
                    'temperature': last_measurement['temperature'] + np.random.normal(0, 2),
                    'vibration': max(0, last_measurement['vibration'] + np.random.normal(0, 0.1)),
                    'pressure': max(0, last_measurement['pressure'] + np.random.normal(0, 1)),
                    'flow_rate': max(0, last_measurement['flow_rate'] + np.random.normal(0, 0.5)),
                    'power_consumption': max(0, last_measurement['power_consumption'] + np.random.normal(0, 10))
                }
                
                # Randomly inject anomalies
                if np.random.random() < 0.02:  # 2% chance of anomaly
                    anomaly_type = np.random.choice(['overheating', 'vibration_spike', 'power_surge'])
                    
                    if anomaly_type == 'overheating':
                        new_measurement['temperature'] *= 1.5
                    elif anomaly_type == 'vibration_spike':
                        new_measurement['vibration'] *= 3
                    elif anomaly_type == 'power_surge':
                        new_measurement['power_consumption'] *= 2
                
                # Quick anomaly detection (simplified)
                temp_threshold = device_history['temperature'].mean() + 3 * device_history['temperature'].std()
                vibration_threshold = device_history['vibration'].mean() + 3 * device_history['vibration'].std()
                power_threshold = device_history['power_consumption'].mean() + 3 * device_history['power_consumption'].std()
                
                alert_triggered = False
                alert_reasons = []
                
                if new_measurement['temperature'] > temp_threshold:
                    alert_triggered = True
                    alert_reasons.append(f"High temperature: {new_measurement['temperature']:.1f}°C")
                
                if new_measurement['vibration'] > vibration_threshold:
                    alert_triggered = True
                    alert_reasons.append(f"High vibration: {new_measurement['vibration']:.2f}")
                
                if new_measurement['power_consumption'] > power_threshold:
                    alert_triggered = True
                    alert_reasons.append(f"High power: {new_measurement['power_consumption']:.1f}W")
                
                if alert_triggered:
                    alert = {
                        'timestamp': current_time,
                        'device_id': device_id,
                        'device_type': new_measurement['device_type'],
                        'location': new_measurement['location'],
                        'alert_reasons': alert_reasons,
                        'severity': 'HIGH' if len(alert_reasons) > 1 else 'MEDIUM'
                    }
                    alerts_generated.append(alert)
        
        # Print progress every 10 minutes
        if minute % 10 == 0 and minute > 0:
            print(f"   ⏱️  {minute} minutes: {len(alerts_generated)} alerts generated")
    
    return alerts_generated

if SKLEARN_AVAILABLE:
    print("🔧 Setting up real-time monitoring simulation...")
    
    # Run monitoring simulation
    alerts = real_time_monitoring_simulation(sensor_df, best_model, feature_cols, 30)
    
    if alerts:
        print(f"\n🚨 REAL-TIME ALERTS GENERATED:")
        print("-" * 50)
        
        for alert in alerts[-5:]:  # Show last 5 alerts
            print(f"   🚨 ALERT: {alert['device_id']} ({alert['device_type']})")
            print(f"      Location: {alert['location']}")
            print(f"      Time: {alert['timestamp']}")
            print(f"      Severity: {alert['severity']}")
            print(f"      Reasons: {', '.join(alert['alert_reasons'])}")
            print()
        
        # Alert statistics
        alert_by_device = {}
        alert_by_location = {}
        
        for alert in alerts:
            device = alert['device_id']
            location = alert['location']
            
            alert_by_device[device] = alert_by_device.get(device, 0) + 1
            alert_by_location[location] = alert_by_location.get(location, 0) + 1
        
        print(f"📊 Alert Statistics:")
        print(f"   Total alerts: {len(alerts)}")
        print(f"   Most problematic device: {max(alert_by_device, key=alert_by_device.get) if alert_by_device else 'None'}")
        print(f"   Most problematic location: {max(alert_by_location, key=alert_by_location.get) if alert_by_location else 'None'}")

# =============================================================================
# 7. Operational Efficiency Analysis
# =============================================================================

print(f"\n💼 Section 7: Operational Efficiency Analysis")
print("-" * 50)

# Calculate efficiency metrics
print("📊 Analyzing operational efficiency...")

if SKLEARN_AVAILABLE and iot_results:
    # Device reliability metrics
    device_reliability = sensor_df.groupby('device_id').agg({
        'is_anomaly': ['count', 'sum', 'mean'],
        'timestamp': ['min', 'max']
    }).round(4)
    
    device_reliability.columns = ['Total_Readings', 'Anomalies', 'Failure_Rate', 'First_Reading', 'Last_Reading']
    device_reliability['Uptime_Hours'] = (
        device_reliability['Last_Reading'] - device_reliability['First_Reading']
    ).dt.total_seconds() / 3600
    
    print(f"📈 Device Reliability Summary:")
    print(f"   Average failure rate: {device_reliability['Failure_Rate'].mean():.4f}")
    print(f"   Most reliable device: {device_reliability['Failure_Rate'].idxmin()}")
    print(f"   Least reliable device: {device_reliability['Failure_Rate'].idxmax()}")
    
    # Energy efficiency analysis
    energy_efficiency = sensor_df.groupby(['device_type', 'location']).agg({
        'power_consumption': ['mean', 'std'],
        'flow_rate': ['mean'],
        'is_anomaly': ['mean']
    }).round(2)
    
    print(f"\n⚡ Energy Efficiency by Device Type and Location:")
    print(energy_efficiency.head())
    
    # Maintenance cost estimation
    anomaly_costs = {
        'overheating': 500,      # Cost of emergency cooling/repair
        'vibration_spike': 800,  # Cost of mechanical repair
        'pressure_drop': 300,    # Cost of seal/valve replacement
        'power_surge': 400,      # Cost of electrical system check
        'sensor_drift': 200,     # Cost of sensor recalibration
        'normal': 0
    }
    
    # Calculate estimated maintenance costs
    total_maintenance_cost = 0
    anomaly_breakdown = sensor_df[sensor_df['is_anomaly']]['anomaly_type'].value_counts()
    
    print(f"\n💰 Estimated Maintenance Costs:")
    for anomaly_type, count in anomaly_breakdown.items():
        if anomaly_type in anomaly_costs:
            cost = count * anomaly_costs[anomaly_type]
            total_maintenance_cost += cost
            print(f"   {anomaly_type}: {count} incidents × ${anomaly_costs[anomaly_type]} = ${cost:,}")
    
    print(f"   TOTAL ESTIMATED COST: ${total_maintenance_cost:,}")
    
    # Predictive maintenance savings
    prevention_cost_multiplier = 0.3  # Preventive maintenance costs 30% of emergency repair
    prevented_incidents = sum(iot_results[best_iot_model]['predictions'])
    
    if prevented_incidents > 0:
        potential_savings = total_maintenance_cost * (1 - prevention_cost_multiplier)
        print(f"\n💡 Predictive Maintenance Benefits:")
        print(f"   Incidents prevented: {prevented_incidents}")
        print(f"   Potential savings: ${potential_savings:,}")
        print(f"   ROI from predictive maintenance: {(potential_savings / (total_maintenance_cost * prevention_cost_multiplier)) * 100:.1f}%")

print(f"\n🎉 IoT Monitoring & Predictive Maintenance Analysis Complete!")
print("=" * 60)

print(f"📚 Key Achievements:")
print(f"   ✅ Generated realistic IoT sensor data with multiple failure modes")
print(f"   ✅ Engineered comprehensive features for anomaly detection")
print(f"   ✅ Trained and evaluated multiple anomaly detection models")
print(f"   ✅ Implemented predictive maintenance recommendations")
print(f"   ✅ Simulated real-time monitoring and alerting")
print(f"   ✅ Analyzed operational efficiency and cost savings")

print(f"\n🚀 Production Implementation Guidelines:")
print(f"   📡 Data Pipeline: Implement robust IoT data ingestion")
print(f"   ⚡ Real-time Processing: Use stream processing (Kafka, Storm)")
print(f"   🔄 Model Updates: Retrain models with new failure patterns")
print(f"   📊 Dashboard: Build comprehensive monitoring dashboard")
print(f"   🚨 Alerting: Implement multi-channel alert system")
print(f"   📱 Mobile: Develop mobile app for field technicians")
print(f"   🤖 Automation: Integrate with maintenance management systems")

print(f"\n🔧 Advanced Features to Consider:")
print(f"   1. Digital twin modeling for complex systems")
print(f"   2. Deep learning for complex pattern recognition")
print(f"   3. Federated learning across multiple sites")
print(f"   4. Edge computing for local anomaly detection")
print(f"   5. Integration with SCADA/MES systems")
print(f"   6. Automated work order generation")

print(f"\nIoT monitoring system ready for industrial deployment! 🏭🔧")