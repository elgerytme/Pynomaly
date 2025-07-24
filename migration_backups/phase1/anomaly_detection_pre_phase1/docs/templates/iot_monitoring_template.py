#!/usr/bin/env python3
"""
IoT Device Monitoring Template
=============================

Ready-to-use template for monitoring IoT sensors and detecting device malfunctions.
Copy this file and modify the data loading section for your use case.

Usage:
    python iot_monitoring_template.py

Requirements:
    - pandas
    - numpy
    - anomaly_detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from anomaly_detection import DetectionService, StreamingService
import time

def generate_sample_sensor_data(days=7, devices=20):
    """
    Generate sample IoT sensor data for testing.
    Replace this function with your actual data loading logic.
    """
    np.random.seed(42)
    
    # Time range
    start_time = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start_time, periods=days*24, freq='H')
    
    data = []
    
    for device_id in range(devices):
        # Normal device behavior
        base_temp = 20 + np.random.normal(0, 2)  # Device-specific baseline
        temp_drift = np.random.normal(0, 0.1, len(timestamps))  # Slow drift
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            
            # Daily temperature cycle
            daily_cycle = 5 * np.sin(2 * np.pi * hour / 24)
            
            # Simulate device malfunction (5% of devices, 10% of time)
            is_malfunctioning = (device_id < devices * 0.05) and (np.random.random() < 0.1)
            
            if is_malfunctioning:
                # Malfunctioning device patterns
                temperature = base_temp + daily_cycle + temp_drift[i] + np.random.normal(15, 5)
                humidity = np.random.uniform(10, 30)  # Very low humidity
                power_consumption = np.random.uniform(150, 300)  # High power
                vibration = np.random.uniform(2, 5)  # High vibration
                battery_level = max(0, 100 - np.random.exponential(20))  # Rapid drain
            else:
                # Normal device behavior
                temperature = base_temp + daily_cycle + temp_drift[i] + np.random.normal(0, 1)
                humidity = np.random.normal(50, 5)  # Normal humidity
                power_consumption = np.random.normal(100, 10)  # Normal power
                vibration = np.random.normal(0.5, 0.1)  # Low vibration
                battery_level = max(20, 100 - i * 0.1 + np.random.normal(0, 2))  # Slow drain
            
            data.append({
                'timestamp': timestamp,
                'device_id': device_id,
                'temperature': temperature,
                'humidity': max(0, min(100, humidity)),
                'power_consumption': max(0, power_consumption),
                'vibration': max(0, vibration),
                'battery_level': max(0, min(100, battery_level)),
                'is_malfunctioning': is_malfunctioning
            })
    
    return pd.DataFrame(data)

def load_your_sensor_data():
    """
    Replace this function with your actual IoT data loading logic.
    
    Expected columns:
    - timestamp: Reading timestamp
    - device_id: Unique device identifier
    - temperature: Temperature reading (°C)
    - humidity: Humidity percentage (0-100)
    - power_consumption: Power usage (W)
    - vibration: Vibration level
    - battery_level: Battery percentage (0-100)
    
    Returns:
        pd.DataFrame: Sensor readings
    """
    # OPTION 1: Load from time series database
    # import influxdb_client
    # client = influxdb_client.InfluxDBClient(url="http://localhost:8086", token="your_token")
    # query = 'from(bucket:"sensors") |> range(start: -7d)'
    # return client.query_api().query_data_frame(query)
    
    # OPTION 2: Load from CSV files
    # return pd.read_csv('sensor_data.csv', parse_dates=['timestamp'])
    
    # OPTION 3: Load from MQTT/Kafka stream (for real-time monitoring)
    # from kafka import KafkaConsumer
    # consumer = KafkaConsumer('sensor-topic')
    # return process_kafka_messages(consumer)
    
    # OPTION 4: Use sample data for testing
    print("📝 Using sample sensor data for demonstration")
    return generate_sample_sensor_data()

def preprocess_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for IoT anomaly detection.
    
    Args:
        df: Raw sensor data
        
    Returns:
        pd.DataFrame: Processed data with features
    """
    print("🔧 Engineering IoT features...")
    
    # Sort by device and timestamp
    df = df.sort_values(['device_id', 'timestamp']).copy()
    
    # Feature engineering by device
    device_features = []
    
    for device_id in df['device_id'].unique():
        device_data = df[df['device_id'] == device_id].copy()
        
        # Rolling statistics (last 6 hours)
        window = min(6, len(device_data))
        device_data['temp_rolling_mean'] = device_data['temperature'].rolling(window).mean()
        device_data['temp_rolling_std'] = device_data['temperature'].rolling(window).std()
        device_data['power_rolling_mean'] = device_data['power_consumption'].rolling(window).mean()
        
        # Rate of change
        device_data['temp_change_rate'] = device_data['temperature'].diff()
        device_data['battery_drain_rate'] = -device_data['battery_level'].diff()
        
        # Deviation from baseline
        temp_baseline = device_data['temperature'].quantile(0.5)
        device_data['temp_deviation'] = abs(device_data['temperature'] - temp_baseline)
        
        # Time-based features
        device_data['hour'] = device_data['timestamp'].dt.hour
        device_data['is_night'] = ((device_data['hour'] < 6) | (device_data['hour'] > 22)).astype(int)
        
        # Operational status indicators
        device_data['low_battery'] = (device_data['battery_level'] < 20).astype(int)
        device_data['high_power'] = (device_data['power_consumption'] > device_data['power_consumption'].quantile(0.8)).astype(int)
        device_data['high_vibration'] = (device_data['vibration'] > 1.0).astype(int)
        
        device_features.append(device_data)
    
    # Combine all devices
    processed_df = pd.concat(device_features, ignore_index=True)
    
    # Fill NaN values from rolling calculations
    processed_df = processed_df.fillna(method='bfill').fillna(0)
    
    print(f"   ✅ Created features for {processed_df['device_id'].nunique()} devices")
    return processed_df

def detect_device_anomalies_batch(df: pd.DataFrame, contamination: float = 0.05):
    """
    Detect anomalies in batch mode (historical analysis).
    
    Args:
        df: Processed sensor data
        contamination: Expected malfunction rate
        
    Returns:
        tuple: (results_df, detection_result)
    """
    print(f"🔍 Running batch anomaly detection...")
    print(f"   Expected malfunction rate: {contamination*100:.1f}%")
    
    # Select features for detection
    feature_columns = [
        'temperature', 'humidity', 'power_consumption', 'vibration', 'battery_level',
        'temp_rolling_mean', 'temp_rolling_std', 'power_rolling_mean',
        'temp_change_rate', 'battery_drain_rate', 'temp_deviation',
        'low_battery', 'high_power', 'high_vibration'
    ]
    
    features = df[feature_columns].values
    
    # Detect anomalies
    service = DetectionService()
    result = service.detect(
        data=features,
        algorithm='isolation_forest',
        contamination=contamination,
        n_estimators=150,
        random_state=42
    )
    
    # Add results to dataframe
    results_df = df.copy()
    results_df['anomaly_score'] = result.scores
    results_df['is_anomaly'] = result.predictions == -1
    
    print(f"   ✅ Batch detection complete in {result.processing_time:.2f}s")
    return results_df, result

def setup_realtime_monitoring(contamination: float = 0.05):
    """
    Set up real-time IoT monitoring.
    
    Args:
        contamination: Expected malfunction rate
        
    Returns:
        StreamingService: Configured streaming service
    """
    print("🔄 Setting up real-time monitoring...")
    
    def on_device_malfunction(sample, score, metadata):
        """Callback when device malfunction is detected"""
        device_id = metadata.get('device_id', 'Unknown')
        timestamp = metadata.get('timestamp', datetime.now())
        
        print(f"🚨 DEVICE MALFUNCTION DETECTED!")
        print(f"   Device ID: {device_id}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Anomaly Score: {score:.3f}")
        print(f"   Temperature: {sample[0]:.1f}°C")
        print(f"   Power: {sample[2]:.1f}W")
        print(f"   Battery: {sample[4]:.1f}%")
        print(f"   🔔 Alert sent to maintenance team!")
        print("-" * 50)
    
    # Configure streaming service
    streaming = StreamingService()
    streaming.configure(
        algorithm='isolation_forest',
        window_size=100,
        contamination=contamination,
        callback=on_device_malfunction
    )
    
    print("   ✅ Real-time monitoring configured")
    return streaming

def simulate_realtime_monitoring(streaming_service, duration_minutes: int = 5):
    """
    Simulate real-time IoT data processing.
    
    Args:
        streaming_service: Configured streaming service
        duration_minutes: How long to run simulation
    """
    print(f"📡 Starting real-time simulation for {duration_minutes} minutes...")
    print("   (Press Ctrl+C to stop early)")
    
    device_count = 10
    start_time = time.time()
    sample_count = 0
    
    try:
        while (time.time() - start_time) < (duration_minutes * 60):
            # Simulate sensor reading from random device
            device_id = np.random.randint(0, device_count)
            
            # Generate sensor reading
            if np.random.random() < 0.02:  # 2% chance of malfunction
                # Malfunctioning device
                sample = np.array([
                    np.random.normal(45, 10),    # High temperature
                    np.random.uniform(10, 30),   # Low humidity
                    np.random.uniform(150, 250), # High power
                    np.random.uniform(2, 4),     # High vibration
                    np.random.uniform(5, 25)     # Low battery
                ])
            else:
                # Normal device
                sample = np.array([
                    np.random.normal(22, 2),     # Normal temperature
                    np.random.normal(50, 5),     # Normal humidity
                    np.random.normal(100, 10),   # Normal power
                    np.random.normal(0.5, 0.2),  # Low vibration
                    np.random.uniform(30, 90)    # Good battery
                ])
            
            # Process sample
            metadata = {
                'device_id': device_id,
                'timestamp': datetime.now()
            }
            
            result = streaming_service.process_sample(sample, metadata=metadata)
            sample_count += 1
            
            if not result.is_anomaly:
                if sample_count % 20 == 0:  # Show periodic updates
                    print(f"✅ Processed {sample_count} samples from {device_count} devices")
            
            time.sleep(0.5)  # Simulate 2 readings per second
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring stopped by user after {sample_count} samples")
    
    print(f"📊 Monitoring session complete: {sample_count} samples processed")

def analyze_device_health(results_df: pd.DataFrame):
    """
    Analyze overall device health and generate maintenance recommendations.
    
    Args:
        results_df: Results from batch anomaly detection
    """
    print(f"\n📊 Device Health Analysis")
    print("=" * 50)
    
    # Overall statistics
    total_readings = len(results_df)
    total_anomalies = results_df['is_anomaly'].sum()
    devices_with_issues = results_df[results_df['is_anomaly']]['device_id'].nunique()
    total_devices = results_df['device_id'].nunique()
    
    print(f"📈 Overall Health Metrics:")
    print(f"   Total readings analyzed: {total_readings:,}")
    print(f"   Anomalous readings: {total_anomalies:,} ({total_anomalies/total_readings*100:.1f}%)")
    print(f"   Devices with issues: {devices_with_issues}/{total_devices} ({devices_with_issues/total_devices*100:.1f}%)")
    
    # Device-specific analysis
    device_health = results_df.groupby('device_id').agg({
        'is_anomaly': ['count', 'sum'],
        'anomaly_score': 'mean',
        'temperature': 'mean',
        'battery_level': 'mean',
        'power_consumption': 'mean'
    }).round(2)
    
    device_health.columns = ['total_readings', 'anomaly_count', 'avg_anomaly_score', 
                           'avg_temperature', 'avg_battery', 'avg_power']
    device_health['anomaly_rate'] = (device_health['anomaly_count'] / device_health['total_readings'] * 100).round(1)
    
    # Identify devices needing attention
    high_risk_devices = device_health[device_health['anomaly_rate'] > 10].sort_values('anomaly_rate', ascending=False)
    
    if len(high_risk_devices) > 0:
        print(f"\n🚨 High-Risk Devices (>10% anomaly rate):")
        for device_id, stats in high_risk_devices.head(5).iterrows():
            print(f"   Device {device_id}: {stats['anomaly_rate']:.1f}% anomaly rate")
            print(f"      Avg temp: {stats['avg_temperature']:.1f}°C, Battery: {stats['avg_battery']:.1f}%")
    
    # Maintenance recommendations
    print(f"\n💡 Maintenance Recommendations:")
    
    # Low battery devices
    low_battery = device_health[device_health['avg_battery'] < 30]
    if len(low_battery) > 0:
        print(f"   🔋 {len(low_battery)} devices need battery replacement")
    
    # High temperature devices
    high_temp = device_health[device_health['avg_temperature'] > 30]
    if len(high_temp) > 0:
        print(f"   🌡️ {len(high_temp)} devices running hot - check ventilation")
    
    # High power consumption devices
    high_power = device_health[device_health['avg_power'] > 120]
    if len(high_power) > 0:
        print(f"   ⚡ {len(high_power)} devices consuming excess power")
    
    return device_health

def save_monitoring_results(results_df: pd.DataFrame, device_health: pd.DataFrame):
    """
    Save monitoring results and health analysis.
    
    Args:
        results_df: Detailed anomaly detection results
        device_health: Device health summary
    """
    # Save detailed results
    anomaly_file = 'iot_anomaly_results.csv'
    results_df.to_csv(anomaly_file, index=False)
    
    # Save device health summary
    health_file = 'device_health_summary.csv'
    device_health.to_csv(health_file)
    
    print(f"\n💾 Results saved:")
    print(f"   Detailed results: {anomaly_file}")
    print(f"   Device health summary: {health_file}")

def main():
    """
    Main IoT monitoring pipeline.
    """
    print("🏭 IoT Device Monitoring System")
    print("=" * 50)
    
    # Step 1: Load sensor data
    print("\n📡 Loading sensor data...")
    df = load_your_sensor_data()
    print(f"   Loaded {len(df):,} readings from {df['device_id'].nunique()} devices")
    
    # Step 2: Preprocess and engineer features
    processed_df = preprocess_sensor_features(df)
    
    # Step 3: Batch anomaly detection (historical analysis)
    results_df, detection_result = detect_device_anomalies_batch(processed_df)
    
    # Step 4: Analyze device health
    device_health = analyze_device_health(results_df)
    
    # Step 5: Save results
    save_monitoring_results(results_df, device_health)
    
    # Step 6: Optional real-time monitoring demo
    print(f"\n🔄 Real-time Monitoring Demo")
    choice = input("   Start real-time monitoring simulation? (y/n): ").lower().strip()
    
    if choice == 'y':
        streaming_service = setup_realtime_monitoring()
        simulate_realtime_monitoring(streaming_service, duration_minutes=2)
    
    print(f"\n✅ IoT monitoring analysis complete!")
    print(f"💡 Tip: Set up automated alerts for devices with high anomaly rates")

if __name__ == "__main__":
    main()