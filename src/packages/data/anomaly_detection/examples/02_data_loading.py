#!/usr/bin/env python3
"""
Data Loading Examples for Anomaly Detection Package

This example demonstrates various ways to load and prepare data for anomaly detection,
including CSV files, JSON, databases, APIs, and streaming sources.
"""

import numpy as np
import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService
    from anomaly_detection.domain.entities.dataset import Dataset
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


class DataLoader:
    """Utility class for loading data from various sources."""
    
    def __init__(self):
        self.service = DetectionService()
    
    def load_csv(self, 
                  filepath: str, 
                  target_column: Optional[str] = None,
                  feature_columns: Optional[List[str]] = None,
                  sep: str = ',',
                  encoding: str = 'utf-8') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_column: Column name for labels (if supervised)
            feature_columns: List of feature columns to use
            sep: Delimiter
            encoding: File encoding
            
        Returns:
            X: Feature matrix
            y: Labels (if target_column specified)
        """
        print(f"Loading data from CSV: {filepath}")
        
        # Read CSV
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Select features
        if feature_columns:
            X = df[feature_columns].values
        else:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_cols:
                numeric_cols.remove(target_column)
            X = df[numeric_cols].values
        
        # Extract target if specified
        y = df[target_column].values if target_column and target_column in df.columns else None
        
        print(f"Feature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def load_json(self, 
                  filepath: str,
                  data_key: str = 'data',
                  features_key: str = 'features',
                  labels_key: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to JSON file
            data_key: Key containing the data
            features_key: Key for features within data
            labels_key: Key for labels (optional)
            
        Returns:
            X: Feature matrix
            y: Labels (if available)
        """
        print(f"Loading data from JSON: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract features
        if data_key in data:
            if isinstance(data[data_key], list):
                X = np.array(data[data_key])
            elif features_key in data[data_key]:
                X = np.array(data[data_key][features_key])
            else:
                X = np.array(data[data_key])
        else:
            # Try to extract from root
            X = np.array(data)
        
        # Extract labels if available
        y = None
        if labels_key:
            if data_key in data and labels_key in data[data_key]:
                y = np.array(data[data_key][labels_key])
            elif labels_key in data:
                y = np.array(data[labels_key])
        
        print(f"Feature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def load_from_database(self,
                          connection_string: str,
                          query: str,
                          feature_columns: Optional[List[str]] = None,
                          target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from database using SQL query.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            feature_columns: Columns to use as features
            target_column: Column for labels
            
        Returns:
            X: Feature matrix
            y: Labels (if available)
        """
        print(f"Loading data from database...")
        
        # For SQLite example
        if connection_string.startswith('sqlite://'):
            db_path = connection_string.replace('sqlite://', '')
            conn = sqlite3.connect(db_path)
        else:
            # For other databases, you would use appropriate connectors
            raise NotImplementedError("Only SQLite is implemented in this example")
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Query returned {len(df)} rows with {len(df.columns)} columns")
        
        # Extract features and target
        if feature_columns:
            X = df[feature_columns].values
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_cols:
                numeric_cols.remove(target_column)
            X = df[numeric_cols].values
        
        y = df[target_column].values if target_column and target_column in df.columns else None
        
        return X, y
    
    def load_from_api(self,
                      api_url: str,
                      headers: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, Any]] = None,
                      data_path: Optional[List[str]] = None) -> np.ndarray:
        """
        Load data from REST API.
        
        Args:
            api_url: API endpoint URL
            headers: Request headers
            params: Query parameters
            data_path: Path to data in JSON response
            
        Returns:
            X: Feature matrix
        """
        import requests
        
        print(f"Loading data from API: {api_url}")
        
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Navigate to data using path
        if data_path:
            for key in data_path:
                data = data[key]
        
        X = np.array(data)
        print(f"Loaded data shape: {X.shape}")
        
        return X
    
    def create_streaming_simulator(self,
                                  data: np.ndarray,
                                  batch_size: int = 10,
                                  delay: float = 0.1) -> Any:
        """
        Create a streaming data simulator.
        
        Args:
            data: Source data to stream
            batch_size: Size of each batch
            delay: Delay between batches (seconds)
            
        Yields:
            Batches of data
        """
        import time
        
        print(f"Creating streaming simulator with batch_size={batch_size}")
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            yield batch
            time.sleep(delay)
    
    def preprocess_data(self,
                       X: np.ndarray,
                       handle_missing: str = 'drop',
                       scaling: str = 'standard',
                       remove_duplicates: bool = True) -> np.ndarray:
        """
        Preprocess data for anomaly detection.
        
        Args:
            X: Raw feature matrix
            handle_missing: How to handle missing values ('drop', 'mean', 'median', 'zero')
            scaling: Scaling method ('standard', 'minmax', 'robust', None)
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            X_processed: Preprocessed data
        """
        print("\nPreprocessing data...")
        original_shape = X.shape
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(X)
        
        # Handle missing values
        if handle_missing == 'drop':
            df = df.dropna()
            print(f"Dropped {original_shape[0] - len(df)} rows with missing values")
        elif handle_missing == 'mean':
            df = df.fillna(df.mean())
        elif handle_missing == 'median':
            df = df.fillna(df.median())
        elif handle_missing == 'zero':
            df = df.fillna(0)
        
        # Remove duplicates
        if remove_duplicates:
            original_len = len(df)
            df = df.drop_duplicates()
            print(f"Removed {original_len - len(df)} duplicate rows")
        
        # Convert back to array
        X = df.values
        
        # Scaling
        if scaling:
            if scaling == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif scaling == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif scaling == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                scaler = None
            
            if scaler:
                X = scaler.fit_transform(X)
                print(f"Applied {scaling} scaling")
        
        print(f"Final shape: {X.shape}")
        return X


def create_sample_files(temp_dir: Path) -> Dict[str, Path]:
    """Create sample data files for demonstration."""
    files = {}
    
    # Create sample CSV
    csv_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) * 2,
        'feature3': np.random.exponential(2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'label': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
    csv_path = temp_dir / 'sample_data.csv'
    csv_data.to_csv(csv_path, index=False)
    files['csv'] = csv_path
    
    # Create sample JSON
    json_data = {
        'metadata': {
            'source': 'sensor_network',
            'timestamp': '2024-01-23T10:00:00Z'
        },
        'data': {
            'features': np.random.randn(50, 3).tolist(),
            'labels': np.random.choice([0, 1], 50, p=[0.85, 0.15]).tolist()
        }
    }
    json_path = temp_dir / 'sample_data.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    files['json'] = json_path
    
    # Create sample SQLite database
    db_path = temp_dir / 'sample_data.db'
    conn = sqlite3.connect(db_path)
    
    # Create and populate table
    sensor_data = pd.DataFrame({
        'sensor_id': np.repeat(range(1, 6), 20),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.normal(50, 10, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'is_anomaly': np.random.choice([0, 1], 100, p=[0.92, 0.08])
    })
    sensor_data.to_sql('sensor_readings', conn, index=False, if_exists='replace')
    conn.close()
    files['db'] = db_path
    
    return files


def example_1_csv_loading():
    """Example 1: Loading data from CSV files."""
    print("\n" + "="*60)
    print("Example 1: Loading Data from CSV Files")
    print("="*60)
    
    loader = DataLoader()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        files = create_sample_files(temp_path)
        
        # Basic CSV loading
        print("\n1.1 Basic CSV Loading:")
        X, y = loader.load_csv(files['csv'])
        
        # CSV with specific columns
        print("\n1.2 Loading specific columns:")
        X_subset, y = loader.load_csv(
            files['csv'],
            feature_columns=['feature1', 'feature2'],
            target_column='label'
        )
        
        # Detect anomalies
        print("\n1.3 Running anomaly detection on CSV data:")
        service = DetectionService()
        result = service.detect_anomalies(X_subset, algorithm='iforest')
        print(f"Detected {result.anomaly_count} anomalies out of {len(X_subset)} samples")


def example_2_json_loading():
    """Example 2: Loading data from JSON files."""
    print("\n" + "="*60)
    print("Example 2: Loading Data from JSON Files")
    print("="*60)
    
    loader = DataLoader()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        files = create_sample_files(temp_path)
        
        # Load from JSON
        print("\n2.1 Loading from nested JSON structure:")
        X, y = loader.load_json(
            files['json'],
            data_key='data',
            features_key='features',
            labels_key='labels'
        )
        
        # Detect anomalies
        print("\n2.2 Running anomaly detection on JSON data:")
        service = DetectionService()
        result = service.detect_anomalies(X, algorithm='lof')
        print(f"Detected {result.anomaly_count} anomalies")
        
        # Compare with true labels if available
        if y is not None:
            true_anomalies = np.sum(y == 1)
            print(f"True anomalies in data: {true_anomalies}")


def example_3_database_loading():
    """Example 3: Loading data from databases."""
    print("\n" + "="*60)
    print("Example 3: Loading Data from Databases")
    print("="*60)
    
    loader = DataLoader()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        files = create_sample_files(temp_path)
        
        # Load from database
        print("\n3.1 Loading from SQLite database:")
        X, y = loader.load_from_database(
            connection_string=f"sqlite://{files['db']}",
            query="""
                SELECT temperature, humidity, pressure, is_anomaly
                FROM sensor_readings
                WHERE sensor_id IN (1, 2, 3)
            """,
            feature_columns=['temperature', 'humidity', 'pressure'],
            target_column='is_anomaly'
        )
        
        # Aggregate by sensor
        print("\n3.2 Loading aggregated data:")
        X_agg, _ = loader.load_from_database(
            connection_string=f"sqlite://{files['db']}",
            query="""
                SELECT 
                    sensor_id,
                    AVG(temperature) as avg_temp,
                    STD(temperature) as std_temp,
                    AVG(humidity) as avg_humidity,
                    STD(humidity) as std_humidity,
                    AVG(pressure) as avg_pressure,
                    STD(pressure) as std_pressure
                FROM sensor_readings
                GROUP BY sensor_id
            """,
            feature_columns=['avg_temp', 'std_temp', 'avg_humidity', 
                           'std_humidity', 'avg_pressure', 'std_pressure']
        )
        
        print("\n3.3 Detecting anomalies in sensor data:")
        service = DetectionService()
        result = service.detect_anomalies(X, algorithm='iforest')
        print(f"Detected {result.anomaly_count} anomalous readings")


def example_4_preprocessing():
    """Example 4: Data preprocessing for anomaly detection."""
    print("\n" + "="*60)
    print("Example 4: Data Preprocessing")
    print("="*60)
    
    loader = DataLoader()
    
    # Create data with issues
    print("Creating data with missing values and outliers...")
    X = np.random.randn(100, 3)
    
    # Add missing values
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan
    
    # Add extreme outliers
    X[95:98] = X[95:98] * 10
    
    # Add duplicates
    X[50:55] = X[45:50]
    
    print(f"Original data shape: {X.shape}")
    print(f"Missing values: {np.sum(np.isnan(X))}")
    
    # Preprocess data
    print("\n4.1 Preprocessing with different strategies:")
    
    # Strategy 1: Drop missing values
    X1 = loader.preprocess_data(
        X.copy(),
        handle_missing='drop',
        scaling='standard',
        remove_duplicates=True
    )
    
    # Strategy 2: Impute missing values
    X2 = loader.preprocess_data(
        X.copy(),
        handle_missing='mean',
        scaling='robust',  # Robust to outliers
        remove_duplicates=True
    )
    
    # Compare detection results
    service = DetectionService()
    
    print("\n4.2 Comparing detection results:")
    result1 = service.detect_anomalies(X1, algorithm='iforest')
    result2 = service.detect_anomalies(X2, algorithm='iforest')
    
    print(f"Strategy 1 (drop missing): {result1.anomaly_count} anomalies in {len(X1)} samples")
    print(f"Strategy 2 (impute missing): {result2.anomaly_count} anomalies in {len(X2)} samples")


def example_5_streaming_data():
    """Example 5: Streaming data processing."""
    print("\n" + "="*60)
    print("Example 5: Streaming Data Processing")
    print("="*60)
    
    loader = DataLoader()
    service = DetectionService()
    
    # Create base data
    print("Creating streaming data source...")
    base_data = np.random.randn(200, 3)
    
    # Train initial model
    print("Training initial model on historical data...")
    service.fit(base_data[:100], algorithm='iforest')
    
    # Create streaming simulator
    stream = loader.create_streaming_simulator(
        base_data[100:],
        batch_size=10,
        delay=0.1
    )
    
    # Process stream
    print("\nProcessing streaming data:")
    total_anomalies = 0
    batch_count = 0
    
    try:
        for batch in stream:
            batch_count += 1
            result = service.predict(batch, algorithm='iforest')
            total_anomalies += result.anomaly_count
            
            if result.anomaly_count > 0:
                print(f"Batch {batch_count}: {result.anomaly_count} anomalies detected")
            
            # Simulate stopping after 5 batches for demo
            if batch_count >= 5:
                break
    
    except KeyboardInterrupt:
        print("\nStreaming stopped by user")
    
    print(f"\nProcessed {batch_count} batches")
    print(f"Total anomalies detected: {total_anomalies}")


def example_6_multi_source_loading():
    """Example 6: Loading and combining data from multiple sources."""
    print("\n" + "="*60)
    print("Example 6: Multi-Source Data Loading")
    print("="*60)
    
    loader = DataLoader()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        files = create_sample_files(temp_path)
        
        # Load from multiple sources
        print("Loading data from multiple sources...")
        
        # Source 1: CSV
        X_csv, _ = loader.load_csv(files['csv'], feature_columns=['feature1', 'feature2'])
        
        # Source 2: JSON
        X_json, _ = loader.load_json(files['json'])
        
        # Source 3: Database
        X_db, _ = loader.load_from_database(
            connection_string=f"sqlite://{files['db']}",
            query="SELECT temperature, humidity FROM sensor_readings LIMIT 50",
        )
        
        # Ensure compatible dimensions
        min_features = min(X_csv.shape[1], X_json.shape[1], X_db.shape[1])
        X_csv = X_csv[:, :min_features]
        X_json = X_json[:, :min_features]
        X_db = X_db[:, :min_features]
        
        # Combine data
        print("\nCombining data sources:")
        X_combined = np.vstack([X_csv, X_json, X_db])
        print(f"Combined data shape: {X_combined.shape}")
        
        # Add source labels for tracking
        source_labels = np.concatenate([
            np.full(len(X_csv), 'csv'),
            np.full(len(X_json), 'json'),
            np.full(len(X_db), 'database')
        ])
        
        # Detect anomalies
        print("\nDetecting anomalies in combined data:")
        service = DetectionService()
        result = service.detect_anomalies(X_combined, algorithm='iforest')
        
        # Analyze anomalies by source
        anomaly_indices = np.where(result.predictions == -1)[0]
        anomaly_sources = source_labels[anomaly_indices]
        
        print(f"\nTotal anomalies: {result.anomaly_count}")
        for source in ['csv', 'json', 'database']:
            count = np.sum(anomaly_sources == source)
            print(f"  From {source}: {count}")


def example_7_custom_data_formats():
    """Example 7: Loading custom data formats."""
    print("\n" + "="*60)
    print("Example 7: Custom Data Format Loading")
    print("="*60)
    
    # Example: Time series data with multiple sensors
    print("7.1 Loading time series data:")
    
    # Create sample time series data
    dates = pd.date_range('2024-01-01', periods=1000, freq='h')
    ts_data = pd.DataFrame({
        'timestamp': dates,
        'sensor_1': np.sin(np.arange(1000) * 0.1) + np.random.randn(1000) * 0.1,
        'sensor_2': np.cos(np.arange(1000) * 0.1) + np.random.randn(1000) * 0.1,
        'sensor_3': np.random.randn(1000).cumsum()
    })
    
    # Add some anomalies
    ts_data.loc[100:110, 'sensor_1'] += 3
    ts_data.loc[500:505, 'sensor_2'] *= 5
    
    # Extract features for anomaly detection
    print("Extracting features from time series...")
    
    # Rolling statistics
    window_size = 24  # 24 hours
    features = pd.DataFrame()
    
    for col in ['sensor_1', 'sensor_2', 'sensor_3']:
        features[f'{col}_mean'] = ts_data[col].rolling(window_size).mean()
        features[f'{col}_std'] = ts_data[col].rolling(window_size).std()
        features[f'{col}_diff'] = ts_data[col].diff()
    
    # Remove NaN values from rolling
    features = features.dropna()
    X_ts = features.values
    
    print(f"Feature matrix shape: {X_ts.shape}")
    
    # Detect anomalies
    service = DetectionService()
    result = service.detect_anomalies(X_ts, algorithm='iforest')
    
    print(f"Detected {result.anomaly_count} anomalies in time series")
    
    # Example: Text data (converted to numerical features)
    print("\n7.2 Loading text data with feature extraction:")
    
    # Sample log data
    logs = [
        "INFO: User login successful",
        "ERROR: Database connection failed",
        "INFO: Data processing completed",
        "CRITICAL: System memory exceeded",
        "WARNING: High CPU usage detected",
        "ERROR: Authentication failed for user admin",
        "INFO: Backup completed successfully",
        "CRITICAL: Disk space critically low",
    ]
    
    # Simple feature extraction
    log_features = []
    for log in logs:
        features = [
            len(log),  # Length
            log.count(' '),  # Word count
            1 if 'ERROR' in log else 0,
            1 if 'CRITICAL' in log else 0,
            1 if 'WARNING' in log else 0,
            1 if 'failed' in log.lower() else 0,
        ]
        log_features.append(features)
    
    X_logs = np.array(log_features)
    print(f"Log feature matrix shape: {X_logs.shape}")
    
    # Detect anomalous logs
    result = service.detect_anomalies(X_logs, algorithm='lof', contamination=0.25)
    anomaly_indices = np.where(result.predictions == -1)[0]
    
    print(f"\nAnomalous logs detected:")
    for idx in anomaly_indices:
        print(f"  - {logs[idx]}")


def main():
    """Run all data loading examples."""
    print("\n" + "="*60)
    print("ANOMALY DETECTION - DATA LOADING EXAMPLES")
    print("="*60)
    
    examples = [
        ("CSV File Loading", example_1_csv_loading),
        ("JSON File Loading", example_2_json_loading),
        ("Database Loading", example_3_database_loading),
        ("Data Preprocessing", example_4_preprocessing),
        ("Streaming Data", example_5_streaming_data),
        ("Multi-Source Loading", example_6_multi_source_loading),
        ("Custom Data Formats", example_7_custom_data_formats)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-7): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()