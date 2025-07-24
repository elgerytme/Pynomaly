#!/usr/bin/env python3
"""
Fraud Detection Template
=======================

Ready-to-use template for detecting fraudulent transactions.
Copy this file and modify the data loading section for your use case.

Usage:
    python fraud_detection_template.py

Requirements:
    - pandas
    - numpy  
    - anomaly_detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from anomaly_detection import DetectionService, EnsembleService

def generate_sample_transaction_data(n_samples=10000):
    """
    Generate sample transaction data for testing.
    Replace this function with your actual data loading logic.
    """
    np.random.seed(42)
    
    # Normal transactions
    normal_count = int(n_samples * 0.99)
    normal_transactions = {
        'amount': np.random.lognormal(3, 1.5, normal_count),  # $20-300 typical
        'hour': np.random.choice(range(6, 23), normal_count),  # Business hours
        'merchant_category': np.random.choice([1, 2, 3, 4], normal_count, p=[0.4, 0.3, 0.2, 0.1]),
        'days_since_last': np.random.exponential(2, normal_count),  # Days between transactions
        'location_risk': np.random.beta(2, 8, normal_count),  # Low risk locations
    }
    
    # Fraudulent transactions  
    fraud_count = n_samples - normal_count
    fraud_transactions = {
        'amount': np.random.lognormal(6, 1, fraud_count),  # High amounts
        'hour': np.random.choice(range(24), fraud_count),  # Any time
        'merchant_category': np.random.choice([1, 2, 3, 4], fraud_count),
        'days_since_last': np.random.exponential(0.1, fraud_count),  # Rapid transactions
        'location_risk': np.random.beta(8, 2, fraud_count),  # High risk locations
    }
    
    # Combine data
    data = pd.DataFrame({
        'transaction_id': range(n_samples),
        'amount': np.concatenate([normal_transactions['amount'], fraud_transactions['amount']]),
        'hour': np.concatenate([normal_transactions['hour'], fraud_transactions['hour']]),
        'merchant_category': np.concatenate([normal_transactions['merchant_category'], fraud_transactions['merchant_category']]),
        'days_since_last': np.concatenate([normal_transactions['days_since_last'], fraud_transactions['days_since_last']]),
        'location_risk': np.concatenate([normal_transactions['location_risk'], fraud_transactions['location_risk']]),
        'is_fraud': [False] * normal_count + [True] * fraud_count
    })
    
    # Shuffle the data
    return data.sample(frac=1).reset_index(drop=True)

def load_your_data():
    """
    Replace this function with your actual data loading logic.
    
    Expected columns:
    - amount: Transaction amount
    - hour: Hour of day (0-23)
    - merchant_category: Category code
    - days_since_last: Days since last transaction
    - location_risk: Risk score for location (0-1)
    
    Returns:
        pd.DataFrame: Transaction data
    """
    # OPTION 1: Load from CSV
    # return pd.read_csv('transactions.csv')
    
    # OPTION 2: Load from database
    # import sqlalchemy
    # engine = sqlalchemy.create_engine('your_connection_string')
    # return pd.read_sql('SELECT * FROM transactions', engine)
    
    # OPTION 3: Use sample data for testing
    print("📝 Using sample data for demonstration")
    return generate_sample_transaction_data()

def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare features for anomaly detection.
    
    Args:
        df: Raw transaction data
        
    Returns:
        np.ndarray: Processed feature matrix
    """
    features = df.copy()
    
    # Feature engineering
    features['amount_log'] = np.log1p(features['amount'])
    features['is_weekend'] = (features['hour'] > 18) | (features['hour'] < 8)
    features['high_risk_category'] = features['merchant_category'].isin([3, 4])
    features['rapid_transaction'] = features['days_since_last'] < 0.1
    
    # Select features for detection
    feature_columns = [
        'amount_log',
        'hour', 
        'merchant_category',
        'days_since_last',
        'location_risk',
        'is_weekend',
        'high_risk_category', 
        'rapid_transaction'
    ]
    
    # Convert boolean columns to integers
    for col in ['is_weekend', 'high_risk_category', 'rapid_transaction']:
        features[col] = features[col].astype(int)
    
    return features[feature_columns].values

def detect_fraud_single_algorithm(features: np.ndarray, contamination: float = 0.01):
    """
    Detect fraud using a single algorithm.
    
    Args:
        features: Processed feature matrix
        contamination: Expected fraud rate
        
    Returns:
        DetectionResult: Anomaly detection results
    """
    print(f"🔍 Running Isolation Forest detection...")
    print(f"   Expected fraud rate: {contamination*100:.1f}%")
    
    service = DetectionService()
    result = service.detect(
        data=features,
        algorithm='isolation_forest',
        contamination=contamination,
        n_estimators=200,
        random_state=42
    )
    
    print(f"   ✅ Detection complete in {result.processing_time:.2f}s")
    return result

def detect_fraud_ensemble(features: np.ndarray, contamination: float = 0.01):
    """
    Detect fraud using ensemble methods for better accuracy.
    
    Args:
        features: Processed feature matrix
        contamination: Expected fraud rate
        
    Returns:
        DetectionResult: Ensemble detection results
    """
    print(f"🎯 Running Ensemble detection...")
    print(f"   Algorithms: Isolation Forest + LOF + One-Class SVM")
    print(f"   Expected fraud rate: {contamination*100:.1f}%")
    
    ensemble = EnsembleService()
    result = ensemble.detect(
        data=features,
        algorithms=['isolation_forest', 'lof', 'one_class_svm'],
        method='voting',
        contamination=contamination
    )
    
    print(f"   ✅ Ensemble detection complete in {result.processing_time:.2f}s")
    return result

def analyze_results(df: pd.DataFrame, result, method_name: str):
    """
    Analyze and display detection results.
    
    Args:
        df: Original data with true labels
        result: Detection results
        method_name: Name of detection method
    """
    print(f"\n📊 {method_name} Results:")
    print(f"{'='*50}")
    
    # Basic statistics
    total_transactions = len(df)
    detected_fraud = np.sum(result.predictions == -1)
    
    print(f"📈 Total transactions analyzed: {total_transactions:,}")
    print(f"🚨 Fraudulent transactions detected: {detected_fraud:,}")
    print(f"📊 Detection rate: {detected_fraud/total_transactions*100:.2f}%")
    
    # If we have true labels, calculate accuracy metrics
    if 'is_fraud' in df.columns:
        true_fraud = df['is_fraud'].sum()
        
        # Convert predictions to match true labels format
        predicted_fraud = result.predictions == -1
        actual_fraud = df['is_fraud'].values
        
        # Calculate metrics
        true_positives = np.sum(predicted_fraud & actual_fraud)
        false_positives = np.sum(predicted_fraud & ~actual_fraud)
        false_negatives = np.sum(~predicted_fraud & actual_fraud)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n🎯 Performance Metrics:")
        print(f"   True fraud cases: {true_fraud}")
        print(f"   Correctly detected: {true_positives}")
        print(f"   False positives: {false_positives}")
        print(f"   Missed fraud: {false_negatives}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
    
    # Show top suspicious transactions
    fraud_indices = np.where(result.predictions == -1)[0]
    if len(fraud_indices) > 0:
        print(f"\n🔥 Top 5 Most Suspicious Transactions:")
        top_suspicious = fraud_indices[np.argsort(result.scores[fraud_indices])[-5:]]
        
        for i, idx in enumerate(reversed(top_suspicious), 1):
            score = result.scores[idx]
            amount = df.iloc[idx]['amount']
            hour = df.iloc[idx]['hour']
            risk = df.iloc[idx]['location_risk']
            print(f"   {i}. Transaction {idx}: ${amount:.2f} at {hour:02d}:00, "
                  f"Risk: {risk:.3f}, Score: {score:.3f}")

def save_results(df: pd.DataFrame, result, output_file: str = 'fraud_detection_results.csv'):
    """
    Save detection results to CSV file.
    
    Args:
        df: Original data
        result: Detection results  
        output_file: Output filename
    """
    # Add results to dataframe
    output_df = df.copy()
    output_df['anomaly_score'] = result.scores
    output_df['is_detected_fraud'] = result.predictions == -1
    output_df['confidence'] = np.where(
        result.predictions == -1,
        'High' if result.scores.max() > 0.7 else 'Medium',
        'Low'
    )
    
    # Sort by anomaly score (highest first)
    output_df = output_df.sort_values('anomaly_score', ascending=False)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    print(f"   Total records: {len(output_df)}")
    print(f"   Fraud detected: {np.sum(output_df['is_detected_fraud'])}")

def main():
    """
    Main fraud detection pipeline.
    """
    print("🚨 Fraud Detection System")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📂 Loading transaction data...")
    df = load_your_data()
    print(f"   Loaded {len(df):,} transactions")
    
    # Step 2: Preprocess features
    print("\n🔧 Preprocessing features...")
    features = preprocess_features(df)
    print(f"   Created {features.shape[1]} features for {features.shape[0]} transactions")
    
    # Step 3: Detect fraud using single algorithm
    single_result = detect_fraud_single_algorithm(features)
    analyze_results(df, single_result, "Isolation Forest")
    
    # Step 4: Detect fraud using ensemble (more accurate)
    ensemble_result = detect_fraud_ensemble(features)  
    analyze_results(df, ensemble_result, "Ensemble Method")
    
    # Step 5: Save results (using ensemble results)
    save_results(df, ensemble_result)
    
    print(f"\n✅ Fraud detection complete!")
    print(f"💡 Tip: Review the saved results and adjust contamination rate if needed")

if __name__ == "__main__":
    main()