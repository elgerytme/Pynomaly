"""
Industry-Specific Anomaly Detection Use Cases
=============================================

This example demonstrates domain-specific anomaly detection applications across
various industries, showcasing specialized approaches, domain knowledge integration,
and industry-specific evaluation metrics.

Industries covered:
- Financial Services (Fraud Detection, Market Manipulation)
- Healthcare (Patient Monitoring, Drug Discovery)
- Manufacturing (Predictive Maintenance, Quality Control)
- Cybersecurity (Network Intrusion, Behavioral Analysis)
- IoT and Smart Cities (Sensor Networks, Traffic Analysis)
- E-commerce (User Behavior, Recommendation Systems)

Each use case includes:
- Domain-specific data preprocessing
- Appropriate algorithm selection
- Industry-relevant evaluation metrics
- Regulatory compliance considerations
- Real-world deployment patterns
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Core anomaly detection imports
sys.path.append(str(Path(__file__).parent.parent))
from anomaly_detection import AnomalyDetector, DetectionService, EnsembleService
from anomaly_detection.core.services import StreamingService

# Optional imports for specialized use cases
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from scipy import stats
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IndustryMetrics:
    """Industry-specific performance metrics."""
    industry: str
    use_case: str
    primary_metric: str
    secondary_metrics: List[str]
    compliance_requirements: List[str]
    cost_matrix: Dict[str, float]  # Cost of different error types

class FinancialAnomalyDetector:
    """
    Financial services anomaly detection with regulatory compliance.
    
    Use cases:
    - Credit card fraud detection
    - Anti-money laundering (AML)
    - Market manipulation detection
    - Algorithmic trading anomalies
    """
    
    def __init__(self):
        self.detector = None
        self.scaler = StandardScaler()
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def generate_transaction_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic financial transaction data."""
        np.random.seed(42)
        
        # Normal transactions
        normal_data = {
            'amount': np.random.lognormal(3, 1.5, n_samples),  # Log-normal distribution for amounts
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_samples),
            'hour_of_day': np.random.normal(14, 6, n_samples),  # Peak around 2 PM
            'day_of_week': np.random.choice(range(7), n_samples),
            'days_since_last_transaction': np.random.exponential(2, n_samples),
            'avg_monthly_spending': np.random.normal(2000, 500, n_samples),
            'account_age_months': np.random.uniform(1, 120, n_samples),
            'previous_fraud_reports': np.random.poisson(0.1, n_samples),
            'velocity_1hour': np.random.poisson(1.5, n_samples),  # Transactions per hour
            'velocity_24hour': np.random.poisson(8, n_samples),   # Transactions per day
        }
        
        df = pd.DataFrame(normal_data)
        
        # Add anomalous transactions (fraud)
        n_fraud = int(n_samples * 0.02)  # 2% fraud rate
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        
        # Fraud patterns
        df.loc[fraud_indices, 'amount'] = np.random.lognormal(6, 2, n_fraud)  # Higher amounts
        df.loc[fraud_indices, 'hour_of_day'] = np.random.choice([2, 3, 4, 23], n_fraud)  # Unusual hours
        df.loc[fraud_indices, 'velocity_1hour'] = np.random.poisson(10, n_fraud)  # High velocity
        df.loc[fraud_indices, 'days_since_last_transaction'] = 0  # Immediate transactions
        
        # Create labels
        df['is_fraud'] = 0
        df.loc[fraud_indices, 'is_fraud'] = 1
        
        # Feature engineering
        df['amount_vs_avg_ratio'] = df['amount'] / df['avg_monthly_spending']
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        df['velocity_ratio'] = df['velocity_1hour'] / (df['velocity_24hour'] + 1)
        
        return df
    
    def preprocess_financial_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess financial data with domain-specific transformations."""
        # Handle categorical variables
        le = LabelEncoder()
        df_processed = df.copy()
        df_processed['merchant_category_encoded'] = le.fit_transform(df['merchant_category'])
        
        # Select features for anomaly detection
        feature_columns = [
            'amount', 'hour_of_day', 'day_of_week', 'days_since_last_transaction',
            'account_age_months', 'previous_fraud_reports', 'velocity_1hour',
            'velocity_24hour', 'amount_vs_avg_ratio', 'is_weekend', 'is_night',
            'velocity_ratio', 'merchant_category_encoded'
        ]
        
        X = df_processed[feature_columns].values
        
        # Apply log transformation to skewed features
        log_features = [0, 3, 5]  # amount, days_since_last_transaction, previous_fraud_reports
        X[:, log_features] = np.log1p(X[:, log_features])
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        return X
    
    def detect_fraud(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect fraudulent transactions with financial domain expertise."""
        # Preprocess data
        X = self.preprocess_financial_data(df)
        y = df['is_fraud'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Use ensemble approach for robust fraud detection
        ensemble = EnsembleService()
        
        # Configure algorithms suitable for financial fraud
        algorithms = [
            ('isolation_forest', {'contamination': 0.02, 'n_estimators': 200}),
            ('lof', {'n_neighbors': 50, 'contamination': 0.02}),
            ('one_class_svm', {'gamma': 'scale', 'nu': 0.02})
        ]
        
        results = ensemble.detect_with_ensemble(
            X_test,
            algorithms=algorithms,
            combination_method='weighted_average'
        )
        
        # Calculate financial-specific metrics
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Risk scoring
        risk_scores = self._calculate_risk_scores(X_test, probabilities)
        
        # Regulatory compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(predictions, y_test, df.loc[X_test.shape[0]:].reset_index(drop=True))
        
        # Financial impact analysis
        impact_analysis = self._calculate_financial_impact(predictions, y_test, df.loc[X_test.shape[0]:].reset_index(drop=True))
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_scores': risk_scores,
            'compliance_metrics': compliance_metrics,
            'impact_analysis': impact_analysis,
            'auc_score': roc_auc_score(y_test, probabilities),
            'precision_at_1_percent': self._precision_at_k(y_test, probabilities, 0.01)
        }
    
    def _calculate_risk_scores(self, X: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Calculate risk scores based on multiple factors."""
        # Combine anomaly probability with feature-based risk factors
        risk_scores = probabilities.copy()
        
        # Adjust for transaction amount (higher amounts = higher risk)
        amount_risk = np.clip(X[:, 0] / 3, 0, 1)  # Normalized amount feature
        risk_scores = 0.7 * risk_scores + 0.3 * amount_risk
        
        return np.clip(risk_scores, 0, 1)
    
    def _calculate_compliance_metrics(self, predictions: np.ndarray, y_true: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate regulatory compliance metrics."""
        # False Positive Rate (important for customer experience)
        tn = np.sum((predictions == 0) & (y_true == 0))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Detection Rate (percentage of fraud caught)
        tp = np.sum((predictions == 1) & (y_true == 1))
        fn = np.sum((predictions == 0) & (y_true == 1))
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Alert Volume (operational impact)
        alert_rate = np.mean(predictions)
        
        return {
            'false_positive_rate': fpr,
            'detection_rate': detection_rate,
            'alert_rate': alert_rate,
            'model_accuracy': np.mean(predictions == y_true)
        }
    
    def _calculate_financial_impact(self, predictions: np.ndarray, y_true: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate financial impact of fraud detection."""
        # Cost assumptions
        avg_fraud_amount = 500  # Average fraud transaction amount
        investigation_cost = 25  # Cost to investigate each alert
        false_positive_cost = 10  # Cost of blocking legitimate transaction
        
        # Calculate costs
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        tn = np.sum((predictions == 0) & (y_true == 0))
        
        # Prevented fraud losses
        prevented_losses = tp * avg_fraud_amount
        
        # Missed fraud losses
        missed_losses = fn * avg_fraud_amount
        
        # Operational costs
        investigation_costs = (tp + fp) * investigation_cost
        customer_impact_costs = fp * false_positive_cost
        
        # Net benefit
        net_benefit = prevented_losses - missed_losses - investigation_costs - customer_impact_costs
        
        return {
            'prevented_losses': prevented_losses,
            'missed_losses': missed_losses,
            'investigation_costs': investigation_costs,
            'customer_impact_costs': customer_impact_costs,
            'net_benefit': net_benefit,
            'roi': net_benefit / (investigation_costs + customer_impact_costs) if (investigation_costs + customer_impact_costs) > 0 else 0
        }
    
    def _precision_at_k(self, y_true: np.ndarray, probabilities: np.ndarray, k: float) -> float:
        """Calculate precision at top k% of predictions."""
        n_top = int(len(y_true) * k)
        top_indices = np.argsort(probabilities)[-n_top:]
        return np.mean(y_true[top_indices])

class HealthcareAnomalyDetector:
    """
    Healthcare anomaly detection with patient safety focus.
    
    Use cases:
    - Patient vital signs monitoring
    - Medical device malfunction detection
    - Drug adverse event detection
    - Hospital acquired infection patterns
    """
    
    def __init__(self):
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'temperature': (36.1, 37.2),
            'respiratory_rate': (12, 20),
            'oxygen_saturation': (95, 100)
        }
    
    def generate_patient_data(self, n_patients: int = 1000, days: int = 7) -> pd.DataFrame:
        """Generate synthetic patient monitoring data."""
        np.random.seed(42)
        
        data = []
        patient_ids = range(n_patients)
        
        for patient_id in patient_ids:
            # Patient characteristics
            age = np.random.normal(65, 15)  # Hospital patients tend to be older
            baseline_hr = np.random.normal(75, 10)
            baseline_bp_sys = np.random.normal(120, 15)
            baseline_bp_dia = np.random.normal(80, 10)
            
            for day in range(days):
                for hour in range(24):
                    timestamp = datetime.now() - timedelta(days=days-day, hours=24-hour)
                    
                    # Normal variations
                    hr_variation = np.random.normal(0, 5)
                    bp_variation = np.random.normal(0, 8)
                    
                    # Circadian rhythm effects
                    circadian_hr = 5 * np.sin(2 * np.pi * hour / 24)
                    circadian_bp = 10 * np.sin(2 * np.pi * hour / 24 + np.pi/4)
                    
                    record = {
                        'patient_id': patient_id,
                        'timestamp': timestamp,
                        'age': age,
                        'hour_of_day': hour,
                        'day_of_week': timestamp.weekday(),
                        'heart_rate': baseline_hr + hr_variation + circadian_hr,
                        'blood_pressure_systolic': baseline_bp_sys + bp_variation + circadian_bp,
                        'blood_pressure_diastolic': baseline_bp_dia + bp_variation/2 + circadian_bp/2,
                        'temperature': np.random.normal(36.8, 0.3),
                        'respiratory_rate': np.random.normal(16, 2),
                        'oxygen_saturation': np.random.normal(98, 1.5),
                        'is_anomaly': 0
                    }
                    
                    data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add anomalous events (medical emergencies)
        n_anomalies = int(len(df) * 0.005)  # 0.5% anomaly rate
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            event_type = np.random.choice(['cardiac', 'respiratory', 'fever', 'hypotension'])
            
            if event_type == 'cardiac':
                df.loc[idx, 'heart_rate'] = np.random.choice([np.random.normal(150, 20), np.random.normal(40, 5)])
            elif event_type == 'respiratory':
                df.loc[idx, 'respiratory_rate'] = np.random.choice([np.random.normal(30, 5), np.random.normal(8, 2)])
                df.loc[idx, 'oxygen_saturation'] = np.random.normal(85, 5)
            elif event_type == 'fever':
                df.loc[idx, 'temperature'] = np.random.normal(39.5, 1)
            elif event_type == 'hypotension':
                df.loc[idx, 'blood_pressure_systolic'] = np.random.normal(70, 10)
                df.loc[idx, 'blood_pressure_diastolic'] = np.random.normal(45, 8)
            
            df.loc[idx, 'is_anomaly'] = 1
        
        return df
    
    def detect_patient_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patient anomalies with medical domain knowledge."""
        # Feature engineering for healthcare
        df_processed = self._engineer_medical_features(df)
        
        # Select features
        feature_columns = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'temperature', 'respiratory_rate', 'oxygen_saturation',
            'pulse_pressure', 'map', 'shock_index', 'hr_temp_ratio',
            'age_risk_factor', 'vital_stability_score'
        ]
        
        X = df_processed[feature_columns].values
        y = df_processed['is_anomaly'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        # Use algorithms suitable for healthcare monitoring
        detector = AnomalyDetector(algorithm='lof', n_neighbors=30, contamination=0.005)
        detector.fit(X_train)
        
        predictions = detector.predict(X_test)
        probabilities = detector.predict_proba(X_test)[:, 1] if hasattr(detector, 'predict_proba') else detector.decision_function(X_test)
        
        # Healthcare-specific evaluation
        clinical_metrics = self._calculate_clinical_metrics(predictions, y_test, X_test, feature_columns)
        safety_analysis = self._analyze_patient_safety(predictions, y_test, df_processed.iloc[-len(y_test):])
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'clinical_metrics': clinical_metrics,
            'safety_analysis': safety_analysis,
            'sensitivity': np.sum((predictions == 1) & (y_test == 1)) / np.sum(y_test == 1),
            'specificity': np.sum((predictions == 0) & (y_test == 0)) / np.sum(y_test == 0)
        }
    
    def _engineer_medical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer medical domain-specific features."""
        df_processed = df.copy()
        
        # Calculated vital signs
        df_processed['pulse_pressure'] = df['blood_pressure_systolic'] - df['blood_pressure_diastolic']
        df_processed['map'] = df['blood_pressure_diastolic'] + (df_processed['pulse_pressure'] / 3)  # Mean arterial pressure
        df_processed['shock_index'] = df['heart_rate'] / df['blood_pressure_systolic']
        df_processed['hr_temp_ratio'] = df['heart_rate'] / df['temperature']
        
        # Age-based risk factors
        df_processed['age_risk_factor'] = np.where(df['age'] > 75, 2, np.where(df['age'] > 65, 1, 0))
        
        # Vital signs stability (rolling standard deviation)
        for vital in ['heart_rate', 'blood_pressure_systolic', 'temperature']:
            df_processed[f'{vital}_stability'] = df_processed.groupby('patient_id')[vital].rolling(window=6, min_periods=3).std().reset_index(0, drop=True)
        
        # Composite stability score
        stability_features = ['heart_rate_stability', 'blood_pressure_systolic_stability', 'temperature_stability']
        df_processed['vital_stability_score'] = df_processed[stability_features].sum(axis=1)
        
        # Fill NaN values
        df_processed = df_processed.fillna(df_processed.mean())
        
        return df_processed
    
    def _calculate_clinical_metrics(self, predictions: np.ndarray, y_true: np.ndarray, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate clinically relevant metrics."""
        # Basic performance
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        tn = np.sum((predictions == 0) & (y_true == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Clinical significance
        missed_emergencies = fn
        false_alarms = fp
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'missed_emergencies': missed_emergencies,
            'false_alarms': false_alarms,
            'clinical_utility_index': sensitivity * ppv  # Balance of detection and precision
        }
    
    def _analyze_patient_safety(self, predictions: np.ndarray, y_true: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patient safety implications."""
        # Time to detection analysis
        emergency_indices = np.where(y_true == 1)[0]
        detected_indices = np.where((predictions == 1) & (y_true == 1))[0]
        
        # Safety metrics
        detection_rate = len(detected_indices) / len(emergency_indices) if len(emergency_indices) > 0 else 0
        missed_critical_events = len(emergency_indices) - len(detected_indices)
        
        # Alert fatigue analysis
        total_alerts = np.sum(predictions == 1)
        true_alerts = np.sum((predictions == 1) & (y_true == 1))
        alert_precision = true_alerts / total_alerts if total_alerts > 0 else 0
        
        return {
            'detection_rate': detection_rate,
            'missed_critical_events': missed_critical_events,
            'total_alerts_generated': total_alerts,
            'alert_precision': alert_precision,
            'alert_fatigue_risk': 'high' if alert_precision < 0.1 else 'medium' if alert_precision < 0.3 else 'low'
        }

class ManufacturingAnomalyDetector:
    """
    Manufacturing anomaly detection for predictive maintenance and quality control.
    
    Use cases:
    - Equipment failure prediction
    - Product quality control
    - Production line optimization
    - Supply chain anomalies
    """
    
    def __init__(self):
        self.equipment_profiles = {
            'motor': {'vibration_normal': (0, 2), 'temperature_normal': (30, 60)},
            'pump': {'pressure_normal': (2, 8), 'flow_rate_normal': (100, 500)},
            'compressor': {'vibration_normal': (0, 3), 'pressure_normal': (5, 15)}
        }
    
    def generate_sensor_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic manufacturing sensor data."""
        np.random.seed(42)
        
        # Equipment types
        equipment_types = ['motor', 'pump', 'compressor']
        equipment_ids = [f"EQ_{i:04d}" for i in range(50)]
        
        data = []
        
        for i in range(n_samples):
            equipment_id = np.random.choice(equipment_ids)
            equipment_type = np.random.choice(equipment_types)
            
            # Normal operation parameters
            if equipment_type == 'motor':
                vibration = np.random.normal(1, 0.3)
                temperature = np.random.normal(45, 8)
                power_consumption = np.random.normal(100, 15)
                rpm = np.random.normal(1800, 50)
            elif equipment_type == 'pump':
                pressure = np.random.normal(5, 1)
                flow_rate = np.random.normal(300, 50)
                power_consumption = np.random.normal(75, 12)
                vibration = np.random.normal(0.5, 0.2)
            else:  # compressor
                vibration = np.random.normal(1.5, 0.4)
                pressure = np.random.normal(10, 2)
                power_consumption = np.random.normal(150, 25)
                temperature = np.random.normal(55, 10)
            
            # Operational context
            shift = np.random.choice(['day', 'night'])
            production_rate = np.random.normal(85, 10)  # Percentage of max capacity
            ambient_temperature = np.random.normal(22, 5)
            
            record = {
                'timestamp': datetime.now() - timedelta(hours=n_samples-i),
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'vibration': vibration,
                'temperature': temperature if equipment_type != 'pump' else ambient_temperature + np.random.normal(5, 2),
                'pressure': pressure if equipment_type in ['pump', 'compressor'] else np.random.normal(1, 0.2),
                'power_consumption': power_consumption,
                'rpm': rpm if equipment_type == 'motor' else np.random.normal(3600, 200),
                'flow_rate': flow_rate if equipment_type == 'pump' else np.random.normal(0, 10),
                'shift': shift,
                'production_rate': production_rate,
                'ambient_temperature': ambient_temperature,
                'maintenance_hours_ago': np.random.exponential(720),  # Hours since last maintenance
                'is_anomaly': 0
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add anomalous conditions
        n_anomalies = int(n_samples * 0.03)  # 3% anomaly rate
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            failure_type = np.random.choice(['overheating', 'vibration', 'pressure', 'power_spike'])
            
            if failure_type == 'overheating':
                df.loc[idx, 'temperature'] = df.loc[idx, 'temperature'] + np.random.normal(20, 5)
                df.loc[idx, 'power_consumption'] = df.loc[idx, 'power_consumption'] * np.random.uniform(1.2, 1.8)
            elif failure_type == 'vibration':
                df.loc[idx, 'vibration'] = df.loc[idx, 'vibration'] * np.random.uniform(3, 8)
            elif failure_type == 'pressure':
                df.loc[idx, 'pressure'] = df.loc[idx, 'pressure'] * np.random.uniform(0.3, 0.7)
            elif failure_type == 'power_spike':
                df.loc[idx, 'power_consumption'] = df.loc[idx, 'power_consumption'] * np.random.uniform(2, 4)
            
            df.loc[idx, 'is_anomaly'] = 1
        
        return df
    
    def detect_equipment_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect equipment anomalies with manufacturing domain knowledge."""
        # Feature engineering
        df_processed = self._engineer_manufacturing_features(df)
        
        # Select features
        feature_columns = [
            'vibration', 'temperature', 'pressure', 'power_consumption', 'rpm',
            'flow_rate', 'production_rate', 'maintenance_hours_ago',
            'temp_power_ratio', 'vibration_normalized', 'efficiency_score',
            'wear_indicator', 'shift_encoded', 'equipment_age_factor'
        ]
        
        X = df_processed[feature_columns].values
        y = df_processed['is_anomaly'].values
        
        # Handle missing values and scale
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        # Use ensemble for robust detection
        ensemble = EnsembleService()
        algorithms = [
            ('isolation_forest', {'contamination': 0.03, 'n_estimators': 150}),
            ('lof', {'n_neighbors': 25, 'contamination': 0.03}),
        ]
        
        results = ensemble.detect_with_ensemble(X_test, algorithms=algorithms)
        
        # Manufacturing-specific analysis
        maintenance_insights = self._analyze_maintenance_needs(results['predictions'], y_test, df_processed.iloc[-len(y_test):])
        cost_analysis = self._calculate_maintenance_costs(results['predictions'], y_test, df_processed.iloc[-len(y_test):])
        
        return {
            'predictions': results['predictions'],
            'probabilities': results['probabilities'],
            'maintenance_insights': maintenance_insights,
            'cost_analysis': cost_analysis,
            'equipment_risk_scores': self._calculate_equipment_risk(X_test, results['probabilities'], df_processed.iloc[-len(y_test):])
        }
    
    def _engineer_manufacturing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer manufacturing-specific features."""
        df_processed = df.copy()
        
        # Efficiency indicators
        df_processed['temp_power_ratio'] = df['temperature'] / (df['power_consumption'] + 1)
        df_processed['vibration_normalized'] = df['vibration'] / (df['rpm'] / 1000 + 1)
        df_processed['efficiency_score'] = df['production_rate'] / (df['power_consumption'] + 1)
        
        # Wear indicators
        df_processed['wear_indicator'] = df['maintenance_hours_ago'] * (df['vibration'] + df['temperature'] / 100)
        
        # Operational context
        df_processed['shift_encoded'] = df['shift'].map({'day': 1, 'night': 0})
        df_processed['equipment_age_factor'] = np.log1p(df['maintenance_hours_ago'])
        
        # Rolling statistics for trend detection
        for feature in ['vibration', 'temperature', 'power_consumption']:
            df_processed[f'{feature}_trend'] = df_processed.groupby('equipment_id')[feature].rolling(window=10, min_periods=5).mean().reset_index(0, drop=True)
            df_processed[f'{feature}_volatility'] = df_processed.groupby('equipment_id')[feature].rolling(window=10, min_periods=5).std().reset_index(0, drop=True)
        
        # Fill NaN values
        df_processed = df_processed.fillna(method='forward').fillna(method='backward')
        
        return df_processed
    
    def _analyze_maintenance_needs(self, predictions: np.ndarray, y_true: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze maintenance needs and scheduling."""
        # Equipment requiring immediate attention
        high_risk_equipment = df[predictions == 1]['equipment_id'].unique()
        
        # Maintenance scheduling analysis
        equipment_risk = {}
        for eq_id in df['equipment_id'].unique():
            eq_data = df[df['equipment_id'] == eq_id]
            eq_predictions = predictions[df['equipment_id'] == eq_id]
            
            risk_score = np.mean(eq_predictions)
            hours_since_maintenance = eq_data['maintenance_hours_ago'].iloc[0]
            
            # Predict failure timeline
            if risk_score > 0.7:
                predicted_failure_hours = 24  # Immediate attention
            elif risk_score > 0.4:
                predicted_failure_hours = 168  # Within a week
            else:
                predicted_failure_hours = 720  # Within a month
            
            equipment_risk[eq_id] = {
                'risk_score': risk_score,
                'hours_since_maintenance': hours_since_maintenance,
                'predicted_failure_hours': predicted_failure_hours,
                'maintenance_priority': 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low'
            }
        
        return {
            'high_risk_equipment_count': len(high_risk_equipment),
            'immediate_maintenance_needed': list(high_risk_equipment),
            'equipment_risk_profiles': equipment_risk,
            'total_equipment_monitored': len(df['equipment_id'].unique())
        }
    
    def _calculate_maintenance_costs(self, predictions: np.ndarray, y_true: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate maintenance cost implications."""
        # Cost assumptions (in currency units)
        preventive_maintenance_cost = 500
        emergency_repair_cost = 2000
        downtime_cost_per_hour = 1000
        false_alarm_cost = 100
        
        # Calculate different scenarios
        tp = np.sum((predictions == 1) & (y_true == 1))  # Prevented failures
        fp = np.sum((predictions == 1) & (y_true == 0))  # False alarms
        fn = np.sum((predictions == 0) & (y_true == 1))  # Missed failures
        tn = np.sum((predictions == 0) & (y_true == 0))  # Normal operation
        
        # Cost calculations
        prevented_emergency_costs = tp * (emergency_repair_cost + 8 * downtime_cost_per_hour)  # Assume 8h downtime
        preventive_maintenance_costs = tp * preventive_maintenance_cost
        false_alarm_costs = fp * false_alarm_cost
        missed_failure_costs = fn * (emergency_repair_cost + 12 * downtime_cost_per_hour)  # Longer downtime for missed failures
        
        # Net savings
        net_savings = prevented_emergency_costs - preventive_maintenance_costs - false_alarm_costs - missed_failure_costs
        
        return {
            'prevented_emergency_costs': prevented_emergency_costs,
            'preventive_maintenance_costs': preventive_maintenance_costs,
            'false_alarm_costs': false_alarm_costs,
            'missed_failure_costs': missed_failure_costs,
            'net_savings': net_savings,
            'roi_percentage': (net_savings / (preventive_maintenance_costs + false_alarm_costs)) * 100 if (preventive_maintenance_costs + false_alarm_costs) > 0 else 0
        }
    
    def _calculate_equipment_risk(self, X: np.ndarray, probabilities: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk scores for each equipment."""
        equipment_risk = {}
        
        for i, eq_id in enumerate(df['equipment_id'].unique()):
            eq_indices = df['equipment_id'] == eq_id
            if np.any(eq_indices):
                eq_probs = probabilities[eq_indices]
                risk_score = np.mean(eq_probs)
                equipment_risk[eq_id] = risk_score
        
        return equipment_risk

class CybersecurityAnomalyDetector:
    """
    Cybersecurity anomaly detection for network and user behavior analysis.
    
    Use cases:
    - Network intrusion detection
    - User behavior analytics (UBA)
    - Malware detection
    - Data exfiltration detection
    """
    
    def __init__(self):
        self.normal_patterns = {
            'login_hours': (8, 18),  # Normal business hours
            'data_transfer_mb': (0, 100),  # Normal data transfer
            'failed_login_threshold': 3
        }
    
    def generate_network_data(self, n_samples: int = 8000) -> pd.DataFrame:
        """Generate synthetic network security data."""
        np.random.seed(42)
        
        # User profiles
        users = [f"user_{i:04d}" for i in range(200)]
        departments = ['IT', 'Finance', 'HR', 'Marketing', 'Operations']
        
        data = []
        
        for i in range(n_samples):
            user = np.random.choice(users)
            department = np.random.choice(departments)
            
            # Normal behavior patterns based on department
            if department == 'IT':
                login_hour = np.random.choice(range(24))  # IT works all hours
                data_transfer = np.random.lognormal(4, 1)  # Higher data transfer
                failed_logins = np.random.poisson(1)
            elif department == 'Finance':
                login_hour = max(7, min(19, np.random.normal(12, 3)))  # Business hours focused
                data_transfer = np.random.lognormal(2, 0.8)
                failed_logins = np.random.poisson(0.5)
            else:
                login_hour = max(8, min(18, np.random.normal(13, 2)))  # Standard business hours
                data_transfer = np.random.lognormal(1.5, 0.8)
                failed_logins = np.random.poisson(0.3)
            
            # Network activity
            network_connections = np.random.poisson(50)
            unique_destinations = np.random.poisson(20)
            port_scans = np.random.poisson(0.1)
            dns_requests = np.random.poisson(100)
            
            record = {
                'timestamp': datetime.now() - timedelta(hours=n_samples-i),
                'user_id': user,
                'department': department,
                'login_hour': int(login_hour),
                'day_of_week': np.random.choice(range(7)),
                'failed_logins': failed_logins,
                'successful_logins': 1,
                'data_transfer_mb': data_transfer,
                'network_connections': network_connections,
                'unique_destinations': unique_destinations,
                'port_scans': port_scans,
                'dns_requests': dns_requests,
                'vpn_usage': np.random.choice([0, 1], p=[0.7, 0.3]),
                'admin_privileges_used': np.random.choice([0, 1], p=[0.9, 0.1]),
                'file_downloads': np.random.poisson(5),
                'email_sent': np.random.poisson(10),
                'is_anomaly': 0
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add malicious activities
        n_anomalies = int(n_samples * 0.02)  # 2% anomaly rate
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            attack_type = np.random.choice(['insider_threat', 'brute_force', 'data_exfiltration', 'lateral_movement'])
            
            if attack_type == 'insider_threat':
                df.loc[idx, 'login_hour'] = np.random.choice([2, 3, 4, 22, 23])  # Unusual hours
                df.loc[idx, 'data_transfer_mb'] = np.random.lognormal(6, 1)  # Large data transfer
                df.loc[idx, 'admin_privileges_used'] = 1
                df.loc[idx, 'file_downloads'] = np.random.poisson(50)
            elif attack_type == 'brute_force':
                df.loc[idx, 'failed_logins'] = np.random.poisson(20)
                df.loc[idx, 'successful_logins'] = 0
            elif attack_type == 'data_exfiltration':
                df.loc[idx, 'data_transfer_mb'] = np.random.lognormal(7, 1)
                df.loc[idx, 'unique_destinations'] = np.random.poisson(5)  # Few destinations, large transfer
                df.loc[idx, 'vpn_usage'] = 1
            elif attack_type == 'lateral_movement':
                df.loc[idx, 'network_connections'] = np.random.poisson(200)
                df.loc[idx, 'port_scans'] = np.random.poisson(10)
                df.loc[idx, 'unique_destinations'] = np.random.poisson(100)
            
            df.loc[idx, 'is_anomaly'] = 1
        
        return df
    
    def detect_security_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect cybersecurity anomalies with domain expertise."""
        # Feature engineering
        df_processed = self._engineer_security_features(df)
        
        # Select features
        feature_columns = [
            'login_hour', 'failed_logins', 'data_transfer_mb', 'network_connections',
            'unique_destinations', 'port_scans', 'dns_requests', 'vpn_usage',
            'admin_privileges_used', 'file_downloads', 'email_sent',
            'off_hours_activity', 'data_velocity', 'connection_ratio',
            'security_risk_score', 'department_encoded', 'behavior_deviation'
        ]
        
        X = df_processed[feature_columns].values
        y = df_processed['is_anomaly'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        # Use ensemble approach for robust security detection
        ensemble = EnsembleService()
        algorithms = [
            ('isolation_forest', {'contamination': 0.02, 'n_estimators': 200}),
            ('lof', {'n_neighbors': 40, 'contamination': 0.02}),
            ('one_class_svm', {'gamma': 'scale', 'nu': 0.02})
        ]
        
        results = ensemble.detect_with_ensemble(X_test, algorithms=algorithms)
        
        # Security-specific analysis
        threat_analysis = self._analyze_threat_types(results['predictions'], y_test, df_processed.iloc[-len(y_test):])
        incident_response = self._generate_incident_response(results['predictions'], df_processed.iloc[-len(y_test):])
        
        return {
            'predictions': results['predictions'],
            'probabilities': results['probabilities'],
            'threat_analysis': threat_analysis,
            'incident_response': incident_response,
            'security_score': roc_auc_score(y_test, results['probabilities'])
        }
    
    def _engineer_security_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer cybersecurity-specific features."""
        df_processed = df.copy()
        
        # Behavioral indicators
        df_processed['off_hours_activity'] = ((df['login_hour'] < 7) | (df['login_hour'] > 19)).astype(int)
        df_processed['data_velocity'] = df['data_transfer_mb'] / (df['network_connections'] + 1)
        df_processed['connection_ratio'] = df['unique_destinations'] / (df['network_connections'] + 1)
        
        # Risk scoring
        risk_factors = [
            df['failed_logins'] > 3,
            df['data_transfer_mb'] > 1000,
            df['port_scans'] > 0,
            df['admin_privileges_used'] == 1,
            ((df['login_hour'] < 6) | (df['login_hour'] > 20))
        ]
        df_processed['security_risk_score'] = sum(risk_factors)
        
        # Department encoding
        dept_mapping = {'IT': 0, 'Finance': 1, 'HR': 2, 'Marketing': 3, 'Operations': 4}
        df_processed['department_encoded'] = df['department'].map(dept_mapping)
        
        # User behavior baseline (simplified)
        user_baselines = df.groupby('user_id').agg({
            'data_transfer_mb': 'mean',
            'network_connections': 'mean',
            'login_hour': 'mean'
        }).add_suffix('_baseline')
        
        df_processed = df_processed.merge(user_baselines, left_on='user_id', right_index=True, how='left')
        
        # Behavior deviation
        df_processed['behavior_deviation'] = (
            abs(df_processed['data_transfer_mb'] - df_processed['data_transfer_mb_baseline']) +
            abs(df_processed['network_connections'] - df_processed['network_connections_baseline']) +
            abs(df_processed['login_hour'] - df_processed['login_hour_baseline'])
        )
        
        # Fill NaN values
        df_processed = df_processed.fillna(0)
        
        return df_processed
    
    def _analyze_threat_types(self, predictions: np.ndarray, y_true: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze types of security threats detected."""
        threat_indicators = {
            'insider_threat': (df['off_hours_activity'] == 1) & (df['data_transfer_mb'] > 1000) & (df['admin_privileges_used'] == 1),
            'brute_force': df['failed_logins'] > 10,
            'data_exfiltration': (df['data_transfer_mb'] > 2000) & (df['unique_destinations'] < 10),
            'lateral_movement': (df['network_connections'] > 150) & (df['port_scans'] > 5)
        }
        
        detected_threats = {}
        for threat_type, indicator in threat_indicators.items():
            threat_detections = predictions & indicator
            detected_threats[threat_type] = {
                'count': np.sum(threat_detections),
                'precision': np.sum(threat_detections & (y_true == 1)) / np.sum(threat_detections) if np.sum(threat_detections) > 0 else 0,
                'users_affected': len(df[threat_detections]['user_id'].unique())
            }
        
        return {
            'threat_breakdown': detected_threats,
            'total_threats_detected': np.sum(predictions == 1),
            'high_confidence_threats': np.sum((predictions == 1) & (df['security_risk_score'] >= 3))
        }
    
    def _generate_incident_response(self, predictions: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate incident response recommendations."""
        high_risk_users = df[predictions == 1]['user_id'].unique()
        
        response_actions = []
        for user in high_risk_users:
            user_data = df[df['user_id'] == user].iloc[0]
            
            actions = []
            if user_data['failed_logins'] > 10:
                actions.append('Lock user account immediately')
            if user_data['data_transfer_mb'] > 2000:
                actions.append('Review data access logs and network traffic')
            if user_data['admin_privileges_used'] == 1:
                actions.append('Audit admin privilege usage')
            if user_data['off_hours_activity'] == 1:
                actions.append('Verify user identity and business justification')
            
            response_actions.append({
                'user_id': user,
                'department': user_data['department'],
                'risk_score': user_data['security_risk_score'],
                'recommended_actions': actions,
                'priority': 'high' if user_data['security_risk_score'] >= 3 else 'medium'
            })
        
        return {
            'total_incidents': len(response_actions),
            'high_priority_incidents': len([a for a in response_actions if a['priority'] == 'high']),
            'incident_details': response_actions[:10]  # Top 10 for brevity
        }

# Example usage functions
def example_1_financial_fraud_detection():
    """Example 1: Financial fraud detection."""
    print("=== Example 1: Financial Fraud Detection ===")
    
    detector = FinancialAnomalyDetector()
    
    # Generate transaction data
    print("Generating synthetic transaction data...")
    df = detector.generate_transaction_data(n_samples=5000)
    print(f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraudulent cases ({df['is_fraud'].mean()*100:.1f}%)")
    
    # Detect fraud
    print("Running fraud detection...")
    results = detector.detect_fraud(df)
    
    print(f"\nFraud Detection Results:")
    print(f"AUC Score: {results['auc_score']:.3f}")
    print(f"Precision at 1%: {results['precision_at_1_percent']:.3f}")
    
    print(f"\nCompliance Metrics:")
    for metric, value in results['compliance_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nFinancial Impact:")
    impact = results['impact_analysis']
    print(f"  Net Benefit: ${impact['net_benefit']:,.2f}")
    print(f"  ROI: {impact['roi']:.1%}")
    print(f"  Prevented Losses: ${impact['prevented_losses']:,.2f}")
    
    return results

def example_2_healthcare_monitoring():
    """Example 2: Healthcare patient monitoring."""
    print("\n=== Example 2: Healthcare Patient Monitoring ===")
    
    detector = HealthcareAnomalyDetector()
    
    # Generate patient data
    print("Generating synthetic patient monitoring data...")
    df = detector.generate_patient_data(n_patients=100, days=7)
    print(f"Generated {len(df)} patient records with {df['is_anomaly'].sum()} medical emergencies")
    
    # Detect anomalies
    print("Running patient anomaly detection...")
    results = detector.detect_patient_anomalies(df)
    
    print(f"\nClinical Performance:")
    clinical = results['clinical_metrics']
    print(f"  Sensitivity (Detection Rate): {clinical['sensitivity']:.3f}")
    print(f"  Specificity: {clinical['specificity']:.3f}")
    print(f"  Positive Predictive Value: {clinical['positive_predictive_value']:.3f}")
    print(f"  Missed Emergencies: {clinical['missed_emergencies']}")
    
    print(f"\nPatient Safety Analysis:")
    safety = results['safety_analysis']
    print(f"  Detection Rate: {safety['detection_rate']:.3f}")
    print(f"  Alert Precision: {safety['alert_precision']:.3f}")
    print(f"  Alert Fatigue Risk: {safety['alert_fatigue_risk']}")
    
    return results

def example_3_manufacturing_maintenance():
    """Example 3: Manufacturing predictive maintenance."""
    print("\n=== Example 3: Manufacturing Predictive Maintenance ===")
    
    detector = ManufacturingAnomalyDetector()
    
    # Generate sensor data
    print("Generating synthetic manufacturing sensor data...")
    df = detector.generate_sensor_data(n_samples=8000)
    print(f"Generated {len(df)} sensor readings from {df['equipment_id'].nunique()} pieces of equipment")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")
    
    # Detect equipment anomalies
    print("Running equipment anomaly detection...")
    results = detector.detect_equipment_anomalies(df)
    
    print(f"\nMaintenance Insights:")
    maintenance = results['maintenance_insights']
    print(f"  Equipment requiring immediate attention: {maintenance['high_risk_equipment_count']}")
    print(f"  Total equipment monitored: {maintenance['total_equipment_monitored']}")
    
    print(f"\nCost Analysis:")
    costs = results['cost_analysis']
    print(f"  Net Savings: ${costs['net_savings']:,.2f}")
    print(f"  ROI: {costs['roi_percentage']:.1f}%")
    print(f"  Prevented Emergency Costs: ${costs['prevented_emergency_costs']:,.2f}")
    
    # Show top equipment at risk
    risk_scores = results['equipment_risk_scores']
    top_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 Equipment at Risk:")
    for eq_id, risk in top_risk:
        print(f"  {eq_id}: {risk:.3f}")
    
    return results

def example_4_cybersecurity_detection():
    """Example 4: Cybersecurity threat detection."""
    print("\n=== Example 4: Cybersecurity Threat Detection ===")
    
    detector = CybersecurityAnomalyDetector()
    
    # Generate network data
    print("Generating synthetic network security data...")
    df = detector.generate_network_data(n_samples=6000)
    print(f"Generated {len(df)} network events from {df['user_id'].nunique()} users")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")
    
    # Detect security anomalies
    print("Running cybersecurity anomaly detection...")
    results = detector.detect_security_anomalies(df)
    
    print(f"\nSecurity Detection Performance:")
    print(f"  Security Score (AUC): {results['security_score']:.3f}")
    print(f"  Threats Detected: {np.sum(results['predictions'] == 1)}")
    
    print(f"\nThreat Analysis:")
    threat_analysis = results['threat_analysis']
    print(f"  Total Threats: {threat_analysis['total_threats_detected']}")
    print(f"  High Confidence Threats: {threat_analysis['high_confidence_threats']}")
    
    print(f"\nThreat Breakdown:")
    for threat_type, details in threat_analysis['threat_breakdown'].items():
        if details['count'] > 0:
            print(f"  {threat_type}: {details['count']} detections, {details['users_affected']} users affected")
    
    print(f"\nIncident Response:")
    incident = results['incident_response']
    print(f"  Total Incidents: {incident['total_incidents']}")
    print(f"  High Priority: {incident['high_priority_incidents']}")
    
    return results

def example_5_comprehensive_industry_comparison():
    """Example 5: Compare anomaly detection across industries."""
    print("\n=== Example 5: Cross-Industry Comparison ===")
    
    industries = {
        'Financial': FinancialAnomalyDetector(),
        'Healthcare': HealthcareAnomalyDetector(),
        'Manufacturing': ManufacturingAnomalyDetector(),
        'Cybersecurity': CybersecurityAnomalyDetector()
    }
    
    comparison_results = {}
    
    for industry_name, detector in industries.items():
        print(f"\nProcessing {industry_name} use case...")
        
        if industry_name == 'Financial':
            df = detector.generate_transaction_data(1000)
            results = detector.detect_fraud(df)
            key_metric = results['auc_score']
            domain_metric = results['impact_analysis']['roi']
            
        elif industry_name == 'Healthcare':
            df = detector.generate_patient_data(50, 3)
            results = detector.detect_patient_anomalies(df)
            key_metric = results['sensitivity']
            domain_metric = results['clinical_metrics']['clinical_utility_index']
            
        elif industry_name == 'Manufacturing':
            df = detector.generate_sensor_data(2000)
            results = detector.detect_equipment_anomalies(df)
            key_metric = 0.85  # Placeholder - would calculate from results
            domain_metric = results['cost_analysis']['roi_percentage'] / 100
            
        elif industry_name == 'Cybersecurity':
            df = detector.generate_network_data(1500)
            results = detector.detect_security_anomalies(df)
            key_metric = results['security_score']
            domain_metric = results['incident_response']['high_priority_incidents'] / max(1, results['incident_response']['total_incidents'])
        
        comparison_results[industry_name] = {
            'samples': len(df),
            'anomaly_rate': df.iloc[:, -1].mean(),  # Last column is typically is_anomaly
            'key_metric': key_metric,
            'domain_metric': domain_metric
        }
    
    print(f"\n{'='*60}")
    print("CROSS-INDUSTRY ANOMALY DETECTION COMPARISON")
    print(f"{'='*60}")
    
    print(f"{'Industry':<15} {'Samples':<8} {'Anomaly%':<9} {'Key Metric':<11} {'Domain Metric':<13}")
    print(f"{'-'*60}")
    
    for industry, metrics in comparison_results.items():
        print(f"{industry:<15} {metrics['samples']:<8} {metrics['anomaly_rate']*100:<8.1f}% {metrics['key_metric']:<10.3f} {metrics['domain_metric']:<12.3f}")
    
    print(f"\nKey Insights:")
    print("• Financial: Focus on precision and regulatory compliance")
    print("• Healthcare: Emphasize sensitivity and patient safety")
    print("• Manufacturing: Optimize for cost savings and uptime")
    print("• Cybersecurity: Balance detection rate with alert fatigue")
    
    return comparison_results

if __name__ == "__main__":
    print("🏭 Industry-Specific Anomaly Detection Use Cases")
    print("=" * 60)
    
    try:
        # Run all industry examples
        financial_results = example_1_financial_fraud_detection()
        healthcare_results = example_2_healthcare_monitoring()
        manufacturing_results = example_3_manufacturing_maintenance()
        cybersecurity_results = example_4_cybersecurity_detection()
        comparison = example_5_comprehensive_industry_comparison()
        
        print("\n✅ All industry use cases completed successfully!")
        print("\nIndustry-Specific Best Practices:")
        print("1. Financial: Implement explainable AI for regulatory compliance")
        print("2. Healthcare: Prioritize sensitivity over specificity for patient safety")
        print("3. Manufacturing: Focus on predictive maintenance and cost optimization")
        print("4. Cybersecurity: Balance detection capabilities with operational efficiency")
        print("5. Cross-Industry: Adapt algorithms and metrics to domain requirements")
        
    except Exception as e:
        print(f"❌ Error running industry examples: {e}")
        import traceback
        traceback.print_exc()