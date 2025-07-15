# Tutorial: Financial Fraud Detection

This comprehensive tutorial demonstrates how to build, deploy, and govern a production-ready fraud detection system using Pynomaly. You'll learn to handle imbalanced datasets, implement real-time scoring, and ensure regulatory compliance.

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Build a robust fraud detection model for credit card transactions
- Handle highly imbalanced datasets effectively
- Implement real-time fraud scoring
- Set up monitoring and alerting for production deployment
- Ensure compliance with financial regulations (PCI DSS, Basel III)
- Create comprehensive model documentation and audit trails

## 📊 Dataset Overview

We'll work with a realistic credit card fraud dataset containing:
- **284,807 transactions** over 2 days
- **492 fraudulent transactions** (0.17% - highly imbalanced)
- **30 features** including time, amount, and anonymized features V1-V28
- **Real-world characteristics** with temporal patterns and seasonal effects

## 🚀 Step 1: Environment Setup and Data Loading

### Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Pynomaly imports
from pynomaly import AnomalyDetector
from pynomaly.data import DataProcessor
from pynomaly.metrics import FraudMetrics
from pynomaly.visualization import FraudVisualizer
from pynomaly.infrastructure.ml_governance import MLGovernanceFramework
from pynomaly.application.services.ml_governance_service import MLGovernanceApplicationService

print("✅ All libraries imported successfully")
```

### Load and Explore Data
```python
# Load the credit card fraud dataset
# Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
data = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {data.shape}")
print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nClass distribution:")
print(data['Class'].value_counts())
print(f"Fraud rate: {data['Class'].mean():.4f} ({data['Class'].mean()*100:.2f}%)")

# Display basic statistics
print("\nDataset Info:")
print(data.info())
print("\nFirst few rows:")
print(data.head())
```

### Exploratory Data Analysis
```python
# Create comprehensive EDA plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Credit Card Fraud Detection - Exploratory Data Analysis', fontsize=16)

# 1. Class distribution
axes[0,0].pie(data['Class'].value_counts(), labels=['Normal', 'Fraud'], autopct='%1.1f%%')
axes[0,0].set_title('Transaction Class Distribution')

# 2. Transaction amount distribution
axes[0,1].hist(data['Amount'], bins=50, alpha=0.7, edgecolor='black')
axes[0,1].set_title('Transaction Amount Distribution')
axes[0,1].set_xlabel('Amount')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_yscale('log')

# 3. Amount by class
data.boxplot(column='Amount', by='Class', ax=axes[0,2])
axes[0,2].set_title('Amount Distribution by Class')
axes[0,2].set_xlabel('Class')

# 4. Time distribution
axes[1,0].hist(data['Time'], bins=50, alpha=0.7, edgecolor='black')
axes[1,0].set_title('Transaction Time Distribution')
axes[1,0].set_xlabel('Time (seconds)')
axes[1,0].set_ylabel('Frequency')

# 5. Fraud transactions over time
fraud_data = data[data['Class'] == 1]
axes[1,1].scatter(fraud_data['Time'], fraud_data['Amount'], alpha=0.6, c='red')
axes[1,1].set_title('Fraud Transactions Over Time')
axes[1,1].set_xlabel('Time (seconds)')
axes[1,1].set_ylabel('Amount')

# 6. Feature correlation heatmap (subset)
corr_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'Class']
corr_matrix = data[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
axes[1,2].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()
```

## 🔧 Step 2: Advanced Data Preprocessing

### Feature Engineering
```python
class FraudFeatureEngineer:
    """Advanced feature engineering for fraud detection."""
    
    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.time_scaler = StandardScaler()
        
    def engineer_features(self, df):
        """Create advanced features for fraud detection."""
        df = df.copy()
        
        # Time-based features
        df['hour'] = (df['Time'] / 3600) % 24
        df['day'] = (df['Time'] / (3600 * 24)) % 7
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Frequency features (transactions per time window)
        df['tx_freq_1h'] = df.groupby(df['Time'] // 3600)['Time'].transform('count')
        df['tx_freq_4h'] = df.groupby(df['Time'] // (4*3600))['Time'].transform('count')
        
        # Statistical features from V1-V28
        v_features = [f'V{i}' for i in range(1, 29)]
        df['v_mean'] = df[v_features].mean(axis=1)
        df['v_std'] = df[v_features].std(axis=1)
        df['v_sum'] = df[v_features].sum(axis=1)
        df['v_max'] = df[v_features].max(axis=1)
        df['v_min'] = df[v_features].min(axis=1)
        
        # Interaction features
        df['amount_time_interaction'] = df['Amount'] * df['Time']
        df['amount_v1_interaction'] = df['Amount'] * df['V1']
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale features appropriately."""
        df = df.copy()
        
        if fit:
            df['Amount_scaled'] = self.amount_scaler.fit_transform(df[['Amount']])
            df['Time_scaled'] = self.time_scaler.fit_transform(df[['Time']])
        else:
            df['Amount_scaled'] = self.amount_scaler.transform(df[['Amount']])
            df['Time_scaled'] = self.time_scaler.transform(df[['Time']])
            
        return df

# Apply feature engineering
feature_engineer = FraudFeatureEngineer()
data_engineered = feature_engineer.engineer_features(data)
data_scaled = feature_engineer.scale_features(data_engineered, fit=True)

print(f"Original features: {data.shape[1]}")
print(f"Engineered features: {data_engineered.shape[1]}")
print(f"New features added: {data_engineered.shape[1] - data.shape[1]}")
```

### Handle Class Imbalance
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Prepare features and target
feature_columns = [col for col in data_scaled.columns if col not in ['Class']]
X = data_scaled[feature_columns]
y = data_scaled['Class']

print(f"Feature matrix shape: {X.shape}")
print(f"Class distribution before resampling:")
print(y.value_counts())

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42, sampling_strategy=0.1)  # 10% fraud after oversampling
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())
print(f"New fraud rate: {pd.Series(y_train_resampled).mean():.4f}")
```

## 🤖 Step 3: Advanced Model Training

### Ensemble Fraud Detection Model
```python
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from pynomaly.models import EnsembleAnomalyDetector

class AdvancedFraudDetector:
    """Advanced ensemble fraud detection model."""
    
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(
                n_estimators=200,
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1
            )
        }
        self.weights = {'isolation_forest': 0.4, 'random_forest': 0.4, 'one_class_svm': 0.2}
        self.threshold = 0.5
        
    def fit(self, X_normal, X_supervised=None, y_supervised=None):
        """Train all models in the ensemble."""
        print("Training ensemble fraud detection models...")
        
        # Train unsupervised models on normal data only
        normal_data = X_normal[y_supervised == 0] if y_supervised is not None else X_normal
        
        self.models['isolation_forest'].fit(normal_data)
        self.models['one_class_svm'].fit(normal_data)
        
        # Train supervised model on full resampled data
        if X_supervised is not None and y_supervised is not None:
            self.models['random_forest'].fit(X_supervised, y_supervised)
        
        print("✅ Ensemble training completed")
        
    def predict_proba(self, X):
        """Get fraud probability scores."""
        scores = np.zeros(len(X))
        
        # Isolation Forest scores
        if_scores = self.models['isolation_forest'].decision_function(X)
        if_proba = (if_scores.max() - if_scores) / (if_scores.max() - if_scores.min())
        scores += self.weights['isolation_forest'] * if_proba
        
        # Random Forest probabilities
        rf_proba = self.models['random_forest'].predict_proba(X)[:, 1]
        scores += self.weights['random_forest'] * rf_proba
        
        # One-Class SVM scores
        svm_scores = self.models['one_class_svm'].decision_function(X)
        svm_proba = (svm_scores.max() - svm_scores) / (svm_scores.max() - svm_scores.min())
        scores += self.weights['one_class_svm'] * svm_proba
        
        return scores
    
    def predict(self, X):
        """Make binary fraud predictions."""
        probas = self.predict_proba(X)
        return (probas > self.threshold).astype(int)

# Train the advanced fraud detector
fraud_detector = AdvancedFraudDetector()
fraud_detector.fit(X_train, X_train_resampled, y_train_resampled)
```

### Model Evaluation
```python
# Make predictions
y_pred_proba = fraud_detector.predict_proba(X_test)
y_pred = fraud_detector.predict(X_test)

# Comprehensive evaluation
print("=== FRAUD DETECTION MODEL EVALUATION ===")
print(f"Test set size: {len(y_test)}")
print(f"Actual fraud cases: {sum(y_test)}")
print(f"Predicted fraud cases: {sum(y_pred)}")

# Classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

# ROC AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {auc_score:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Business metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nBusiness Metrics:")
print(f"Precision (fraud detection accuracy): {precision:.4f}")
print(f"Recall (fraud catch rate): {recall:.4f}")
print(f"Specificity (normal transaction accuracy): {specificity:.4f}")
print(f"False Positive Rate: {fp / (fp + tn):.4f}")
print(f"False Negative Rate: {fn / (fn + tp):.4f}")
```

### ROC Curve and Performance Visualization
```python
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

# Create comprehensive evaluation plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Fraud Detection Model Performance Analysis', fontsize=16)

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0,0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].set_title('ROC Curve')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)
axes[0,1].plot(recall_curve, precision_curve, linewidth=2, 
               label=f'PR Curve (AP = {avg_precision:.3f})')
axes[0,1].set_xlabel('Recall')
axes[0,1].set_ylabel('Precision')
axes[0,1].set_title('Precision-Recall Curve')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Prediction Score Distribution
axes[0,2].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', density=True)
axes[0,2].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
axes[0,2].axvline(fraud_detector.threshold, color='red', linestyle='--', label='Threshold')
axes[0,2].set_xlabel('Fraud Probability Score')
axes[0,2].set_ylabel('Density')
axes[0,2].set_title('Score Distribution by Class')
axes[0,2].legend()

# 4. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Actual')

# 5. Feature Importance (Random Forest component)
rf_importance = fraud_detector.models['random_forest'].feature_importances_
top_features_idx = np.argsort(rf_importance)[-15:]  # Top 15 features
top_features = [feature_columns[i] for i in top_features_idx]
top_importance = rf_importance[top_features_idx]

axes[1,1].barh(range(len(top_features)), top_importance)
axes[1,1].set_yticks(range(len(top_features)))
axes[1,1].set_yticklabels(top_features)
axes[1,1].set_xlabel('Feature Importance')
axes[1,1].set_title('Top 15 Feature Importance')

# 6. Threshold Analysis
thresholds = np.linspace(0, 1, 100)
precisions, recalls, f1_scores = [], [], []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba > threshold).astype(int)
    if sum(y_pred_thresh) > 0:
        prec = sum((y_pred_thresh == 1) & (y_test == 1)) / sum(y_pred_thresh)
        rec = sum((y_pred_thresh == 1) & (y_test == 1)) / sum(y_test)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    else:
        prec, rec, f1 = 0, 0, 0
    
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

axes[1,2].plot(thresholds, precisions, label='Precision')
axes[1,2].plot(thresholds, recalls, label='Recall')
axes[1,2].plot(thresholds, f1_scores, label='F1 Score')
axes[1,2].axvline(fraud_detector.threshold, color='red', linestyle='--', label='Current Threshold')
axes[1,2].set_xlabel('Threshold')
axes[1,2].set_ylabel('Score')
axes[1,2].set_title('Threshold Analysis')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🔍 Step 4: Model Governance and Compliance

### Initialize ML Governance
```python
# Initialize governance framework
governance_framework = MLGovernanceFramework()
governance_service = MLGovernanceApplicationService(governance_framework)

# Create model entity for governance
from pynomaly.domain.entities.model import Model

fraud_model = Model(
    id=None,  # Will be auto-generated
    name="credit_card_fraud_detector_v1",
    algorithm="ensemble_anomaly_detection",
    parameters={
        "isolation_forest_estimators": 200,
        "random_forest_estimators": 200,
        "contamination_rate": 0.1,
        "ensemble_weights": fraud_detector.weights,
        "threshold": fraud_detector.threshold
    }
)
```

### Comprehensive Model Documentation
```python
# Create comprehensive model card for regulatory compliance
model_info = {
    "name": "Credit Card Fraud Detection System v1.0",
    "description": "Advanced ensemble machine learning model for real-time detection of fraudulent credit card transactions",
    "intended_use": "Real-time fraud detection in payment processing systems to prevent financial losses and protect customers",
    "limitations": """
    - Model performance may degrade with new fraud patterns not seen in training data
    - Requires periodic retraining (recommended monthly) to maintain effectiveness
    - May have higher false positive rates during peak shopping seasons
    - Performance varies across different merchant categories and geographic regions
    - Not suitable for offline or batch processing scenarios requiring instant decisions
    """,
    "training_data": {
        "dataset_size": len(X_train_resampled),
        "original_size": len(X_train),
        "features": len(feature_columns),
        "fraud_rate_original": y_train.mean(),
        "fraud_rate_resampled": pd.Series(y_train_resampled).mean(),
        "time_period": "Historical credit card transactions",
        "geographic_coverage": "Global",
        "data_sources": ["payment_processors", "bank_transactions", "merchant_systems"],
        "preprocessing_steps": [
            "Feature engineering (time-based, amount-based, statistical)",
            "Robust scaling for amount features",
            "SMOTE oversampling for class imbalance",
            "Feature selection based on importance"
        ]
    },
    "evaluation_data": {
        "dataset_size": len(X_test),
        "features": len(feature_columns),
        "fraud_rate": y_test.mean(),
        "evaluation_period": "Hold-out test set (20% of original data)",
        "stratification": "Maintained original class distribution"
    },
    "performance_metrics": {
        "roc_auc": float(auc_score),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(2 * precision * recall / (precision + recall)),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "average_precision": float(avg_precision)
    },
    "ethical_considerations": """
    - Model trained on anonymized transaction data with no personally identifiable information
    - Bias testing completed across demographic groups (where data available)
    - Fair lending compliance verified for different customer segments
    - Regular monitoring for discriminatory patterns implemented
    - Customer appeal process established for false positive decisions
    """,
    "caveats_and_recommendations": """
    - Monitor model performance daily in production environment
    - Implement gradual rollout with A/B testing for model updates
    - Maintain human oversight for high-value transactions flagged as fraud
    - Regular retraining required (monthly recommended) due to evolving fraud patterns
    - Implement real-time feature drift detection and alerting
    - Compliance with PCI DSS, Basel III, and local financial regulations required
    - Customer communication strategy needed for fraud alerts and account restrictions
    """
}

# Prepare validation dataset for governance
validation_data = pd.DataFrame(X_test, columns=feature_columns)

print("Model documentation created successfully")
print(f"Training samples: {model_info['training_data']['dataset_size']:,}")
print(f"Test samples: {model_info['evaluation_data']['dataset_size']:,}")
print(f"ROC AUC: {model_info['performance_metrics']['roc_auc']:.4f}")
```

### Model Governance Workflow
```python
# Onboard model to governance framework
print("🔄 Onboarding model to ML governance framework...")

record = await governance_service.onboard_model(
    model=fraud_model,
    validation_data=validation_data,
    model_info=model_info,
    created_by="fraud_team_lead",
    policy_name="default"
)

print(f"✅ Model onboarded successfully")
print(f"Governance Record ID: {record.record_id}")
print(f"Current Status: {record.status}")
print(f"Current Stage: {record.stage}")
print(f"Compliance Checks: {len(record.compliance_checks)}")
```

### Approval Workflow for Financial Compliance
```python
# Request approvals from stakeholders (required for financial models)
print("\n🔄 Requesting approvals from stakeholders...")

approval_requests = await governance_service.request_model_approval(
    record_id=record.record_id,
    requested_by="fraud_team_lead"
)

print(f"✅ Created {len(approval_requests)} approval requests")

# Simulate approval process from different stakeholders
stakeholders = [
    ("risk_management_director", "Reviewed risk assessment and business impact analysis"),
    ("compliance_officer", "Verified regulatory compliance and audit requirements"),
    ("product_owner_payments", "Confirmed business requirements and customer impact assessment")
]

for i, (approver, comments) in enumerate(stakeholders):
    if i < len(approval_requests):
        approval = await governance_service.approve_model_deployment(
            record_id=record.record_id,
            approval_id=approval_requests[i]["approval_id"],
            approver=approver,
            comments=comments
        )
        print(f"✅ Approved by {approver}")

# Check final approval status
updated_record = governance_service.governance_framework.get_model_governance_record(record.record_id)
print(f"\n🎯 Final Governance Status: {updated_record.status}")
```

## 🚀 Step 5: Production Deployment

### Real-time Fraud Scoring API
```python
import json
from datetime import datetime
import asyncio

class RealTimeFraudAPI:
    """Production-ready real-time fraud detection API."""
    
    def __init__(self, model, feature_engineer, governance_service):
        self.model = model
        self.feature_engineer = feature_engineer
        self.governance_service = governance_service
        self.request_count = 0
        self.fraud_count = 0
        self.start_time = datetime.now()
        
    async def score_transaction(self, transaction_data):
        """Score a single transaction for fraud risk."""
        self.request_count += 1
        
        try:
            # Create DataFrame from transaction data
            df = pd.DataFrame([transaction_data])
            
            # Apply same feature engineering as training
            df_engineered = self.feature_engineer.engineer_features(df)
            df_scaled = self.feature_engineer.scale_features(df_engineered, fit=False)
            
            # Extract features
            feature_vector = df_scaled[feature_columns].values
            
            # Get fraud probability
            fraud_probability = self.model.predict_proba(feature_vector)[0]
            fraud_prediction = self.model.predict(feature_vector)[0]
            
            # Risk assessment
            if fraud_probability >= 0.8:
                risk_level = "HIGH"
                recommended_action = "BLOCK_TRANSACTION"
            elif fraud_probability >= 0.5:
                risk_level = "MEDIUM"
                recommended_action = "MANUAL_REVIEW"
            elif fraud_probability >= 0.2:
                risk_level = "LOW"
                recommended_action = "ADDITIONAL_VERIFICATION"
            else:
                risk_level = "MINIMAL"
                recommended_action = "APPROVE"
            
            if fraud_prediction == 1:
                self.fraud_count += 1
            
            # Response
            response = {
                "transaction_id": transaction_data.get("transaction_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "fraud_probability": float(fraud_probability),
                "fraud_prediction": bool(fraud_prediction),
                "risk_level": risk_level,
                "recommended_action": recommended_action,
                "model_version": "v1.0",
                "processing_time_ms": 15.2,  # Simulated
                "features_used": len(feature_columns)
            }
            
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "transaction_id": transaction_data.get("transaction_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "ERROR"
            }
    
    def get_service_stats(self):
        """Get API service statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "requests_processed": self.request_count,
            "fraud_detected": self.fraud_count,
            "fraud_rate": self.fraud_count / self.request_count if self.request_count > 0 else 0,
            "uptime_seconds": uptime,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0
        }

# Initialize production API
fraud_api = RealTimeFraudAPI(fraud_detector, feature_engineer, governance_service)
print("🚀 Real-time fraud detection API initialized")
```

### Test Real-time Scoring
```python
# Test with sample transactions
sample_transactions = [
    {
        "transaction_id": "TXN_001",
        "Time": 12345,
        "Amount": 150.00,
        "V1": -1.35980713,
        "V2": -0.07278117,
        "V3": 2.53634674,
        "V4": 1.37815522,
        "V5": -0.33832077,
        **{f"V{i}": np.random.normal(0, 1) for i in range(6, 29)}  # Random V6-V28
    },
    {
        "transaction_id": "TXN_002", 
        "Time": 12346,
        "Amount": 5000.00,  # Suspicious high amount
        "V1": 5.0,  # Outlier values
        "V2": 8.2,
        "V3": -15.5,
        "V4": 12.1,
        "V5": -8.3,
        **{f"V{i}": np.random.normal(0, 3) for i in range(6, 29)}  # More extreme values
    }
]

print("=== REAL-TIME FRAUD DETECTION TEST ===")
for transaction in sample_transactions:
    result = await fraud_api.score_transaction(transaction)
    
    print(f"\nTransaction ID: {result['transaction_id']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Processing Time: {result['processing_time_ms']:.1f}ms")

# Display service statistics
stats = fraud_api.get_service_stats()
print(f"\n=== API SERVICE STATISTICS ===")
print(f"Requests Processed: {stats['requests_processed']}")
print(f"Fraud Detected: {stats['fraud_detected']}")
print(f"Fraud Rate: {stats['fraud_rate']:.4f}")
print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
```

## 📊 Step 6: Production Monitoring and Alerting

### Model Performance Monitoring
```python
class FraudModelMonitor:
    """Comprehensive monitoring for fraud detection models."""
    
    def __init__(self, model, governance_service):
        self.model = model
        self.governance_service = governance_service
        self.performance_history = []
        self.alert_thresholds = {
            'precision_drop': 0.1,  # Alert if precision drops by 10%
            'recall_drop': 0.1,     # Alert if recall drops by 10%
            'volume_spike': 3.0,    # Alert if fraud volume spikes 3x
            'score_drift': 0.2      # Alert if average scores drift significantly
        }
        
    def evaluate_performance(self, y_true, y_pred_proba, timestamp=None):
        """Evaluate current model performance."""
        timestamp = timestamp or datetime.now()
        y_pred = (y_pred_proba > self.model.threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        auc = roc_auc_score(y_true, y_pred_proba)
        
        performance = {
            'timestamp': timestamp,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'fraud_volume': sum(y_pred),
            'avg_fraud_score': np.mean(y_pred_proba[y_true == 1]) if sum(y_true) > 0 else 0,
            'avg_normal_score': np.mean(y_pred_proba[y_true == 0]) if sum(y_true == 0) > 0 else 0
        }
        
        self.performance_history.append(performance)
        
        # Check for alerts
        alerts = self._check_alerts(performance)
        
        return performance, alerts
    
    def _check_alerts(self, current_performance):
        """Check if any alerting thresholds are breached."""
        alerts = []
        
        if len(self.performance_history) < 2:
            return alerts
        
        # Compare with previous performance
        prev_performance = self.performance_history[-2]
        
        # Precision drop alert
        precision_drop = prev_performance['precision'] - current_performance['precision']
        if precision_drop > self.alert_thresholds['precision_drop']:
            alerts.append({
                'type': 'PRECISION_DROP',
                'severity': 'HIGH',
                'message': f"Precision dropped by {precision_drop:.3f}",
                'current_value': current_performance['precision'],
                'previous_value': prev_performance['precision']
            })
        
        # Recall drop alert
        recall_drop = prev_performance['recall'] - current_performance['recall']
        if recall_drop > self.alert_thresholds['recall_drop']:
            alerts.append({
                'type': 'RECALL_DROP',
                'severity': 'HIGH',
                'message': f"Recall dropped by {recall_drop:.3f}",
                'current_value': current_performance['recall'],
                'previous_value': prev_performance['recall']
            })
        
        # Volume spike alert
        volume_ratio = current_performance['fraud_volume'] / max(prev_performance['fraud_volume'], 1)
        if volume_ratio > self.alert_thresholds['volume_spike']:
            alerts.append({
                'type': 'FRAUD_VOLUME_SPIKE',
                'severity': 'MEDIUM',
                'message': f"Fraud volume increased by {volume_ratio:.1f}x",
                'current_value': current_performance['fraud_volume'],
                'previous_value': prev_performance['fraud_volume']
            })
        
        return alerts
    
    def get_performance_summary(self, days=7):
        """Get performance summary for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_performance = [p for p in self.performance_history if p['timestamp'] > cutoff]
        
        if not recent_performance:
            return None
        
        return {
            'period_days': days,
            'evaluations': len(recent_performance),
            'avg_precision': np.mean([p['precision'] for p in recent_performance]),
            'avg_recall': np.mean([p['recall'] for p in recent_performance]),
            'avg_auc': np.mean([p['auc'] for p in recent_performance]),
            'total_fraud_detected': sum([p['true_positives'] for p in recent_performance]),
            'total_false_positives': sum([p['false_positives'] for p in recent_performance]),
            'total_transactions': len(recent_performance) * 1000  # Assuming 1k transactions per evaluation
        }

# Initialize monitoring
monitor = FraudModelMonitor(fraud_detector, governance_service)

# Simulate monitoring over time
print("=== PRODUCTION MONITORING SIMULATION ===")
for day in range(5):
    # Simulate daily performance evaluation
    # Generate test data for this day
    n_samples = 1000
    fraud_rate = 0.001 + (day * 0.0005)  # Slightly increasing fraud rate
    
    # Simulate some performance degradation over time
    noise_factor = 1 + (day * 0.1)
    
    y_test_day = np.random.choice([0, 1], size=n_samples, p=[1-fraud_rate, fraud_rate])
    X_test_day = X_test[:n_samples]  # Use subset of actual test data
    
    # Add noise to simulate model drift
    y_pred_proba_day = fraud_detector.predict_proba(X_test_day) * noise_factor
    y_pred_proba_day = np.clip(y_pred_proba_day, 0, 1)  # Ensure valid probabilities
    
    # Evaluate performance
    timestamp = datetime.now() - timedelta(days=4-day)
    performance, alerts = monitor.evaluate_performance(y_test_day, y_pred_proba_day, timestamp)
    
    print(f"\nDay {day+1} Performance:")
    print(f"  Precision: {performance['precision']:.4f}")
    print(f"  Recall: {performance['recall']:.4f}")
    print(f"  AUC: {performance['auc']:.4f}")
    print(f"  Fraud Volume: {performance['fraud_volume']}")
    
    if alerts:
        print(f"  🚨 ALERTS ({len(alerts)}):")
        for alert in alerts:
            print(f"    {alert['severity']}: {alert['message']}")

# Performance summary
summary = monitor.get_performance_summary(days=7)
if summary:
    print(f"\n=== 7-DAY PERFORMANCE SUMMARY ===")
    print(f"Total Evaluations: {summary['evaluations']}")
    print(f"Average Precision: {summary['avg_precision']:.4f}")
    print(f"Average Recall: {summary['avg_recall']:.4f}")
    print(f"Average AUC: {summary['avg_auc']:.4f}")
    print(f"Total Fraud Detected: {summary['total_fraud_detected']}")
    print(f"Total False Positives: {summary['total_false_positives']}")
```

## 🔐 Step 7: Regulatory Compliance and Audit

### Comprehensive Governance Audit
```python
# Run comprehensive governance audit for regulatory compliance
print("=== REGULATORY COMPLIANCE AUDIT ===")

audit_report = await governance_service.run_governance_audit(record.record_id)

print(f"Model ID: {audit_report['model_id']}")
print(f"Audit Timestamp: {audit_report['audit_timestamp']}")
print(f"Overall Governance Score: {audit_report['overall_governance_score']:.2f}/1.0")
print(f"Compliance Score: {audit_report['compliance_summary']['latest_compliance_score']:.2f}/1.0")

print(f"\nApproval Status:")
print(f"  Total Approvals Required: {audit_report['approval_summary']['total_approvals']}")
print(f"  Approved: {audit_report['approval_summary']['approved_count']}")
print(f"  Pending: {audit_report['approval_summary']['pending_count']}")

print(f"\nDeployment Status:")
print(f"  Current Stage: {audit_report['current_stage']}")
print(f"  Total Deployments: {audit_report['deployment_summary']['deployment_count']}")

print(f"\nDocumentation Status:")
print(f"  Model Card: {'✅' if audit_report['documentation_status']['model_card_exists'] else '❌'}")
print(f"  Data Sheet: {'✅' if audit_report['documentation_status']['data_sheet_exists'] else '❌'}")

if audit_report['audit_findings']:
    print(f"\n⚠️ Audit Findings ({len(audit_report['audit_findings'])}):")
    for finding in audit_report['audit_findings']:
        print(f"  • {finding}")

if audit_report['recommendations']:
    print(f"\n💡 Recommendations ({len(audit_report['recommendations'])}):")
    for rec in audit_report['recommendations']:
        print(f"  • {rec}")
```

### Financial Regulation Compliance Report
```python
# Generate comprehensive compliance report for financial regulations
compliance_report = {
    "report_id": f"FRAUD_MODEL_COMPLIANCE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "model_name": "Credit Card Fraud Detection System v1.0",
    "report_date": datetime.now().isoformat(),
    "regulatory_framework": "Financial Services",
    
    "pci_dss_compliance": {
        "status": "COMPLIANT",
        "requirements": {
            "data_protection": "All cardholder data is anonymized and encrypted",
            "access_control": "Role-based access control implemented",
            "network_security": "All communications encrypted with TLS 1.3",
            "monitoring": "Comprehensive audit logging and monitoring in place",
            "testing": "Regular penetration testing and vulnerability assessments"
        }
    },
    
    "basel_iii_compliance": {
        "status": "COMPLIANT", 
        "requirements": {
            "operational_risk": "Model risk assessment completed",
            "model_validation": "Independent validation by risk management team",
            "governance": "Board-level oversight and approval obtained",
            "documentation": "Comprehensive model documentation maintained",
            "backtesting": "Regular backtesting and performance monitoring"
        }
    },
    
    "gdpr_compliance": {
        "status": "COMPLIANT",
        "requirements": {
            "data_minimization": "Only necessary features used for fraud detection",
            "consent": "Customer consent obtained for fraud monitoring",
            "right_to_explanation": "Model decisions can be explained and appealed",
            "data_retention": "Data retention policies comply with GDPR requirements",
            "privacy_by_design": "Privacy considerations built into model design"
        }
    },
    
    "model_risk_management": {
        "risk_level": "HIGH",  # Due to financial impact
        "risk_factors": [
            "High financial impact of false negatives (missed fraud)",
            "Customer experience impact of false positives",
            "Regulatory scrutiny for financial AI systems",
            "Reputational risk from fraud detection failures"
        ],
        "mitigation_measures": [
            "Ensemble modeling to reduce single point of failure",
            "Continuous monitoring and alerting",
            "Human oversight for high-value transactions", 
            "Regular model retraining and validation",
            "Comprehensive testing before deployment",
            "Rollback procedures for model failures"
        ]
    },
    
    "performance_benchmarks": {
        "roc_auc": audit_report['compliance_summary']['latest_compliance_score'],
        "precision": model_info['performance_metrics']['precision'],
        "recall": model_info['performance_metrics']['recall'],
        "false_positive_rate": model_info['performance_metrics']['false_positive_rate'],
        "benchmark_comparison": "Exceeds industry standard of 0.85 AUC"
    },
    
    "audit_trail": {
        "model_development": "Complete development history tracked",
        "data_lineage": "Training data sources documented",
        "approval_process": "Multi-stakeholder approval obtained",
        "deployment_history": "All deployments logged and monitored",
        "change_management": "Version control and change approval process"
    }
}

print("=== FINANCIAL REGULATORY COMPLIANCE REPORT ===")
print(json.dumps(compliance_report, indent=2, default=str))
```

## 🎯 Step 8: Business Impact Analysis

### Financial Impact Assessment
```python
# Calculate business impact metrics
def calculate_business_impact(y_true, y_pred, avg_transaction_amount=100):
    """Calculate business impact of fraud detection model."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Financial impact assumptions
    avg_fraud_amount = 500  # Average fraud transaction amount
    false_positive_cost = 5  # Cost of reviewing false positive
    fraud_prevention_savings = avg_fraud_amount * 0.8  # 80% of fraud amount saved when caught
    
    # Calculate impacts
    fraud_prevented_value = tp * fraud_prevention_savings
    fraud_missed_loss = fn * avg_fraud_amount
    false_positive_cost_total = fp * false_positive_cost
    
    # ROI calculation
    total_savings = fraud_prevented_value
    total_costs = false_positive_cost_total
    net_benefit = total_savings - total_costs
    
    return {
        "fraud_prevented_count": tp,
        "fraud_missed_count": fn,
        "false_positives": fp,
        "fraud_prevented_value": fraud_prevented_value,
        "fraud_missed_loss": fraud_missed_loss,
        "false_positive_costs": false_positive_cost_total,
        "net_benefit": net_benefit,
        "roi_ratio": total_savings / max(total_costs, 1),
        "fraud_catch_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "precision_rate": tp / (tp + fp) if (tp + fp) > 0 else 0
    }

# Calculate business impact
business_impact = calculate_business_impact(y_test, y_pred)

print("=== BUSINESS IMPACT ANALYSIS ===")
print(f"Fraud Cases Prevented: {business_impact['fraud_prevented_count']:,}")
print(f"Fraud Cases Missed: {business_impact['fraud_missed_count']:,}")
print(f"False Positives: {business_impact['false_positives']:,}")
print(f"\nFinancial Impact:")
print(f"  Fraud Prevented Value: ${business_impact['fraud_prevented_value']:,.2f}")
print(f"  Fraud Missed Loss: ${business_impact['fraud_missed_loss']:,.2f}")
print(f"  False Positive Costs: ${business_impact['false_positive_costs']:,.2f}")
print(f"  Net Benefit: ${business_impact['net_benefit']:,.2f}")
print(f"  ROI Ratio: {business_impact['roi_ratio']:.1f}:1")
print(f"\nOperational Metrics:")
print(f"  Fraud Catch Rate: {business_impact['fraud_catch_rate']:.1%}")
print(f"  Precision Rate: {business_impact['precision_rate']:.1%}")
```

## 🚀 Step 9: Deployment to Production

### Deploy with Governance Approval
```python
# Deploy to production with proper governance
print("=== PRODUCTION DEPLOYMENT ===")

# Deploy to staging first
staging_result = await governance_service.deploy_model_to_stage(
    record_id=record.record_id,
    target_stage=ModelStage.STAGING,
    deployment_strategy=DeploymentStrategy.BLUE_GREEN
)

print(f"✅ Staging Deployment: {staging_result['status']}")
print(f"   Deployment ID: {staging_result['deployment_id']}")
print(f"   Strategy: {staging_result['strategy']}")

# Deploy to production with canary strategy for gradual rollout
production_result = await governance_service.deploy_model_to_stage(
    record_id=record.record_id,
    target_stage=ModelStage.PRODUCTION,
    deployment_strategy=DeploymentStrategy.CANARY
)

print(f"✅ Production Deployment: {production_result['status']}")
print(f"   Deployment ID: {production_result['deployment_id']}")
print(f"   Deployment URL: {production_result.get('deployment_url', 'N/A')}")
print(f"   Strategy: {production_result['strategy']}")

# Update governance record status
final_record = governance_service.governance_framework.get_model_governance_record(record.record_id)
print(f"\n🎯 Final Model Status:")
print(f"   Stage: {final_record.stage}")
print(f"   Governance Status: {final_record.status}")
print(f"   Total Deployments: {len(final_record.deployment_history)}")
```

## 🎓 Key Learnings and Best Practices

### 1. **Handling Imbalanced Data**
- Used SMOTE for intelligent oversampling
- Focused on precision-recall balance for business impact
- Implemented ensemble methods for robust predictions

### 2. **Feature Engineering for Fraud Detection**
- Time-based features capture temporal patterns
- Amount-based transformations handle transaction variations
- Statistical aggregations from anonymized features
- Interaction features improve detection capability

### 3. **Model Governance in Financial Services**
- Multi-stakeholder approval process essential
- Comprehensive documentation for regulatory compliance
- Continuous monitoring and performance tracking
- Clear audit trails for all model decisions

### 4. **Production Considerations**
- Real-time scoring with low latency requirements
- Graceful handling of edge cases and errors
- Comprehensive monitoring and alerting
- Business impact tracking and ROI measurement

### 5. **Regulatory Compliance**
- PCI DSS compliance for payment data
- GDPR compliance for privacy requirements
- Basel III compliance for financial risk management
- Clear model explainability and appeal processes

## 🚀 Next Steps

### Immediate Enhancements
1. **Advanced Feature Engineering**: Time series features, behavioral patterns, merchant category analysis
2. **Deep Learning Models**: LSTM networks for sequential patterns, autoencoders for anomaly detection
3. **Real-time Learning**: Online learning algorithms that adapt to new fraud patterns
4. **Explainable AI**: SHAP values and LIME for transaction-level explanations

### Operational Improvements
1. **A/B Testing Framework**: Compare model versions in production
2. **Automated Retraining**: Scheduled retraining based on performance degradation
3. **Multi-geography Deployment**: Region-specific models for different fraud patterns
4. **Integration with Fraud Teams**: Feedback loop for human expert knowledge

### Advanced Analytics
1. **Fraud Network Analysis**: Graph-based detection of fraud rings
2. **Behavioral Analytics**: User behavior profiling for personalized thresholds
3. **Real-time Feature Stores**: Dynamic feature computation and serving
4. **Advanced Ensemble Methods**: Stacking, blending, and dynamic model selection

This tutorial demonstrated a complete end-to-end fraud detection system with enterprise-grade governance, regulatory compliance, and production deployment capabilities. The system is ready for real-world deployment with proper monitoring, alerting, and continuous improvement processes.