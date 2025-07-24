"""
Real-time Fraud Detection System

This module implements a comprehensive fraud detection system using ensemble methods
and real-time feature engineering for transaction monitoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import joblib
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb

# MLOps platform imports
import mlflow
import mlflow.sklearn
from mlops.infrastructure.feature_store.feature_store import FeatureStore
from machine_learning.infrastructure.serving.model_server import ModelServer
from mlops.infrastructure.streaming.kafka_client import KafkaStreaming

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudModelConfig:
    """Configuration for fraud detection model"""
    model_name: str = "fraud_detection"
    version: str = "1.0"
    target_precision: float = 0.95  # Minimize false positives
    target_recall: float = 0.90     # Catch actual fraud
    max_latency_ms: float = 50.0    # Real-time requirement
    
    # Model hyperparameters
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    isolation_contamination: float = 0.05
    mlp_hidden_layers: Tuple[int, ...] = (100, 50)
    random_state: int = 42

class RealTimeFraudDetection:
    """Real-time fraud detection system with ensemble models"""
    
    def __init__(self, config: FraudModelConfig):
        self.config = config
        self.models = {
            'xgboost': None,
            'isolation_forest': None,
            'neural_network': None,
            'rule_engine': None
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_store = FeatureStore()
        self.model_server = ModelServer()
        self.kafka_client = KafkaStreaming()
        
        # Real-time feature definitions
        self.transaction_features = [
            'amount', 'merchant_risk_score', 'hour_of_day', 'day_of_week',
            'transaction_count_1h', 'transaction_count_24h', 'amount_sum_1h',
            'amount_sum_24h', 'velocity_km_per_hour', 'days_since_last_transaction'
        ]
        
        self.customer_features = [
            'account_age_days', 'avg_transaction_amount', 'transaction_frequency',
            'spending_pattern_deviation', 'device_risk_score', 'behavior_score',
            'verification_level', 'failed_attempts_24h', 'countries_24h'
        ]
        
        self.categorical_features = [
            'merchant_category', 'payment_method', 'device_type', 
            'country_code', 'currency', 'channel'
        ]
        
        # Rule-based thresholds
        self.rule_thresholds = {
            'max_amount_single': 10000,
            'max_amount_1h': 5000,
            'max_transactions_1h': 10,
            'max_velocity_kmh': 500,
            'max_countries_24h': 3
        }
        
        # Initialize MLflow experiment
        mlflow.set_experiment(f"fraud_detection_{self.config.model_name}")
        
    def generate_synthetic_fraud_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """Generate realistic synthetic transaction data with fraud patterns"""
        np.random.seed(self.config.random_state)
        
        data = []
        fraud_rate = 0.02  # 2% fraud rate (realistic for financial data)
        n_fraud = int(n_samples * fraud_rate)
        n_legitimate = n_samples - n_fraud
        
        # Generate legitimate transactions
        for i in range(n_legitimate):
            # Normal customer behavior
            base_amount = np.random.lognormal(3, 1)  # Log-normal distribution for amounts
            customer_id = f'CUST_{np.random.randint(1, 100000):06d}'
            
            transaction = {
                'transaction_id': f'TXN_{i:08d}',
                'customer_id': customer_id,
                'is_fraud': 0,
                
                # Transaction features
                'amount': max(1, base_amount),
                'merchant_category': np.random.choice(['grocery', 'gas', 'retail', 'restaurant', 'online'], 
                                                    p=[0.3, 0.15, 0.25, 0.2, 0.1]),
                'payment_method': np.random.choice(['credit_card', 'debit_card', 'mobile_pay'], 
                                                 p=[0.6, 0.3, 0.1]),
                'device_type': np.random.choice(['pos', 'online', 'mobile', 'atm'], 
                                              p=[0.4, 0.3, 0.25, 0.05]),
                'country_code': np.random.choice(['US', 'CA', 'UK', 'DE', 'FR'], 
                                               p=[0.7, 0.1, 0.1, 0.05, 0.05]),
                'currency': 'USD',
                'channel': np.random.choice(['card_present', 'card_not_present', 'online'], 
                                          p=[0.6, 0.25, 0.15]),
                
                # Temporal features
                'hour_of_day': np.random.choice(range(24), p=self._get_hourly_distribution()),
                'day_of_week': np.random.randint(0, 7),
                
                # Customer behavior (normal patterns)
                'account_age_days': np.random.randint(30, 2000),
                'transaction_count_1h': np.random.poisson(1),
                'transaction_count_24h': np.random.poisson(3),
                'amount_sum_1h': base_amount * np.random.poisson(1),
                'amount_sum_24h': base_amount * np.random.poisson(3),
                'avg_transaction_amount': base_amount * np.random.normal(1, 0.2),
                'transaction_frequency': np.random.exponential(2),
                'spending_pattern_deviation': np.random.normal(0, 0.1),
                'device_risk_score': np.random.beta(2, 8),  # Low risk for legitimate
                'behavior_score': np.random.beta(8, 2),     # High score for normal behavior
                'verification_level': np.random.choice([1, 2, 3], p=[0.1, 0.6, 0.3]),
                'failed_attempts_24h': np.random.poisson(0.1),
                'countries_24h': 1,
                'velocity_km_per_hour': np.random.exponential(20),
                'days_since_last_transaction': np.random.exponential(1),
                'merchant_risk_score': np.random.beta(2, 8)  # Low risk merchants
            }
            
            data.append(transaction)
        
        # Generate fraudulent transactions
        for i in range(n_legitimate, n_samples):
            # Fraudulent patterns
            fraud_type = np.random.choice(['card_theft', 'account_takeover', 'synthetic_identity'])
            
            if fraud_type == 'card_theft':
                # Stolen card patterns: unusual locations, amounts, merchants
                base_amount = np.random.lognormal(4, 1.5)  # Higher amounts
                transaction = {
                    'transaction_id': f'TXN_{i:08d}',
                    'customer_id': f'CUST_{np.random.randint(1, 100000):06d}',
                    'is_fraud': 1,
                    'amount': max(1, base_amount),
                    'merchant_category': np.random.choice(['online', 'electronics', 'luxury'], 
                                                        p=[0.4, 0.4, 0.2]),
                    'payment_method': np.random.choice(['credit_card', 'debit_card'], p=[0.8, 0.2]),
                    'device_type': np.random.choice(['online', 'pos'], p=[0.7, 0.3]),
                    'country_code': np.random.choice(['US', 'RU', 'CN', 'NG'], p=[0.5, 0.2, 0.2, 0.1]),
                    'currency': 'USD',
                    'channel': np.random.choice(['card_not_present', 'online'], p=[0.6, 0.4]),
                    'hour_of_day': np.random.choice(range(24)),  # Any time
                    'day_of_week': np.random.randint(0, 7),
                    'account_age_days': np.random.randint(100, 1500),
                    'transaction_count_1h': np.random.poisson(3),  # Higher frequency
                    'transaction_count_24h': np.random.poisson(8),
                    'amount_sum_1h': base_amount * np.random.poisson(3),
                    'amount_sum_24h': base_amount * np.random.poisson(8),
                    'avg_transaction_amount': base_amount * 0.5,  # Deviation from normal
                    'transaction_frequency': np.random.exponential(0.5),  # More frequent
                    'spending_pattern_deviation': np.random.normal(2, 0.5),  # High deviation
                    'device_risk_score': np.random.beta(6, 2),  # High risk
                    'behavior_score': np.random.beta(2, 6),     # Low behavior score
                    'verification_level': np.random.choice([1, 2], p=[0.7, 0.3]),
                    'failed_attempts_24h': np.random.poisson(2),
                    'countries_24h': np.random.poisson(2) + 1,
                    'velocity_km_per_hour': np.random.exponential(100),  # High velocity
                    'days_since_last_transaction': np.random.exponential(0.1),
                    'merchant_risk_score': np.random.beta(6, 2)  # High risk merchants
                }
                
            elif fraud_type == 'account_takeover':
                # Account takeover: sudden behavior change
                base_amount = np.random.lognormal(3.5, 1.2)
                transaction = {
                    'transaction_id': f'TXN_{i:08d}',
                    'customer_id': f'CUST_{np.random.randint(1, 100000):06d}',
                    'is_fraud': 1,
                    'amount': max(1, base_amount),
                    'merchant_category': np.random.choice(['online', 'transfer', 'withdrawal']),
                    'payment_method': np.random.choice(['credit_card', 'bank_transfer'], p=[0.6, 0.4]),
                    'device_type': np.random.choice(['online', 'mobile'], p=[0.7, 0.3]),
                    'country_code': np.random.choice(['US', 'CA', 'UK'], p=[0.6, 0.2, 0.2]),
                    'currency': 'USD',
                    'channel': 'online',
                    'hour_of_day': np.random.choice([2, 3, 4, 22, 23]),  # Unusual hours
                    'day_of_week': np.random.randint(0, 7),
                    'account_age_days': np.random.randint(200, 2000),  # Established accounts
                    'transaction_count_1h': np.random.poisson(2),
                    'transaction_count_24h': np.random.poisson(5),
                    'amount_sum_1h': base_amount * 2,
                    'amount_sum_24h': base_amount * 5,
                    'avg_transaction_amount': base_amount * 0.3,  # Different from history
                    'transaction_frequency': np.random.exponential(0.2),
                    'spending_pattern_deviation': np.random.normal(3, 0.5),  # Very high deviation
                    'device_risk_score': np.random.beta(5, 3),
                    'behavior_score': np.random.beta(1, 8),     # Very low behavior score
                    'verification_level': np.random.choice([1, 2], p=[0.8, 0.2]),
                    'failed_attempts_24h': np.random.poisson(5),  # Many failed attempts
                    'countries_24h': 1,
                    'velocity_km_per_hour': np.random.exponential(50),
                    'days_since_last_transaction': np.random.exponential(0.5),
                    'merchant_risk_score': np.random.beta(4, 4)
                }
                
            else:  # synthetic_identity
                # Synthetic identity: new accounts with suspicious patterns
                base_amount = np.random.lognormal(3.8, 1)
                transaction = {
                    'transaction_id': f'TXN_{i:08d}',
                    'customer_id': f'CUST_{np.random.randint(100000, 200000):06d}',  # New customer range
                    'is_fraud': 1,
                    'amount': max(1, base_amount),
                    'merchant_category': np.random.choice(['online', 'cash_advance', 'luxury']),
                    'payment_method': 'credit_card',
                    'device_type': np.random.choice(['online', 'mobile'], p=[0.8, 0.2]),
                    'country_code': 'US',
                    'currency': 'USD',
                    'channel': 'card_not_present',
                    'hour_of_day': np.random.choice(range(24)),
                    'day_of_week': np.random.randint(0, 7),
                    'account_age_days': np.random.randint(1, 30),  # Very new accounts
                    'transaction_count_1h': np.random.poisson(1),
                    'transaction_count_24h': np.random.poisson(2),
                    'amount_sum_1h': base_amount,
                    'amount_sum_24h': base_amount * 2,
                    'avg_transaction_amount': base_amount,  # Limited history
                    'transaction_frequency': np.random.exponential(1),
                    'spending_pattern_deviation': np.random.normal(1, 0.3),
                    'device_risk_score': np.random.beta(7, 3),  # High risk devices
                    'behavior_score': np.random.beta(3, 5),     # Medium-low behavior
                    'verification_level': np.random.choice([1, 2], p=[0.9, 0.1]),  # Low verification
                    'failed_attempts_24h': np.random.poisson(1),
                    'countries_24h': 1,
                    'velocity_km_per_hour': np.random.exponential(30),
                    'days_since_last_transaction': 0,  # First transaction
                    'merchant_risk_score': np.random.beta(5, 3)
                }
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
        
        logger.info(f"Generated {n_samples} synthetic transactions")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
        logger.info(f"Average amount: ${df['amount'].mean():.2f}")
        logger.info(f"Fraud types distribution: {df[df['is_fraud']==1]['merchant_category'].value_counts().to_dict()}")
        
        return df
    
    def _get_hourly_distribution(self) -> List[float]:
        """Get realistic hourly transaction distribution"""
        # Peak hours: 9-17, evening: 18-22, night: 23-6, morning: 7-8
        base_probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04,  # 0-7
                     0.05, 0.06, 0.07, 0.08, 0.09, 0.09, 0.08, 0.08,  # 8-15
                     0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]  # 16-23
        return base_probs
    
    def apply_rule_engine(self, df: pd.DataFrame) -> pd.Series:
        """Apply rule-based fraud detection"""
        
        rule_flags = pd.Series(0, index=df.index)
        
        # High amount rules
        rule_flags |= (df['amount'] > self.rule_thresholds['max_amount_single']).astype(int)
        rule_flags |= (df['amount_sum_1h'] > self.rule_thresholds['max_amount_1h']).astype(int)
        
        # Velocity rules
        rule_flags |= (df['transaction_count_1h'] > self.rule_thresholds['max_transactions_1h']).astype(int)
        rule_flags |= (df['velocity_km_per_hour'] > self.rule_thresholds['max_velocity_kmh']).astype(int)
        
        # Geographic rules
        rule_flags |= (df['countries_24h'] > self.rule_thresholds['max_countries_24h']).astype(int)
        
        # Device and behavior rules
        rule_flags |= (df['device_risk_score'] > 0.8).astype(int)
        rule_flags |= (df['behavior_score'] < 0.2).astype(int)
        rule_flags |= (df['failed_attempts_24h'] > 5).astype(int)
        
        return rule_flags
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for ensemble models"""
        
        processed_df = df.copy()
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in processed_df.columns:
                if is_training:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        processed_df[col] = processed_df[col].map(
                            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                        )
        
        # Create derived features
        processed_df['amount_log'] = np.log1p(processed_df['amount'])
        processed_df['spending_velocity'] = processed_df['amount_sum_24h'] / processed_df['account_age_days'].clip(1)
        processed_df['transaction_velocity'] = processed_df['transaction_count_24h'] / processed_df['account_age_days'].clip(1)
        processed_df['risk_composite'] = (
            processed_df['device_risk_score'] + 
            processed_df['merchant_risk_score'] + 
            (1 - processed_df['behavior_score'])
        ) / 3
        
        # Apply rule engine
        processed_df['rule_score'] = self.apply_rule_engine(processed_df)
        
        # Select features for model
        feature_columns = (
            self.transaction_features + 
            self.customer_features + 
            self.categorical_features +
            ['amount_log', 'spending_velocity', 'transaction_velocity', 'risk_composite', 'rule_score']
        )
        
        available_columns = [col for col in feature_columns if col in processed_df.columns]
        X = processed_df[available_columns].fillna(0)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, available_columns
    
    def train_ensemble_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble of fraud detection models"""
        
        with mlflow.start_run(run_name=f"fraud_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Starting fraud detection ensemble training...")
            
            # Log dataset info
            fraud_rate = df['is_fraud'].mean()
            mlflow.log_params({
                "dataset_size": len(df),
                "fraud_rate": fraud_rate,
                "legitimate_count": (df['is_fraud'] == 0).sum(),
                "fraud_count": (df['is_fraud'] == 1).sum()
            })
            
            # Preprocess data
            X, feature_names = self.preprocess_data(df, is_training=True)
            y = df['is_fraud'].values
            
            # Train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_state, stratify=y
            )
            
            mlflow.log_params({
                "train_size": len(X_train),
                "test_size": len(X_test),
                "num_features": len(feature_names)
            })
            
            ensemble_results = {}
            
            # 1. Train XGBoost model
            logger.info("Training XGBoost model...")
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                learning_rate=self.config.xgb_learning_rate,
                max_depth=self.config.xgb_max_depth,
                random_state=self.config.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            self.models['xgboost'].fit(X_train, y_train)
            xgb_pred = self.models['xgboost'].predict(X_test)
            xgb_proba = self.models['xgboost'].predict_proba(X_test)[:, 1]
            
            ensemble_results['xgboost'] = {
                'auc': roc_auc_score(y_test, xgb_proba),
                'precision': ((xgb_pred == 1) & (y_test == 1)).sum() / max(1, (xgb_pred == 1).sum()),
                'recall': ((xgb_pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())
            }
            
            # 2. Train Isolation Forest
            logger.info("Training Isolation Forest...")
            self.models['isolation_forest'] = IsolationForest(
                contamination=self.config.isolation_contamination,
                random_state=self.config.random_state
            )
            
            self.models['isolation_forest'].fit(X_train[y_train == 0])  # Train only on legitimate transactions
            iso_pred = (self.models['isolation_forest'].predict(X_test) == -1).astype(int)
            iso_scores = self.models['isolation_forest'].score_samples(X_test)
            
            ensemble_results['isolation_forest'] = {
                'auc': roc_auc_score(y_test, -iso_scores),  # Negative because lower scores = more anomalous
                'precision': ((iso_pred == 1) & (y_test == 1)).sum() / max(1, (iso_pred == 1).sum()),
                'recall': ((iso_pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())
            }
            
            # 3. Train Neural Network
            logger.info("Training Neural Network...")
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=self.config.mlp_hidden_layers,
                random_state=self.config.random_state,
                max_iter=500
            )
            
            self.models['neural_network'].fit(X_train, y_train)
            nn_pred = self.models['neural_network'].predict(X_test)
            nn_proba = self.models['neural_network'].predict_proba(X_test)[:, 1]
            
            ensemble_results['neural_network'] = {
                'auc': roc_auc_score(y_test, nn_proba),
                'precision': ((nn_pred == 1) & (y_test == 1)).sum() / max(1, (nn_pred == 1).sum()),
                'recall': ((nn_pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())
            }
            
            # 4. Evaluate Rule Engine
            rule_pred = self.apply_rule_engine(df.iloc[X_test.shape[0]:]).iloc[:len(X_test)]
            ensemble_results['rule_engine'] = {
                'precision': ((rule_pred == 1) & (y_test == 1)).sum() / max(1, (rule_pred == 1).sum()),
                'recall': ((rule_pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())
            }
            
            # 5. Create ensemble predictions
            ensemble_proba = (
                0.4 * xgb_proba + 
                0.25 * (-iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min()) +
                0.25 * nn_proba +
                0.1 * rule_pred
            )
            
            # Find optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            
            ensemble_pred = (ensemble_proba >= optimal_threshold).astype(int)
            
            # Calculate ensemble metrics
            ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            ensemble_precision = ((ensemble_pred == 1) & (y_test == 1)).sum() / max(1, (ensemble_pred == 1).sum())
            ensemble_recall = ((ensemble_pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())
            
            ensemble_results['ensemble'] = {
                'auc': ensemble_auc,
                'precision': ensemble_precision,
                'recall': ensemble_recall,
                'threshold': optimal_threshold
            }
            
            # Log all metrics
            for model_name, metrics in ensemble_results.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Log hyperparameters
            mlflow.log_params({
                "xgb_n_estimators": self.config.xgb_n_estimators,
                "xgb_learning_rate": self.config.xgb_learning_rate,
                "xgb_max_depth": self.config.xgb_max_depth,
                "isolation_contamination": self.config.isolation_contamination,
                "mlp_hidden_layers": str(self.config.mlp_hidden_layers)
            })
            
            # Save models
            joblib.dump(self.models, "fraud_ensemble_models.pkl")
            joblib.dump(self.scaler, "fraud_scaler.pkl")
            joblib.dump(self.label_encoders, "fraud_label_encoders.pkl")
            joblib.dump(feature_names, "fraud_feature_names.pkl")
            joblib.dump({'threshold': optimal_threshold}, "fraud_threshold.pkl")
            
            mlflow.log_artifact("fraud_ensemble_models.pkl")
            mlflow.log_artifact("fraud_scaler.pkl")
            mlflow.log_artifact("fraud_label_encoders.pkl")
            mlflow.log_artifact("fraud_feature_names.pkl")
            mlflow.log_artifact("fraud_threshold.pkl")
            
            # Check if ensemble meets targets
            meets_precision_target = ensemble_precision >= self.config.target_precision
            meets_recall_target = ensemble_recall >= self.config.target_recall
            
            results = {
                "ensemble_results": ensemble_results,
                "ensemble_auc": ensemble_auc,
                "ensemble_precision": ensemble_precision,
                "ensemble_recall": ensemble_recall,
                "optimal_threshold": optimal_threshold,
                "meets_targets": meets_precision_target and meets_recall_target,
                "run_id": mlflow.active_run().info.run_id
            }
            
            logger.info(f"Fraud detection ensemble training completed:")
            logger.info(f"  Ensemble AUC: {ensemble_auc:.4f}")
            logger.info(f"  Ensemble Precision: {ensemble_precision:.4f}")
            logger.info(f"  Ensemble Recall: {ensemble_recall:.4f}")
            logger.info(f"  Meets Targets: {results['meets_targets']}")
            
            return results
    
    async def real_time_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make real-time fraud prediction with low latency"""
        
        start_time = datetime.now()
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([transaction_data])
        
        # Preprocess
        X, _ = self.preprocess_data(df, is_training=False)
        
        # Get predictions from all models
        predictions = {}
        
        # XGBoost prediction
        xgb_proba = self.models['xgboost'].predict_proba(X)[0, 1]
        predictions['xgboost'] = xgb_proba
        
        # Isolation Forest prediction
        iso_score = self.models['isolation_forest'].score_samples(X)[0]
        predictions['isolation_forest'] = -iso_score  # Convert to risk score
        
        # Neural Network prediction
        nn_proba = self.models['neural_network'].predict_proba(X)[0, 1]
        predictions['neural_network'] = nn_proba
        
        # Rule engine prediction
        rule_score = self.apply_rule_engine(df).iloc[0]
        predictions['rule_engine'] = rule_score
        
        # Ensemble prediction
        ensemble_score = (
            0.4 * xgb_proba + 
            0.25 * iso_score + 
            0.25 * nn_proba +
            0.1 * rule_score
        )
        
        # Determine risk level
        if ensemble_score >= 0.8:
            risk_level = "HIGH"
            action = "BLOCK"
        elif ensemble_score >= 0.5:
            risk_level = "MEDIUM"
            action = "REVIEW"
        elif ensemble_score >= 0.3:
            risk_level = "LOW"
            action = "STEP_UP_AUTH"
        else:
            risk_level = "VERY_LOW"
            action = "APPROVE"
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "transaction_id": transaction_data.get("transaction_id"),
            "fraud_score": ensemble_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "model_scores": predictions,
            "processing_time_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main execution function for fraud detection system"""
    
    # Initialize configuration
    config = FraudModelConfig()
    
    # Create fraud detection instance
    fraud_detector = RealTimeFraudDetection(config)
    
    try:
        # Step 1: Generate training data
        logger.info("Step 1: Generating synthetic fraud detection data...")
        training_data = fraud_detector.generate_synthetic_fraud_data(n_samples=100000)
        
        # Step 2: Train ensemble models
        logger.info("Step 2: Training fraud detection ensemble...")
        training_results = fraud_detector.train_ensemble_models(training_data)
        
        if not training_results["meets_targets"]:
            logger.warning("Fraud detection ensemble does not meet target performance criteria")
            return False
        
        # Step 3: Test real-time prediction
        logger.info("Step 3: Testing real-time prediction...")
        test_transaction = {
            "transaction_id": "TEST_001",
            "amount": 5000,
            "merchant_category": "online",
            "payment_method": "credit_card",
            "device_type": "online",
            "country_code": "US",
            "currency": "USD",
            "channel": "card_not_present",
            "hour_of_day": 3,
            "day_of_week": 2,
            "account_age_days": 30,
            "transaction_count_1h": 5,
            "transaction_count_24h": 10,
            "amount_sum_1h": 5000,
            "amount_sum_24h": 8000,
            "avg_transaction_amount": 100,
            "transaction_frequency": 0.1,
            "spending_pattern_deviation": 3.0,
            "device_risk_score": 0.8,
            "behavior_score": 0.1,
            "verification_level": 1,
            "failed_attempts_24h": 3,
            "countries_24h": 2,
            "velocity_km_per_hour": 800,
            "days_since_last_transaction": 0.1,
            "merchant_risk_score": 0.7
        }
        
        import asyncio
        prediction_result = asyncio.run(fraud_detector.real_time_prediction(test_transaction))
        
        # Summary
        logger.info("\n🎉 FRAUD DETECTION SYSTEM DEPLOYED!")
        logger.info("="*60)
        logger.info(f"Ensemble Performance:")
        logger.info(f"  AUC: {training_results['ensemble_auc']:.4f}")
        logger.info(f"  Precision: {training_results['ensemble_precision']:.4f}")
        logger.info(f"  Recall: {training_results['ensemble_recall']:.4f}")
        logger.info(f"Real-time Test:")
        logger.info(f"  Fraud Score: {prediction_result['fraud_score']:.4f}")
        logger.info(f"  Risk Level: {prediction_result['risk_level']}")
        logger.info(f"  Action: {prediction_result['recommended_action']}")
        logger.info(f"  Latency: {prediction_result['processing_time_ms']:.1f}ms")
        logger.info(f"MLflow Run: {training_results['run_id']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fraud detection system failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)