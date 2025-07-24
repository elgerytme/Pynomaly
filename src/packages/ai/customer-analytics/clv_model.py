"""
Customer Lifetime Value (CLV) Prediction Model

This module implements customer lifetime value prediction using the MLOps platform,
demonstrating scalability and feature reuse across multiple use cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# MLOps platform imports
import mlflow
import mlflow.sklearn
from mlops.infrastructure.feature_store.feature_store import FeatureStore
from machine_learning.infrastructure.serving.model_server import ModelServer
from mlops.infrastructure.explainability.model_explainability_framework import (
    ModelExplainabilityFramework, ExplanationRequest, ExplanationMethod, ExplanationScope
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CLVModelConfig:
    """Configuration for Customer Lifetime Value model"""
    model_name: str = "customer_lifetime_value"
    version: str = "1.0"
    target_mae_percentage: float = 0.15  # 15% MAE threshold
    target_r2_score: float = 0.75
    max_latency_ms: float = 200.0
    prediction_horizon_months: int = 12
    
    # Model hyperparameters
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 8
    min_samples_split: int = 20
    random_state: int = 42

class CustomerLifetimeValueModel:
    """Customer Lifetime Value prediction model using gradient boosting"""
    
    def __init__(self, config: CLVModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_store = FeatureStore()
        self.model_server = ModelServer()
        self.explainer = ModelExplainabilityFramework()
        
        # Feature definitions for CLV prediction
        self.numerical_features = [
            # Transactional features
            'avg_order_value', 'purchase_frequency', 'recency_days',
            'total_spent', 'order_count', 'avg_days_between_orders',
            'seasonal_spend_variance', 'payment_method_diversity',
            
            # Behavioral features
            'website_sessions', 'avg_session_duration', 'pages_per_session',
            'email_open_rate', 'email_click_rate', 'support_tickets',
            'product_categories_explored', 'review_count', 'referral_count',
            
            # Account features
            'account_tenure_days', 'subscription_value', 'discount_usage',
            'loyalty_points_earned', 'loyalty_points_redeemed'
        ]
        
        self.categorical_features = [
            'preferred_channel', 'subscription_type', 'customer_segment',
            'geographic_region', 'acquisition_source', 'preferred_payment_method',
            'device_type', 'communication_preference'
        ]
        
        # Initialize MLflow experiment
        mlflow.set_experiment(f"customer_analytics_{self.config.model_name}")
        
    def generate_synthetic_clv_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic customer data for CLV modeling"""
        np.random.seed(self.config.random_state)
        
        # Customer segments with different value profiles
        segments = ['high_value', 'medium_value', 'low_value', 'new_customer']
        segment_weights = [0.15, 0.35, 0.35, 0.15]
        
        data = []
        
        for i in range(n_samples):
            # Assign customer segment
            segment = np.random.choice(segments, p=segment_weights)
            
            # Generate features based on segment
            if segment == 'high_value':
                base_spend = np.random.normal(200, 50)
                frequency = np.random.normal(8, 2)
                tenure = np.random.normal(800, 200)
                engagement_multiplier = 1.5
            elif segment == 'medium_value':
                base_spend = np.random.normal(80, 20)
                frequency = np.random.normal(4, 1)
                tenure = np.random.normal(400, 150)
                engagement_multiplier = 1.0
            elif segment == 'low_value':
                base_spend = np.random.normal(30, 15)
                frequency = np.random.normal(2, 0.5)
                tenure = np.random.normal(200, 100)
                engagement_multiplier = 0.6
            else:  # new_customer
                base_spend = np.random.normal(50, 25)
                frequency = np.random.normal(1.5, 0.5)
                tenure = np.random.normal(30, 15)
                engagement_multiplier = 0.8
            
            # Calculate derived features
            order_count = max(1, int(frequency * tenure / 30))
            total_spent = max(0, base_spend * order_count + np.random.normal(0, base_spend * 0.1))
            avg_order_value = total_spent / order_count if order_count > 0 else 0
            
            customer_data = {
                'customer_id': f'CLV_{i:06d}',
                'customer_segment': segment,
                
                # Transactional features
                'avg_order_value': max(10, avg_order_value),
                'purchase_frequency': max(0.1, frequency),
                'recency_days': max(1, int(np.random.exponential(30))),
                'total_spent': total_spent,
                'order_count': order_count,
                'avg_days_between_orders': max(1, 30 / frequency if frequency > 0 else 30),
                'seasonal_spend_variance': np.random.normal(0.2, 0.1),
                'payment_method_diversity': np.random.poisson(2) + 1,
                
                # Behavioral features
                'website_sessions': max(0, int(np.random.normal(
                    order_count * 3 * engagement_multiplier, order_count * 0.5))),
                'avg_session_duration': max(60, np.random.normal(300, 100) * engagement_multiplier),
                'pages_per_session': max(1, np.random.normal(5, 2) * engagement_multiplier),
                'email_open_rate': np.clip(np.random.normal(0.25, 0.1) * engagement_multiplier, 0, 1),
                'email_click_rate': np.clip(np.random.normal(0.05, 0.02) * engagement_multiplier, 0, 1),
                'support_tickets': max(0, int(np.random.poisson(1.5 / engagement_multiplier))),
                'product_categories_explored': max(1, int(np.random.normal(3, 1) * engagement_multiplier)),
                'review_count': max(0, int(np.random.poisson(order_count * 0.1))),
                'referral_count': max(0, int(np.random.poisson(0.5 * engagement_multiplier))),
                
                # Account features
                'account_tenure_days': max(1, int(tenure)),
                'subscription_value': max(0, np.random.normal(50, 20) if np.random.random() < 0.3 else 0),
                'discount_usage': np.clip(np.random.normal(0.15, 0.05), 0, 1),
                'loyalty_points_earned': max(0, int(total_spent * np.random.normal(0.1, 0.02))),
                'loyalty_points_redeemed': 0,  # Will be calculated
                
                # Categorical features
                'preferred_channel': np.random.choice(['online', 'mobile', 'store', 'phone'], 
                                                    p=[0.4, 0.35, 0.2, 0.05]),
                'subscription_type': np.random.choice(['none', 'basic', 'premium'], 
                                                    p=[0.6, 0.3, 0.1]),
                'geographic_region': np.random.choice(['north', 'south', 'east', 'west'], 
                                                    p=[0.3, 0.2, 0.25, 0.25]),
                'acquisition_source': np.random.choice(['organic', 'paid_search', 'social', 'referral'], 
                                                     p=[0.3, 0.25, 0.25, 0.2]),
                'preferred_payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'apple_pay'],
                                                           p=[0.4, 0.25, 0.2, 0.15]),
                'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], 
                                              p=[0.4, 0.5, 0.1]),
                'communication_preference': np.random.choice(['email', 'sms', 'push', 'none'],
                                                           p=[0.5, 0.25, 0.15, 0.1])
            }
            
            # Calculate loyalty points redeemed
            customer_data['loyalty_points_redeemed'] = min(
                customer_data['loyalty_points_earned'],
                int(customer_data['loyalty_points_earned'] * np.random.uniform(0, 0.7))
            )
            
            data.append(customer_data)
        
        df = pd.DataFrame(data)
        
        # Generate CLV target variable with realistic correlations
        def calculate_clv(row):
            # Base CLV calculation
            base_clv = (
                row['avg_order_value'] * 
                row['purchase_frequency'] * 
                self.config.prediction_horizon_months
            )
            
            # Adjustment factors
            tenure_factor = min(2.0, 1 + (row['account_tenure_days'] / 365))
            engagement_factor = 1 + (row['email_open_rate'] + row['email_click_rate'])
            loyalty_factor = 1 + (row['loyalty_points_redeemed'] / max(1, row['loyalty_points_earned']))
            recency_factor = max(0.5, 1 - (row['recency_days'] / 365))
            
            # Premium adjustments
            subscription_bonus = row['subscription_value'] * 12 if row['subscription_value'] > 0 else 0
            
            # Calculate final CLV
            predicted_clv = (
                base_clv * 
                tenure_factor * 
                engagement_factor * 
                loyalty_factor * 
                recency_factor
            ) + subscription_bonus
            
            # Add some noise
            noise = np.random.normal(0, predicted_clv * 0.1)
            return max(0, predicted_clv + noise)
        
        df['customer_lifetime_value'] = df.apply(calculate_clv, axis=1)
        
        logger.info(f"Generated {n_samples} synthetic CLV records")
        logger.info(f"Average CLV: ${df['customer_lifetime_value'].mean():.2f}")
        logger.info(f"CLV std: ${df['customer_lifetime_value'].std():.2f}")
        logger.info(f"Segment distribution: {df['customer_segment'].value_counts().to_dict()}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for model training/inference"""
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
                        # Handle unseen categories
                        processed_df[col] = processed_df[col].map(
                            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                        )
        
        # Create additional derived features
        if 'total_spent' in processed_df.columns and 'order_count' in processed_df.columns:
            processed_df['spend_per_order_ratio'] = (
                processed_df['total_spent'] / processed_df['order_count'].replace(0, 1)
            )
            
        if 'loyalty_points_earned' in processed_df.columns and 'loyalty_points_redeemed' in processed_df.columns:
            processed_df['loyalty_engagement_ratio'] = (
                processed_df['loyalty_points_redeemed'] / 
                processed_df['loyalty_points_earned'].replace(0, 1)
            )
            
        if 'email_open_rate' in processed_df.columns and 'email_click_rate' in processed_df.columns:
            processed_df['email_engagement_score'] = (
                processed_df['email_open_rate'] + processed_df['email_click_rate'] * 2
            )
        
        # Update numerical features list
        derived_features = ['spend_per_order_ratio', 'loyalty_engagement_ratio', 'email_engagement_score']
        all_numerical_features = self.numerical_features + derived_features
        
        # Select features for model
        feature_columns = all_numerical_features + self.categorical_features
        available_columns = [col for col in feature_columns if col in processed_df.columns]
        
        X = processed_df[available_columns].fillna(0)
        
        # Scale numerical features only
        numerical_cols = [col for col in available_columns if col in all_numerical_features]
        if is_training:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X.values, available_columns
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the CLV model with comprehensive evaluation"""
        
        with mlflow.start_run(run_name=f"clv_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Starting CLV model training...")
            
            # Log dataset info
            mlflow.log_params({
                "dataset_size": len(df),
                "avg_clv": df['customer_lifetime_value'].mean(),
                "clv_std": df['customer_lifetime_value'].std(),
                "prediction_horizon_months": self.config.prediction_horizon_months
            })
            
            # Preprocess data
            X, feature_names = self.preprocess_data(df, is_training=True)
            y = df['customer_lifetime_value'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_state
            )
            
            # Log data split info
            mlflow.log_params({
                "train_size": len(X_train),
                "test_size": len(X_test),
                "num_features": len(feature_names)
            })
            
            # Initialize and train model
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=self.config.random_state
            )
            
            # Log hyperparameters
            mlflow.log_params({
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "max_depth": self.config.max_depth,
                "min_samples_split": self.config.min_samples_split
            })
            
            # Train model
            start_time = datetime.now()
            self.model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Calculate percentage errors
            train_mae_pct = train_mae / y_train.mean()
            test_mae_pct = test_mae / y_test.mean()
            
            # Log metrics
            mlflow.log_metrics({
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae_percentage": train_mae_pct,
                "test_mae_percentage": test_mae_pct
            })
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=5, scoring='neg_mean_absolute_error'
            )
            cv_mae = -cv_scores.mean()
            cv_mae_std = cv_scores.std()
            
            mlflow.log_metrics({
                "cv_mae_mean": cv_mae,
                "cv_mae_std": cv_mae_std
            })
            
            # Feature importance
            feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]:
                mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Log model artifacts
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=self.config.model_name
            )
            
            # Save preprocessing artifacts
            joblib.dump(self.scaler, "clv_scaler.pkl")
            joblib.dump(self.label_encoders, "clv_label_encoders.pkl")
            joblib.dump(feature_names, "clv_feature_names.pkl")
            
            mlflow.log_artifact("clv_scaler.pkl")
            mlflow.log_artifact("clv_label_encoders.pkl")
            mlflow.log_artifact("clv_feature_names.pkl")
            
            # Check if model meets target performance
            meets_mae_target = test_mae_pct <= self.config.target_mae_percentage
            meets_r2_target = test_r2 >= self.config.target_r2_score
            
            mlflow.log_params({
                "meets_mae_target": meets_mae_target,
                "meets_r2_target": meets_r2_target
            })
            
            results = {
                "model": self.model,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "test_mae_percentage": test_mae_pct,
                "test_r2": test_r2,
                "cv_mae": cv_mae,
                "feature_importance": feature_importance,
                "meets_targets": meets_mae_target and meets_r2_target,
                "run_id": mlflow.active_run().info.run_id
            }
            
            logger.info(f"CLV model training completed:")
            logger.info(f"  Test MAE: ${test_mae:.2f} ({test_mae_pct:.1%})")
            logger.info(f"  Test R²: {test_r2:.4f}")
            logger.info(f"  CV MAE: ${cv_mae:.2f} ± ${cv_mae_std:.2f}")
            logger.info(f"  Meets Targets: {results['meets_targets']}")
            
            return results
    
    def deploy_model(self, run_id: str) -> str:
        """Deploy CLV model to serving infrastructure"""
        
        logger.info(f"Deploying CLV model from run {run_id}...")
        
        # Register model in model server
        model_id = self.model_server.register_model(
            name=self.config.model_name,
            version=self.config.version,
            model_path=f"runs:/{run_id}/model",
            metadata={
                "description": "Customer Lifetime Value prediction model",
                "target_mae_percentage": self.config.target_mae_percentage,
                "target_r2_score": self.config.target_r2_score,
                "max_latency_ms": self.config.max_latency_ms,
                "prediction_horizon_months": self.config.prediction_horizon_months,
                "features": self.numerical_features + self.categorical_features,
                "use_case": "customer_analytics",
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Deploy for serving
        deployment_id = self.model_server.deploy_model(
            model_id=model_id,
            deployment_config={
                "replicas": 3,
                "cpu": "1000m",
                "memory": "2Gi",
                "max_latency_ms": self.config.max_latency_ms,
                "health_check_path": "/health",
                "metrics_enabled": True,
                "autoscaling": {
                    "min_replicas": 2,
                    "max_replicas": 10,
                    "target_cpu": 70
                }
            }
        )
        
        logger.info(f"CLV model deployed successfully:")
        logger.info(f"  Model ID: {model_id}")
        logger.info(f"  Deployment ID: {deployment_id}")
        
        return deployment_id
    
    def calculate_business_impact(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate business impact metrics for CLV model"""
        
        # Marketing ROI calculation
        avg_clv = predictions_df['predicted_clv'].mean()
        customer_count = len(predictions_df)
        
        # Segment customers by predicted CLV
        high_value_threshold = predictions_df['predicted_clv'].quantile(0.8)
        medium_value_threshold = predictions_df['predicted_clv'].quantile(0.5)
        
        high_value_customers = (predictions_df['predicted_clv'] >= high_value_threshold).sum()
        medium_value_customers = (
            (predictions_df['predicted_clv'] >= medium_value_threshold) & 
            (predictions_df['predicted_clv'] < high_value_threshold)
        ).sum()
        low_value_customers = (predictions_df['predicted_clv'] < medium_value_threshold).sum()
        
        # Calculate targeted marketing impact
        # Assumption: 20% improvement in retention for targeted customers
        retention_improvement = 0.20
        
        # High-value customers get premium treatment
        high_value_impact = high_value_customers * avg_clv * 0.3 * retention_improvement
        
        # Medium-value customers get standard treatment  
        medium_value_impact = medium_value_customers * avg_clv * 0.2 * retention_improvement
        
        # Low-value customers get cost-effective treatment
        low_value_impact = low_value_customers * avg_clv * 0.1 * retention_improvement
        
        total_revenue_impact = high_value_impact + medium_value_impact + low_value_impact
        
        # Cost savings from targeted marketing
        # Assumption: 30% reduction in marketing costs through better targeting
        marketing_cost_savings = customer_count * 50 * 0.30  # $50 avg marketing cost per customer
        
        business_metrics = {
            "total_customers_scored": customer_count,
            "avg_predicted_clv": avg_clv,
            "high_value_customers": high_value_customers,
            "medium_value_customers": medium_value_customers,
            "low_value_customers": low_value_customers,
            "annual_revenue_impact": total_revenue_impact,
            "annual_cost_savings": marketing_cost_savings,
            "total_annual_benefit": total_revenue_impact + marketing_cost_savings,
            "roi_multiple": (total_revenue_impact + marketing_cost_savings) / 500000  # $500K implementation cost
        }
        
        return business_metrics

def main():
    """Main execution function for CLV model"""
    
    # Initialize configuration
    config = CLVModelConfig()
    
    # Create model instance
    clv_model = CustomerLifetimeValueModel(config)
    
    try:
        # Step 1: Generate training data
        logger.info("Step 1: Generating synthetic CLV training data...")
        training_data = clv_model.generate_synthetic_clv_data(n_samples=15000)
        
        # Step 2: Train model
        logger.info("Step 2: Training CLV model...")
        training_results = clv_model.train_model(training_data)
        
        if not training_results["meets_targets"]:
            logger.warning("CLV model does not meet target performance criteria")
            return False
        
        # Step 3: Deploy model
        logger.info("Step 3: Deploying CLV model...")
        deployment_id = clv_model.deploy_model(training_results["run_id"])
        
        # Step 4: Generate predictions for business impact analysis
        logger.info("Step 4: Calculating business impact...")
        validation_data = clv_model.generate_synthetic_clv_data(n_samples=5000)
        X_val, _ = clv_model.preprocess_data(validation_data, is_training=False)
        predictions = clv_model.model.predict(X_val)
        
        validation_data['predicted_clv'] = predictions
        business_impact = clv_model.calculate_business_impact(validation_data)
        
        # Summary
        logger.info("\n🎉 CLV MODEL DEPLOYMENT SUCCESSFUL!")
        logger.info("="*60)
        logger.info(f"Model Performance:")
        logger.info(f"  Test MAE: {training_results['test_mae_percentage']:.1%}")
        logger.info(f"  Test R²: {training_results['test_r2']:.4f}")
        logger.info(f"Business Impact:")
        logger.info(f"  Annual Revenue Impact: ${business_impact['annual_revenue_impact']:,.0f}")
        logger.info(f"  Annual Cost Savings: ${business_impact['annual_cost_savings']:,.0f}")
        logger.info(f"  Total Annual Benefit: ${business_impact['total_annual_benefit']:,.0f}")
        logger.info(f"  ROI Multiple: {business_impact['roi_multiple']:.1f}x")
        logger.info(f"Deployment:")
        logger.info(f"  Deployment ID: {deployment_id}")
        logger.info(f"  MLflow Run ID: {training_results['run_id']}")
        
        return True
        
    except Exception as e:
        logger.error(f"CLV model pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)