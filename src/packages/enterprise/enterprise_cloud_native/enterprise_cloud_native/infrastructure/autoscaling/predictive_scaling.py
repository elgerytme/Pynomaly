"""
Predictive Scaling Engine

Implements machine learning-based predictive scaling using historical
metrics and patterns to proactively scale resources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from uuid import UUID

from structlog import get_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...domain.entities.autoscaling import PredictiveScalingPolicy

logger = get_logger(__name__)


class PredictiveScalingEngine:
    """
    Predictive scaling engine using machine learning models.
    
    Analyzes historical resource usage patterns and workload metrics
    to predict future resource requirements and scaling actions.
    """
    
    def __init__(self):
        self.models: Dict[UUID, Any] = {}  # Policy ID -> ML model
        self.scalers: Dict[UUID, StandardScaler] = {}  # Policy ID -> feature scaler
        self.training_data: Dict[UUID, pd.DataFrame] = {}  # Policy ID -> training data
        
        # Model configurations
        self.model_configs = {
            "linear": LinearRegression,
            "random_forest": RandomForestRegressor,
            "ensemble": None  # Will use ensemble of models
        }
        
        logger.info("PredictiveScalingEngine initialized")
    
    async def initialize_model(
        self,
        policy: PredictiveScalingPolicy,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Initialize ML model for a predictive scaling policy."""
        logger.info("Initializing ML model", policy_id=policy.id, model_type=policy.prediction_model_type)
        
        try:
            # Initialize model based on type
            if policy.prediction_model_type == "ensemble":
                model = self._create_ensemble_model(policy)
            else:
                model_class = self.model_configs.get(policy.prediction_model_type, LinearRegression)
                model = model_class(**policy.model_config)
            
            # Initialize feature scaler
            scaler = StandardScaler()
            
            # Store model and scaler
            self.models[policy.id] = model
            self.scalers[policy.id] = scaler
            
            # Initialize with historical data if provided
            if historical_data is not None and len(historical_data) > 0:
                await self.train_model(policy, historical_data)
            
            return {
                "status": "initialized",
                "model_type": policy.prediction_model_type,
                "features": policy.feature_columns,
                "target": policy.target_metric
            }
            
        except Exception as e:
            logger.error("Failed to initialize model", policy_id=policy.id, error=str(e))
            raise
    
    async def train_model(
        self,
        policy: PredictiveScalingPolicy,
        training_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train ML model with historical data."""
        logger.info("Training ML model", policy_id=policy.id, data_points=len(training_data))
        
        try:
            # Validate data
            if len(training_data) < 10:
                raise ValueError("Insufficient training data (minimum 10 points required)")
            
            # Prepare features and target
            features = training_data[policy.feature_columns].values
            target = training_data[policy.target_metric].values
            
            # Handle missing values
            features = np.nan_to_num(features)
            target = np.nan_to_num(target)
            
            # Scale features
            scaler = self.scalers[policy.id]
            features_scaled = scaler.fit_transform(features)
            
            # Train model
            model = self.models[policy.id]
            
            if policy.prediction_model_type == "ensemble":
                accuracy = await self._train_ensemble_model(model, features_scaled, target)
            else:
                model.fit(features_scaled, target)
                
                # Calculate accuracy
                predictions = model.predict(features_scaled)
                accuracy = 1.0 - mean_absolute_error(target, predictions) / (np.max(target) - np.min(target) + 1e-8)
                accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
            
            # Store training data
            self.training_data[policy.id] = training_data
            
            # Update policy with training results
            policy.record_training(accuracy)
            
            logger.info("Model training completed", policy_id=policy.id, accuracy=accuracy)
            
            return {
                "status": "trained",
                "accuracy": accuracy,
                "data_points": len(training_data),
                "features": policy.feature_columns,
                "model_params": self._get_model_parameters(model, policy.prediction_model_type)
            }
            
        except Exception as e:
            logger.error("Failed to train model", policy_id=policy.id, error=str(e))
            raise
    
    async def predict_scaling_action(
        self,
        policy: PredictiveScalingPolicy,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict scaling action based on current metrics."""
        logger.debug("Predicting scaling action", policy_id=policy.id)
        
        try:
            # Check if model is trained
            if policy.id not in self.models or not policy.last_trained_at:
                return {
                    "action": "none",
                    "reason": "Model not trained",
                    "confidence": 0.0
                }
            
            # Prepare features
            feature_values = []
            for feature in policy.feature_columns:
                if feature in current_metrics:
                    feature_values.append(current_metrics[feature])
                else:
                    # Use historical average if feature missing
                    if policy.id in self.training_data:
                        avg_value = self.training_data[policy.id][feature].mean()
                        feature_values.append(avg_value)
                    else:
                        feature_values.append(0.0)
            
            features = np.array([feature_values])
            
            # Scale features
            scaler = self.scalers[policy.id]
            features_scaled = scaler.transform(features)
            
            # Make prediction
            model = self.models[policy.id]
            
            if policy.prediction_model_type == "ensemble":
                predicted_value, confidence = await self._predict_with_ensemble(model, features_scaled)
            else:
                predicted_value = model.predict(features_scaled)[0]
                confidence = await self._calculate_prediction_confidence(policy, features_scaled, predicted_value)
            
            # Determine scaling action
            action = "none"
            if predicted_value > policy.scale_up_threshold:
                action = "scale_up"
            elif predicted_value < policy.scale_down_threshold:
                action = "scale_down"
            
            # Calculate scaling magnitude
            scaling_magnitude = await self._calculate_scaling_magnitude(policy, predicted_value, current_metrics)
            
            result = {
                "action": action,
                "predicted_value": float(predicted_value),
                "confidence": float(confidence),
                "scaling_magnitude": scaling_magnitude,
                "threshold_upper": policy.scale_up_threshold,
                "threshold_lower": policy.scale_down_threshold,
                "current_metrics": current_metrics,
                "prediction_horizon": policy.prediction_horizon_minutes
            }
            
            # Update policy with prediction
            policy.update_prediction(result, confidence)
            
            logger.debug("Scaling prediction completed", 
                        policy_id=policy.id, action=action, confidence=confidence)
            
            return result
            
        except Exception as e:
            logger.error("Failed to predict scaling action", policy_id=policy.id, error=str(e))
            return {
                "action": "none",
                "reason": f"Prediction error: {str(e)}",
                "confidence": 0.0
            }
    
    async def update_model_with_feedback(
        self,
        policy: PredictiveScalingPolicy,
        actual_metrics: Dict[str, float],
        prediction_timestamp: datetime
    ) -> Dict[str, Any]:
        """Update model with actual outcome feedback."""
        logger.debug("Updating model with feedback", policy_id=policy.id)
        
        try:
            # Add new data point to training data
            if policy.id in self.training_data:
                new_row = {**actual_metrics, "timestamp": prediction_timestamp}
                new_data = pd.DataFrame([new_row])
                self.training_data[policy.id] = pd.concat([
                    self.training_data[policy.id],
                    new_data
                ], ignore_index=True)
                
                # Keep only recent data
                cutoff_date = datetime.utcnow() - timedelta(days=policy.historical_data_days)
                self.training_data[policy.id] = self.training_data[policy.id][
                    self.training_data[policy.id]["timestamp"] >= cutoff_date
                ]
                
                # Retrain if enough new data accumulated
                if len(self.training_data[policy.id]) % 50 == 0:  # Retrain every 50 data points
                    await self.train_model(policy, self.training_data[policy.id])
            
            return {"status": "updated", "data_points": len(self.training_data.get(policy.id, []))}
            
        except Exception as e:
            logger.error("Failed to update model with feedback", policy_id=policy.id, error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def get_model_metrics(
        self,
        policy: PredictiveScalingPolicy
    ) -> Dict[str, Any]:
        """Get model performance metrics."""
        try:
            if policy.id not in self.models or policy.id not in self.training_data:
                return {"error": "Model not found or not trained"}
            
            training_data = self.training_data[policy.id]
            if len(training_data) < 10:
                return {"error": "Insufficient training data"}
            
            # Prepare data
            features = training_data[policy.feature_columns].values
            target = training_data[policy.target_metric].values
            
            features = np.nan_to_num(features)
            target = np.nan_to_num(target)
            
            # Scale features
            scaler = self.scalers[policy.id]
            features_scaled = scaler.transform(features)
            
            # Make predictions
            model = self.models[policy.id]
            if policy.prediction_model_type == "ensemble":
                predictions = []
                for i in range(len(features_scaled)):
                    pred, _ = await self._predict_with_ensemble(model, features_scaled[i:i+1])
                    predictions.append(pred)
                predictions = np.array(predictions)
            else:
                predictions = model.predict(features_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(target, predictions)
            mse = mean_squared_error(target, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate accuracy
            accuracy = 1.0 - mae / (np.max(target) - np.min(target) + 1e-8)
            accuracy = max(0.0, min(1.0, accuracy))
            
            return {
                "model_type": policy.prediction_model_type,
                "training_samples": len(training_data),
                "accuracy": float(accuracy),
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "last_trained": policy.last_trained_at.isoformat() if policy.last_trained_at else None,
                "features": policy.feature_columns,
                "target": policy.target_metric
            }
            
        except Exception as e:
            logger.error("Failed to get model metrics", policy_id=policy.id, error=str(e))
            return {"error": str(e)}
    
    def cleanup_model(self, policy_id: UUID) -> bool:
        """Clean up model resources for a policy."""
        logger.info("Cleaning up model resources", policy_id=policy_id)
        
        try:
            self.models.pop(policy_id, None)
            self.scalers.pop(policy_id, None)
            self.training_data.pop(policy_id, None)
            
            return True
            
        except Exception as e:
            logger.error("Failed to cleanup model", policy_id=policy_id, error=str(e))
            return False
    
    # Private helper methods
    
    def _create_ensemble_model(self, policy: PredictiveScalingPolicy) -> Dict[str, Any]:
        """Create ensemble model combining multiple algorithms."""
        return {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "weights": [0.3, 0.7]  # Weights for ensemble
        }
    
    async def _train_ensemble_model(
        self,
        ensemble: Dict[str, Any],
        features: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Train ensemble model."""
        # Train individual models
        ensemble["linear"].fit(features, target)
        ensemble["random_forest"].fit(features, target)
        
        # Calculate ensemble accuracy
        linear_pred = ensemble["linear"].predict(features)
        rf_pred = ensemble["random_forest"].predict(features)
        
        # Weighted ensemble prediction
        weights = ensemble["weights"]
        ensemble_pred = weights[0] * linear_pred + weights[1] * rf_pred
        
        # Calculate accuracy
        accuracy = 1.0 - mean_absolute_error(target, ensemble_pred) / (np.max(target) - np.min(target) + 1e-8)
        return max(0.0, min(1.0, accuracy))
    
    async def _predict_with_ensemble(
        self,
        ensemble: Dict[str, Any],
        features: np.ndarray
    ) -> Tuple[float, float]:
        """Make prediction with ensemble model."""
        # Get predictions from individual models
        linear_pred = ensemble["linear"].predict(features)[0]
        rf_pred = ensemble["random_forest"].predict(features)[0]
        
        # Weighted ensemble prediction
        weights = ensemble["weights"]
        prediction = weights[0] * linear_pred + weights[1] * rf_pred
        
        # Calculate confidence (inverse of prediction variance)
        variance = weights[0] * (linear_pred - prediction) ** 2 + weights[1] * (rf_pred - prediction) ** 2
        confidence = 1.0 / (1.0 + variance)
        
        return prediction, confidence
    
    async def _calculate_prediction_confidence(
        self,
        policy: PredictiveScalingPolicy,
        features: np.ndarray,
        prediction: float
    ) -> float:
        """Calculate prediction confidence based on historical performance."""
        try:
            if policy.id not in self.training_data:
                return 0.5  # Default confidence
            
            training_data = self.training_data[policy.id]
            if len(training_data) < 10:
                return 0.5
            
            # Calculate prediction confidence based on model accuracy and data quality
            base_confidence = policy.model_accuracy or 0.5
            
            # Adjust based on feature completeness
            feature_completeness = len([f for f in policy.feature_columns if f in training_data.columns])
            feature_completeness_ratio = feature_completeness / len(policy.feature_columns)
            
            # Adjust based on data recency
            latest_data = training_data["timestamp"].max() if "timestamp" in training_data.columns else datetime.utcnow()
            if isinstance(latest_data, str):
                latest_data = datetime.fromisoformat(latest_data)
            
            data_age_hours = (datetime.utcnow() - latest_data).total_seconds() / 3600
            recency_factor = max(0.1, 1.0 - (data_age_hours / (24 * policy.historical_data_days)))
            
            # Combined confidence
            confidence = base_confidence * feature_completeness_ratio * recency_factor
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default confidence on error
    
    async def _calculate_scaling_magnitude(
        self,
        policy: PredictiveScalingPolicy,
        predicted_value: float,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate scaling magnitude based on prediction."""
        try:
            current_value = current_metrics.get(policy.target_metric, 0.5)
            
            # Calculate percentage change
            change_percent = ((predicted_value - current_value) / max(current_value, 0.1)) * 100
            
            # Determine scaling factor
            if abs(change_percent) < 10:
                scaling_factor = 1.1  # Small adjustment
            elif abs(change_percent) < 30:
                scaling_factor = 1.25  # Moderate adjustment
            else:
                scaling_factor = 1.5  # Large adjustment
            
            if change_percent < 0:
                scaling_factor = 1 / scaling_factor
            
            return {
                "change_percent": float(change_percent),
                "scaling_factor": float(scaling_factor),
                "urgency": "low" if abs(change_percent) < 20 else "medium" if abs(change_percent) < 50 else "high"
            }
            
        except Exception:
            return {
                "change_percent": 0.0,
                "scaling_factor": 1.0,
                "urgency": "low"
            }
    
    def _get_model_parameters(self, model: Any, model_type: str) -> Dict[str, Any]:
        """Get model parameters for logging."""
        try:
            if model_type == "linear":
                return {
                    "coefficients": model.coef_.tolist() if hasattr(model, "coef_") else [],
                    "intercept": float(model.intercept_) if hasattr(model, "intercept_") else 0.0
                }
            elif model_type == "random_forest":
                return {
                    "n_estimators": model.n_estimators if hasattr(model, "n_estimators") else 100,
                    "max_depth": model.max_depth if hasattr(model, "max_depth") else None,
                    "feature_importances": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else []
                }
            elif model_type == "ensemble":
                return {
                    "models": ["linear", "random_forest"],
                    "weights": model.get("weights", [])
                }
            else:
                return {}
        except Exception:
            return {}