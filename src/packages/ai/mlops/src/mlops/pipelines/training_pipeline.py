"""
Advanced ML Training Pipeline

This module implements a comprehensive ML training pipeline with automated
feature engineering, hyperparameter optimization, model validation, and deployment.
"""

import logging
import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import boto3
from google.cloud import storage as gcs
import redis
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for ML training pipeline"""
    pipeline_id: str
    model_type: str
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cross_validation_folds: int = 5
    hyperparameter_optimization: str = "optuna"  # optuna, grid, random
    n_trials: int = 100
    scoring_metric: str = "accuracy"
    model_registry_name: str = "mlops_models"
    staging_threshold: float = 0.85
    production_threshold: float = 0.90
    feature_importance_threshold: float = 0.01
    drift_detection_enabled: bool = True
    auto_deployment_enabled: bool = False

@dataclass
class TrainingMetrics:
    """Training metrics data structure"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_score: float
    training_time: float
    model_size_mb: float
    feature_importance: Dict[str, float]

@dataclass
class PipelineResult:
    """Pipeline execution result"""
    pipeline_id: str
    model_name: str
    model_version: str
    metrics: TrainingMetrics
    model_path: str
    feature_names: List[str]
    preprocessing_artifacts: Dict[str, str]
    status: str
    error_message: Optional[str] = None
    execution_time: float = 0.0

class FeatureEngineer:
    """Advanced feature engineering capabilities"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
    
    def engineer_features(self, df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        logger.info("Starting feature engineering...")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Create datetime features
        df_processed = self._create_datetime_features(df_processed)
        
        # Create interaction features
        df_processed = self._create_interaction_features(df_processed, config.feature_columns)
        
        # Create aggregation features
        df_processed = self._create_aggregation_features(df_processed)
        
        # Encode categorical variables
        df_processed = self._encode_categorical_features(df_processed)
        
        # Scale numerical features
        df_processed = self._scale_numerical_features(df_processed)
        
        # Feature selection based on importance
        df_processed = self._select_important_features(df_processed, config)
        
        logger.info(f"Feature engineering completed. Shape: {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent imputation"""
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # Use median for numerical columns
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    # Use mode for categorical columns
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'unknown', inplace=True)
        return df
    
    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        numerical_columns = df[feature_columns].select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) >= 2:
            for i, col1 in enumerate(numerical_columns):
                for col2 in numerical_columns[i+1:i+3]:  # Limit to avoid feature explosion
                    if col1 != col2:
                        df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                        df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features"""
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) > 0:
            df['numerical_sum'] = df[numerical_columns].sum(axis=1)
            df['numerical_mean'] = df[numerical_columns].mean(axis=1)
            df['numerical_std'] = df[numerical_columns].std(axis=1)
            df['numerical_max'] = df[numerical_columns].max(axis=1)
            df['numerical_min'] = df[numerical_columns].min(axis=1)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if df[col].nunique() <= 10:  # One-hot encode low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=[f'{col}_{cat}' for cat in encoder.categories_[0]],
                    index=df.index
                )
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
                self.encoders[col] = encoder
            else:  # Label encode high cardinality
                encoder = LabelEncoder()
                df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
                df.drop(col, axis=1, inplace=True)
                self.encoders[col] = encoder
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            self.scalers['numerical'] = scaler
        
        return df
    
    def _select_important_features(self, df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
        """Select important features based on variance threshold"""
        from sklearn.feature_selection import VarianceThreshold
        
        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        selected_features = selector.fit_transform(df.drop(config.target_column, axis=1, errors='ignore'))
        
        feature_names = df.drop(config.target_column, axis=1, errors='ignore').columns[selector.get_support()]
        selected_df = pd.DataFrame(selected_features, columns=feature_names, index=df.index)
        
        # Add target column back if it exists
        if config.target_column in df.columns:
            selected_df[config.target_column] = df[config.target_column]
        
        return selected_df

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.study = None
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using specified method"""
        logger.info(f"Starting hyperparameter optimization using {self.config.hyperparameter_optimization}")
        
        if self.config.hyperparameter_optimization == "optuna":
            return self._optimize_with_optuna(X_train, y_train, X_val, y_val)
        elif self.config.hyperparameter_optimization == "grid":
            return self._optimize_with_grid_search(X_train, y_train)
        elif self.config.hyperparameter_optimization == "random":
            return self._optimize_with_random_search(X_train, y_train)
        else:
            logger.warning("No optimization method specified, using default parameters")
            return self._get_default_params()
    
    def _optimize_with_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize using Optuna"""
        def objective(trial):
            params = self._suggest_params(trial)
            model = self._create_model(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.config.n_trials)
        
        return self.study.best_params
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna trial"""
        if self.config.model_type == "random_forest":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
        elif self.config.model_type == "gradient_boosting":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        elif self.config.model_type == "logistic_regression":
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
        else:
            return {}
    
    def _optimize_with_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize using Grid Search"""
        param_grid = self._get_param_grid()
        model = self._create_model({})
        
        grid_search = GridSearchCV(
            model, param_grid, cv=self.config.cross_validation_folds,
            scoring=self.config.scoring_metric, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_
    
    def _optimize_with_random_search(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize using Random Search"""
        param_distributions = self._get_param_distributions()
        model = self._create_model({})
        
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=self.config.n_trials,
            cv=self.config.cross_validation_folds, scoring=self.config.scoring_metric,
            n_jobs=-1, random_state=self.config.random_state
        )
        random_search.fit(X_train, y_train)
        
        return random_search.best_params_
    
    def _create_model(self, params: Dict[str, Any]):
        """Create model with given parameters"""
        if self.config.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.config.random_state, **params)
        elif self.config.model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.config.random_state, **params)
        elif self.config.model_type == "logistic_regression":
            return LogisticRegression(random_state=self.config.random_state, **params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _get_param_grid(self) -> Dict[str, List]:
        """Get parameter grid for grid search"""
        if self.config.model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.config.model_type == "gradient_boosting":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.config.model_type == "logistic_regression":
            return {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        else:
            return {}
    
    def _get_param_distributions(self) -> Dict[str, Any]:
        """Get parameter distributions for random search"""
        from scipy.stats import randint, uniform
        
        if self.config.model_type == "random_forest":
            return {
                'n_estimators': randint(50, 300),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            }
        elif self.config.model_type == "gradient_boosting":
            return {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4)
            }
        elif self.config.model_type == "logistic_regression":
            return {
                'C': uniform(0.001, 99.999),
                'solver': ['liblinear', 'lbfgs']
            }
        else:
            return {}
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters"""
        if self.config.model_type == "random_forest":
            return {'n_estimators': 100, 'max_depth': 10, 'random_state': self.config.random_state}
        elif self.config.model_type == "gradient_boosting":
            return {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': self.config.random_state}
        elif self.config.model_type == "logistic_regression":
            return {'C': 1.0, 'random_state': self.config.random_state}
        else:
            return {}

class MLTrainingPipeline:
    """Advanced ML training pipeline with comprehensive capabilities"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.optimizer = HyperparameterOptimizer(config)
        self.mlflow_client = MlflowClient()
        self.model = None
        self.best_params = None
        
        # Initialize MLflow
        mlflow.set_experiment(f"mlops_{config.pipeline_id}")
    
    async def run_pipeline(self, data_path: str) -> PipelineResult:
        """Run the complete ML training pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting ML training pipeline: {self.config.pipeline_id}")
        
        try:
            # Load and prepare data
            df = await self._load_data(data_path)
            
            # Feature engineering
            df_processed = self.feature_engineer.engineer_features(df, self.config)
            
            # Split data
            X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(df_processed)
            
            # Hyperparameter optimization
            self.best_params = self.optimizer.optimize(X_train, y_train, X_val, y_val)
            
            # Train final model
            self.model = self._train_final_model(X_train, y_train, self.best_params)
            
            # Evaluate model
            metrics = self._evaluate_model(X_test, y_test, X_train, y_train)
            
            # Save model and artifacts
            model_path, preprocessing_artifacts = await self._save_model_artifacts()
            
            # Register model in MLflow
            model_version = await self._register_model(metrics)
            
            # Generate pipeline result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = PipelineResult(
                pipeline_id=self.config.pipeline_id,
                model_name=f"{self.config.model_type}_{self.config.pipeline_id}",
                model_version=model_version,
                metrics=metrics,
                model_path=model_path,
                feature_names=list(X_train.columns) if hasattr(X_train, 'columns') else [],
                preprocessing_artifacts=preprocessing_artifacts,
                status="success",
                execution_time=execution_time
            )
            
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                pipeline_id=self.config.pipeline_id,
                model_name="",
                model_version="",
                metrics=TrainingMetrics(0, 0, 0, 0, 0, 0, 0, 0, {}),
                model_path="",
                feature_names=[],
                preprocessing_artifacts={},
                status="failed",
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various sources"""
        logger.info(f"Loading data from: {data_path}")
        
        if data_path.startswith('s3://'):
            # Load from S3
            return await self._load_from_s3(data_path)
        elif data_path.startswith('gs://'):
            # Load from Google Cloud Storage
            return await self._load_from_gcs(data_path)
        elif data_path.startswith('postgresql://'):
            # Load from PostgreSQL
            return await self._load_from_postgres(data_path)
        else:
            # Load from local file
            return pd.read_csv(data_path)
    
    async def _load_from_s3(self, s3_path: str) -> pd.DataFrame:
        """Load data from S3"""
        s3 = boto3.client('s3')
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    
    async def _load_from_gcs(self, gcs_path: str) -> pd.DataFrame:
        """Load data from Google Cloud Storage"""
        client = gcs.Client()
        bucket_name, blob_name = gcs_path.replace('gs://', '').split('/', 1)
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        content = blob.download_as_text()
        from io import StringIO
        return pd.read_csv(StringIO(content))
    
    async def _load_from_postgres(self, connection_string: str) -> pd.DataFrame:
        """Load data from PostgreSQL"""
        engine = create_engine(connection_string)
        query = f"SELECT * FROM {self.config.pipeline_id}_data"
        return pd.read_sql(query, engine)
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Split data into train, test, and validation sets"""
        logger.info("Splitting data into train/test/validation sets")
        
        # Separate features and target
        X = df.drop(self.config.target_column, axis=1)
        y = df[self.config.target_column]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        # Second split: train and validation
        val_size = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.config.random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def _train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          best_params: Dict[str, Any]):
        """Train the final model with best parameters"""
        logger.info("Training final model with optimized parameters")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", self.config.model_type)
            mlflow.log_param("pipeline_id", self.config.pipeline_id)
            
            # Create and train model
            model = self.optimizer._create_model(best_params)
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            return model
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series,
                       X_train: pd.DataFrame, y_train: pd.Series) -> TrainingMetrics:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model performance")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (for binary classification)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                   cv=self.config.cross_validation_folds)
        cv_score = cv_scores.mean()
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
            feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Model size
        model_size_mb = len(pickle.dumps(self.model)) / (1024 * 1024)
        
        metrics = TrainingMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cross_val_score=cv_score,
            training_time=0.0,  # Would need to track this during training
            model_size_mb=model_size_mb,
            feature_importance=feature_importance
        )
        
        # Log metrics to MLflow
        mlflow.log_metrics(asdict(metrics))
        
        logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics
    
    async def _save_model_artifacts(self) -> Tuple[str, Dict[str, str]]:
        """Save model and preprocessing artifacts"""
        logger.info("Saving model and preprocessing artifacts")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.model_type}_{self.config.pipeline_id}_{timestamp}"
        
        # Save model
        model_path = f"models/{model_name}.pkl"
        joblib.dump(self.model, model_path)
        
        # Save preprocessing artifacts
        preprocessing_artifacts = {}
        
        # Save scalers
        if self.feature_engineer.scalers:
            scaler_path = f"artifacts/{model_name}_scalers.pkl"
            joblib.dump(self.feature_engineer.scalers, scaler_path)
            preprocessing_artifacts['scalers'] = scaler_path
        
        # Save encoders
        if self.feature_engineer.encoders:
            encoder_path = f"artifacts/{model_name}_encoders.pkl"
            joblib.dump(self.feature_engineer.encoders, encoder_path)
            preprocessing_artifacts['encoders'] = encoder_path
        
        # Save feature stats
        if self.feature_engineer.feature_stats:
            stats_path = f"artifacts/{model_name}_feature_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.feature_engineer.feature_stats, f)
            preprocessing_artifacts['feature_stats'] = stats_path
        
        return model_path, preprocessing_artifacts
    
    async def _register_model(self, metrics: TrainingMetrics) -> str:
        """Register model in MLflow model registry"""
        logger.info("Registering model in MLflow registry")
        
        model_name = f"{self.config.model_type}_{self.config.pipeline_id}"
        
        # Register model
        model_version = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name
        )
        
        # Determine stage based on performance
        stage = "None"
        if metrics.accuracy >= self.config.production_threshold:
            stage = "Production"
        elif metrics.accuracy >= self.config.staging_threshold:
            stage = "Staging"
        
        # Transition model to appropriate stage
        if stage != "None":
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
        
        logger.info(f"Model registered as version {model_version.version} in stage {stage}")
        return model_version.version

# Pipeline orchestration functions
async def run_training_pipeline(config: PipelineConfig, data_path: str) -> PipelineResult:
    """Run a complete training pipeline"""
    pipeline = MLTrainingPipeline(config)
    return await pipeline.run_pipeline(data_path)

async def run_batch_training(configs: List[PipelineConfig], data_paths: List[str]) -> List[PipelineResult]:
    """Run multiple training pipelines in parallel"""
    logger.info(f"Starting batch training of {len(configs)} pipelines")
    
    tasks = [
        run_training_pipeline(config, data_path)
        for config, data_path in zip(configs, data_paths)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = [r for r in results if isinstance(r, PipelineResult) and r.status == "success"]
    failed_results = [r for r in results if not isinstance(r, PipelineResult) or r.status == "failed"]
    
    logger.info(f"Batch training completed - Success: {len(successful_results)}, Failed: {len(failed_results)}")
    
    return results