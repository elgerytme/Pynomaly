"""
Enhanced ML Pipeline Framework for Anomaly Detection

Comprehensive machine learning pipeline framework implementing:
- DAG-based workflow orchestration
- Model training with multiple algorithms  
- Hyperparameter optimization with Optuna
- Model evaluation and selection
- AutoML capabilities for feature engineering and algorithm recommendation

This addresses Issue #143: Phase 2.2: Data Science Package - Machine Learning Pipeline Framework
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from pynomaly.domain.entities.pipeline import Pipeline, PipelineStep, PipelineType, StepType
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger


class MLPipelineStage(Enum):
    """Stages in ML pipeline for anomaly detection."""
    
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    ALGORITHM_SELECTION = "algorithm_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_SELECTION = "model_selection"
    MODEL_VALIDATION = "model_validation"
    PERFORMANCE_TESTING = "performance_testing"
    MODEL_DEPLOYMENT = "model_deployment"


class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter tuning."""
    
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall" 
    MAXIMIZE_F1 = "maximize_f1"
    MAXIMIZE_ROC_AUC = "maximize_roc_auc"
    MINIMIZE_FALSE_POSITIVES = "minimize_false_positives"
    BALANCED_PRECISION_RECALL = "balanced_precision_recall"


class FeatureEngineeringStrategy(Enum):
    """Feature engineering strategies for anomaly detection."""
    
    STATISTICAL_FEATURES = "statistical_features"
    TEMPORAL_FEATURES = "temporal_features"
    INTERACTION_FEATURES = "interaction_features"
    POLYNOMIAL_FEATURES = "polynomial_features"
    DOMAIN_SPECIFIC = "domain_specific"
    AUTO_FEATURES = "auto_features"


@dataclass
class MLPipelineConfig:
    """Configuration for ML pipeline execution."""
    
    pipeline_id: UUID = field(default_factory=uuid4)
    pipeline_name: str = ""
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    feature_engineering_config: Dict[str, Any] = field(default_factory=dict)
    algorithms_to_evaluate: List[str] = field(default_factory=lambda: [
        "IsolationForest", "OneClassSVM", "LocalOutlierFactor", "EllipticEnvelope"
    ])
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1", "roc_auc"
    ])
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    max_trials: int = 100
    timeout_seconds: int = 3600  # 1 hour default
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        "min_precision": 0.7,
        "min_recall": 0.6,
        "min_f1": 0.65,
        "min_roc_auc": 0.75
    })
    gpu_acceleration: bool = False
    distributed_processing: bool = False
    feature_selection_enabled: bool = True
    auto_feature_engineering: bool = True
    ensemble_methods: bool = True


@dataclass
class MLExperimentResult:
    """Results from ML pipeline experiment."""
    
    experiment_id: UUID = field(default_factory=uuid4)
    pipeline_config: MLPipelineConfig = field(default_factory=MLPipelineConfig)
    algorithm_name: str = ""
    best_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: Dict[str, List[float]] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_artifacts: Dict[str, str] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    training_dataset_size: int = 0
    validation_dataset_size: int = 0
    optimization_trials: int = 0
    best_trial_number: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    error_message: Optional[str] = None


@dataclass
class AutoMLRecommendation:
    """AutoML recommendations for algorithm and configuration."""
    
    recommended_algorithms: List[str] = field(default_factory=list)
    algorithm_rankings: Dict[str, float] = field(default_factory=dict)
    dataset_characteristics: Dict[str, Any] = field(default_factory=dict)
    feature_engineering_recommendations: List[str] = field(default_factory=list)
    hyperparameter_recommendations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


class MLPipelineFramework:
    """Enhanced ML Pipeline Framework for Anomaly Detection.
    
    Provides comprehensive machine learning pipeline capabilities including:
    - DAG-based workflow orchestration
    - Multi-algorithm model training
    - Hyperparameter optimization with Optuna
    - Model evaluation and selection
    - AutoML capabilities
    """
    
    def __init__(
        self,
        storage_directory: str = "ml_pipeline_artifacts",
        enable_logging: bool = True,
        log_level: str = "INFO"
    ):
        """Initialize the ML Pipeline Framework.
        
        Args:
            storage_directory: Directory for storing pipeline artifacts
            enable_logging: Whether to enable structured logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        if enable_logging:
            self.logger = StructuredLogger("ml_pipeline_framework")
            self.logger.setLevel(getattr(logging, log_level.upper()))
        else:
            self.logger = logging.getLogger("ml_pipeline_framework")
            self.logger.addHandler(logging.NullHandler())
        
        # Pipeline storage
        self.experiments: Dict[UUID, MLExperimentResult] = {}
        self.active_pipelines: Dict[UUID, Pipeline] = {}
        
        # Algorithm registry
        self._initialize_algorithm_registry()
        
        # Feature engineering functions
        self._initialize_feature_engineering()
        
        self.logger.info("ML Pipeline Framework initialized")
    
    def _initialize_algorithm_registry(self) -> None:
        """Initialize the algorithm registry with available algorithms."""
        self.algorithm_registry = {
            "IsolationForest": {
                "class": IsolationForest,
                "type": "unsupervised",
                "suitable_for": ["tabular", "numeric"],
                "hyperparameter_space": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_samples": ["auto", 0.5, 0.7, 0.9],
                    "contamination": [0.01, 0.05, 0.1, 0.15, 0.2],
                    "max_features": [0.5, 0.7, 0.9, 1.0],
                    "bootstrap": [True, False]
                }
            },
            "OneClassSVM": {
                "class": None,  # Will be imported dynamically
                "type": "unsupervised", 
                "suitable_for": ["tabular", "numeric"],
                "hyperparameter_space": {
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                    "nu": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                    "degree": [2, 3, 4]  # For poly kernel
                }
            },
            "LocalOutlierFactor": {
                "class": None,  # Will be imported dynamically
                "type": "unsupervised",
                "suitable_for": ["tabular", "numeric"], 
                "hyperparameter_space": {
                    "n_neighbors": [5, 10, 20, 30, 50],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [10, 20, 30, 40, 50],
                    "metric": ["minkowski", "euclidean", "manhattan"],
                    "contamination": [0.01, 0.05, 0.1, 0.15, 0.2]
                }
            },
            "EllipticEnvelope": {
                "class": None,  # Will be imported dynamically
                "type": "unsupervised",
                "suitable_for": ["tabular", "numeric"],
                "hyperparameter_space": {
                    "contamination": [0.01, 0.05, 0.1, 0.15, 0.2],
                    "support_fraction": [None, 0.5, 0.7, 0.9],
                    "store_precision": [True, False]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.algorithm_registry["XGBClassifier"] = {
                "class": xgb.XGBClassifier,
                "type": "supervised",
                "suitable_for": ["tabular", "numeric", "categorical"],
                "hyperparameter_space": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [3, 4, 5, 6, 7, 8],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 0.5, 1.0],
                    "reg_lambda": [0, 0.1, 0.5, 1.0]
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.algorithm_registry["LGBMClassifier"] = {
                "class": lgb.LGBMClassifier,
                "type": "supervised",
                "suitable_for": ["tabular", "numeric", "categorical"],
                "hyperparameter_space": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [3, 4, 5, 6, 7],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "num_leaves": [15, 31, 63, 127],
                    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
                    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 0.5, 1.0],
                    "reg_lambda": [0, 0.1, 0.5, 1.0]
                }
            }
            
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            self.algorithm_registry["CatBoostClassifier"] = {
                "class": cb.CatBoostClassifier,
                "type": "supervised",
                "suitable_for": ["tabular", "numeric", "categorical"],
                "hyperparameter_space": {
                    "iterations": [100, 200, 300, 500],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "l2_leaf_reg": [1, 3, 5, 7, 9],
                    "border_count": [32, 64, 128, 255],
                    "random_strength": [0, 1, 2, 3]
                }
            }
            
        self.logger.info(f"Initialized algorithm registry with {len(self.algorithm_registry)} algorithms")
    
    def _initialize_feature_engineering(self) -> None:
        """Initialize feature engineering functions."""
        self.feature_engineering_functions = {
            "statistical_features": self._create_statistical_features,
            "temporal_features": self._create_temporal_features,
            "interaction_features": self._create_interaction_features,
            "polynomial_features": self._create_polynomial_features,
            "domain_specific": self._create_domain_specific_features,
            "auto_features": self._create_auto_features
        }
        
        self.logger.info("Initialized feature engineering functions")
    
    async def create_ml_pipeline(
        self,
        pipeline_name: str,
        description: str,
        config: MLPipelineConfig,
        created_by: str = "system"
    ) -> Pipeline:
        """Create a new ML pipeline for anomaly detection.
        
        Args:
            pipeline_name: Name of the pipeline
            description: Description of the pipeline
            config: Pipeline configuration
            created_by: User creating the pipeline
            
        Returns:
            Created Pipeline entity
        """
        self.logger.info(f"Creating ML pipeline: {pipeline_name}")
        
        # Create pipeline entity
        pipeline = Pipeline(
            name=pipeline_name,
            description=description,
            pipeline_type=PipelineType.TRAINING,
            created_by=created_by,
            environment="development"
        )
        
        # Add pipeline steps based on configuration
        await self._build_pipeline_steps(pipeline, config)
        
        # Store pipeline
        self.active_pipelines[pipeline.id] = pipeline
        
        self.logger.info(f"Created ML pipeline {pipeline.id} with {len(pipeline.steps)} steps")
        return pipeline
    
    async def _build_pipeline_steps(self, pipeline: Pipeline, config: MLPipelineConfig) -> None:
        """Build pipeline steps based on configuration."""
        
        step_order = 1
        
        # Data loading step
        data_loading_step = PipelineStep(
            name="data_loading",
            step_type=StepType.DATA_LOADING,
            description="Load and validate input data for anomaly detection",
            order=step_order,
            configuration=config.dataset_config,
            timeout_seconds=300
        )
        pipeline.add_step(data_loading_step)
        step_order += 1
        
        # Data validation step
        data_validation_step = PipelineStep(
            name="data_validation",
            step_type=StepType.DATA_VALIDATION,
            description="Validate data quality and schema",
            order=step_order,
            configuration={"validation_rules": "anomaly_detection_schema"},
            dependencies=[data_loading_step.id],
            timeout_seconds=180
        )
        pipeline.add_step(data_validation_step)
        step_order += 1
        
        # Data preprocessing step
        preprocessing_step = PipelineStep(
            name="data_preprocessing",
            step_type=StepType.DATA_PREPROCESSING,
            description="Preprocess data for anomaly detection models",
            order=step_order,
            configuration=config.preprocessing_config,
            dependencies=[data_validation_step.id],
            timeout_seconds=600
        )
        pipeline.add_step(preprocessing_step)
        step_order += 1
        
        # Feature engineering step (if enabled)
        if config.auto_feature_engineering:
            feature_engineering_step = PipelineStep(
                name="feature_engineering",
                step_type=StepType.FEATURE_ENGINEERING,
                description="Generate features for anomaly detection",
                order=step_order,
                configuration=config.feature_engineering_config,
                dependencies=[preprocessing_step.id],
                timeout_seconds=900
            )
            pipeline.add_step(feature_engineering_step)
            step_order += 1
            last_step_id = feature_engineering_step.id
        else:
            last_step_id = preprocessing_step.id
        
        # Algorithm selection and training steps
        for algorithm in config.algorithms_to_evaluate:
            # Hyperparameter optimization step
            hyperparam_step = PipelineStep(
                name=f"hyperparameter_optimization_{algorithm}",
                step_type=StepType.MODEL_TRAINING,
                description=f"Optimize hyperparameters for {algorithm}",
                order=step_order,
                configuration={
                    "algorithm": algorithm,
                    "optimization_config": config.optimization_config,
                    "max_trials": config.max_trials
                },
                dependencies=[last_step_id],
                timeout_seconds=config.timeout_seconds
            )
            pipeline.add_step(hyperparam_step)
            step_order += 1
            
            # Model training step
            training_step = PipelineStep(
                name=f"model_training_{algorithm}",
                step_type=StepType.MODEL_TRAINING,
                description=f"Train {algorithm} model with optimized parameters",
                order=step_order,
                configuration={"algorithm": algorithm},
                dependencies=[hyperparam_step.id],
                timeout_seconds=1800
            )
            pipeline.add_step(training_step)
            step_order += 1
            
            # Model evaluation step
            evaluation_step = PipelineStep(
                name=f"model_evaluation_{algorithm}",
                step_type=StepType.MODEL_VALIDATION,
                description=f"Evaluate {algorithm} model performance",
                order=step_order,
                configuration={
                    "algorithm": algorithm,
                    "evaluation_metrics": config.evaluation_metrics
                },
                dependencies=[training_step.id],
                timeout_seconds=300
            )
            pipeline.add_step(evaluation_step)
            step_order += 1
        
        # Model selection step
        model_selection_step = PipelineStep(
            name="model_selection",
            step_type=StepType.MODEL_VALIDATION,
            description="Select best performing model",
            order=step_order,
            configuration={
                "selection_criteria": config.performance_targets,
                "algorithms": config.algorithms_to_evaluate
            },
            dependencies=[
                step.id for step in pipeline.steps 
                if step.name.startswith("model_evaluation_")
            ],
            timeout_seconds=180
        )
        pipeline.add_step(model_selection_step)
        step_order += 1
        
        # Model deployment step
        deployment_step = PipelineStep(
            name="model_deployment",
            step_type=StepType.MODEL_DEPLOYMENT,
            description="Deploy selected model for production use",
            order=step_order,
            configuration={"deployment_environment": "staging"},
            dependencies=[model_selection_step.id],
            timeout_seconds=600
        )
        pipeline.add_step(deployment_step)
    
    async def execute_pipeline(
        self,
        pipeline_id: UUID,
        input_data: Union[pd.DataFrame, np.ndarray, str],
        target_data: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Execute an ML pipeline for anomaly detection.
        
        Args:
            pipeline_id: ID of the pipeline to execute
            input_data: Input data (DataFrame, array, or file path)
            target_data: Target labels (for supervised methods)
            
        Returns:
            Pipeline execution results
        """
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.active_pipelines[pipeline_id]
        self.logger.info(f"Executing pipeline: {pipeline.name} ({pipeline_id})")
        
        start_time = time.time()
        execution_context = {
            "pipeline_id": pipeline_id,
            "start_time": start_time,
            "input_data": input_data,
            "target_data": target_data,
            "step_results": {},
            "artifacts": {},
            "current_data": None,
            "models": {},
            "performance_results": {}
        }
        
        try:
            # Execute pipeline steps in order
            for step in pipeline.execution_order:
                if not step.is_enabled:
                    continue
                
                self.logger.info(f"Executing step: {step.name}")
                step_start_time = time.time()
                
                # Execute step based on type
                step_result = await self._execute_pipeline_step(step, execution_context)
                
                step_duration = time.time() - step_start_time
                execution_context["step_results"][step.name] = {
                    "result": step_result,
                    "duration_seconds": step_duration,
                    "status": "completed"
                }
                
                self.logger.info(f"Completed step {step.name} in {step_duration:.2f}s")
            
            # Compile final results
            total_duration = time.time() - start_time
            
            results = {
                "pipeline_id": str(pipeline_id),
                "pipeline_name": pipeline.name,
                "execution_time_seconds": total_duration,
                "status": "completed",
                "step_results": execution_context["step_results"],
                "model_performance": execution_context["performance_results"],
                "best_model": self._select_best_model(execution_context["performance_results"]),
                "artifacts": execution_context["artifacts"],
                "execution_summary": self._generate_execution_summary(execution_context)
            }
            
            self.logger.info(f"Pipeline execution completed in {total_duration:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                "pipeline_id": str(pipeline_id),
                "status": "failed",
                "error_message": str(e),
                "execution_time_seconds": time.time() - start_time,
                "step_results": execution_context["step_results"]
            }
    
    async def _execute_pipeline_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        
        if step.step_type == StepType.DATA_LOADING:
            return await self._execute_data_loading_step(step, context)
        elif step.step_type == StepType.DATA_VALIDATION:
            return await self._execute_data_validation_step(step, context)
        elif step.step_type == StepType.DATA_PREPROCESSING:
            return await self._execute_preprocessing_step(step, context)
        elif step.step_type == StepType.FEATURE_ENGINEERING:
            return await self._execute_feature_engineering_step(step, context)
        elif step.step_type == StepType.MODEL_TRAINING:
            if "hyperparameter_optimization" in step.name:
                return await self._execute_hyperparameter_optimization_step(step, context)
            else:
                return await self._execute_model_training_step(step, context)
        elif step.step_type == StepType.MODEL_VALIDATION:
            if "evaluation" in step.name:
                return await self._execute_model_evaluation_step(step, context)
            else:
                return await self._execute_model_selection_step(step, context)
        elif step.step_type == StepType.MODEL_DEPLOYMENT:
            return await self._execute_model_deployment_step(step, context)
        else:
            raise ValueError(f"Unknown step type: {step.step_type}")
    
    async def _execute_data_loading_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data loading step."""
        
        input_data = context["input_data"]
        
        # Load data based on input type
        if isinstance(input_data, str):
            # File path
            if input_data.endswith('.csv'):
                data = pd.read_csv(input_data)
            elif input_data.endswith('.parquet'):
                data = pd.read_parquet(input_data)
            else:
                raise ValueError(f"Unsupported file format: {input_data}")
        elif isinstance(input_data, pd.DataFrame):
            data = input_data.copy()
        elif isinstance(input_data, np.ndarray):
            data = pd.DataFrame(input_data)
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
        
        # Store in context
        context["current_data"] = data
        
        return {
            "data_shape": data.shape,
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": data.isnull().sum().to_dict()
        }
    
    async def _execute_data_validation_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data validation step."""
        
        data = context["current_data"]
        
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for empty data
        if data.empty:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Dataset is empty")
        
        # Check for sufficient data size
        if len(data) < 100:
            validation_results["warnings"].append(
                f"Small dataset size: {len(data)} rows. Consider using more data."
            )
        
        # Check for excessive missing values
        missing_percentage = (data.isnull().sum() / len(data) * 100)
        high_missing_cols = missing_percentage[missing_percentage > 50].to_dict()
        if high_missing_cols:
            validation_results["warnings"].append(
                f"Columns with >50% missing values: {high_missing_cols}"
            )
        
        # Check for constant columns
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
            validation_results["warnings"].append(
                f"Constant columns detected: {constant_cols}"
            )
        
        return validation_results
    
    async def _execute_preprocessing_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data preprocessing step."""
        
        data = context["current_data"].copy()
        config = step.configuration
        
        # Handle missing values
        missing_strategy = config.get("missing_value_strategy", "mean")
        if missing_strategy == "drop":
            data = data.dropna()
        elif missing_strategy == "mean":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        elif missing_strategy == "median":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        # Handle categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            # Simple label encoding for categorical variables
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
        
        # Scale numeric features
        scaling_method = config.get("scaling_method", "standard")
        if scaling_method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        elif scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        # Update context
        context["current_data"] = data
        
        return {
            "processed_shape": data.shape,
            "scaling_method": scaling_method,
            "missing_value_strategy": missing_strategy,
            "categorical_columns_encoded": list(categorical_cols)
        }
    
    async def _execute_feature_engineering_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute feature engineering step."""
        
        data = context["current_data"].copy()
        config = step.configuration
        
        original_feature_count = len(data.columns)
        
        # Apply configured feature engineering strategies
        strategies = config.get("strategies", ["statistical_features"])
        
        for strategy in strategies:
            if strategy in self.feature_engineering_functions:
                data = self.feature_engineering_functions[strategy](data, config)
        
        # Update context
        context["current_data"] = data
        
        return {
            "original_features": original_feature_count,
            "engineered_features": len(data.columns),
            "new_features_added": len(data.columns) - original_feature_count,
            "strategies_applied": strategies
        }
    
    async def _execute_hyperparameter_optimization_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute hyperparameter optimization step."""
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, using default hyperparameters")
            return await self._execute_default_hyperparameter_selection(step, context)
        
        algorithm_name = step.configuration["algorithm"]
        max_trials = step.configuration.get("max_trials", 50)
        
        data = context["current_data"]
        target = context.get("target_data")
        
        # Split data for optimization
        if target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2, random_state=42
            )
        else:
            # For unsupervised methods, use entire dataset
            X_train = data
            X_test = data.sample(frac=0.3, random_state=42)
            y_train = None
            y_test = None
        
        # Create optimization study
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            return self._optimize_hyperparameters(
                trial, algorithm_name, X_train, X_test, y_train, y_test
            )
        
        # Run optimization
        study.optimize(objective, n_trials=max_trials, timeout=step.timeout_seconds)
        
        best_params = study.best_params
        best_score = study.best_value
        
        # Store results in context
        context[f"best_params_{algorithm_name}"] = best_params
        context[f"best_score_{algorithm_name}"] = best_score
        
        return {
            "algorithm": algorithm_name,
            "best_parameters": best_params,
            "best_score": best_score,
            "trials_completed": len(study.trials),
            "optimization_time": time.time() - context["start_time"]
        }
    
    def _optimize_hyperparameters(
        self,
        trial: 'optuna.Trial',
        algorithm_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None
    ) -> float:
        """Optimize hyperparameters for a specific algorithm."""
        
        # Get hyperparameter space for algorithm
        param_space = self.algorithm_registry[algorithm_name]["hyperparameter_space"]
        
        # Sample hyperparameters
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) for v in param_values):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
        
        try:
            # Create and train model
            model = self._create_algorithm_instance(algorithm_name, params)
            
            if y_train is not None:
                # Supervised training
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_pred_proba)
                else:
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred, average='weighted')
            else:
                # Unsupervised training
                model.fit(X_train)
                
                # For anomaly detection, we evaluate based on outlier scores
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                    # Use negative scores (outliers have lower scores)
                    score = -np.mean(scores)
                elif hasattr(model, "score_samples"):
                    scores = model.score_samples(X_test)
                    score = -np.mean(scores)
                else:
                    # Fallback: use prediction consistency
                    y_pred = model.predict(X_test)
                    outlier_ratio = np.sum(y_pred == -1) / len(y_pred)
                    score = abs(outlier_ratio - 0.1)  # Target ~10% outliers
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial failed for {algorithm_name}: {str(e)}")
            return -float('inf')
    
    async def _execute_default_hyperparameter_selection(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback hyperparameter selection when Optuna is not available."""
        
        algorithm_name = step.configuration["algorithm"]
        
        # Use default/recommended hyperparameters
        default_params = {
            "IsolationForest": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42
            },
            "OneClassSVM": {
                "nu": 0.1,
                "gamma": "scale"
            },
            "LocalOutlierFactor": {
                "n_neighbors": 20,
                "contamination": 0.1
            },
            "EllipticEnvelope": {
                "contamination": 0.1,
                "random_state": 42
            }
        }
        
        best_params = default_params.get(algorithm_name, {})
        
        # Store in context
        context[f"best_params_{algorithm_name}"] = best_params
        context[f"best_score_{algorithm_name}"] = 0.0
        
        return {
            "algorithm": algorithm_name,
            "best_parameters": best_params,
            "best_score": 0.0,
            "method": "default_parameters"
        }
    
    async def _execute_model_training_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model training step."""
        
        algorithm_name = step.configuration["algorithm"]
        best_params = context.get(f"best_params_{algorithm_name}", {})
        
        data = context["current_data"]
        target = context.get("target_data")
        
        # Create model with optimized parameters
        model = self._create_algorithm_instance(algorithm_name, best_params)
        
        # Train model
        training_start_time = time.time()
        
        if target is not None:
            model.fit(data, target)
        else:
            model.fit(data)
        
        training_time = time.time() - training_start_time
        
        # Store model in context
        context["models"][algorithm_name] = model
        
        # Save model artifact
        model_path = self.storage_directory / f"model_{algorithm_name}_{context['pipeline_id']}.joblib"
        joblib.dump(model, model_path)
        context["artifacts"][f"model_{algorithm_name}"] = str(model_path)
        
        return {
            "algorithm": algorithm_name,
            "training_time_seconds": training_time,
            "model_parameters": best_params,
            "model_saved_to": str(model_path)
        }
    
    async def _execute_model_evaluation_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model evaluation step."""
        
        algorithm_name = step.configuration["algorithm"]
        evaluation_metrics = step.configuration["evaluation_metrics"]
        
        model = context["models"][algorithm_name]
        data = context["current_data"]
        target = context.get("target_data")
        
        # Evaluate model performance
        performance_metrics = {}
        
        if target is not None:
            # Supervised evaluation
            y_pred = model.predict(data)
            
            if "precision" in evaluation_metrics:
                performance_metrics["precision"] = precision_score(
                    target, y_pred, average='weighted', zero_division=0
                )
            
            if "recall" in evaluation_metrics:
                performance_metrics["recall"] = recall_score(
                    target, y_pred, average='weighted', zero_division=0
                )
            
            if "f1" in evaluation_metrics:
                performance_metrics["f1"] = f1_score(
                    target, y_pred, average='weighted', zero_division=0
                )
            
            if "accuracy" in evaluation_metrics:
                performance_metrics["accuracy"] = accuracy_score(target, y_pred)
            
            if "roc_auc" in evaluation_metrics and hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(data)[:, 1]
                    performance_metrics["roc_auc"] = roc_auc_score(target, y_pred_proba)
                except ValueError:
                    performance_metrics["roc_auc"] = 0.5
        
        else:
            # Unsupervised evaluation - use internal metrics
            y_pred = model.predict(data)
            outlier_ratio = np.sum(y_pred == -1) / len(y_pred)
            
            performance_metrics["outlier_ratio"] = outlier_ratio
            
            if hasattr(model, "decision_function"):
                decision_scores = model.decision_function(data)
                performance_metrics["mean_decision_score"] = np.mean(decision_scores)
                performance_metrics["std_decision_score"] = np.std(decision_scores)
            
            # Cross-validation score if possible
            try:
                cv_scores = cross_val_score(model, data, cv=3, scoring='neg_log_loss')
                performance_metrics["cv_score_mean"] = np.mean(cv_scores)
                performance_metrics["cv_score_std"] = np.std(cv_scores)
            except Exception:
                pass
        
        # Store performance results
        context["performance_results"][algorithm_name] = performance_metrics
        
        return {
            "algorithm": algorithm_name,
            "performance_metrics": performance_metrics,
            "evaluation_completed": True
        }
    
    async def _execute_model_selection_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model selection step."""
        
        selection_criteria = step.configuration["selection_criteria"]
        algorithms = step.configuration["algorithms"]
        
        performance_results = context["performance_results"]
        
        # Select best model based on criteria
        best_model = None
        best_score = -float('inf')
        best_algorithm = None
        
        for algorithm in algorithms:
            if algorithm not in performance_results:
                continue
            
            metrics = performance_results[algorithm]
            
            # Calculate composite score based on criteria
            composite_score = 0
            weight_sum = 0
            
            for metric, target_value in selection_criteria.items():
                if metric.startswith("min_"):
                    metric_name = metric[4:]  # Remove "min_" prefix
                    if metric_name in metrics:
                        # Score based on how much we exceed minimum
                        score = max(0, metrics[metric_name] - target_value)
                        composite_score += score
                        weight_sum += 1
            
            # Normalize composite score
            if weight_sum > 0:
                composite_score /= weight_sum
            
            if composite_score > best_score:
                best_score = composite_score
                best_algorithm = algorithm
                best_model = context["models"][algorithm]
        
        # Store selection results
        context["selected_model"] = best_model
        context["selected_algorithm"] = best_algorithm
        
        return {
            "selected_algorithm": best_algorithm,
            "selection_score": best_score,
            "all_algorithm_scores": {
                alg: context["performance_results"].get(alg, {})
                for alg in algorithms
            }
        }
    
    async def _execute_model_deployment_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute model deployment step."""
        
        selected_algorithm = context.get("selected_algorithm")
        selected_model = context.get("selected_model")
        
        if not selected_algorithm or selected_model is None:
            raise ValueError("No model selected for deployment")
        
        deployment_env = step.configuration.get("deployment_environment", "staging")
        
        # Create deployment package
        deployment_package = {
            "model": selected_model,
            "algorithm": selected_algorithm,
            "performance_metrics": context["performance_results"][selected_algorithm],
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "pipeline_id": context["pipeline_id"]
        }
        
        # Save deployment package
        deployment_path = self.storage_directory / f"deployment_{context['pipeline_id']}_{deployment_env}.joblib"
        joblib.dump(deployment_package, deployment_path)
        
        context["artifacts"]["deployment_package"] = str(deployment_path)
        
        return {
            "deployed_algorithm": selected_algorithm,
            "deployment_environment": deployment_env,
            "deployment_package_path": str(deployment_path),
            "deployment_timestamp": deployment_package["deployment_timestamp"]
        }
    
    def _create_algorithm_instance(self, algorithm_name: str, parameters: Dict[str, Any]) -> BaseEstimator:
        """Create an instance of the specified algorithm with parameters."""
        
        if algorithm_name not in self.algorithm_registry:
            raise ValueError(f"Algorithm {algorithm_name} not supported")
        
        algorithm_info = self.algorithm_registry[algorithm_name]
        
        # Handle dynamic imports
        if algorithm_info["class"] is None:
            if algorithm_name == "OneClassSVM":
                from sklearn.svm import OneClassSVM
                algorithm_class = OneClassSVM
            elif algorithm_name == "LocalOutlierFactor":
                from sklearn.neighbors import LocalOutlierFactor
                algorithm_class = LocalOutlierFactor
                parameters["novelty"] = True  # Required for anomaly detection
            elif algorithm_name == "EllipticEnvelope":
                from sklearn.covariance import EllipticEnvelope
                algorithm_class = EllipticEnvelope
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
        else:
            algorithm_class = algorithm_info["class"]
        
        return algorithm_class(**parameters)
    
    def _select_best_model(self, performance_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Select the best model from performance results."""
        
        if not performance_results:
            return {"algorithm": None, "reason": "No models evaluated"}
        
        # Simple selection based on F1 score or outlier ratio
        best_algorithm = None
        best_score = -float('inf')
        
        for algorithm, metrics in performance_results.items():
            if "f1" in metrics:
                score = metrics["f1"]
            elif "roc_auc" in metrics:
                score = metrics["roc_auc"]
            elif "outlier_ratio" in metrics:
                # For unsupervised methods, prefer moderate outlier ratios
                ratio = metrics["outlier_ratio"]
                score = 1.0 - abs(ratio - 0.1)  # Target 10% outliers
            else:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_algorithm = algorithm
        
        return {
            "algorithm": best_algorithm,
            "score": best_score,
            "metrics": performance_results.get(best_algorithm, {})
        }
    
    def _generate_execution_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of pipeline execution."""
        
        return {
            "total_steps": len(context["step_results"]),
            "successful_steps": len([
                r for r in context["step_results"].values() 
                if r["status"] == "completed"
            ]),
            "total_execution_time": time.time() - context["start_time"],
            "models_trained": len(context.get("models", {})),
            "best_performing_algorithm": self._select_best_model(
                context.get("performance_results", {})
            )["algorithm"],
            "artifacts_created": len(context.get("artifacts", {})),
            "data_shape": getattr(context.get("current_data"), "shape", None)
        }
    
    # Feature Engineering Functions
    
    def _create_statistical_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create statistical features for anomaly detection."""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Rolling statistics
            window_size = config.get("rolling_window", 5)
            if len(data) >= window_size:
                data[f"{col}_rolling_mean"] = data[col].rolling(window=window_size).mean()
                data[f"{col}_rolling_std"] = data[col].rolling(window=window_size).std()
                data[f"{col}_rolling_min"] = data[col].rolling(window=window_size).min()
                data[f"{col}_rolling_max"] = data[col].rolling(window=window_size).max()
            
            # Z-score
            data[f"{col}_zscore"] = (data[col] - data[col].mean()) / data[col].std()
            
            # Percentile ranks
            data[f"{col}_percentile"] = data[col].rank(pct=True)
        
        # Fill NaN values created by rolling operations
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data
    
    def _create_temporal_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create temporal features if timestamp column exists."""
        
        timestamp_col = config.get("timestamp_column")
        if timestamp_col and timestamp_col in data.columns:
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            
            # Extract temporal components
            data["hour"] = data[timestamp_col].dt.hour
            data["day_of_week"] = data[timestamp_col].dt.dayofweek
            data["month"] = data[timestamp_col].dt.month
            data["quarter"] = data[timestamp_col].dt.quarter
            
            # Cyclical encoding
            data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
            data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
            data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
            data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
        
        return data
    
    def _create_interaction_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        max_interactions = config.get("max_interactions", 10)
        
        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Multiplicative interaction
                data[f"{col1}_x_{col2}"] = data[col1] * data[col2]
                
                # Ratio interaction (avoid division by zero)
                data[f"{col1}_div_{col2}"] = data[col1] / (data[col2] + 1e-8)
                
                interaction_count += 2
            
            if interaction_count >= max_interactions:
                break
        
        return data
    
    def _create_polynomial_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create polynomial features."""
        
        from sklearn.preprocessing import PolynomialFeatures
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        degree = config.get("polynomial_degree", 2)
        max_features = config.get("max_polynomial_features", 50)
        
        if len(numeric_cols) > 0:
            # Use subset of columns to avoid explosion in feature count
            selected_cols = numeric_cols[:min(5, len(numeric_cols))]
            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(data[selected_cols])
            
            # Create DataFrame with polynomial features
            feature_names = poly.get_feature_names_out(selected_cols)
            poly_df = pd.DataFrame(
                poly_features[:, len(selected_cols):],  # Exclude original features
                columns=feature_names[len(selected_cols):],
                index=data.index
            )
            
            # Limit number of features
            if len(poly_df.columns) > max_features:
                poly_df = poly_df.iloc[:, :max_features]
            
            data = pd.concat([data, poly_df], axis=1)
        
        return data
    
    def _create_domain_specific_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create domain-specific features for anomaly detection."""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Distance from center
        if len(numeric_cols) >= 2:
            # Euclidean distance from centroid
            centroid = data[numeric_cols].mean()
            distances = np.sqrt(((data[numeric_cols] - centroid) ** 2).sum(axis=1))
            data["distance_from_center"] = distances
            
            # Mahalanobis distance (if possible)
            try:
                from scipy.spatial.distance import mahalanobis
                cov_matrix = data[numeric_cols].cov()
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
                
                mahal_distances = []
                for _, row in data[numeric_cols].iterrows():
                    dist = mahalanobis(row, centroid, inv_cov_matrix)
                    mahal_distances.append(dist)
                
                data["mahalanobis_distance"] = mahal_distances
            except Exception:
                pass
        
        # Density estimation
        if len(numeric_cols) >= 1:
            from sklearn.neighbors import LocalOutlierFactor
            
            lof = LocalOutlierFactor(n_neighbors=min(20, len(data) // 2))
            lof_scores = lof.fit_predict(data[numeric_cols])
            data["local_outlier_factor"] = lof.negative_outlier_factor_
        
        return data
    
    def _create_auto_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Automatically create features using multiple strategies."""
        
        # Apply multiple feature engineering strategies
        data = self._create_statistical_features(data, config)
        data = self._create_interaction_features(data, config)
        data = self._create_domain_specific_features(data, config)
        
        return data
    
    async def get_automl_recommendations(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> AutoMLRecommendation:
        """Get AutoML recommendations for algorithm and configuration.
        
        Args:
            data: Input data for analysis
            target: Target labels (optional)
            
        Returns:
            AutoML recommendations
        """
        self.logger.info("Generating AutoML recommendations")
        
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Analyze dataset characteristics
        dataset_chars = self._analyze_dataset_characteristics(data, target)
        
        # Recommend algorithms based on dataset characteristics
        algorithm_rankings = self._rank_algorithms_for_dataset(dataset_chars)
        
        # Generate feature engineering recommendations
        feature_recommendations = self._recommend_feature_engineering(dataset_chars)
        
        # Generate hyperparameter recommendations
        hyperparam_recommendations = self._recommend_hyperparameters(dataset_chars)
        
        # Generate performance predictions
        performance_predictions = self._predict_algorithm_performance(dataset_chars)
        
        # Generate reasoning
        reasoning = self._generate_recommendation_reasoning(
            dataset_chars, algorithm_rankings, feature_recommendations
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_recommendation_confidence(dataset_chars)
        
        recommendation = AutoMLRecommendation(
            recommended_algorithms=list(algorithm_rankings.keys())[:3],
            algorithm_rankings=algorithm_rankings,
            dataset_characteristics=dataset_chars,
            feature_engineering_recommendations=feature_recommendations,
            hyperparameter_recommendations=hyperparam_recommendations,
            performance_predictions=performance_predictions,
            reasoning=reasoning,
            confidence_score=confidence_score
        )
        
        self.logger.info(f"Generated AutoML recommendations with {confidence_score:.2f} confidence")
        return recommendation
    
    def _analyze_dataset_characteristics(
        self,
        data: pd.DataFrame,
        target: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics for AutoML recommendations."""
        
        characteristics = {
            "n_samples": len(data),
            "n_features": len(data.columns),
            "n_numeric_features": len(data.select_dtypes(include=[np.number]).columns),
            "n_categorical_features": len(data.select_dtypes(include=['object']).columns),
            "missing_value_ratio": data.isnull().sum().sum() / (data.shape[0] * data.shape[1]),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "has_target": target is not None,
            "is_supervised": target is not None,
        }
        
        # Analyze numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            characteristics.update({
                "mean_feature_correlation": abs(numeric_data.corr()).mean().mean(),
                "max_feature_correlation": abs(numeric_data.corr()).max().max(),
                "feature_variance_ratio": numeric_data.var().max() / (numeric_data.var().min() + 1e-8),
                "has_outliers": self._detect_outliers_in_data(numeric_data),
                "data_distribution": self._analyze_data_distribution(numeric_data)
            })
        
        # Analyze target if available
        if target is not None:
            if hasattr(target, 'value_counts'):
                target_counts = target.value_counts()
                characteristics.update({
                    "n_classes": len(target_counts),
                    "class_imbalance_ratio": target_counts.max() / target_counts.min(),
                    "minority_class_ratio": target_counts.min() / len(target)
                })
        
        # Dataset size category
        if characteristics["n_samples"] < 1000:
            characteristics["size_category"] = "small"
        elif characteristics["n_samples"] < 100000:
            characteristics["size_category"] = "medium"
        else:
            characteristics["size_category"] = "large"
        
        # Feature complexity
        if characteristics["n_features"] < 10:
            characteristics["feature_complexity"] = "low"
        elif characteristics["n_features"] < 100:
            characteristics["feature_complexity"] = "medium"
        else:
            characteristics["feature_complexity"] = "high"
        
        return characteristics
    
    def _detect_outliers_in_data(self, data: pd.DataFrame) -> bool:
        """Detect if data contains outliers using IQR method."""
        
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio = outliers / len(data)
            
            if outlier_ratio > 0.05:  # More than 5% outliers
                return True
        
        return False
    
    def _analyze_data_distribution(self, data: pd.DataFrame) -> str:
        """Analyze the distribution characteristics of numeric data."""
        
        try:
            from scipy import stats
            
            # Test for normality using Shapiro-Wilk test (sample if data is large)
            sample_size = min(5000, len(data))
            sample_data = data.sample(n=sample_size) if len(data) > sample_size else data
            
            normal_features = 0
            total_features = len(sample_data.columns)
            
            for col in sample_data.columns:
                _, p_value = stats.shapiro(sample_data[col].dropna())
                if p_value > 0.05:  # Normal distribution
                    normal_features += 1
            
            normal_ratio = normal_features / total_features
            
            if normal_ratio > 0.7:
                return "normal"
            elif normal_ratio > 0.3:
                return "mixed"
            else:
                return "non_normal"
                
        except ImportError:
            # Fallback without scipy
            skewness_values = []
            for col in data.columns:
                skew = data[col].skew()
                skewness_values.append(abs(skew))
            
            mean_skewness = np.mean(skewness_values)
            
            if mean_skewness < 0.5:
                return "normal"
            elif mean_skewness < 2.0:
                return "moderately_skewed"
            else:
                return "highly_skewed"
    
    def _rank_algorithms_for_dataset(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Rank algorithms based on dataset characteristics."""
        
        algorithm_scores = {}
        
        for algorithm_name, algorithm_info in self.algorithm_registry.items():
            score = 0.0
            
            # Size-based scoring
            if characteristics["size_category"] == "small":
                if algorithm_name in ["LocalOutlierFactor", "EllipticEnvelope"]:
                    score += 0.3
            elif characteristics["size_category"] == "medium":
                if algorithm_name in ["IsolationForest", "OneClassSVM"]:
                    score += 0.3
            else:  # large
                if algorithm_name in ["IsolationForest", "XGBClassifier", "LGBMClassifier"]:
                    score += 0.3
            
            # Feature complexity scoring
            if characteristics["feature_complexity"] == "high":
                if algorithm_name in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]:
                    score += 0.2
            
            # Supervised vs unsupervised
            if characteristics["is_supervised"]:
                if algorithm_info["type"] == "supervised":
                    score += 0.2
            else:
                if algorithm_info["type"] == "unsupervised":
                    score += 0.2
            
            # Outlier handling
            if characteristics.get("has_outliers", False):
                if algorithm_name in ["IsolationForest", "LocalOutlierFactor"]:
                    score += 0.15
            
            # Data distribution
            if characteristics.get("data_distribution") == "normal":
                if algorithm_name in ["EllipticEnvelope", "OneClassSVM"]:
                    score += 0.1
            
            # Missing values tolerance
            if characteristics["missing_value_ratio"] > 0.1:
                if algorithm_name in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]:
                    score += 0.1
            
            # Categorical features
            if characteristics["n_categorical_features"] > 0:
                if algorithm_name in ["CatBoostClassifier", "LGBMClassifier"]:
                    score += 0.1
            
            algorithm_scores[algorithm_name] = score
        
        # Sort by score
        return dict(sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True))
    
    def _recommend_feature_engineering(self, characteristics: Dict[str, Any]) -> List[str]:
        """Recommend feature engineering strategies."""
        
        recommendations = []
        
        # Always recommend statistical features for anomaly detection
        recommendations.append("statistical_features")
        
        # Interaction features for medium complexity datasets
        if characteristics["feature_complexity"] in ["medium", "high"]:
            recommendations.append("interaction_features")
        
        # Domain-specific features for outlier detection
        recommendations.append("domain_specific")
        
        # Polynomial features for small datasets
        if characteristics["size_category"] == "small":
            recommendations.append("polynomial_features")
        
        # Auto features for complex datasets
        if (characteristics["feature_complexity"] == "high" or 
            characteristics["size_category"] == "large"):
            recommendations.append("auto_features")
        
        return recommendations
    
    def _recommend_hyperparameters(self, characteristics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Recommend hyperparameters for each algorithm."""
        
        recommendations = {}
        
        for algorithm_name in self.algorithm_registry.keys():
            if algorithm_name == "IsolationForest":
                if characteristics["size_category"] == "large":
                    recommendations[algorithm_name] = {
                        "n_estimators": 200,
                        "max_samples": 0.7,
                        "contamination": 0.1
                    }
                else:
                    recommendations[algorithm_name] = {
                        "n_estimators": 100,
                        "max_samples": "auto",
                        "contamination": 0.1
                    }
            
            elif algorithm_name == "OneClassSVM":
                if characteristics.get("has_outliers", False):
                    recommendations[algorithm_name] = {
                        "nu": 0.05,
                        "gamma": "scale"
                    }
                else:
                    recommendations[algorithm_name] = {
                        "nu": 0.1,
                        "gamma": "scale"
                    }
            
            elif algorithm_name == "LocalOutlierFactor":
                if characteristics["size_category"] == "small":
                    recommendations[algorithm_name] = {
                        "n_neighbors": 10,
                        "contamination": 0.1
                    }
                else:
                    recommendations[algorithm_name] = {
                        "n_neighbors": 20,
                        "contamination": 0.1
                    }
        
        return recommendations
    
    def _predict_algorithm_performance(self, characteristics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Predict algorithm performance based on dataset characteristics."""
        
        predictions = {}
        
        for algorithm_name in self.algorithm_registry.keys():
            # Simple heuristic-based performance prediction
            base_performance = 0.7
            
            # Adjust based on algorithm suitability
            if algorithm_name == "IsolationForest":
                if characteristics["size_category"] in ["medium", "large"]:
                    base_performance += 0.1
                if characteristics.get("has_outliers", False):
                    base_performance += 0.1
            
            elif algorithm_name == "OneClassSVM":
                if characteristics.get("data_distribution") == "normal":
                    base_performance += 0.1
                if characteristics["size_category"] == "small":
                    base_performance += 0.05
            
            elif algorithm_name == "LocalOutlierFactor":
                if characteristics["size_category"] == "small":
                    base_performance += 0.1
                if characteristics["feature_complexity"] == "low":
                    base_performance += 0.05
            
            # Add some variability
            predictions[algorithm_name] = {
                "precision": min(0.95, base_performance + np.random.normal(0, 0.05)),
                "recall": min(0.95, base_performance + np.random.normal(0, 0.05)),
                "f1": min(0.95, base_performance + np.random.normal(0, 0.03)),
                "roc_auc": min(0.98, base_performance + 0.1 + np.random.normal(0, 0.03))
            }
        
        return predictions
    
    def _generate_recommendation_reasoning(
        self,
        characteristics: Dict[str, Any],
        algorithm_rankings: Dict[str, float],
        feature_recommendations: List[str]
    ) -> List[str]:
        """Generate reasoning for recommendations."""
        
        reasoning = []
        
        # Dataset size reasoning
        if characteristics["size_category"] == "small":
            reasoning.append(
                "Small dataset detected - recommending algorithms that work well with limited data"
            )
        elif characteristics["size_category"] == "large":
            reasoning.append(
                "Large dataset detected - recommending scalable algorithms like IsolationForest"
            )
        
        # Feature complexity reasoning
        if characteristics["feature_complexity"] == "high":
            reasoning.append(
                "High-dimensional dataset - recommending tree-based algorithms and feature engineering"
            )
        
        # Outlier reasoning
        if characteristics.get("has_outliers", False):
            reasoning.append(
                "Outliers detected in data - prioritizing robust anomaly detection algorithms"
            )
        
        # Supervised vs unsupervised
        if not characteristics["is_supervised"]:
            reasoning.append(
                "No target labels provided - focusing on unsupervised anomaly detection methods"
            )
        
        # Top algorithm reasoning
        top_algorithm = list(algorithm_rankings.keys())[0]
        reasoning.append(f"Recommending {top_algorithm} as the primary algorithm based on dataset characteristics")
        
        return reasoning
    
    def _calculate_recommendation_confidence(self, characteristics: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendations."""
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence for clear dataset characteristics
        if characteristics["size_category"] in ["small", "large"]:
            confidence += 0.1
        
        if characteristics["feature_complexity"] in ["low", "high"]:
            confidence += 0.1
        
        if characteristics.get("has_outliers") is not None:
            confidence += 0.1
        
        if characteristics["is_supervised"]:
            confidence += 0.1
        
        # Lower confidence for edge cases
        if characteristics["missing_value_ratio"] > 0.5:
            confidence -= 0.1
        
        if characteristics["n_samples"] < 50:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    async def get_pipeline_status(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get status of a pipeline."""
        
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.active_pipelines[pipeline_id]
        
        return {
            "pipeline_id": str(pipeline_id),
            "name": pipeline.name,
            "status": pipeline.status.value,
            "step_count": len(pipeline.steps),
            "enabled_steps": len(pipeline.enabled_steps),
            "created_at": pipeline.created_at.isoformat(),
            "is_active": pipeline.is_active
        }
    
    async def list_available_algorithms(self) -> Dict[str, Any]:
        """List all available algorithms with their capabilities."""
        
        algorithms = {}
        
        for name, info in self.algorithm_registry.items():
            algorithms[name] = {
                "type": info["type"],
                "suitable_for": info["suitable_for"],
                "available": info["class"] is not None or name in [
                    "OneClassSVM", "LocalOutlierFactor", "EllipticEnvelope"
                ]
            }
        
        return {
            "total_algorithms": len(algorithms),
            "algorithms": algorithms,
            "dependencies": {
                "optuna_available": OPTUNA_AVAILABLE,
                "xgboost_available": XGBOOST_AVAILABLE,
                "lightgbm_available": LIGHTGBM_AVAILABLE,
                "catboost_available": CATBOOST_AVAILABLE
            }
        }