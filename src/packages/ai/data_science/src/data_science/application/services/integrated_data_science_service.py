"""
Integrated Data Science Service - Phase 2.8 Integration Layer
Orchestrates data profiling, quality assessment, statistical analysis, ML pipelines, and performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from ...domain.entities.data_science_model import DataScienceModel
from ...domain.entities.machine_learning_pipeline import MachineLearningPipeline
from ..dto.ml_pipeline_dto import MLPipelineDTO
from ..dto.statistical_analysis_dto import StatisticalAnalysisDTO
from ..dto.visualization_dto import VisualizationDTO

# Import data profiling services
from ....data_profiling.application.services.profiling_engine import ProfilingEngine, ProfilingConfig
from ....data_profiling.application.services.pattern_discovery_service import PatternDiscoveryService
from ....data_profiling.application.services.statistical_profiling_service import StatisticalProfilingService

# Import data quality services  
from ....data_quality.application.services.quality_assessment_service import QualityAssessmentService
from ....data_quality.application.services.validation_engine import ValidationEngine
from ....data_quality.application.services.data_cleansing_engine import DataCleansingEngine

logger = logging.getLogger(__name__)


@dataclass
class DataScienceWorkflowConfig:
    """Configuration for integrated data science workflows."""
    
    # Profiling configuration
    enable_profiling: bool = True
    profiling_sample_size: int = 10000
    enable_advanced_patterns: bool = True
    
    # Quality assessment configuration
    enable_quality_assessment: bool = True
    quality_threshold: float = 0.8
    enable_auto_cleansing: bool = False
    
    # Statistical analysis configuration
    enable_statistical_analysis: bool = True
    confidence_level: float = 0.95
    enable_hypothesis_testing: bool = True
    
    # ML pipeline configuration
    enable_ml_pipeline: bool = True
    auto_feature_engineering: bool = True
    enable_hyperparameter_tuning: bool = True
    max_training_time_minutes: int = 60
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    memory_limit_gb: int = 8
    
    # Integration settings
    enable_cross_package_optimization: bool = True
    enable_workflow_monitoring: bool = True
    enable_result_validation: bool = True


@dataclass
class WorkflowExecutionMetrics:
    """Metrics collected during workflow execution."""
    
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time_seconds: Optional[float] = None
    
    # Stage timings
    profiling_time_seconds: Optional[float] = None
    quality_assessment_time_seconds: Optional[float] = None
    statistical_analysis_time_seconds: Optional[float] = None
    ml_pipeline_time_seconds: Optional[float] = None
    
    # Resource usage
    peak_memory_usage_mb: Optional[float] = None
    cpu_utilization_percent: Optional[float] = None
    
    # Data characteristics
    dataset_rows: Optional[int] = None
    dataset_columns: Optional[int] = None
    dataset_size_mb: Optional[float] = None
    
    # Quality metrics
    data_quality_score: Optional[float] = None
    feature_count: Optional[int] = None
    model_performance_score: Optional[float] = None
    
    # Status and errors
    status: str = "running"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IntegratedWorkflowResult:
    """Comprehensive result from integrated data science workflow."""
    
    workflow_id: str
    execution_metrics: WorkflowExecutionMetrics
    
    # Core results
    profiling_results: Optional[Dict[str, Any]] = None
    quality_results: Optional[Dict[str, Any]] = None
    statistical_results: Optional[Dict[str, Any]] = None
    ml_results: Optional[Dict[str, Any]] = None
    
    # Integrated insights
    data_insights: Optional[Dict[str, Any]] = None
    quality_recommendations: List[str] = field(default_factory=list)
    feature_engineering_suggestions: List[str] = field(default_factory=list)
    model_recommendations: List[str] = field(default_factory=list)
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None
    
    # Final outputs
    processed_dataset: Optional[pd.DataFrame] = None
    trained_models: List[Dict[str, Any]] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None


class IntegratedDataScienceService:
    """
    Unified service that orchestrates all data science packages for end-to-end workflows.
    Implements Phase 2.8 integration requirements with performance optimization.
    """
    
    def __init__(self, config: DataScienceWorkflowConfig = None):
        self.config = config or DataScienceWorkflowConfig()
        
        # Initialize individual package services
        self._initialize_services()
        
        # Performance monitoring
        self.active_workflows: Dict[str, WorkflowExecutionMetrics] = {}
        self.workflow_history: List[WorkflowExecutionMetrics] = []
        
        # Caching for performance optimization
        self.result_cache: Dict[str, Any] = {} if self.config.enable_caching else None
        
        logger.info("Initialized IntegratedDataScienceService with advanced orchestration")
    
    def _initialize_services(self):
        """Initialize all data science package services."""
        try:
            # Data profiling services
            profiling_config = ProfilingConfig(
                enable_sampling=True,
                sample_size=self.config.profiling_sample_size,
                enable_advanced_patterns=self.config.enable_advanced_patterns,
                max_workers=self.config.max_workers
            )
            self.profiling_engine = ProfilingEngine(profiling_config)
            self.pattern_discovery = PatternDiscoveryService()
            self.statistical_profiling = StatisticalProfilingService()
            
            # Data quality services
            self.quality_assessment = QualityAssessmentService()
            self.validation_engine = ValidationEngine()
            self.data_cleansing = DataCleansingEngine()
            
            # Initialize other services as they become available
            self.ml_pipeline_service = None  # Will be initialized when ML pipeline is available
            self.feature_engineering_service = None
            self.visualization_service = None
            
            logger.info("Successfully initialized all available data science services")
            
        except Exception as e:
            logger.error(f"Failed to initialize some services: {e}")
            # Continue with available services
    
    async def execute_integrated_workflow(
        self,
        data: Union[pd.DataFrame, str, Dict[str, Any]],
        workflow_id: str = None,
        target_column: str = None,
        workflow_type: str = "comprehensive"
    ) -> IntegratedWorkflowResult:
        """
        Execute a comprehensive integrated data science workflow.
        
        Args:
            data: Input data (DataFrame, file path, or data source config)
            workflow_id: Unique identifier for this workflow execution
            target_column: Target column for ML modeling (if applicable)
            workflow_type: Type of workflow ('profiling', 'quality', 'ml', 'comprehensive')
        
        Returns:
            IntegratedWorkflowResult with all analysis results and insights
        """
        workflow_id = workflow_id or f"workflow_{int(time.time())}"
        
        # Initialize workflow tracking
        metrics = WorkflowExecutionMetrics(
            workflow_id=workflow_id,
            start_time=datetime.now()
        )
        self.active_workflows[workflow_id] = metrics
        
        try:
            logger.info(f"Starting integrated workflow {workflow_id} (type: {workflow_type})")
            
            # Step 1: Data loading and preprocessing
            df = await self._load_and_preprocess_data(data, metrics)
            
            # Step 2: Execute workflow stages based on configuration and type
            if workflow_type in ["comprehensive", "profiling"]:
                profiling_results = await self._execute_profiling_stage(df, metrics)
            else:
                profiling_results = None
            
            if workflow_type in ["comprehensive", "quality"]:
                quality_results = await self._execute_quality_stage(df, metrics, profiling_results)
            else:
                quality_results = None
            
            if workflow_type in ["comprehensive", "statistical"]:
                statistical_results = await self._execute_statistical_stage(df, metrics)
            else:
                statistical_results = None
            
            if workflow_type in ["comprehensive", "ml"] and target_column:
                ml_results = await self._execute_ml_stage(df, target_column, metrics, profiling_results)
            else:
                ml_results = None
            
            # Step 3: Generate integrated insights
            insights = await self._generate_integrated_insights(
                df, profiling_results, quality_results, statistical_results, ml_results
            )
            
            # Step 4: Create visualization data
            visualization_data = await self._create_visualization_data(
                df, profiling_results, quality_results, statistical_results, ml_results
            )
            
            # Step 5: Finalize metrics and create result
            metrics.end_time = datetime.now()
            metrics.total_execution_time_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.status = "completed"
            
            result = IntegratedWorkflowResult(
                workflow_id=workflow_id,
                execution_metrics=metrics,
                profiling_results=profiling_results,
                quality_results=quality_results,
                statistical_results=statistical_results,
                ml_results=ml_results,
                data_insights=insights,
                visualization_data=visualization_data,
                processed_dataset=df
            )
            
            # Clean up active workflow tracking
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            self.workflow_history.append(metrics)
            
            logger.info(f"Completed integrated workflow {workflow_id} in {metrics.total_execution_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            metrics.end_time = datetime.now()
            metrics.status = "failed"
            metrics.errors.append(str(e))
            
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            # Return partial result with error information
            return IntegratedWorkflowResult(
                workflow_id=workflow_id,
                execution_metrics=metrics
            )
    
    async def _load_and_preprocess_data(self, data: Union[pd.DataFrame, str, Dict[str, Any]], 
                                       metrics: WorkflowExecutionMetrics) -> pd.DataFrame:
        """Load and preprocess input data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, str):
            # Load from file path
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.parquet'):
                df = pd.read_parquet(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Update metrics
        metrics.dataset_rows = len(df)
        metrics.dataset_columns = len(df.columns)
        metrics.dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        logger.info(f"Loaded dataset: {metrics.dataset_rows} rows, {metrics.dataset_columns} columns")
        
        return df
    
    async def _execute_profiling_stage(self, df: pd.DataFrame, 
                                     metrics: WorkflowExecutionMetrics) -> Dict[str, Any]:
        """Execute data profiling stage."""
        if not self.config.enable_profiling:
            return None
        
        start_time = time.time()
        
        try:
            logger.info("Executing data profiling stage...")
            
            # Run comprehensive profiling
            profile = self.profiling_engine.profile_dataset(
                df, 
                dataset_id=metrics.workflow_id,
                use_advanced_orchestrator=self.config.enable_advanced_patterns
            )
            
            # Additional pattern discovery
            patterns = self.pattern_discovery.discover(df)
            
            # Statistical profiling
            statistical_profile = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                statistical_profile[col] = self.statistical_profiling.analyze(df[[col]])
            
            profiling_results = {
                'schema_profile': profile,
                'patterns': patterns,
                'statistical_profile': statistical_profile,
                'profiling_summary': self.profiling_engine.get_profiling_summary(profile)
            }
            
            metrics.profiling_time_seconds = time.time() - start_time
            logger.info(f"Profiling stage completed in {metrics.profiling_time_seconds:.2f}s")
            
            return profiling_results
            
        except Exception as e:
            metrics.errors.append(f"Profiling stage failed: {str(e)}")
            logger.error(f"Profiling stage failed: {e}")
            return None
    
    async def _execute_quality_stage(self, df: pd.DataFrame, 
                                   metrics: WorkflowExecutionMetrics,
                                   profiling_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute data quality assessment stage."""
        if not self.config.enable_quality_assessment:
            return None
        
        start_time = time.time()
        
        try:
            logger.info("Executing data quality stage...")
            
            # Use profiling results for enhanced quality assessment
            schema_profile = profiling_results.get('schema_profile') if profiling_results else None
            
            # Run quality assessment
            quality_assessment = self.quality_assessment.assess_quality(schema_profile, df)
            
            # Validation engine results
            validation_results = self.validation_engine.validate_dataset(df)
            
            # Auto-cleansing if enabled
            cleansed_data = None
            if self.config.enable_auto_cleansing and quality_assessment.overall_score < self.config.quality_threshold:
                cleansed_data = self.data_cleansing.cleanse_dataset(df, quality_assessment)
            
            quality_results = {
                'quality_assessment': quality_assessment,
                'validation_results': validation_results,
                'cleansed_data': cleansed_data,
                'quality_score': quality_assessment.overall_score if quality_assessment else None
            }
            
            metrics.quality_assessment_time_seconds = time.time() - start_time
            metrics.data_quality_score = quality_assessment.overall_score if quality_assessment else None
            
            logger.info(f"Quality stage completed in {metrics.quality_assessment_time_seconds:.2f}s")
            
            return quality_results
            
        except Exception as e:
            metrics.errors.append(f"Quality stage failed: {str(e)}")
            logger.error(f"Quality stage failed: {e}")
            return None
    
    async def _execute_statistical_stage(self, df: pd.DataFrame, 
                                       metrics: WorkflowExecutionMetrics) -> Dict[str, Any]:
        """Execute statistical analysis stage."""
        if not self.config.enable_statistical_analysis:
            return None
        
        start_time = time.time()
        
        try:
            logger.info("Executing statistical analysis stage...")
            
            # Descriptive statistics
            descriptive_stats = df.describe(include='all')
            
            # Correlation analysis for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr() if len(numeric_df.columns) > 1 else None
            
            # Distribution analysis
            distribution_analysis = {}
            for col in numeric_df.columns:
                try:
                    from scipy import stats
                    # Test for normality
                    statistic, p_value = stats.normaltest(numeric_df[col].dropna())
                    distribution_analysis[col] = {
                        'normality_test_statistic': statistic,
                        'normality_p_value': p_value,
                        'is_normal': p_value > 0.05,
                        'skewness': numeric_df[col].skew(),
                        'kurtosis': numeric_df[col].kurtosis()
                    }
                except ImportError:
                    # Fallback without scipy
                    distribution_analysis[col] = {
                        'skewness': numeric_df[col].skew(),
                        'kurtosis': numeric_df[col].kurtosis()
                    }
            
            # Outlier detection
            outliers = {}
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(numeric_df[col])) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            
            statistical_results = {
                'descriptive_statistics': descriptive_stats.to_dict(),
                'correlation_matrix': correlation_matrix.to_dict() if correlation_matrix is not None else None,
                'distribution_analysis': distribution_analysis,
                'outlier_analysis': outliers,
                'summary': {
                    'total_numeric_columns': len(numeric_df.columns),
                    'total_categorical_columns': len(df.select_dtypes(include=['object']).columns),
                    'missing_values_total': df.isnull().sum().sum(),
                    'duplicate_rows': df.duplicated().sum()
                }
            }
            
            metrics.statistical_analysis_time_seconds = time.time() - start_time
            logger.info(f"Statistical stage completed in {metrics.statistical_analysis_time_seconds:.2f}s")
            
            return statistical_results
            
        except Exception as e:
            metrics.errors.append(f"Statistical stage failed: {str(e)}")
            logger.error(f"Statistical stage failed: {e}")
            return None
    
    async def _execute_ml_stage(self, df: pd.DataFrame, target_column: str,
                              metrics: WorkflowExecutionMetrics,
                              profiling_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute machine learning pipeline stage."""
        if not self.config.enable_ml_pipeline or not target_column:
            return None
        
        start_time = time.time()
        
        try:
            logger.info("Executing ML pipeline stage...")
            
            # Basic feature engineering based on profiling results
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
            
            # Simple preprocessing
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
            
            # Handle categorical variables
            X_processed = X.copy()
            label_encoders = {}
            
            for col in X.select_dtypes(include=['object']).columns:
                if X[col].nunique() < 50:  # Only encode if reasonable number of categories
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X[col].fillna('missing'))
                    label_encoders[col] = le
                else:
                    X_processed = X_processed.drop(columns=[col])
            
            # Fill missing values
            X_processed = X_processed.fillna(X_processed.mean() if X_processed.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
            
            # Determine problem type and train model
            if y.dtype == 'object' or y.nunique() < 20:
                # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                performance_score = accuracy_score(y_test, y_pred)
                performance_metric = 'accuracy'
                detailed_metrics = classification_report(y_test, y_pred, output_dict=True)
            else:
                # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                performance_score = 1 - (mean_squared_error(y_test, y_pred) / y_test.var())  # R² score approximation
                performance_metric = 'r2_score'
                detailed_metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
            
            # Feature importance
            feature_importance = dict(zip(X_processed.columns, model.feature_importances_))
            
            ml_results = {
                'model': model,
                'model_type': type(model).__name__,
                'performance_score': performance_score,
                'performance_metric': performance_metric,
                'detailed_metrics': detailed_metrics,
                'feature_importance': feature_importance,
                'label_encoders': label_encoders,
                'training_features': list(X_processed.columns),
                'target_column': target_column,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            metrics.ml_pipeline_time_seconds = time.time() - start_time
            metrics.model_performance_score = performance_score
            metrics.feature_count = len(X_processed.columns)
            
            logger.info(f"ML stage completed in {metrics.ml_pipeline_time_seconds:.2f}s, {performance_metric}: {performance_score:.3f}")
            
            return ml_results
            
        except Exception as e:
            metrics.errors.append(f"ML stage failed: {str(e)}")
            logger.error(f"ML stage failed: {e}")
            return None
    
    async def _generate_integrated_insights(self, df: pd.DataFrame,
                                          profiling_results: Dict[str, Any],
                                          quality_results: Dict[str, Any],
                                          statistical_results: Dict[str, Any],
                                          ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated insights across all analysis stages."""
        insights = {
            'data_overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'recommendations': [],
            'key_findings': [],
            'data_quality_insights': [],
            'statistical_insights': [],
            'ml_insights': []
        }
        
        # Data quality insights
        if quality_results and quality_results.get('quality_assessment'):
            qa = quality_results['quality_assessment']
            quality_score = qa.overall_score
            
            insights['data_quality_insights'].append(f"Overall data quality score: {quality_score:.2f}")
            
            if quality_score < 0.7:
                insights['recommendations'].append("Data quality is below threshold - consider data cleansing")
            elif quality_score > 0.9:
                insights['key_findings'].append("Excellent data quality detected")
        
        # Statistical insights
        if statistical_results:
            stats = statistical_results.get('summary', {})
            if stats.get('missing_values_total', 0) > 0:
                insights['statistical_insights'].append(f"Missing values detected: {stats['missing_values_total']} total")
            
            if stats.get('duplicate_rows', 0) > 0:
                insights['recommendations'].append(f"Consider removing {stats['duplicate_rows']} duplicate rows")
        
        # ML insights
        if ml_results:
            performance = ml_results.get('performance_score', 0)
            metric = ml_results.get('performance_metric', 'score')
            
            insights['ml_insights'].append(f"Model {metric}: {performance:.3f}")
            
            if performance > 0.8:
                insights['key_findings'].append("High-performing model achieved")
            elif performance < 0.6:
                insights['recommendations'].append("Model performance is low - consider feature engineering or different algorithms")
            
            # Feature importance insights
            if ml_results.get('feature_importance'):
                top_features = sorted(ml_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
                top_feature_names = [f[0] for f in top_features]
                insights['ml_insights'].append(f"Top important features: {', '.join(top_feature_names)}")
        
        # Cross-package insights
        if profiling_results and ml_results:
            insights['key_findings'].append("Comprehensive analysis completed across profiling, quality, and ML stages")
        
        return insights
    
    async def _create_visualization_data(self, df: pd.DataFrame,
                                       profiling_results: Dict[str, Any],
                                       quality_results: Dict[str, Any],
                                       statistical_results: Dict[str, Any],
                                       ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization data for dashboard rendering."""
        viz_data = {
            'data_overview': {
                'column_types': df.dtypes.value_counts().to_dict(),
                'missing_values_by_column': df.isnull().sum().to_dict(),
                'data_shape': {'rows': len(df), 'columns': len(df.columns)}
            }
        }
        
        # Statistical visualizations
        if statistical_results:
            viz_data['statistical'] = {
                'correlation_heatmap': statistical_results.get('correlation_matrix'),
                'outlier_summary': statistical_results.get('outlier_analysis')
            }
        
        # Quality visualizations
        if quality_results and quality_results.get('quality_assessment'):
            qa = quality_results['quality_assessment']
            viz_data['quality'] = {
                'quality_scores': {
                    'overall': qa.overall_score,
                    'completeness': qa.completeness_score,
                    'consistency': qa.consistency_score,
                    'accuracy': qa.accuracy_score
                }
            }
        
        # ML visualizations
        if ml_results:
            viz_data['ml'] = {
                'feature_importance': ml_results.get('feature_importance'),
                'model_performance': {
                    'score': ml_results.get('performance_score'),
                    'metric': ml_results.get('performance_metric')
                }
            }
        
        return viz_data
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecutionMetrics]:
        """Get current status of an active workflow."""
        return self.active_workflows.get(workflow_id)
    
    def get_workflow_history(self, limit: int = 10) -> List[WorkflowExecutionMetrics]:
        """Get workflow execution history."""
        return self.workflow_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all workflows."""
        if not self.workflow_history:
            return {'message': 'No completed workflows'}
        
        completed_workflows = [w for w in self.workflow_history if w.status == 'completed']
        
        if not completed_workflows:
            return {'message': 'No successfully completed workflows'}
        
        execution_times = [w.total_execution_time_seconds for w in completed_workflows if w.total_execution_time_seconds]
        
        return {
            'total_workflows': len(self.workflow_history),
            'successful_workflows': len(completed_workflows),
            'average_execution_time': np.mean(execution_times) if execution_times else 0,
            'min_execution_time': np.min(execution_times) if execution_times else 0,
            'max_execution_time': np.max(execution_times) if execution_times else 0,
            'active_workflows': len(self.active_workflows)
        }
    
    def clear_cache(self):
        """Clear all cached results."""
        if self.result_cache:
            self.result_cache.clear()
            logger.info("Result cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all integrated services."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'performance': self.get_performance_metrics()
        }
        
        # Check individual services
        try:
            if self.profiling_engine:
                health_status['services']['profiling'] = 'available'
            
            if self.quality_assessment:
                health_status['services']['quality'] = 'available'
                
            if self.validation_engine:
                health_status['services']['validation'] = 'available'
                
            health_status['services']['statistical_analysis'] = 'available'
            
            # Check system resources
            import psutil
            health_status['system'] = {
                'memory_usage_percent': psutil.virtual_memory().percent,
                'cpu_usage_percent': psutil.cpu_percent(),
                'active_workflows': len(self.active_workflows)
            }
            
        except ImportError:
            health_status['system'] = {'note': 'System monitoring not available (psutil not installed)'}
        except Exception as e:
            health_status['status'] = 'degraded'
            health_status['error'] = str(e)
        
        return health_status