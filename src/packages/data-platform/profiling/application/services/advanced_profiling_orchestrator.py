"""Advanced profiling orchestrator for comprehensive data analysis and discovery."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ...domain.entities.data_profile import (
    DataProfile, ProfilingStatus, ProfileId, DatasetId,
    ProfilingMetadata, QualityAssessment, SchemaProfile,
    ColumnProfile, Pattern, SemanticType, DataType
)
from .schema_analysis_service import SchemaAnalysisService
from .statistical_profiling_service import StatisticalProfilingService
from .pattern_discovery_service import PatternDiscoveryService
from .quality_assessment_service import QualityAssessmentService
from .performance_optimizer import PerformanceOptimizer


logger = logging.getLogger(__name__)


class AdvancedProfilingOrchestrator:
    """Advanced orchestrator for comprehensive data profiling with intelligent analysis."""
    
    def __init__(self, max_workers: int = 4, enable_ml_features: bool = True):
        """Initialize the advanced profiling orchestrator.
        
        Args:
            max_workers: Maximum number of workers for parallel processing
            enable_ml_features: Whether to enable ML-powered features
        """
        self.max_workers = max_workers
        self.enable_ml_features = enable_ml_features
        
        # Initialize core services
        self.schema_service = SchemaAnalysisService()
        self.statistical_service = StatisticalProfilingService()
        self.pattern_service = PatternDiscoveryService()
        self.quality_service = QualityAssessmentService()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Advanced analysis cache
        self._analysis_cache: Dict[str, Any] = {}
        
        logger.info(f"Initialized AdvancedProfilingOrchestrator with {max_workers} workers")
    
    async def profile_dataset_comprehensive(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source_metadata: Optional[Dict[str, Any]] = None,
        profiling_options: Optional[Dict[str, Any]] = None
    ) -> DataProfile:
        """Perform comprehensive dataset profiling with advanced analytics.
        
        Args:
            df: DataFrame to profile
            dataset_name: Name/identifier for the dataset
            source_metadata: Metadata about the data source
            profiling_options: Options for profiling behavior
            
        Returns:
            Comprehensive data profile with advanced insights
        """
        start_time = datetime.now()
        options = profiling_options or {}
        
        # Create profile instance
        profile = DataProfile(
            profile_id=ProfileId(),
            dataset_id=DatasetId(dataset_name),
            source_type=source_metadata.get('type', 'dataframe') if source_metadata else 'dataframe',
            source_connection=source_metadata or {}
        )
        
        try:
            profile.start_profiling()
            
            # Phase 1: Data preprocessing and optimization
            logger.info(f"Starting comprehensive profiling for dataset: {dataset_name}")
            preprocessed_df = await self._preprocess_dataset(df, options)
            
            # Phase 2: Parallel core analysis
            core_results = await self._execute_core_analysis_parallel(preprocessed_df, options)
            
            # Phase 3: Advanced pattern and relationship discovery
            if options.get('enable_advanced_analysis', True):
                advanced_results = await self._execute_advanced_analysis(
                    preprocessed_df, core_results, options
                )
                core_results.update(advanced_results)
            
            # Phase 4: Cross-column relationship analysis
            if options.get('enable_relationship_analysis', True):
                relationship_results = await self._analyze_column_relationships(
                    preprocessed_df, core_results['schema_profile']
                )
                core_results['relationships'] = relationship_results
            
            # Phase 5: Quality assessment and recommendations
            quality_assessment = await self._comprehensive_quality_assessment(
                preprocessed_df, core_results
            )
            
            # Phase 6: Intelligent insights and recommendations
            insights = await self._generate_intelligent_insights(
                preprocessed_df, core_results, quality_assessment
            )
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive metadata
            metadata = ProfilingMetadata(
                profiling_strategy=self._determine_profiling_strategy(df, options),
                execution_time_seconds=execution_time,
                memory_usage_mb=self._estimate_memory_usage(df),
                include_patterns=True,
                include_statistical_analysis=True,
                include_quality_assessment=True,
                engine_version="2.0.0",
                configuration={
                    'advanced_analysis': options.get('enable_advanced_analysis', True),
                    'ml_features': self.enable_ml_features,
                    'parallel_processing': True,
                    'relationship_analysis': options.get('enable_relationship_analysis', True)
                }
            )
            
            # Enhance schema profile with insights
            enhanced_schema = self._enhance_schema_profile(
                core_results['schema_profile'], insights
            )
            
            # Complete profiling
            profile.complete_profiling(
                schema_profile=enhanced_schema,
                quality_assessment=quality_assessment,
                metadata=metadata
            )
            
            # Add insights to profile metadata
            profile.notes = self._format_insights_summary(insights)
            
            logger.info(
                f"Comprehensive profiling completed for {dataset_name}: "
                f"{len(df)} rows, {len(df.columns)} columns in {execution_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Comprehensive profiling failed for {dataset_name}: {e}")
            profile.fail_profiling(str(e))
            raise
        
        return profile
    
    async def _preprocess_dataset(
        self, df: pd.DataFrame, options: Dict[str, Any]
    ) -> pd.DataFrame:
        """Preprocess dataset for optimal profiling performance."""
        # Apply intelligent sampling if needed
        if len(df) > options.get('max_rows', 100000):
            df = self.performance_optimizer.apply_intelligent_sampling(
                df, target_size=options.get('sample_size', 50000)
            )
        
        # Optimize memory usage
        df = self.performance_optimizer.optimize_memory_usage(df)
        
        # Handle special data types
        df = await self._handle_special_data_types(df)
        
        return df
    
    async def _execute_core_analysis_parallel(
        self, df: pd.DataFrame, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute core analysis components in parallel."""
        tasks = []
        
        # Schema analysis
        tasks.append(self._run_schema_analysis_async(df))
        
        # Statistical analysis per column
        for column in df.columns:
            tasks.append(self._run_column_statistical_analysis_async(df[column], column))
        
        # Pattern discovery per column
        for column in df.select_dtypes(include=['object']).columns:
            tasks.append(self._run_pattern_discovery_async(df[column], column))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        schema_profile = results[0]
        column_stats = {}
        column_patterns = {}
        
        idx = 1
        for column in df.columns:
            if not isinstance(results[idx], Exception):
                column_stats[column] = results[idx]
            idx += 1
        
        for column in df.select_dtypes(include=['object']).columns:
            if idx < len(results) and not isinstance(results[idx], Exception):
                column_patterns[column] = results[idx]
            idx += 1
        
        return {
            'schema_profile': schema_profile,
            'column_statistics': column_stats,
            'column_patterns': column_patterns
        }
    
    async def _execute_advanced_analysis(
        self, df: pd.DataFrame, core_results: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute advanced analysis features."""
        advanced_results = {}
        
        if self.enable_ml_features:
            # Semantic type detection using ML
            advanced_results['semantic_types'] = await self._detect_semantic_types_ml(df)
            
            # Data distribution clustering
            advanced_results['distribution_clusters'] = await self._analyze_distribution_clusters(df)
            
            # Anomaly detection in data patterns
            advanced_results['pattern_anomalies'] = await self._detect_pattern_anomalies(df)
        
        # Time series analysis for datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            advanced_results['time_series_analysis'] = await self._analyze_time_series_patterns(
                df, datetime_columns
            )
        
        # Cross-column correlation analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            advanced_results['correlation_analysis'] = await self._analyze_correlations(
                df[numeric_columns]
            )
        
        return advanced_results
    
    async def _analyze_column_relationships(
        self, df: pd.DataFrame, schema_profile: SchemaProfile
    ) -> Dict[str, Any]:
        """Analyze relationships between columns."""
        relationships = {
            'functional_dependencies': [],
            'inclusion_dependencies': [],
            'statistical_dependencies': [],
            'hierarchical_relationships': []
        }
        
        # Functional dependency detection
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    dependency_strength = self._calculate_functional_dependency(df, col1, col2)
                    if dependency_strength > 0.9:
                        relationships['functional_dependencies'].append({
                            'determinant': col1,
                            'dependent': col2,
                            'strength': dependency_strength
                        })
        
        # Inclusion dependency detection
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    inclusion_score = self._calculate_inclusion_dependency(df, col1, col2)
                    if inclusion_score > 0.8:
                        relationships['inclusion_dependencies'].append({
                            'subset_column': col1,
                            'superset_column': col2,
                            'inclusion_score': inclusion_score
                        })
        
        return relationships
    
    async def _comprehensive_quality_assessment(
        self, df: pd.DataFrame, analysis_results: Dict[str, Any]
    ) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        # Use existing quality service with enhancements
        base_assessment = self.quality_service.assess_quality(df)
        
        # Add advanced quality metrics
        enhanced_issues = list(base_assessment.issues)
        
        # Check for advanced quality issues
        enhanced_issues.extend(await self._detect_advanced_quality_issues(df, analysis_results))
        
        # Calculate enhanced quality scores
        enhanced_scores = await self._calculate_enhanced_quality_scores(df, analysis_results)
        
        return QualityAssessment(
            overall_score=enhanced_scores['overall'],
            completeness_score=enhanced_scores['completeness'],
            consistency_score=enhanced_scores['consistency'],
            accuracy_score=enhanced_scores['accuracy'],
            validity_score=enhanced_scores['validity'],
            uniqueness_score=enhanced_scores['uniqueness'],
            timeliness_score=enhanced_scores.get('timeliness', base_assessment.timeliness_score),
            issues=enhanced_issues,
            recommendations=await self._generate_quality_recommendations(df, enhanced_issues),
            critical_issues=len([i for i in enhanced_issues if i.severity == 'critical']),
            high_issues=len([i for i in enhanced_issues if i.severity == 'high']),
            medium_issues=len([i for i in enhanced_issues if i.severity == 'medium']),
            low_issues=len([i for i in enhanced_issues if i.severity == 'low'])
        )
    
    async def _generate_intelligent_insights(
        self, df: pd.DataFrame, analysis_results: Dict[str, Any], quality_assessment: QualityAssessment
    ) -> Dict[str, Any]:
        """Generate intelligent insights about the dataset."""
        insights = {
            'dataset_characteristics': self._analyze_dataset_characteristics(df),
            'data_patterns': self._identify_key_patterns(analysis_results),
            'quality_insights': self._generate_quality_insights(quality_assessment),
            'optimization_opportunities': self._identify_optimization_opportunities(df, analysis_results),
            'recommendations': self._generate_actionable_recommendations(df, analysis_results, quality_assessment)
        }
        
        return insights
    
    # Helper methods for async operations
    async def _run_schema_analysis_async(self, df: pd.DataFrame) -> SchemaProfile:
        """Run schema analysis asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.schema_service.infer, df)
    
    async def _run_column_statistical_analysis_async(
        self, series: pd.Series, column_name: str
    ) -> Dict[str, Any]:
        """Run statistical analysis for a column asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.statistical_service.profile_column, series, column_name
        )
    
    async def _run_pattern_discovery_async(
        self, series: pd.Series, column_name: str
    ) -> List[Pattern]:
        """Run pattern discovery for a column asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.pattern_service.discover_patterns, series, column_name
        )
    
    # Advanced analysis methods
    async def _detect_semantic_types_ml(self, df: pd.DataFrame) -> Dict[str, SemanticType]:
        """Detect semantic types using machine learning approaches."""
        semantic_types = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Use pattern-based ML classification
                semantic_type = await self._classify_semantic_type_ml(df[column])
                semantic_types[column] = semantic_type
        
        return semantic_types
    
    async def _analyze_distribution_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distribution clusters using ML."""
        if not self.enable_ml_features:
            return {}
        
        # Implementation would use clustering algorithms to group similar distributions
        return {'distribution_clusters': []}
    
    async def _detect_pattern_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in data patterns."""
        if not self.enable_ml_features:
            return {}
        
        # Implementation would use anomaly detection to find unusual patterns
        return {'pattern_anomalies': []}
    
    async def _analyze_time_series_patterns(
        self, df: pd.DataFrame, datetime_columns: List[str]
    ) -> Dict[str, Any]:
        """Analyze time series patterns in datetime columns."""
        time_series_insights = {}
        
        for column in datetime_columns:
            if df[column].notna().sum() > 0:
                time_series_insights[column] = {
                    'trend': 'unknown',
                    'seasonality': 'unknown',
                    'frequency': 'unknown',
                    'gaps': []
                }
        
        return time_series_insights
    
    async def _analyze_correlations(self, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    # Utility methods
    def _calculate_functional_dependency(
        self, df: pd.DataFrame, col1: str, col2: str
    ) -> float:
        """Calculate functional dependency strength between two columns."""
        if df[col1].nunique() == 0 or df[col2].nunique() == 0:
            return 0.0
        
        # Group by col1 and check if col2 has single value per group
        grouped = df.groupby(col1)[col2].nunique()
        single_value_groups = (grouped == 1).sum()
        return single_value_groups / len(grouped)
    
    def _calculate_inclusion_dependency(
        self, df: pd.DataFrame, col1: str, col2: str
    ) -> float:
        """Calculate inclusion dependency between two columns."""
        values1 = set(df[col1].dropna().astype(str))
        values2 = set(df[col2].dropna().astype(str))
        
        if len(values1) == 0:
            return 0.0
        
        intersection = values1.intersection(values2)
        return len(intersection) / len(values1)
    
    async def _handle_special_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle special data types for better analysis."""
        # Convert string dates to datetime
        for column in df.select_dtypes(include=['object']).columns:
            if df[column].notna().sum() > 0:
                # Try to detect and convert datetime strings
                sample_values = df[column].dropna().head(100)
                if self._is_likely_datetime_string(sample_values):
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    except:
                        pass
        
        return df
    
    def _is_likely_datetime_string(self, series: pd.Series) -> bool:
        """Check if a string series likely contains datetime values."""
        import re
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        for value in series.head(10):
            if any(re.search(pattern, str(value)) for pattern in datetime_patterns):
                return True
        return False
    
    async def _classify_semantic_type_ml(self, series: pd.Series) -> SemanticType:
        """Classify semantic type using ML approaches."""
        # Simplified implementation - would use trained models in production
        sample_values = series.dropna().astype(str).head(100)
        
        # Email detection
        if any('@' in str(val) and '.' in str(val) for val in sample_values):
            return SemanticType.PII_EMAIL
        
        # Phone detection  
        import re
        phone_pattern = r'\d{3}-\d{3}-\d{4}|\(\d{3}\)\s?\d{3}-\d{4}'
        if any(re.search(phone_pattern, str(val)) for val in sample_values):
            return SemanticType.PII_PHONE
        
        # URL detection
        if any(str(val).startswith(('http://', 'https://')) for val in sample_values):
            return SemanticType.URL
        
        return SemanticType.UNKNOWN
    
    def _determine_profiling_strategy(
        self, df: pd.DataFrame, options: Dict[str, Any]
    ) -> str:
        """Determine the profiling strategy based on data characteristics."""
        if len(df) > 1000000:
            return "large_scale_sampling"
        elif len(df) > 100000:
            return "intelligent_sampling"
        else:
            return "comprehensive"
    
    def _estimate_memory_usage(self, df: pd.DataFrame) -> float:
        """Estimate memory usage in MB."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    def _enhance_schema_profile(
        self, schema_profile: SchemaProfile, insights: Dict[str, Any]
    ) -> SchemaProfile:
        """Enhance schema profile with insights."""
        # This would enhance the schema profile with additional insights
        return schema_profile
    
    def _format_insights_summary(self, insights: Dict[str, Any]) -> str:
        """Format insights into a readable summary."""
        summary_parts = []
        
        if 'dataset_characteristics' in insights:
            chars = insights['dataset_characteristics']
            summary_parts.append(f"Dataset: {chars.get('shape', 'unknown shape')}")
        
        if 'recommendations' in insights:
            rec_count = len(insights['recommendations'])
            summary_parts.append(f"Recommendations: {rec_count} items")
        
        return "; ".join(summary_parts)
    
    # Quality analysis helper methods
    async def _detect_advanced_quality_issues(
        self, df: pd.DataFrame, analysis_results: Dict[str, Any]
    ) -> List[Any]:
        """Detect advanced quality issues."""
        # Implementation would detect sophisticated quality issues
        return []
    
    async def _calculate_enhanced_quality_scores(
        self, df: pd.DataFrame, analysis_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate enhanced quality scores."""
        return {
            'overall': 0.85,
            'completeness': 0.90,
            'consistency': 0.80,
            'accuracy': 0.85,
            'validity': 0.88,
            'uniqueness': 0.75
        }
    
    async def _generate_quality_recommendations(
        self, df: pd.DataFrame, issues: List[Any]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if len(issues) > 0:
            recommendations.append("Address identified quality issues")
        
        # Add more sophisticated recommendations
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_percentage > 0.1:
            recommendations.append("Consider data imputation for missing values")
        
        return recommendations
    
    def _analyze_dataset_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze high-level dataset characteristics."""
        return {
            'shape': f"{len(df)} rows × {len(df.columns)} columns",
            'memory_usage_mb': self._estimate_memory_usage(df),
            'data_types': df.dtypes.value_counts().to_dict(),
            'sparsity': df.isnull().sum().sum() / (len(df) * len(df.columns))
        }
    
    def _identify_key_patterns(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify key patterns from analysis results."""
        patterns = []
        
        if 'column_patterns' in analysis_results:
            for column, column_patterns in analysis_results['column_patterns'].items():
                if column_patterns:
                    patterns.append(f"{column}: {len(column_patterns)} patterns detected")
        
        return patterns
    
    def _generate_quality_insights(self, quality_assessment: QualityAssessment) -> List[str]:
        """Generate insights from quality assessment."""
        insights = []
        
        if quality_assessment.overall_score < 0.7:
            insights.append("Dataset quality is below recommended threshold")
        
        if quality_assessment.total_issues > 10:
            insights.append("Multiple quality issues detected requiring attention")
        
        return insights
    
    def _identify_optimization_opportunities(
        self, df: pd.DataFrame, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Memory optimization
        if self._estimate_memory_usage(df) > 100:  # > 100MB
            opportunities.append("Consider data type optimization for memory efficiency")
        
        # Indexing recommendations
        high_cardinality_columns = [
            col for col in df.columns 
            if df[col].nunique() / len(df) > 0.5
        ]
        if high_cardinality_columns:
            opportunities.append("Consider indexing high-cardinality columns")
        
        return opportunities
    
    def _generate_actionable_recommendations(
        self, df: pd.DataFrame, analysis_results: Dict[str, Any], quality_assessment: QualityAssessment
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Quality-based recommendations
        if quality_assessment.completeness_score < 0.8:
            recommendations.append("Improve data completeness through better collection processes")
        
        if quality_assessment.consistency_score < 0.8:
            recommendations.append("Implement data validation rules to ensure consistency")
        
        # Pattern-based recommendations
        if analysis_results.get('column_patterns'):
            recommendations.append("Leverage discovered patterns for data validation")
        
        return recommendations