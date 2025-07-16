"""Tests for the AdvancedProfilingOrchestrator."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from src.packages.data_profiling.application.services.advanced_profiling_orchestrator import (
    AdvancedProfilingOrchestrator
)
from src.packages.data_profiling.domain.entities.data_profile import (
    DataProfile, ProfilingStatus, SchemaProfile, QualityAssessment,
    ColumnProfile, DataType, SemanticType
)


class TestAdvancedProfilingOrchestrator:
    """Test AdvancedProfilingOrchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = AdvancedProfilingOrchestrator(
            max_workers=2,
            enable_ml_features=True
        )
        
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'id': range(1, 101),
            'name': [f'User_{i}' for i in range(1, 101)],
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.uniform(30000, 100000, 100),
            'join_date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'active': np.random.choice([True, False], 100)
        })
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.max_workers == 2
        assert self.orchestrator.enable_ml_features is True
        assert self.orchestrator.schema_service is not None
        assert self.orchestrator.statistical_service is not None
        assert self.orchestrator.pattern_service is not None
        assert self.orchestrator.quality_service is not None
        assert self.orchestrator.performance_optimizer is not None
        assert isinstance(self.orchestrator._analysis_cache, dict)
    
    def test_orchestrator_initialization_no_ml(self):
        """Test orchestrator initialization without ML features."""
        orchestrator = AdvancedProfilingOrchestrator(
            max_workers=4,
            enable_ml_features=False
        )
        
        assert orchestrator.max_workers == 4
        assert orchestrator.enable_ml_features is False
    
    @pytest.mark.asyncio
    async def test_profile_dataset_comprehensive_basic(self):
        """Test comprehensive dataset profiling."""
        # Mock all the services
        mock_schema_profile = SchemaProfile(
            total_columns=7,
            total_rows=100,
            columns=[
                ColumnProfile(
                    column_name="id",
                    data_type=DataType.INTEGER,
                    nullable=False,
                    unique_count=100,
                    total_count=100
                )
            ]
        )
        
        with patch.object(self.orchestrator, '_preprocess_dataset') as mock_preprocess:
            mock_preprocess.return_value = self.sample_df
            
            with patch.object(self.orchestrator, '_execute_core_analysis_parallel') as mock_core:
                mock_core.return_value = {
                    'schema_profile': mock_schema_profile,
                    'column_statistics': {},
                    'column_patterns': {}
                }
                
                with patch.object(self.orchestrator, '_execute_advanced_analysis') as mock_advanced:
                    mock_advanced.return_value = {}
                    
                    with patch.object(self.orchestrator, '_analyze_column_relationships') as mock_relationships:
                        mock_relationships.return_value = {}
                        
                        with patch.object(self.orchestrator, '_comprehensive_quality_assessment') as mock_quality:
                            mock_quality.return_value = QualityAssessment()
                            
                            with patch.object(self.orchestrator, '_generate_intelligent_insights') as mock_insights:
                                mock_insights.return_value = {}
                                
                                result = await self.orchestrator.profile_dataset_comprehensive(
                                    df=self.sample_df,
                                    dataset_name="test_dataset"
                                )
                                
                                assert isinstance(result, DataProfile)
                                assert result.status == ProfilingStatus.COMPLETED
                                assert result.dataset_id.value == "test_dataset"
                                assert result.schema_profile == mock_schema_profile
    
    @pytest.mark.asyncio
    async def test_profile_dataset_comprehensive_with_options(self):
        """Test comprehensive profiling with custom options."""
        profiling_options = {
            'enable_advanced_analysis': True,
            'enable_relationship_analysis': True,
            'max_rows': 50,
            'sample_size': 25
        }
        
        source_metadata = {
            'type': 'csv',
            'connection': {'path': '/test.csv'}
        }
        
        with patch.object(self.orchestrator, '_preprocess_dataset') as mock_preprocess:
            mock_preprocess.return_value = self.sample_df
            
            with patch.object(self.orchestrator, '_execute_core_analysis_parallel') as mock_core:
                mock_core.return_value = {
                    'schema_profile': SchemaProfile(),
                    'column_statistics': {},
                    'column_patterns': {}
                }
                
                with patch.object(self.orchestrator, '_execute_advanced_analysis') as mock_advanced:
                    mock_advanced.return_value = {}
                    
                    with patch.object(self.orchestrator, '_analyze_column_relationships') as mock_relationships:
                        mock_relationships.return_value = {}
                        
                        with patch.object(self.orchestrator, '_comprehensive_quality_assessment') as mock_quality:
                            mock_quality.return_value = QualityAssessment()
                            
                            with patch.object(self.orchestrator, '_generate_intelligent_insights') as mock_insights:
                                mock_insights.return_value = {}
                                
                                result = await self.orchestrator.profile_dataset_comprehensive(
                                    df=self.sample_df,
                                    dataset_name="test_dataset",
                                    source_metadata=source_metadata,
                                    profiling_options=profiling_options
                                )
                                
                                assert result.source_type == 'csv'
                                assert result.source_connection == {'path': '/test.csv'}
                                
                                # Verify options were passed to preprocessing
                                mock_preprocess.assert_called_once_with(self.sample_df, profiling_options)
    
    @pytest.mark.asyncio
    async def test_profile_dataset_comprehensive_error_handling(self):
        """Test error handling in comprehensive profiling."""
        with patch.object(self.orchestrator, '_preprocess_dataset') as mock_preprocess:
            mock_preprocess.side_effect = Exception("Preprocessing failed")
            
            result = await self.orchestrator.profile_dataset_comprehensive(
                df=self.sample_df,
                dataset_name="test_dataset"
            )
            
            assert result.status == ProfilingStatus.FAILED
            assert "Preprocessing failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_preprocess_dataset_sampling(self):
        """Test dataset preprocessing with sampling."""
        large_df = pd.DataFrame({
            'col1': range(50000),
            'col2': [f'value_{i}' for i in range(50000)]
        })
        
        options = {
            'max_rows': 10000,
            'sample_size': 5000
        }
        
        with patch.object(self.orchestrator.performance_optimizer, 'apply_intelligent_sampling') as mock_sampling:
            mock_sampling.return_value = large_df.head(5000)
            
            with patch.object(self.orchestrator.performance_optimizer, 'optimize_memory_usage') as mock_memory:
                mock_memory.return_value = large_df.head(5000)
                
                with patch.object(self.orchestrator, '_handle_special_data_types') as mock_special:
                    mock_special.return_value = large_df.head(5000)
                    
                    result = await self.orchestrator._preprocess_dataset(large_df, options)
                    
                    mock_sampling.assert_called_once_with(large_df, target_size=5000)
                    mock_memory.assert_called_once()
                    mock_special.assert_called_once()
                    assert len(result) == 5000
    
    @pytest.mark.asyncio
    async def test_preprocess_dataset_no_sampling(self):
        """Test dataset preprocessing without sampling."""
        options = {
            'max_rows': 10000,
            'sample_size': 5000
        }
        
        with patch.object(self.orchestrator.performance_optimizer, 'optimize_memory_usage') as mock_memory:
            mock_memory.return_value = self.sample_df
            
            with patch.object(self.orchestrator, '_handle_special_data_types') as mock_special:
                mock_special.return_value = self.sample_df
                
                result = await self.orchestrator._preprocess_dataset(self.sample_df, options)
                
                mock_memory.assert_called_once()
                mock_special.assert_called_once()
                assert len(result) == len(self.sample_df)
    
    @pytest.mark.asyncio
    async def test_execute_core_analysis_parallel(self):
        """Test core analysis execution."""
        mock_schema_profile = SchemaProfile()
        
        with patch.object(self.orchestrator, '_run_schema_analysis_async') as mock_schema:
            mock_schema.return_value = mock_schema_profile
            
            with patch.object(self.orchestrator, '_run_column_statistical_analysis_async') as mock_stats:
                mock_stats.return_value = {'mean': 10.0, 'std': 2.0}
                
                with patch.object(self.orchestrator, '_run_pattern_discovery_async') as mock_patterns:
                    mock_patterns.return_value = []
                    
                    result = await self.orchestrator._execute_core_analysis_parallel(
                        self.sample_df, {}
                    )
                    
                    assert 'schema_profile' in result
                    assert 'column_statistics' in result
                    assert 'column_patterns' in result
                    assert result['schema_profile'] == mock_schema_profile
    
    @pytest.mark.asyncio
    async def test_execute_advanced_analysis_with_ml(self):
        """Test advanced analysis with ML features enabled."""
        core_results = {
            'schema_profile': SchemaProfile()
        }
        
        with patch.object(self.orchestrator, '_detect_semantic_types_ml') as mock_semantic:
            mock_semantic.return_value = {'email': SemanticType.PII_EMAIL}
            
            with patch.object(self.orchestrator, '_analyze_distribution_clusters') as mock_clusters:
                mock_clusters.return_value = {'distribution_clusters': []}
                
                with patch.object(self.orchestrator, '_detect_pattern_anomalies') as mock_anomalies:
                    mock_anomalies.return_value = {'pattern_anomalies': []}
                    
                    with patch.object(self.orchestrator, '_analyze_correlations') as mock_correlations:
                        mock_correlations.return_value = {'correlation_matrix': {}}
                        
                        result = await self.orchestrator._execute_advanced_analysis(
                            self.sample_df, core_results, {}
                        )
                        
                        assert 'semantic_types' in result
                        assert 'distribution_clusters' in result
                        assert 'pattern_anomalies' in result
                        assert 'correlation_analysis' in result
    
    @pytest.mark.asyncio
    async def test_execute_advanced_analysis_without_ml(self):
        """Test advanced analysis without ML features."""
        orchestrator = AdvancedProfilingOrchestrator(enable_ml_features=False)
        core_results = {
            'schema_profile': SchemaProfile()
        }
        
        with patch.object(orchestrator, '_analyze_correlations') as mock_correlations:
            mock_correlations.return_value = {'correlation_matrix': {}}
            
            result = await orchestrator._execute_advanced_analysis(
                self.sample_df, core_results, {}
            )
            
            # Should not include ML features
            assert 'semantic_types' not in result
            assert 'distribution_clusters' not in result
            assert 'pattern_anomalies' not in result
            assert 'correlation_analysis' in result
    
    @pytest.mark.asyncio
    async def test_analyze_column_relationships(self):
        """Test column relationship analysis."""
        # Create DataFrame with relationships
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'user_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'department_id': [1, 1, 2, 2, 3],
            'department_name': ['Engineering', 'Engineering', 'Sales', 'Sales', 'Marketing']
        })
        
        schema_profile = SchemaProfile()
        
        result = await self.orchestrator._analyze_column_relationships(df, schema_profile)
        
        assert 'functional_dependencies' in result
        assert 'inclusion_dependencies' in result
        assert 'statistical_dependencies' in result
        assert 'hierarchical_relationships' in result
        assert isinstance(result['functional_dependencies'], list)
        assert isinstance(result['inclusion_dependencies'], list)
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_assessment(self):
        """Test comprehensive quality assessment."""
        analysis_results = {
            'schema_profile': SchemaProfile(),
            'column_statistics': {},
            'column_patterns': {}
        }
        
        base_assessment = QualityAssessment(
            overall_score=0.8,
            completeness_score=0.9,
            consistency_score=0.7
        )
        
        with patch.object(self.orchestrator.quality_service, 'assess_quality') as mock_assess:
            mock_assess.return_value = base_assessment
            
            with patch.object(self.orchestrator, '_detect_advanced_quality_issues') as mock_issues:
                mock_issues.return_value = []
                
                with patch.object(self.orchestrator, '_calculate_enhanced_quality_scores') as mock_scores:
                    mock_scores.return_value = {
                        'overall': 0.85,
                        'completeness': 0.90,
                        'consistency': 0.80,
                        'accuracy': 0.85,
                        'validity': 0.88,
                        'uniqueness': 0.75
                    }
                    
                    with patch.object(self.orchestrator, '_generate_quality_recommendations') as mock_recommendations:
                        mock_recommendations.return_value = ['Recommendation 1', 'Recommendation 2']
                        
                        result = await self.orchestrator._comprehensive_quality_assessment(
                            self.sample_df, analysis_results
                        )
                        
                        assert isinstance(result, QualityAssessment)
                        assert result.overall_score == 0.85
                        assert result.completeness_score == 0.90
                        assert result.consistency_score == 0.80
                        assert result.accuracy_score == 0.85
                        assert result.validity_score == 0.88
                        assert result.uniqueness_score == 0.75
    
    @pytest.mark.asyncio
    async def test_generate_intelligent_insights(self):
        """Test intelligent insights generation."""
        analysis_results = {
            'schema_profile': SchemaProfile(),
            'column_statistics': {},
            'column_patterns': {}
        }
        
        quality_assessment = QualityAssessment()
        
        with patch.object(self.orchestrator, '_analyze_dataset_characteristics') as mock_chars:
            mock_chars.return_value = {'shape': '100 rows × 7 columns'}
            
            with patch.object(self.orchestrator, '_identify_key_patterns') as mock_patterns:
                mock_patterns.return_value = ['Pattern 1', 'Pattern 2']
                
                with patch.object(self.orchestrator, '_generate_quality_insights') as mock_quality:
                    mock_quality.return_value = ['Quality insight 1']
                    
                    with patch.object(self.orchestrator, '_identify_optimization_opportunities') as mock_optimization:
                        mock_optimization.return_value = ['Optimization 1']
                        
                        with patch.object(self.orchestrator, '_generate_actionable_recommendations') as mock_recommendations:
                            mock_recommendations.return_value = ['Recommendation 1']
                            
                            result = await self.orchestrator._generate_intelligent_insights(
                                self.sample_df, analysis_results, quality_assessment
                            )
                            
                            assert 'dataset_characteristics' in result
                            assert 'data_patterns' in result
                            assert 'quality_insights' in result
                            assert 'optimization_opportunities' in result
                            assert 'recommendations' in result
    
    @pytest.mark.asyncio
    async def test_run_schema_analysis_async(self):
        """Test async schema analysis."""
        mock_schema_profile = SchemaProfile()
        
        with patch.object(self.orchestrator.schema_service, 'infer') as mock_infer:
            mock_infer.return_value = mock_schema_profile
            
            result = await self.orchestrator._run_schema_analysis_async(self.sample_df)
            
            assert result == mock_schema_profile
            mock_infer.assert_called_once_with(self.sample_df)
    
    @pytest.mark.asyncio
    async def test_run_column_statistical_analysis_async(self):
        """Test async column statistical analysis."""
        mock_stats = {'mean': 10.0, 'std': 2.0}
        
        with patch.object(self.orchestrator.statistical_service, 'profile_column') as mock_profile:
            mock_profile.return_value = mock_stats
            
            result = await self.orchestrator._run_column_statistical_analysis_async(
                self.sample_df['age'], 'age'
            )
            
            assert result == mock_stats
            mock_profile.assert_called_once_with(self.sample_df['age'], 'age')
    
    @pytest.mark.asyncio
    async def test_run_pattern_discovery_async(self):
        """Test async pattern discovery."""
        mock_patterns = [{'pattern_type': 'email', 'frequency': 10}]
        
        with patch.object(self.orchestrator.pattern_service, 'discover_patterns') as mock_discover:
            mock_discover.return_value = mock_patterns
            
            result = await self.orchestrator._run_pattern_discovery_async(
                self.sample_df['email'], 'email'
            )
            
            assert result == mock_patterns
            mock_discover.assert_called_once_with(self.sample_df['email'], 'email')
    
    @pytest.mark.asyncio
    async def test_detect_semantic_types_ml(self):
        """Test ML-based semantic type detection."""
        result = await self.orchestrator._detect_semantic_types_ml(self.sample_df)
        
        assert isinstance(result, dict)
        # Should detect email semantic type
        assert 'email' in result
        assert result['email'] == SemanticType.PII_EMAIL
    
    @pytest.mark.asyncio
    async def test_analyze_correlations(self):
        """Test correlation analysis."""
        numeric_df = self.sample_df.select_dtypes(include=[np.number])
        
        result = await self.orchestrator._analyze_correlations(numeric_df)
        
        assert 'correlation_matrix' in result
        assert 'strong_correlations' in result
        assert isinstance(result['correlation_matrix'], dict)
        assert isinstance(result['strong_correlations'], list)
    
    def test_calculate_functional_dependency(self):
        """Test functional dependency calculation."""
        # Create test data with functional dependency
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'user_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        # user_id -> user_name (should have high dependency)
        dependency = self.orchestrator._calculate_functional_dependency(df, 'user_id', 'user_name')
        assert dependency == 1.0
        
        # user_name -> user_id (should also have high dependency for unique names)
        dependency = self.orchestrator._calculate_functional_dependency(df, 'user_name', 'user_id')
        assert dependency == 1.0
    
    def test_calculate_inclusion_dependency(self):
        """Test inclusion dependency calculation."""
        # Create test data with inclusion dependency
        df = pd.DataFrame({
            'subset_col': ['A', 'B', 'C'],
            'superset_col': ['A', 'B', 'C', 'D', 'E']
        })
        
        # subset_col ⊆ superset_col
        inclusion = self.orchestrator._calculate_inclusion_dependency(df, 'subset_col', 'superset_col')
        assert inclusion == 1.0
        
        # superset_col ⊄ subset_col
        inclusion = self.orchestrator._calculate_inclusion_dependency(df, 'superset_col', 'subset_col')
        assert inclusion == 0.6  # 3/5 values are included
    
    @pytest.mark.asyncio
    async def test_handle_special_data_types(self):
        """Test handling of special data types."""
        # Create DataFrame with string dates
        df = pd.DataFrame({
            'date_string': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'regular_string': ['A', 'B', 'C']
        })
        
        result = await self.orchestrator._handle_special_data_types(df)
        
        # Should convert date strings to datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date_string'])
        assert pd.api.types.is_object_dtype(result['regular_string'])
    
    def test_is_likely_datetime_string(self):
        """Test datetime string detection."""
        # Test various datetime formats
        datetime_series = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        assert self.orchestrator._is_likely_datetime_string(datetime_series) is True
        
        date_series = pd.Series(['01/01/2023', '01/02/2023', '01/03/2023'])
        assert self.orchestrator._is_likely_datetime_string(date_series) is True
        
        regular_series = pd.Series(['A', 'B', 'C'])
        assert self.orchestrator._is_likely_datetime_string(regular_series) is False
    
    @pytest.mark.asyncio
    async def test_classify_semantic_type_ml(self):
        """Test ML-based semantic type classification."""
        # Test email detection
        email_series = pd.Series(['user1@example.com', 'user2@test.org', 'user3@company.net'])
        result = await self.orchestrator._classify_semantic_type_ml(email_series)
        assert result == SemanticType.PII_EMAIL
        
        # Test phone detection
        phone_series = pd.Series(['123-456-7890', '(555) 123-4567', '987-654-3210'])
        result = await self.orchestrator._classify_semantic_type_ml(phone_series)
        assert result == SemanticType.PII_PHONE
        
        # Test URL detection
        url_series = pd.Series(['https://example.com', 'http://test.org', 'https://company.net'])
        result = await self.orchestrator._classify_semantic_type_ml(url_series)
        assert result == SemanticType.URL
        
        # Test unknown type
        unknown_series = pd.Series(['A', 'B', 'C'])
        result = await self.orchestrator._classify_semantic_type_ml(unknown_series)
        assert result == SemanticType.UNKNOWN
    
    def test_determine_profiling_strategy(self):
        """Test profiling strategy determination."""
        # Small dataset
        small_df = pd.DataFrame({'col1': range(1000)})
        strategy = self.orchestrator._determine_profiling_strategy(small_df, {})
        assert strategy == "comprehensive"
        
        # Medium dataset
        medium_df = pd.DataFrame({'col1': range(500000)})
        strategy = self.orchestrator._determine_profiling_strategy(medium_df, {})
        assert strategy == "intelligent_sampling"
        
        # Large dataset
        large_df = pd.DataFrame({'col1': range(2000000)})
        strategy = self.orchestrator._determine_profiling_strategy(large_df, {})
        assert strategy == "large_scale_sampling"
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        memory_usage = self.orchestrator._estimate_memory_usage(self.sample_df)
        assert isinstance(memory_usage, float)
        assert memory_usage > 0
    
    def test_format_insights_summary(self):
        """Test insights summary formatting."""
        insights = {
            'dataset_characteristics': {'shape': '100 rows × 7 columns'},
            'recommendations': ['Rec 1', 'Rec 2', 'Rec 3']
        }
        
        summary = self.orchestrator._format_insights_summary(insights)
        assert "Dataset: 100 rows × 7 columns" in summary
        assert "Recommendations: 3 items" in summary
    
    def test_analyze_dataset_characteristics(self):
        """Test dataset characteristics analysis."""
        characteristics = self.orchestrator._analyze_dataset_characteristics(self.sample_df)
        
        assert 'shape' in characteristics
        assert 'memory_usage_mb' in characteristics
        assert 'data_types' in characteristics
        assert 'sparsity' in characteristics
        
        assert characteristics['shape'] == "100 rows × 7 columns"
        assert isinstance(characteristics['memory_usage_mb'], float)
        assert isinstance(characteristics['data_types'], dict)
        assert isinstance(characteristics['sparsity'], float)
    
    def test_identify_optimization_opportunities(self):
        """Test optimization opportunity identification."""
        # Create large DataFrame to trigger memory optimization suggestion
        large_data = np.random.random((10000, 50))
        large_df = pd.DataFrame(large_data)
        
        opportunities = self.orchestrator._identify_optimization_opportunities(large_df, {})
        
        assert isinstance(opportunities, list)
        # Should suggest memory optimization for large dataset
        assert any('memory efficiency' in opp for opp in opportunities)
    
    def test_generate_actionable_recommendations(self):
        """Test actionable recommendations generation."""
        # Create quality assessment with low scores
        quality_assessment = QualityAssessment(
            completeness_score=0.7,
            consistency_score=0.6
        )
        
        analysis_results = {
            'column_patterns': {'email': [{'pattern': 'email'}]}
        }
        
        recommendations = self.orchestrator._generate_actionable_recommendations(
            self.sample_df, analysis_results, quality_assessment
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should suggest improvements for low quality scores
        assert any('completeness' in rec for rec in recommendations)
        assert any('consistency' in rec for rec in recommendations)
        assert any('patterns' in rec for rec in recommendations)