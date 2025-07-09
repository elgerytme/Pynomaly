"""Integration tests for advanced analytics features."""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from pynomaly.domain.entities import Dataset, DetectionResult, Anomaly, Detector
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType
from pynomaly.features.advanced_analytics import (
    AdvancedAnalytics,
    AnalyticsEngine,
    TimeSeriesAnalyzer,
    PatternDetector,
    AnomalyExplainer,
    AnalysisType,
    get_analytics_engine,
)


class MockDetector(Detector):
    """Mock detector for testing."""
    
    def __init__(self):
        self.algorithm = "mock_detector"
        self.is_fitted = False
    
    def fit(self, dataset: Dataset) -> None:
        """Mock fit method."""
        self.is_fitted = True
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Mock predict method."""
        anomalies = []
        for i in range(min(3, len(dataset.data))):
            anomaly = Anomaly(
                id=f"anomaly_{i}",
                score=AnomalyScore(0.8 + i * 0.05),
                type=AnomalyType.POINT,
                timestamp=datetime.now(),
                data_point=dataset.data.iloc[i].to_dict(),
            )
            anomalies.append(anomaly)
        
        return DetectionResult(
            anomalies=anomalies,
            threshold=0.7,
            metadata={
                "algorithm": self.algorithm,
                "execution_time_ms": 100.0,
            }
        )


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    return pd.DataFrame({
        'timestamp': dates,
        'value': values,
        'feature_1': np.random.randn(len(dates)),
        'feature_2': np.random.randn(len(dates)),
    })


@pytest.fixture
def sample_dataset(sample_time_series_data):
    """Create sample dataset for testing."""
    return Dataset(
        name="test_dataset",
        data=sample_time_series_data,
        description="Test dataset for integration testing",
    )


@pytest.fixture
def sample_detection_results(sample_dataset):
    """Create sample detection results for testing."""
    detector = MockDetector()
    detector.fit(sample_dataset)
    return [detector.predict(sample_dataset)]


@pytest.mark.asyncio
class TestAdvancedAnalyticsIntegration:
    """Integration tests for advanced analytics system."""
    
    async def test_time_series_analyzer_integration(self, sample_time_series_data):
        """Test time series analyzer integration."""
        analyzer = TimeSeriesAnalyzer()
        
        # Test time series analysis
        result = await analyzer.analyze_time_series(
            sample_time_series_data,
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Verify result structure
        assert result.analysis_type == AnalysisType.TIME_SERIES
        assert isinstance(result.insights, dict)
        assert "data_points" in result.insights
        assert "time_range" in result.insights
        assert "value_statistics" in result.insights
        assert "seasonality" in result.insights
        assert "trend" in result.insights
        assert "anomalies" in result.insights
        
        # Verify insights content
        assert result.insights["data_points"] == len(sample_time_series_data)
        assert result.insights["value_statistics"]["mean"] is not None
        assert result.insights["value_statistics"]["std"] is not None
        
        # Verify visualizations
        assert len(result.visualizations) > 0
        assert any(viz["type"] == "line_chart" for viz in result.visualizations)
        
        # Verify recommendations
        assert isinstance(result.recommendations, list)
        assert result.confidence_score > 0
        assert result.execution_time_ms > 0
    
    async def test_pattern_detector_integration(self, sample_dataset, sample_detection_results):
        """Test pattern detector integration."""
        detector = PatternDetector()
        
        # Test pattern detection
        result = await detector.detect_patterns(sample_detection_results, sample_dataset)
        
        # Verify result structure
        assert result.analysis_type == AnalysisType.PATTERN_DETECTION
        assert isinstance(result.insights, dict)
        assert "total_anomalies" in result.insights
        assert "detection_results" in result.insights
        assert "temporal_patterns" in result.insights
        assert "spatial_patterns" in result.insights
        assert "type_patterns" in result.insights
        assert "severity_patterns" in result.insights
        
        # Verify insights content
        assert result.insights["total_anomalies"] > 0
        assert result.insights["detection_results"] == len(sample_detection_results)
        
        # Verify visualizations
        assert isinstance(result.visualizations, list)
        
        # Verify recommendations
        assert isinstance(result.recommendations, list)
        assert result.confidence_score > 0
        assert result.execution_time_ms > 0
    
    async def test_anomaly_explainer_integration(self, sample_dataset, sample_detection_results):
        """Test anomaly explainer integration."""
        explainer = AnomalyExplainer()
        
        # Get first anomaly from detection results
        anomaly = sample_detection_results[0].anomalies[0]
        detection_result = sample_detection_results[0]
        
        # Test anomaly explanation
        result = await explainer.explain_anomaly(anomaly, sample_dataset, detection_result)
        
        # Verify result structure
        assert result.analysis_type == AnalysisType.ANOMALY_EXPLANATION
        assert isinstance(result.insights, dict)
        assert "anomaly_id" in result.insights
        assert "explanation_type" in result.insights
        assert "feature_contributions" in result.insights
        assert "statistical_explanation" in result.insights
        assert "contextual_explanation" in result.insights
        assert "similarity_analysis" in result.insights
        
        # Verify insights content
        assert result.insights["anomaly_id"] == anomaly.id
        assert result.insights["explanation_type"] == "feature_based"
        
        # Verify visualizations
        assert isinstance(result.visualizations, list)
        assert len(result.visualizations) > 0
        
        # Verify recommendations
        assert isinstance(result.recommendations, list)
        assert result.confidence_score > 0
        assert result.execution_time_ms > 0
    
    async def test_analytics_engine_integration(self, sample_dataset, sample_detection_results):
        """Test analytics engine integration."""
        engine = AnalyticsEngine()
        
        # Test comprehensive analysis
        results = await engine.run_comprehensive_analysis(
            sample_dataset,
            sample_detection_results,
            analysis_types=[AnalysisType.TIME_SERIES, AnalysisType.PATTERN_DETECTION]
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert AnalysisType.TIME_SERIES in results
        assert AnalysisType.PATTERN_DETECTION in results
        
        # Verify each analysis result
        for analysis_type, result in results.items():
            assert result.analysis_type == analysis_type
            assert isinstance(result.insights, dict)
            assert isinstance(result.visualizations, list)
            assert isinstance(result.recommendations, list)
            assert result.confidence_score > 0
            assert result.execution_time_ms > 0
        
        # Test anomaly explanation
        anomalies = sample_detection_results[0].anomalies
        explanations = await engine.explain_anomalies(
            anomalies[:2],  # Limit to first 2 anomalies
            sample_dataset,
            sample_detection_results[0]
        )
        
        # Verify explanations
        assert len(explanations) == 2
        for explanation in explanations:
            assert explanation.analysis_type == AnalysisType.ANOMALY_EXPLANATION
            assert isinstance(explanation.insights, dict)
            assert "anomaly_id" in explanation.insights
        
        # Verify analysis history
        assert len(engine.analysis_history) > 0
        history_entry = engine.analysis_history[-1]
        assert "timestamp" in history_entry
        assert "dataset_name" in history_entry
        assert "analysis_types" in history_entry
        assert "results_count" in history_entry
    
    async def test_advanced_analytics_facade_integration(self, sample_dataset, sample_detection_results):
        """Test advanced analytics facade integration."""
        analytics = AdvancedAnalytics()
        
        # Test dataset analysis
        report = await analytics.analyze_dataset(sample_dataset, sample_detection_results)
        
        # Verify report structure
        assert isinstance(report, dict)
        assert "dataset_name" in report
        assert "analysis_timestamp" in report
        assert "analyses_performed" in report
        assert "results" in report
        assert "summary" in report
        
        # Verify report content
        assert report["dataset_name"] == sample_dataset.name
        assert isinstance(report["analyses_performed"], list)
        assert len(report["analyses_performed"]) > 0
        
        # Verify results
        for analysis_type in report["analyses_performed"]:
            assert analysis_type in report["results"]
            result = report["results"][analysis_type]
            assert "analysis_type" in result
            assert "insights" in result
            assert "visualizations" in result
            assert "recommendations" in result
        
        # Verify summary
        summary = report["summary"]
        assert "total_analyses" in summary
        assert "key_insights" in summary
        assert "overall_recommendations" in summary
        assert "confidence_scores" in summary
        assert summary["total_analyses"] > 0
    
    async def test_global_analytics_engine_integration(self, sample_dataset):
        """Test global analytics engine integration."""
        # Test global engine retrieval
        engine1 = get_analytics_engine()
        engine2 = get_analytics_engine()
        
        # Verify singleton behavior
        assert engine1 is engine2
        assert isinstance(engine1, AdvancedAnalytics)
        
        # Test global engine functionality
        mock_results = [MockDetector().predict(sample_dataset)]
        report = await engine1.analyze_dataset(sample_dataset, mock_results)
        
        # Verify functionality
        assert isinstance(report, dict)
        assert "dataset_name" in report
        assert report["dataset_name"] == sample_dataset.name
    
    async def test_analytics_error_handling(self, sample_dataset):
        """Test analytics error handling."""
        engine = AnalyticsEngine()
        
        # Test with invalid dataset
        invalid_dataset = Dataset(
            name="invalid_dataset",
            data=pd.DataFrame(),  # Empty dataframe
            description="Invalid dataset for testing",
        )
        
        # Test time series analysis with invalid data
        analyzer = TimeSeriesAnalyzer()
        with pytest.raises(Exception):
            await analyzer.analyze_time_series(
                invalid_dataset.data,
                timestamp_col='nonexistent_column',
                value_col='nonexistent_value'
            )
        
        # Test pattern detection with empty results
        detector = PatternDetector()
        result = await detector.detect_patterns([], invalid_dataset)
        
        # Should handle empty results gracefully
        assert result.analysis_type == AnalysisType.PATTERN_DETECTION
        assert result.insights["total_anomalies"] == 0
    
    async def test_analytics_performance(self, sample_dataset, sample_detection_results):
        """Test analytics performance characteristics."""
        engine = AnalyticsEngine()
        
        # Test multiple concurrent analyses
        tasks = []
        for i in range(3):
            task = engine.run_comprehensive_analysis(
                sample_dataset,
                sample_detection_results,
                analysis_types=[AnalysisType.TIME_SERIES, AnalysisType.PATTERN_DETECTION]
            )
            tasks.append(task)
        
        # Run analyses concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Verify all analyses completed
        assert len(results_list) == 3
        for results in results_list:
            assert isinstance(results, dict)
            assert len(results) == 2  # TIME_SERIES and PATTERN_DETECTION
            
            # Check execution times are reasonable
            for result in results.values():
                assert result.execution_time_ms > 0
                assert result.execution_time_ms < 10000  # Should be under 10 seconds
    
    async def test_analytics_caching(self, sample_dataset, sample_detection_results):
        """Test analytics caching behavior."""
        explainer = AnomalyExplainer()
        
        # Get first anomaly
        anomaly = sample_detection_results[0].anomalies[0]
        detection_result = sample_detection_results[0]
        
        # First explanation call
        start_time = datetime.now()
        result1 = await explainer.explain_anomaly(anomaly, sample_dataset, detection_result)
        first_duration = (datetime.now() - start_time).total_seconds()
        
        # Second explanation call (should be cached)
        start_time = datetime.now()
        result2 = await explainer.explain_anomaly(anomaly, sample_dataset, detection_result)
        second_duration = (datetime.now() - start_time).total_seconds()
        
        # Verify results are identical
        assert result1.insights["anomaly_id"] == result2.insights["anomaly_id"]
        assert result1.analysis_type == result2.analysis_type
        
        # Second call should be faster (cached)
        assert second_duration <= first_duration
    
    async def test_analytics_data_types(self):
        """Test analytics with different data types."""
        # Create mixed data types dataset
        mixed_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'numeric_int': np.random.randint(0, 100, 100),
            'numeric_float': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'boolean': np.random.choice([True, False], 100),
            'text': [f"text_{i}" for i in range(100)],
        })
        
        dataset = Dataset(
            name="mixed_types_dataset",
            data=mixed_data,
            description="Dataset with mixed data types",
        )
        
        # Test time series analysis
        analyzer = TimeSeriesAnalyzer()
        result = await analyzer.analyze_time_series(
            mixed_data,
            timestamp_col='timestamp',
            value_col='numeric_float'
        )
        
        # Verify analysis handles mixed types
        assert result.analysis_type == AnalysisType.TIME_SERIES
        assert isinstance(result.insights, dict)
        assert result.insights["data_points"] == 100
        
        # Test pattern detection
        detector = PatternDetector()
        mock_results = [MockDetector().predict(dataset)]
        result = await detector.detect_patterns(mock_results, dataset)
        
        # Verify pattern detection handles mixed types
        assert result.analysis_type == AnalysisType.PATTERN_DETECTION
        assert isinstance(result.insights, dict)
        assert result.insights["total_anomalies"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])