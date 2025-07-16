"""
Comprehensive test suite for executive reporting components.
Tests executive scorecard, business impact analysis, strategic analytics, and performance optimization.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from src.packages.data_quality.domain.entities.executive_scorecard import (
    ExecutiveScorecard, QualityKPI, BusinessUnit, QualityMaturityStage,
    TrendDirection, FinancialImpact, QualityROIAnalysis, IndustryBenchmark,
    BusinessImpactSeverity, QualityMetricSnapshot
)
from src.packages.data_quality.domain.entities.data_quality_assessment import QualityAssessment
from src.packages.data_quality.application.services.business_impact_analysis_service import (
    BusinessImpactAnalysisService, ImpactCalculationConfig, BusinessMetrics
)
from src.packages.data_quality.application.services.strategic_quality_analytics_service import (
    StrategicQualityAnalyticsService, PredictionConfig, QualityPrediction,
    InvestmentOptimization, CompetitiveAnalysis
)
from src.packages.data_quality.application.services.executive_reporting_service import (
    ExecutiveReportingService, ReportingConfig, ExecutiveReport
)
from src.packages.data_quality.infrastructure.performance.dashboard_performance_optimizer import (
    DashboardPerformanceOptimizer, CacheConfig, PerformanceConfig
)


class TestExecutiveScorecard:
    """Test executive scorecard domain entity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.business_unit = BusinessUnit(
            unit_id="bu_001",
            unit_name="Engineering",
            division="Technology",
            region="North America",
            head_executive="John Doe",
            data_steward="Jane Smith",
            quality_budget=100000.0,
            employee_count=250,
            revenue_impact=0.3
        )
        
        self.quality_kpi = QualityKPI(
            kpi_id="overall_quality",
            name="Overall Data Quality",
            description="Comprehensive quality score",
            calculation_method="Weighted average",
            target_value=0.95,
            current_value=0.87,
            trend_direction=TrendDirection.IMPROVING,
            business_units=[self.business_unit]
        )
        
        self.financial_impact = FinancialImpact(
            severity=BusinessImpactSeverity.HIGH,
            revenue_impact=50000.0,
            cost_impact=25000.0,
            productivity_impact=15000.0,
            compliance_cost=10000.0,
            opportunity_cost=20000.0,
            affected_business_units=["Engineering", "Sales"]
        )
    
    def test_scorecard_creation(self):
        """Test executive scorecard creation."""
        scorecard = ExecutiveScorecard(
            organization_name="Test Corp",
            reporting_period="Q1 2024",
            overall_quality_score=0.87,
            quality_kpis=[self.quality_kpi],
            current_maturity_stage=QualityMaturityStage.MANAGED,
            target_maturity_stage=QualityMaturityStage.DEFINED,
            financial_impacts=[self.financial_impact]
        )
        
        assert scorecard.organization_name == "Test Corp"
        assert scorecard.overall_quality_score == 0.87
        assert len(scorecard.quality_kpis) == 1
        assert scorecard.current_maturity_stage == QualityMaturityStage.MANAGED
        assert len(scorecard.financial_impacts) == 1
    
    def test_executive_summary_generation(self):
        """Test executive summary generation."""
        scorecard = ExecutiveScorecard(
            organization_name="Test Corp",
            overall_quality_score=0.87,
            current_maturity_stage=QualityMaturityStage.MANAGED,
            financial_impacts=[self.financial_impact],
            top_quality_risks=["Data completeness issues"],
            improvement_opportunities=["Automated validation"]
        )
        
        summary = scorecard.get_executive_summary()
        
        assert "87.0%" in summary["overall_quality_score"]
        assert summary["maturity_stage"] == "Managed"
        assert summary["total_financial_impact"] == "$120,000"
        assert summary["critical_risks"] == 0
        assert summary["top_risk"] == "Data completeness issues"
        assert summary["key_opportunity"] == "Automated validation"
    
    def test_quality_health_index_calculation(self):
        """Test quality health index calculation."""
        scorecard = ExecutiveScorecard(
            quality_kpis=[self.quality_kpi],
            current_maturity_stage=QualityMaturityStage.DEFINED,
            financial_impacts=[self.financial_impact]
        )
        
        health_index = scorecard.calculate_quality_health_index()
        
        assert 0 <= health_index <= 100
        assert isinstance(health_index, float)
    
    def test_kpi_achievement_percentage(self):
        """Test KPI achievement percentage calculation."""
        assert self.quality_kpi.achievement_percentage == pytest.approx(91.58, rel=1e-2)
        assert self.quality_kpi.status_indicator == "🟢"
    
    def test_financial_impact_calculations(self):
        """Test financial impact calculations."""
        assert self.financial_impact.total_financial_impact == 120000.0
        
        # Test annualized impact with recurring frequency
        recurring_impact = FinancialImpact(
            revenue_impact=10000.0,
            cost_impact=5000.0,
            recurring_frequency="monthly"
        )
        
        assert recurring_impact.annualized_impact == 180000.0  # 15k * 12 months


class TestBusinessImpactAnalysisService:
    """Test business impact analysis service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ImpactCalculationConfig()
        self.business_metrics = BusinessMetrics()
        self.service = BusinessImpactAnalysisService(self.config, self.business_metrics)
        
        self.quality_assessment = QualityAssessment(
            assessment_id="qa_001",
            overall_score=0.75,
            completeness_score=0.80,
            accuracy_score=0.70,
            consistency_score=0.75,
            validity_score=0.78,
            uniqueness_score=0.85,
            timeliness_score=0.72
        )
    
    @pytest.mark.asyncio
    async def test_quality_issue_impact_analysis(self):
        """Test quality issue impact analysis."""
        impact = await self.service.analyze_quality_issue_impact(
            self.quality_assessment,
            affected_data_volume=1000.0,
            business_context={"business_unit": "Engineering"}
        )
        
        assert isinstance(impact, FinancialImpact)
        assert impact.severity == BusinessImpactSeverity.MEDIUM  # Based on 0.75 overall score
        assert impact.total_financial_impact > 0
        assert "Engineering" in impact.affected_business_units
        assert len(impact.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_roi_calculation(self):
        """Test ROI calculation for quality initiatives."""
        implementation_plan = {
            "base_implementation_cost": 100000.0,
            "consulting_days": 30,
            "user_count": 50,
            "employees_to_train": 100
        }
        
        expected_improvements = {
            "overall_quality_improvement": 0.15,
            "accuracy_improvement": 0.20,
            "compliance_improvement": 0.10
        }
        
        roi_analysis = await self.service.calculate_quality_initiative_roi(
            "Automated Quality Monitoring",
            implementation_plan,
            expected_improvements,
            timeline_months=12
        )
        
        assert isinstance(roi_analysis, QualityROIAnalysis)
        assert roi_analysis.total_investment > 0
        assert roi_analysis.total_returns > 0
        assert roi_analysis.roi_percentage != 0
        assert roi_analysis.payback_period_months is not None
    
    @pytest.mark.asyncio
    async def test_business_unit_performance_analysis(self):
        """Test business unit performance analysis."""
        business_units = [
            BusinessUnit(unit_id="bu_001", unit_name="Engineering", division="Tech"),
            BusinessUnit(unit_id="bu_002", unit_name="Sales", division="Business")
        ]
        
        quality_assessments = {
            "bu_001": self.quality_assessment,
            "bu_002": QualityAssessment(
                assessment_id="qa_002",
                overall_score=0.85
            )
        }
        
        results = await self.service.analyze_business_unit_performance(
            business_units,
            quality_assessments
        )
        
        assert len(results) == 2
        assert "bu_001" in results
        assert "bu_002" in results
        
        bu_001_result = results["bu_001"]
        assert bu_001_result["unit_name"] == "Engineering"
        assert bu_001_result["quality_score"] == 0.75
        assert bu_001_result["financial_impact"] > 0
        assert isinstance(bu_001_result["performance_score"], float)
    
    def test_impact_severity_determination(self):
        """Test impact severity determination."""
        # Test critical severity
        critical_assessment = QualityAssessment(overall_score=0.4)
        severity = self.service._determine_impact_severity(critical_assessment)
        assert severity == BusinessImpactSeverity.CRITICAL
        
        # Test high severity
        high_assessment = QualityAssessment(overall_score=0.6)
        severity = self.service._determine_impact_severity(high_assessment)
        assert severity == BusinessImpactSeverity.HIGH
        
        # Test negligible severity
        good_assessment = QualityAssessment(overall_score=0.98)
        severity = self.service._determine_impact_severity(good_assessment)
        assert severity == BusinessImpactSeverity.NEGLIGIBLE


class TestStrategicQualityAnalyticsService:
    """Test strategic quality analytics service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PredictionConfig()
        self.service = StrategicQualityAnalyticsService(self.config)
        
        # Create historical assessment data
        self.historical_assessments = []
        for i in range(30):
            assessment = QualityAssessment(
                assessment_id=f"qa_{i:03d}",
                assessment_timestamp=datetime.now() - timedelta(days=i),
                overall_score=0.8 + (np.random.random() - 0.5) * 0.2,
                completeness_score=0.85 + (np.random.random() - 0.5) * 0.15,
                accuracy_score=0.82 + (np.random.random() - 0.5) * 0.18
            )
            self.historical_assessments.append(assessment)
    
    @pytest.mark.asyncio
    async def test_quality_metrics_prediction(self):
        """Test quality metrics prediction."""
        metric_names = ["overall_score", "completeness_score", "accuracy_score"]
        
        predictions = await self.service.predict_quality_metrics(
            metric_names,
            self.historical_assessments,
            business_context={"business_events": []},
            prediction_horizon_days=30
        )
        
        assert len(predictions) == 3
        
        for metric_name in metric_names:
            assert metric_name in predictions
            prediction = predictions[metric_name]
            
            assert isinstance(prediction, QualityPrediction)
            assert prediction.metric_name == metric_name
            assert prediction.prediction_horizon_days == 30
            assert 0 <= prediction.predicted_value <= 1
            assert prediction.confidence_interval_lower <= prediction.predicted_value
            assert prediction.predicted_value <= prediction.confidence_interval_upper
            assert prediction.risk_level in ["low", "medium", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_investment_optimization(self):
        """Test quality investment optimization."""
        current_scores = {
            "completeness": 0.75,
            "accuracy": 0.70,
            "consistency": 0.80
        }
        
        improvement_targets = {
            "completeness": 0.90,
            "accuracy": 0.85,
            "consistency": 0.88
        }
        
        optimization = await self.service.optimize_quality_investment(
            current_scores,
            available_budget=500000.0,
            improvement_targets=improvement_targets,
            business_constraints={"cost_multiplier": 1.2}
        )
        
        assert isinstance(optimization, InvestmentOptimization)
        assert optimization.recommended_budget == 500000.0
        assert optimization.expected_roi > 0
        assert len(optimization.area_allocations) > 0
        assert len(optimization.priority_initiatives) > 0
        assert optimization.quality_improvement_projection > 0
    
    @pytest.mark.asyncio
    async def test_competitive_analysis(self):
        """Test competitive position analysis."""
        org_metrics = {
            "overall_quality_score": 0.82,
            "completeness_score": 0.85,
            "accuracy_score": 0.80
        }
        
        benchmarks = [
            IndustryBenchmark(
                industry_sector="technology",
                median_quality_score=0.78,
                top_quartile_score=0.89,
                bottom_quartile_score=0.65
            )
        ]
        
        analysis = await self.service.analyze_competitive_position(
            org_metrics,
            benchmarks,
            market_context={"competitive_pressure": "high"}
        )
        
        assert isinstance(analysis, CompetitiveAnalysis)
        assert analysis.organization_position in ["leader", "challenger", "follower", "niche"]
        assert 0 <= analysis.market_position_score <= 1
        assert analysis.quality_maturity_gap >= 0
        assert len(analysis.competitive_advantages) >= 0
        assert len(analysis.improvement_opportunities) >= 0
    
    @pytest.mark.asyncio
    async def test_strategic_recommendations_generation(self):
        """Test strategic recommendations generation."""
        # Create mock predictions
        predictions = {
            "overall_score": QualityPrediction(
                metric_name="overall_score",
                prediction_timestamp=datetime.now(),
                prediction_horizon_days=30,
                predicted_value=0.75,
                confidence_interval_lower=0.70,
                confidence_interval_upper=0.80,
                prediction_confidence=0.85,
                model_accuracy=0.75,
                historical_variance=0.05,
                trend_strength=0.1,
                risk_level="medium",
                recommended_actions=["Monitor trends closely"]
            )
        }
        
        investment_optimization = InvestmentOptimization(
            optimization_id="opt_001",
            investment_scenario="test",
            recommended_budget=100000.0,
            expected_roi=150.0,
            confidence_score=0.8
        )
        
        competitive_analysis = CompetitiveAnalysis(
            analysis_id="comp_001",
            comparison_date=datetime.now(),
            organization_position="challenger",
            market_position_score=0.7,
            quality_maturity_gap=0.1,
            investment_gap_percentage=5.0
        )
        
        recommendations = await self.service.generate_strategic_recommendations(
            predictions,
            investment_optimization,
            competitive_analysis,
            business_objectives={"growth_focus": True}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_feature_generation(self):
        """Test feature generation for predictions."""
        # Create test time series
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        values = np.random.random(30) * 0.2 + 0.8
        
        metric_series = pd.Series(values, index=dates, name="test_metric")
        
        # Test temporal features
        df = pd.DataFrame({"value": metric_series})
        temporal_features = self.service._generate_temporal_features(df, {})
        
        assert "day_of_week" in temporal_features.columns
        assert "month" in temporal_features.columns
        assert "is_weekend" in temporal_features.columns
        
        # Test trend features
        trend_features = self.service._generate_trend_features(df, {})
        
        assert "ma_7" in trend_features.columns
        assert "trend_7" in trend_features.columns
        assert "roc_1" in trend_features.columns


class TestExecutiveReportingService:
    """Test executive reporting service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ReportingConfig()
        self.business_impact_service = Mock(spec=BusinessImpactAnalysisService)
        self.analytics_service = Mock(spec=StrategicQualityAnalyticsService)
        
        self.service = ExecutiveReportingService(
            self.config,
            self.business_impact_service,
            self.analytics_service
        )
        
        # Mock business units
        self.business_units = [
            BusinessUnit(unit_id="bu_001", unit_name="Engineering", division="Tech"),
            BusinessUnit(unit_id="bu_002", unit_name="Sales", division="Business")
        ]
    
    @pytest.mark.asyncio
    async def test_executive_scorecard_generation(self):
        """Test executive scorecard generation."""
        with patch.object(self.service, '_get_aggregated_data') as mock_aggregated:
            mock_aggregated.return_value = {
                "quality_assessments": [
                    {
                        "timestamp": datetime.now(),
                        "overall_score": 0.85,
                        "completeness_score": 0.88,
                        "accuracy_score": 0.82,
                        "business_unit": "Engineering"
                    }
                ],
                "quality_issues": [],
                "business_metrics": {"annual_revenue": 100_000_000},
                "historical_trends": {"overall_score": [0.8] * 30},
                "benchmark_data": {"industry_median": 0.78}
            }
            
            scorecard = await self.service.generate_executive_scorecard(
                "Test Corp",
                reporting_period="Q1 2024",
                business_units=self.business_units
            )
            
            assert isinstance(scorecard, ExecutiveScorecard)
            assert scorecard.organization_name == "Test Corp"
            assert scorecard.reporting_period == "Q1 2024"
            assert scorecard.overall_quality_score > 0
            assert len(scorecard.quality_kpis) > 0
            assert scorecard.current_maturity_stage is not None
    
    @pytest.mark.asyncio
    async def test_executive_report_generation(self):
        """Test executive report generation."""
        with patch.object(self.service, 'generate_executive_scorecard') as mock_scorecard:
            mock_scorecard.return_value = ExecutiveScorecard(
                organization_name="Test Corp",
                reporting_period="Q1 2024",
                overall_quality_score=0.85
            )
            
            report = await self.service.generate_executive_report(
                "summary",
                "Test Corp",
                {"reporting_period": "Q1 2024"}
            )
            
            assert isinstance(report, ExecutiveReport)
            assert report.report_type == "summary"
            assert report.executive_scorecard is not None
            assert len(report.key_insights) > 0
            assert len(report.recommendations) > 0
            assert report.generation_metrics is not None
    
    @pytest.mark.asyncio
    async def test_real_time_dashboard_data(self):
        """Test real-time dashboard data generation."""
        with patch.object(self.service, '_get_aggregated_data') as mock_aggregated:
            mock_aggregated.return_value = {
                "quality_assessments": [
                    {"overall_score": 0.85, "timestamp": datetime.now()}
                ],
                "quality_issues": [
                    {"severity": "critical", "category": "completeness"},
                    {"severity": "high", "category": "accuracy"}
                ],
                "historical_trends": {
                    "overall_score": [0.8, 0.82, 0.85],
                    "completeness": [0.85, 0.87, 0.88]
                }
            }
            
            dashboard_data = await self.service.get_real_time_dashboard_data({
                "quality_score_widget": True,
                "issues_widget": True,
                "trends_widget": True
            })
            
            assert "core_metrics" in dashboard_data
            assert "widgets" in dashboard_data
            assert "metadata" in dashboard_data
            
            core_metrics = dashboard_data["core_metrics"]
            assert "overall_quality_score" in core_metrics
            assert "total_issues" in core_metrics
            assert "critical_issues" in core_metrics
            
            widgets = dashboard_data["widgets"]
            assert "quality_score" in widgets
            assert "issues_summary" in widgets
            assert "performance_trends" in widgets
            
            # Check performance target
            load_time = dashboard_data["metadata"]["load_time_ms"]
            assert load_time < 2000  # Target: sub-2s loading
    
    @pytest.mark.asyncio
    async def test_subscription_management(self):
        """Test real-time update subscriptions."""
        callback_called = False
        received_data = None
        
        def test_callback(data):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = data
        
        # Subscribe to updates
        await self.service.subscribe_to_updates(
            "test_subscriber",
            test_callback,
            ["quality_metrics"]
        )
        
        # Verify subscription
        assert "quality_metrics" in self.service.subscribers
        assert len(self.service.subscribers["quality_metrics"]) == 1
        
        # Test notification
        test_data = {"test": "data"}
        await self.service._notify_subscribers("quality_metrics", test_data)
        
        assert callback_called
        assert received_data == test_data
        
        # Unsubscribe
        await self.service.unsubscribe_from_updates("test_subscriber")
        assert len(self.service.subscribers["quality_metrics"]) == 0
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache_key = self.service._generate_cache_key(
            "test_query",
            {"param1": "value1", "param2": 123}
        )
        
        assert cache_key.startswith("dashboard:test_query:")
        assert len(cache_key.split(":")) == 3
        
        # Test deterministic generation
        cache_key2 = self.service._generate_cache_key(
            "test_query",
            {"param2": 123, "param1": "value1"}  # Different order
        )
        
        assert cache_key == cache_key2  # Should be the same
    
    def test_quality_trend_calculation(self):
        """Test quality trend calculation."""
        data = {
            "historical_trends": {
                "overall_score": [0.8] * 10 + [0.85] * 10,  # Improving trend
                "completeness_score": [0.9] * 20,           # Stable trend
                "accuracy_score": [0.85] * 10 + [0.80] * 10  # Declining trend
            }
        }
        
        trends = self.service._calculate_quality_trends(data)
        
        assert trends["overall_score"] == TrendDirection.IMPROVING
        assert trends["completeness_score"] == TrendDirection.STABLE
        assert trends["accuracy_score"] == TrendDirection.DECLINING


class TestDashboardPerformanceOptimizer:
    """Test dashboard performance optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache_config = CacheConfig(enable_redis_cache=False)  # Disable Redis for tests
        self.performance_config = PerformanceConfig()
        self.optimizer = DashboardPerformanceOptimizer(self.cache_config, self.performance_config)
    
    @pytest.mark.asyncio
    async def test_query_optimization_with_cache(self):
        """Test query optimization with caching."""
        # Mock query function
        async def mock_query(**params):
            await asyncio.sleep(0.1)  # Simulate query time
            return {"result": "test_data", "params": params}
        
        # First call (cache miss)
        result1 = await self.optimizer.optimize_dashboard_query(
            "test_query",
            mock_query,
            {"param1": "value1"},
            cache_ttl=300
        )
        
        assert result1["cache_hit"] is False
        assert "data" in result1
        assert result1["load_time_ms"] > 0
        
        # Second call (cache hit)
        result2 = await self.optimizer.optimize_dashboard_query(
            "test_query",
            mock_query,
            {"param1": "value1"},
            cache_ttl=300
        )
        
        assert result2["cache_hit"] is True
        assert result2["data"] == result1["data"]
        assert result2["load_time_ms"] < result1["load_time_ms"]  # Should be faster
    
    @pytest.mark.asyncio
    async def test_data_optimization(self):
        """Test data optimization for DataFrames."""
        # Create test DataFrame
        large_df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.random(1000),
            'category': ['A', 'B', 'C'] * 334
        })
        
        optimized_data = await self.optimizer._optimize_result_data(large_df)
        
        assert isinstance(optimized_data, list)  # Should be converted to records
        assert len(optimized_data) <= 1000
    
    @pytest.mark.asyncio
    async def test_response_optimization(self):
        """Test response optimization with compression."""
        large_data = {"data": "x" * 5000}  # Large data to trigger compression
        
        optimized_response = await self.optimizer.optimize_data_response(
            large_data,
            response_format="json",
            enable_compression=True
        )
        
        assert optimized_response["compressed"] is True
        assert optimized_response["compression_ratio"] > 1.0
        assert optimized_response["final_size"] < optimized_response["original_size"]
    
    def test_cache_statistics(self):
        """Test cache statistics retrieval."""
        stats = self.optimizer.get_cache_statistics()
        
        assert "hit_rate" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "memory_cache_size" in stats
        assert "redis_available" in stats
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        metrics = self.optimizer.get_performance_metrics()
        
        assert hasattr(metrics, 'total_load_time_ms')
        assert hasattr(metrics, 'cache_hit_rate')
        assert hasattr(metrics, 'data_reduction_ratio')
        assert hasattr(metrics, 'concurrent_requests')
    
    def test_lru_cache_functionality(self):
        """Test LRU cache implementation."""
        from src.packages.data_quality.infrastructure.performance.dashboard_performance_optimizer import LRUCache
        
        cache = LRUCache(capacity=3)
        
        # Add items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
        # Add fourth item (should evict least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key4") == "value4"
    
    def test_data_compressor(self):
        """Test data compression utilities."""
        from src.packages.data_quality.infrastructure.performance.dashboard_performance_optimizer import DataCompressor
        
        test_data = "This is a test string that should be compressed" * 100
        
        # Test compression
        assert DataCompressor.should_compress(test_data, threshold=100)
        
        compressed = DataCompressor.compress(test_data)
        assert len(compressed) < len(test_data.encode('utf-8'))
        
        # Test decompression
        decompressed = DataCompressor.decompress(compressed)
        assert decompressed == test_data
    
    def test_query_optimizer(self):
        """Test query optimization utilities."""
        from src.packages.data_quality.infrastructure.performance.dashboard_performance_optimizer import QueryOptimizer
        
        # Create test DataFrame
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 15, 30, 25],
            'region': ['North', 'South', 'North', 'East', 'South']
        })
        
        # Test filtering
        filtered = QueryOptimizer.optimize_dataframe_query(df, {'category': 'A'})
        assert len(filtered) == 2
        assert all(filtered['category'] == 'A')
        
        # Test range filtering
        range_filtered = QueryOptimizer.optimize_dataframe_query(
            df, {'value': {'min': 15, 'max': 25}}
        )
        assert len(range_filtered) == 3
        assert all(range_filtered['value'] >= 15)
        assert all(range_filtered['value'] <= 25)
        
        # Test aggregation
        aggregated = QueryOptimizer.optimize_aggregation(
            df, ['category'], {'value': 'mean'}
        )
        assert len(aggregated) == 3  # A, B, C categories
        assert 'value' in aggregated.columns


class TestIntegrationScenarios:
    """Test integration scenarios across all components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.business_impact_service = BusinessImpactAnalysisService()
        self.analytics_service = StrategicQualityAnalyticsService()
        self.reporting_service = ExecutiveReportingService(
            business_impact_service=self.business_impact_service,
            analytics_service=self.analytics_service
        )
        self.optimizer = DashboardPerformanceOptimizer()
    
    @pytest.mark.asyncio
    async def test_end_to_end_reporting_workflow(self):
        """Test complete end-to-end reporting workflow."""
        # 1. Generate historical data
        historical_assessments = []
        for i in range(30):
            assessment = QualityAssessment(
                assessment_id=f"qa_{i:03d}",
                assessment_timestamp=datetime.now() - timedelta(days=i),
                overall_score=0.8 + (np.random.random() - 0.5) * 0.2
            )
            historical_assessments.append(assessment)
        
        # 2. Analyze business impact
        impact = await self.business_impact_service.analyze_quality_issue_impact(
            historical_assessments[0],  # Latest assessment
            affected_data_volume=1000.0
        )
        assert isinstance(impact, FinancialImpact)
        
        # 3. Generate predictions
        predictions = await self.analytics_service.predict_quality_metrics(
            ["overall_score"],
            historical_assessments
        )
        assert "overall_score" in predictions
        
        # 4. Generate executive report
        with patch.object(self.reporting_service, '_get_aggregated_data') as mock_data:
            mock_data.return_value = {
                "quality_assessments": [asdict(a) for a in historical_assessments[:5]],
                "quality_issues": [],
                "business_metrics": {},
                "historical_trends": {"overall_score": [0.8] * 30},
                "benchmark_data": {}
            }
            
            report = await self.reporting_service.generate_executive_report(
                "comprehensive",
                "Test Corp"
            )
            
            assert isinstance(report, ExecutiveReport)
            assert report.executive_scorecard is not None
            assert len(report.key_insights) > 0
        
        # 5. Optimize dashboard performance
        async def mock_dashboard_query():
            return {"scorecard": asdict(report.executive_scorecard)}
        
        optimized_result = await self.optimizer.optimize_dashboard_query(
            "comprehensive_dashboard",
            mock_dashboard_query
        )
        
        assert "data" in optimized_result
        assert optimized_result["load_time_ms"] < 5000  # Should be performant
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        async def simulate_concurrent_requests():
            tasks = []
            
            for i in range(10):  # Simulate 10 concurrent requests
                async def mock_query():
                    await asyncio.sleep(0.1)
                    return {"result": f"data_{i}"}
                
                task = self.optimizer.optimize_dashboard_query(
                    f"query_{i}",
                    mock_query
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        start_time = time.time()
        results = await simulate_concurrent_requests()
        total_time = time.time() - start_time
        
        assert len(results) == 10
        assert all("data" in result for result in results)
        assert total_time < 5.0  # Should handle concurrent load efficiently
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': range(10000),
            'col2': np.random.random(10000),
            'col3': ['category_' + str(i % 100) for i in range(10000)]
        })
        
        # Test memory optimization
        initial_memory = large_data.memory_usage(deep=True).sum()
        
        # Simulate optimization
        optimized_data = large_data.sample(n=5000)  # Sample for performance
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        assert optimized_memory < initial_memory
        reduction_ratio = initial_memory / optimized_memory
        assert reduction_ratio > 1.5  # At least 50% reduction


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])