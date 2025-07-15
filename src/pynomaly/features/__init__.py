"""Advanced features and capabilities for Pynomaly."""

from .advanced_analytics import (
    AdvancedAnalytics,
    AnalyticsEngine,
    AnomalyExplainer,
    PatternDetector,
    TimeSeriesAnalyzer,
    TrendAnalyzer,
    get_analytics_engine,
)
from .api_gateway import (
    APIGateway,
    APIVersioning,
    EndpointManager,
    RequestProcessor,
    ResponseProcessor,
    get_api_gateway,
)
from .feature_engineering import (
    FeatureEngineer,
    FeatureExtractor,
    FeaturePipeline,
    FeatureSelector,
    FeatureTransformer,
    get_feature_engineer,
)
from .model_management import (
    AutoMLPipeline,
    ModelDeployment,
    ModelManager,
    ModelMonitoring,
    ModelRegistry,
    ModelVersioning,
    get_model_manager,
)
from .real_time_processing import (
    EventProcessor,
    RealTimeDetector,
    StreamingConfig,
    StreamingPipeline,
    StreamProcessor,
    get_stream_processor,
)

__all__ = [
    # Advanced analytics
    "AdvancedAnalytics",
    "AnalyticsEngine",
    "TimeSeriesAnalyzer",
    "PatternDetector",
    "TrendAnalyzer",
    "AnomalyExplainer",
    "get_analytics_engine",
    # Model management
    "ModelManager",
    "ModelRegistry",
    "ModelVersioning",
    "ModelDeployment",
    "ModelMonitoring",
    "AutoMLPipeline",
    "get_model_manager",
    # Real-time processing
    "StreamProcessor",
    "RealTimeDetector",
    "StreamingPipeline",
    "EventProcessor",
    "StreamingConfig",
    "get_stream_processor",
    # API gateway
    "APIGateway",
    "EndpointManager",
    "APIVersioning",
    "RequestProcessor",
    "ResponseProcessor",
    "get_api_gateway",
    # Feature engineering
    "FeatureEngineer",
    "FeatureExtractor",
    "FeatureSelector",
    "FeatureTransformer",
    "FeaturePipeline",
    "get_feature_engineer",
]
