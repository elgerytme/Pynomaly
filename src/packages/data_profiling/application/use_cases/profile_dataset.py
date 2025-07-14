from datetime import datetime
from ...infrastructure.adapters.file_adapter import get_file_adapter
from ..services.schema_analysis_service import SchemaAnalysisService
from ..services.statistical_profiling_service import StatisticalProfilingService
from ..services.pattern_discovery_service import PatternDiscoveryService
from ...domain.entities.profiles import DataProfile

class ProfileDatasetUseCase:
    """Use case to orchestrate data profiling for various data sources."""
    def __init__(self) -> None:
        self.schema_service = SchemaAnalysisService()
        self.stats_service = StatisticalProfilingService()
        self.pattern_service = PatternDiscoveryService()

    def execute(self, path: str) -> DataProfile:
        """Profile the dataset at the given path and return a DataProfile."""
        adapter = get_file_adapter(path)
        df = adapter.load(path)
        schema = self.schema_service.infer(df)
        stats = self.stats_service.analyze(df)
        patterns = self.pattern_service.discover(df)
        # Attach patterns to column profiles
        for col_profile in schema.columns:
            if col_profile.column_name in patterns:
                # bypass frozen dataclass to set patterns
                object.__setattr__(col_profile, 'patterns', patterns[col_profile.column_name])
        now = datetime.utcnow()
        profile = DataProfile(
            schema_profile=schema,
            statistical_profile=stats,
            content_profile=None,
            quality_assessment=None,
            profiling_metadata={'path': path},
            created_at=now,
            last_updated=now,
            version='1.0'
        )
        return profile