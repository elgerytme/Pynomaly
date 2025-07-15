"""Execute Data Profiling Use Case."""

from typing import Dict, Any, Optional, List
import pandas as pd
import structlog

from ...domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus, 
    SchemaProfile, QualityAssessment, ProfilingMetadata
)
from ...domain.repositories.data_profile_repository import DataProfileRepository

logger = structlog.get_logger(__name__)


class ExecuteDataProfilingUseCase:
    """Use case for executing data profiling."""
    
    def __init__(
        self,
        repository: DataProfileRepository,
        profiling_service: Optional[Any] = None
    ):
        self.repository = repository
        self.profiling_service = profiling_service
    
    async def execute(
        self,
        dataset_id: DatasetId,
        data: pd.DataFrame,
        source_type: str = "dataframe",
        source_connection: Optional[Dict[str, Any]] = None,
        profiling_config: Optional[Dict[str, Any]] = None
    ) -> DataProfile:
        """Execute data profiling use case."""
        
        # Create new profile entity
        profile = DataProfile(
            profile_id=ProfileId(),
            dataset_id=dataset_id,
            source_type=source_type,
            source_connection=source_connection or {}
        )
        
        logger.info(
            "Executing data profiling",
            profile_id=str(profile.profile_id.value),
            dataset_id=str(dataset_id.value),
            source_type=source_type
        )
        
        try:
            # Start the profiling
            profile.start_profiling()
            await self.repository.save(profile)
            
            # Use the profiling service if available
            if self.profiling_service:
                completed_profile = await self.profiling_service.execute_profiling(
                    profile, data, profiling_config or {}
                )
                await self.repository.save(completed_profile)
                return completed_profile
            else:
                # Basic fallback implementation
                schema_profile = self._create_basic_schema_profile(data)
                quality_assessment = self._create_basic_quality_assessment(data)
                metadata = self._create_basic_metadata(data)
                
                profile.complete_profiling(
                    schema_profile=schema_profile,
                    quality_assessment=quality_assessment,
                    metadata=metadata
                )
                await self.repository.save(profile)
                return profile
                
        except Exception as e:
            logger.error(
                "Data profiling failed",
                profile_id=str(profile.profile_id.value),
                error=str(e)
            )
            profile.fail_profiling(f"Profiling execution failed: {str(e)}")
            await self.repository.save(profile)
            raise
    
    def _create_basic_schema_profile(self, data: pd.DataFrame) -> SchemaProfile:
        """Create a basic schema profile from DataFrame."""
        from ...domain.entities.data_profile import (
            ColumnProfile, DataType, CardinalityLevel, ValueDistribution
        )
        
        columns = []
        for col_name in data.columns:
            col_data = data[col_name]
            
            # Infer data type
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = DataType.FLOAT if col_data.dtype.kind == 'f' else DataType.INTEGER
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = DataType.DATETIME
            elif pd.api.types.is_bool_dtype(col_data):
                data_type = DataType.BOOLEAN
            else:
                data_type = DataType.STRING
            
            # Calculate distribution
            unique_count = col_data.nunique()
            null_count = col_data.isnull().sum()
            total_count = len(col_data)
            completeness_ratio = (total_count - null_count) / total_count if total_count > 0 else 0
            
            distribution = ValueDistribution(
                unique_count=unique_count,
                null_count=null_count,
                total_count=total_count,
                completeness_ratio=completeness_ratio
            )
            
            # Determine cardinality
            if unique_count < 10:
                cardinality = CardinalityLevel.LOW
            elif unique_count < 100:
                cardinality = CardinalityLevel.MEDIUM
            elif unique_count < 1000:
                cardinality = CardinalityLevel.HIGH
            else:
                cardinality = CardinalityLevel.VERY_HIGH
            
            column_profile = ColumnProfile(
                column_name=col_name,
                data_type=data_type,
                nullable=col_data.isnull().any(),
                distribution=distribution,
                cardinality=cardinality,
                quality_score=completeness_ratio * 100
            )
            columns.append(column_profile)
        
        return SchemaProfile(
            table_name="dataset",
            total_columns=len(data.columns),
            total_rows=len(data),
            columns=columns
        )
    
    def _create_basic_quality_assessment(self, data: pd.DataFrame) -> QualityAssessment:
        """Create a basic quality assessment from DataFrame."""
        # Calculate basic quality scores
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # Simple quality scoring
        overall_score = completeness_score * 0.9  # Basic score based on completeness
        
        return QualityAssessment(
            overall_score=overall_score,
            completeness_score=completeness_score,
            consistency_score=0.8,  # Placeholder
            accuracy_score=0.8,     # Placeholder
            validity_score=0.8,     # Placeholder
            uniqueness_score=0.8,   # Placeholder
            recommendations=["Complete data profiling for detailed quality assessment"]
        )
    
    def _create_basic_metadata(self, data: pd.DataFrame) -> ProfilingMetadata:
        """Create basic profiling metadata."""
        return ProfilingMetadata(
            profiling_strategy="basic",
            execution_time_seconds=1.0,
            include_patterns=False,
            include_statistical_analysis=False,
            include_quality_assessment=True
        )
    
    async def get_profile_by_id(
        self, 
        profile_id: ProfileId
    ) -> Optional[DataProfile]:
        """Get profile by ID."""
        return await self.repository.get_by_id(profile_id)
    
    async def get_latest_profile_by_dataset(
        self, 
        dataset_id: DatasetId
    ) -> Optional[DataProfile]:
        """Get the latest profile for a dataset."""
        return await self.repository.get_latest_by_dataset_id(dataset_id)
    
    async def get_profiles_by_dataset(
        self, 
        dataset_id: DatasetId
    ) -> List[DataProfile]:
        """Get all profiles for a dataset."""
        return await self.repository.get_by_dataset_id(dataset_id)