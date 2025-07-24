from uuid import UUID
from typing import Dict, Any
import pandas as pd

from ..ports.data_profile_repository import DataProfileRepository
from ...domain.entities.data_profile import DataProfile, ColumnProfile, ProfileStatistics, DataType
from ...infrastructure.adapters.data_source_adapter import DataSourceAdapter


class DataProfilingService:
    """Service for data profiling."""

    def __init__(self, data_profile_repository: DataProfileRepository):
        self.data_profile_repository = data_profile_repository

    def create_profile(self, dataset_name: str, data_source_adapter: DataSourceAdapter, source_config: Dict[str, Any]) -> DataProfile:
        """Create a new data profile.

        Args:
            dataset_name: The name of the dataset to profile.
            data_source_adapter: An instance of a DataSourceAdapter to read the data.
            source_config: Configuration for the data source adapter.

        Returns:
            The created DataProfile entity.
        """
        data_profile = DataProfile(dataset_name=dataset_name)
        data_profile.start_profiling()

        try:
            df = data_source_adapter.read_data(source_config)
            data_profile.total_rows = len(df)
            data_profile.total_columns = len(df.columns)

            column_profiles = []
            for col_name in df.columns:
                column_data = df[col_name]
                stats = ProfileStatistics(
                    total_count=len(column_data),
                    null_count=column_data.isnull().sum(),
                    distinct_count=column_data.nunique()
                )
                # Basic data type inference
                inferred_type = DataType.UNKNOWN
                if pd.api.types.is_integer_dtype(column_data):
                    inferred_type = DataType.INTEGER
                elif pd.api.types.is_float_dtype(column_data):
                    inferred_type = DataType.FLOAT
                elif pd.api.types.is_bool_dtype(column_data):
                    inferred_type = DataType.BOOLEAN
                elif pd.api.types.is_datetime64_any_dtype(column_data):
                    inferred_type = DataType.DATETIME
                elif pd.api.types.is_string_dtype(column_data):
                    inferred_type = DataType.STRING

                col_profile = ColumnProfile(
                    column_name=col_name,
                    data_type=inferred_type,
                    statistics=stats
                )
                column_profiles.append(col_profile)

            data_profile.column_profiles = column_profiles
            data_profile.complete_profiling()

        except Exception as e:
            data_profile.fail_profiling(str(e))
            raise

        self.data_profile_repository.save(data_profile)
        return data_profile

    def get_profile(self, profile_id: UUID) -> DataProfile:
        """Get a data profile by its ID."""
        return self.data_profile_repository.get_by_id(profile_id)