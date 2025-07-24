
from typing import Optional
from uuid import UUID
import numpy as np

from sqlalchemy.orm import Session

from ...application.ports.data_profile_repository import DataProfileRepository
from ...domain.entities.data_profile import DataProfile, ColumnProfile, ProfileStatistics, DataType, ProfileStatus
from ..database.models import DataProfileModel


class SQLAlchemyDataProfileRepository(DataProfileRepository):
    """SQLAlchemy implementation of DataProfileRepository."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, id: UUID) -> Optional[DataProfile]:
        """Get a data profile by its ID."""
        model = self.session.query(DataProfileModel).filter_by(id=id).first()
        if model:
            return self._to_entity(model)
        return None

    def save(self, data_profile: DataProfile) -> None:
        """Save a data profile."""
        model = self.session.query(DataProfileModel).filter_by(id=data_profile.id).first()
        if model:
            self._update_model_from_entity(model, data_profile)
        else:
            model = self._to_model(data_profile)
            self.session.add(model)
        self.session.commit()

    def _to_entity(self, model: DataProfileModel) -> DataProfile:
        """Convert a SQLAlchemy model to a DataProfile entity."""
        column_profiles = []
        if model.column_profiles:
            for cp_dict in model.column_profiles:
                stats = ProfileStatistics(**cp_dict["statistics"])
                inferred_type_str = cp_dict.get("inferred_type")
                inferred_type = DataType(inferred_type_str) if inferred_type_str else None

                column_profiles.append(ColumnProfile(
                    id=UUID(cp_dict["id"]),
                    column_name=cp_dict["column_name"],
                    data_type=DataType(cp_dict["data_type"]),
                    inferred_type=inferred_type,
                    position=cp_dict["position"],
                    is_nullable=cp_dict["is_nullable"],
                    is_primary_key=cp_dict["is_primary_key"],
                    is_foreign_key=cp_dict["is_foreign_key"],
                    foreign_key_table=cp_dict["foreign_key_table"],
                    foreign_key_column=cp_dict["foreign_key_column"],
                    statistics=stats,
                    common_patterns=cp_dict["common_patterns"],
                    format_patterns=cp_dict["format_patterns"],
                    regex_patterns=cp_dict["regex_patterns"],
                    top_values=cp_dict["top_values"],
                    sample_values=cp_dict["sample_values"],
                    invalid_values=cp_dict["invalid_values"],
                    quality_score=cp_dict["quality_score"],
                    anomaly_count=cp_dict["anomaly_count"],
                    outlier_count=cp_dict["outlier_count"],
                    created_at=cp_dict["created_at"],
                    updated_at=cp_dict["updated_at"],
                ))

        return DataProfile(
            id=model.id,
            dataset_name=model.dataset_name,
            table_name=model.table_name,
            schema_name=model.schema_name,
            status=ProfileStatus(model.status),
            version=model.version,
            total_rows=model.total_rows,
            total_columns=model.total_columns,
            file_size_bytes=model.file_size_bytes,
            column_profiles=column_profiles,
            completeness_score=model.completeness_score,
            uniqueness_score=model.uniqueness_score,
            validity_score=model.validity_score,
            overall_quality_score=model.overall_quality_score,
            primary_keys=model.primary_keys,
            foreign_keys=model.foreign_keys,
            relationships=model.relationships,
            profiling_started_at=model.profiling_started_at,
            profiling_completed_at=model.profiling_completed_at,
            profiling_duration_ms=model.profiling_duration_ms,
            sample_size=model.sample_size,
            sampling_method=model.sampling_method,
            created_at=model.created_at,
            created_by=model.created_by,
            updated_at=model.updated_at,
            updated_by=model.updated_by,
            config=model.config,
            tags=model.tags,
        )

    def _to_model(self, entity: DataProfile) -> DataProfileModel:
        """Convert a DataProfile entity to a SQLAlchemy model."""
        # Helper to convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(elem) for elem in obj]
            return obj

        return DataProfileModel(
            id=entity.id,
            dataset_name=entity.dataset_name,
            table_name=entity.table_name,
            schema_name=entity.schema_name,
            status=entity.status.value,
            version=entity.version,
            total_rows=entity.total_rows,
            total_columns=entity.total_columns,
            file_size_bytes=entity.file_size_bytes,
            column_profiles=[convert_numpy_types(cp.to_dict()) for cp in entity.column_profiles],
            completeness_score=convert_numpy_types(entity.completeness_score),
            uniqueness_score=convert_numpy_types(entity.uniqueness_score),
            validity_score=convert_numpy_types(entity.validity_score),
            overall_quality_score=convert_numpy_types(entity.overall_quality_score),
            primary_keys=convert_numpy_types(entity.primary_keys),
            foreign_keys=convert_numpy_types(entity.foreign_keys),
            relationships=convert_numpy_types(entity.relationships),
            profiling_started_at=entity.profiling_started_at,
            profiling_completed_at=entity.profiling_completed_at,
            profiling_duration_ms=convert_numpy_types(entity.profiling_duration_ms),
            sample_size=convert_numpy_types(entity.sample_size),
            sampling_method=entity.sampling_method,
            created_at=entity.created_at,
            created_by=entity.created_by,
            updated_at=entity.updated_at,
            updated_by=entity.updated_by,
            config=convert_numpy_types(entity.config),
            tags=convert_numpy_types(entity.tags),
        )

    def _update_model_from_entity(self, model: DataProfileModel, entity: DataProfile) -> None:
        """Update a SQLAlchemy model from a DataProfile entity."""
        # Helper to convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(elem) for elem in obj]
            return obj

        model.dataset_name = entity.dataset_name
        model.table_name = entity.table_name
        model.schema_name = entity.schema_name
        model.status = entity.status.value
        model.version = entity.version
        model.total_rows = entity.total_rows
        model.total_columns = entity.total_columns
        model.file_size_bytes = entity.file_size_bytes
        model.column_profiles = [convert_numpy_types(cp.to_dict()) for cp in entity.column_profiles]
        model.completeness_score = convert_numpy_types(entity.completeness_score)
        model.uniqueness_score = convert_numpy_types(entity.uniqueness_score)
        model.validity_score = convert_numpy_types(entity.validity_score)
        model.overall_quality_score = convert_numpy_types(entity.overall_quality_score)
        model.primary_keys = convert_numpy_types(entity.primary_keys)
        model.foreign_keys = convert_numpy_types(entity.foreign_keys)
        model.relationships = convert_numpy_types(entity.relationships)
        model.profiling_started_at = entity.profiling_started_at
        model.profiling_completed_at = entity.profiling_completed_at
        model.profiling_duration_ms = convert_numpy_types(entity.profiling_duration_ms)
        model.sample_size = convert_numpy_types(entity.sample_size)
        model.sampling_method = entity.sampling_method
        model.created_at = entity.created_at
        model.created_by = entity.created_by
        model.updated_at = entity.updated_at
        model.updated_by = entity.updated_by
        model.config = convert_numpy_types(entity.config)
        model.tags = convert_numpy_types(entity.tags)
