"""Data domain interfaces for cross-package communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..domain.abstractions import ServiceInterface


class DataProcessingInterface(ServiceInterface):
    """Interface for data processing services."""
    
    @abstractmethod
    async def process_data(self, data: Any, processing_config: Dict[str, Any]) -> Any:
        """Process data according to configuration."""
        pass
    
    @abstractmethod
    async def validate_data(self, data: Any, validation_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data against rules."""
        pass
    
    @abstractmethod
    async def transform_data(self, data: Any, transformation_config: Dict[str, Any]) -> Any:
        """Transform data according to configuration."""
        pass
    
    @abstractmethod
    async def clean_data(self, data: Any, cleaning_config: Optional[Dict[str, Any]] = None) -> Any:
        """Clean data by removing/fixing issues."""
        pass
    
    @abstractmethod
    async def aggregate_data(self, data: Any, aggregation_config: Dict[str, Any]) -> Any:
        """Aggregate data according to configuration."""
        pass


class QualityAssessmentInterface(ServiceInterface):
    """Interface for data quality assessment services."""
    
    @abstractmethod
    async def assess_quality(self, dataset_id: UUID) -> Dict[str, Any]:
        """Assess overall data quality for a dataset."""
        pass
    
    @abstractmethod
    async def run_quality_checks(self, dataset_id: UUID, check_ids: List[UUID]) -> List[Dict[str, Any]]:
        """Run specific quality checks on a dataset."""
        pass
    
    @abstractmethod
    async def create_quality_rule(self, rule_data: Dict[str, Any]) -> UUID:
        """Create a new data quality rule."""
        pass
    
    @abstractmethod
    async def get_quality_report(self, dataset_id: UUID, time_range: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get quality report for a dataset."""
        pass
    
    @abstractmethod
    async def get_quality_trends(self, dataset_id: UUID, metric: str, time_range: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get quality trends for specific metrics."""
        pass
    
    @abstractmethod
    async def detect_anomalies(self, dataset_id: UUID, column: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect data anomalies in dataset or specific column."""
        pass


class ProfilingEngineInterface(ServiceInterface):
    """Interface for data profiling services."""
    
    @abstractmethod
    async def profile_dataset(self, dataset_id: UUID, profiling_config: Optional[Dict[str, Any]] = None) -> UUID:
        """Start profiling job for a dataset."""
        pass
    
    @abstractmethod
    async def get_profile(self, profile_id: UUID) -> Optional[Dict[str, Any]]:
        """Get data profile results."""
        pass
    
    @abstractmethod
    async def get_column_profile(self, profile_id: UUID, column_name: str) -> Optional[Dict[str, Any]]:
        """Get profile for specific column."""
        pass
    
    @abstractmethod
    async def get_profile_summary(self, profile_id: UUID) -> Dict[str, Any]:
        """Get summary of data profile."""
        pass
    
    @abstractmethod
    async def compare_profiles(self, profile_id_1: UUID, profile_id_2: UUID) -> Dict[str, Any]:
        """Compare two data profiles."""
        pass
    
    @abstractmethod
    async def get_data_types(self, profile_id: UUID) -> Dict[str, str]:
        """Get inferred data types for all columns."""
        pass


class ValidationEngineInterface(ServiceInterface):
    """Interface for data validation services."""
    
    @abstractmethod
    async def validate_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against schema."""
        pass
    
    @abstractmethod
    async def validate_constraints(self, data: Any, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data against constraints."""
        pass
    
    @abstractmethod
    async def validate_business_rules(self, data: Any, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data against business rules."""
        pass
    
    @abstractmethod
    async def create_validation_rule(self, rule_data: Dict[str, Any]) -> UUID:
        """Create a new validation rule."""
        pass
    
    @abstractmethod
    async def get_validation_results(self, validation_id: UUID) -> Optional[Dict[str, Any]]:
        """Get validation results."""
        pass


class DataCatalogInterface(ServiceInterface):
    """Interface for data catalog services."""
    
    @abstractmethod
    async def register_dataset(self, dataset_metadata: Dict[str, Any]) -> UUID:
        """Register a new dataset in catalog."""
        pass
    
    @abstractmethod
    async def get_dataset(self, dataset_id: UUID) -> Optional[Dict[str, Any]]:
        """Get dataset metadata."""
        pass
    
    @abstractmethod
    async def search_datasets(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search datasets by query and filters."""
        pass
    
    @abstractmethod
    async def update_dataset_metadata(self, dataset_id: UUID, metadata: Dict[str, Any]) -> bool:
        """Update dataset metadata."""
        pass
    
    @abstractmethod
    async def add_dataset_tags(self, dataset_id: UUID, tags: List[str]) -> bool:
        """Add tags to dataset."""
        pass
    
    @abstractmethod
    async def get_dataset_lineage(self, dataset_id: UUID) -> Dict[str, Any]:
        """Get data lineage for dataset."""
        pass


class DataIngestionInterface(ServiceInterface):
    """Interface for data ingestion services."""
    
    @abstractmethod
    async def ingest_batch_data(self, source_config: Dict[str, Any], destination_config: Dict[str, Any]) -> UUID:
        """Ingest batch data from source to destination."""
        pass
    
    @abstractmethod
    async def ingest_streaming_data(self, stream_config: Dict[str, Any]) -> UUID:
        """Start streaming data ingestion."""
        pass
    
    @abstractmethod
    async def get_ingestion_job(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """Get ingestion job status."""
        pass
    
    @abstractmethod
    async def list_ingestion_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List ingestion jobs with optional status filter."""
        pass
    
    @abstractmethod
    async def cancel_ingestion_job(self, job_id: UUID) -> bool:
        """Cancel a running ingestion job."""
        pass
    
    @abstractmethod
    async def validate_source_connection(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate connection to data source."""
        pass


class DataLineageInterface(ServiceInterface):
    """Interface for data lineage services."""
    
    @abstractmethod
    async def track_transformation(self, transformation_data: Dict[str, Any]) -> UUID:
        """Track a data transformation."""
        pass
    
    @abstractmethod
    async def get_upstream_lineage(self, dataset_id: UUID, depth: int = 5) -> Dict[str, Any]:
        """Get upstream data lineage."""
        pass
    
    @abstractmethod
    async def get_downstream_lineage(self, dataset_id: UUID, depth: int = 5) -> Dict[str, Any]:
        """Get downstream data lineage."""
        pass
    
    @abstractmethod
    async def get_column_lineage(self, dataset_id: UUID, column_name: str) -> Dict[str, Any]:
        """Get lineage for specific column."""
        pass
    
    @abstractmethod
    async def find_impact_analysis(self, dataset_id: UUID) -> Dict[str, Any]:
        """Find impact of changes to dataset."""
        pass
    
    @abstractmethod
    async def get_lineage_graph(self, dataset_id: UUID) -> Dict[str, Any]:
        """Get lineage as graph structure."""
        pass