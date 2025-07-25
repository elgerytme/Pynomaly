"""Dependency injection container for data quality services."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar
from pathlib import Path

# Domain interfaces
from data_quality.domain.interfaces.data_processing_operations import (
    DataProfilingPort,
    DataValidationPort,
    StatisticalAnalysisPort,
    DataSamplingPort,
    DataTransformationPort
)
from data_quality.domain.interfaces.external_system_operations import (
    DataSourcePort,
    FileSystemPort,
    NotificationPort,
    ReportingPort,
    MetadataPort,
    CloudStoragePort
)
from data_quality.domain.interfaces.quality_assessment_operations import (
    RuleEvaluationPort,
    QualityMetricsPort,
    AnomalyDetectionPort,
    QualityMonitoringPort,
    DataLineagePort
)

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class DataQualityContainerConfig:
    """Configuration for the data quality dependency injection container."""
    
    # Data processing configuration
    enable_file_data_processing: bool = True
    enable_spark_data_processing: bool = False
    enable_pandas_profiling: bool = True
    data_storage_path: str = "data_quality_storage"
    
    # External system configuration
    enable_file_system: bool = True
    enable_cloud_storage: bool = False
    enable_email_notifications: bool = False
    enable_slack_notifications: bool = False
    
    # Quality assessment configuration
    enable_statistical_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_quality_monitoring: bool = True
    
    # Database configuration
    enable_postgresql: bool = False
    enable_sqlite: bool = True
    database_url: str = "sqlite:///data_quality.db"
    
    # Reporting configuration
    enable_pdf_reports: bool = True
    enable_html_reports: bool = True
    reports_path: str = "reports"
    
    # Environment settings
    environment: str = "development"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize default configurations."""
        pass


class DataQualityContainer:
    """Dependency injection container for data quality services."""
    
    _instance: Optional['DataQualityContainer'] = None
    
    def __init__(self, config: Optional[DataQualityContainerConfig] = None):
        """Initialize the container with configuration.
        
        Args:
            config: Container configuration
        """
        self._config = config or DataQualityContainerConfig()
        self._singletons: Dict[Type, Any] = {}
        self._configure_adapters()
        self._configure_domain_services()
        
        logger.info(f"Data Quality container initialized for {self._config.environment} environment")
    
    def _configure_adapters(self):
        """Configure infrastructure adapters based on configuration."""
        self._configure_data_processing_adapters()
        self._configure_external_system_adapters()
        self._configure_quality_assessment_adapters()
    
    def _configure_data_processing_adapters(self):
        """Configure data processing adapters."""
        # File-based data processing (default)
        if self._config.enable_file_data_processing:
            try:
                from data_quality.infrastructure.adapters.file_based.data_processing_adapters import (
                    FileBasedDataProfiling,
                    FileBasedDataValidation,
                    FileBasedStatisticalAnalysis
                )
                
                storage_path = Path(self._config.data_storage_path)
                
                self._singletons[DataProfilingPort] = FileBasedDataProfiling(str(storage_path / "profiling"))
                self._singletons[DataValidationPort] = FileBasedDataValidation(str(storage_path / "validation"))
                self._singletons[StatisticalAnalysisPort] = FileBasedStatisticalAnalysis(str(storage_path / "analysis"))
                
                logger.info("File-based data processing adapters configured")
                return
                
            except ImportError:
                logger.warning("File data processing adapters not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_data_processing_stubs()
    
    def _configure_external_system_adapters(self):
        """Configure external system adapters."""
        # File system adapter (default)
        if self._config.enable_file_system:
            try:
                from data_quality.infrastructure.adapters.file_based.external_system_adapters import (
                    FileBasedDataSource,
                    LocalFileSystem,
                    FileBasedMetadata
                )
                
                storage_path = Path(self._config.data_storage_path)
                
                self._singletons[DataSourcePort] = FileBasedDataSource(str(storage_path / "datasources"))
                self._singletons[FileSystemPort] = LocalFileSystem()
                self._singletons[MetadataPort] = FileBasedMetadata(str(storage_path / "metadata"))
                
                logger.info("File-based external system adapters configured")
                
            except ImportError:
                logger.warning("File external system adapters not available, falling back to stubs")
        
        # Cloud storage
        if self._config.enable_cloud_storage:
            try:
                from data_quality.infrastructure.adapters.cloud.cloud_storage_adapters import (
                    S3CloudStorageAdapter
                )
                
                self._singletons[CloudStoragePort] = S3CloudStorageAdapter()
                logger.info("Cloud storage adapters configured")
                
            except ImportError:
                logger.warning("Cloud storage adapters not available")
        
        # Notification adapters
        if self._config.enable_email_notifications:
            try:
                from data_quality.infrastructure.adapters.notification.email_adapter import EmailNotificationAdapter
                
                self._singletons[NotificationPort] = EmailNotificationAdapter()
                logger.info("Email notification adapter configured")
                
            except ImportError:
                logger.warning("Email notification adapter not available")
        
        # Fallback to stub implementations for missing adapters
        self._configure_external_system_stubs()
    
    def _configure_quality_assessment_adapters(self):
        """Configure quality assessment adapters."""
        # Statistical analysis and quality assessment
        if self._config.enable_statistical_analysis:
            try:
                from data_quality.infrastructure.adapters.file_based.quality_assessment_adapters import (
                    FileBasedRuleEvaluation,
                    FileBasedQualityMetrics,
                    FileBasedAnomalyDetection,
                    FileBasedQualityMonitoring,
                    FileBasedDataLineage
                )
                
                storage_path = Path(self._config.data_storage_path)
                
                self._singletons[RuleEvaluationPort] = FileBasedRuleEvaluation(str(storage_path / "rules"))
                self._singletons[QualityMetricsPort] = FileBasedQualityMetrics(str(storage_path / "metrics"))
                self._singletons[DataLineagePort] = FileBasedDataLineage(str(storage_path / "lineage"))
                
                if self._config.enable_anomaly_detection:
                    self._singletons[AnomalyDetectionPort] = FileBasedAnomalyDetection(str(storage_path / "anomalies"))
                
                if self._config.enable_quality_monitoring:
                    self._singletons[QualityMonitoringPort] = FileBasedQualityMonitoring(str(storage_path / "monitoring"))
                
                logger.info("File-based quality assessment adapters configured")
                return
                
            except ImportError:
                logger.warning("Quality assessment adapters not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_quality_assessment_stubs()
    
    def _configure_data_processing_stubs(self):
        """Configure data processing stub implementations."""
        from data_quality.infrastructure.adapters.stubs.data_processing_stubs import (
            DataProfilingStub,
            DataValidationStub,
            StatisticalAnalysisStub,
            DataSamplingStub,
            DataTransformationStub
        )
        
        self._singletons[DataProfilingPort] = DataProfilingStub()
        self._singletons[DataValidationPort] = DataValidationStub()
        self._singletons[StatisticalAnalysisPort] = StatisticalAnalysisStub()
        self._singletons[DataSamplingPort] = DataSamplingStub()
        self._singletons[DataTransformationPort] = DataTransformationStub()
        
        logger.info("Data processing stubs configured")
    
    def _configure_external_system_stubs(self):
        """Configure external system stub implementations."""
        from data_quality.infrastructure.adapters.stubs.external_system_stubs import (
            DataSourceStub,
            FileSystemStub,
            NotificationStub,
            ReportingStub,
            MetadataStub,
            CloudStorageStub
        )
        
        # Only configure stubs for missing services
        if DataSourcePort not in self._singletons:
            self._singletons[DataSourcePort] = DataSourceStub()
        if FileSystemPort not in self._singletons:
            self._singletons[FileSystemPort] = FileSystemStub()
        if NotificationPort not in self._singletons:
            self._singletons[NotificationPort] = NotificationStub()
        if ReportingPort not in self._singletons:
            self._singletons[ReportingPort] = ReportingStub()
        if MetadataPort not in self._singletons:
            self._singletons[MetadataPort] = MetadataStub()
        if CloudStoragePort not in self._singletons:
            self._singletons[CloudStoragePort] = CloudStorageStub()
        
        logger.info("External system stubs configured")
    
    def _configure_quality_assessment_stubs(self):
        """Configure quality assessment stub implementations."""
        from data_quality.infrastructure.adapters.stubs.quality_assessment_stubs import (
            RuleEvaluationStub,
            QualityMetricsStub,
            AnomalyDetectionStub,
            QualityMonitoringStub,
            DataLineageStub
        )
        
        # Only configure stubs for missing services
        if RuleEvaluationPort not in self._singletons:
            self._singletons[RuleEvaluationPort] = RuleEvaluationStub()
        if QualityMetricsPort not in self._singletons:
            self._singletons[QualityMetricsPort] = QualityMetricsStub()
        if AnomalyDetectionPort not in self._singletons:
            self._singletons[AnomalyDetectionPort] = AnomalyDetectionStub()
        if QualityMonitoringPort not in self._singletons:
            self._singletons[QualityMonitoringPort] = QualityMonitoringStub()
        if DataLineagePort not in self._singletons:
            self._singletons[DataLineagePort] = DataLineageStub()
        
        logger.info("Quality assessment stubs configured")
    
    def _configure_domain_services(self):
        """Configure domain services with dependency injection."""
        try:
            from data_quality.domain.services.data_quality_service import DataQualityService
            from data_quality.domain.services.data_profiling_service import DataProfilingService
            from data_quality.domain.services.quality_monitoring_service import QualityMonitoringService
            
            # Data profiling service
            profiling_service = DataProfilingService(
                data_profiling_port=self.get(DataProfilingPort),
                statistical_analysis_port=self.get(StatisticalAnalysisPort),
                data_source_port=self.get(DataSourcePort),
                metadata_port=self.get(MetadataPort)
            )
            self._singletons[DataProfilingService] = profiling_service
            
            # Data quality service
            quality_service = DataQualityService(
                data_validation_port=self.get(DataValidationPort),
                rule_evaluation_port=self.get(RuleEvaluationPort),
                quality_metrics_port=self.get(QualityMetricsPort),
                data_source_port=self.get(DataSourcePort),
                notification_port=self.get(NotificationPort),
                reporting_port=self.get(ReportingPort)
            )
            self._singletons[DataQualityService] = quality_service
            
            # Quality monitoring service
            monitoring_service = QualityMonitoringService(
                quality_monitoring_port=self.get(QualityMonitoringPort),
                anomaly_detection_port=self.get(AnomalyDetectionPort),
                data_lineage_port=self.get(DataLineagePort),
                notification_port=self.get(NotificationPort),
                metadata_port=self.get(MetadataPort)
            )
            self._singletons[QualityMonitoringService] = monitoring_service
            
            logger.info("Domain services configured")
            
        except ImportError as e:
            logger.warning(f"Could not configure domain services: {e}")
    
    def get(self, interface: Type[T]) -> T:
        """Get a service instance by interface type.
        
        Args:
            interface: The interface type to retrieve
            
        Returns:
            Service instance implementing the interface
            
        Raises:
            ValueError: If service is not registered
        """
        if interface in self._singletons:
            return self._singletons[interface]
        
        raise ValueError(f"Service not registered: {interface.__name__}")
    
    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered.
        
        Args:
            interface: The interface type to check
            
        Returns:
            True if service is registered
        """
        return interface in self._singletons
    
    def register_singleton(self, interface: Type[T], implementation: T):
        """Register a singleton service.
        
        Args:
            interface: The interface type
            implementation: The implementation instance
        """
        self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the container configuration.
        
        Returns:
            Configuration summary
        """
        return {
            "environment": self._config.environment,
            "data_processing": {
                "file_enabled": self._config.enable_file_data_processing,
                "spark_enabled": self._config.enable_spark_data_processing,
                "pandas_profiling_enabled": self._config.enable_pandas_profiling,
                "storage_path": self._config.data_storage_path
            },
            "external_systems": {
                "file_system_enabled": self._config.enable_file_system,
                "cloud_storage_enabled": self._config.enable_cloud_storage,
                "email_notifications_enabled": self._config.enable_email_notifications,
                "slack_notifications_enabled": self._config.enable_slack_notifications
            },
            "quality_assessment": {
                "statistical_analysis_enabled": self._config.enable_statistical_analysis,
                "anomaly_detection_enabled": self._config.enable_anomaly_detection,
                "quality_monitoring_enabled": self._config.enable_quality_monitoring
            },
            "database": {
                "postgresql_enabled": self._config.enable_postgresql,
                "sqlite_enabled": self._config.enable_sqlite,
                "url": self._config.database_url
            },
            "reporting": {
                "pdf_reports_enabled": self._config.enable_pdf_reports,
                "html_reports_enabled": self._config.enable_html_reports,
                "reports_path": self._config.reports_path
            },
            "registered_services": {
                "singletons": [service.__name__ for service in self._singletons.keys()],
                "count": len(self._singletons)
            }
        }
    
    def configure_data_processing(
        self,
        enable_file: Optional[bool] = None,
        enable_spark: Optional[bool] = None,
        enable_pandas_profiling: Optional[bool] = None,
        storage_path: Optional[str] = None
    ):
        """Reconfigure data processing at runtime.
        
        Args:
            enable_file: Enable file-based processing
            enable_spark: Enable Spark processing
            enable_pandas_profiling: Enable pandas profiling
            storage_path: Storage path for data
        """
        if enable_file is not None:
            self._config.enable_file_data_processing = enable_file
        if enable_spark is not None:
            self._config.enable_spark_data_processing = enable_spark
        if enable_pandas_profiling is not None:
            self._config.enable_pandas_profiling = enable_pandas_profiling
        if storage_path is not None:
            self._config.data_storage_path = storage_path
        
        # Reconfigure adapters
        self._configure_data_processing_adapters()
        logger.info("Data processing configuration updated")
    
    def configure_quality_assessment(
        self,
        enable_statistical_analysis: Optional[bool] = None,
        enable_anomaly_detection: Optional[bool] = None,
        enable_quality_monitoring: Optional[bool] = None
    ):
        """Reconfigure quality assessment at runtime.
        
        Args:
            enable_statistical_analysis: Enable statistical analysis
            enable_anomaly_detection: Enable anomaly detection
            enable_quality_monitoring: Enable quality monitoring
        """
        if enable_statistical_analysis is not None:
            self._config.enable_statistical_analysis = enable_statistical_analysis
        if enable_anomaly_detection is not None:
            self._config.enable_anomaly_detection = enable_anomaly_detection
        if enable_quality_monitoring is not None:
            self._config.enable_quality_monitoring = enable_quality_monitoring
        
        # Reconfigure adapters
        self._configure_quality_assessment_adapters()
        logger.info("Quality assessment configuration updated")


# Global container instance
_global_container: Optional[DataQualityContainer] = None


def get_container(config: Optional[DataQualityContainerConfig] = None) -> DataQualityContainer:
    """Get the global container instance.
    
    Args:
        config: Optional configuration for new container
        
    Returns:
        Global container instance
    """
    global _global_container
    
    if _global_container is None:
        _global_container = DataQualityContainer(config)
    
    return _global_container


def reset_container():
    """Reset the global container instance."""
    global _global_container
    _global_container = None