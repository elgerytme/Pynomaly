"""Standardized logging for data quality package using the shared framework."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import from shared infrastructure
sys.path.append(str(Path(__file__).parents[6] / "shared" / "infrastructure"))

try:
    from logging.structured_logging import (
        StructuredLogger,
        BaseLoggerConfig,
        PerformanceTimer,
        LogLevel,
        LogFormat,
        create_logger_from_env,
        get_logger,
        setup_package_logging,
        log_decorator,
        log_context,
        STRUCTLOG_AVAILABLE
    )
    SHARED_LOGGING_AVAILABLE = True
except ImportError:
    # Fallback if shared infrastructure is not available
    SHARED_LOGGING_AVAILABLE = False
    import logging
    
    class StructuredLogger:
        def __init__(self, package_name: str):
            self.logger = logging.getLogger(package_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        def debug(self, message: str, **kwargs): self.logger.debug(f"{message} | {kwargs}")
        def info(self, message: str, **kwargs): self.logger.info(f"{message} | {kwargs}")
        def warning(self, message: str, **kwargs): self.logger.warning(f"{message} | {kwargs}")
        def error(self, message: str, **kwargs): self.logger.error(f"{message} | {kwargs}")
        def critical(self, message: str, **kwargs): self.logger.critical(f"{message} | {kwargs}")
        
        def set_context(self, **kwargs): pass
        def clear_context(self): pass
        def set_request_context(self, **kwargs): pass
        def clear_request_context(self): pass
        
        def log_performance(self, operation: str, duration_ms: float, success: bool = True, **kwargs):
            self.info(f"Performance: {operation}", duration_ms=duration_ms, success=success, **kwargs)
        
        def log_data_quality(self, dataset_name: str, check_name: str, result: str, score: Optional[float] = None, **kwargs):
            self.info(f"Data Quality: {check_name}", dataset_name=dataset_name, result=result, score=score, **kwargs)
    
    class PerformanceTimer:
        def __init__(self, logger, operation: str, **context):
            self.logger = logger
            self.operation = operation
            self.context = context
        
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def log_decorator(operation=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def log_context(logger, **context):
        from contextlib import nullcontext
        return nullcontext()


class QualityLogger(StructuredLogger):
    """Enhanced logger for data quality operations."""
    
    def __init__(self, config: Optional[Any] = None):
        if SHARED_LOGGING_AVAILABLE:
            if config is None:
                # Use environment-based configuration
                super().__init__(create_logger_from_env("quality")._logger)
            else:
                super().__init__(config, create_logger_from_env("quality")._logger)
        else:
            super().__init__("quality")
    
    def log_validation_start(
        self,
        dataset_name: str,
        rule_count: int,
        data_size: int,
        **kwargs
    ) -> None:
        """Log the start of data validation."""
        self.info(
            f"Starting data validation for dataset: {dataset_name}",
            dataset_name=dataset_name,
            rule_count=rule_count,
            data_size=data_size,
            validation_stage="start",
            **kwargs
        )
    
    def log_validation_complete(
        self,
        dataset_name: str,
        total_checks: int,
        passed_checks: int,
        failed_checks: int,
        warnings: int,
        overall_score: float,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log the completion of data validation."""
        self.log_data_quality(
            dataset_name=dataset_name,
            check_name="overall_validation",
            result="completed",
            score=overall_score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            duration_ms=duration_ms,
            validation_stage="complete",
            **kwargs
        )
    
    def log_rule_execution(
        self,
        rule_name: str,
        dataset_name: str,
        rule_type: str,
        result: str,
        score: Optional[float] = None,
        duration_ms: Optional[float] = None,
        error_count: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log individual rule execution."""
        log_data = {
            "rule_name": rule_name,
            "dataset_name": dataset_name,
            "rule_type": rule_type,
            "result": result,
            "rule_execution": True
        }
        
        if score is not None:
            log_data["score"] = score
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        if error_count is not None:
            log_data["error_count"] = error_count
        
        log_data.update(kwargs)
        
        if result == "failed" or result == "error":
            self.warning(f"Rule execution: {rule_name}", **log_data)
        else:
            self.info(f"Rule execution: {rule_name}", **log_data)
    
    def log_profiling_start(
        self,
        dataset_name: str,
        column_count: int,
        row_count: int,
        **kwargs
    ) -> None:
        """Log the start of data profiling."""
        self.info(
            f"Starting data profiling for dataset: {dataset_name}",
            dataset_name=dataset_name,
            column_count=column_count,
            row_count=row_count,
            profiling_stage="start",
            **kwargs
        )
    
    def log_profiling_complete(
        self,
        dataset_name: str,
        profile_stats: Dict[str, Any],
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log the completion of data profiling."""
        self.info(
            f"Data profiling completed for dataset: {dataset_name}",
            dataset_name=dataset_name,
            duration_ms=duration_ms,
            profiling_stage="complete",
            profile_stats=profile_stats,
            **kwargs
        )
    
    def log_column_profile(
        self,
        dataset_name: str,
        column_name: str,
        data_type: str,
        null_count: int,
        unique_count: int,
        **kwargs
    ) -> None:
        """Log column profiling results."""
        self.info(
            f"Column profile: {column_name}",
            dataset_name=dataset_name,
            column_name=column_name,
            data_type=data_type,
            null_count=null_count,
            unique_count=unique_count,
            column_profiling=True,
            **kwargs
        )
    
    def log_data_ingestion(
        self,
        source_path: str,
        dataset_name: str,
        rows_ingested: int,
        columns_ingested: int,
        ingestion_method: str,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log data ingestion operations."""
        self.info(
            f"Data ingested: {dataset_name}",
            source_path=source_path,
            dataset_name=dataset_name,
            rows_ingested=rows_ingested,
            columns_ingested=columns_ingested,
            ingestion_method=ingestion_method,
            duration_ms=duration_ms,
            data_ingestion=True,
            **kwargs
        )
    
    def log_quality_threshold_breach(
        self,
        dataset_name: str,
        threshold_type: str,  # warning, error, critical
        threshold_value: float,
        actual_score: float,
        check_name: str,
        **kwargs
    ) -> None:
        """Log quality threshold breaches."""
        log_data = {
            "dataset_name": dataset_name,
            "threshold_type": threshold_type,
            "threshold_value": threshold_value,
            "actual_score": actual_score,
            "check_name": check_name,
            "threshold_breach": True,
            **kwargs
        }
        
        if threshold_type == "critical":
            self.critical(f"Critical quality threshold breached: {check_name}", **log_data)
        elif threshold_type == "error":
            self.error(f"Error quality threshold breached: {check_name}", **log_data)
        else:
            self.warning(f"Warning quality threshold breached: {check_name}", **log_data)
    
    def log_data_export(
        self,
        dataset_name: str,
        export_path: str,
        export_format: str,
        rows_exported: int,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log data export operations."""
        self.info(
            f"Data exported: {dataset_name}",
            dataset_name=dataset_name,
            export_path=export_path,
            export_format=export_format,
            rows_exported=rows_exported,
            duration_ms=duration_ms,
            data_export=True,
            **kwargs
        )
    
    def log_cache_operation(
        self,
        operation: str,  # hit, miss, store, evict
        key: str,
        cache_type: str = "validation",
        **kwargs
    ) -> None:
        """Log cache operations."""
        self.debug(
            f"Cache {operation}: {key}",
            operation=operation,
            key=key,
            cache_type=cache_type,
            cache_operation=True,
            **kwargs
        )
    
    def log_webhook_event(
        self,
        event_type: str,
        webhook_url: str,
        status_code: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        success: bool = True,
        **kwargs
    ) -> None:
        """Log webhook events."""
        log_data = {
            "event_type": event_type,
            "webhook_url": webhook_url,
            "success": success,
            "webhook_event": True,
            **kwargs
        }
        
        if status_code is not None:
            log_data["status_code"] = status_code
        if response_time_ms is not None:
            log_data["response_time_ms"] = response_time_ms
        
        if success:
            self.info(f"Webhook sent: {event_type}", **log_data)
        else:
            self.error(f"Webhook failed: {event_type}", **log_data)


# Global logger instance
_quality_logger: Optional[QualityLogger] = None


def get_quality_logger() -> QualityLogger:
    """Get the global quality logger instance."""
    global _quality_logger
    if _quality_logger is None:
        _quality_logger = QualityLogger()
    return _quality_logger


def setup_quality_logging(
    level: str = LogLevel.INFO,
    format_type: str = LogFormat.JSON,
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: Optional[str] = None,
    **config_kwargs
) -> QualityLogger:
    """Setup logging for the quality package."""
    global _quality_logger
    
    if SHARED_LOGGING_AVAILABLE:
        config = BaseLoggerConfig(
            package_name="quality",
            level=level,
            format_type=format_type,
            enable_console=enable_console,
            enable_file=enable_file,
            file_path=file_path,
            **config_kwargs
        )
        
        logger = setup_package_logging("quality", config)
        _quality_logger = QualityLogger(config)
    else:
        _quality_logger = QualityLogger()
    
    return _quality_logger


# Convenience decorators for quality operations
def log_validation_operation(operation_name: Optional[str] = None):
    """Decorator for validation operations."""
    return log_decorator(
        operation=operation_name,
        log_duration=True,
        level=LogLevel.INFO
    )


def log_profiling_operation(operation_name: Optional[str] = None):
    """Decorator for profiling operations."""
    return log_decorator(
        operation=operation_name,
        log_duration=True,
        level=LogLevel.INFO
    )


def log_data_operation(operation_name: Optional[str] = None):
    """Decorator for data operations."""
    return log_decorator(
        operation=operation_name,
        log_duration=True,
        level=LogLevel.INFO
    )


# Backwards compatibility with the old logger
def configure_quality_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Configure logging for the entire data quality package (legacy)."""
    if config is None:
        config = {}
    
    level = config.get('log_level', LogLevel.INFO)
    format_type = config.get('log_format', LogFormat.JSON)
    
    setup_quality_logging(
        level=level,
        format_type=format_type,
        enable_console=True,
        enable_file=config.get('enable_file_logging', False),
        file_path=config.get('log_file_path')
    )


def get_logger(name: str = "quality") -> QualityLogger:
    """Get configured logger for data quality services (legacy)."""
    return get_quality_logger()


__all__ = [
    # Main logger class
    "QualityLogger",
    
    # Factory functions
    "get_quality_logger",
    "setup_quality_logging",
    
    # Decorators
    "log_validation_operation",
    "log_profiling_operation", 
    "log_data_operation",
    
    # Legacy compatibility
    "configure_quality_logging",
    "get_logger",
    
    # Re-exports from shared infrastructure
    "PerformanceTimer",
    "log_context",
    "LogLevel",
    "LogFormat",
    
    # Feature detection
    "SHARED_LOGGING_AVAILABLE"
]