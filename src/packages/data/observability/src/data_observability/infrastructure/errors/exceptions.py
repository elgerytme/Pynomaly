"""Custom exceptions for data observability."""

from typing import Any, Dict, Optional


class DataObservabilityError(Exception):
    """Base exception for all data observability errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": self.error_code,
            "message": self.message,
        }
        
        if self.details:
            result["details"] = self.details
        
        if self.cause:
            result["cause"] = str(self.cause)
        
        return result


class ValidationError(DataObservabilityError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        
        super().__init__(message, error_code="VALIDATION_ERROR", details=details, **kwargs)
        self.field = field
        self.value = value


class AssetNotFoundError(DataObservabilityError):
    """Raised when a data asset is not found."""
    
    def __init__(
        self,
        asset_id: str,
        asset_type: Optional[str] = None,
        **kwargs
    ):
        message = f"Asset not found: {asset_id}"
        if asset_type:
            message = f"{asset_type} asset not found: {asset_id}"
        
        details = {"asset_id": asset_id}
        if asset_type:
            details["asset_type"] = asset_type
        
        super().__init__(
            message,
            error_code="ASSET_NOT_FOUND",
            details=details,
            **kwargs
        )
        self.asset_id = asset_id
        self.asset_type = asset_type


class ConfigurationError(DataObservabilityError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            **kwargs
        )
        self.config_key = config_key


class DatabaseError(DataObservabilityError):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if table:
            details["table"] = table
        
        super().__init__(
            message,
            error_code="DATABASE_ERROR",
            details=details,
            **kwargs
        )
        self.operation = operation
        self.table = table


class ServiceError(DataObservabilityError):
    """Raised when service operations fail."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if service:
            details["service"] = service
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message,
            error_code="SERVICE_ERROR",
            details=details,
            **kwargs
        )
        self.service = service
        self.operation = operation


class AuthenticationError(DataObservabilityError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        username: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if username:
            details["username"] = username
        
        super().__init__(
            message,
            error_code="AUTHENTICATION_ERROR",
            details=details,
            **kwargs
        )
        self.username = username


class AuthorizationError(DataObservabilityError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        username: Optional[str] = None,
        required_permission: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if username:
            details["username"] = username
        if required_permission:
            details["required_permission"] = required_permission
        
        super().__init__(
            message,
            error_code="AUTHORIZATION_ERROR",
            details=details,
            **kwargs
        )
        self.username = username
        self.required_permission = required_permission


class LineageError(DataObservabilityError):
    """Raised when data lineage operations fail."""
    
    def __init__(
        self,
        message: str,
        source_asset: Optional[str] = None,
        target_asset: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if source_asset:
            details["source_asset"] = source_asset
        if target_asset:
            details["target_asset"] = target_asset
        
        super().__init__(
            message,
            error_code="LINEAGE_ERROR",
            details=details,
            **kwargs
        )
        self.source_asset = source_asset
        self.target_asset = target_asset


class QualityError(DataObservabilityError):
    """Raised when data quality operations fail."""
    
    def __init__(
        self,
        message: str,
        asset_id: Optional[str] = None,
        metric: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if asset_id:
            details["asset_id"] = asset_id
        if metric:
            details["metric"] = metric
        
        super().__init__(
            message,
            error_code="QUALITY_ERROR",
            details=details,
            **kwargs
        )
        self.asset_id = asset_id
        self.metric = metric


class PipelineError(DataObservabilityError):
    """Raised when pipeline health operations fail."""
    
    def __init__(
        self,
        message: str,
        pipeline_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if pipeline_id:
            details["pipeline_id"] = pipeline_id
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message,
            error_code="PIPELINE_ERROR",
            details=details,
            **kwargs
        )
        self.pipeline_id = pipeline_id
        self.operation = operation


class ExternalSystemError(DataObservabilityError):
    """Raised when external system integration fails."""
    
    def __init__(
        self,
        message: str,
        system: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if system:
            details["system"] = system
        if endpoint:
            details["endpoint"] = endpoint
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(
            message,
            error_code="EXTERNAL_SYSTEM_ERROR",
            details=details,
            **kwargs
        )
        self.system = system
        self.endpoint = endpoint
        self.status_code = status_code


class RateLimitError(DataObservabilityError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if reset_time:
            details["reset_time"] = reset_time
        
        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            details=details,
            **kwargs
        )
        self.limit = limit
        self.reset_time = reset_time