"""Domain adapters for standardized cross-domain data transformation.

This module provides adapters and transformation patterns to enable clean
integration between domains while preserving domain boundaries and preventing
tight coupling.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import (
    Any, Dict, List, Optional, Type, TypeVar, Generic,
    Union, Callable, Protocol, runtime_checkable
)
from enum import Enum

import structlog

from .cross_domain_patterns import (
    CrossDomainMessage, IntegrationContext, IntegrationResult,
    IntegrationStatus, AntiCorruptionLayer
)
from ..infrastructure.exceptions.base_exceptions import (
    BaseApplicationError, ErrorCategory, ErrorSeverity
)


logger = structlog.get_logger()

T = TypeVar('T')
S = TypeVar('S')


class AdapterError(BaseApplicationError):
    """Domain adapter errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class TransformationStrategy(Enum):
    """Data transformation strategies."""
    DIRECT_MAPPING = "direct_mapping"
    FIELD_MAPPING = "field_mapping"
    CUSTOM_TRANSFORM = "custom_transform"
    AGGREGATION = "aggregation"
    ENRICHMENT = "enrichment"


@runtime_checkable
class DomainEntity(Protocol):
    """Protocol for domain entities that can be adapted."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainEntity':
        """Create entity from dictionary."""
        ...


@dataclass
class FieldMapping:
    """Mapping configuration for field transformation."""
    source_field: str
    target_field: str
    transform_function: Optional[Callable[[Any], Any]] = None
    required: bool = True
    default_value: Any = None


@dataclass
class ValidationRule:
    """Validation rule for adapted data."""
    field: str
    validator: Callable[[Any], bool]
    error_message: str


class DataTransformer(Generic[S, T]):
    """Generic data transformer between domain types."""
    
    def __init__(
        self,
        source_type: Type[S],
        target_type: Type[T],
        field_mappings: List[FieldMapping] = None,
        validation_rules: List[ValidationRule] = None,
        transformation_strategy: TransformationStrategy = TransformationStrategy.FIELD_MAPPING
    ):
        self.source_type = source_type
        self.target_type = target_type
        self.field_mappings = field_mappings or []
        self.validation_rules = validation_rules or []
        self.transformation_strategy = transformation_strategy
    
    def transform(self, source: S, context: Optional[IntegrationContext] = None) -> T:
        """Transform source object to target type."""
        try:
            if self.transformation_strategy == TransformationStrategy.DIRECT_MAPPING:
                return self._direct_transform(source)
            elif self.transformation_strategy == TransformationStrategy.FIELD_MAPPING:
                return self._field_mapping_transform(source)
            elif self.transformation_strategy == TransformationStrategy.CUSTOM_TRANSFORM:
                return self._custom_transform(source, context)
            else:
                raise AdapterError(f"Unsupported transformation strategy: {self.transformation_strategy}")
                
        except Exception as e:
            logger.error("Data transformation failed", 
                        source_type=self.source_type.__name__,
                        target_type=self.target_type.__name__,
                        error=str(e))
            raise AdapterError(f"Transformation failed: {e}") from e
    
    def _direct_transform(self, source: S) -> T:
        """Direct transformation (assumes compatible types)."""
        if hasattr(source, 'to_dict') and hasattr(self.target_type, 'from_dict'):
            source_dict = source.to_dict()
            return self.target_type.from_dict(source_dict)
        else:
            raise AdapterError("Direct transformation not supported for these types")
    
    def _field_mapping_transform(self, source: S) -> T:
        """Transform using field mappings."""
        # Convert source to dict if possible
        if hasattr(source, 'to_dict'):
            source_dict = source.to_dict()
        elif isinstance(source, dict):
            source_dict = source
        else:
            source_dict = asdict(source) if hasattr(source, '__dataclass_fields__') else {}
        
        # Apply field mappings
        target_dict = {}
        for mapping in self.field_mappings:
            source_value = source_dict.get(mapping.source_field)
            
            if source_value is None:
                if mapping.required and mapping.default_value is None:
                    raise AdapterError(f"Required field missing: {mapping.source_field}")
                source_value = mapping.default_value
            
            # Apply transformation function if provided
            if mapping.transform_function:
                source_value = mapping.transform_function(source_value)
            
            target_dict[mapping.target_field] = source_value
        
        # Validate transformed data
        self._validate_data(target_dict)
        
        # Create target object
        if hasattr(self.target_type, 'from_dict'):
            return self.target_type.from_dict(target_dict)
        else:
            return self.target_type(**target_dict)
    
    def _custom_transform(self, source: S, context: Optional[IntegrationContext]) -> T:
        """Custom transformation (to be overridden by subclasses)."""
        raise NotImplementedError("Custom transformation must be implemented by subclass")
    
    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate transformed data against rules."""
        for rule in self.validation_rules:
            field_value = data.get(rule.field)
            if not rule.validator(field_value):
                raise AdapterError(f"Validation failed for field {rule.field}: {rule.error_message}")


class DomainAdapter(ABC):
    """Abstract base class for domain adapters."""
    
    def __init__(self, source_domain: str, target_domain: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.transformers: Dict[str, DataTransformer] = {}
    
    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """Get list of operations this adapter supports."""
        pass
    
    @abstractmethod
    async def adapt_request(
        self, 
        operation: str, 
        payload: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt request from source domain to target domain format."""
        pass
    
    @abstractmethod
    async def adapt_response(
        self,
        operation: str,
        response: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt response from target domain to source domain format."""
        pass
    
    def register_transformer(self, operation: str, transformer: DataTransformer) -> None:
        """Register a data transformer for an operation."""
        self.transformers[operation] = transformer
    
    def get_transformer(self, operation: str) -> Optional[DataTransformer]:
        """Get transformer for operation."""
        return self.transformers.get(operation)


class StandardDomainAdapter(DomainAdapter):
    """Standard implementation of domain adapter with configurable transformations."""
    
    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        operation_mappings: Dict[str, str] = None,
        field_mappings: Dict[str, List[FieldMapping]] = None
    ):
        super().__init__(source_domain, target_domain)
        self.operation_mappings = operation_mappings or {}
        self.field_mappings = field_mappings or {}
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return list(self.operation_mappings.keys())
    
    async def adapt_request(
        self,
        operation: str,
        payload: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt request payload."""
        try:
            # Map operation name if needed
            target_operation = self.operation_mappings.get(operation, operation)
            
            # Apply field mappings if available
            field_maps = self.field_mappings.get(operation, [])
            if field_maps:
                adapted_payload = {}
                for mapping in field_maps:
                    source_value = payload.get(mapping.source_field)
                    
                    if source_value is None:
                        if mapping.required and mapping.default_value is None:
                            raise AdapterError(f"Required field missing: {mapping.source_field}")
                        source_value = mapping.default_value
                    
                    if mapping.transform_function:
                        source_value = mapping.transform_function(source_value)
                    
                    adapted_payload[mapping.target_field] = source_value
                
                # Include any unmapped fields
                for key, value in payload.items():
                    if key not in [m.source_field for m in field_maps]:
                        adapted_payload[key] = value
                
                return adapted_payload
            
            return payload
            
        except Exception as e:
            logger.error("Request adaptation failed",
                        source_domain=self.source_domain,
                        target_domain=self.target_domain,
                        operation=operation,
                        error=str(e))
            raise AdapterError(f"Request adaptation failed: {e}") from e
    
    async def adapt_response(
        self,
        operation: str,
        response: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt response payload."""
        try:
            # For standard adapter, response adaptation is symmetric to request
            # This can be overridden for more complex scenarios
            response_field_maps = self.field_mappings.get(f"{operation}_response", [])
            
            if response_field_maps:
                adapted_response = {}
                for mapping in response_field_maps:
                    source_value = response.get(mapping.source_field)
                    
                    if source_value is not None:
                        if mapping.transform_function:
                            source_value = mapping.transform_function(source_value)
                        adapted_response[mapping.target_field] = source_value
                
                # Include unmapped fields
                for key, value in response.items():
                    if key not in [m.source_field for m in response_field_maps]:
                        adapted_response[key] = value
                
                return adapted_response
            
            return response
            
        except Exception as e:
            logger.error("Response adaptation failed",
                        source_domain=self.source_domain,
                        target_domain=self.target_domain,
                        operation=operation,
                        error=str(e))
            raise AdapterError(f"Response adaptation failed: {e}") from e


class EnrichmentAdapter(DomainAdapter):
    """Adapter that enriches data during transformation."""
    
    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        enrichment_functions: Dict[str, Callable[[Dict[str, Any], IntegrationContext], Dict[str, Any]]] = None
    ):
        super().__init__(source_domain, target_domain)
        self.enrichment_functions = enrichment_functions or {}
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return list(self.enrichment_functions.keys())
    
    async def adapt_request(
        self,
        operation: str,
        payload: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt and enrich request."""
        try:
            enrichment_fn = self.enrichment_functions.get(operation)
            if enrichment_fn:
                enriched_payload = enrichment_fn(payload.copy(), context)
                return enriched_payload
            
            return payload
            
        except Exception as e:
            logger.error("Request enrichment failed",
                        source_domain=self.source_domain,
                        target_domain=self.target_domain,
                        operation=operation,
                        error=str(e))
            raise AdapterError(f"Request enrichment failed: {e}") from e
    
    async def adapt_response(
        self,
        operation: str,
        response: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt response (no enrichment by default)."""
        return response


class AggregationAdapter(DomainAdapter):
    """Adapter that aggregates data from multiple sources."""
    
    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        aggregation_configs: Dict[str, Dict[str, Any]] = None
    ):
        super().__init__(source_domain, target_domain)
        self.aggregation_configs = aggregation_configs or {}
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return list(self.aggregation_configs.keys())
    
    async def adapt_request(
        self,
        operation: str,
        payload: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt request for aggregation."""
        config = self.aggregation_configs.get(operation, {})
        
        # Add aggregation metadata
        adapted_payload = payload.copy()
        adapted_payload['_aggregation_config'] = config
        adapted_payload['_context'] = asdict(context)
        
        return adapted_payload
    
    async def adapt_response(
        self,
        operation: str,
        response: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt aggregated response."""
        try:
            config = self.aggregation_configs.get(operation, {})
            aggregation_type = config.get('type', 'merge')
            
            if aggregation_type == 'merge':
                return self._merge_response(response, config)
            elif aggregation_type == 'sum':
                return self._sum_response(response, config)
            elif aggregation_type == 'average':
                return self._average_response(response, config)
            else:
                return response
                
        except Exception as e:
            logger.error("Response aggregation failed",
                        operation=operation,
                        error=str(e))
            raise AdapterError(f"Response aggregation failed: {e}") from e
    
    def _merge_response(self, response: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple response objects."""
        if isinstance(response.get('data'), list):
            merged_data = {}
            for item in response['data']:
                if isinstance(item, dict):
                    merged_data.update(item)
            return {'data': merged_data, 'metadata': response.get('metadata', {})}
        return response
    
    def _sum_response(self, response: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Sum numeric fields in response."""
        sum_fields = config.get('sum_fields', [])
        if isinstance(response.get('data'), list) and sum_fields:
            result = {}
            for field in sum_fields:
                result[field] = sum(
                    item.get(field, 0) for item in response['data'] 
                    if isinstance(item.get(field), (int, float))
                )
            return {'data': result, 'metadata': response.get('metadata', {})}
        return response
    
    def _average_response(self, response: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Average numeric fields in response."""
        avg_fields = config.get('avg_fields', [])
        if isinstance(response.get('data'), list) and avg_fields:
            result = {}
            data_count = len(response['data'])
            for field in avg_fields:
                total = sum(
                    item.get(field, 0) for item in response['data']
                    if isinstance(item.get(field), (int, float))
                )
                result[field] = total / data_count if data_count > 0 else 0
            return {'data': result, 'metadata': response.get('metadata', {})}
        return response


class CachingAdapter(DomainAdapter):
    """Adapter with caching capabilities."""
    
    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        cache_ttl_seconds: int = 300,
        cacheable_operations: List[str] = None
    ):
        super().__init__(source_domain, target_domain)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cacheable_operations = set(cacheable_operations or [])
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return list(self.cacheable_operations)
    
    async def adapt_request(
        self,
        operation: str,
        payload: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt request with cache check."""
        if operation in self.cacheable_operations:
            cache_key = self._generate_cache_key(operation, payload, context)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                # Mark as cached
                adapted_payload = payload.copy()
                adapted_payload['_cached_response'] = cached_response
                return adapted_payload
        
        return payload
    
    async def adapt_response(
        self,
        operation: str,
        response: Dict[str, Any],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """Adapt response with caching."""
        if operation in self.cacheable_operations:
            # Check if this was a cached response
            if '_cached_response' not in response:
                # Cache the response
                cache_key = self._generate_cache_key(operation, response, context)
                self._cache_response(cache_key, response)
        
        return response
    
    def _generate_cache_key(
        self,
        operation: str,
        data: Dict[str, Any],
        context: IntegrationContext
    ) -> str:
        """Generate cache key."""
        import hashlib
        key_data = f"{operation}:{json.dumps(data, sort_keys=True)}:{context.user_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        if cache_key not in self._cache:
            return None
        
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return None
        
        # Check if cache is still valid
        if (datetime.now(timezone.utc) - cache_time).total_seconds() > self.cache_ttl_seconds:
            # Cache expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        return self._cache[cache_key]
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response."""
        self._cache[cache_key] = response.copy()
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        self._cache_timestamps.clear()


class DomainAdapterRegistry:
    """Registry for managing domain adapters."""
    
    def __init__(self):
        self._adapters: Dict[str, DomainAdapter] = {}
        self.logger = structlog.get_logger()
    
    def register_adapter(self, adapter: DomainAdapter) -> None:
        """Register a domain adapter."""
        adapter_key = f"{adapter.source_domain}->{adapter.target_domain}"
        self._adapters[adapter_key] = adapter
        
        self.logger.info(
            "Domain adapter registered",
            source_domain=adapter.source_domain,
            target_domain=adapter.target_domain,
            supported_operations=adapter.get_supported_operations()
        )
    
    def get_adapter(self, source_domain: str, target_domain: str) -> Optional[DomainAdapter]:
        """Get adapter for domain pair."""
        adapter_key = f"{source_domain}->{target_domain}"
        return self._adapters.get(adapter_key)
    
    async def adapt_cross_domain_message(
        self,
        message: CrossDomainMessage,
        adaptation_type: str = "request"
    ) -> CrossDomainMessage:
        """Adapt cross-domain message using registered adapters."""
        adapter = self.get_adapter(message.source_domain, message.target_domain)
        if not adapter:
            # No adapter found, return message unchanged
            return message
        
        try:
            if adaptation_type == "request":
                adapted_payload = await adapter.adapt_request(
                    message.operation,
                    message.payload,
                    message.context
                )
            else:  # response
                adapted_payload = await adapter.adapt_response(
                    message.operation,
                    message.payload,
                    message.context
                )
            
            # Create new message with adapted payload
            adapted_message = CrossDomainMessage(
                message_id=message.message_id,
                message_type=message.message_type,
                priority=message.priority,
                source_domain=message.source_domain,
                target_domain=message.target_domain,
                operation=message.operation,
                payload=adapted_payload,
                context=message.context,
                created_at=message.created_at,
                expires_at=message.expires_at,
                retry_count=message.retry_count,
                max_retries=message.max_retries
            )
            
            return adapted_message
            
        except Exception as e:
            self.logger.error("Message adaptation failed",
                            source_domain=message.source_domain,
                            target_domain=message.target_domain,
                            operation=message.operation,
                            adaptation_type=adaptation_type,
                            error=str(e))
            raise AdapterError(f"Message adaptation failed: {e}") from e
    
    def get_registered_adapters(self) -> List[Dict[str, str]]:
        """Get list of registered adapters."""
        return [
            {
                "source_domain": adapter.source_domain,
                "target_domain": adapter.target_domain,
                "operations": adapter.get_supported_operations()
            }
            for adapter in self._adapters.values()
        ]


# Global adapter registry instance
_adapter_registry: Optional[DomainAdapterRegistry] = None


def get_adapter_registry() -> DomainAdapterRegistry:
    """Get global adapter registry instance."""
    global _adapter_registry
    if _adapter_registry is None:
        _adapter_registry = DomainAdapterRegistry()
    return _adapter_registry


# Convenience functions
def register_standard_adapter(
    source_domain: str,
    target_domain: str,
    operation_mappings: Dict[str, str] = None,
    field_mappings: Dict[str, List[FieldMapping]] = None
) -> StandardDomainAdapter:
    """Register a standard domain adapter."""
    adapter = StandardDomainAdapter(
        source_domain=source_domain,
        target_domain=target_domain,
        operation_mappings=operation_mappings,
        field_mappings=field_mappings
    )
    
    registry = get_adapter_registry()
    registry.register_adapter(adapter)
    
    return adapter


def register_enrichment_adapter(
    source_domain: str,
    target_domain: str,
    enrichment_functions: Dict[str, Callable] = None
) -> EnrichmentAdapter:
    """Register an enrichment adapter."""
    adapter = EnrichmentAdapter(
        source_domain=source_domain,
        target_domain=target_domain,
        enrichment_functions=enrichment_functions
    )
    
    registry = get_adapter_registry()
    registry.register_adapter(adapter)
    
    return adapter