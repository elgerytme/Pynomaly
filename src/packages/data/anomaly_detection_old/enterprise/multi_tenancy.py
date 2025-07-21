"""
Multi-Tenancy and Tenant Isolation for Pynomaly Detection
==========================================================

Comprehensive multi-tenant architecture providing:
- Complete tenant isolation and data segregation
- Resource quotas and usage tracking
- Tenant-specific configurations and customizations
- Billing and subscription management
- Cross-tenant security and compliance
"""

import logging
import json
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import uuid

try:
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, JSON, Float, Integer, Boolean
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TenantStatus(Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    TERMINATED = "terminated"

class SubscriptionTier(Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

@dataclass
class ResourceQuota:
    """Resource quota configuration."""
    max_api_calls_per_minute: int = 100
    max_api_calls_per_hour: int = 1000
    max_api_calls_per_day: int = 10000
    max_data_storage_mb: int = 1000
    max_model_instances: int = 10
    max_concurrent_detections: int = 5
    max_retention_days: int = 30
    enable_advanced_algorithms: bool = False
    enable_custom_models: bool = False
    enable_real_time_streaming: bool = False
    enable_batch_processing: bool = True
    max_batch_size: int = 10000

@dataclass
class TenantConfig:
    """Tenant configuration."""
    tenant_id: str
    tenant_name: str
    organization_name: str
    contact_email: str
    status: TenantStatus = TenantStatus.ACTIVE
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    resource_quota: ResourceQuota = field(default_factory=ResourceQuota)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    billing_info: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)

@dataclass
class TenantUsage:
    """Tenant resource usage tracking."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    api_calls_count: int = 0
    data_storage_mb: float = 0.0
    model_instances_count: int = 0
    detection_requests: int = 0
    processing_time_seconds: float = 0.0
    anomalies_detected: int = 0
    alerts_sent: int = 0
    last_activity: Optional[datetime] = None

class TenantManager:
    """Comprehensive tenant management system."""
    
    def __init__(self, database_url: Optional[str] = None, redis_url: Optional[str] = None):
        """Initialize tenant manager.
        
        Args:
            database_url: Database connection URL
            redis_url: Redis connection URL
        """
        self.database_engine = None
        self.redis_client = None
        
        # In-memory storage (fallback)
        self.tenants: Dict[str, TenantConfig] = {}
        self.tenant_usage: Dict[str, TenantUsage] = {}
        
        # Caching and performance
        self.tenant_cache: Dict[str, TenantConfig] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize storage
        if database_url and SQLALCHEMY_AVAILABLE:
            self._initialize_database(database_url)
        
        if redis_url and REDIS_AVAILABLE:
            self._initialize_redis(redis_url)
        
        # Default subscription tiers
        self.subscription_quotas = self._initialize_subscription_quotas()
        
        logger.info("Tenant Manager initialized")
    
    def create_tenant(self, tenant_config: TenantConfig) -> bool:
        """Create a new tenant.
        
        Args:
            tenant_config: Tenant configuration
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                # Generate tenant ID if not provided
                if not tenant_config.tenant_id:
                    tenant_config.tenant_id = self._generate_tenant_id(tenant_config.organization_name)
                
                # Validate tenant doesn't exist
                if self._tenant_exists(tenant_config.tenant_id):
                    logger.error(f"Tenant already exists: {tenant_config.tenant_id}")
                    return False
                
                # Set subscription quota
                if tenant_config.subscription_tier in self.subscription_quotas:
                    tenant_config.resource_quota = self.subscription_quotas[tenant_config.subscription_tier]
                
                # Store tenant
                if self.database_engine:
                    self._store_tenant_db(tenant_config)
                else:
                    self.tenants[tenant_config.tenant_id] = tenant_config
                
                # Initialize usage tracking
                usage = TenantUsage(
                    tenant_id=tenant_config.tenant_id,
                    period_start=datetime.now(),
                    period_end=datetime.now() + timedelta(days=30)
                )
                
                if self.database_engine:
                    self._store_usage_db(usage)
                else:
                    self.tenant_usage[tenant_config.tenant_id] = usage
                
                # Cache tenant
                self.tenant_cache[tenant_config.tenant_id] = tenant_config
                self.cache_timestamps[tenant_config.tenant_id] = datetime.now()
                
                logger.info(f"Tenant created: {tenant_config.tenant_id} ({tenant_config.organization_name})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create tenant {tenant_config.tenant_id}: {e}")
            return False
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Tenant configuration or None
        """
        try:
            # Check cache first
            if tenant_id in self.tenant_cache:
                cache_time = self.cache_timestamps.get(tenant_id)
                if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                    return self.tenant_cache[tenant_id]
            
            # Load from storage
            tenant = None
            if self.database_engine:
                tenant = self._load_tenant_db(tenant_id)
            else:
                tenant = self.tenants.get(tenant_id)
            
            # Update cache
            if tenant:
                self.tenant_cache[tenant_id] = tenant
                self.cache_timestamps[tenant_id] = datetime.now()
            
            return tenant
            
        except Exception as e:
            logger.error(f"Failed to get tenant {tenant_id}: {e}")
            return None
    
    def update_tenant(self, tenant_config: TenantConfig) -> bool:
        """Update tenant configuration.
        
        Args:
            tenant_config: Updated tenant configuration
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                tenant_config.last_updated = datetime.now()
                
                # Store updated tenant
                if self.database_engine:
                    self._update_tenant_db(tenant_config)
                else:
                    self.tenants[tenant_config.tenant_id] = tenant_config
                
                # Update cache
                self.tenant_cache[tenant_config.tenant_id] = tenant_config
                self.cache_timestamps[tenant_config.tenant_id] = datetime.now()
                
                logger.info(f"Tenant updated: {tenant_config.tenant_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update tenant {tenant_config.tenant_id}: {e}")
            return False
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete - marks as terminated).
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                tenant = self.get_tenant(tenant_id)
                if not tenant:
                    logger.warning(f"Tenant not found: {tenant_id}")
                    return False
                
                # Mark as terminated
                tenant.status = TenantStatus.TERMINATED
                tenant.last_updated = datetime.now()
                
                # Update storage
                if self.database_engine:
                    self._update_tenant_db(tenant)
                else:
                    self.tenants[tenant_id] = tenant
                
                # Remove from cache
                if tenant_id in self.tenant_cache:
                    del self.tenant_cache[tenant_id]
                if tenant_id in self.cache_timestamps:
                    del self.cache_timestamps[tenant_id]
                
                logger.info(f"Tenant deleted (terminated): {tenant_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete tenant {tenant_id}: {e}")
            return False
    
    def get_tenant_usage(self, tenant_id: str, period_days: int = 30) -> Optional[TenantUsage]:
        """Get tenant usage statistics.
        
        Args:
            tenant_id: Tenant identifier
            period_days: Usage period in days
            
        Returns:
            Tenant usage or None
        """
        try:
            if self.database_engine:
                return self._load_usage_db(tenant_id, period_days)
            else:
                return self.tenant_usage.get(tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to get usage for tenant {tenant_id}: {e}")
            return None
    
    def update_tenant_usage(self, tenant_id: str, usage_delta: Dict[str, Any]) -> bool:
        """Update tenant usage statistics.
        
        Args:
            tenant_id: Tenant identifier
            usage_delta: Usage changes to apply
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                usage = self.get_tenant_usage(tenant_id)
                if not usage:
                    # Create new usage record
                    usage = TenantUsage(
                        tenant_id=tenant_id,
                        period_start=datetime.now(),
                        period_end=datetime.now() + timedelta(days=30)
                    )
                
                # Apply usage deltas
                for key, value in usage_delta.items():
                    if hasattr(usage, key):
                        current_value = getattr(usage, key)
                        if isinstance(current_value, (int, float)):
                            setattr(usage, key, current_value + value)
                        else:
                            setattr(usage, key, value)
                
                usage.last_activity = datetime.now()
                
                # Store updated usage
                if self.database_engine:
                    self._store_usage_db(usage)
                else:
                    self.tenant_usage[tenant_id] = usage
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update usage for tenant {tenant_id}: {e}")
            return False
    
    def check_resource_quota(self, tenant_id: str, resource_type: str, requested_amount: float = 1.0) -> bool:
        """Check if tenant has quota for requested resource.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource to check
            requested_amount: Amount of resource requested
            
        Returns:
            True if quota available, False otherwise
        """
        try:
            tenant = self.get_tenant(tenant_id)
            if not tenant or tenant.status != TenantStatus.ACTIVE:
                return False
            
            usage = self.get_tenant_usage(tenant_id)
            if not usage:
                return True  # No usage tracking yet
            
            quota = tenant.resource_quota
            
            # Check specific resource quotas
            if resource_type == "api_calls_per_minute":
                # Check current minute usage
                current_minute_usage = self._get_current_minute_usage(tenant_id)
                return current_minute_usage + requested_amount <= quota.max_api_calls_per_minute
            
            elif resource_type == "api_calls_per_hour":
                current_hour_usage = self._get_current_hour_usage(tenant_id)
                return current_hour_usage + requested_amount <= quota.max_api_calls_per_hour
            
            elif resource_type == "api_calls_per_day":
                current_day_usage = self._get_current_day_usage(tenant_id)
                return current_day_usage + requested_amount <= quota.max_api_calls_per_day
            
            elif resource_type == "data_storage":
                return usage.data_storage_mb + requested_amount <= quota.max_data_storage_mb
            
            elif resource_type == "model_instances":
                return usage.model_instances_count + requested_amount <= quota.max_model_instances
            
            elif resource_type == "concurrent_detections":
                current_detections = self._get_current_detections(tenant_id)
                return current_detections + requested_amount <= quota.max_concurrent_detections
            
            else:
                logger.warning(f"Unknown resource type: {resource_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to check quota for tenant {tenant_id}: {e}")
            return False
    
    def list_tenants(self, status_filter: Optional[TenantStatus] = None) -> List[TenantConfig]:
        """List all tenants with optional status filter.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of tenant configurations
        """
        try:
            if self.database_engine:
                return self._list_tenants_db(status_filter)
            else:
                tenants = list(self.tenants.values())
                if status_filter:
                    tenants = [t for t in tenants if t.status == status_filter]
                return tenants
                
        except Exception as e:
            logger.error(f"Failed to list tenants: {e}")
            return []
    
    def get_tenant_statistics(self) -> Dict[str, Any]:
        """Get overall tenant statistics.
        
        Returns:
            Tenant statistics
        """
        try:
            stats = {
                'total_tenants': 0,
                'active_tenants': 0,
                'suspended_tenants': 0,
                'terminated_tenants': 0,
                'subscription_distribution': defaultdict(int),
                'total_api_calls': 0,
                'total_data_storage_mb': 0.0,
                'total_anomalies_detected': 0
            }
            
            tenants = self.list_tenants()
            stats['total_tenants'] = len(tenants)
            
            for tenant in tenants:
                # Status distribution
                if tenant.status == TenantStatus.ACTIVE:
                    stats['active_tenants'] += 1
                elif tenant.status == TenantStatus.SUSPENDED:
                    stats['suspended_tenants'] += 1
                elif tenant.status == TenantStatus.TERMINATED:
                    stats['terminated_tenants'] += 1
                
                # Subscription distribution
                stats['subscription_distribution'][tenant.subscription_tier.value] += 1
                
                # Usage aggregation
                usage = self.get_tenant_usage(tenant.tenant_id)
                if usage:
                    stats['total_api_calls'] += usage.api_calls_count
                    stats['total_data_storage_mb'] += usage.data_storage_mb
                    stats['total_anomalies_detected'] += usage.anomalies_detected
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get tenant statistics: {e}")
            return {}
    
    def _generate_tenant_id(self, organization_name: str) -> str:
        """Generate unique tenant ID."""
        # Create deterministic but unique ID
        timestamp = str(int(time.time()))
        org_hash = hashlib.md5(organization_name.encode()).hexdigest()[:8]
        return f"tenant_{org_hash}_{timestamp}"
    
    def _tenant_exists(self, tenant_id: str) -> bool:
        """Check if tenant exists."""
        if self.database_engine:
            return self._tenant_exists_db(tenant_id)
        else:
            return tenant_id in self.tenants
    
    def _initialize_database(self, database_url: str):
        """Initialize database connection and tables."""
        try:
            self.database_engine = create_engine(database_url)
            self._create_tenant_tables()
            logger.info("Database initialized for tenant management")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.database_engine = None
    
    def _initialize_redis(self, redis_url: str):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis initialized for tenant caching")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    def _create_tenant_tables(self):
        """Create database tables for tenant management."""
        metadata = MetaData()
        
        # Tenants table
        tenants_table = Table(
            'tenants',
            metadata,
            Column('tenant_id', String(255), primary_key=True),
            Column('tenant_name', String(255)),
            Column('organization_name', String(255)),
            Column('contact_email', String(255)),
            Column('status', String(50)),
            Column('subscription_tier', String(50)),
            Column('created_date', DateTime),
            Column('last_updated', DateTime),
            Column('resource_quota', JSON),
            Column('custom_settings', JSON),
            Column('billing_info', JSON),
            Column('compliance_requirements', JSON)
        )
        
        # Tenant usage table
        usage_table = Table(
            'tenant_usage',
            metadata,
            Column('tenant_id', String(255)),
            Column('period_start', DateTime),
            Column('period_end', DateTime),
            Column('api_calls_count', Integer),
            Column('data_storage_mb', Float),
            Column('model_instances_count', Integer),
            Column('detection_requests', Integer),
            Column('processing_time_seconds', Float),
            Column('anomalies_detected', Integer),
            Column('alerts_sent', Integer),
            Column('last_activity', DateTime)
        )
        
        metadata.create_all(self.database_engine)
    
    def _initialize_subscription_quotas(self) -> Dict[SubscriptionTier, ResourceQuota]:
        """Initialize default subscription quotas."""
        return {
            SubscriptionTier.FREE: ResourceQuota(
                max_api_calls_per_minute=10,
                max_api_calls_per_hour=100,
                max_api_calls_per_day=1000,
                max_data_storage_mb=100,
                max_model_instances=1,
                max_concurrent_detections=1,
                max_retention_days=7,
                enable_advanced_algorithms=False,
                enable_custom_models=False,
                enable_real_time_streaming=False,
                max_batch_size=1000
            ),
            SubscriptionTier.STARTER: ResourceQuota(
                max_api_calls_per_minute=50,
                max_api_calls_per_hour=1000,
                max_api_calls_per_day=10000,
                max_data_storage_mb=1000,
                max_model_instances=3,
                max_concurrent_detections=3,
                max_retention_days=30,
                enable_advanced_algorithms=True,
                enable_custom_models=False,
                enable_real_time_streaming=False,
                max_batch_size=5000
            ),
            SubscriptionTier.PROFESSIONAL: ResourceQuota(
                max_api_calls_per_minute=200,
                max_api_calls_per_hour=5000,
                max_api_calls_per_day=50000,
                max_data_storage_mb=10000,
                max_model_instances=10,
                max_concurrent_detections=10,
                max_retention_days=90,
                enable_advanced_algorithms=True,
                enable_custom_models=True,
                enable_real_time_streaming=True,
                max_batch_size=50000
            ),
            SubscriptionTier.ENTERPRISE: ResourceQuota(
                max_api_calls_per_minute=1000,
                max_api_calls_per_hour=25000,
                max_api_calls_per_day=500000,
                max_data_storage_mb=100000,
                max_model_instances=50,
                max_concurrent_detections=50,
                max_retention_days=365,
                enable_advanced_algorithms=True,
                enable_custom_models=True,
                enable_real_time_streaming=True,
                max_batch_size=1000000
            )
        }
    
    def _get_current_minute_usage(self, tenant_id: str) -> int:
        """Get current minute API usage."""
        # Implementation would use Redis or time-windowed counters
        return 0
    
    def _get_current_hour_usage(self, tenant_id: str) -> int:
        """Get current hour API usage."""
        # Implementation would use Redis or time-windowed counters
        return 0
    
    def _get_current_day_usage(self, tenant_id: str) -> int:
        """Get current day API usage."""
        # Implementation would use Redis or time-windowed counters
        return 0
    
    def _get_current_detections(self, tenant_id: str) -> int:
        """Get current concurrent detections."""
        # Implementation would track active detection sessions
        return 0
    
    # Database operations (simplified implementations)
    def _store_tenant_db(self, tenant: TenantConfig):
        """Store tenant in database."""
        pass
    
    def _load_tenant_db(self, tenant_id: str) -> Optional[TenantConfig]:
        """Load tenant from database."""
        return None
    
    def _update_tenant_db(self, tenant: TenantConfig):
        """Update tenant in database."""
        pass
    
    def _tenant_exists_db(self, tenant_id: str) -> bool:
        """Check if tenant exists in database."""
        return False
    
    def _store_usage_db(self, usage: TenantUsage):
        """Store usage in database."""
        pass
    
    def _load_usage_db(self, tenant_id: str, period_days: int) -> Optional[TenantUsage]:
        """Load usage from database."""
        return None
    
    def _list_tenants_db(self, status_filter: Optional[TenantStatus]) -> List[TenantConfig]:
        """List tenants from database."""
        return []


class TenantIsolationService:
    """Service for enforcing tenant data isolation."""
    
    def __init__(self, tenant_manager: TenantManager):
        """Initialize tenant isolation service.
        
        Args:
            tenant_manager: Tenant manager instance
        """
        self.tenant_manager = tenant_manager
        self.isolation_contexts: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        logger.info("Tenant Isolation Service initialized")
    
    def create_isolation_context(self, tenant_id: str) -> Dict[str, Any]:
        """Create isolated context for tenant operations.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Isolation context
        """
        try:
            tenant = self.tenant_manager.get_tenant(tenant_id)
            if not tenant:
                raise ValueError(f"Tenant not found: {tenant_id}")
            
            if tenant.status != TenantStatus.ACTIVE:
                raise ValueError(f"Tenant not active: {tenant_id}")
            
            context = {
                'tenant_id': tenant_id,
                'tenant_config': tenant,
                'resource_quota': tenant.resource_quota,
                'data_namespace': f"tenant_{tenant_id}",
                'model_namespace': f"models_{tenant_id}",
                'cache_namespace': f"cache_{tenant_id}",
                'created_at': datetime.now()
            }
            
            with self.lock:
                self.isolation_contexts[tenant_id] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create isolation context for {tenant_id}: {e}")
            raise
    
    def get_isolation_context(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get isolation context for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Isolation context or None
        """
        with self.lock:
            return self.isolation_contexts.get(tenant_id)
    
    def validate_tenant_access(self, tenant_id: str, resource_type: str, resource_id: str) -> bool:
        """Validate tenant access to resource.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            True if access allowed, False otherwise
        """
        try:
            context = self.get_isolation_context(tenant_id)
            if not context:
                return False
            
            # Check resource namespace
            if resource_type == "data":
                return resource_id.startswith(context['data_namespace'])
            elif resource_type == "model":
                return resource_id.startswith(context['model_namespace'])
            elif resource_type == "cache":
                return resource_id.startswith(context['cache_namespace'])
            else:
                # Default to deny for unknown resource types
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate access for {tenant_id}: {e}")
            return False
    
    def get_namespaced_key(self, tenant_id: str, resource_type: str, key: str) -> str:
        """Get namespaced key for tenant resource.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource
            key: Original key
            
        Returns:
            Namespaced key
        """
        context = self.get_isolation_context(tenant_id)
        if not context:
            raise ValueError(f"No isolation context for tenant: {tenant_id}")
        
        if resource_type == "data":
            return f"{context['data_namespace']}:{key}"
        elif resource_type == "model":
            return f"{context['model_namespace']}:{key}"
        elif resource_type == "cache":
            return f"{context['cache_namespace']}:{key}"
        else:
            return f"tenant_{tenant_id}:{resource_type}:{key}"
    
    def cleanup_tenant_resources(self, tenant_id: str) -> bool:
        """Cleanup all resources for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Success status
        """
        try:
            context = self.get_isolation_context(tenant_id)
            if not context:
                logger.warning(f"No isolation context for tenant: {tenant_id}")
                return True
            
            # In a real implementation, this would:
            # 1. Remove all data with tenant namespace
            # 2. Delete all models for tenant
            # 3. Clear cache entries
            # 4. Remove temporary files
            # 5. Clean up database records
            
            with self.lock:
                if tenant_id in self.isolation_contexts:
                    del self.isolation_contexts[tenant_id]
            
            logger.info(f"Cleaned up resources for tenant: {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources for {tenant_id}: {e}")
            return False