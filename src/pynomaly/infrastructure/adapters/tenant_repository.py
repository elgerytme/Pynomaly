"""Repository implementation for tenant persistence."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from pynomaly.domain.entities.tenant import (
    ResourceQuota,
    ResourceQuotaType,
    SubscriptionTier,
    Tenant,
    TenantConfiguration,
    TenantStatus,
)
from pynomaly.shared.protocols.repository import Repository

Base = declarative_base()


class TenantModel(Base):
    """SQLAlchemy model for tenant persistence."""

    __tablename__ = "tenants"

    # Primary fields
    tenant_id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), nullable=False, index=True)
    subscription_tier = Column(String(50), nullable=False, index=True)

    # Contact information
    contact_email = Column(String(255), nullable=False)
    admin_user_id = Column(PG_UUID(as_uuid=True), nullable=True)
    billing_contact = Column(String(255))

    # Timestamps
    created_at = Column(DateTime, nullable=False, index=True)
    activated_at = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, nullable=True, index=True)
    updated_at = Column(DateTime, nullable=False)

    # Isolation and security
    database_schema = Column(String(255), nullable=False)
    encryption_key_id = Column(String(255), nullable=False)
    network_isolation_config = Column(JSONB)

    # Usage tracking
    total_api_requests = Column(Integer, default=0)
    total_cpu_hours = Column(Float, default=0.0)
    total_storage_gb = Column(Float, default=0.0)
    last_billing_date = Column(DateTime, nullable=True)

    # Configuration and metadata
    configuration = Column(JSONB)
    resource_quotas = Column(JSONB)
    tags = Column(JSONB)
    metadata = Column(JSONB)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_tenant_status_tier", "status", "subscription_tier"),
        Index("idx_tenant_created_at", "created_at"),
        Index("idx_tenant_last_activity", "last_activity"),
    )


class DatabaseTenantRepository(Repository[Tenant]):
    """Database repository implementation for tenants."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, tenant: Tenant) -> Tenant:
        """Save tenant to database."""
        tenant_model = self._to_model(tenant)

        existing = (
            self.session.query(TenantModel)
            .filter_by(tenant_id=tenant.tenant_id)
            .first()
        )
        if existing:
            # Update existing
            for key, value in tenant_model.__dict__.items():
                if not key.startswith("_"):
                    setattr(existing, key, value)
            self.session.commit()
        else:
            # Create new
            self.session.add(tenant_model)
            self.session.commit()

        return tenant

    def get_by_id(self, tenant_id: UUID) -> Tenant | None:
        """Get tenant by ID."""
        tenant_model = (
            self.session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
        )
        if tenant_model:
            return self._from_model(tenant_model)
        return None

    def get_by_name(self, name: str) -> Tenant | None:
        """Get tenant by name."""
        tenant_model = self.session.query(TenantModel).filter_by(name=name).first()
        if tenant_model:
            return self._from_model(tenant_model)
        return None

    def list_all(self) -> list[Tenant]:
        """List all tenants."""
        tenant_models = self.session.query(TenantModel).all()
        return [self._from_model(model) for model in tenant_models]

    def list_by_status(self, status: TenantStatus) -> list[Tenant]:
        """List tenants by status."""
        tenant_models = (
            self.session.query(TenantModel).filter_by(status=status.value).all()
        )
        return [self._from_model(model) for model in tenant_models]

    def list_by_subscription_tier(self, tier: SubscriptionTier) -> list[Tenant]:
        """List tenants by subscription tier."""
        tenant_models = (
            self.session.query(TenantModel)
            .filter_by(subscription_tier=tier.value)
            .all()
        )
        return [self._from_model(model) for model in tenant_models]

    def search_tenants(
        self,
        status: TenantStatus | None = None,
        subscription_tier: SubscriptionTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """Search tenants with filters."""
        query = self.session.query(TenantModel)

        if status:
            query = query.filter(TenantModel.status == status.value)

        if subscription_tier:
            query = query.filter(
                TenantModel.subscription_tier == subscription_tier.value
            )

        query = query.order_by(TenantModel.created_at.desc())
        query = query.offset(offset).limit(limit)

        tenant_models = query.all()
        return [self._from_model(model) for model in tenant_models]

    def delete(self, tenant_id: UUID) -> bool:
        """Delete tenant by ID."""
        result = self.session.query(TenantModel).filter_by(tenant_id=tenant_id).delete()
        self.session.commit()
        return result > 0

    def exists(self, tenant_id: UUID) -> bool:
        """Check if tenant exists."""
        return (
            self.session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
            is not None
        )

    def count_by_status(self) -> dict[str, int]:
        """Count tenants by status."""
        from sqlalchemy import func

        results = (
            self.session.query(TenantModel.status, func.count(TenantModel.tenant_id))
            .group_by(TenantModel.status)
            .all()
        )

        return dict(results)

    def count_by_subscription_tier(self) -> dict[str, int]:
        """Count tenants by subscription tier."""
        from sqlalchemy import func

        results = (
            self.session.query(
                TenantModel.subscription_tier, func.count(TenantModel.tenant_id)
            )
            .group_by(TenantModel.subscription_tier)
            .all()
        )

        return dict(results)

    def get_tenants_needing_quota_reset(self) -> list[Tenant]:
        """Get tenants that need quota reset for new billing period."""
        # Get tenants whose last billing date is more than 30 days ago
        from sqlalchemy import or_

        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        tenant_models = (
            self.session.query(TenantModel)
            .filter(
                or_(
                    TenantModel.last_billing_date.is_(None),
                    TenantModel.last_billing_date < thirty_days_ago,
                )
            )
            .filter(TenantModel.status == TenantStatus.ACTIVE.value)
            .all()
        )

        return [self._from_model(model) for model in tenant_models]

    def update_last_activity(self, tenant_id: UUID, activity_time: datetime) -> bool:
        """Update last activity time for a tenant."""
        result = (
            self.session.query(TenantModel)
            .filter_by(tenant_id=tenant_id)
            .update({"last_activity": activity_time, "updated_at": datetime.utcnow()})
        )
        self.session.commit()
        return result > 0

    def _to_model(self, tenant: Tenant) -> TenantModel:
        """Convert tenant entity to database model."""
        return TenantModel(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            display_name=tenant.display_name,
            description=tenant.description,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            contact_email=tenant.contact_email,
            admin_user_id=tenant.admin_user_id,
            billing_contact=tenant.billing_contact,
            created_at=tenant.created_at,
            activated_at=tenant.activated_at,
            last_activity=tenant.last_activity,
            updated_at=tenant.updated_at,
            database_schema=tenant.database_schema,
            encryption_key_id=tenant.encryption_key_id,
            network_isolation_config=tenant.network_isolation_config,
            total_api_requests=tenant.total_api_requests,
            total_cpu_hours=tenant.total_cpu_hours,
            total_storage_gb=tenant.total_storage_gb,
            last_billing_date=tenant.last_billing_date,
            configuration=self._serialize_configuration(tenant.configuration),
            resource_quotas=self._serialize_quotas(tenant.resource_quotas),
            tags=tenant.tags,
            metadata=tenant.metadata,
        )

    def _from_model(self, model: TenantModel) -> Tenant:
        """Convert database model to tenant entity."""
        tenant = Tenant(
            tenant_id=model.tenant_id,
            name=model.name,
            display_name=model.display_name,
            description=model.description or "",
            status=TenantStatus(model.status),
            subscription_tier=SubscriptionTier(model.subscription_tier),
            contact_email=model.contact_email,
            admin_user_id=model.admin_user_id,
            billing_contact=model.billing_contact or "",
            created_at=model.created_at,
            activated_at=model.activated_at,
            last_activity=model.last_activity,
            updated_at=model.updated_at,
            database_schema=model.database_schema,
            encryption_key_id=model.encryption_key_id,
            network_isolation_config=model.network_isolation_config or {},
            total_api_requests=model.total_api_requests,
            total_cpu_hours=model.total_cpu_hours,
            total_storage_gb=model.total_storage_gb,
            last_billing_date=model.last_billing_date,
            tags=model.tags or {},
            metadata=model.metadata or {},
        )

        # Deserialize configuration
        if model.configuration:
            tenant.configuration = self._deserialize_configuration(model.configuration)

        # Deserialize quotas
        if model.resource_quotas:
            tenant.resource_quotas = self._deserialize_quotas(model.resource_quotas)

        return tenant

    def _serialize_configuration(self, config: TenantConfiguration) -> dict:
        """Serialize tenant configuration to JSON."""
        return {
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "max_model_size_mb": config.max_model_size_mb,
            "allowed_algorithms": list(config.allowed_algorithms),
            "allowed_data_formats": list(config.allowed_data_formats),
            "enable_auto_scaling": config.enable_auto_scaling,
            "enable_gpu_access": config.enable_gpu_access,
            "enable_advanced_analytics": config.enable_advanced_analytics,
            "data_retention_days": config.data_retention_days,
            "backup_enabled": config.backup_enabled,
            "monitoring_level": config.monitoring_level,
            "custom_settings": config.custom_settings,
        }

    def _deserialize_configuration(self, data: dict) -> TenantConfiguration:
        """Deserialize tenant configuration from JSON."""
        return TenantConfiguration(
            max_concurrent_jobs=data.get("max_concurrent_jobs", 5),
            max_model_size_mb=data.get("max_model_size_mb", 1000),
            allowed_algorithms=set(data.get("allowed_algorithms", [])),
            allowed_data_formats=set(
                data.get("allowed_data_formats", ["csv", "json", "parquet"])
            ),
            enable_auto_scaling=data.get("enable_auto_scaling", True),
            enable_gpu_access=data.get("enable_gpu_access", False),
            enable_advanced_analytics=data.get("enable_advanced_analytics", True),
            data_retention_days=data.get("data_retention_days", 365),
            backup_enabled=data.get("backup_enabled", True),
            monitoring_level=data.get("monitoring_level", "standard"),
            custom_settings=data.get("custom_settings", {}),
        )

    def _serialize_quotas(self, quotas: dict[ResourceQuotaType, ResourceQuota]) -> dict:
        """Serialize resource quotas to JSON."""
        return {
            quota_type.value: {
                "quota_type": quota_type.value,
                "limit": quota.limit,
                "used": quota.used,
                "period_start": quota.period_start.isoformat(),
                "period_end": (
                    quota.period_end.isoformat() if quota.period_end else None
                ),
                "is_unlimited": quota.is_unlimited,
            }
            for quota_type, quota in quotas.items()
        }

    def _deserialize_quotas(self, data: dict) -> dict[ResourceQuotaType, ResourceQuota]:
        """Deserialize resource quotas from JSON."""
        quotas = {}

        for quota_type_str, quota_data in data.items():
            try:
                quota_type = ResourceQuotaType(quota_type_str)
                quota = ResourceQuota(
                    quota_type=quota_type,
                    limit=quota_data["limit"],
                    used=quota_data["used"],
                    period_start=datetime.fromisoformat(quota_data["period_start"]),
                    period_end=(
                        datetime.fromisoformat(quota_data["period_end"])
                        if quota_data.get("period_end")
                        else None
                    ),
                    is_unlimited=quota_data.get("is_unlimited", False),
                )
                quotas[quota_type] = quota
            except (ValueError, KeyError):
                # Skip invalid quota data
                continue

        return quotas


class InMemoryTenantRepository(Repository[Tenant]):
    """In-memory repository implementation for tenants (for testing/development)."""

    def __init__(self):
        self._tenants: dict[UUID, Tenant] = {}
        self._tenants_by_name: dict[str, UUID] = {}

    def save(self, tenant: Tenant) -> Tenant:
        """Save tenant to memory."""
        tenant.updated_at = datetime.utcnow()
        self._tenants[tenant.tenant_id] = tenant
        self._tenants_by_name[tenant.name] = tenant.tenant_id
        return tenant

    def get_by_id(self, tenant_id: UUID) -> Tenant | None:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_by_name(self, name: str) -> Tenant | None:
        """Get tenant by name."""
        tenant_id = self._tenants_by_name.get(name)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None

    def list_all(self) -> list[Tenant]:
        """List all tenants."""
        return list(self._tenants.values())

    def list_by_status(self, status: TenantStatus) -> list[Tenant]:
        """List tenants by status."""
        return [tenant for tenant in self._tenants.values() if tenant.status == status]

    def list_by_subscription_tier(self, tier: SubscriptionTier) -> list[Tenant]:
        """List tenants by subscription tier."""
        return [
            tenant
            for tenant in self._tenants.values()
            if tenant.subscription_tier == tier
        ]

    def search_tenants(
        self,
        status: TenantStatus | None = None,
        subscription_tier: SubscriptionTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """Search tenants with filters."""
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        if subscription_tier:
            tenants = [t for t in tenants if t.subscription_tier == subscription_tier]

        # Sort by creation date (newest first)
        tenants.sort(key=lambda t: t.created_at, reverse=True)

        return tenants[offset : offset + limit]

    def delete(self, tenant_id: UUID) -> bool:
        """Delete tenant by ID."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            del self._tenants[tenant_id]
            self._tenants_by_name.pop(tenant.name, None)
            return True
        return False

    def exists(self, tenant_id: UUID) -> bool:
        """Check if tenant exists."""
        return tenant_id in self._tenants

    def count_by_status(self) -> dict[str, int]:
        """Count tenants by status."""
        counts = {}
        for tenant in self._tenants.values():
            status = tenant.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts

    def count_by_subscription_tier(self) -> dict[str, int]:
        """Count tenants by subscription tier."""
        counts = {}
        for tenant in self._tenants.values():
            tier = tenant.subscription_tier.value
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def update_last_activity(self, tenant_id: UUID, activity_time: datetime) -> bool:
        """Update last activity time for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.last_activity = activity_time
            tenant.updated_at = datetime.utcnow()
            return True
        return False
