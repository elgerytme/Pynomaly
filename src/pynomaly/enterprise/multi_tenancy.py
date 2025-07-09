#!/usr/bin/env python3
"""
Enterprise Multi-Tenancy System for Pynomaly.
This module provides comprehensive multi-tenant capabilities with isolation and resource management.
"""

import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    String,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()


class TenantStatus(Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    PENDING = "pending"
    DELETED = "deleted"


class UserRole(Enum):
    """User role enumeration."""

    TENANT_ADMIN = "tenant_admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class ResourceType(Enum):
    """Resource type enumeration."""

    MODELS = "models"
    EXPERIMENTS = "experiments"
    DEPLOYMENTS = "deployments"
    STORAGE = "storage"
    COMPUTE = "compute"
    API_CALLS = "api_calls"


# Database Models
class Tenant(Base):
    """Tenant database model."""

    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=False)
    domain = Column(String(255), nullable=True)
    status = Column(String(50), nullable=False, default=TenantStatus.ACTIVE.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings = Column(JSON, default=dict)
    resource_limits = Column(JSON, default=dict)
    billing_info = Column(JSON, default=dict)

    # Relationships
    users = relationship("TenantUser", back_populates="tenant")
    resources = relationship("TenantResource", back_populates="tenant")
    audit_logs = relationship("AuditLog", back_populates="tenant")


class TenantUser(Base):
    """Tenant user database model."""

    __tablename__ = "tenant_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    email = Column(String(255), nullable=False)
    username = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    permissions = Column(JSON, default=list)
    preferences = Column(JSON, default=dict)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")


class TenantResource(Base):
    """Tenant resource tracking model."""

    __tablename__ = "tenant_resources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(255), nullable=False)
    resource_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

    # Relationships
    tenant = relationship("Tenant", back_populates="resources")


class AuditLog(Base):
    """Audit log database model."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("tenant_users.id"), nullable=True)
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    request_id = Column(String(100), nullable=True)
    details = Column(JSON, default=dict)
    status = Column(String(50), nullable=False)

    # Relationships
    tenant = relationship("Tenant", back_populates="audit_logs")
    user = relationship("TenantUser", back_populates="audit_logs")


# Pydantic Models
@dataclass
class TenantInfo:
    """Tenant information structure."""

    id: str
    name: str
    display_name: str
    domain: str | None
    status: TenantStatus
    created_at: datetime
    updated_at: datetime
    settings: dict[str, Any]
    resource_limits: dict[str, Any]
    billing_info: dict[str, Any]


@dataclass
class TenantUserInfo:
    """Tenant user information structure."""

    id: str
    tenant_id: str
    email: str
    username: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: datetime | None
    permissions: list[str]
    preferences: dict[str, Any]


class TenantCreateRequest(BaseModel):
    """Tenant creation request."""

    name: str = Field(..., min_length=3, max_length=255)
    display_name: str = Field(..., min_length=3, max_length=255)
    domain: str | None = Field(None, max_length=255)
    admin_email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    admin_username: str = Field(..., min_length=3, max_length=255)
    admin_password: str = Field(..., min_length=8)
    settings: dict[str, Any] | None = Field(default_factory=dict)
    resource_limits: dict[str, Any] | None = Field(default_factory=dict)


class TenantUpdateRequest(BaseModel):
    """Tenant update request."""

    display_name: str | None = Field(None, min_length=3, max_length=255)
    domain: str | None = Field(None, max_length=255)
    status: TenantStatus | None = None
    settings: dict[str, Any] | None = None
    resource_limits: dict[str, Any] | None = None


class UserCreateRequest(BaseModel):
    """User creation request."""

    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    username: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)
    role: UserRole
    permissions: list[str] | None = Field(default_factory=list)


class LoginRequest(BaseModel):
    """Login request."""

    username: str
    password: str
    tenant_name: str | None = None


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant_id: str
    user_id: str
    role: str


class MultiTenantManager:
    """Multi-tenant management system."""

    def __init__(self, database_url: str, jwt_secret: str):
        """Initialize multi-tenant manager."""
        self.database_url = database_url
        self.jwt_secret = jwt_secret
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Create tables
        Base.metadata.create_all(bind=self.engine)

        # Default resource limits
        self.default_resource_limits = {
            ResourceType.MODELS.value: 100,
            ResourceType.EXPERIMENTS.value: 500,
            ResourceType.DEPLOYMENTS.value: 50,
            ResourceType.STORAGE.value: 1024 * 1024 * 1024 * 10,  # 10GB
            ResourceType.COMPUTE.value: 1000,  # hours
            ResourceType.API_CALLS.value: 10000,  # per day
        }

        logger.info("Multi-tenant manager initialized")

    def get_db(self):
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def hash_password(self, password: str) -> str:
        """Hash password."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self, user_data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create JWT access token."""
        to_encode = user_data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm="HS256")
        return encoded_jwt

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def create_tenant(self, request: TenantCreateRequest) -> TenantInfo:
        """Create new tenant."""
        db = next(self.get_db())

        try:
            # Check if tenant already exists
            existing_tenant = (
                db.query(Tenant).filter(Tenant.name == request.name).first()
            )
            if existing_tenant:
                raise HTTPException(
                    status_code=400, detail="Tenant name already exists"
                )

            # Create tenant
            tenant = Tenant(
                name=request.name,
                display_name=request.display_name,
                domain=request.domain,
                status=TenantStatus.ACTIVE.value,
                settings=request.settings,
                resource_limits=request.resource_limits or self.default_resource_limits,
            )

            db.add(tenant)
            db.commit()
            db.refresh(tenant)

            # Create admin user
            admin_user = TenantUser(
                tenant_id=tenant.id,
                email=request.admin_email,
                username=request.admin_username,
                password_hash=self.hash_password(request.admin_password),
                role=UserRole.TENANT_ADMIN.value,
                permissions=["*"],  # Full permissions for admin
            )

            db.add(admin_user)
            db.commit()

            # Create tenant-specific schemas/directories
            await self._setup_tenant_resources(str(tenant.id))

            logger.info(f"✅ Tenant created: {tenant.name} ({tenant.id})")

            return TenantInfo(
                id=str(tenant.id),
                name=tenant.name,
                display_name=tenant.display_name,
                domain=tenant.domain,
                status=TenantStatus(tenant.status),
                created_at=tenant.created_at,
                updated_at=tenant.updated_at,
                settings=tenant.settings,
                resource_limits=tenant.resource_limits,
                billing_info=tenant.billing_info,
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create tenant: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create tenant: {str(e)}"
            )
        finally:
            db.close()

    async def _setup_tenant_resources(self, tenant_id: str):
        """Set up tenant-specific resources."""
        # Create tenant-specific directories
        tenant_dirs = [
            f"tenants/{tenant_id}/models",
            f"tenants/{tenant_id}/experiments",
            f"tenants/{tenant_id}/deployments",
            f"tenants/{tenant_id}/data",
            f"tenants/{tenant_id}/logs",
        ]

        for dir_path in tenant_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ Tenant resources set up: {tenant_id}")

    async def get_tenant(self, tenant_id: str) -> TenantInfo | None:
        """Get tenant by ID."""
        db = next(self.get_db())

        try:
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant:
                return None

            return TenantInfo(
                id=str(tenant.id),
                name=tenant.name,
                display_name=tenant.display_name,
                domain=tenant.domain,
                status=TenantStatus(tenant.status),
                created_at=tenant.created_at,
                updated_at=tenant.updated_at,
                settings=tenant.settings,
                resource_limits=tenant.resource_limits,
                billing_info=tenant.billing_info,
            )

        finally:
            db.close()

    async def update_tenant(
        self, tenant_id: str, request: TenantUpdateRequest
    ) -> TenantInfo:
        """Update tenant."""
        db = next(self.get_db())

        try:
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant:
                raise HTTPException(status_code=404, detail="Tenant not found")

            # Update fields
            if request.display_name:
                tenant.display_name = request.display_name
            if request.domain:
                tenant.domain = request.domain
            if request.status:
                tenant.status = request.status.value
            if request.settings:
                tenant.settings = request.settings
            if request.resource_limits:
                tenant.resource_limits = request.resource_limits

            tenant.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(tenant)

            logger.info(f"✅ Tenant updated: {tenant.name}")

            return TenantInfo(
                id=str(tenant.id),
                name=tenant.name,
                display_name=tenant.display_name,
                domain=tenant.domain,
                status=TenantStatus(tenant.status),
                created_at=tenant.created_at,
                updated_at=tenant.updated_at,
                settings=tenant.settings,
                resource_limits=tenant.resource_limits,
                billing_info=tenant.billing_info,
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update tenant: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to update tenant: {str(e)}"
            )
        finally:
            db.close()

    async def list_tenants(self, skip: int = 0, limit: int = 100) -> list[TenantInfo]:
        """List all tenants."""
        db = next(self.get_db())

        try:
            tenants = db.query(Tenant).offset(skip).limit(limit).all()

            return [
                TenantInfo(
                    id=str(tenant.id),
                    name=tenant.name,
                    display_name=tenant.display_name,
                    domain=tenant.domain,
                    status=TenantStatus(tenant.status),
                    created_at=tenant.created_at,
                    updated_at=tenant.updated_at,
                    settings=tenant.settings,
                    resource_limits=tenant.resource_limits,
                    billing_info=tenant.billing_info,
                )
                for tenant in tenants
            ]

        finally:
            db.close()

    async def create_user(
        self, tenant_id: str, request: UserCreateRequest
    ) -> TenantUserInfo:
        """Create new user for tenant."""
        db = next(self.get_db())

        try:
            # Check if tenant exists
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant:
                raise HTTPException(status_code=404, detail="Tenant not found")

            # Check if user already exists
            existing_user = (
                db.query(TenantUser)
                .filter(
                    TenantUser.tenant_id == tenant_id, TenantUser.email == request.email
                )
                .first()
            )
            if existing_user:
                raise HTTPException(status_code=400, detail="User already exists")

            # Create user
            user = TenantUser(
                tenant_id=tenant_id,
                email=request.email,
                username=request.username,
                password_hash=self.hash_password(request.password),
                role=request.role.value,
                permissions=request.permissions,
            )

            db.add(user)
            db.commit()
            db.refresh(user)

            logger.info(f"✅ User created: {user.email} for tenant {tenant.name}")

            return TenantUserInfo(
                id=str(user.id),
                tenant_id=str(user.tenant_id),
                email=user.email,
                username=user.username,
                role=UserRole(user.role),
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
                permissions=user.permissions,
                preferences=user.preferences,
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create user: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create user: {str(e)}"
            )
        finally:
            db.close()

    async def authenticate_user(self, request: LoginRequest) -> TenantUserInfo | None:
        """Authenticate user."""
        db = next(self.get_db())

        try:
            # Build query
            query = db.query(TenantUser).filter(TenantUser.username == request.username)

            # Filter by tenant if specified
            if request.tenant_name:
                query = query.join(Tenant).filter(Tenant.name == request.tenant_name)

            user = query.first()

            if not user or not self.verify_password(
                request.password, user.password_hash
            ):
                return None

            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()

            return TenantUserInfo(
                id=str(user.id),
                tenant_id=str(user.tenant_id),
                email=user.email,
                username=user.username,
                role=UserRole(user.role),
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
                permissions=user.permissions,
                preferences=user.preferences,
            )

        finally:
            db.close()

    async def get_user(self, user_id: str) -> TenantUserInfo | None:
        """Get user by ID."""
        db = next(self.get_db())

        try:
            user = db.query(TenantUser).filter(TenantUser.id == user_id).first()
            if not user:
                return None

            return TenantUserInfo(
                id=str(user.id),
                tenant_id=str(user.tenant_id),
                email=user.email,
                username=user.username,
                role=UserRole(user.role),
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
                permissions=user.permissions,
                preferences=user.preferences,
            )

        finally:
            db.close()

    async def check_resource_limits(
        self, tenant_id: str, resource_type: ResourceType
    ) -> bool:
        """Check if tenant has reached resource limits."""
        db = next(self.get_db())

        try:
            # Get tenant
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant:
                return False

            # Get current resource count
            current_count = (
                db.query(TenantResource)
                .filter(
                    TenantResource.tenant_id == tenant_id,
                    TenantResource.resource_type == resource_type.value,
                )
                .count()
            )

            # Check limit
            limit = tenant.resource_limits.get(
                resource_type.value, self.default_resource_limits[resource_type.value]
            )

            return current_count < limit

        finally:
            db.close()

    async def track_resource_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        resource_id: str,
        resource_name: str,
        metadata: dict[str, Any] = None,
    ):
        """Track resource usage for tenant."""
        db = next(self.get_db())

        try:
            resource = TenantResource(
                tenant_id=tenant_id,
                resource_type=resource_type.value,
                resource_id=resource_id,
                resource_name=resource_name,
                metadata=metadata or {},
            )

            db.add(resource)
            db.commit()

            logger.info(
                f"✅ Resource tracked: {resource_type.value} for tenant {tenant_id}"
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to track resource: {e}")
        finally:
            db.close()


# Global multi-tenant manager instance
multi_tenant_manager = None


def get_multi_tenant_manager() -> MultiTenantManager:
    """Get multi-tenant manager instance."""
    global multi_tenant_manager
    if multi_tenant_manager is None:
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost/pynomaly"
        )
        jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        multi_tenant_manager = MultiTenantManager(database_url, jwt_secret)
    return multi_tenant_manager


# FastAPI Security
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> TenantUserInfo:
    """Get current authenticated user."""
    manager = get_multi_tenant_manager()
    payload = manager.verify_token(credentials.credentials)

    user = await manager.get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_current_tenant(
    user: TenantUserInfo = Depends(get_current_user),
) -> TenantInfo:
    """Get current tenant."""
    manager = get_multi_tenant_manager()
    tenant = await manager.get_tenant(user.tenant_id)

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    return tenant


def require_permission(permission: str):
    """Decorator to require specific permission."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            if not user or (
                permission not in user.permissions and "*" not in user.permissions
            ):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Make components available for import
__all__ = [
    "MultiTenantManager",
    "TenantInfo",
    "TenantUserInfo",
    "TenantStatus",
    "UserRole",
    "ResourceType",
    "TenantCreateRequest",
    "TenantUpdateRequest",
    "UserCreateRequest",
    "LoginRequest",
    "TokenResponse",
    "get_multi_tenant_manager",
    "get_current_user",
    "get_current_tenant",
    "require_permission",
]
