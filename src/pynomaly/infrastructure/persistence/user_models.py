import json
from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import VARCHAR, TypeDecorator

# Create a separate base for user models to avoid circular imports
Base = declarative_base()


class JSONType(TypeDecorator):
    """JSON type that works with both PostgreSQL and SQLite."""

    impl = VARCHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class UUIDType(TypeDecorator):
    """UUID type that works with both PostgreSQL and SQLite."""

    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGUUID())
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return UUID(value)
        return value


# Many-to-many association table for users and roles
user_roles_association = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUIDType, ForeignKey("users.id")),
    Column("role_id", UUIDType, ForeignKey("roles.id")),
)

# Many-to-many association table for roles and permissions
role_permissions_association = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", UUIDType, ForeignKey("roles.id")),
    Column("permission_id", UUIDType, ForeignKey("permissions.id")),
)


class APIKeyModel(Base):
    """SQLAlchemy model for API Keys."""

    __tablename__ = "api_keys"

    id = Column(UUIDType, primary_key=True)
    key_hash = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    user_id = Column(UUIDType, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="api_keys")


class UserModel(Base):
    """SQLAlchemy model for User entity."""

    __tablename__ = "users"

    id = Column(UUIDType, primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    username = Column(String(255), nullable=False, unique=True)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)  # UserStatus enum
    is_active = Column(Boolean, default=True)
    settings = Column(JSONType, default=dict)
    last_login_at = Column(DateTime, nullable=True)
    email_verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    roles = relationship(
        "RoleModel", secondary=user_roles_association, back_populates="users"
    )
    api_keys = relationship(
        "APIKeyModel", back_populates="user", cascade="all, delete-orphan"
    )


class RoleModel(Base):
    """SQLAlchemy model for Role entity."""

    __tablename__ = "roles"

    id = Column(UUIDType, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    is_system_role = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    users = relationship(
        "UserModel", secondary=user_roles_association, back_populates="roles"
    )
    permissions = relationship(
        "PermissionModel",
        secondary=role_permissions_association,
        back_populates="roles",
    )


class PermissionModel(Base):
    """SQLAlchemy model for Permission entity."""

    __tablename__ = "permissions"

    id = Column(UUIDType, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    resource = Column(String(255), nullable=False)
    action = Column(String(255), nullable=False)
    description = Column(Text)

    roles = relationship(
        "RoleModel",
        secondary=role_permissions_association,
        back_populates="permissions",
    )


class TenantModel(Base):
    """SQLAlchemy model for Tenant entity."""

    __tablename__ = "tenants"

    id = Column(UUIDType, primary_key=True)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), nullable=False, unique=True)
    plan = Column(String(50), nullable=False)  # TenantPlan enum
    status = Column(String(50), nullable=False)  # TenantStatus enum
    contact_email = Column(String(255))
    billing_email = Column(String(255))
    settings = Column(JSONType, default=dict)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
