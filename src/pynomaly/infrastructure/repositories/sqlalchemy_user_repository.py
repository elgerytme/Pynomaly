"""
SQLAlchemy implementation of user management repositories.
"""

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, Float, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB

from pynomaly.domain.entities.user import (
    User, Tenant, UserSession, UserTenantRole, UserRole, UserStatus,
    TenantStatus, TenantPlan, TenantLimits, TenantUsage, Permission
)
from pynomaly.domain.repositories.user_repository import (
    UserRepositoryProtocol, TenantRepositoryProtocol, SessionRepositoryProtocol
)
from pynomaly.shared.types import UserId, TenantId

Base = declarative_base()

# Association table for user-tenant-roles
user_tenant_roles = Table(
    'user_tenant_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('tenant_id', UUID(as_uuid=True), ForeignKey('tenants.id'), primary_key=True),
    Column('role', String(50), nullable=False),
    Column('permissions', JSONB),
    Column('granted_at', DateTime, default=datetime.utcnow),
    Column('granted_by', UUID(as_uuid=True)),
    Column('expires_at', DateTime)
)


class UserModel(Base):
    """SQLAlchemy model for User entity."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    email_verified_at = Column(DateTime)
    settings = Column(JSONB, default=dict)
    
    # Relationships
    tenants = relationship("TenantModel", secondary=user_tenant_roles, back_populates="users")
    sessions = relationship("UserSessionModel", back_populates="user")


class TenantModel(Base):
    """SQLAlchemy model for Tenant entity."""
    __tablename__ = 'tenants'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True, nullable=False, index=True)
    plan = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)
    contact_email = Column(String(255), default='')
    billing_email = Column(String(255), default='')
    settings = Column(JSONB, default=dict)
    
    # Limits
    max_users = Column(Integer, default=10)
    max_datasets = Column(Integer, default=100)
    max_models = Column(Integer, default=50)
    max_detections_per_month = Column(Integer, default=10000)
    max_storage_gb = Column(Integer, default=10)
    max_api_calls_per_minute = Column(Integer, default=100)
    max_concurrent_detections = Column(Integer, default=5)
    
    # Usage
    users_count = Column(Integer, default=0)
    datasets_count = Column(Integer, default=0)
    models_count = Column(Integer, default=0)
    detections_this_month = Column(Integer, default=0)
    storage_used_gb = Column(Float, default=0.0)
    api_calls_this_minute = Column(Integer, default=0)
    concurrent_detections = Column(Integer, default=0)
    usage_last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("UserModel", secondary=user_tenant_roles, back_populates="tenants")


class UserSessionModel(Base):
    """SQLAlchemy model for UserSession entity."""
    __tablename__ = 'user_sessions'
    
    id = Column(String(255), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String(45), default='')
    user_agent = Column(Text, default='')
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("UserModel", back_populates="sessions")


class SQLAlchemyUserRepository(UserRepositoryProtocol):
    """SQLAlchemy implementation of UserRepositoryProtocol."""
    
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory
    
    def _to_domain_user(self, user_model: UserModel) -> User:
        """Convert SQLAlchemy model to domain entity."""
        # Get tenant roles from association table
        tenant_roles = []
        with self._session_factory() as session:
            result = session.execute(
                user_tenant_roles.select().where(user_tenant_roles.c.user_id == user_model.id)
            )
            for row in result:
                permissions = set()
                if row.permissions:
                    for perm_data in row.permissions:
                        permissions.add(Permission(**perm_data))
                
                tenant_roles.append(UserTenantRole(
                    user_id=UserId(str(user_model.id)),
                    tenant_id=TenantId(str(row.tenant_id)),
                    role=UserRole(row.role),
                    permissions=permissions,
                    granted_at=row.granted_at,
                    granted_by=UserId(str(row.granted_by)) if row.granted_by else None,
                    expires_at=row.expires_at
                ))
        
        return User(
            id=UserId(str(user_model.id)),
            email=user_model.email,
            username=user_model.username,
            first_name=user_model.first_name,
            last_name=user_model.last_name,
            status=UserStatus(user_model.status),
            tenant_roles=tenant_roles,
            created_at=user_model.created_at,
            updated_at=user_model.updated_at,
            last_login_at=user_model.last_login_at,
            email_verified_at=user_model.email_verified_at,
            password_hash=user_model.password_hash,
            settings=user_model.settings or {}
        )
    
    def _to_model_user(self, user: User) -> UserModel:
        """Convert domain entity to SQLAlchemy model."""
        return UserModel(
            id=user.id,
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            status=user.status.value,
            password_hash=user.password_hash,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            email_verified_at=user.email_verified_at,
            settings=user.settings
        )
    
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        with self._session_factory() as session:
            user_model = self._to_model_user(user)
            session.add(user_model)
            
            # Add tenant roles
            for tenant_role in user.tenant_roles:
                permissions_data = [
                    {
                        "name": p.name,
                        "resource": p.resource,
                        "action": p.action,
                        "description": p.description
                    }
                    for p in tenant_role.permissions
                ]
                
                session.execute(
                    user_tenant_roles.insert().values(
                        user_id=user.id,
                        tenant_id=tenant_role.tenant_id,
                        role=tenant_role.role.value,
                        permissions=permissions_data,
                        granted_at=tenant_role.granted_at,
                        granted_by=tenant_role.granted_by,
                        expires_at=tenant_role.expires_at
                    )
                )
            
            session.commit()
            session.refresh(user_model)
            return self._to_domain_user(user_model)
    
    async def get_user_by_id(self, user_id: UserId) -> Optional[User]:
        """Get user by ID."""
        with self._session_factory() as session:
            user_model = session.query(UserModel).filter(UserModel.id == user_id).first()
            return self._to_domain_user(user_model) if user_model else None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        with self._session_factory() as session:
            user_model = session.query(UserModel).filter(UserModel.email == email).first()
            return self._to_domain_user(user_model) if user_model else None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with self._session_factory() as session:
            user_model = session.query(UserModel).filter(UserModel.username == username).first()
            return self._to_domain_user(user_model) if user_model else None
    
    async def update_user(self, user: User) -> User:
        """Update existing user."""
        with self._session_factory() as session:
            user_model = session.query(UserModel).filter(UserModel.id == user.id).first()
            if not user_model:
                raise ValueError("User not found")
            
            # Update fields
            user_model.email = user.email
            user_model.username = user.username
            user_model.first_name = user.first_name
            user_model.last_name = user.last_name
            user_model.status = user.status.value
            user_model.password_hash = user.password_hash
            user_model.updated_at = user.updated_at
            user_model.last_login_at = user.last_login_at
            user_model.email_verified_at = user.email_verified_at
            user_model.settings = user.settings
            
            session.commit()
            session.refresh(user_model)
            return self._to_domain_user(user_model)
    
    async def delete_user(self, user_id: UserId) -> bool:
        """Delete user."""
        with self._session_factory() as session:
            # Delete tenant role associations
            session.execute(
                user_tenant_roles.delete().where(user_tenant_roles.c.user_id == user_id)
            )
            
            # Delete user
            deleted = session.query(UserModel).filter(UserModel.id == user_id).delete()
            session.commit()
            return deleted > 0
    
    async def get_users_by_tenant(self, tenant_id: TenantId) -> List[User]:
        """Get all users for a tenant."""
        with self._session_factory() as session:
            result = session.execute(
                user_tenant_roles.select().where(user_tenant_roles.c.tenant_id == tenant_id)
            )
            user_ids = [row.user_id for row in result]
            
            user_models = session.query(UserModel).filter(UserModel.id.in_(user_ids)).all()
            return [self._to_domain_user(model) for model in user_models]
    
    async def add_user_to_tenant(self, user_id: UserId, tenant_id: TenantId, role: UserRole) -> UserTenantRole:
        """Add user to tenant with role."""
        with self._session_factory() as session:
            from pynomaly.domain.entities.user import get_default_permissions
            
            permissions_data = [
                {
                    "name": p.name,
                    "resource": p.resource,
                    "action": p.action,
                    "description": p.description
                }
                for p in get_default_permissions(role)
            ]
            
            session.execute(
                user_tenant_roles.insert().values(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    role=role.value,
                    permissions=permissions_data,
                    granted_at=datetime.utcnow()
                )
            )
            session.commit()
            
            return UserTenantRole(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                permissions=get_default_permissions(role),
                granted_at=datetime.utcnow()
            )
    
    async def remove_user_from_tenant(self, user_id: UserId, tenant_id: TenantId) -> bool:
        """Remove user from tenant."""
        with self._session_factory() as session:
            deleted = session.execute(
                user_tenant_roles.delete().where(
                    (user_tenant_roles.c.user_id == user_id) &
                    (user_tenant_roles.c.tenant_id == tenant_id)
                )
            )
            session.commit()
            return deleted.rowcount > 0
    
    async def update_user_role_in_tenant(self, user_id: UserId, tenant_id: TenantId, role: UserRole) -> UserTenantRole:
        """Update user's role in tenant."""
        with self._session_factory() as session:
            from pynomaly.domain.entities.user import get_default_permissions
            
            permissions_data = [
                {
                    "name": p.name,
                    "resource": p.resource,
                    "action": p.action,
                    "description": p.description
                }
                for p in get_default_permissions(role)
            ]
            
            session.execute(
                user_tenant_roles.update().where(
                    (user_tenant_roles.c.user_id == user_id) &
                    (user_tenant_roles.c.tenant_id == tenant_id)
                ).values(
                    role=role.value,
                    permissions=permissions_data
                )
            )
            session.commit()
            
            return UserTenantRole(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                permissions=get_default_permissions(role),
                granted_at=datetime.utcnow()
            )


class SQLAlchemyTenantRepository(TenantRepositoryProtocol):
    """SQLAlchemy implementation of TenantRepositoryProtocol."""
    
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory
    
    def _to_domain_tenant(self, tenant_model: TenantModel) -> Tenant:
        """Convert SQLAlchemy model to domain entity."""
        limits = TenantLimits(
            max_users=tenant_model.max_users,
            max_datasets=tenant_model.max_datasets,
            max_models=tenant_model.max_models,
            max_detections_per_month=tenant_model.max_detections_per_month,
            max_storage_gb=tenant_model.max_storage_gb,
            max_api_calls_per_minute=tenant_model.max_api_calls_per_minute,
            max_concurrent_detections=tenant_model.max_concurrent_detections
        )
        
        usage = TenantUsage(
            users_count=tenant_model.users_count,
            datasets_count=tenant_model.datasets_count,
            models_count=tenant_model.models_count,
            detections_this_month=tenant_model.detections_this_month,
            storage_used_gb=tenant_model.storage_used_gb,
            api_calls_this_minute=tenant_model.api_calls_this_minute,
            concurrent_detections=tenant_model.concurrent_detections,
            last_updated=tenant_model.usage_last_updated
        )
        
        return Tenant(
            id=TenantId(str(tenant_model.id)),
            name=tenant_model.name,
            domain=tenant_model.domain,
            plan=TenantPlan(tenant_model.plan),
            status=TenantStatus(tenant_model.status),
            limits=limits,
            usage=usage,
            created_at=tenant_model.created_at,
            updated_at=tenant_model.updated_at,
            expires_at=tenant_model.expires_at,
            contact_email=tenant_model.contact_email,
            billing_email=tenant_model.billing_email,
            settings=tenant_model.settings or {}
        )
    
    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create a new tenant."""
        with self._session_factory() as session:
            tenant_model = TenantModel(
                id=tenant.id,
                name=tenant.name,
                domain=tenant.domain,
                plan=tenant.plan.value,
                status=tenant.status.value,
                created_at=tenant.created_at,
                updated_at=tenant.updated_at,
                expires_at=tenant.expires_at,
                contact_email=tenant.contact_email,
                billing_email=tenant.billing_email,
                settings=tenant.settings,
                # Limits
                max_users=tenant.limits.max_users,
                max_datasets=tenant.limits.max_datasets,
                max_models=tenant.limits.max_models,
                max_detections_per_month=tenant.limits.max_detections_per_month,
                max_storage_gb=tenant.limits.max_storage_gb,
                max_api_calls_per_minute=tenant.limits.max_api_calls_per_minute,
                max_concurrent_detections=tenant.limits.max_concurrent_detections,
                # Usage
                users_count=tenant.usage.users_count,
                datasets_count=tenant.usage.datasets_count,
                models_count=tenant.usage.models_count,
                detections_this_month=tenant.usage.detections_this_month,
                storage_used_gb=tenant.usage.storage_used_gb,
                api_calls_this_minute=tenant.usage.api_calls_this_minute,
                concurrent_detections=tenant.usage.concurrent_detections,
                usage_last_updated=tenant.usage.last_updated
            )
            
            session.add(tenant_model)
            session.commit()
            session.refresh(tenant_model)
            return self._to_domain_tenant(tenant_model)
    
    async def get_tenant_by_id(self, tenant_id: TenantId) -> Optional[Tenant]:
        """Get tenant by ID."""
        with self._session_factory() as session:
            tenant_model = session.query(TenantModel).filter(TenantModel.id == tenant_id).first()
            return self._to_domain_tenant(tenant_model) if tenant_model else None
    
    async def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        with self._session_factory() as session:
            tenant_model = session.query(TenantModel).filter(TenantModel.domain == domain).first()
            return self._to_domain_tenant(tenant_model) if tenant_model else None
    
    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update existing tenant."""
        with self._session_factory() as session:
            tenant_model = session.query(TenantModel).filter(TenantModel.id == tenant.id).first()
            if not tenant_model:
                raise ValueError("Tenant not found")
            
            # Update fields
            tenant_model.name = tenant.name
            tenant_model.domain = tenant.domain
            tenant_model.plan = tenant.plan.value
            tenant_model.status = tenant.status.value
            tenant_model.updated_at = tenant.updated_at
            tenant_model.expires_at = tenant.expires_at
            tenant_model.contact_email = tenant.contact_email
            tenant_model.billing_email = tenant.billing_email
            tenant_model.settings = tenant.settings
            
            session.commit()
            session.refresh(tenant_model)
            return self._to_domain_tenant(tenant_model)
    
    async def delete_tenant(self, tenant_id: TenantId) -> bool:
        """Delete tenant."""
        with self._session_factory() as session:
            deleted = session.query(TenantModel).filter(TenantModel.id == tenant_id).delete()
            session.commit()
            return deleted > 0
    
    async def list_tenants(self, limit: int = 100, offset: int = 0) -> List[Tenant]:
        """List all tenants with pagination."""
        with self._session_factory() as session:
            tenant_models = session.query(TenantModel).offset(offset).limit(limit).all()
            return [self._to_domain_tenant(model) for model in tenant_models]
    
    async def update_tenant_usage(self, tenant_id: TenantId, usage_updates: dict) -> bool:
        """Update tenant usage statistics."""
        with self._session_factory() as session:
            tenant_model = session.query(TenantModel).filter(TenantModel.id == tenant_id).first()
            if not tenant_model:
                return False
            
            # Apply updates
            for field, value in usage_updates.items():
                if hasattr(tenant_model, field):
                    if isinstance(value, str) and value.startswith('+'):
                        # Increment operation
                        increment = int(value[1:])
                        current = getattr(tenant_model, field)
                        setattr(tenant_model, field, current + increment)
                    else:
                        setattr(tenant_model, field, value)
            
            tenant_model.usage_last_updated = datetime.utcnow()
            session.commit()
            return True


class SQLAlchemySessionRepository(SessionRepositoryProtocol):
    """SQLAlchemy implementation of SessionRepositoryProtocol."""
    
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory
    
    def _to_domain_session(self, session_model: UserSessionModel) -> UserSession:
        """Convert SQLAlchemy model to domain entity."""
        return UserSession(
            id=session_model.id,
            user_id=UserId(str(session_model.user_id)),
            tenant_id=TenantId(str(session_model.tenant_id)) if session_model.tenant_id else None,
            created_at=session_model.created_at,
            expires_at=session_model.expires_at,
            ip_address=session_model.ip_address,
            user_agent=session_model.user_agent,
            is_active=session_model.is_active,
            last_activity=session_model.last_activity
        )
    
    async def create_session(self, session: UserSession) -> UserSession:
        """Create a new user session."""
        with self._session_factory() as db_session:
            session_model = UserSessionModel(
                id=session.id,
                user_id=session.user_id,
                tenant_id=session.tenant_id,
                created_at=session.created_at,
                expires_at=session.expires_at,
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                is_active=session.is_active,
                last_activity=session.last_activity
            )
            
            db_session.add(session_model)
            db_session.commit()
            db_session.refresh(session_model)
            return self._to_domain_session(session_model)
    
    async def get_session_by_id(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        with self._session_factory() as db_session:
            session_model = db_session.query(UserSessionModel).filter(
                UserSessionModel.id == session_id
            ).first()
            return self._to_domain_session(session_model) if session_model else None
    
    async def update_session(self, session: UserSession) -> UserSession:
        """Update existing session."""
        with self._session_factory() as db_session:
            session_model = db_session.query(UserSessionModel).filter(
                UserSessionModel.id == session.id
            ).first()
            if not session_model:
                raise ValueError("Session not found")
            
            session_model.expires_at = session.expires_at
            session_model.ip_address = session.ip_address
            session_model.user_agent = session.user_agent
            session_model.is_active = session.is_active
            session_model.last_activity = session.last_activity
            
            db_session.commit()
            db_session.refresh(session_model)
            return self._to_domain_session(session_model)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        with self._session_factory() as db_session:
            deleted = db_session.query(UserSessionModel).filter(
                UserSessionModel.id == session_id
            ).delete()
            db_session.commit()
            return deleted > 0
    
    async def get_active_sessions_for_user(self, user_id: UserId) -> List[UserSession]:
        """Get all active sessions for a user."""
        with self._session_factory() as db_session:
            session_models = db_session.query(UserSessionModel).filter(
                UserSessionModel.user_id == user_id,
                UserSessionModel.is_active == True,
                UserSessionModel.expires_at > datetime.utcnow()
            ).all()
            return [self._to_domain_session(model) for model in session_models]
    
    async def delete_all_sessions_for_user(self, user_id: UserId) -> bool:
        """Delete all sessions for a user."""
        with self._session_factory() as db_session:
            deleted = db_session.query(UserSessionModel).filter(
                UserSessionModel.user_id == user_id
            ).delete()
            db_session.commit()
            return deleted > 0