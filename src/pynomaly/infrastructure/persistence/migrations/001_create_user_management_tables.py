"""Database migration script for user management and metrics tables.

This migration creates tables for:
- Users
- Tenants
- Roles
- User-Tenant-Role associations
- Metrics

Revision ID: 001
Create Date: 2024-01-01 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


def upgrade():
    """Create user management and metrics tables."""
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('first_name', sa.String(100)),
        sa.Column('last_name', sa.String(100)),
        sa.Column('status', sa.String(50)),
        sa.Column('password_hash', sa.String(255)),
        sa.Column('settings', sa.JSON),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        sa.Column('last_login_at', sa.DateTime),
        sa.Column('email_verified_at', sa.DateTime),
    )
    
    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('domain', sa.String(255), nullable=False),
        sa.Column('plan', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('limits', sa.JSON),
        sa.Column('usage', sa.JSON),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        sa.Column('expires_at', sa.DateTime),
        sa.Column('contact_email', sa.String(255)),
        sa.Column('billing_email', sa.String(255)),
        sa.Column('settings', sa.JSON),
    )
    
    # Create roles table
    op.create_table(
        'roles',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('permissions', sa.JSON),
        sa.Column('is_system_role', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )
    
    # Create user_roles association table
    op.create_table(
        'user_roles',
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('tenant_id', sa.String(36), sa.ForeignKey('tenants.id'), primary_key=True),
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id'), primary_key=True),
        sa.Column('permissions', sa.JSON),
        sa.Column('granted_at', sa.DateTime, nullable=False),
        sa.Column('granted_by', sa.String(36), sa.ForeignKey('users.id')),
        sa.Column('expires_at', sa.DateTime),
    )
    
    # Create metrics table
    op.create_table(
        'metrics',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('value', sa.Float, nullable=False),
        sa.Column('unit', sa.String(50)),
        sa.Column('tags', sa.JSON),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('entity_type', sa.String(100)),
        sa.Column('entity_id', sa.String(36)),
        sa.Column('metadata', sa.JSON),
    )
    
    # Create indexes for better performance
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_status', 'users', ['status'])
    
    op.create_index('idx_tenants_domain', 'tenants', ['domain'])
    op.create_index('idx_tenants_plan', 'tenants', ['plan'])
    op.create_index('idx_tenants_status', 'tenants', ['status'])
    
    op.create_index('idx_roles_name', 'roles', ['name'])
    op.create_index('idx_roles_system', 'roles', ['is_system_role'])
    
    op.create_index('idx_user_roles_user', 'user_roles', ['user_id'])
    op.create_index('idx_user_roles_tenant', 'user_roles', ['tenant_id'])
    op.create_index('idx_user_roles_role', 'user_roles', ['role_id'])
    
    op.create_index('idx_metrics_name', 'metrics', ['name'])
    op.create_index('idx_metrics_timestamp', 'metrics', ['timestamp'])
    op.create_index('idx_metrics_entity', 'metrics', ['entity_type', 'entity_id'])


def downgrade():
    """Drop user management and metrics tables."""
    
    # Drop indexes
    op.drop_index('idx_metrics_entity')
    op.drop_index('idx_metrics_timestamp')
    op.drop_index('idx_metrics_name')
    
    op.drop_index('idx_user_roles_role')
    op.drop_index('idx_user_roles_tenant')
    op.drop_index('idx_user_roles_user')
    
    op.drop_index('idx_roles_system')
    op.drop_index('idx_roles_name')
    
    op.drop_index('idx_tenants_status')
    op.drop_index('idx_tenants_plan')
    op.drop_index('idx_tenants_domain')
    
    op.drop_index('idx_users_status')
    op.drop_index('idx_users_username')
    op.drop_index('idx_users_email')
    
    # Drop tables
    op.drop_table('metrics')
    op.drop_table('user_roles')
    op.drop_table('roles')
    op.drop_table('tenants')
    op.drop_table('users')
