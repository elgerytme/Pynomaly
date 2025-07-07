"""Migration 001: Add User, Role, Permission, and Tenant tables.

This migration adds the following tables:
- users: User accounts
- roles: User roles with descriptions
- permissions: Granular permissions for roles  
- tenants: Multi-tenant organizations
- user_roles: Many-to-many relationship between users and roles

It also seeds the database with default system roles and permissions.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..database import DatabaseManager
from ..database_repositories import UserModel, RoleModel, TenantModel, UserRoleModel
from ..seed_data import seed_default_roles_and_permissions, seed_custom_roles

logger = logging.getLogger(__name__)


def upgrade(db_manager: DatabaseManager) -> bool:
    """Apply migration 001: Add user, role, permission, and tenant tables.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Applying migration 001: Add user, role, permission, and tenant tables")
        
        # Create tables
        engine = db_manager.engine
        
        # Create user-related tables
        UserModel.__table__.create(engine, checkfirst=True)
        logger.info("Created users table")
        
        RoleModel.__table__.create(engine, checkfirst=True)
        logger.info("Created roles table")
        
        TenantModel.__table__.create(engine, checkfirst=True)
        logger.info("Created tenants table")
        
        UserRoleModel.__table__.create(engine, checkfirst=True)
        logger.info("Created user_roles association table")
        
        # Seed default data
        with db_manager.get_session() as session:
            if not seed_default_roles_and_permissions(session):
                logger.warning("Failed to seed default roles and permissions")
                return False
            
            if not seed_custom_roles(session):
                logger.warning("Failed to seed custom roles")
        
        logger.info("Successfully applied migration 001")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Database error in migration 001: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in migration 001: {e}")
        return False


def downgrade(db_manager: DatabaseManager) -> bool:
    """Rollback migration 001: Remove user, role, permission, and tenant tables.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Rolling back migration 001: Remove user, role, permission, and tenant tables")
        
        engine = db_manager.engine
        
        # Drop tables in reverse order
        UserRoleModel.__table__.drop(engine, checkfirst=True)
        logger.info("Dropped user_roles association table")
        
        RoleModel.__table__.drop(engine, checkfirst=True)
        logger.info("Dropped roles table")
        
        TenantModel.__table__.drop(engine, checkfirst=True)
        logger.info("Dropped tenants table")
        
        UserModel.__table__.drop(engine, checkfirst=True)
        logger.info("Dropped users table")
        
        logger.info("Successfully rolled back migration 001")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Database error in migration 001 rollback: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in migration 001 rollback: {e}")
        return False


def check_applied(db_manager: DatabaseManager) -> bool:
    """Check if migration 001 has been applied.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        True if applied, False otherwise
    """
    try:
        engine = db_manager.engine
        
        # Check if all required tables exist
        required_tables = ['users', 'roles', 'tenants', 'user_roles']
        existing_tables = engine.table_names()
        
        for table in required_tables:
            if table not in existing_tables:
                return False
        
        # Check if default roles exist
        with db_manager.get_session() as session:
            system_roles = session.query(RoleModel).filter(RoleModel.is_system_role == True).count()
            if system_roles == 0:
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking migration 001 status: {e}")
        return False


# Migration metadata
MIGRATION_ID = "001"
MIGRATION_NAME = "add_user_roles_permissions"
MIGRATION_DESCRIPTION = "Add User, Role, Permission, and Tenant tables with many-to-many relationships"
DEPENDS_ON = []  # No dependencies for first migration
