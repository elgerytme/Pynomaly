"""Database seed data for roles and permissions."""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from pynomaly.domain.entities.user import UserRole, DEFAULT_PERMISSIONS
from pynomaly.shared.types import generate_role_id
from .user_models import RoleModel, PermissionModel

logger = logging.getLogger(__name__)


def seed_default_roles_and_permissions(session: Session) -> bool:
    """Seed default roles and permissions.
    
    Args:
        session: Database session
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Seeding default roles and permissions...")
        
        # Check if roles already exist
        existing_roles = session.query(RoleModel).filter(RoleModel.is_system_role == True).all()
        if existing_roles:
            logger.info("System roles already exist, skipping seed")
            return True
        
        # Create roles and their permissions
        for role_enum, permissions_set in DEFAULT_PERMISSIONS.items():
            # Create role
            role = RoleModel(
                id=str(uuid4()),
                name=role_enum.value,
                description=f"System role: {role_enum.value.replace('_', ' ').title()}",
                is_system_role=True,
                created_at=datetime.utcnow()
            )
            session.add(role)
            session.flush()  # Flush to get the role ID
            
            # Create permissions for this role
            for permission in permissions_set:
                perm = PermissionModel(
                    id=str(uuid4()),
                    name=permission.name,
                    resource=permission.resource,
                    action=permission.action,
                    description=permission.description,
                    role_id=role.id
                )
                session.add(perm)
            
            logger.info(f"Created role: {role.name} with {len(permissions_set)} permissions")
        
        session.commit()
        logger.info("Successfully seeded default roles and permissions")
        return True
        
    except Exception as e:
        logger.error(f"Failed to seed roles and permissions: {e}")
        session.rollback()
        return False


def create_custom_role(session: Session, name: str, description: str = "", permissions: list = None) -> str | None:
    """Create a custom role with specified permissions.
    
    Args:
        session: Database session
        name: Role name
        description: Role description
        permissions: List of permission tuples (name, resource, action, description)
        
    Returns:
        Role ID if successful, None otherwise
    """
    try:
        # Check if role already exists
        existing_role = session.query(RoleModel).filter(RoleModel.name == name).first()
        if existing_role:
            logger.warning(f"Role '{name}' already exists")
            return str(existing_role.id)
        
        # Create role
        role = RoleModel(
            id=str(uuid4()),
            name=name,
            description=description,
            is_system_role=False,
            created_at=datetime.utcnow()
        )
        session.add(role)
        session.flush()
        
        # Create permissions if provided
        if permissions:
            for perm_data in permissions:
                if len(perm_data) == 4:
                    perm_name, resource, action, desc = perm_data
                else:
                    perm_name, resource, action = perm_data[:3]
                    desc = f"{action} {resource}"
                
                perm = PermissionModel(
                    id=str(uuid4()),
                    name=perm_name,
                    resource=resource,
                    action=action,
                    description=desc,
                    role_id=role.id
                )
                session.add(perm)
        
        session.commit()
        logger.info(f"Created custom role: {name}")
        return str(role.id)
        
    except Exception as e:
        logger.error(f"Failed to create custom role '{name}': {e}")
        session.rollback()
        return None


def get_role_by_name(session: Session, name: str) -> RoleModel | None:
    """Get role by name.
    
    Args:
        session: Database session
        name: Role name
        
    Returns:
        Role model if found, None otherwise
    """
    return session.query(RoleModel).filter(RoleModel.name == name).first()


def list_all_roles(session: Session) -> list[RoleModel]:
    """List all roles.
    
    Args:
        session: Database session
        
    Returns:
        List of role models
    """
    return session.query(RoleModel).all()


def delete_role(session: Session, role_id: str) -> bool:
    """Delete a role and its permissions.
    
    Args:
        session: Database session
        role_id: Role ID to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        role = session.query(RoleModel).filter(RoleModel.id == role_id).first()
        if not role:
            logger.warning(f"Role with ID '{role_id}' not found")
            return False
        
        if role.is_system_role:
            logger.error(f"Cannot delete system role: {role.name}")
            return False
        
        # Delete permissions first
        session.query(PermissionModel).filter(PermissionModel.role_id == role_id).delete()
        
        # Delete role
        session.delete(role)
        session.commit()
        
        logger.info(f"Deleted role: {role.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete role '{role_id}': {e}")
        session.rollback()
        return False


# Predefined custom roles that might be useful
CUSTOM_ROLES = [
    {
        "name": "api_user",
        "description": "API access for external integrations",
        "permissions": [
            ("api.read", "api", "read", "Read API endpoints"),
            ("detection.run", "detection", "run", "Run detections via API"),
            ("dataset.view", "dataset", "view", "View datasets via API"),
        ]
    },
    {
        "name": "read_only_admin",
        "description": "Read-only administrative access",
        "permissions": [
            ("tenant.view", "tenant", "view", "View tenant information"),
            ("user.view", "user", "view", "View user information"),
            ("dataset.view", "dataset", "view", "View all datasets"),
            ("model.view", "model", "view", "View all models"),
            ("detection.view", "detection", "view", "View all detections"),
            ("billing.view", "billing", "view", "View billing information"),
        ]
    }
]


def seed_custom_roles(session: Session) -> bool:
    """Seed predefined custom roles.
    
    Args:
        session: Database session
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Seeding custom roles...")
        
        for role_data in CUSTOM_ROLES:
            create_custom_role(
                session,
                role_data["name"],
                role_data["description"],
                role_data["permissions"]
            )
        
        logger.info("Successfully seeded custom roles")
        return True
        
    except Exception as e:
        logger.error(f"Failed to seed custom roles: {e}")
        return False
