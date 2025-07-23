"""Persistence layer for infrastructure components.

This module provides database connectivity, repository patterns, and data access
abstractions. It supports multiple database backends and implements common
patterns like Unit of Work, Repository, and Connection Pooling.

Example usage:
    from infrastructure.persistence import DatabaseConnection, BaseRepository
    
    db = DatabaseConnection()
    repository = UserRepository(db)
    users = await repository.find_all()
"""

from .database import DatabaseConnection, get_database
from .repository import BaseRepository, Repository
from .unit_of_work import UnitOfWork
from .migrations import MigrationManager

__all__ = [
    "DatabaseConnection",
    "get_database",
    "BaseRepository", 
    "Repository",
    "UnitOfWork",
    "MigrationManager"
]