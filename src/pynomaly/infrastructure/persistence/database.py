"""Database connection and session management."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .database_repositories import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self, database_url: str, echo: bool = False):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url
        self.echo = echo
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        if self._engine is None:
            # Configure engine based on database type
            if self.database_url.startswith('sqlite:'):
                # SQLite configuration
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    poolclass=StaticPool,
                    connect_args={'check_same_thread': False}
                )
                
                # Enable foreign keys for SQLite
                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()
            
            elif self.database_url.startswith('postgresql:'):
                # PostgreSQL configuration
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True
                )
            
            else:
                # Default configuration
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo
                )
        
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory
    
    def create_tables(self) -> None:
        """Create all database tables."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session (context manager)."""
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager = None


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialize global database manager.
    
    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements
        
    Returns:
        Database manager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo)
    return _db_manager


def get_database_manager() -> DatabaseManager:
    """Get global database manager.
    
    Returns:
        Database manager instance
        
    Raises:
        RuntimeError: If database not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_manager


def get_session() -> Generator[Session, None, None]:
    """Get database session from global manager.
    
    Returns:
        Database session
        
    Raises:
        RuntimeError: If database not initialized
    """
    return get_database_manager().get_session()


# Common database URLs for development
SQLITE_MEMORY_URL = "sqlite:///:memory:"
SQLITE_FILE_URL = "sqlite:///./pynomaly.db"
POSTGRESQL_LOCAL_URL = "postgresql://pynomaly:pynomaly@localhost:5432/pynomaly"