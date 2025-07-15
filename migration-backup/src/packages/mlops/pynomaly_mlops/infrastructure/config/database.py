"""Database Configuration

Database setup and configuration for the MLOps platform.
"""

import os
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine as sync_create_engine
from sqlalchemy.ext.asyncio import create_async_engine as async_create_engine, AsyncEngine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "mlops"
    username: str = "mlops_user"
    password: str = "mlops_password"
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # SSL settings
    sslmode: str = "prefer"
    sslcert: Optional[str] = None
    sslkey: Optional[str] = None
    sslrootcert: Optional[str] = None
    
    # Additional options
    echo: bool = False  # Set to True for SQL logging
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables.
        
        Returns:
            DatabaseConfig instance
        """
        return cls(
            host=os.getenv('MLOPS_DB_HOST', 'localhost'),
            port=int(os.getenv('MLOPS_DB_PORT', '5432')),
            database=os.getenv('MLOPS_DB_NAME', 'mlops'),
            username=os.getenv('MLOPS_DB_USER', 'mlops_user'),
            password=os.getenv('MLOPS_DB_PASSWORD', 'mlops_password'),
            
            pool_size=int(os.getenv('MLOPS_DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('MLOPS_DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('MLOPS_DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('MLOPS_DB_POOL_RECYCLE', '3600')),
            
            sslmode=os.getenv('MLOPS_DB_SSLMODE', 'prefer'),
            sslcert=os.getenv('MLOPS_DB_SSLCERT'),
            sslkey=os.getenv('MLOPS_DB_SSLKEY'),
            sslrootcert=os.getenv('MLOPS_DB_SSLROOTCERT'),
            
            echo=os.getenv('MLOPS_DB_ECHO', 'false').lower() == 'true',
        )
    
    def get_sync_url(self) -> str:
        """Get synchronous database URL.
        
        Returns:
            PostgreSQL connection URL for sync operations
        """
        url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Add SSL parameters
        params = []
        if self.sslmode != "prefer":
            params.append(f"sslmode={self.sslmode}")
        if self.sslcert:
            params.append(f"sslcert={self.sslcert}")
        if self.sslkey:
            params.append(f"sslkey={self.sslkey}")
        if self.sslrootcert:
            params.append(f"sslrootcert={self.sslrootcert}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    def get_async_url(self) -> str:
        """Get asynchronous database URL.
        
        Returns:
            PostgreSQL connection URL for async operations
        """
        url = f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Add SSL parameters
        params = []
        if self.sslmode != "prefer":
            params.append(f"ssl={self.sslmode}")
        if self.sslcert:
            params.append(f"sslcert={self.sslcert}")
        if self.sslkey:
            params.append(f"sslkey={self.sslkey}")
        if self.sslrootcert:
            params.append(f"sslrootcert={self.sslrootcert}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url


def create_engine(config: DatabaseConfig) -> Engine:
    """Create synchronous SQLAlchemy engine.
    
    Args:
        config: Database configuration
        
    Returns:
        SQLAlchemy Engine instance
    """
    return sync_create_engine(
        config.get_sync_url(),
        poolclass=QueuePool,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout,
        pool_recycle=config.pool_recycle,
        echo=config.echo,
        
        # Connection parameters
        connect_args={
            "options": "-c timezone=utc"
        }
    )


def create_async_engine(config: DatabaseConfig) -> AsyncEngine:
    """Create asynchronous SQLAlchemy engine.
    
    Args:
        config: Database configuration
        
    Returns:
        SQLAlchemy AsyncEngine instance
    """
    return async_create_engine(
        config.get_async_url(),
        poolclass=QueuePool,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout,
        pool_recycle=config.pool_recycle,
        echo=config.echo,
        
        # Connection parameters
        connect_args={
            "server_settings": {
                "timezone": "utc"
            }
        }
    )