"""Database configuration and connection settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "pynomaly_production"
    username: str = "pynomaly"
    password: str = "pynomaly"
    
    # Connection options
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    
    # SSL/TLS settings
    ssl_mode: str = "prefer"
    ssl_cert: str | None = None
    ssl_key: str | None = None
    ssl_ca: str | None = None
    
    # Additional connection parameters
    connect_timeout: int = 10
    command_timeout: int = 30
    
    @property
    def connection_url(self) -> str:
        """Generate database connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def connection_params(self) -> dict:
        """Get connection parameters for SQLAlchemy."""
        params = {
            "echo": self.echo,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_pre_ping": self.pool_pre_ping,
        }
        
        # Add SSL parameters if configured
        if self.ssl_mode != "disable":
            connect_args = {"sslmode": self.ssl_mode}
            if self.ssl_cert:
                connect_args["sslcert"] = self.ssl_cert
            if self.ssl_key:
                connect_args["sslkey"] = self.ssl_key
            if self.ssl_ca:
                connect_args["sslrootcert"] = self.ssl_ca
            
            params["connect_args"] = connect_args
        
        return params


@dataclass
class MongoDBConfig:
    """MongoDB configuration settings."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 27017
    database: str = "pynomaly_production"
    username: str | None = None
    password: str | None = None
    
    # Connection options
    max_pool_size: int = 100
    min_pool_size: int = 0
    max_idle_time_ms: int = 30000
    connect_timeout_ms: int = 10000
    server_selection_timeout_ms: int = 30000
    
    # SSL/TLS settings
    ssl_enabled: bool = False
    ssl_cert_reqs: str = "CERT_REQUIRED"
    ssl_ca_certs: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    
    # Replica set settings
    replica_set: str | None = None
    read_preference: str = "primary"
    
    @property
    def connection_url(self) -> str:
        """Generate MongoDB connection URL."""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        
        url = f"mongodb://{auth}{self.host}:{self.port}/{self.database}"
        
        # Add replica set if configured
        if self.replica_set:
            url += f"?replicaSet={self.replica_set}"
        
        return url
    
    @property
    def connection_params(self) -> dict:
        """Get connection parameters for PyMongo."""
        params = {
            "maxPoolSize": self.max_pool_size,
            "minPoolSize": self.min_pool_size,
            "maxIdleTimeMS": self.max_idle_time_ms,
            "connectTimeoutMS": self.connect_timeout_ms,
            "serverSelectionTimeoutMS": self.server_selection_timeout_ms,
        }
        
        # Add SSL parameters if enabled
        if self.ssl_enabled:
            params["ssl"] = True
            params["ssl_cert_reqs"] = self.ssl_cert_reqs
            if self.ssl_ca_certs:
                params["ssl_ca_certs"] = self.ssl_ca_certs
            if self.ssl_certfile:
                params["ssl_certfile"] = self.ssl_certfile
            if self.ssl_keyfile:
                params["ssl_keyfile"] = self.ssl_keyfile
        
        # Add replica set if configured
        if self.replica_set:
            params["replicaSet"] = self.replica_set
            params["readPreference"] = self.read_preference
        
        return params


class DatabaseSettings(BaseModel):
    """Pydantic model for database settings validation."""
    
    # Database type
    database_type: Literal["postgresql", "mongodb", "sqlite"] = Field(
        default="postgresql",
        description="Type of database to use"
    )
    
    # PostgreSQL settings
    postgresql: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="PostgreSQL configuration"
    )
    
    # MongoDB settings
    mongodb: MongoDBConfig = Field(
        default_factory=MongoDBConfig,
        description="MongoDB configuration"
    )
    
    # SQLite settings
    sqlite_path: str = Field(
        default="./pynomaly.db",
        description="Path to SQLite database file"
    )
    
    # General settings
    create_tables: bool = Field(
        default=True,
        description="Whether to create database tables on startup"
    )
    
    migration_enabled: bool = Field(
        default=True,
        description="Whether to run database migrations"
    )
    
    @classmethod
    def from_environment(cls) -> DatabaseSettings:
        """Create database settings from environment variables."""
        
        # Database type
        db_type = os.getenv("DB_TYPE", "postgresql")
        
        # PostgreSQL settings
        postgresql = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "pynomaly_production"),
            username=os.getenv("DB_USER", "pynomaly"),
            password=os.getenv("DB_PASSWORD", "pynomaly"),
            echo=os.getenv("DB_ECHO", "False").lower() == "true",
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_pre_ping=os.getenv("DB_POOL_PRE_PING", "True").lower() == "true",
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer"),
            ssl_cert=os.getenv("DB_SSL_CERT"),
            ssl_key=os.getenv("DB_SSL_KEY"),
            ssl_ca=os.getenv("DB_SSL_CA"),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
            command_timeout=int(os.getenv("DB_COMMAND_TIMEOUT", "30")),
        )
        
        # MongoDB settings
        mongodb = MongoDBConfig(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", "27017")),
            database=os.getenv("MONGO_DB", "pynomaly_production"),
            username=os.getenv("MONGO_USER"),
            password=os.getenv("MONGO_PASSWORD"),
            max_pool_size=int(os.getenv("MONGO_MAX_POOL_SIZE", "100")),
            min_pool_size=int(os.getenv("MONGO_MIN_POOL_SIZE", "0")),
            max_idle_time_ms=int(os.getenv("MONGO_MAX_IDLE_TIME_MS", "30000")),
            connect_timeout_ms=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "10000")),
            server_selection_timeout_ms=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "30000")),
            ssl_enabled=os.getenv("MONGO_SSL_ENABLED", "False").lower() == "true",
            ssl_cert_reqs=os.getenv("MONGO_SSL_CERT_REQS", "CERT_REQUIRED"),
            ssl_ca_certs=os.getenv("MONGO_SSL_CA_CERTS"),
            ssl_certfile=os.getenv("MONGO_SSL_CERTFILE"),
            ssl_keyfile=os.getenv("MONGO_SSL_KEYFILE"),
            replica_set=os.getenv("MONGO_REPLICA_SET"),
            read_preference=os.getenv("MONGO_READ_PREFERENCE", "primary"),
        )
        
        # General settings
        create_tables = os.getenv("DB_CREATE_TABLES", "True").lower() == "true"
        migration_enabled = os.getenv("DB_MIGRATION_ENABLED", "True").lower() == "true"
        sqlite_path = os.getenv("SQLITE_PATH", "./pynomaly.db")
        
        return cls(
            database_type=db_type,  # type: ignore
            postgresql=postgresql,
            mongodb=mongodb,
            sqlite_path=sqlite_path,
            create_tables=create_tables,
            migration_enabled=migration_enabled,
        )


# Default production database settings
PRODUCTION_DATABASE_SETTINGS = DatabaseSettings(
    database_type="postgresql",
    postgresql=DatabaseConfig(
        host="localhost",
        port=5432,
        database="pynomaly_production",
        username="pynomaly",
        password="pynomaly",
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        ssl_mode="prefer",
    ),
    create_tables=True,
    migration_enabled=True,
)

# Default development database settings
DEVELOPMENT_DATABASE_SETTINGS = DatabaseSettings(
    database_type="sqlite",
    sqlite_path="./pynomaly_dev.db",
    create_tables=True,
    migration_enabled=True,
)

# Default test database settings
TEST_DATABASE_SETTINGS = DatabaseSettings(
    database_type="sqlite",
    sqlite_path=":memory:",
    create_tables=True,
    migration_enabled=False,
)


def get_database_settings() -> DatabaseSettings:
    """Get database settings based on environment."""
    environment = os.getenv("PYNOMALY_ENV", "development").lower()
    
    if environment == "production":
        return DatabaseSettings.from_environment()
    elif environment == "development":
        return DEVELOPMENT_DATABASE_SETTINGS
    elif environment == "test":
        return TEST_DATABASE_SETTINGS
    else:
        # Default to development
        return DEVELOPMENT_DATABASE_SETTINGS
