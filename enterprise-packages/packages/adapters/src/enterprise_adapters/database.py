"""Database adapter implementations for enterprise applications.

This module provides adapters for various database systems including
SQL databases (PostgreSQL, MySQL, SQLite) and NoSQL databases (MongoDB).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from enterprise_core import HealthStatus, InfrastructureError
from pydantic import Field

from .base import AdapterConfiguration, BaseAdapter, adapter

logger = logging.getLogger(__name__)


class DatabaseConfiguration(AdapterConfiguration):
    """Configuration for database adapters."""

    adapter_type: str = Field(..., description="Database adapter type")
    database: str = Field(..., description="Database name")
    echo: bool = Field(default=False, description="Enable SQL echoing")
    pool_pre_ping: bool = Field(default=True, description="Enable connection pool pre-ping")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")


class DatabaseAdapter(BaseAdapter):
    """Base class for database adapters."""

    def __init__(self, config: DatabaseConfiguration) -> None:
        super().__init__(config)
        self.db_config = config

    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query."""
        pass

    @abstractmethod
    async def execute_many(self, query: str, parameters_list: List[Dict[str, Any]]) -> Any:
        """Execute a query with multiple parameter sets."""
        pass

    @abstractmethod
    async def fetch_one(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from a query."""
        pass

    @abstractmethod
    async def fetch_all(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from a query."""
        pass

    @abstractmethod
    async def begin_transaction(self) -> Any:
        """Begin a database transaction."""
        pass

    @abstractmethod
    async def commit_transaction(self, transaction: Any) -> None:
        """Commit a database transaction."""
        pass

    @abstractmethod
    async def rollback_transaction(self, transaction: Any) -> None:
        """Rollback a database transaction."""
        pass


@adapter("sqlalchemy")
class SQLAlchemyAdapter(DatabaseAdapter):
    """SQLAlchemy adapter for SQL databases."""

    def __init__(self, config: DatabaseConfiguration) -> None:
        super().__init__(config)
        self._engine: Optional[Any] = None
        self._session_factory: Optional[Any] = None

    async def _create_connection(self) -> Any:
        """Create SQLAlchemy engine and session factory."""
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        except ImportError:
            raise InfrastructureError(
                "SQLAlchemy not installed. Install with: pip install 'enterprise-adapters[database]'",
                error_code="DEPENDENCY_MISSING",
            )

        # Build connection URL
        if self.config.connection_string:
            url = self.config.connection_string
        else:
            if not all([self.config.host, self.config.username, self.config.password]):
                raise InfrastructureError(
                    "Database connection requires either connection_string or host/username/password",
                    error_code="CONFIGURATION_ERROR",
                )

            url = f"postgresql+asyncpg://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port or 5432}/{self.db_config.database}"

        # Create engine
        self._engine = create_async_engine(
            url,
            echo=self.db_config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=self.db_config.pool_pre_ping,
            pool_recycle=self.db_config.pool_recycle,
        )

        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

        return self._engine

    async def _close_connection(self) -> None:
        """Close SQLAlchemy engine."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def _test_connection(self) -> bool:
        """Test the SQLAlchemy connection."""
        if not self._engine:
            return False

        try:
            async with self._engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception:
            return False

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query."""
        if not self._session_factory:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        async with self.with_retry():
            async with self._session_factory() as session:
                try:
                    from sqlalchemy import text
                    result = await session.execute(text(query), parameters or {})
                    await session.commit()
                    return result
                except Exception as e:
                    await session.rollback()
                    raise InfrastructureError(
                        f"Database query failed: {e}",
                        error_code="QUERY_FAILED",
                        details={"query": query, "parameters": parameters},
                        cause=e,
                    )

    async def execute_many(self, query: str, parameters_list: List[Dict[str, Any]]) -> Any:
        """Execute a query with multiple parameter sets."""
        if not self._session_factory:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        async with self.with_retry():
            async with self._session_factory() as session:
                try:
                    from sqlalchemy import text
                    result = await session.execute(text(query), parameters_list)
                    await session.commit()
                    return result
                except Exception as e:
                    await session.rollback()
                    raise InfrastructureError(
                        f"Database bulk query failed: {e}",
                        error_code="BULK_QUERY_FAILED",
                        details={"query": query, "parameter_count": len(parameters_list)},
                        cause=e,
                    )

    async def fetch_one(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from a query."""
        result = await self.execute_query(query, parameters)
        row = result.fetchone()
        return dict(row) if row else None

    async def fetch_all(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from a query."""
        result = await self.execute_query(query, parameters)
        rows = result.fetchall()
        return [dict(row) for row in rows]

    async def begin_transaction(self) -> Any:
        """Begin a database transaction."""
        if not self._session_factory:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        session = self._session_factory()
        transaction = await session.begin()
        return {"session": session, "transaction": transaction}

    async def commit_transaction(self, transaction: Any) -> None:
        """Commit a database transaction."""
        try:
            await transaction["transaction"].commit()
            await transaction["session"].close()
        except Exception as e:
            await transaction["session"].close()
            raise InfrastructureError(
                f"Transaction commit failed: {e}",
                error_code="COMMIT_FAILED",
                cause=e,
            )

    async def rollback_transaction(self, transaction: Any) -> None:
        """Rollback a database transaction."""
        try:
            await transaction["transaction"].rollback()
            await transaction["session"].close()
        except Exception as e:
            await transaction["session"].close()
            raise InfrastructureError(
                f"Transaction rollback failed: {e}",
                error_code="ROLLBACK_FAILED",
                cause=e,
            )


@adapter("mongodb")
class MongoDBAdapter(DatabaseAdapter):
    """MongoDB adapter for NoSQL document database."""

    def __init__(self, config: DatabaseConfiguration) -> None:
        super().__init__(config)
        self._client: Optional[Any] = None
        self._database: Optional[Any] = None

    async def _create_connection(self) -> Any:
        """Create MongoDB client and database connection."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise InfrastructureError(
                "Motor (MongoDB async driver) not installed. Install with: pip install 'enterprise-adapters[database]'",
                error_code="DEPENDENCY_MISSING",
            )

        # Build connection URL
        if self.config.connection_string:
            url = self.config.connection_string
        else:
            if not all([self.config.host, self.config.username, self.config.password]):
                raise InfrastructureError(
                    "MongoDB connection requires either connection_string or host/username/password",
                    error_code="CONFIGURATION_ERROR",
                )

            url = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port or 27017}/{self.db_config.database}"

        # Create client
        self._client = AsyncIOMotorClient(
            url,
            maxPoolSize=self.config.pool_size,
            serverSelectionTimeoutMS=self.config.timeout * 1000,
        )

        # Get database
        self._database = self._client[self.db_config.database]

        return self._client

    async def _close_connection(self) -> None:
        """Close MongoDB client."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None

    async def _test_connection(self) -> bool:
        """Test the MongoDB connection."""
        if not self._client:
            return False

        try:
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a MongoDB operation (not applicable for traditional queries)."""
        raise NotImplementedError("MongoDB uses collection-based operations, not SQL queries")

    async def execute_many(self, query: str, parameters_list: List[Dict[str, Any]]) -> Any:
        """Execute multiple MongoDB operations."""
        raise NotImplementedError("MongoDB uses collection-based operations, not SQL queries")

    async def fetch_one(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single document (not applicable for traditional queries)."""
        raise NotImplementedError("MongoDB uses collection-based operations, not SQL queries")

    async def fetch_all(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all documents (not applicable for traditional queries)."""
        raise NotImplementedError("MongoDB uses collection-based operations, not SQL queries")

    async def begin_transaction(self) -> Any:
        """Begin a MongoDB transaction."""
        if not self._client:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        session = await self._client.start_session()
        session.start_transaction()
        return session

    async def commit_transaction(self, transaction: Any) -> None:
        """Commit a MongoDB transaction."""
        try:
            await transaction.commit_transaction()
            await transaction.end_session()
        except Exception as e:
            await transaction.end_session()
            raise InfrastructureError(
                f"Transaction commit failed: {e}",
                error_code="COMMIT_FAILED",
                cause=e,
            )

    async def rollback_transaction(self, transaction: Any) -> None:
        """Rollback a MongoDB transaction."""
        try:
            await transaction.abort_transaction()
            await transaction.end_session()
        except Exception as e:
            await transaction.end_session()
            raise InfrastructureError(
                f"Transaction rollback failed: {e}",
                error_code="ROLLBACK_FAILED",
                cause=e,
            )

    # MongoDB-specific methods
    def get_collection(self, collection_name: str) -> Any:
        """Get a MongoDB collection."""
        if not self._database:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        return self._database[collection_name]

    async def insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a document into a collection."""
        if not self._database:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        async with self.with_retry():
            try:
                result = await self._database[collection].insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                raise InfrastructureError(
                    f"Document insert failed: {e}",
                    error_code="INSERT_FAILED",
                    details={"collection": collection},
                    cause=e,
                )

    async def find_documents(
        self,
        collection: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find documents in a collection."""
        if not self._database:
            raise InfrastructureError("Database not connected", error_code="NOT_CONNECTED")

        async with self.with_retry():
            try:
                cursor = self._database[collection].find(filter_dict or {})
                if limit:
                    cursor = cursor.limit(limit)

                documents = []
                async for doc in cursor:
                    # Convert ObjectId to string
                    doc["_id"] = str(doc["_id"])
                    documents.append(doc)

                return documents
            except Exception as e:
                raise InfrastructureError(
                    f"Document find failed: {e}",
                    error_code="FIND_FAILED",
                    details={"collection": collection, "filter": filter_dict},
                    cause=e,
                )
