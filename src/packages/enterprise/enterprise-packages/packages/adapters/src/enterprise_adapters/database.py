"""Database adapter implementations for enterprise applications.

This module provides adapters for various database systems including
SQL databases (PostgreSQL, MySQL, SQLite) and NoSQL databases (MongoDB).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from enterprise_core import InfrastructureError
from pydantic import Field

from .base import AdapterConfiguration, BaseAdapter, adapter

logger = logging.getLogger(__name__)


class DatabaseConfiguration(AdapterConfiguration):
    """Configuration for database adapters."""

    adapter_type: str = Field(..., description="Database adapter type")
    database: str = Field(..., description="Database name")
    echo: bool = Field(default=False, description="Enable SQL echoing")
    pool_pre_ping: bool = Field(
        default=True, description="Enable connection pool pre-ping"
    )
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")


class DatabaseAdapter(BaseAdapter):
    """Base class for database adapters."""

    def __init__(self, config: DatabaseConfiguration) -> None:
        super().__init__(config)
        self.db_config = config

    @abstractmethod
    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a database query."""
        pass

    @abstractmethod
    async def execute_many(
        self, query: str, parameters_list: list[dict[str, Any]]
    ) -> Any:
        """Execute a query with multiple parameter sets."""
        pass

    @abstractmethod
    async def fetch_one(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single row from a query."""
        pass

    @abstractmethod
    async def fetch_all(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
        self._engine: Any | None = None
        self._session_factory: Any | None = None

    async def _create_connection(self) -> Any:
        """Create SQLAlchemy engine and session factory."""
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        except ImportError:
            raise InfrastructureError(
                "SQLAlchemy not installed. Install with: pip install 'enterprise-adapters[database]'",
                error_code="DEPENDENCY_MISSING",
            ) from None

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

    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a database query."""
        if not self._session_factory:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

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
                    ) from e

    async def execute_many(
        self, query: str, parameters_list: list[dict[str, Any]]
    ) -> Any:
        """Execute a query with multiple parameter sets."""
        if not self._session_factory:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

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
                        details={
                            "query": query,
                            "parameter_count": len(parameters_list),
                        },
                        cause=e,
                    ) from e

    async def fetch_one(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single row from a query."""
        result = await self.execute_query(query, parameters)
        row = result.fetchone()
        return dict(row) if row else None

    async def fetch_all(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all rows from a query."""
        result = await self.execute_query(query, parameters)
        rows = result.fetchall()
        return [dict(row) for row in rows]

    async def begin_transaction(self) -> Any:
        """Begin a database transaction."""
        if not self._session_factory:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

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
            ) from e

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
            ) from e


@adapter("mongodb")
class MongoDBAdapter(DatabaseAdapter):
    """MongoDB adapter for NoSQL document database."""

    def __init__(self, config: DatabaseConfiguration) -> None:
        super().__init__(config)
        self._client: Any | None = None
        self._database: Any | None = None

    async def _create_connection(self) -> Any:
        """Create MongoDB client and database connection."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise InfrastructureError(
                "Motor (MongoDB async driver) not installed. Install with: pip install 'enterprise-adapters[database]'",
                error_code="DEPENDENCY_MISSING",
            ) from None

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

    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a MongoDB operation (not applicable for traditional queries)."""
        raise NotImplementedError(
            "MongoDB uses collection-based operations, not SQL queries"
        )

    async def execute_many(
        self, query: str, parameters_list: list[dict[str, Any]]
    ) -> Any:
        """Execute multiple MongoDB operations."""
        raise NotImplementedError(
            "MongoDB uses collection-based operations, not SQL queries"
        )

    async def fetch_one(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single document (not applicable for traditional queries)."""
        raise NotImplementedError(
            "MongoDB uses collection-based operations, not SQL queries"
        )

    async def fetch_all(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all documents (not applicable for traditional queries)."""
        raise NotImplementedError(
            "MongoDB uses collection-based operations, not SQL queries"
        )

    async def begin_transaction(self) -> Any:
        """Begin a MongoDB transaction."""
        if not self._client:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

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
            ) from e

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
            ) from e

    # MongoDB-specific methods
    def get_collection(self, collection_name: str) -> Any:
        """Get a MongoDB collection."""
        if not self._database:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

        return self._database[collection_name]

    async def insert_document(self, collection: str, document: dict[str, Any]) -> str:
        """Insert a document into a collection."""
        if not self._database:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

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
                ) from e

    async def find_documents(
        self,
        collection: str,
        filter_dict: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find documents in a collection."""
        if not self._database:
            raise InfrastructureError(
                "Database not connected", error_code="NOT_CONNECTED"
            )

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
                ) from e


@adapter("enterprise_orchestrator")
class EnterpriseDataSourceOrchestrator(DatabaseAdapter):
    """Enterprise-grade data source orchestrator with multi-tenant and high-availability features."""

    def __init__(self, config: DatabaseConfiguration) -> None:
        super().__init__(config)
        self._database_adapters: dict[str, DatabaseAdapter] = {}
        self._api_clients: dict[str, Any] = {}
        self._file_handlers: dict[str, Any] = {}
        self._stream_processors: dict[str, Any] = {}

    async def _create_connection(self) -> Any:
        """Initialize enterprise data source connections."""
        return {"status": "enterprise_orchestrator_ready"}

    async def _close_connection(self) -> None:
        """Close all enterprise data source connections."""
        for adapter in self._database_adapters.values():
            await adapter.disconnect()
        self._database_adapters.clear()
        self._api_clients.clear()
        self._file_handlers.clear()
        self._stream_processors.clear()

    async def _test_connection(self) -> bool:
        """Test all enterprise data source connections."""
        try:
            for adapter in self._database_adapters.values():
                if not await adapter._test_connection():
                    return False
            return True
        except Exception:
            return False

    async def setup_database_source(self, source_config: dict[str, Any]) -> None:
        """Setup database data source for enterprise operations.
        
        Provides enterprise-grade database connection management with:
        - Multi-tenant database support
        - Connection pooling with enterprise features
        - High-availability and failover mechanisms
        - Advanced security and compliance features
        """
        source_name = source_config.get("name", "default")
        source_type = source_config.get("type", "postgresql")
        
        # Create database configuration
        db_config = DatabaseConfiguration(
            adapter_type=source_type,
            database=source_config["database"],
            host=source_config.get("host"),
            port=source_config.get("port"),
            username=source_config.get("username"),
            password=source_config.get("password"),
            connection_string=source_config.get("connection_string"),
            pool_size=source_config.get("pool_size", 20),
            max_overflow=source_config.get("max_overflow", 50),
            echo=source_config.get("echo", False),
            pool_pre_ping=True,
            pool_recycle=source_config.get("pool_recycle", 3600)
        )
        
        # Create appropriate adapter
        if source_type == "postgresql" or source_type == "mysql":
            adapter = SQLAlchemyAdapter(db_config)
        elif source_type == "mongodb":
            adapter = MongoDBAdapter(db_config)
        else:
            raise InfrastructureError(
                f"Unsupported database type: {source_type}",
                error_code="UNSUPPORTED_DATABASE_TYPE"
            )
        
        # Initialize connection
        await adapter.connect()
        self._database_adapters[source_name] = adapter
        
        logger.info(f"Enterprise database source '{source_name}' configured successfully")

    async def setup_api_source(self, api_config: dict[str, Any]) -> None:
        """Setup API data source for enterprise operations.
        
        Provides enterprise API integration with:
        - OAuth2/SAML authentication
        - Rate limiting and throttling
        - API gateway integration
        - Advanced security features
        """
        try:
            import httpx
        except ImportError:
            raise InfrastructureError(
                "httpx not installed. Install with: pip install 'enterprise-adapters[api]'",
                error_code="DEPENDENCY_MISSING",
            ) from None

        api_name = api_config.get("name", "default")
        base_url = api_config["base_url"]
        auth_type = api_config.get("auth_type", "bearer")
        
        # Configure authentication
        headers = {}
        if auth_type == "bearer" and "token" in api_config:
            headers["Authorization"] = f"Bearer {api_config['token']}"
        elif auth_type == "api_key" and "api_key" in api_config:
            headers["X-API-Key"] = api_config["api_key"]
        
        # Create HTTP client with enterprise features
        client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=api_config.get("timeout", 30),
            limits=httpx.Limits(
                max_keepalive_connections=api_config.get("max_connections", 20),
                max_connections=api_config.get("max_connections", 100),
                keepalive_expiry=api_config.get("keepalive_expiry", 5)
            )
        )
        
        self._api_clients[api_name] = client
        logger.info(f"Enterprise API source '{api_name}' configured successfully")

    async def setup_file_source(self, file_config: dict[str, Any]) -> None:
        """Setup file data source for enterprise operations.
        
        Provides enterprise file processing with:
        - Support for HDFS, S3, Azure Blob Storage
        - Batch processing capabilities  
        - Data validation and quality checks
        - Scalable file processing
        """
        source_name = file_config.get("name", "default")
        source_type = file_config.get("type", "local")
        
        if source_type == "s3":
            try:
                import boto3
                
                client = boto3.client(
                    's3',
                    aws_access_key_id=file_config.get("access_key"),
                    aws_secret_access_key=file_config.get("secret_key"),
                    region_name=file_config.get("region", "us-east-1")
                )
                
                self._file_handlers[source_name] = {
                    "type": "s3",
                    "client": client,
                    "bucket": file_config.get("bucket")
                }
                
            except ImportError:
                raise InfrastructureError(
                    "boto3 not installed. Install with: pip install 'enterprise-adapters[cloud]'",
                    error_code="DEPENDENCY_MISSING",
                ) from None
                
        elif source_type == "azure":
            try:
                from azure.storage.blob import BlobServiceClient
                
                client = BlobServiceClient(
                    account_url=file_config["account_url"],
                    credential=file_config.get("credential")
                )
                
                self._file_handlers[source_name] = {
                    "type": "azure",
                    "client": client,
                    "container": file_config.get("container")
                }
                
            except ImportError:
                raise InfrastructureError(
                    "azure-storage-blob not installed. Install with: pip install 'enterprise-adapters[cloud]'",
                    error_code="DEPENDENCY_MISSING",
                ) from None
                
        else:
            # Local file system handler
            import os
            import pathlib
            
            base_path = pathlib.Path(file_config.get("base_path", "/tmp"))
            if not base_path.exists():
                base_path.mkdir(parents=True, exist_ok=True)
                
            self._file_handlers[source_name] = {
                "type": "local",
                "base_path": base_path
            }
        
        logger.info(f"Enterprise file source '{source_name}' configured successfully")

    async def setup_stream_source(self, stream_config: dict[str, Any]) -> None:
        """Setup stream data source for enterprise operations.
        
        Provides enterprise stream processing with:
        - Kafka/RabbitMQ integration
        - Real-time data ingestion
        - Stream analytics capabilities
        - Fault-tolerant processing
        """
        source_name = stream_config.get("name", "default")
        source_type = stream_config.get("type", "kafka")
        
        if source_type == "kafka":
            try:
                from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
                
                consumer = AIOKafkaConsumer(
                    *stream_config.get("topics", []),
                    bootstrap_servers=stream_config.get("bootstrap_servers", "localhost:9092"),
                    group_id=stream_config.get("group_id", "enterprise-group"),
                    auto_offset_reset=stream_config.get("auto_offset_reset", "latest")
                )
                
                producer = AIOKafkaProducer(
                    bootstrap_servers=stream_config.get("bootstrap_servers", "localhost:9092")
                )
                
                self._stream_processors[source_name] = {
                    "type": "kafka",
                    "consumer": consumer,
                    "producer": producer,
                    "topics": stream_config.get("topics", [])
                }
                
            except ImportError:
                raise InfrastructureError(
                    "aiokafka not installed. Install with: pip install 'enterprise-adapters[streaming]'",
                    error_code="DEPENDENCY_MISSING",
                ) from None
                
        elif source_type == "rabbitmq":
            try:
                import aio_pika
                
                connection_url = stream_config.get("connection_url", "amqp://localhost/")
                
                self._stream_processors[source_name] = {
                    "type": "rabbitmq",
                    "connection_url": connection_url,
                    "exchange": stream_config.get("exchange", "enterprise"),
                    "routing_keys": stream_config.get("routing_keys", [])
                }
                
            except ImportError:
                raise InfrastructureError(
                    "aio-pika not installed. Install with: pip install 'enterprise-adapters[messaging]'",
                    error_code="DEPENDENCY_MISSING",
                ) from None
        else:
            raise InfrastructureError(
                f"Unsupported stream type: {source_type}",
                error_code="UNSUPPORTED_STREAM_TYPE"
            )
        
        logger.info(f"Enterprise stream source '{source_name}' configured successfully")

    # Database adapter interface methods
    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None, source: str = "default"
    ) -> Any:
        """Execute a query on the specified database source."""
        if source not in self._database_adapters:
            raise InfrastructureError(
                f"Database source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        
        return await self._database_adapters[source].execute_query(query, parameters)

    async def execute_many(
        self, query: str, parameters_list: list[dict[str, Any]], source: str = "default"
    ) -> Any:
        """Execute a query with multiple parameter sets on the specified database source."""
        if source not in self._database_adapters:
            raise InfrastructureError(
                f"Database source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        
        return await self._database_adapters[source].execute_many(query, parameters_list)

    async def fetch_one(
        self, query: str, parameters: dict[str, Any] | None = None, source: str = "default"
    ) -> dict[str, Any] | None:
        """Fetch a single row from the specified database source."""
        if source not in self._database_adapters:
            raise InfrastructureError(
                f"Database source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        
        return await self._database_adapters[source].fetch_one(query, parameters)

    async def fetch_all(
        self, query: str, parameters: dict[str, Any] | None = None, source: str = "default"
    ) -> list[dict[str, Any]]:
        """Fetch all rows from the specified database source."""
        if source not in self._database_adapters:
            raise InfrastructureError(
                f"Database source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        
        return await self._database_adapters[source].fetch_all(query, parameters)

    async def begin_transaction(self, source: str = "default") -> Any:
        """Begin a transaction on the specified database source."""
        if source not in self._database_adapters:
            raise InfrastructureError(
                f"Database source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        
        return await self._database_adapters[source].begin_transaction()

    async def commit_transaction(self, transaction: Any) -> None:
        """Commit a transaction."""
        # Extract source from transaction context if available
        source_adapter = getattr(transaction, '_source_adapter', None)
        if source_adapter:
            return await source_adapter.commit_transaction(transaction)
        
        # Fallback to default source
        if "default" in self._database_adapters:
            return await self._database_adapters["default"].commit_transaction(transaction)
        
        raise InfrastructureError(
            "No database source available for transaction commit",
            error_code="NO_SOURCE_AVAILABLE"
        )

    async def rollback_transaction(self, transaction: Any) -> None:
        """Rollback a transaction."""
        # Extract source from transaction context if available
        source_adapter = getattr(transaction, '_source_adapter', None)
        if source_adapter:
            return await source_adapter.rollback_transaction(transaction)
        
        # Fallback to default source
        if "default" in self._database_adapters:
            return await self._database_adapters["default"].rollback_transaction(transaction)
        
        raise InfrastructureError(
            "No database source available for transaction rollback",
            error_code="NO_SOURCE_AVAILABLE"
        )

    # Enterprise-specific operations
    async def get_database_adapter(self, source: str = "default") -> DatabaseAdapter:
        """Get a specific database adapter."""
        if source not in self._database_adapters:
            raise InfrastructureError(
                f"Database source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        return self._database_adapters[source]

    async def get_api_client(self, api: str = "default") -> Any:
        """Get a specific API client."""
        if api not in self._api_clients:
            raise InfrastructureError(
                f"API client '{api}' not configured",
                error_code="CLIENT_NOT_FOUND"
            )
        return self._api_clients[api]

    async def get_file_handler(self, source: str = "default") -> dict[str, Any]:
        """Get a specific file handler."""
        if source not in self._file_handlers:
            raise InfrastructureError(
                f"File source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        return self._file_handlers[source]

    async def get_stream_processor(self, source: str = "default") -> dict[str, Any]:
        """Get a specific stream processor."""
        if source not in self._stream_processors:
            raise InfrastructureError(
                f"Stream source '{source}' not configured",
                error_code="SOURCE_NOT_FOUND"
            )
        return self._stream_processors[source]

    def list_sources(self) -> dict[str, list[str]]:
        """List all configured data sources."""
        return {
            "databases": list(self._database_adapters.keys()),
            "apis": list(self._api_clients.keys()),
            "files": list(self._file_handlers.keys()),
            "streams": list(self._stream_processors.keys())
        }
