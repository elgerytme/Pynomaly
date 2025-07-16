"""MongoDB adapter for persistence operations."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import motor.motor_asyncio
from pymongo import IndexModel
from pymongo.errors import DuplicateKeyError

from monorepo.infrastructure.config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)


class MongoDBAdapter:
    """MongoDB adapter for async database operations."""

    def __init__(self, config: DatabaseConfig):
        """Initialize MongoDB adapter.

        Args:
            config: Database configuration
        """
        self.config = config
        self.client: motor.motor_asyncio.AsyncIOMotorClient | None = None
        self.database: motor.motor_asyncio.AsyncIOMotorDatabase | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to MongoDB database."""
        try:
            connection_url = self.config.get_connection_url()
            self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_url)

            # Get database name from config
            database_name = self.config.database or "monorepo"
            self.database = self.client[database_name]

            # Test connection
            await self.ping()
            self._connected = True
            logger.info(f"Connected to MongoDB database: {database_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")

    async def ping(self) -> bool:
        """Test database connection.

        Returns:
            True if connection is successful
        """
        try:
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False

    async def drop_database(self) -> None:
        """Drop the current database."""
        if self.database:
            await self.client.drop_database(self.database.name)
            logger.info(f"Dropped database: {self.database.name}")

    async def get_database_info(self) -> dict[str, Any]:
        """Get database information.

        Returns:
            Database information dictionary
        """
        if not self.database:
            return {"connected": False}

        try:
            stats = await self.database.command("dbstats")
            return {
                "connected": True,
                "database_name": self.database.name,
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"connected": False, "error": str(e)}

    async def insert_document(
        self,
        collection_name: str,
        document: dict[str, Any],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> motor.motor_asyncio.AsyncIOMotorCollection.InsertOneResult:
        """Insert a document into a collection.

        Args:
            collection_name: Name of the collection
            document: Document to insert
            session: Optional transaction session

        Returns:
            Insert result
        """
        try:
            collection = self.database[collection_name]
            result = await collection.insert_one(document, session=session)
            return result
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error in {collection_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to insert document in {collection_name}: {e}")
            raise

    async def find_document(
        self,
        collection_name: str,
        filter_dict: dict[str, Any],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> dict[str, Any] | None:
        """Find a single document in a collection.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            session: Optional transaction session

        Returns:
            Found document or None
        """
        try:
            collection = self.database[collection_name]
            result = await collection.find_one(filter_dict, session=session)
            return result
        except Exception as e:
            logger.error(f"Failed to find document in {collection_name}: {e}")
            raise

    async def find_documents(
        self,
        collection_name: str,
        filter_dict: dict[str, Any],
        limit: int | None = None,
        skip: int | None = None,
        sort: list[tuple] | None = None,
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> list[dict[str, Any]]:
        """Find multiple documents in a collection.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: Sort specification
            session: Optional transaction session

        Returns:
            List of found documents
        """
        try:
            collection = self.database[collection_name]
            cursor = collection.find(filter_dict, session=session)

            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            results = await cursor.to_list(length=limit)
            return results
        except Exception as e:
            logger.error(f"Failed to find documents in {collection_name}: {e}")
            raise

    async def update_document(
        self,
        collection_name: str,
        filter_dict: dict[str, Any],
        update_dict: dict[str, Any],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> motor.motor_asyncio.AsyncIOMotorCollection.UpdateResult:
        """Update a document in a collection.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            update_dict: Update specification
            session: Optional transaction session

        Returns:
            Update result
        """
        try:
            collection = self.database[collection_name]
            result = await collection.update_one(
                filter_dict, update_dict, session=session
            )
            return result
        except Exception as e:
            logger.error(f"Failed to update document in {collection_name}: {e}")
            raise

    async def delete_document(
        self,
        collection_name: str,
        filter_dict: dict[str, Any],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> motor.motor_asyncio.AsyncIOMotorCollection.DeleteResult:
        """Delete a document from a collection.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            session: Optional transaction session

        Returns:
            Delete result
        """
        try:
            collection = self.database[collection_name]
            result = await collection.delete_one(filter_dict, session=session)
            return result
        except Exception as e:
            logger.error(f"Failed to delete document in {collection_name}: {e}")
            raise

    async def aggregate(
        self,
        collection_name: str,
        pipeline: list[dict[str, Any]],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> list[dict[str, Any]]:
        """Execute an aggregation pipeline.

        Args:
            collection_name: Name of the collection
            pipeline: Aggregation pipeline
            session: Optional transaction session

        Returns:
            Aggregation results
        """
        try:
            collection = self.database[collection_name]
            cursor = collection.aggregate(pipeline, session=session)
            results = await cursor.to_list(length=None)
            return results
        except Exception as e:
            logger.error(f"Failed to execute aggregation in {collection_name}: {e}")
            raise

    async def create_index(
        self,
        collection_name: str,
        keys: Any,
        unique: bool = False,
        background: bool = True,
    ) -> str:
        """Create an index on a collection.

        Args:
            collection_name: Name of the collection
            keys: Index keys specification
            unique: Whether to create unique index
            background: Whether to create index in background

        Returns:
            Index name
        """
        try:
            collection = self.database[collection_name]
            index_name = await collection.create_index(
                keys, unique=unique, background=background
            )
            return index_name
        except Exception as e:
            logger.error(f"Failed to create index in {collection_name}: {e}")
            raise

    async def get_indexes(self, collection_name: str) -> list[dict[str, Any]]:
        """Get all indexes for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of index information
        """
        try:
            collection = self.database[collection_name]
            indexes = await collection.list_indexes().to_list(length=None)
            return indexes
        except Exception as e:
            logger.error(f"Failed to get indexes for {collection_name}: {e}")
            raise

    @asynccontextmanager
    async def start_transaction(
        self,
    ) -> AsyncGenerator[motor.motor_asyncio.AsyncIOMotorClientSession, None]:
        """Start a database transaction.

        Yields:
            Transaction session
        """
        async with await self.client.start_session() as session:
            async with session.start_transaction():
                yield session

    def is_connected(self) -> bool:
        """Check if adapter is connected to database.

        Returns:
            True if connected
        """
        return self._connected

    def get_collection(
        self, collection_name: str
    ) -> motor.motor_asyncio.AsyncIOMotorCollection:
        """Get a collection reference.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection reference
        """
        if not self.database:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]

    async def count_documents(
        self,
        collection_name: str,
        filter_dict: dict[str, Any],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> int:
        """Count documents in a collection.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            session: Optional transaction session

        Returns:
            Number of documents
        """
        try:
            collection = self.database[collection_name]
            count = await collection.count_documents(filter_dict, session=session)
            return count
        except Exception as e:
            logger.error(f"Failed to count documents in {collection_name}: {e}")
            raise

    async def ensure_indexes(
        self, collection_name: str, indexes: list[IndexModel]
    ) -> None:
        """Ensure indexes exist on a collection.

        Args:
            collection_name: Name of the collection
            indexes: List of index models
        """
        try:
            collection = self.database[collection_name]
            if indexes:
                await collection.create_indexes(indexes)
                logger.info(f"Ensured {len(indexes)} indexes for {collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure indexes for {collection_name}: {e}")
            raise

    async def bulk_write(
        self,
        collection_name: str,
        operations: list[Any],
        session: motor.motor_asyncio.AsyncIOMotorClientSession | None = None,
    ) -> motor.motor_asyncio.AsyncIOMotorCollection.BulkWriteResult:
        """Execute bulk write operations.

        Args:
            collection_name: Name of the collection
            operations: List of write operations
            session: Optional transaction session

        Returns:
            Bulk write result
        """
        try:
            collection = self.database[collection_name]
            result = await collection.bulk_write(operations, session=session)
            return result
        except Exception as e:
            logger.error(f"Failed to execute bulk write in {collection_name}: {e}")
            raise
