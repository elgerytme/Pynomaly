"""MongoDB Adapter Implementation for Pynomaly."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import motor.motor_asyncio
    from pymongo import ASCENDING, DESCENDING
    from pymongo.errors import DuplicateKeyError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

from pynomaly.infrastructure.config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)


class MongoDBAdapter:
    """MongoDB database adapter for Pynomaly with production-ready features."""

    def __init__(self, config: DatabaseConfig):
        """Initialize MongoDB adapter with configuration."""
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB dependencies not available. Install with: pip install motor pymongo")
        
        self.config = config
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.get_connection_url())
        self.db = self.client[self.config.database]
        self._connected = False
        logger.info(f"MongoDB adapter initialized for database: {self.config.database}")

    async def connect(self) -> None:
        """Establish connection to MongoDB database."""
        try:
            await self.client.admin.command('ping')
            self._connected = True
            logger.info("Successfully connected to MongoDB")
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
        """Ping the MongoDB server to check connection."""
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False

    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database."""
        try:
            server_info = await self.client.server_info()
            return {
                "connected": self._connected,
                "database_name": self.config.database,
                "server_version": server_info.get("version", "unknown"),
                "server_info": server_info
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                "connected": False,
                "database_name": self.config.database,
                "error": str(e)
            }

    async def insert_document(self, collection: str, document: Dict[str, Any], session=None) -> Any:
        """Insert document into a specified collection."""
        try:
            col = self.db[collection]
            result = await col.insert_one(document, session=session)
            logger.debug(f"Document inserted into {collection}: {result.inserted_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to insert document into {collection}: {e}")
            raise

    async def find_document(self, collection: str, filter_dict: Dict[str, Any], session=None) -> Optional[Dict[str, Any]]:
        """Find a single document in a collection using a filter."""
        try:
            col = self.db[collection]
            result = await col.find_one(filter_dict, session=session)
            logger.debug(f"Document found in {collection}: {result is not None}")
            return result
        except Exception as e:
            logger.error(f"Failed to find document in {collection}: {e}")
            raise

    async def find_documents(self, collection: str, filter_dict: Dict[str, Any], session=None) -> List[Dict[str, Any]]:
        """Find documents in a collection using a filter."""
        try:
            col = self.db[collection]
            cursor = col.find(filter_dict, session=session)
            results = await cursor.to_list(length=None)
            logger.debug(f"Found {len(results)} documents in {collection}")
            return results
        except Exception as e:
            logger.error(f"Failed to find documents in {collection}: {e}")
            raise

    async def update_document(self, collection: str, filter_dict: Dict[str, Any], update: Dict[str, Any], session=None) -> Any:
        """Update a document in the collection."""
        try:
            col = self.db[collection]
            result = await col.update_one(filter_dict, update, session=session)
            logger.debug(f"Document updated in {collection}: {result.modified_count} modified")
            return result
        except Exception as e:
            logger.error(f"Failed to update document in {collection}: {e}")
            raise

    async def delete_document(self, collection: str, filter_dict: Dict[str, Any], session=None) -> Any:
        """Delete a document from the collection."""
        try:
            col = self.db[collection]
            result = await col.delete_one(filter_dict, session=session)
            logger.debug(f"Document deleted from {collection}: {result.deleted_count} deleted")
            return result
        except Exception as e:
            logger.error(f"Failed to delete document from {collection}: {e}")
            raise

    async def aggregate(self, collection: str, pipeline: List[Dict[str, Any]], session=None) -> List[Dict[str, Any]]:
        """Perform an aggregation on the collection."""
        try:
            col = self.db[collection]
            cursor = col.aggregate(pipeline, session=session)
            results = await cursor.to_list(length=None)
            logger.debug(f"Aggregation on {collection} returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to perform aggregation on {collection}: {e}")
            raise

    async def create_index(self, collection: str, keys, **kwargs) -> None:
        """Create an index on the given collection."""
        try:
            col = self.db[collection]
            if isinstance(keys, str):
                keys = [(keys, ASCENDING)]
            await col.create_index(keys, **kwargs)
            logger.debug(f"Index created on {collection}: {keys}")
        except Exception as e:
            logger.error(f"Failed to create index on {collection}: {e}")
            raise

    async def get_indexes(self, collection: str) -> List[Dict[str, Any]]:
        """List indexes on a given collection."""
        try:
            col = self.db[collection]
            cursor = col.list_indexes()
            indexes = await cursor.to_list(length=None)
            logger.debug(f"Retrieved {len(indexes)} indexes from {collection}")
            return indexes
        except Exception as e:
            logger.error(f"Failed to get indexes from {collection}: {e}")
            raise

    async def drop_database(self) -> None:
        """Drops the current database."""
        try:
            await self.client.drop_database(self.config.database)
            logger.warning(f"Database {self.config.database} dropped")
        except Exception as e:
            logger.error(f"Failed to drop database {self.config.database}: {e}")
            raise

    async def start_transaction(self):
        """Start a MongoDB transaction session."""
        try:
            session = await self.client.start_session()
            logger.debug("Transaction session started")
            return session
        except Exception as e:
            logger.error(f"Failed to start transaction: {e}")
            raise

    async def create_collection(self, collection_name: str, **options) -> None:
        """Create a new collection with options."""
        try:
            await self.db.create_collection(collection_name, **options)
            logger.info(f"Collection {collection_name} created")
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    async def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        try:
            await self.db.drop_collection(collection_name)
            logger.info(f"Collection {collection_name} dropped")
        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {e}")
            raise

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        try:
            stats = await self.db.command("collStats", collection_name)
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for collection {collection_name}: {e}")
            raise

    async def ensure_indexes(self, collection_name: str, index_definitions: List[Dict[str, Any]]) -> None:
        """Ensure indexes exist on a collection."""
        try:
            col = self.db[collection_name]
            for index_def in index_definitions:
                keys = index_def.get("keys", [])
                options = index_def.get("options", {})
                await col.create_index(keys, **options)
            logger.info(f"Ensured {len(index_definitions)} indexes on {collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure indexes on {collection_name}: {e}")
            raise

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected to MongoDB."""
        return self._connected

    @property 
    def database_name(self) -> str:
        """Get the current database name."""
        return self.config.database

    @property
    def connection_url(self) -> str:
        """Get the connection URL (without credentials)."""
        url = self.config.get_connection_url()
        # Remove credentials for logging
        if "@" in url:
            scheme, rest = url.split("://", 1)
            if "@" in rest:
                credentials, host_part = rest.split("@", 1)
                return f"{scheme}://{host_part}"
        return url
