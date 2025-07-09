"""MongoDB Adapter Implementation for Pynomaly."""

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

class MongoDBAdapter:
    """MongoDB database adapter for Pynomaly."""

    def __init__(self, config):
        """Initialize MongoDB adapter with config."""
        self.config = config
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.get_connection_url())
        self.db = self.client[self.config.database]

    async def connect(self):
        """Establish connection to MongoDB database."""
        # Verify connection by pinging the server
        await self.client.admin.command('ping')

    async def close(self):
        """Close MongoDB connection."""
        self.client.close()

    async def ping(self) -> bool:
        """Ping the MongoDB server to check connection."""
        try:
            await self.client.admin.command('ping')
            return True
        except Exception:
            return False

    async def get_database_info(self):
        """Get information about the current database."""
        return {
            "connected": True,
            "database_name": self.config.database
        }

    async def insert_document(self, collection: str, document: dict, session=None):
        """Insert document into a specified collection."""
        col = self.db[collection]
        return await col.insert_one(document, session=session)

    async def find_document(self, collection: str, filter: dict, session=None):
        """Find a single document in a collection using a filter."""
        col = self.db[collection]
        return await col.find_one(filter, session=session)

    async def find_documents(self, collection: str, filter: dict, session=None):
        """Find documents in a collection using a filter."""
        col = self.db[collection]
        cursor = col.find(filter, session=session)
        return await cursor.to_list(length=None)

    async def update_document(self, collection: str, filter: dict, update: dict, session=None):
        """Update a document in the collection."""
        col = self.db[collection]
        return await col.update_one(filter, update, session=session)

    async def delete_document(self, collection: str, filter: dict, session=None):
        """Delete a document from the collection."""
        col = self.db[collection]
        return await col.delete_one(filter, session=session)

    async def aggregate(self, collection: str, pipeline: list, session=None):
        """Perform an aggregation on the collection."""
        col = self.db[collection]
        cursor = col.aggregate(pipeline, session=session)
        return await cursor.to_list(length=None)

    async def create_index(self, collection: str, keys: list, **kwargs):
        """Create an index on the given collection."""
        col = self.db[collection]
        if isinstance(keys, str):
            keys = [(keys, ASCENDING)]
        await col.create_index(keys, **kwargs)

    async def get_indexes(self, collection: str):
        """List indexes on a given collection."""
        col = self.db[collection]
        return await col.list_indexes().to_list(length=None)

    async def drop_database(self):
        """Drops the current database."""
        await self.client.drop_database(self.config.database)

    async def start_transaction(self):
        """Start a MongoDB transaction session."""
        return await self.client.start_session()
