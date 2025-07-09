"""Tests for MongoDB adapter implementation."""

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.infrastructure.config.database_config import DatabaseConfig, DatabaseType
from pynomaly.infrastructure.persistence.mongodb_adapter import MongoDBAdapter


class TestMongoDBAdapter:
    """Test MongoDB adapter functionality."""
    
    @pytest.fixture
    def mongodb_config(self):
        """MongoDB configuration for testing."""
        return DatabaseConfig(
            db_type=DatabaseType.MONGODB,
            host="localhost",
            port=27017,
            database="pynomaly_test",
            username="test_user",
            password="test_password"
        )
    
    @pytest.fixture
    async def mongodb_adapter(self, mongodb_config):
        """MongoDB adapter instance."""
        adapter = MongoDBAdapter(mongodb_config)
        await adapter.connect()
        
        # Clean up any existing test data
        await adapter.drop_database()
        
        yield adapter
        
        # Clean up after test
        await adapter.drop_database()
        await adapter.close()
    
    @pytest.mark.asyncio
    async def test_mongodb_connection(self, mongodb_adapter):
        """Test MongoDB connection establishment."""
        assert await mongodb_adapter.ping() is True
        
        # Test database creation
        info = await mongodb_adapter.get_database_info()
        assert info["connected"] is True
        assert info["database_name"] == "pynomaly_test"
    
    @pytest.mark.asyncio
    async def test_document_insertion(self, mongodb_adapter):
        """Test document insertion with validation."""
        test_document = {
            "id": str(uuid4()),
            "name": "test_document",
            "data": {"key": "value"},
            "timestamp": datetime.utcnow()
        }
        
        # Insert document
        result = await mongodb_adapter.insert_document("test_collection", test_document)
        assert result.acknowledged is True
        assert result.inserted_id is not None
        
        # Verify document exists
        retrieved = await mongodb_adapter.find_document("test_collection", {"id": test_document["id"]})
        assert retrieved is not None
        assert retrieved["name"] == "test_document"
        assert retrieved["data"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_document_querying(self, mongodb_adapter):
        """Test document querying with filters."""
        # Insert test documents
        documents = [
            {"id": str(uuid4()), "type": "detector", "algorithm": "isolation_forest", "score": 0.85},
            {"id": str(uuid4()), "type": "detector", "algorithm": "one_class_svm", "score": 0.92},
            {"id": str(uuid4()), "type": "dataset", "name": "test_dataset", "rows": 1000}
        ]
        
        for doc in documents:
            await mongodb_adapter.insert_document("test_collection", doc)
        
        # Test filtering by type
        detectors = await mongodb_adapter.find_documents("test_collection", {"type": "detector"})
        assert len(detectors) == 2
        
        # Test filtering by algorithm
        isolation_forest = await mongodb_adapter.find_documents(
            "test_collection", 
            {"algorithm": "isolation_forest"}
        )
        assert len(isolation_forest) == 1
        assert isolation_forest[0]["score"] == 0.85
        
        # Test range queries
        high_score = await mongodb_adapter.find_documents(
            "test_collection", 
            {"score": {"$gte": 0.9}}
        )
        assert len(high_score) == 1
        assert high_score[0]["algorithm"] == "one_class_svm"
    
    @pytest.mark.asyncio
    async def test_document_updates(self, mongodb_adapter):
        """Test document updates and modifications."""
        test_document = {
            "id": str(uuid4()),
            "name": "update_test",
            "status": "created",
            "version": 1
        }
        
        # Insert document
        await mongodb_adapter.insert_document("test_collection", test_document)
        
        # Update document
        update_result = await mongodb_adapter.update_document(
            "test_collection",
            {"id": test_document["id"]},
            {"$set": {"status": "trained", "version": 2}}
        )
        assert update_result.modified_count == 1
        
        # Verify update
        updated = await mongodb_adapter.find_document("test_collection", {"id": test_document["id"]})
        assert updated["status"] == "trained"
        assert updated["version"] == 2
        assert updated["name"] == "update_test"  # Unchanged field
    
    @pytest.mark.asyncio
    async def test_document_deletion(self, mongodb_adapter):
        """Test document deletion."""
        test_document = {
            "id": str(uuid4()),
            "name": "delete_test",
            "temporary": True
        }
        
        # Insert document
        await mongodb_adapter.insert_document("test_collection", test_document)
        
        # Verify existence
        found = await mongodb_adapter.find_document("test_collection", {"id": test_document["id"]})
        assert found is not None
        
        # Delete document
        delete_result = await mongodb_adapter.delete_document(
            "test_collection",
            {"id": test_document["id"]}
        )
        assert delete_result.deleted_count == 1
        
        # Verify deletion
        not_found = await mongodb_adapter.find_document("test_collection", {"id": test_document["id"]})
        assert not_found is None
    
    @pytest.mark.asyncio
    async def test_aggregation_operations(self, mongodb_adapter):
        """Test MongoDB aggregation pipeline."""
        # Insert test data
        analytics_data = [
            {"detector_id": "det1", "score": 0.85, "timestamp": datetime.utcnow(), "anomaly": True},
            {"detector_id": "det1", "score": 0.92, "timestamp": datetime.utcnow(), "anomaly": True},
            {"detector_id": "det2", "score": 0.15, "timestamp": datetime.utcnow(), "anomaly": False},
            {"detector_id": "det2", "score": 0.88, "timestamp": datetime.utcnow(), "anomaly": True},
        ]
        
        for data in analytics_data:
            await mongodb_adapter.insert_document("analytics", data)
        
        # Test aggregation pipeline
        pipeline = [
            {"$group": {
                "_id": "$detector_id",
                "avg_score": {"$avg": "$score"},
                "anomaly_count": {"$sum": {"$cond": ["$anomaly", 1, 0]}},
                "total_count": {"$sum": 1}
            }},
            {"$sort": {"avg_score": -1}}
        ]
        
        results = await mongodb_adapter.aggregate("analytics", pipeline)
        assert len(results) == 2
        
        # Verify aggregation results
        det1_result = next(r for r in results if r["_id"] == "det1")
        assert det1_result["avg_score"] == 0.885  # (0.85 + 0.92) / 2
        assert det1_result["anomaly_count"] == 2
        assert det1_result["total_count"] == 2
    
    @pytest.mark.asyncio
    async def test_indexing_operations(self, mongodb_adapter):
        """Test index creation and management."""
        # Create indexes
        await mongodb_adapter.create_index("test_collection", "id", unique=True)
        await mongodb_adapter.create_index("test_collection", [("type", 1), ("timestamp", -1)])
        
        # Verify indexes
        indexes = await mongodb_adapter.get_indexes("test_collection")
        index_names = [idx["name"] for idx in indexes]
        
        assert "id_1" in index_names
        assert "type_1_timestamp_-1" in index_names
        
        # Test unique constraint
        doc1 = {"id": "unique_test", "name": "first"}
        doc2 = {"id": "unique_test", "name": "second"}
        
        await mongodb_adapter.insert_document("test_collection", doc1)
        
        # Second insert should fail due to unique constraint
        with pytest.raises(Exception):  # Should raise DuplicateKeyError
            await mongodb_adapter.insert_document("test_collection", doc2)
    
    @pytest.mark.asyncio
    async def test_transaction_support(self, mongodb_adapter):
        """Test MongoDB transaction support."""
        # Start transaction
        async with mongodb_adapter.start_transaction() as session:
            # Insert documents in transaction
            await mongodb_adapter.insert_document(
                "test_collection", 
                {"id": "tx_test_1", "value": 1},
                session=session
            )
            await mongodb_adapter.insert_document(
                "test_collection", 
                {"id": "tx_test_2", "value": 2},
                session=session
            )
            
            # Documents should not be visible outside transaction yet
            outside_tx = await mongodb_adapter.find_document("test_collection", {"id": "tx_test_1"})
            assert outside_tx is None
            
            # But visible within transaction
            within_tx = await mongodb_adapter.find_document(
                "test_collection", 
                {"id": "tx_test_1"}, 
                session=session
            )
            assert within_tx is not None
        
        # After transaction commit, documents should be visible
        committed = await mongodb_adapter.find_document("test_collection", {"id": "tx_test_1"})
        assert committed is not None
        assert committed["value"] == 1
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, mongodb_adapter):
        """Test connection pool management and concurrent operations."""
        # Test concurrent operations
        async def insert_operation(i):
            document = {
                "id": f"concurrent_test_{i}",
                "thread_id": i,
                "timestamp": datetime.utcnow()
            }
            return await mongodb_adapter.insert_document("concurrent_test", document)
        
        # Run concurrent operations
        tasks = [insert_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        assert len(results) == 10
        assert all(r.acknowledged for r in results)
        
        # Verify all documents were inserted
        all_docs = await mongodb_adapter.find_documents("concurrent_test", {})
        assert len(all_docs) == 10
        
        # Verify unique thread IDs
        thread_ids = [doc["thread_id"] for doc in all_docs]
        assert len(set(thread_ids)) == 10
