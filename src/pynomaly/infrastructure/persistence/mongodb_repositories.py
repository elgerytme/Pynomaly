"""MongoDB repository implementations for production scalability."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.shared.protocols.repository_protocol import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class MongoDBManager:
    """MongoDB connection and database manager."""

    def __init__(
        self,
        connection_string: str,
        database_name: str = "pynomaly_production",
        **kwargs,
    ):
        """Initialize MongoDB manager.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name
            **kwargs: Additional MongoClient parameters
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client_kwargs = kwargs
        self._client = None
        self._database = None

    @property
    def client(self) -> MongoClient:
        """Get MongoDB client."""
        if self._client is None:
            self._client = MongoClient(self.connection_string, **self.client_kwargs)
        return self._client

    @property
    def database(self) -> Database:
        """Get database."""
        if self._database is None:
            self._database = self.client[self.database_name]
        return self._database

    def create_indexes(self) -> None:
        """Create necessary indexes for optimal performance."""
        try:
            # Detector collection indexes
            detectors = self.database["detectors"]
            detectors.create_index("id", unique=True)
            detectors.create_index("name")
            detectors.create_index("algorithm_name")
            detectors.create_index("is_fitted")
            detectors.create_index("created_at")

            # Dataset collection indexes
            datasets = self.database["datasets"]
            datasets.create_index("id", unique=True)
            datasets.create_index("name")
            datasets.create_index("created_at")
            datasets.create_index("metadata")

            # Detection results collection indexes
            results = self.database["detection_results"]
            results.create_index("id", unique=True)
            results.create_index("detector_id")
            results.create_index("dataset_id")
            results.create_index("timestamp")
            results.create_index([("detector_id", 1), ("dataset_id", 1)])

            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")

    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


class MongoDBDetectorRepository(DetectorRepositoryProtocol):
    """MongoDB-backed detector repository."""

    def __init__(self, db_manager: MongoDBManager):
        """Initialize with MongoDB manager."""
        self.db_manager = db_manager
        self.collection: Collection = db_manager.database["detectors"]

    def save(self, entity: Detector) -> None:
        """Save detector to MongoDB."""
        detector_doc = {
            "id": str(entity.id),
            "name": entity.name,
            "algorithm_name": entity.algorithm_name,
            "parameters": entity.parameters,
            "is_fitted": entity.is_fitted,
            "metadata": entity.metadata,
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
        }

        self.collection.replace_one(
            {"id": str(entity.id)}, detector_doc, upsert=True
        )

    def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find detector by ID."""
        doc = self.collection.find_one({"id": str(entity_id)})
        return self._doc_to_entity(doc) if doc else None

    def find_all(self) -> list[Detector]:
        """Find all detectors."""
        docs = self.collection.find()
        return [self._doc_to_entity(doc) for doc in docs]

    def delete(self, entity_id: UUID) -> bool:
        """Delete detector by ID."""
        result = self.collection.delete_one({"id": str(entity_id)})
        return result.deleted_count > 0

    def exists(self, entity_id: UUID) -> bool:
        """Check if detector exists."""
        return self.collection.count_documents({"id": str(entity_id)}) > 0

    def count(self) -> int:
        """Count total detectors."""
        return self.collection.count_documents({})

    def find_by_name(self, name: str) -> Detector | None:
        """Find detector by name."""
        doc = self.collection.find_one({"name": name})
        return self._doc_to_entity(doc) if doc else None

    def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find detectors by algorithm."""
        docs = self.collection.find({"algorithm_name": algorithm_name})
        return [self._doc_to_entity(doc) for doc in docs]

    def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        docs = self.collection.find({"is_fitted": True})
        return [self._doc_to_entity(doc) for doc in docs]

    def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save model artifact."""
        # Use GridFS for large binary data
        from pymongo import GridFS

        fs = GridFS(self.db_manager.database)
        fs.put(artifact, filename=f"detector_model_{detector_id}")

    def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load model artifact."""
        from pymongo import GridFS

        fs = GridFS(self.db_manager.database)
        try:
            grid_out = fs.get_last_version(f"detector_model_{detector_id}")
            return grid_out.read()
        except Exception:
            return None

    def _doc_to_entity(self, doc: dict) -> Detector:
        """Convert MongoDB document to domain entity."""
        return Detector(
            name=doc["name"],
            algorithm_name=doc["algorithm_name"],
            parameters=doc["parameters"],
            is_fitted=doc["is_fitted"],
            metadata=doc["metadata"],
            id=UUID(doc["id"]),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )


class MongoDBDatasetRepository(DatasetRepositoryProtocol):
    """MongoDB-backed dataset repository."""

    def __init__(self, db_manager: MongoDBManager):
        """Initialize with MongoDB manager."""
        self.db_manager = db_manager
        self.collection: Collection = db_manager.database["datasets"]

    def save(self, entity: Dataset) -> None:
        """Save dataset to MongoDB."""
        dataset_doc = {
            "id": str(entity.id),
            "name": entity.name,
            "description": entity.description,
            "file_path": entity.file_path,
            "target_column": entity.target_column,
            "features": entity.features,
            "metadata": entity.metadata,
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
        }

        self.collection.replace_one(
            {"id": str(entity.id)}, dataset_doc, upsert=True
        )

    def find_by_id(self, entity_id: UUID) -> Dataset | None:
        """Find dataset by ID."""
        doc = self.collection.find_one({"id": str(entity_id)})
        return self._doc_to_entity(doc) if doc else None

    def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        docs = self.collection.find()
        return [self._doc_to_entity(doc) for doc in docs]

    def delete(self, entity_id: UUID) -> bool:
        """Delete dataset by ID."""
        result = self.collection.delete_one({"id": str(entity_id)})
        return result.deleted_count > 0

    def exists(self, entity_id: UUID) -> bool:
        """Check if dataset exists."""
        return self.collection.count_documents({"id": str(entity_id)}) > 0

    def count(self) -> int:
        """Count total datasets."""
        return self.collection.count_documents({})

    def find_by_name(self, name: str) -> Dataset | None:
        """Find dataset by name."""
        doc = self.collection.find_one({"name": name})
        return self._doc_to_entity(doc) if doc else None

    def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
        """Find datasets by metadata key-value pair."""
        docs = self.collection.find({f"metadata.{key}": value})
        return [self._doc_to_entity(doc) for doc in docs]

    def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to GridFS."""
        from pymongo import GridFS

        dataset = self.find_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        fs = GridFS(self.db_manager.database)
        
        # Serialize data based on format
        if format == "parquet":
            data_bytes = dataset.data.to_parquet()
        elif format == "csv":
            data_bytes = dataset.data.to_csv(index=False).encode("utf-8")
        else:
            raise ValueError(f"Unsupported format: {format}")

        file_id = fs.put(
            data_bytes,
            filename=f"dataset_data_{dataset_id}.{format}",
            content_type=f"application/{format}",
        )

        return f"mongodb+gridfs://{file_id}"

    def load_data(self, dataset_id: UUID) -> Dataset | None:
        """Load dataset with data from GridFS."""
        dataset = self.find_by_id(dataset_id)
        if not dataset:
            return None

        from pymongo import GridFS

        fs = GridFS(self.db_manager.database)
        try:
            # Try to find data file
            grid_out = fs.get_last_version(f"dataset_data_{dataset_id}.parquet")
            data_bytes = grid_out.read()
            
            # Deserialize data
            import pandas as pd
            import io
            
            dataset.data = pd.read_parquet(io.BytesIO(data_bytes))
            return dataset
        except Exception:
            return dataset  # Return dataset without data if not found

    def _doc_to_entity(self, doc: dict) -> Dataset:
        """Convert MongoDB document to domain entity."""
        return Dataset(
            name=doc["name"],
            data=None,  # Data loaded separately
            description=doc.get("description"),
            file_path=doc.get("file_path"),
            target_column=doc.get("target_column"),
            features=doc.get("features", []),
            metadata=doc.get("metadata", {}),
            id=UUID(doc["id"]),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )


class MongoDBDetectionResultRepository(DetectionResultRepositoryProtocol):
    """MongoDB-backed detection result repository."""

    def __init__(self, db_manager: MongoDBManager):
        """Initialize with MongoDB manager."""
        self.db_manager = db_manager
        self.collection: Collection = db_manager.database["detection_results"]

    def save(self, entity: DetectionResult) -> None:
        """Save detection result to MongoDB."""
        result_doc = {
            "id": str(entity.id),
            "detector_id": str(entity.detector_id),
            "dataset_id": str(entity.dataset_id),
            "scores": [
                {"value": score.value, "confidence": getattr(score, "confidence", None)}
                for score in entity.scores
            ],
            "metadata": entity.metadata,
            "timestamp": entity.timestamp,
        }

        self.collection.replace_one(
            {"id": str(entity.id)}, result_doc, upsert=True
        )

    def find_by_id(self, entity_id: UUID) -> DetectionResult | None:
        """Find detection result by ID."""
        doc = self.collection.find_one({"id": str(entity_id)})
        return self._doc_to_entity(doc) if doc else None

    def find_all(self) -> list[DetectionResult]:
        """Find all detection results."""
        docs = self.collection.find()
        return [self._doc_to_entity(doc) for doc in docs]

    def delete(self, entity_id: UUID) -> bool:
        """Delete detection result by ID."""
        result = self.collection.delete_one({"id": str(entity_id)})
        return result.deleted_count > 0

    def exists(self, entity_id: UUID) -> bool:
        """Check if detection result exists."""
        return self.collection.count_documents({"id": str(entity_id)}) > 0

    def count(self) -> int:
        """Count total detection results."""
        return self.collection.count_documents({})

    def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find results by detector ID."""
        docs = self.collection.find({"detector_id": str(detector_id)})
        return [self._doc_to_entity(doc) for doc in docs]

    def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find results by dataset ID."""
        docs = self.collection.find({"dataset_id": str(dataset_id)})
        return [self._doc_to_entity(doc) for doc in docs]

    def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find most recent detection results."""
        docs = self.collection.find().sort("timestamp", -1).limit(limit)
        return [self._doc_to_entity(doc) for doc in docs]

    def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
        """Get summary statistics for a result."""
        doc = self.collection.find_one({"id": str(result_id)})
        if not doc:
            return {}

        return {
            "id": doc["id"],
            "detector_id": doc["detector_id"],
            "dataset_id": doc["dataset_id"],
            "timestamp": doc["timestamp"].isoformat(),
            "total_scores": len(doc["scores"]),
            "metadata": doc["metadata"],
        }

    def _doc_to_entity(self, doc: dict) -> DetectionResult:
        """Convert MongoDB document to domain entity."""
        from pynomaly.domain.value_objects import AnomalyScore

        scores = [
            AnomalyScore(
                value=score_data["value"],
                confidence=score_data.get("confidence"),
            )
            for score_data in doc["scores"]
        ]

        return DetectionResult(
            detector_id=UUID(doc["detector_id"]),
            dataset_id=UUID(doc["dataset_id"]),
            scores=scores,
            metadata=doc["metadata"],
            id=UUID(doc["id"]),
            timestamp=doc["timestamp"],
        )
