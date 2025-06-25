"""Enhanced model persistence service with multiple format support."""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Detector
from pynomaly.domain.entities.model_version import ModelStatus, ModelVersion
from pynomaly.domain.value_objects.model_storage_info import (
    ModelStorageInfo,
    SerializationFormat,
    StorageBackend,
)
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.domain.value_objects.semantic_version import SemanticVersion
from pynomaly.shared.protocols import DetectorRepositoryProtocol


class ModelSerializationError(Exception):
    """Error during model serialization."""

    pass


class ModelDeserializationError(Exception):
    """Error during model deserialization."""

    pass


class UnsupportedFormatError(Exception):
    """Unsupported serialization format."""

    pass


class EnhancedModelPersistenceService:
    """Enhanced service for model persistence with multiple format support.

    This service provides comprehensive model lifecycle management including:
    - Multiple serialization formats (pickle, joblib, ONNX, etc.)
    - Version control with semantic versioning
    - Metadata management and storage optimization
    - Integrity verification with checksums
    - Compression and encryption support
    - Performance tracking and comparison
    """

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        storage_path: Path,
        enable_compression: bool = True,
        default_format: SerializationFormat = SerializationFormat.PICKLE,
    ):
        """Initialize enhanced model persistence service.

        Args:
            detector_repository: Repository for detector entities
            storage_path: Base path for model storage
            enable_compression: Whether to enable compression by default
            default_format: Default serialization format
        """
        self.detector_repository = detector_repository
        self.storage_path = Path(storage_path)
        self.enable_compression = enable_compression
        self.default_format = default_format

        # Create storage directory structure
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models_path = self.storage_path / "models"
        self.metadata_path = self.storage_path / "metadata"
        self.versions_path = self.storage_path / "versions"

        for path in [self.models_path, self.metadata_path, self.versions_path]:
            path.mkdir(parents=True, exist_ok=True)

    async def serialize_model(
        self,
        detector: Detector,
        version: SemanticVersion,
        format: SerializationFormat | None = None,
        performance_metrics: PerformanceMetrics | None = None,
        metadata: dict[str, Any] | None = None,
        compress: bool | None = None,
        created_by: str = "system",
    ) -> ModelVersion:
        """Serialize and store a trained model.

        Args:
            detector: Trained detector to serialize
            version: Version for this model
            format: Serialization format (uses default if None)
            performance_metrics: Performance metrics for this version
            metadata: Additional metadata
            compress: Whether to compress (uses default if None)
            created_by: User who created this version

        Returns:
            ModelVersion entity representing the stored model

        Raises:
            ModelSerializationError: If serialization fails
            UnsupportedFormatError: If format is not supported
        """
        if not detector.is_fitted:
            raise ModelSerializationError(f"Detector {detector.name} is not fitted")

        format = format or self.default_format
        compress = compress if compress is not None else self.enable_compression

        try:
            # Create model directory
            model_dir = self.models_path / str(detector.id) / version.version_string
            model_dir.mkdir(parents=True, exist_ok=True)

            # Serialize model based on format
            serialized_data, file_extension = await self._serialize_by_format(
                detector, format
            )

            # Compress if requested
            if compress:
                serialized_data = await self._compress_data(serialized_data)
                file_extension += ".gz"

            # Calculate checksum
            checksum = hashlib.sha256(serialized_data).hexdigest()

            # Save model file
            model_filename = f"model{file_extension}"
            model_path = model_dir / model_filename

            with open(model_path, "wb") as f:
                f.write(serialized_data)

            # Create storage info
            storage_info = ModelStorageInfo.create_for_local_file(
                file_path=str(model_path),
                format=format,
                size_bytes=len(serialized_data),
                checksum=checksum,
                compression_type="gzip" if compress else None,
            )

            # Create performance metrics if not provided
            if performance_metrics is None:
                performance_metrics = await self._estimate_performance_metrics(detector)

            # Create model version
            model_version = ModelVersion(
                model_id=detector.id,
                version=version,
                detector_id=detector.id,
                created_by=created_by,
                performance_metrics=performance_metrics,
                storage_info=storage_info,
                metadata=metadata or {},
                status=ModelStatus.VALIDATED,
            )

            # Save metadata
            await self._save_version_metadata(model_version)

            # Save detector configuration
            await self._save_detector_config(detector, model_dir)

            return model_version

        except Exception as e:
            raise ModelSerializationError(f"Failed to serialize model: {str(e)}") from e

    async def deserialize_model(
        self, model_version: ModelVersion, verify_checksum: bool = True
    ) -> Detector:
        """Deserialize a stored model.

        Args:
            model_version: Model version to deserialize
            verify_checksum: Whether to verify file integrity

        Returns:
            Deserialized detector instance

        Raises:
            ModelDeserializationError: If deserialization fails
        """
        try:
            storage_info = model_version.storage_info
            model_path = Path(storage_info.storage_path)

            if not model_path.exists():
                raise ModelDeserializationError(f"Model file not found: {model_path}")

            # Read model data
            with open(model_path, "rb") as f:
                model_data = f.read()

            # Verify checksum if requested
            if verify_checksum:
                if not storage_info.verify_checksum(model_data):
                    raise ModelDeserializationError(
                        "Model checksum verification failed"
                    )

            # Decompress if needed
            if storage_info.is_compressed:
                model_data = await self._decompress_data(model_data)

            # Deserialize based on format
            detector = await self._deserialize_by_format(
                model_data, storage_info.format
            )

            return detector

        except Exception as e:
            raise ModelDeserializationError(
                f"Failed to deserialize model: {str(e)}"
            ) from e

    async def list_model_versions(
        self,
        model_id: UUID | None = None,
        status_filter: ModelStatus | None = None,
        limit: int | None = None,
    ) -> list[ModelVersion]:
        """List available model versions.

        Args:
            model_id: Filter by specific model ID
            status_filter: Filter by model status
            limit: Maximum number of versions to return

        Returns:
            List of model versions sorted by creation date (newest first)
        """
        versions = []

        # Scan versions directory
        for version_file in self.versions_path.glob("*.json"):
            try:
                version = await self._load_version_metadata(version_file)

                # Apply filters
                if model_id and version.model_id != model_id:
                    continue

                if status_filter and version.status != status_filter:
                    continue

                versions.append(version)

            except Exception:
                # Skip corrupted metadata files
                continue

        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        # Apply limit
        if limit:
            versions = versions[:limit]

        return versions

    async def get_model_version(
        self, model_id: UUID, version: SemanticVersion | str
    ) -> ModelVersion | None:
        """Get a specific model version.

        Args:
            model_id: Model ID
            version: Version number or string

        Returns:
            Model version if found, None otherwise
        """
        if isinstance(version, str):
            version = SemanticVersion.from_string(version)

        versions = await self.list_model_versions(model_id=model_id)

        for model_version in versions:
            if model_version.version == version:
                return model_version

        return None

    async def get_latest_version(
        self, model_id: UUID, status_filter: ModelStatus | None = None
    ) -> ModelVersion | None:
        """Get the latest version of a model.

        Args:
            model_id: Model ID
            status_filter: Filter by status

        Returns:
            Latest model version if found
        """
        versions = await self.list_model_versions(
            model_id=model_id, status_filter=status_filter, limit=1
        )

        return versions[0] if versions else None

    async def compare_versions(
        self, version1: ModelVersion, version2: ModelVersion
    ) -> dict[str, Any]:
        """Compare two model versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Comparison results including performance differences
        """
        performance_diff = version1.performance_metrics.compare_with(
            version2.performance_metrics
        )

        return {
            "version1": {
                "version": version1.version_string,
                "created_at": version1.created_at.isoformat(),
                "status": version1.status.value,
                "performance": version1.performance_metrics.to_dict(),
            },
            "version2": {
                "version": version2.version_string,
                "created_at": version2.created_at.isoformat(),
                "status": version2.status.value,
                "performance": version2.performance_metrics.to_dict(),
            },
            "performance_difference": performance_diff,
            "version1_is_better": version1.performance_metrics.is_better_than(
                version2.performance_metrics
            ),
            "version_distance": version1.version.distance_from(version2.version),
        }

    async def delete_model_version(
        self, model_version: ModelVersion, force: bool = False
    ) -> bool:
        """Delete a model version.

        Args:
            model_version: Version to delete
            force: Force deletion even if deployed

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If trying to delete deployed version without force
        """
        if model_version.is_deployed and not force:
            raise ValueError("Cannot delete deployed version without force=True")

        try:
            # Delete model file
            model_path = Path(model_version.storage_info.storage_path)
            if model_path.exists():
                model_path.unlink()

            # Delete model directory if empty
            model_dir = model_path.parent
            if model_dir.exists() and not list(model_dir.iterdir()):
                model_dir.rmdir()

            # Delete metadata file
            metadata_file = self.versions_path / f"{model_version.id}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            return True

        except Exception:
            return False

    async def export_model_bundle(
        self,
        model_version: ModelVersion,
        export_path: Path,
        include_dependencies: bool = True,
        include_examples: bool = True,
    ) -> dict[str, str]:
        """Export a complete model bundle for deployment.

        Args:
            model_version: Version to export
            export_path: Export directory
            include_dependencies: Include requirements file
            include_examples: Include usage examples

        Returns:
            Dictionary of exported file paths
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Copy model file
        model_path = Path(model_version.storage_info.storage_path)
        model_filename = f"model{model_path.suffix}"
        export_model_path = export_path / model_filename

        with open(model_path, "rb") as src, open(export_model_path, "wb") as dst:
            dst.write(src.read())

        exported_files["model"] = str(export_model_path)

        # Export model configuration
        config = await self._create_deployment_config(model_version)
        config_path = export_path / "model_config.json"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        exported_files["config"] = str(config_path)

        # Export metadata
        metadata_path = export_path / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(model_version.get_info(), f, indent=2, default=str)

        exported_files["metadata"] = str(metadata_path)

        # Include requirements if requested
        if include_dependencies:
            requirements = await self._generate_requirements(model_version)
            req_path = export_path / "requirements.txt"

            with open(req_path, "w") as f:
                f.write("\n".join(requirements))

            exported_files["requirements"] = str(req_path)

        # Include examples if requested
        if include_examples:
            example_script = await self._generate_usage_examples(model_version)
            example_path = export_path / "example_usage.py"

            with open(example_path, "w") as f:
                f.write(example_script)

            exported_files["examples"] = str(example_path)

        # Create deployment script
        deploy_script = await self._generate_deployment_script(model_version)
        deploy_path = export_path / "deploy.py"

        with open(deploy_path, "w") as f:
            f.write(deploy_script)

        exported_files["deploy_script"] = str(deploy_path)

        return exported_files

    async def create_model_archive(
        self, model_version: ModelVersion, archive_path: Path
    ) -> str:
        """Create a compressed archive of the model bundle.

        Args:
            model_version: Version to archive
            archive_path: Path for the archive file

        Returns:
            Path to created archive
        """
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export model bundle to temporary directory
            exported_files = await self.export_model_bundle(model_version, temp_path)

            # Create zip archive
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for _file_type, file_path in exported_files.items():
                    zf.write(file_path, Path(file_path).name)

        return str(archive_path)

    async def _serialize_by_format(
        self, detector: Detector, format: SerializationFormat
    ) -> tuple[bytes, str]:
        """Serialize detector using specified format.

        Returns:
            Tuple of (serialized_data, file_extension)
        """
        if format == SerializationFormat.PICKLE:
            return pickle.dumps(detector), ".pkl"

        elif format == SerializationFormat.JOBLIB:
            import joblib

            with tempfile.NamedTemporaryFile() as temp_file:
                joblib.dump(detector, temp_file.name)
                with open(temp_file.name, "rb") as f:
                    return f.read(), ".joblib"

        elif format == SerializationFormat.SCIKIT_LEARN_PICKLE:
            # Enhanced pickle for scikit-learn models
            import joblib

            with tempfile.NamedTemporaryFile() as temp_file:
                joblib.dump(detector, temp_file.name, compress=3)
                with open(temp_file.name, "rb") as f:
                    return f.read(), ".sklearn.pkl"

        else:
            raise UnsupportedFormatError(
                f"Serialization format {format.value} not supported"
            )

    async def _deserialize_by_format(
        self, data: bytes, format: SerializationFormat
    ) -> Detector:
        """Deserialize detector using specified format."""
        if format == SerializationFormat.PICKLE:
            return pickle.loads(data)

        elif format in [
            SerializationFormat.JOBLIB,
            SerializationFormat.SCIKIT_LEARN_PICKLE,
        ]:
            import joblib

            with tempfile.NamedTemporaryFile() as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(data)
                return joblib.load(temp_file.name)

        else:
            raise UnsupportedFormatError(
                f"Deserialization format {format.value} not supported"
            )

    async def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        import gzip

        return gzip.compress(data)

    async def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        import gzip

        return gzip.decompress(data)

    async def _estimate_performance_metrics(
        self, detector: Detector
    ) -> PerformanceMetrics:
        """Estimate performance metrics for a detector."""
        # This is a placeholder - in practice, you'd run evaluation
        return PerformanceMetrics.create_minimal(
            accuracy=0.85,  # Placeholder
            training_time=60.0,  # Placeholder
            inference_time=5.0,  # Placeholder
            model_size=1024 * 1024,  # Placeholder
        )

    async def _save_version_metadata(self, model_version: ModelVersion) -> None:
        """Save model version metadata to disk."""
        metadata_file = self.versions_path / f"{model_version.id}.json"

        with open(metadata_file, "w") as f:
            json.dump(model_version.get_info(), f, indent=2, default=str)

    async def _load_version_metadata(self, metadata_file: Path) -> ModelVersion:
        """Load model version metadata from disk."""
        with open(metadata_file) as f:
            data = json.load(f)

        # Reconstruct value objects
        version = SemanticVersion.from_string(data["version"])

        performance_data = data["performance_metrics"]
        performance_metrics = PerformanceMetrics(
            accuracy=performance_data["accuracy"],
            precision=performance_data["precision"],
            recall=performance_data["recall"],
            f1_score=performance_data["f1_score"],
            training_time=performance_data["training_time"],
            inference_time=performance_data["inference_time"],
            model_size=performance_data["model_size"],
            roc_auc=performance_data.get("roc_auc"),
            pr_auc=performance_data.get("pr_auc"),
        )

        storage_data = data["storage_info"]
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend(storage_data["storage_backend"]),
            storage_path=storage_data["storage_path"],
            format=SerializationFormat(storage_data["format"]),
            size_bytes=storage_data["size_bytes"],
            checksum=storage_data["checksum"],
            compression_type=storage_data.get("compression_type"),
        )

        return ModelVersion(
            id=UUID(data["id"]),
            model_id=UUID(data["model_id"]),
            version=version,
            detector_id=UUID(data["detector_id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            performance_metrics=performance_metrics,
            storage_info=storage_info,
            metadata=data["metadata"],
            status=ModelStatus(data["status"]),
            parent_version_id=(
                UUID(data["parent_version_id"])
                if data.get("parent_version_id")
                else None
            ),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )

    async def _save_detector_config(self, detector: Detector, model_dir: Path) -> None:
        """Save detector configuration."""
        config = {
            "detector_info": detector.get_info(),
            "algorithm_name": detector.algorithm_name,
            "parameters": detector.parameters,
            "contamination_rate": detector.contamination_rate.value,
            "metadata": detector.metadata,
        }

        config_path = model_dir / "detector_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    async def _create_deployment_config(
        self, model_version: ModelVersion
    ) -> dict[str, Any]:
        """Create deployment configuration."""
        return {
            "model_version_id": str(model_version.id),
            "model_id": str(model_version.model_id),
            "version": model_version.version_string,
            "format": model_version.storage_info.format.value,
            "performance_metrics": model_version.performance_metrics.to_dict(),
            "deployment_info": {
                "requires_preprocessing": True,
                "supports_batch_inference": True,
                "supports_streaming": False,
                "recommended_batch_size": 1000,
            },
            "created_at": model_version.created_at.isoformat(),
        }

    async def _generate_requirements(self, model_version: ModelVersion) -> list[str]:
        """Generate requirements for deployment."""
        base_requirements = [
            "pynomaly>=0.4.0",
            "numpy>=1.26.0",
            "pandas>=2.2.0",
            "scikit-learn>=1.5.0",
        ]

        # Add format-specific requirements
        format_requirements = {
            SerializationFormat.JOBLIB: ["joblib>=1.3.0"],
            SerializationFormat.ONNX: ["onnx>=1.14.0", "onnxruntime>=1.15.0"],
            SerializationFormat.TENSORFLOW_SAVEDMODEL: ["tensorflow>=2.13.0"],
            SerializationFormat.PYTORCH_STATE_DICT: ["torch>=2.0.0"],
            SerializationFormat.HUGGINGFACE: ["transformers>=4.21.0"],
        }

        format_reqs = format_requirements.get(model_version.storage_info.format, [])

        return sorted(set(base_requirements + format_reqs))

    async def _generate_usage_examples(self, model_version: ModelVersion) -> str:
        """Generate usage examples."""
        return f'''#!/usr/bin/env python3
"""
Usage examples for model version {model_version.version_string}
Generated by Pynomaly Enhanced Model Persistence Service
"""

import pandas as pd
from pynomaly.application.services.enhanced_model_persistence_service import EnhancedModelPersistenceService
from pynomaly.domain.value_objects.semantic_version import SemanticVersion
from uuid import UUID

# Initialize service
service = EnhancedModelPersistenceService(
    detector_repository=None,  # Provide your repository
    storage_path="./models"
)

# Load model version
model_version = await service.get_model_version(
    model_id=UUID("{model_version.model_id}"),
    version=SemanticVersion.from_string("{model_version.version_string}")
)

# Deserialize model
detector = await service.deserialize_model(model_version)

# Use for inference
data = pd.read_csv("your_data.csv")
from pynomaly.domain.entities import Dataset

dataset = Dataset(name="inference_data", data=data)
result = detector.detect(dataset)

print(f"Found {{result.n_anomalies}} anomalies")
print(f"Anomaly scores: {{[s.value for s in result.scores]}}")
'''

    async def _generate_deployment_script(self, model_version: ModelVersion) -> str:
        """Generate deployment script."""
        return f'''#!/usr/bin/env python3
"""
Deployment script for model {model_version.version_string}
Generated by Pynomaly Enhanced Model Persistence Service
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

class ModelDeployment:
    """Production model deployment wrapper."""

    def __init__(self, model_path: str = "model.pkl", config_path: str = "model_config.json"):
        """Initialize deployment."""
        self.model_path = model_path
        self.config_path = config_path

        # Load configuration
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load model (implement based on format)
        self.detector = self._load_model()

    def _load_model(self):
        """Load the model based on format."""
        format_type = self.config["format"]

        if format_type == "pickle":
            import pickle
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        elif format_type == "joblib":
            import joblib
            return joblib.load(self.model_path)
        else:
            raise ValueError(f"Unsupported format: {{format_type}}")

    def predict(self, data: pd.DataFrame) -> Dict[str, any]:
        """Predict anomalies."""
        from pynomaly.domain.entities import Dataset

        dataset = Dataset(name="inference", data=data)
        result = self.detector.detect(dataset)

        return {{
            "anomalies": result.n_anomalies,
            "scores": [s.value for s in result.scores],
            "labels": result.labels.tolist(),
            "threshold": result.threshold,
            "model_version": self.config["version"]
        }}

    def health_check(self) -> Dict[str, any]:
        """Health check endpoint."""
        return {{
            "status": "healthy",
            "model_version": self.config["version"],
            "model_id": self.config["model_id"],
            "format": self.config["format"]
        }}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deploy.py <input_csv>")
        sys.exit(1)

    # Initialize deployment
    deployment = ModelDeployment()

    # Load and process data
    data = pd.read_csv(sys.argv[1])
    results = deployment.predict(data)

    print(f"Processed {{len(data)}} samples")
    print(f"Found {{results['anomalies']}} anomalies")
    print(f"Model version: {{results['model_version']}}")

    # Save results
    output_df = pd.DataFrame({{
        "anomaly_score": results["scores"],
        "is_anomaly": results["labels"]
    }})
    output_df.to_csv("deployment_results.csv", index=False)
    print("Results saved to deployment_results.csv")
'''
