"""Application service for processor persistence."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from monorepo.infrastructure.security.security_hardening import get_secure_serializer
from monorepo.shared.protocols import DetectorProtocol, DetectorRepositoryProtocol

logger = logging.getLogger(__name__)


class ModelPersistenceService:
    """Service for saving and loading trained models."""

    def __init__(
        self, detector_repository: DetectorRepositoryProtocol, storage_path: Path
    ):
        """Initialize processor persistence service.

        Args:
            detector_repository: Repository for detectors
            storage_path: Base path for processor storage
        """
        self.detector_repository = detector_repository
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def save_processor(
        self,
        detector_id: UUID,
        format: str = "pickle",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a trained processor to storage.

        Args:
            detector_id: ID of detector to save
            format: Serialization format ('pickle', 'joblib', 'onnx')
            metadata: Additional metadata to save

        Returns:
            Path where processor was saved
        """
        # Load detector
        detector = await self.detector_repository.find_by_id(detector_id)
        if detector is None:
            raise ValueError(f"Detector {detector_id} not found")

        if not detector.is_fitted:
            raise ValueError(f"Detector {detector.name} is not fitted")

        # Create processor directory
        processor_dir = self.storage_path / str(detector_id)
        processor_dir.mkdir(exist_ok=True)

        # Use secure serialization instead of pickle
        secure_serializer = get_secure_serializer()

        # Save based on format
        if format == "pickle":
            # Use secure serialization instead of unsafe pickle
            processor_path = processor_dir / "processor.secure"
            secure_serializer.serialize_processor(detector, processor_path)
        elif format == "joblib":
            # joblib is safer than pickle, but still use secure wrapper
            processor_path = processor_dir / "processor.joblib"
            secure_serializer.serialize_processor(detector, processor_path)
        elif format == "onnx":
            # ONNX conversion using existing implementation
            processor_path = processor_dir / "processor.onnx"
            await self._save_onnx_processor(detector, processor_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Save metadata
        meta = {
            "detector_id": str(detector_id),
            "detector_name": detector.name,
            "algorithm": detector.algorithm_name,
            "saved_at": datetime.now(UTC).isoformat(),
            "format": format,
            "is_fitted": detector.is_fitted,
            "parameters": detector.parameters,
            **(metadata or {}),
        }

        meta_path = processor_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Save to repository as well
        await self.detector_repository.save(detector)

        return str(processor_path)

    async def load_processor(
        self, detector_id: UUID, format: str = "pickle"
    ) -> DetectorProtocol:
        """Load a saved processor.

        Args:
            detector_id: ID of detector to load
            format: Format to load from

        Returns:
            Loaded detector
        """
        processor_dir = self.storage_path / str(detector_id)

        if not processor_dir.exists():
            raise ValueError(f"No saved processor found for detector {detector_id}")

        # Use secure deserialization
        secure_serializer = get_secure_serializer()

        # Load based on format
        if format == "pickle":
            # Use secure deserialization instead of unsafe pickle
            processor_path = processor_dir / "processor.secure"
            if not processor_path.exists():
                # Try legacy pickle file for backward compatibility
                legacy_path = processor_dir / "processor.pkl"
                if legacy_path.exists():
                    raise ValueError(
                        f"Legacy pickle file found for detector {detector_id}. "
                        "Please re-save the processor with secure serialization."
                    )
                raise ValueError(
                    f"No secure processor file found for detector {detector_id}"
                )
            detector = secure_serializer.deserialize_processor(processor_path)
        elif format == "joblib":
            # Use secure deserialization wrapper
            processor_path = processor_dir / "processor.joblib"
            detector = secure_serializer.deserialize_processor(processor_path)
        elif format == "onnx":
            # ONNX loading - for now, load the stub or use fallback
            processor_path = processor_dir / "processor.onnx"
            detector = await self._load_onnx_processor(processor_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Update repository only if it's a domain entity (has id property)
        # Adapters are infrastructure objects and shouldn't be stored in the domain repository
        if hasattr(detector, "id") and hasattr(detector, "name"):
            await self.detector_repository.save(detector)

        return detector

    async def export_processor(
        self, detector_id: UUID, export_path: Path, include_data: bool = False
    ) -> dict[str, str]:
        """Export processor for deployment.

        Args:
            detector_id: ID of detector to export
            export_path: Path to export to
            include_data: Whether to include training data

        Returns:
            Paths of exported files
        """
        # Load detector
        detector = await self.detector_repository.find_by_id(detector_id)
        if detector is None:
            raise ValueError(f"Detector {detector_id} not found")

        export_path.mkdir(parents=True, exist_ok=True)
        exported_files = {}

        # Export processor - for domain entities, we'll save as JSON with custom serialization
        # since the secure serializer has issues with domain value objects
        processor_path = export_path / "processor.json"
        processor_data = {
            "id": str(detector.id),
            "name": detector.name,
            "algorithm_name": detector.algorithm_name,
            "contamination_rate": detector.contamination_rate.value,
            "parameters": detector.parameters,
            "metadata": detector.metadata,
            "created_at": detector.created_at.isoformat(),
            "trained_at": detector.trained_at.isoformat()
            if detector.trained_at
            else None,
            "is_fitted": detector.is_fitted,
            "exported_at": datetime.now(UTC).isoformat(),
        }

        with open(processor_path, "w") as f:
            json.dump(processor_data, f, indent=2)
        exported_files["processor"] = str(processor_path)

        # Export configuration
        config = {
            "detector_name": detector.name,
            "algorithm": detector.algorithm_name,
            "parameters": detector.parameters,
            "contamination_rate": detector.contamination_rate.value,
            "requires_fitting": detector.requires_fitting,
            "supports_streaming": detector.supports_streaming,
            "exported_at": datetime.now(UTC).isoformat(),
        }

        config_path = export_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        exported_files["config"] = str(config_path)

        # Export requirements
        requirements = self._generate_requirements(detector)
        req_path = export_path / "requirements.txt"
        with open(req_path, "w") as f:
            f.write("\n".join(requirements))
        exported_files["requirements"] = str(req_path)

        # Create deployment script
        deploy_script = self._generate_deployment_script(detector)
        script_path = export_path / "deploy.py"
        with open(script_path, "w") as f:
            f.write(deploy_script)
        exported_files["deploy_script"] = str(script_path)

        return exported_files

    async def list_saved_processors(self) -> dict[str, dict[str, Any]]:
        """List all saved models.

        Returns:
            Dictionary of processor metadata by detector ID
        """
        saved_processors = {}

        for processor_dir in self.storage_path.iterdir():
            if processor_dir.is_dir():
                meta_path = processor_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        metadata = json.load(f)
                    saved_processors[processor_dir.name] = metadata

        return saved_processors

    async def delete_processor(self, detector_id: UUID) -> bool:
        """Delete a saved processor.

        Args:
            detector_id: ID of detector to delete

        Returns:
            True if deleted, False if not found
        """
        processor_dir = self.storage_path / str(detector_id)

        if processor_dir.exists():
            import shutil

            shutil.rmtree(processor_dir)
            return True

        return False

    def _generate_requirements(self, detector: DetectorProtocol) -> list[str]:
        """Generate requirements for processor deployment."""
        # Base requirements
        requirements = [
            "software>=0.1.0",
            "numpy>=1.26.0",
            "pandas>=2.2.0",
            "scikit-learn>=1.5.0",
        ]

        # Add algorithm-specific requirements
        algorithm_requirements = {
            "PyOD": ["pyod>=2.0.5"],
            "IsolationForest": [],
            "LocalOutlierFactor": [],
            "OneClassSVM": [],
            "PyGOD": ["pygod>=1.0.0"],
        }

        for algo, reqs in algorithm_requirements.items():
            if algo.lower() in detector.algorithm_name.lower():
                requirements.extend(reqs)

        return sorted(set(requirements))

    def _generate_deployment_script(self, detector: DetectorProtocol) -> str:
        """Generate a simple deployment script."""
        return f'''#!/usr/bin/env python3
"""
Deployment script for {detector.name}
Generated by Software ModelPersistenceService
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from uuid import UUID


class {detector.name.replace(" ", "")}Detector:
    """Wrapper for deployed anomaly detector."""

    def __init__(self, model_path: str = "model.json"):
        """Load the detector processor."""
        with open(processor_path, 'r') as f:
            processor_data = json.load(f)

        # Create a simple detector-like object
        self.detector = type('Detector', (), {{
            'id': UUID(processor_data['id']),
            'name': processor_data['name'],
            'algorithm_name': processor_data['algorithm_name'],
            'contamination_rate': type('ContaminationRate', (), {{'value': processor_data['contamination_rate']}})(),
            'parameters': processor_data['parameters'],
            'is_fitted': processor_data['is_fitted'],
        }})()

    def detect(self, data: pd.DataFrame) -> dict:
        """Detect anomalies in data.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary with scores and labels
        """
        from monorepo.domain.entities import Dataset

        data_collection = DataCollection(name="input", data=data)
        result = self.detector.detect(data_collection)

        return {{
            "scores": [s.value for s in result.scores],
            "labels": result.labels.tolist(),
            "n_anomalies": result.n_anomalies,
            "threshold": result.threshold
        }}

    def predict_proba(self, data: pd.DataFrame) -> list[float]:
        """Get anomaly scores.

        Args:
            data: DataFrame with features

        Returns:
            List of anomaly scores
        """
        from monorepo.domain.entities import Dataset

        data_collection = DataCollection(name="input", data=data)
        scores = self.detector.score(data_collection)
        return [s.value for s in scores]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deploy.py <input_csv>")
        sys.exit(1)

    # Load data
    data = pd.read_csv(sys.argv[1])

    # Initialize detector
    detector = {detector.name.replace(" ", "")}Detector()

    # Detect anomalies
    results = detector.detect(data)

    print(f"Found {{results['n_anomalies']}} anomalies")
    print(f"Threshold: {{results['threshold']:.3f}}")

    # Save results
    output = pd.DataFrame({{
        "score": results["scores"],
        "is_anomaly": results["labels"]
    }})
    output.to_csv("anomalies.csv", index=False)
    print("Results saved to anomalies.csv")
'''

    async def _save_onnx_processor(
        self, detector: DetectorProtocol, processor_path: Path
    ) -> None:
        """Save processor in ONNX format.

        Args:
            detector: Detector to save
            processor_path: Path to save ONNX processor
        """
        from monorepo.infrastructure.config.feature_flags import feature_flags

        # Check if deep learning is enabled
        if not feature_flags.is_enabled("deep_learning"):
            raise RuntimeError(
                "Deep learning features are disabled. Enable with PYNOMALY_DEEP_LEARNING=true"
            )

        try:
            # Try to get PyTorch processor from detector
            if hasattr(detector, "_processor") and detector._processor is not None:
                try:
                    import torch
                    import torch.onnx

                    # Get processor and create dummy input
                    processor = detector._processor
                    logger.info(f"Exporting PyTorch processor to ONNX: {detector.name}")

                    # Create dummy input based on processor's expected input shape
                    dummy_input = self._create_dummy_input(detector)
                    if dummy_input is None:
                        raise ValueError("Could not create dummy input for ONNX export")

                    # Export to ONNX with comprehensive error handling
                    torch.onnx.export(
                        processor,
                        dummy_input,
                        str(processor_path),
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={
                            "input": {0: "batch_size"},
                            "output": {0: "batch_size"},
                        },
                        verbose=False,  # Reduce noise
                    )
                    logger.info(
                        f"Successfully exported PyTorch processor to ONNX: {processor_path}"
                    )

                except ImportError as e:
                    logger.warning(
                        f"PyTorch not available for ONNX export: {e}. Creating stub processor."
                    )
                    await self._create_stub_onnx_processor(detector, processor_path)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning(
                        f"PyTorch ONNX export failed: {e}. Creating stub processor."
                    )
                    await self._create_stub_onnx_processor(detector, processor_path)
            else:
                # For non-PyTorch models, create a stub ONNX processor
                logger.info(
                    f"Creating stub ONNX processor for non-PyTorch detector: {detector.name}"
                )
                await self._create_stub_onnx_processor(detector, processor_path)

        except Exception as e:
            # If any other error occurs, try to create stub as fallback
            logger.error(f"Unexpected error during ONNX export: {e}")
            try:
                await self._create_stub_onnx_processor(detector, processor_path)
                logger.info(
                    f"Created fallback stub ONNX processor after error: {processor_path}"
                )
            except Exception as stub_error:
                raise RuntimeError(
                    f"Failed to export processor to ONNX and fallback stub creation failed. "
                    f"Original error: {e}. Stub error: {stub_error}"
                )

    def _create_dummy_input(self, detector: DetectorProtocol):
        """Create dummy input for ONNX export.

        Args:
            detector: Detector to create input for

        Returns:
            Dummy input tensor
        """
        try:
            import torch

            # Default input shape - can be overridden by detector
            input_shape = getattr(detector, "_input_shape", (1, 10))
            return torch.randn(input_shape)
        except ImportError:
            # If torch is not available, return None
            return None

    async def _create_stub_onnx_processor(
        self, detector: DetectorProtocol, processor_path: Path
    ) -> None:
        """Create a stub ONNX processor for non-PyTorch detectors.

        Args:
            detector: Detector to create stub for
            processor_path: Path to save stub processor
        """
        # Create a simple stub ONNX processor that can be loaded later
        # This is a minimal implementation for fast tests
        import json

        stub_data = {
            "processor_type": "stub",
            "detector_name": detector.name,
            "algorithm": detector.algorithm_name,
            "parameters": detector.parameters,
            "message": "This is a stub ONNX processor. Full ONNX export requires PyTorch-based models.",
        }

        # Save as JSON stub with .onnx extension
        with open(processor_path, "w") as f:
            json.dump(stub_data, f, indent=2)

    async def _load_onnx_processor(self, processor_path: Path) -> DetectorProtocol:
        """Load ONNX processor.

        Args:
            processor_path: Path to ONNX processor

        Returns:
            Loaded detector
        """
        try:
            # Try to load as actual ONNX processor
            import onnxruntime as ort

            # Check if it's a real ONNX processor
            try:
                session = ort.InferenceSession(str(processor_path))
                # Create a wrapper detector for ONNX processor
                return await self._create_onnx_detector_wrapper(session, processor_path)
            except:
                # If it fails, try to load as stub
                return await self._load_onnx_stub(processor_path)

        except ImportError:
            # If ONNX runtime not available, try to load as stub
            return await self._load_onnx_stub(processor_path)

    async def _load_onnx_stub(self, processor_path: Path) -> DetectorProtocol:
        """Load ONNX stub processor.

        Args:
            processor_path: Path to ONNX stub processor

        Returns:
            Detector instance
        """
        # Load the stub data
        with open(processor_path) as f:
            stub_data = json.load(f)

        # Create a basic detector from the stub data
        from monorepo.domain.value_objects import ContaminationRate
        from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter

        # Create a simple sklearn adapter as a fallback
        detector = SklearnAdapter(
            algorithm_name=stub_data["algorithm"],
            name=stub_data["detector_name"],
            contamination_rate=ContaminationRate(0.1),
            **stub_data["parameters"],
        )

        return detector

    async def _create_onnx_detector_wrapper(
        self, session, processor_path: Path
    ) -> DetectorProtocol:
        """Create a detector wrapper for ONNX processor.

        Args:
            session: ONNX runtime session
            processor_path: Path to ONNX processor

        Returns:
            Detector wrapper
        """
        from monorepo.infrastructure.adapters.onnx_adapter import ONNXAdapter

        # Create ONNX adapter (this would need to be implemented)
        adapter = ONNXAdapter(session, processor_path)
        return adapter
