"""Application service for model persistence."""

from __future__ import annotations

import asyncio
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly.shared.protocols import DetectorProtocol, DetectorRepositoryProtocol


class ModelPersistenceService:
    """Service for saving and loading trained models."""

    def __init__(
        self, detector_repository: DetectorRepositoryProtocol, storage_path: Path
    ):
        """Initialize model persistence service.

        Args:
            detector_repository: Repository for detectors
            storage_path: Base path for model storage
        """
        self.detector_repository = detector_repository
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def save_model(
        self,
        detector_id: UUID,
        format: str = "pickle",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a trained model to storage.

        Args:
            detector_id: ID of detector to save
            format: Serialization format ('pickle', 'joblib', 'onnx')
            metadata: Additional metadata to save

        Returns:
            Path where model was saved
        """
        # Load detector
        import asyncio

        if hasattr(
            self.detector_repository, "find_by_id"
        ) and asyncio.iscoroutinefunction(self.detector_repository.find_by_id):
            detector = await self.detector_repository.find_by_id(detector_id)
        else:
            detector = self.detector_repository.find_by_id(detector_id)
        if detector is None:
            raise ValueError(f"Detector {detector_id} not found")

        if not detector.is_fitted:
            raise ValueError(f"Detector {detector.name} is not fitted")

        # Create model directory
        model_dir = self.storage_path / str(detector_id)
        model_dir.mkdir(exist_ok=True)

        # Save based on format
        if format == "pickle":
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(detector, f)
        elif format == "joblib":
            import joblib

            model_path = model_dir / "model.joblib"
            joblib.dump(detector, model_path)
        elif format == "onnx":
            # ONNX conversion would require specific implementation
            raise NotImplementedError("ONNX format not yet supported")
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

        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Save to repository as well
        if hasattr(self.detector_repository, "save") and asyncio.iscoroutinefunction(
            self.detector_repository.save
        ):
            await self.detector_repository.save(detector)
        else:
            self.detector_repository.save(detector)

        return str(model_path)

    async def load_model(
        self, detector_id: UUID, format: str = "pickle"
    ) -> DetectorProtocol:
        """Load a saved model.

        Args:
            detector_id: ID of detector to load
            format: Format to load from

        Returns:
            Loaded detector
        """
        model_dir = self.storage_path / str(detector_id)

        if not model_dir.exists():
            raise ValueError(f"No saved model found for detector {detector_id}")

        # Load based on format
        if format == "pickle":
            model_path = model_dir / "model.pkl"
            with open(model_path, "rb") as f:
                detector = pickle.load(f)
        elif format == "joblib":
            import joblib

            model_path = model_dir / "model.joblib"
            detector = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Update repository
        if hasattr(self.detector_repository, "save") and asyncio.iscoroutinefunction(
            self.detector_repository.save
        ):
            await self.detector_repository.save(detector)
        else:
            self.detector_repository.save(detector)

        return detector

    async def export_model(
        self, detector_id: UUID, export_path: Path, include_data: bool = False
    ) -> dict[str, str]:
        """Export model for deployment.

        Args:
            detector_id: ID of detector to export
            export_path: Path to export to
            include_data: Whether to include training data

        Returns:
            Paths of exported files
        """
        # Load detector
        import asyncio

        if hasattr(
            self.detector_repository, "find_by_id"
        ) and asyncio.iscoroutinefunction(self.detector_repository.find_by_id):
            detector = await self.detector_repository.find_by_id(detector_id)
        else:
            detector = self.detector_repository.find_by_id(detector_id)
        if detector is None:
            raise ValueError(f"Detector {detector_id} not found")

        export_path.mkdir(parents=True, exist_ok=True)
        exported_files = {}

        # Export model
        model_path = export_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(detector, f)
        exported_files["model"] = str(model_path)

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

    async def list_saved_models(self) -> dict[str, dict[str, Any]]:
        """List all saved models.

        Returns:
            Dictionary of model metadata by detector ID
        """
        saved_models = {}

        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir():
                meta_path = model_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        metadata = json.load(f)
                    saved_models[model_dir.name] = metadata

        return saved_models

    async def delete_model(self, detector_id: UUID) -> bool:
        """Delete a saved model.

        Args:
            detector_id: ID of detector to delete

        Returns:
            True if deleted, False if not found
        """
        model_dir = self.storage_path / str(detector_id)

        if model_dir.exists():
            import shutil

            shutil.rmtree(model_dir)
            return True

        return False

    def _generate_requirements(self, detector: DetectorProtocol) -> list[str]:
        """Generate requirements for model deployment."""
        # Base requirements
        requirements = [
            "pynomaly>=0.1.0",
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
Generated by Pynomaly ModelPersistenceService
"""

import pickle
import pandas as pd
from pathlib import Path


class {detector.name.replace(" ", "")}Detector:
    """Wrapper for deployed anomaly detector."""

    def __init__(self, model_path: str = "model.pkl"):
        """Load the detector model."""
        with open(model_path, "rb") as f:
            self.detector = pickle.load(f)

    def detect(self, data: pd.DataFrame) -> dict:
        """Detect anomalies in data.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary with scores and labels
        """
        from pynomaly.domain.entities import Dataset

        dataset = Dataset(name="input", data=data)
        result = self.detector.detect(dataset)

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
        from pynomaly.domain.entities import Dataset

        dataset = Dataset(name="input", data=data)
        scores = self.detector.score(dataset)
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
