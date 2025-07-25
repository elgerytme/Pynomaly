"""Model versioning adapter implementation."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import hashlib
import shutil
from dataclasses import asdict

from machine_learning.domain.interfaces.advanced_ml_operations import ModelVersioningPort
from machine_learning.domain.entities.model_version import ModelVersion, ModelStatus

logger = logging.getLogger(__name__)

class FileBasedModelVersioningAdapter(ModelVersioningPort):
    """File-based model versioning implementation."""
    
    def __init__(self, storage_root: str = "/tmp/model_versions"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.storage_root / "metadata"
        self.artifacts_dir = self.storage_root / "artifacts"
        self.metadata_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
    
    async def create_model_version(
        self, 
        model_name: str,
        version_number: str,
        model_data: bytes,
        metadata: Dict[str, Any]
    ) -> ModelVersion:
        """Create a new model version."""
        try:
            # Generate version ID
            version_id = f"{model_name}_{version_number}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create model version entity
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                version_number=version_number,
                parent_version=metadata.get("parent_version"),
                status=ModelStatus.DEVELOPMENT,
                accuracy=metadata.get("accuracy", 0.0),
                precision=metadata.get("precision", 0.0),
                recall=metadata.get("recall", 0.0),
                f1_score=metadata.get("f1_score", 0.0),
                training_dataset=metadata.get("training_dataset", "unknown"),
                validation_dataset=metadata.get("validation_dataset", "unknown"),
                hyperparameters=metadata.get("hyperparameters", {}),
                feature_names=metadata.get("feature_names", []),
                model_size_mb=len(model_data) / (1024 * 1024),
                training_duration_seconds=metadata.get("training_duration_seconds", 0),
                created_by=metadata.get("created_by", "system"),
                created_at=datetime.utcnow().isoformat(),
                description=metadata.get("description", "")
            )
            
            # Store model artifacts
            artifact_path = self.artifacts_dir / f"{version_id}.model"
            with open(artifact_path, 'wb') as f:
                f.write(model_data)
            
            # Store metadata
            metadata_path = self.metadata_dir / f"{version_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(model_version), f, indent=2, default=str)
            
            logger.info(f"Created model version {version_id}")
            return model_version
        
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise
    
    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        try:
            metadata_path = self.metadata_dir / f"{version_id}.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            # Convert dict back to ModelVersion
            model_version = ModelVersion(**{
                k: v for k, v in data.items() 
                if k in ModelVersion.__dataclass_fields__
            })
            
            return model_version
        
        except Exception as e:
            logger.error(f"Failed to get model version {version_id}: {e}")
            return None
    
    async def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        try:
            versions = []
            
            for metadata_file in self.metadata_dir.glob("*.json"):
                if metadata_file.stem.startswith(f"{model_name}_"):
                    version = await self.get_model_version(metadata_file.stem)
                    if version:
                        versions.append(version)
            
            # Sort by creation time, newest first
            versions.sort(key=lambda x: x.created_at, reverse=True)
            return versions
        
        except Exception as e:
            logger.error(f"Failed to list model versions for {model_name}: {e}")
            return []
    
    async def promote_model_version(self, version_id: str, target_status: ModelStatus) -> bool:
        """Promote model version to different environment."""
        try:
            version = await self.get_model_version(version_id)
            if not version:
                return False
            
            # Update status
            version.status = target_status
            if target_status == ModelStatus.PRODUCTION:
                version.deployed_at = datetime.utcnow().isoformat()
            elif target_status == ModelStatus.ARCHIVED:
                version.archived_at = datetime.utcnow().isoformat()
            
            # Save updated metadata
            metadata_path = self.metadata_dir / f"{version_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(version), f, indent=2, default=str)
            
            logger.info(f"Promoted model version {version_id} to {target_status.value}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to promote model version {version_id}: {e}")
            return False
    
    async def rollback_model_version(self, model_name: str, target_version: str) -> bool:
        """Rollback model to previous version."""
        try:
            # Get current production version
            versions = await self.list_model_versions(model_name)
            current_production = None
            target_model = None
            
            for version in versions:
                if version.status == ModelStatus.PRODUCTION:
                    current_production = version
                if version.version_number == target_version:
                    target_model = version
            
            if not target_model:
                logger.error(f"Target version {target_version} not found")
                return False
            
            # Demote current production version
            if current_production:
                await self.promote_model_version(
                    current_production.version_id, 
                    ModelStatus.STAGING
                )
            
            # Promote target version to production
            success = await self.promote_model_version(
                target_model.version_id,
                ModelStatus.PRODUCTION
            )
            
            if success:
                logger.info(f"Rolled back {model_name} to version {target_version}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to rollback model {model_name}: {e}")
            return False
    
    async def compare_model_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare performance metrics between model versions."""
        try:
            model_a = await self.get_model_version(version_a)
            model_b = await self.get_model_version(version_b)
            
            if not model_a or not model_b:
                return {"error": "One or more versions not found"}
            
            comparison = {
                "version_a": {
                    "version_id": model_a.version_id,
                    "version_number": model_a.version_number,
                    "accuracy": model_a.accuracy,
                    "precision": model_a.precision,
                    "recall": model_a.recall,
                    "f1_score": model_a.f1_score,
                    "model_size_mb": model_a.model_size_mb,
                    "training_duration_seconds": model_a.training_duration_seconds
                },
                "version_b": {
                    "version_id": model_b.version_id,
                    "version_number": model_b.version_number,
                    "accuracy": model_b.accuracy,
                    "precision": model_b.precision,
                    "recall": model_b.recall,
                    "f1_score": model_b.f1_score,
                    "model_size_mb": model_b.model_size_mb,
                    "training_duration_seconds": model_b.training_duration_seconds
                },
                "differences": {
                    "accuracy": model_b.accuracy - model_a.accuracy,
                    "precision": model_b.precision - model_a.precision,
                    "recall": model_b.recall - model_a.recall,
                    "f1_score": model_b.f1_score - model_a.f1_score,
                    "model_size_mb": model_b.model_size_mb - model_a.model_size_mb,
                    "training_duration_seconds": model_b.training_duration_seconds - model_a.training_duration_seconds
                },
                "recommendations": self._generate_comparison_recommendations(model_a, model_b)
            }
            
            return comparison
        
        except Exception as e:
            logger.error(f"Failed to compare model versions: {e}")
            return {"error": str(e)}
    
    def _generate_comparison_recommendations(self, model_a: ModelVersion, model_b: ModelVersion) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        if model_b.accuracy > model_a.accuracy:
            recommendations.append(f"Version B shows {(model_b.accuracy - model_a.accuracy):.3f} improvement in accuracy")
        
        if model_b.f1_score > model_a.f1_score:
            recommendations.append(f"Version B shows {(model_b.f1_score - model_a.f1_score):.3f} improvement in F1 score")
        
        if model_b.model_size_mb < model_a.model_size_mb:
            recommendations.append(f"Version B is {(model_a.model_size_mb - model_b.model_size_mb):.1f}MB smaller")
        
        if model_b.training_duration_seconds < model_a.training_duration_seconds:
            reduction = model_a.training_duration_seconds - model_b.training_duration_seconds
            recommendations.append(f"Version B trains {reduction} seconds faster")
        
        # Overall recommendation
        if model_b.accuracy > model_a.accuracy and model_b.f1_score > model_a.f1_score:
            recommendations.append("RECOMMEND: Version B shows superior performance metrics")
        elif model_a.accuracy > model_b.accuracy and model_a.f1_score > model_b.f1_score:
            recommendations.append("RECOMMEND: Version A shows superior performance metrics")
        else:
            recommendations.append("NEUTRAL: Mixed performance improvements, consider business requirements")
        
        return recommendations
    
    async def get_model_artifact(self, version_id: str) -> Optional[bytes]:
        """Get model artifact data."""
        try:
            artifact_path = self.artifacts_dir / f"{version_id}.model"
            if not artifact_path.exists():
                return None
            
            with open(artifact_path, 'rb') as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"Failed to get model artifact {version_id}: {e}")
            return None
    
    async def delete_model_version(self, version_id: str) -> bool:
        """Delete model version and its artifacts."""
        try:
            # Remove metadata file
            metadata_path = self.metadata_dir / f"{version_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove artifact file
            artifact_path = self.artifacts_dir / f"{version_id}.model"
            if artifact_path.exists():
                artifact_path.unlink()
            
            logger.info(f"Deleted model version {version_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete model version {version_id}: {e}")
            return False
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            metadata_files = list(self.metadata_dir.glob("*.json"))
            artifact_files = list(self.artifacts_dir.glob("*.model"))
            
            total_artifact_size = sum(f.stat().st_size for f in artifact_files)
            
            return {
                "total_versions": len(metadata_files),
                "total_artifact_size_mb": total_artifact_size / (1024 * 1024),
                "storage_root": str(self.storage_root),
                "last_updated": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return {"error": str(e)}