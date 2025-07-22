"""Repository for model persistence and retrieval."""

import os
import json
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import shutil
import tempfile

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Neural model persistence will be limited.")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("Joblib not available. Some persistence features may be limited.")

from ...domain.entities.neuro_symbolic_model import NeuroSymbolicModel
from ...domain.entities.knowledge_graph import KnowledgeGraph, Triple
from ..config.settings import get_config


class ModelRepository(ABC):
    """Abstract repository for model persistence."""
    
    @abstractmethod
    def save_model(self, model: NeuroSymbolicModel, version: Optional[str] = None) -> str:
        """Save a neuro-symbolic model and return version identifier."""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str, version: Optional[str] = None) -> NeuroSymbolicModel:
        """Load a neuro-symbolic model by ID and optional version."""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a model. If version is None, deletes all versions."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata."""
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get metadata about a model."""
        pass


class FileSystemModelRepository(ModelRepository):
    """File system-based model repository."""
    
    def __init__(self, base_path: Optional[str] = None):
        config = get_config()
        
        if base_path is None:
            self.base_path = Path(config.storage.base_path) / config.storage.models_path
        else:
            self.base_path = Path(base_path)
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versioning_enabled = config.storage.versioning_enabled
        self.max_versions = config.storage.max_versions
        self.compression = config.storage.compression
        self.save_format = config.storage.save_format
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get the base path for a model."""
        return self.base_path / model_id
    
    def _get_version_path(self, model_id: str, version: str) -> Path:
        """Get the path for a specific model version."""
        return self._get_model_path(model_id) / f"v{version}"
    
    def _generate_version(self, model_id: str) -> str:
        """Generate a new version identifier."""
        if not self.versioning_enabled:
            return "latest"
        
        model_path = self._get_model_path(model_id)
        if not model_path.exists():
            return "001"
        
        # Find highest existing version
        versions = []
        for version_dir in model_path.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith('v'):
                try:
                    version_num = int(version_dir.name[1:])
                    versions.append(version_num)
                except ValueError:
                    continue
        
        next_version = max(versions, default=0) + 1
        return f"{next_version:03d}"
    
    def _cleanup_old_versions(self, model_id: str) -> None:
        """Remove old versions if max_versions limit is exceeded."""
        if not self.versioning_enabled or self.max_versions <= 0:
            return
        
        model_path = self._get_model_path(model_id)
        if not model_path.exists():
            return
        
        # Get all version directories sorted by version number
        version_dirs = []
        for version_dir in model_path.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith('v'):
                try:
                    version_num = int(version_dir.name[1:])
                    version_dirs.append((version_num, version_dir))
                except ValueError:
                    continue
        
        version_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Remove excess versions
        for _, version_dir in version_dirs[self.max_versions:]:
            try:
                shutil.rmtree(version_dir)
            except Exception as e:
                warnings.warn(f"Failed to cleanup old version {version_dir}: {e}")
    
    def _save_metadata(self, model_path: Path, model: NeuroSymbolicModel, version: str) -> None:
        """Save model metadata."""
        metadata = {
            'id': model.id,
            'name': model.name,
            'neural_backbone': model.neural_backbone,
            'symbolic_reasoner': model.symbolic_reasoner,
            'is_trained': model.is_trained,
            'version': model.version,
            'saved_version': version,
            'num_knowledge_graphs': len(model.knowledge_graphs),
            'num_symbolic_constraints': len(model.symbolic_constraints),
            'created_at': datetime.now().isoformat(),
            'save_format': self.save_format,
            'compressed': self.compression
        }
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, model_path: Path) -> Dict[str, Any]:
        """Load model metadata."""
        metadata_path = model_path / 'metadata.json'
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load metadata: {e}")
            return {}
    
    def _save_knowledge_graphs(self, model_path: Path, knowledge_graphs: List[KnowledgeGraph]) -> None:
        """Save knowledge graphs."""
        kg_dir = model_path / 'knowledge_graphs'
        kg_dir.mkdir(exist_ok=True)
        
        for i, kg in enumerate(knowledge_graphs):
            kg_path = kg_dir / f'kg_{i}.json'
            
            kg_data = {
                'id': kg.id,
                'name': kg.name,
                'triples': [
                    {
                        'subject': t.subject,
                        'predicate': t.predicate, 
                        'object': t.object,
                        'confidence': t.confidence,
                        'metadata': t.metadata
                    }
                    for t in kg.triples
                ],
                'rules': [
                    {
                        'id': r.id,
                        'antecedent': r.antecedent,
                        'consequent': r.consequent,
                        'logic_type': r.logic_type.value,
                        'confidence': r.confidence,
                        'metadata': r.metadata
                    }
                    for r in kg.rules
                ],
                'namespaces': kg.namespaces
            }
            
            with open(kg_path, 'w') as f:
                json.dump(kg_data, f, indent=2)
    
    def _load_knowledge_graphs(self, model_path: Path) -> List[KnowledgeGraph]:
        """Load knowledge graphs."""
        kg_dir = model_path / 'knowledge_graphs'
        if not kg_dir.exists():
            return []
        
        knowledge_graphs = []
        
        for kg_file in sorted(kg_dir.glob('kg_*.json')):
            try:
                with open(kg_file, 'r') as f:
                    kg_data = json.load(f)
                
                kg = KnowledgeGraph(
                    id=kg_data['id'],
                    name=kg_data['name'],
                    namespaces=kg_data.get('namespaces', {})
                )
                
                # Load triples
                for triple_data in kg_data.get('triples', []):
                    triple = Triple(
                        subject=triple_data['subject'],
                        predicate=triple_data['predicate'],
                        object=triple_data['object'],
                        confidence=triple_data.get('confidence', 1.0),
                        metadata=triple_data.get('metadata', {})
                    )
                    kg.triples.append(triple)
                
                # Load rules
                from ...infrastructure.adapters.symbolic_adapter import LogicalRule, LogicType
                for rule_data in kg_data.get('rules', []):
                    rule = LogicalRule(
                        id=rule_data['id'],
                        antecedent=rule_data['antecedent'],
                        consequent=rule_data['consequent'],
                        logic_type=LogicType(rule_data['logic_type']),
                        confidence=rule_data.get('confidence', 1.0),
                        metadata=rule_data.get('metadata', {})
                    )
                    kg.rules.append(rule)
                
                knowledge_graphs.append(kg)
                
            except Exception as e:
                warnings.warn(f"Failed to load knowledge graph from {kg_file}: {e}")
        
        return knowledge_graphs
    
    def _save_neural_components(self, model_path: Path, model: NeuroSymbolicModel) -> None:
        """Save neural network components (placeholder for actual implementation)."""
        # In a full implementation, this would save the actual neural network weights
        # For now, we just save a placeholder
        neural_dir = model_path / 'neural'
        neural_dir.mkdir(exist_ok=True)
        
        neural_data = {
            'backbone_type': model.neural_backbone,
            'is_trained': model.is_trained,
            'architecture_info': {
                'type': model.neural_backbone,
                'parameters': {}  # Would contain actual model parameters
            }
        }
        
        with open(neural_dir / 'neural_info.json', 'w') as f:
            json.dump(neural_data, f, indent=2)
        
        # If PyTorch is available and model has neural components, save them
        if HAS_TORCH:
            # Placeholder for actual neural network saving
            # torch.save(model.neural_network.state_dict(), neural_dir / 'model.pth')
            pass
    
    def _load_neural_components(self, model_path: Path) -> Dict[str, Any]:
        """Load neural network components."""
        neural_dir = model_path / 'neural'
        if not neural_dir.exists():
            return {}
        
        neural_info_path = neural_dir / 'neural_info.json'
        if not neural_info_path.exists():
            return {}
        
        try:
            with open(neural_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load neural components: {e}")
            return {}
    
    def save_model(self, model: NeuroSymbolicModel, version: Optional[str] = None) -> str:
        """Save a neuro-symbolic model."""
        if version is None:
            version = self._generate_version(model.id)
        
        model_path = self._get_version_path(model.id, version)
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save metadata
            self._save_metadata(model_path, model, version)
            
            # Save knowledge graphs
            self._save_knowledge_graphs(model_path, model.knowledge_graphs)
            
            # Save neural components
            self._save_neural_components(model_path, model)
            
            # Save symbolic constraints
            constraints_path = model_path / 'symbolic_constraints.json'
            with open(constraints_path, 'w') as f:
                json.dump(model.symbolic_constraints, f, indent=2)
            
            # Cleanup old versions if necessary
            self._cleanup_old_versions(model.id)
            
            return version
            
        except Exception as e:
            # Cleanup failed save
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                except:
                    pass
            raise RuntimeError(f"Failed to save model {model.id}: {e}")
    
    def load_model(self, model_id: str, version: Optional[str] = None) -> NeuroSymbolicModel:
        """Load a neuro-symbolic model."""
        if version is None:
            if self.versioning_enabled:
                # Find latest version
                model_path = self._get_model_path(model_id)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model {model_id} not found")
                
                versions = []
                for version_dir in model_path.iterdir():
                    if version_dir.is_dir() and version_dir.name.startswith('v'):
                        try:
                            version_num = int(version_dir.name[1:])
                            versions.append((version_num, version_dir.name[1:]))
                        except ValueError:
                            continue
                
                if not versions:
                    raise FileNotFoundError(f"No versions found for model {model_id}")
                
                version = max(versions)[1]
            else:
                version = "latest"
        
        model_path = self._get_version_path(model_id, version)
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} version {version} not found")
        
        try:
            # Load metadata
            metadata = self._load_metadata(model_path)
            
            # Load knowledge graphs
            knowledge_graphs = self._load_knowledge_graphs(model_path)
            
            # Load symbolic constraints
            constraints_path = model_path / 'symbolic_constraints.json'
            symbolic_constraints = []
            if constraints_path.exists():
                with open(constraints_path, 'r') as f:
                    symbolic_constraints = json.load(f)
            
            # Load neural components info
            neural_info = self._load_neural_components(model_path)
            
            # Create model
            model = NeuroSymbolicModel(
                id=model_id,
                name=metadata.get('name', f'Model {model_id}'),
                neural_backbone=metadata.get('neural_backbone', 'transformer'),
                symbolic_reasoner=metadata.get('symbolic_reasoner', 'first_order_logic'),
                knowledge_graphs=knowledge_graphs,
                symbolic_constraints=symbolic_constraints,
                is_trained=metadata.get('is_trained', False),
                version=metadata.get('version', '0.1.0')
            )
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id} version {version}: {e}")
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a model or specific version."""
        try:
            if version is None:
                # Delete entire model
                model_path = self._get_model_path(model_id)
                if model_path.exists():
                    shutil.rmtree(model_path)
                    return True
                return False
            else:
                # Delete specific version
                version_path = self._get_version_path(model_id, version)
                if version_path.exists():
                    shutil.rmtree(version_path)
                    
                    # Check if model directory is now empty
                    model_path = self._get_model_path(model_id)
                    if model_path.exists() and not any(model_path.iterdir()):
                        model_path.rmdir()
                    
                    return True
                return False
                
        except Exception as e:
            warnings.warn(f"Failed to delete model {model_id}: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        
        for model_dir in self.base_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_id = model_dir.name
            versions = []
            
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir() and (version_dir.name.startswith('v') or version_dir.name == 'latest'):
                    metadata = self._load_metadata(version_dir)
                    versions.append({
                        'version': version_dir.name[1:] if version_dir.name.startswith('v') else version_dir.name,
                        'created_at': metadata.get('created_at'),
                        'is_trained': metadata.get('is_trained', False),
                        'size_mb': self._get_directory_size(version_dir) / (1024 * 1024)
                    })
            
            if versions:
                # Get latest version metadata for model info
                latest_version_dir = max(
                    model_dir.iterdir(),
                    key=lambda x: x.stat().st_mtime if x.is_dir() else 0
                )
                latest_metadata = self._load_metadata(latest_version_dir)
                
                models.append({
                    'id': model_id,
                    'name': latest_metadata.get('name', model_id),
                    'neural_backbone': latest_metadata.get('neural_backbone'),
                    'symbolic_reasoner': latest_metadata.get('symbolic_reasoner'),
                    'versions': versions,
                    'latest_version': versions[-1]['version'] if versions else None
                })
        
        return models
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        model_path = self._get_model_path(model_id)
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        # Get all versions
        versions = []
        for version_dir in model_path.iterdir():
            if version_dir.is_dir():
                metadata = self._load_metadata(version_dir)
                versions.append({
                    'version': version_dir.name[1:] if version_dir.name.startswith('v') else version_dir.name,
                    'metadata': metadata,
                    'size_mb': self._get_directory_size(version_dir) / (1024 * 1024),
                    'files': [f.name for f in version_dir.rglob('*') if f.is_file()]
                })
        
        # Sort versions
        versions.sort(key=lambda x: x['version'])
        
        return {
            'id': model_id,
            'versions': versions,
            'total_size_mb': self._get_directory_size(model_path) / (1024 * 1024),
            'version_count': len(versions)
        }
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())


class DatabaseModelRepository(ModelRepository):
    """Database-based model repository (placeholder implementation)."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        # In a full implementation, this would initialize database connection
        warnings.warn("Database repository not fully implemented. Using file system fallback.")
        self._fallback = FileSystemModelRepository()
    
    def save_model(self, model: NeuroSymbolicModel, version: Optional[str] = None) -> str:
        return self._fallback.save_model(model, version)
    
    def load_model(self, model_id: str, version: Optional[str] = None) -> NeuroSymbolicModel:
        return self._fallback.load_model(model_id, version)
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        return self._fallback.delete_model(model_id, version)
    
    def list_models(self) -> List[Dict[str, Any]]:
        return self._fallback.list_models()
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        return self._fallback.get_model_info(model_id)


def create_model_repository(repository_type: str = "filesystem", **kwargs) -> ModelRepository:
    """Factory function to create model repositories."""
    if repository_type.lower() == "filesystem":
        return FileSystemModelRepository(**kwargs)
    elif repository_type.lower() == "database":
        return DatabaseModelRepository(**kwargs)
    else:
        raise ValueError(f"Unknown repository type: {repository_type}")


# Global repository instance
_repository: Optional[ModelRepository] = None


def get_model_repository() -> ModelRepository:
    """Get the global model repository instance."""
    global _repository
    if _repository is None:
        config = get_config()
        if config.storage.database_url:
            _repository = create_model_repository("database", database_url=config.storage.database_url)
        else:
            _repository = create_model_repository("filesystem")
    return _repository


def set_model_repository(repository: ModelRepository) -> None:
    """Set the global model repository instance."""
    global _repository
    _repository = repository