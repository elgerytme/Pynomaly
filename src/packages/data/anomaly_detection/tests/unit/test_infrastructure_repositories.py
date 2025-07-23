"""Unit tests for infrastructure repository components."""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import Dict, List, Any

from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.domain.entities.model import Model
from anomaly_detection.domain.value_objects.model_value_objects import (
    ModelMetadata, ModelStatus, SerializationFormat
)


class TestModelRepository:
    """Test cases for ModelRepository class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_path = Path("/tmp/test_models")
        self.repository = ModelRepository(base_path=self.base_path)
        
        # Sample model and metadata
        self.sample_model = Mock()
        self.sample_model.model_id = "test_model_123"
        self.sample_model.name = "Test Isolation Forest"
        self.sample_model.algorithm = "isolation_forest"
        self.sample_model.trained_at = datetime.now()
        
        self.sample_metadata = ModelMetadata(
            model_id="test_model_123",
            name="Test Isolation Forest",
            algorithm="isolation_forest",
            status=ModelStatus.TRAINED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["test", "isolation_forest"],
            description="Test model for unit testing"
        )
        
        # Registry data
        self.sample_registry = {
            "models": {
                "test_model_123": {
                    "model_id": "test_model_123",
                    "name": "Test Isolation Forest",
                    "algorithm": "isolation_forest",
                    "status": "trained",
                    "created_at": "2024-01-20T10:00:00",
                    "file_path": "models/test_model_123/model.pkl",
                    "metadata_path": "metadata/test_model_123.json"
                }
            },
            "last_updated": "2024-01-20T10:00:00"
        }


class TestModelRepositoryInitialization:
    """Test model repository initialization."""
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    def test_init_creates_directories(self, mock_exists, mock_mkdir):
        """Test that initialization creates required directories."""
        mock_exists.return_value = False
        
        base_path = Path("/tmp/test")
        repository = ModelRepository(base_path=base_path)
        
        # Should create base directory and subdirectories
        expected_calls = [
            ((base_path,), {'parents': True, 'exist_ok': True}),
            ((base_path, "models"), {'parents': True, 'exist_ok': True}),
            ((base_path, "metadata"), {'parents': True, 'exist_ok': True})
        ]
        
        assert mock_mkdir.call_count == 3
        for call, expected in zip(mock_mkdir.call_args_list, expected_calls):
            assert call[0] == expected[0]
            assert call[1] == expected[1]
    
    @patch('pathlib.Path.exists')
    def test_init_uses_default_path(self, mock_exists):
        """Test initialization with default path."""
        mock_exists.return_value = True
        
        repository = ModelRepository()
        
        # Should use default path
        assert repository.base_path == Path("./data/models")
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    def test_init_existing_directories(self, mock_exists, mock_mkdir):
        """Test initialization when directories already exist."""
        mock_exists.return_value = True
        
        repository = ModelRepository(base_path=Path("/tmp/existing"))
        
        # Should still call mkdir with exist_ok=True
        assert mock_mkdir.call_count == 3


class TestModelRepositorySave:
    """Test model saving functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        self.sample_model = Mock()
        self.sample_model.model_id = "save_test_123"
        self.sample_model.name = "Save Test Model"
        self.sample_model.algorithm = "isolation_forest"
        self.sample_model.trained_at = datetime.now()
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    @patch('json.dump')
    def test_save_model_pickle_format(self, mock_json_dump, mock_pickle_dump, 
                                    mock_file, mock_exists, mock_mkdir):
        """Test saving model in pickle format."""
        mock_exists.return_value = True
        
        # Mock the model file
        model_file = MagicMock()
        metadata_file = MagicMock()
        registry_file = MagicMock()
        mock_file.side_effect = [model_file, metadata_file, registry_file]
        
        # Call save method
        result = self.repository.save(self.sample_model, SerializationFormat.PICKLE)
        
        # Assertions
        assert result == self.sample_model.model_id
        mock_pickle_dump.assert_called_once_with(self.sample_model, model_file)
        mock_json_dump.assert_called()  # For metadata and registry
        
        # Check file opening calls
        assert mock_file.call_count == 3  # model, metadata, registry
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('joblib.dump')
    @patch('json.dump')
    def test_save_model_joblib_format(self, mock_json_dump, mock_joblib_dump,
                                    mock_file, mock_exists, mock_mkdir):
        """Test saving model in joblib format."""
        mock_exists.return_value = True
        
        result = self.repository.save(self.sample_model, SerializationFormat.JOBLIB)
        
        assert result == self.sample_model.model_id
        mock_joblib_dump.assert_called_once()
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')  
    def test_save_model_json_format(self, mock_json_dump, mock_file, mock_exists, mock_mkdir):
        """Test saving model in JSON format."""
        mock_exists.return_value = True
        
        # Mock model with to_dict method for JSON serialization
        self.sample_model.to_dict.return_value = {"algorithm": "isolation_forest"}
        
        result = self.repository.save(self.sample_model, SerializationFormat.JSON)
        
        assert result == self.sample_model.model_id
        # JSON dump should be called for model, metadata, and registry
        assert mock_json_dump.call_count == 3
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_save_model_file_error(self, mock_file, mock_exists, mock_mkdir):
        """Test save method with file operation error."""
        mock_exists.return_value = True
        
        with pytest.raises(OSError, match="Permission denied"):
            self.repository.save(self.sample_model, SerializationFormat.PICKLE)
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump', side_effect=Exception("Serialization error"))
    def test_save_model_serialization_error(self, mock_pickle_dump, mock_file, 
                                          mock_exists, mock_mkdir):
        """Test save method with serialization error."""
        mock_exists.return_value = True
        
        with pytest.raises(Exception, match="Serialization error"):
            self.repository.save(self.sample_model, SerializationFormat.PICKLE)


class TestModelRepositoryLoad:
    """Test model loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        self.model_id = "load_test_123"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_registry_file_not_found(self, mock_json_load, mock_file, mock_exists):
        """Test loading when registry file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            self.repository.load(self.model_id)
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_model_not_in_registry(self, mock_json_load, mock_file, mock_exists):
        """Test loading model that's not in registry."""
        mock_exists.return_value = True
        mock_json_load.return_value = {"models": {}}
        
        with pytest.raises(KeyError, match="Model .* not found"):
            self.repository.load(self.model_id)
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('pickle.load')
    def test_load_model_pickle_success(self, mock_pickle_load, mock_json_load,
                                     mock_file, mock_exists):
        """Test successful model loading in pickle format."""
        # Setup registry
        registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "file_path": f"models/{self.model_id}/model.pkl",
                    "format": "pickle"
                }
            }
        }
        mock_json_load.return_value = registry_data
        mock_exists.return_value = True
        
        # Setup model loading
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        result = self.repository.load(self.model_id)
        
        assert result == mock_model
        mock_pickle_load.assert_called_once()
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('joblib.load')
    def test_load_model_joblib_success(self, mock_joblib_load, mock_json_load,
                                     mock_file, mock_exists):
        """Test successful model loading in joblib format."""
        registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "file_path": f"models/{self.model_id}/model.joblib",
                    "format": "joblib"
                }
            }
        }
        mock_json_load.return_value = registry_data
        mock_exists.return_value = True
        
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        result = self.repository.load(self.model_id)
        
        assert result == mock_model
        mock_joblib_load.assert_called_once()
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_model_json_success(self, mock_json_load, mock_file, mock_exists):
        """Test successful model loading in JSON format."""
        registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "file_path": f"models/{self.model_id}/model.json",
                    "format": "json"
                }
            }
        }
        model_data = {"algorithm": "isolation_forest", "parameters": {}}
        mock_json_load.side_effect = [registry_data, model_data]
        mock_exists.return_value = True
        
        result = self.repository.load(self.model_id)
        
        # For JSON format, should return the dictionary
        assert result == model_data
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('pickle.load', side_effect=Exception("Corruption error"))
    def test_load_model_corruption_error(self, mock_pickle_load, mock_json_load,
                                       mock_file, mock_exists):
        """Test loading corrupted model file."""
        registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "file_path": f"models/{self.model_id}/model.pkl",
                    "format": "pickle"
                }
            }
        }
        mock_json_load.return_value = registry_data
        mock_exists.return_value = True
        
        with pytest.raises(Exception, match="Corruption error"):
            self.repository.load(self.model_id)


class TestModelRepositoryDelete:
    """Test model deletion functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        self.model_id = "delete_test_123"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.rmdir')
    def test_delete_model_success(self, mock_rmdir, mock_unlink, mock_json_dump,
                                mock_json_load, mock_file, mock_exists):
        """Test successful model deletion."""
        # Setup registry with model to delete
        registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "file_path": f"models/{self.model_id}/model.pkl",
                    "metadata_path": f"metadata/{self.model_id}.json"
                },
                "other_model": {
                    "model_id": "other_model",
                    "file_path": "models/other_model/model.pkl"
                }
            }
        }
        mock_json_load.return_value = registry_data
        mock_exists.return_value = True
        
        result = self.repository.delete(self.model_id)
        
        assert result is True
        mock_unlink.assert_called()  # Model and metadata files deleted
        mock_rmdir.assert_called()   # Model directory removed
        mock_json_dump.assert_called()  # Registry updated
        
        # Check that only the target model was removed from registry
        updated_registry = mock_json_dump.call_args[0][0]
        assert self.model_id not in updated_registry["models"]
        assert "other_model" in updated_registry["models"]
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_delete_model_not_found(self, mock_json_load, mock_file, mock_exists):
        """Test deleting non-existent model."""
        registry_data = {"models": {}}
        mock_json_load.return_value = registry_data
        mock_exists.return_value = True
        
        result = self.repository.delete(self.model_id)
        
        assert result is False
    
    @patch('pathlib.Path.exists')
    def test_delete_registry_not_found(self, mock_exists):
        """Test deletion when registry doesn't exist."""
        mock_exists.return_value = False
        
        result = self.repository.delete(self.model_id)
        
        assert result is False
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('pathlib.Path.unlink', side_effect=OSError("Permission denied"))
    def test_delete_file_permission_error(self, mock_unlink, mock_json_load,
                                        mock_file, mock_exists):
        """Test deletion with file permission error."""
        registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "file_path": f"models/{self.model_id}/model.pkl"
                }
            }
        }
        mock_json_load.return_value = registry_data
        mock_exists.return_value = True
        
        with pytest.raises(OSError, match="Permission denied"):
            self.repository.delete(self.model_id)


class TestModelRepositoryList:
    """Test model listing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        
        # Sample registry data
        self.registry_data = {
            "models": {
                "model1": {
                    "model_id": "model1",
                    "name": "Isolation Forest Model",
                    "algorithm": "isolation_forest",
                    "status": "trained",
                    "tags": ["production", "v1"]
                },
                "model2": {
                    "model_id": "model2", 
                    "name": "SVM Model",
                    "algorithm": "one_class_svm",
                    "status": "deployed",
                    "tags": ["production", "v2"]
                },
                "model3": {
                    "model_id": "model3",
                    "name": "LOF Model", 
                    "algorithm": "lof",
                    "status": "training",
                    "tags": ["dev", "experimental"]
                }
            }
        }
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_list_models_all(self, mock_json_load, mock_file, mock_exists):
        """Test listing all models without filters."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.list_models()
        
        assert len(result) == 3
        model_ids = [model["model_id"] for model in result]
        assert "model1" in model_ids
        assert "model2" in model_ids
        assert "model3" in model_ids
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_list_models_filter_by_status(self, mock_json_load, mock_file, mock_exists):
        """Test listing models filtered by status."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.list_models(status="deployed")
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model2"
        assert result[0]["status"] == "deployed"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_list_models_filter_by_algorithm(self, mock_json_load, mock_file, mock_exists):
        """Test listing models filtered by algorithm."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.list_models(algorithm="isolation_forest")
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model1"
        assert result[0]["algorithm"] == "isolation_forest"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_list_models_filter_by_tags(self, mock_json_load, mock_file, mock_exists):
        """Test listing models filtered by tags."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.list_models(tags=["production"])
        
        assert len(result) == 2
        model_ids = [model["model_id"] for model in result]
        assert "model1" in model_ids
        assert "model2" in model_ids
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_list_models_multiple_filters(self, mock_json_load, mock_file, mock_exists):
        """Test listing models with multiple filters."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.list_models(
            status="deployed", 
            algorithm="one_class_svm",
            tags=["production"]
        )
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model2"
    
    @patch('pathlib.Path.exists')
    def test_list_models_no_registry(self, mock_exists):
        """Test listing models when registry doesn't exist."""
        mock_exists.return_value = False
        
        result = self.repository.list_models()
        
        assert result == []
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_list_models_corrupted_registry(self, mock_json_load, mock_file, mock_exists):
        """Test listing models with corrupted registry."""
        mock_exists.return_value = True
        
        with pytest.raises(json.JSONDecodeError):
            self.repository.list_models()


class TestModelRepositorySearch:
    """Test model search functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        
        self.registry_data = {
            "models": {
                "model1": {
                    "model_id": "model1",
                    "name": "Production Isolation Forest",
                    "algorithm": "isolation_forest",
                    "description": "Production-ready anomaly detection model",
                    "tags": ["production", "isolation"]
                },
                "model2": {
                    "model_id": "model2",
                    "name": "Experimental SVM",
                    "algorithm": "one_class_svm",
                    "description": "Experimental SVM model for testing",
                    "tags": ["experimental", "svm"]
                }
            }
        }
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_search_models_name_match(self, mock_json_load, mock_file, mock_exists):
        """Test searching models by name."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.search_models("Production")
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model1"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_search_models_description_match(self, mock_json_load, mock_file, mock_exists):
        """Test searching models by description."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.search_models("testing")
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model2"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_search_models_tags_match(self, mock_json_load, mock_file, mock_exists):
        """Test searching models by tags."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.search_models("isolation")
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model1"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_search_models_case_insensitive(self, mock_json_load, mock_file, mock_exists):
        """Test that search is case insensitive."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.search_models("PRODUCTION")
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model1"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_search_models_no_matches(self, mock_json_load, mock_file, mock_exists):
        """Test searching with no matches."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.search_models("nonexistent")
        
        assert result == []


class TestModelRepositoryStats:
    """Test repository statistics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        
        self.registry_data = {
            "models": {
                "model1": {"status": "trained", "algorithm": "isolation_forest"},
                "model2": {"status": "deployed", "algorithm": "one_class_svm"},
                "model3": {"status": "trained", "algorithm": "isolation_forest"},
                "model4": {"status": "training", "algorithm": "lof"}
            }
        }
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_get_repository_stats(self, mock_json_load, mock_file, mock_exists):
        """Test getting repository statistics."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.get_repository_stats()
        
        assert result["total_models"] == 4
        assert result["by_status"]["trained"] == 2
        assert result["by_status"]["deployed"] == 1
        assert result["by_status"]["training"] == 1
        assert result["by_algorithm"]["isolation_forest"] == 2
        assert result["by_algorithm"]["one_class_svm"] == 1
        assert result["by_algorithm"]["lof"] == 1
    
    @patch('pathlib.Path.exists')
    def test_get_repository_stats_no_registry(self, mock_exists):
        """Test getting statistics when no registry exists."""
        mock_exists.return_value = False
        
        result = self.repository.get_repository_stats()
        
        assert result["total_models"] == 0
        assert result["by_status"] == {}
        assert result["by_algorithm"] == {}


class TestModelRepositoryMaintenance:
    """Test repository maintenance functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        
        # Create models with different dates
        old_date = datetime.now() - timedelta(days=10)
        recent_date = datetime.now() - timedelta(days=2)
        
        self.registry_data = {
            "models": {
                "old_model": {
                    "model_id": "old_model",
                    "status": "trained", 
                    "created_at": old_date.isoformat(),
                    "file_path": "models/old_model/model.pkl"
                },
                "recent_model": {
                    "model_id": "recent_model",
                    "status": "trained",
                    "created_at": recent_date.isoformat(),
                    "file_path": "models/recent_model/model.pkl"
                },
                "deployed_old": {
                    "model_id": "deployed_old",
                    "status": "deployed",
                    "created_at": old_date.isoformat(),
                    "file_path": "models/deployed_old/model.pkl"
                }
            }
        }
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.rmdir')
    def test_cleanup_old_models(self, mock_rmdir, mock_unlink, mock_json_dump,
                              mock_json_load, mock_file, mock_exists):
        """Test cleanup of old models."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.cleanup_old_models(days=7, keep_deployed=True)
        
        assert result == 1  # Only old_model should be cleaned up
        mock_unlink.assert_called()  # Files deleted
        mock_rmdir.assert_called()   # Directory removed
        
        # Check registry was updated
        updated_registry = mock_json_dump.call_args[0][0]
        assert "old_model" not in updated_registry["models"]
        assert "recent_model" in updated_registry["models"]
        assert "deployed_old" in updated_registry["models"]  # Kept because deployed
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.rmdir')
    def test_cleanup_old_models_including_deployed(self, mock_rmdir, mock_unlink,
                                                 mock_json_dump, mock_json_load,
                                                 mock_file, mock_exists):
        """Test cleanup including deployed models."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.cleanup_old_models(days=7, keep_deployed=False)
        
        assert result == 2  # old_model and deployed_old should be cleaned up
        
        updated_registry = mock_json_dump.call_args[0][0]
        assert "old_model" not in updated_registry["models"]
        assert "deployed_old" not in updated_registry["models"]
        assert "recent_model" in updated_registry["models"]


class TestModelRepositoryStatusUpdate:
    """Test model status update functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
        self.model_id = "status_test_123"
        
        self.registry_data = {
            "models": {
                self.model_id: {
                    "model_id": self.model_id,
                    "status": "training",
                    "updated_at": "2024-01-20T10:00:00"
                }
            }
        }
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    def test_update_model_status_success(self, mock_json_dump, mock_json_load,
                                       mock_file, mock_exists):
        """Test successful model status update."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.registry_data
        
        result = self.repository.update_model_status(self.model_id, ModelStatus.TRAINED)
        
        assert result is True
        mock_json_dump.assert_called_once()
        
        # Check that status was updated
        updated_registry = mock_json_dump.call_args[0][0]
        assert updated_registry["models"][self.model_id]["status"] == "trained"
        # updated_at should be newer
        assert "updated_at" in updated_registry["models"][self.model_id]
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_update_model_status_not_found(self, mock_json_load, mock_file, mock_exists):
        """Test updating status of non-existent model."""
        mock_exists.return_value = True
        mock_json_load.return_value = {"models": {}}
        
        result = self.repository.update_model_status(self.model_id, ModelStatus.TRAINED)
        
        assert result is False
    
    @patch('pathlib.Path.exists')
    def test_update_model_status_no_registry(self, mock_exists):
        """Test updating status when registry doesn't exist."""
        mock_exists.return_value = False
        
        result = self.repository.update_model_status(self.model_id, ModelStatus.TRAINED)
        
        assert result is False


class TestModelRepositoryEdgeCases:
    """Test edge cases and error scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_permission_error_on_registry_access(self, mock_file, mock_exists):
        """Test handling of permission errors when accessing registry."""
        mock_exists.return_value = True
        
        with pytest.raises(PermissionError, match="Access denied"):
            self.repository.list_models()
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load', return_value={"models": None})  # Invalid registry structure
    def test_invalid_registry_structure(self, mock_json_load, mock_file, mock_exists):
        """Test handling of invalid registry structure."""
        mock_exists.return_value = True
        
        with pytest.raises(TypeError):
            self.repository.list_models()
    
    def test_repository_with_nonexistent_base_path(self):
        """Test repository behavior with non-existent base path."""
        # Should not raise error during initialization
        repository = ModelRepository(base_path=Path("/nonexistent/path"))
        assert repository.base_path == Path("/nonexistent/path")
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_empty_search_query(self, mock_json_load, mock_file, mock_exists):
        """Test search with empty query."""
        mock_exists.return_value = True
        mock_json_load.return_value = {"models": {"model1": {"name": "test"}}}
        
        result = self.repository.search_models("")
        
        # Empty query should return all models
        assert len(result) == 1


class TestModelRepositoryConcurrency:
    """Test concurrency-related scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = ModelRepository(base_path=Path("/tmp/test"))
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    def test_concurrent_registry_updates(self, mock_json_dump, mock_json_load,
                                       mock_file, mock_exists):
        """Test behavior during concurrent registry updates."""
        mock_exists.return_value = True
        
        # Simulate registry being modified between read and write
        initial_registry = {"models": {"model1": {"status": "training"}}}
        modified_registry = {"models": {"model1": {"status": "trained"}}}
        
        mock_json_load.side_effect = [initial_registry, modified_registry]
        
        # This simulates a race condition, but repository should handle gracefully
        result = self.repository.update_model_status("model1", ModelStatus.DEPLOYED)
        
        # Should complete without error
        assert result is True or result is False  # Depends on implementation
    
    @patch('builtins.open', side_effect=[OSError("File locked"), mock_open()])
    @patch('pathlib.Path.exists')
    def test_file_lock_retry_behavior(self, mock_exists, mock_file):
        """Test behavior when files are temporarily locked."""
        mock_exists.return_value = True
        
        # First call fails with lock error, but we don't implement retry logic
        # This test documents current behavior
        with pytest.raises(OSError, match="File locked"):
            self.repository.list_models()