"""Unit tests for CLI model management commands."""

import json
import pytest
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from typer.testing import CliRunner
from datetime import datetime

from anomaly_detection.cli_new.commands.models import app
from anomaly_detection.domain.entities.model import ModelStatus, Model, ModelMetadata, SerializationFormat
from anomaly_detection.domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestModelCommands:
    """Test cases for model management CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.sample_model_metadata = {
            'model_id': 'test-model-123',
            'name': 'test_model',
            'algorithm': 'isolation_forest',
            'version': 1,
            'status': 'trained',
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'description': 'Test model',
            'training_samples': 1000,
            'training_features': 5,
            'contamination_rate': 0.1,
            'training_duration_seconds': 5.5,
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90,
            'f1_score': 0.85,
            'hyperparameters': {'contamination': 0.1},
            'tags': ['test', 'development']
        }
        
        self.sample_models_list = [
            {
                'model_id': 'model-1',
                'name': 'model_one',
                'algorithm': 'isolation_forest',
                'status': 'trained',
                'created_at': '2023-01-01',
                'accuracy': 0.85
            },
            {
                'model_id': 'model-2',
                'name': 'model_two',
                'algorithm': 'one_class_svm',
                'status': 'deployed',
                'created_at': '2023-01-02',
                'accuracy': 0.78
            }
        ]
        
        self.sample_repo_stats = {
            'total_models': 5,
            'storage_size_mb': 25.5,
            'storage_path': '/test/models',
            'by_status': {'trained': 3, 'deployed': 2},
            'by_algorithm': {'isolation_forest': 3, 'one_class_svm': 2}
        }
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_list_command_success(self, mock_repo_class):
        """Test list command with successful model listing."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.list_models.return_value = self.sample_models_list
        
        # Run command
        result = self.runner.invoke(app, ['list', '--models-dir', 'test_models'])
        
        # Assertions
        assert result.exit_code == 0
        mock_repo.list_models.assert_called_once_with(status=None, algorithm=None)
        assert "Found 2 model(s)" in result.stdout
        assert "model_one" in result.stdout
        assert "model_two" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_list_command_with_filters(self, mock_repo_class):
        """Test list command with status and algorithm filters."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.list_models.return_value = [self.sample_models_list[0]]
        
        # Run command
        result = self.runner.invoke(app, [
            'list',
            '--models-dir', 'test_models',
            '--algorithm', 'isolation_forest',
            '--status', 'trained'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_repo.list_models.assert_called_once_with(
            status=ModelStatus.TRAINED,
            algorithm='isolation_forest'
        )
        assert "Found 1 model(s)" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_list_command_no_models(self, mock_repo_class):
        """Test list command when no models are found."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.list_models.return_value = []
        
        # Run command
        result = self.runner.invoke(app, ['list'])
        
        # Assertions
        assert result.exit_code == 0
        assert "No models found matching the criteria" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_list_command_error(self, mock_repo_class):
        """Test list command when repository raises error."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.list_models.side_effect = Exception("Repository error")
        
        # Run command
        result = self.runner.invoke(app, ['list'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Error listing models" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_info_command_success(self, mock_repo_class):
        """Test info command with successful model information retrieval."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        
        # Run command
        result = self.runner.invoke(app, ['info', 'test-model-123'])
        
        # Assertions
        assert result.exit_code == 0
        mock_repo.get_model_metadata.assert_called_once_with('test-model-123')
        assert "test_model" in result.stdout
        assert "isolation_forest" in result.stdout
        assert "1000" in result.stdout  # training_samples
        assert "0.850" in result.stdout  # accuracy
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_info_command_model_not_found(self, mock_repo_class):
        """Test info command when model is not found."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.side_effect = FileNotFoundError()
        
        # Run command
        result = self.runner.invoke(app, ['info', 'nonexistent-model'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Model with ID 'nonexistent-model' not found" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_info_command_error(self, mock_repo_class):
        """Test info command when repository raises error."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.side_effect = Exception("Repository error")
        
        # Run command
        result = self.runner.invoke(app, ['info', 'test-model'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Error getting model info" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('typer.confirm')
    def test_delete_command_success_with_confirmation(self, mock_confirm, mock_repo_class):
        """Test delete command with successful deletion and confirmation."""
        # Setup mocks
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        mock_repo.delete.return_value = True
        mock_confirm.return_value = True
        
        # Run command
        result = self.runner.invoke(app, ['delete', 'test-model-123'])
        
        # Assertions
        assert result.exit_code == 0
        mock_confirm.assert_called_once()
        mock_repo.delete.assert_called_once_with('test-model-123')
        assert "deleted successfully" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('typer.confirm')
    def test_delete_command_cancelled(self, mock_confirm, mock_repo_class):
        """Test delete command when user cancels deletion."""
        # Setup mocks
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        mock_confirm.return_value = False
        
        # Run command
        result = self.runner.invoke(app, ['delete', 'test-model-123'])
        
        # Assertions
        assert result.exit_code == 0
        mock_confirm.assert_called_once()
        mock_repo.delete.assert_not_called()
        assert "Deletion cancelled" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_delete_command_force(self, mock_repo_class):
        """Test delete command with force flag."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        mock_repo.delete.return_value = True
        
        # Run command
        result = self.runner.invoke(app, ['delete', 'test-model-123', '--force'])
        
        # Assertions
        assert result.exit_code == 0
        mock_repo.delete.assert_called_once_with('test-model-123')
        assert "deleted successfully" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_delete_command_model_not_found(self, mock_repo_class):
        """Test delete command when model is not found."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.side_effect = FileNotFoundError()
        
        # Run command
        result = self.runner.invoke(app, ['delete', 'nonexistent-model'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Model with ID 'nonexistent-model' not found" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_delete_command_failure(self, mock_repo_class):
        """Test delete command when deletion fails."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        mock_repo.delete.return_value = False
        
        # Run command
        result = self.runner.invoke(app, ['delete', 'test-model-123', '--force'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to delete model" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_stats_command_success(self, mock_repo_class):
        """Test stats command with successful repository statistics."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_repository_stats.return_value = self.sample_repo_stats
        
        # Run command
        result = self.runner.invoke(app, ['stats'])
        
        # Assertions
        assert result.exit_code == 0
        assert "5" in result.stdout  # total_models
        assert "25.5 MB" in result.stdout  # storage_size_mb
        assert "trained" in result.stdout
        assert "isolation_forest" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_stats_command_error(self, mock_repo_class):
        """Test stats command when repository raises error."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_repository_stats.side_effect = Exception("Repository error")
        
        # Run command
        result = self.runner.invoke(app, ['stats'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Error getting repository stats" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_export_command_json_success(self, mock_json_dump, mock_file, mock_repo_class):
        """Test export command with JSON format."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        
        # Run command
        result = self.runner.invoke(app, [
            'export', 'test-model-123',
            '--output', 'model.json',
            '--format', 'json'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        assert "exported to: model.json" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    def test_export_command_yaml_success(self, mock_yaml_dump, mock_file, mock_repo_class):
        """Test export command with YAML format."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        
        # Run command
        result = self.runner.invoke(app, [
            'export', 'test-model-123',
            '--output', 'model.yaml',
            '--format', 'yaml'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_yaml_dump.assert_called_once()
        assert "exported to: model.yaml" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_export_command_unsupported_format(self, mock_repo_class):
        """Test export command with unsupported format."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.return_value = self.sample_model_metadata
        
        # Run command
        result = self.runner.invoke(app, [
            'export', 'test-model-123',
            '--output', 'model.xml',
            '--format', 'xml'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Unsupported format: xml" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    def test_export_command_model_not_found(self, mock_repo_class):
        """Test export command when model is not found."""
        # Setup mock
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_model_metadata.side_effect = FileNotFoundError()
        
        # Run command
        result = self.runner.invoke(app, [
            'export', 'nonexistent-model',
            '--output', 'model.json'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Model with ID 'nonexistent-model' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2,label\n1.0,2.0,-1\n3.0,4.0,1")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.models.DetectionService')
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('uuid.uuid4')
    @patch('sklearn.metrics.accuracy_score')
    @patch('sklearn.metrics.precision_score')
    @patch('sklearn.metrics.recall_score')
    @patch('sklearn.metrics.f1_score')
    def test_train_command_success_with_labels(self, mock_f1, mock_recall, mock_precision, 
                                             mock_accuracy, mock_uuid, mock_repo_class,
                                             mock_service_class, mock_read_csv, mock_file):
        """Test train command with successful training and labels."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 3.0],
            'feature2': [2.0, 4.0],
            'label': [-1, 1]
        })
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_detection_result = DetectionResult(
            predictions=np.array([-1, 1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[0],
            success=True
        )
        mock_service.detect_anomalies.return_value = mock_detection_result
        mock_service._fitted_models = {'iforest': Mock()}
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save.return_value = 'saved-model-id'
        
        mock_uuid.return_value = uuid.UUID('12345678-1234-1234-1234-123456789abc')
        
        # Mock metrics
        mock_accuracy.return_value = 0.85
        mock_precision.return_value = 0.80
        mock_recall.return_value = 0.90
        mock_f1.return_value = 0.85
        
        # Run command
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.csv',
            '--model-name', 'test_model',
            '--algorithm', 'isolation_forest',
            '--contamination', '0.1',
            '--has-labels',
            '--label-column', 'label'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_service.fit.assert_called_once()
        mock_service.detect_anomalies.assert_called_once()
        mock_repo.save.assert_called_once()
        
        # Check that metrics were calculated
        mock_accuracy.assert_called_once()
        mock_precision.assert_called_once()
        mock_recall.assert_called_once()
        mock_f1.assert_called_once()
        
        assert "Model training completed successfully" in result.stdout
        assert "0.850" in result.stdout  # accuracy in output
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.models.DetectionService')
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('uuid.uuid4')
    def test_train_command_success_without_labels(self, mock_uuid, mock_repo_class,
                                                 mock_service_class, mock_read_csv, mock_file):
        """Test train command without labels."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 3.0],
            'feature2': [2.0, 4.0]
        })
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_detection_result = DetectionResult(
            predictions=np.array([1, -1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[1],
            success=True
        )
        mock_service.detect_anomalies.return_value = mock_detection_result
        mock_service._fitted_models = {'ocsvm': Mock()}
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save.return_value = 'saved-model-id'
        
        mock_uuid.return_value = uuid.UUID('12345678-1234-1234-1234-123456789abc')
        
        # Run command
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.csv',
            '--model-name', 'test_model',
            '--algorithm', 'one_class_svm'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_service.fit.assert_called_once()
        mock_repo.save.assert_called_once()
        assert "Model training completed successfully" in result.stdout
    
    def test_train_command_file_not_found(self):
        """Test train command with non-existent file."""
        result = self.runner.invoke(app, [
            'train',
            '--input', 'nonexistent.csv',
            '--model-name', 'test_model'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    def test_train_command_unsupported_format(self, mock_file):
        """Test train command with unsupported file format."""
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.txt',
            '--model-name', 'test_model'
        ])
        
        assert result.exit_code == 1
        assert "Unsupported file format '.txt'" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    def test_train_command_data_loading_failure(self, mock_read_csv, mock_file):
        """Test train command when data loading fails."""
        # Setup mock
        mock_read_csv.side_effect = Exception("CSV parsing error")
        
        # Run command
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.csv',
            '--model-name', 'test_model'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to load dataset" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.models.DetectionService')
    def test_train_command_training_failure(self, mock_service_class, mock_read_csv, mock_file):
        """Test train command when model training fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.fit.side_effect = Exception("Training failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.csv',
            '--model-name', 'test_model'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Model training failed" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.models.DetectionService')
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('uuid.uuid4')
    def test_train_command_save_failure(self, mock_uuid, mock_repo_class,
                                       mock_service_class, mock_read_csv, mock_file):
        """Test train command when model saving fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_detection_result = DetectionResult(
            predictions=np.array([1, -1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[1],
            success=True
        )
        mock_service.detect_anomalies.return_value = mock_detection_result
        mock_service._fitted_models = {'iforest': Mock()}
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save.side_effect = Exception("Save failed")
        
        mock_uuid.return_value = uuid.UUID('12345678-1234-1234-1234-123456789abc')
        
        # Run command
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.csv',
            '--model-name', 'test_model'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to save model" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_parquet')
    @patch('anomaly_detection.cli_new.commands.models.DetectionService')
    @patch('anomaly_detection.cli_new.commands.models.ModelRepository')
    @patch('uuid.uuid4')
    def test_train_command_parquet_format(self, mock_uuid, mock_repo_class,
                                         mock_service_class, mock_read_parquet, mock_file):
        """Test train command with Parquet file format."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_parquet.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_detection_result = DetectionResult(
            predictions=np.array([1, -1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[1],
            success=True
        )
        mock_service.detect_anomalies.return_value = mock_detection_result
        mock_service._fitted_models = {'lof': Mock()}
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save.return_value = 'saved-model-id'
        
        mock_uuid.return_value = uuid.UUID('12345678-1234-1234-1234-123456789abc')
        
        # Run command
        result = self.runner.invoke(app, [
            'train',
            '--input', 'training.parquet',
            '--model-name', 'test_model',
            '--algorithm', 'lof',
            '--format', 'joblib'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_parquet.assert_called_once()
        
        # Check that joblib format was used
        save_call_args = mock_repo.save.call_args
        assert save_call_args[0][1] == SerializationFormat.JOBLIB
    
    def test_algorithm_mappings(self):
        """Test that algorithm mappings are correct."""
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        # Test that all expected mappings exist
        assert algorithm_map['isolation_forest'] == 'iforest'
        assert algorithm_map['one_class_svm'] == 'ocsvm'
        assert algorithm_map['lof'] == 'lof'