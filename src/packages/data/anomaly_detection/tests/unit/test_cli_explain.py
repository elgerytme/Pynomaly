"""Unit tests for CLI explainability commands."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from typer.testing import CliRunner

from anomaly_detection.cli_new.commands.explain import app
from anomaly_detection.domain.services.explainability_service import ExplainerType
from anomaly_detection.domain.entities.explanation import Explanation


class TestExplainCommands:
    """Test cases for explainability CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.sample_explanation = Explanation(
            is_anomaly=True,
            prediction_confidence=0.85,
            feature_importance={'feature1': 0.6, 'feature2': 0.4, 'feature3': 0.2},
            top_features=[
                {'rank': 1, 'feature_name': 'feature1', 'value': 2.5, 'importance': 0.6},
                {'rank': 2, 'feature_name': 'feature2', 'value': 1.8, 'importance': 0.4},
                {'rank': 3, 'feature_name': 'feature3', 'value': 0.9, 'importance': 0.2}
            ],
            data_sample=np.array([2.5, 1.8, 0.9]),
            base_value=0.1,
            metadata={'explainer': 'feature_importance'}
        )
        
        self.sample_csv_data = "feature1,feature2,feature3\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
        self.sample_json_data = '[{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}]'
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2,feature3\n1.0,2.0,3.0\n2.5,1.8,0.9")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_csv_success(self, mock_explain_service, mock_detection_service,
                                       mock_read_csv, mock_file):
        """Test sample command with CSV file input."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 2.5],
            'feature2': [2.0, 1.8],
            'feature3': [3.0, 0.9]
        })
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.csv',
            '--sample', '1',
            '--algorithm', 'isolation_forest',
            '--explainer', 'feature_importance'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_csv.assert_called_once()
        mock_explain.explain_prediction.assert_called_once()
        
        # Check that correct parameters were passed
        call_args = mock_explain.explain_prediction.call_args
        assert call_args[1]['algorithm'] == 'iforest'  # mapped algorithm
        assert call_args[1]['explainer_type'] == ExplainerType.FEATURE_IMPORTANCE
        
        # Check output contains explanation information
        assert "Explanation generated successfully" in result.stdout
        assert "feature1" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data='[{"feature1": 1.0}]')
    @patch('pandas.read_json')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_json_success(self, mock_explain_service, mock_detection_service,
                                        mock_read_json, mock_file):
        """Test sample command with JSON file input."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 2.0]})
        mock_read_json.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.json',
            '--explainer', 'shap'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_json.assert_called_once()
        
        # Check SHAP explainer was selected
        call_args = mock_explain.explain_prediction.call_args
        assert call_args[1]['explainer_type'] == ExplainerType.SHAP
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_parquet')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_parquet_success(self, mock_explain_service, mock_detection_service,
                                           mock_read_parquet, mock_file):
        """Test sample command with Parquet file input."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_parquet.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.parquet',
            '--explainer', 'lime'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_parquet.assert_called_once()
        
        # Check LIME explainer was selected and training data was passed
        call_args = mock_explain.explain_prediction.call_args
        assert call_args[1]['explainer_type'] == ExplainerType.LIME
        assert call_args[1]['training_data'] is not None
    
    def test_sample_command_file_not_found(self):
        """Test sample command with non-existent file."""
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'nonexistent.csv'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    def test_sample_command_unsupported_format(self, mock_file):
        """Test sample command with unsupported file format."""
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.xml'
        ])
        
        assert result.exit_code == 1
        assert "Unsupported file format '.xml'" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0")
    @patch('pandas.read_csv')
    def test_sample_command_sample_index_out_of_range(self, mock_read_csv, mock_file):
        """Test sample command with sample index out of range."""
        # Setup mock
        mock_df = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        mock_read_csv.return_value = mock_df
        
        # Run command with index beyond data size
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.csv',
            '--sample', '5'
        ])
        
        assert result.exit_code == 1
        assert "Sample index 5 out of range" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.ModelRepository')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_with_saved_model(self, mock_explain_service, mock_detection_service,
                                           mock_repo_class, mock_read_csv, mock_file):
        """Test sample command with saved model."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_model = Mock()
        mock_model.model_object = Mock()
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.load.return_value = mock_model
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        mock_detection._fitted_models = {}
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.csv',
            '--model-id', 'saved-model-123'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_repo.load.assert_called_once_with('saved-model-123')
        assert "Loaded saved model: saved-model-123" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.ModelRepository')
    def test_sample_command_model_load_failure(self, mock_repo_class, mock_read_csv, mock_file):
        """Test sample command when model loading fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.load.side_effect = Exception("Model not found")
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.csv',
            '--model-id', 'invalid-model-id'
        ])
        
        assert result.exit_code == 1
        assert "Failed to load model invalid-model-id" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    @patch('json.dump')
    def test_sample_command_with_output_file(self, mock_json_dump, mock_explain_service,
                                           mock_detection_service, mock_read_csv, mock_file):
        """Test sample command with output file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.csv',
            '--output', 'explanation.json'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        
        # Check output data structure
        dumped_data = mock_json_dump.call_args[0][0]
        assert 'sample_index' in dumped_data
        assert 'algorithm' in dumped_data
        assert 'explainer_type' in dumped_data
        assert 'prediction' in dumped_data
        assert 'feature_importance' in dumped_data
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_explanation_failure(self, mock_explain_service, mock_detection_service,
                                              mock_read_csv, mock_file):
        """Test sample command when explanation fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.side_effect = Exception("Explanation failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'sample',
            '--input', 'test.csv'
        ])
        
        assert result.exit_code == 1
        assert "Explanation failed" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_batch_command_success(self, mock_explain_service, mock_detection_service,
                                  mock_read_csv, mock_file):
        """Test batch command with successful batch explanation."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 3.0, 5.0],
            'feature2': [2.0, 4.0, 6.0]
        })
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'batch',
            '--input', 'test.csv',
            '--max-samples', '2',
            '--explainer', 'permutation'
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert mock_explain.explain_prediction.call_count == 2  # max_samples
        assert "Generated 2 explanations" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_batch_command_anomalies_only(self, mock_explain_service, mock_detection_service,
                                         mock_read_csv, mock_file):
        """Test batch command with anomalies only filter."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 3.0, 5.0],
            'feature2': [2.0, 4.0, 6.0]
        })
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        # Mock detection result with some anomalies
        mock_detection_result = Mock()
        mock_detection_result.predictions = np.array([1, -1, 1])  # One anomaly at index 1
        mock_detection.detect_anomalies.return_value = mock_detection_result
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'batch',
            '--input', 'test.csv',
            '--anomalies-only'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_detection.detect_anomalies.assert_called_once()
        assert "Found 1 anomalies" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    @patch('json.dump')
    def test_batch_command_with_output_file(self, mock_json_dump, mock_explain_service,
                                          mock_detection_service, mock_read_csv, mock_file):
        """Test batch command with output file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Run command
        result = self.runner.invoke(app, [
            'batch',
            '--input', 'test.csv',
            '--output', 'batch_results.json'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        
        # Check output data structure
        dumped_data = mock_json_dump.call_args[0][0]
        assert 'algorithm' in dumped_data
        assert 'explainer_type' in dumped_data
        assert 'global_feature_importance' in dumped_data
        assert 'explanations' in dumped_data
    
    def test_batch_command_file_not_found(self):
        """Test batch command with non-existent file."""
        result = self.runner.invoke(app, [
            'batch',
            '--input', 'nonexistent.csv'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_batch_command_explanation_failure(self, mock_explain_service, mock_detection_service,
                                             mock_read_csv, mock_file):
        """Test batch command when explanation fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.side_effect = Exception("Explanation failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'batch',
            '--input', 'test.csv'
        ])
        
        assert result.exit_code == 1
        assert "Batch explanation failed" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_global_importance_command_success(self, mock_explain_service, mock_detection_service,
                                             mock_read_csv, mock_file):
        """Test global_importance command with successful analysis."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 3.0, 5.0],
            'feature2': [2.0, 4.0, 6.0],
            'feature3': [1.5, 3.5, 5.5]
        })
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        
        # Mock global feature importance
        global_importance = {
            'feature1': 0.6,
            'feature2': 0.3,
            'feature3': 0.1
        }
        mock_explain.get_global_feature_importance.return_value = global_importance
        
        # Run command
        result = self.runner.invoke(app, [
            'global-importance',
            '--input', 'test.csv',
            '--algorithm', 'isolation_forest',
            '--samples', '50'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_explain.get_global_feature_importance.assert_called_once()
        
        # Check that global importance was analyzed
        call_args = mock_explain.get_global_feature_importance.call_args
        assert call_args[1]['n_samples'] == 50
        assert call_args[1]['algorithm'] == 'iforest'
        
        assert "Global feature importance analysis complete" in result.stdout
        assert "feature1" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    @patch('json.dump')
    def test_global_importance_command_with_output_file(self, mock_json_dump, mock_explain_service,
                                                      mock_detection_service, mock_read_csv, mock_file):
        """Test global_importance command with output file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        
        global_importance = {'feature1': 0.7, 'feature2': 0.3}
        mock_explain.get_global_feature_importance.return_value = global_importance
        
        # Run command
        result = self.runner.invoke(app, [
            'global-importance',
            '--input', 'test.csv',
            '--output', 'global_importance.json'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        
        # Check output data structure
        dumped_data = mock_json_dump.call_args[0][0]
        assert 'algorithm' in dumped_data
        assert 'samples_analyzed' in dumped_data
        assert 'global_feature_importance' in dumped_data
        assert 'importance_ranking' in dumped_data
        assert 'statistics' in dumped_data
    
    def test_global_importance_command_file_not_found(self):
        """Test global_importance command with non-existent file."""
        result = self.runner.invoke(app, [
            'global-importance',
            '--input', 'nonexistent.csv'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_global_importance_command_analysis_failure(self, mock_explain_service, mock_detection_service,
                                                       mock_read_csv, mock_file):
        """Test global_importance command when analysis fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.get_global_feature_importance.side_effect = Exception("Analysis failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'global-importance',
            '--input', 'test.csv'
        ])
        
        assert result.exit_code == 1
        assert "Global importance analysis failed" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_available_command_success(self, mock_explain_service, mock_detection_service):
        """Test available command with successful explainer listing."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.get_available_explainers.return_value = [
            'shap', 'lime', 'permutation', 'feature_importance'
        ]
        
        # Run command
        result = self.runner.invoke(app, ['available'])
        
        # Assertions
        assert result.exit_code == 0
        mock_explain.get_available_explainers.assert_called_once()
        assert "Available Explainer Types" in result.stdout
        assert "shap" in result.stdout
        assert "lime" in result.stdout
        assert "permutation" in result.stdout
        assert "feature_importance" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_available_command_missing_explainers(self, mock_explain_service, mock_detection_service):
        """Test available command when some explainers are missing."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.get_available_explainers.return_value = [
            'permutation', 'feature_importance'  # Missing SHAP and LIME
        ]
        
        # Run command
        result = self.runner.invoke(app, ['available'])
        
        # Assertions
        assert result.exit_code == 0
        assert "pip install shap lime" in result.stdout
    
    def test_explainer_type_mappings(self):
        """Test that explainer type mappings are correct."""
        explainer_type_map = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        
        # Test that all expected mappings exist
        assert explainer_type_map['shap'] == ExplainerType.SHAP
        assert explainer_type_map['lime'] == ExplainerType.LIME
        assert explainer_type_map['permutation'] == ExplainerType.PERMUTATION
        assert explainer_type_map['feature_importance'] == ExplainerType.FEATURE_IMPORTANCE
    
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
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_different_explainers(self, mock_explain_service, mock_detection_service,
                                                mock_read_csv, mock_file):
        """Test sample command with different explainer types."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Test with different explainers
        explainers = ['shap', 'lime', 'permutation', 'feature_importance']
        for explainer in explainers:
            result = self.runner.invoke(app, [
                'sample',
                '--input', 'test.csv',
                '--explainer', explainer
            ])
            
            assert result.exit_code == 0
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.explain.DetectionService')
    @patch('anomaly_detection.cli_new.commands.explain.ExplainabilityService')
    def test_sample_command_different_algorithms(self, mock_explain_service, mock_detection_service,
                                                mock_read_csv, mock_file):
        """Test sample command with different algorithms."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_explain = Mock()
        mock_explain_service.return_value = mock_explain
        mock_explain.explain_prediction.return_value = self.sample_explanation
        
        # Test with different algorithms
        algorithms = ['isolation_forest', 'one_class_svm', 'lof']
        for algorithm in algorithms:
            result = self.runner.invoke(app, [
                'sample',
                '--input', 'test.csv',
                '--algorithm', algorithm
            ])
            
            assert result.exit_code == 0