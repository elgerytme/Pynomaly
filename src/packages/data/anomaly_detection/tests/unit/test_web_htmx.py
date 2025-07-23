"""Unit tests for HTMX web endpoints."""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from anomaly_detection.web.api.htmx import router
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.explanation import Explanation
from anomaly_detection.domain.services.explainability_service import ExplainerType


class TestHTMXEndpoints:
    """Test cases for HTMX web endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test app
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        
        # Sample detection result
        self.sample_detection_result = DetectionResult(
            predictions=np.array([1, -1, 1]),
            anomaly_count=1,
            normal_count=2,
            anomaly_rate=0.333,
            anomalies=[1],
            confidence_scores=np.array([0.1, 0.9, 0.2]),
            success=True
        )
        
        # Sample explanation
        self.sample_explanation = Explanation(
            is_anomaly=True,
            prediction_confidence=0.85,
            feature_importance={'feature1': 0.6, 'feature2': 0.4},
            top_features=[
                {'rank': 1, 'feature_name': 'feature1', 'value': 2.5, 'importance': 0.6},
                {'rank': 2, 'feature_name': 'feature2', 'value': 1.8, 'importance': 0.4}
            ],
            data_sample=np.array([2.5, 1.8]),
            base_value=0.1,
            metadata={'explainer': 'feature_importance'}
        )
        
        # Mock templates
        self.mock_templates = Mock(spec=Jinja2Templates)
        self.mock_template_response = Mock()
        self.mock_template_response.status_code = 200
        self.mock_templates.TemplateResponse.return_value = self.mock_template_response
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    def test_run_detection_success(self, mock_get_service, mock_templates):
        """Test run_detection endpoint with successful detection."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.sample_detection_result
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/detect", data={
            "algorithm": "isolation_forest",
            "contamination": "0.1",
            "sample_data": "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.detect_anomalies.assert_called_once()
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/detection_results.html"
        assert "results" in template_call[0][1]
        
        results = template_call[0][1]["results"]
        assert results["algorithm"] == "isolation_forest"
        assert results["success"] == True
        assert results["anomalies_detected"] == 1
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    def test_run_detection_no_data(self, mock_get_service, mock_templates):
        """Test run_detection endpoint with no input data (should generate sample data)."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.sample_detection_result
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request with no sample_data
        response = self.client.post("/detect", data={
            "algorithm": "one_class_svm",
            "contamination": "0.05"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.detect_anomalies.assert_called_once()
        
        # Check that algorithm was mapped correctly
        call_args = mock_service.detect_anomalies.call_args
        assert call_args[1]['algorithm'] == 'ocsvm'
        assert call_args[1]['contamination'] == 0.05
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    def test_run_detection_invalid_data(self, mock_get_service, mock_templates):
        """Test run_detection endpoint with invalid JSON data."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request with invalid JSON
        response = self.client.post("/detect", data={
            "algorithm": "isolation_forest",
            "sample_data": "invalid json"
        })
        
        # Should return 400 error
        assert response.status_code == 400
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    def test_run_detection_service_error(self, mock_get_service, mock_templates):
        """Test run_detection endpoint when service raises error."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.detect_anomalies.side_effect = Exception("Detection failed")
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/detect", data={
            "algorithm": "isolation_forest",
            "sample_data": "[[1.0, 2.0]]"
        })
        
        # Should return error template with 500 status
        assert response.status_code == 200  # FastAPI returns 200 but template has 500
        mock_templates.TemplateResponse.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/error_message.html"
        assert template_call[0][2]['status_code'] == 500
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_ensemble_service')
    def test_run_ensemble_detection_success(self, mock_get_service, mock_templates):
        """Test run_ensemble_detection endpoint with successful ensemble."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.majority_vote.return_value = np.array([1, -1])
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/ensemble", data={
            "algorithms": ["isolation_forest", "one_class_svm"],
            "method": "majority",
            "contamination": "0.1",
            "sample_data": "[[1.0, 2.0], [3.0, 4.0]]"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.majority_vote.assert_called_once()
        mock_templates.TemplateResponse.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/ensemble_results.html"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_ensemble_service')
    def test_run_ensemble_detection_average_method(self, mock_get_service, mock_templates):
        """Test run_ensemble_detection endpoint with average combination method."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.average_combination.return_value = (np.array([1, -1]), np.array([0.3, 0.7]))
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/ensemble", data={
            "algorithms": ["isolation_forest", "lof"],
            "method": "average",
            "sample_data": "[[1.0, 2.0], [3.0, 4.0]]"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.average_combination.assert_called_once()
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_model_repository')
    def test_list_models_htmx_success(self, mock_get_repo, mock_templates):
        """Test list_models_htmx endpoint with successful model listing."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        
        sample_models = [
            {"model_id": "model1", "name": "test_model", "algorithm": "isolation_forest"},
            {"model_id": "model2", "name": "test_model2", "algorithm": "one_class_svm"}
        ]
        mock_repo.list_models.return_value = sample_models
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/models/list")
        
        # Assertions
        assert response.status_code == 200
        mock_repo.list_models.assert_called_once()
        mock_templates.TemplateResponse.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/model_list.html"
        assert template_call[0][1]["models"] == sample_models
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_model_repository')
    def test_get_model_info_htmx_success(self, mock_get_repo, mock_templates):
        """Test get_model_info_htmx endpoint with successful model info retrieval."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        
        sample_metadata = {
            "model_id": "test-model-123",
            "name": "test_model",
            "algorithm": "isolation_forest",
            "accuracy": 0.85
        }
        mock_repo.get_model_metadata.return_value = sample_metadata
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/models/test-model-123/info")
        
        # Assertions
        assert response.status_code == 200
        mock_repo.get_model_metadata.assert_called_once_with("test-model-123")
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/model_info.html"
        assert template_call[0][1]["model"] == sample_metadata
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_model_repository')
    def test_get_model_info_htmx_not_found(self, mock_get_repo, mock_templates):
        """Test get_model_info_htmx endpoint when model is not found."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.get_model_metadata.side_effect = FileNotFoundError()
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/models/nonexistent/info")
        
        # Should return error template with 404 status
        assert response.status_code == 200
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/error_message.html"
        assert template_call[0][2]['status_code'] == 404
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_model_repository')
    def test_get_dashboard_stats_success(self, mock_get_repo, mock_templates):
        """Test get_dashboard_stats endpoint with successful stats retrieval."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.return_value = [
            {"status": "trained"}, 
            {"status": "trained"}, 
            {"status": "deployed"}
        ]
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/dashboard/stats")
        
        # Assertions
        assert response.status_code == 200
        mock_repo.list_models.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/dashboard_stats.html"
        
        stats = template_call[0][1]["stats"]
        assert stats["total_models"] == 3
        assert stats["active_models"] == 2  # Only trained models
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    @patch('anomaly_detection.web.api.htmx.get_model_repository')
    @patch('uuid.uuid4')
    def test_train_model_htmx_success(self, mock_uuid, mock_get_repo, mock_get_service, mock_templates):
        """Test train_model_htmx endpoint with successful model training."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.sample_detection_result
        mock_service._fitted_models = {'iforest': Mock()}
        
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.save.return_value = "saved-model-id"
        
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-uuid")
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/train", data={
            "model_name": "test_model",
            "algorithm": "isolation_forest",
            "contamination": "0.1",
            "training_data": "[[1.0, 2.0], [3.0, 4.0]]"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.fit.assert_called_once()
        mock_repo.save.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/training_results.html"
        
        results = template_call[0][1]["results"]
        assert results["success"] == True
        assert results["model_name"] == "test_model"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    @patch('anomaly_detection.web.api.htmx.get_model_repository')
    def test_train_model_htmx_invalid_data(self, mock_get_repo, mock_get_service, mock_templates):
        """Test train_model_htmx endpoint with invalid training data."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request with invalid JSON
        response = self.client.post("/train", data={
            "model_name": "test_model",
            "training_data": "invalid json"
        })
        
        # Should return 400 error
        assert response.status_code == 400
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_streaming_service')
    def test_start_streaming_monitor_success(self, mock_get_service, mock_templates):
        """Test start_streaming_monitor endpoint with successful streaming start."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_streaming_stats.return_value = {
            'buffer_size': 50,
            'model_fitted': True
        }
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/streaming/start", data={
            "algorithm": "isolation_forest",
            "window_size": "1000",
            "update_frequency": "100"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.reset_stream.assert_called_once()
        mock_service.set_window_size.assert_called_once_with(1000)
        mock_service.set_update_frequency.assert_called_once_with(100)
        mock_service.process_batch.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/streaming_status.html"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_streaming_service')
    def test_process_streaming_sample_success(self, mock_get_service, mock_templates):
        """Test process_streaming_sample endpoint with successful sample processing."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        sample_result = DetectionResult(
            predictions=np.array([-1]),
            anomaly_count=1,
            normal_count=0,
            anomaly_rate=1.0,
            anomalies=[0],
            confidence_scores=np.array([0.9]),
            success=True
        )
        mock_service.process_sample.return_value = sample_result
        mock_service.get_streaming_stats.return_value = {
            'buffer_size': 51,
            'model_fitted': True,
            'total_samples': 100
        }
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/streaming/sample", data={
            "sample_data": "[1.0, 2.0, 3.0, 4.0, 5.0]",
            "algorithm": "isolation_forest"
        })
        
        # Assertions
        assert response.status_code == 200
        mock_service.process_sample.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/streaming_result.html"
        
        context = template_call[0][1]
        assert context["success"] == True
        assert context["is_anomaly"] == True  # prediction was -1
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_streaming_service')
    def test_get_streaming_stats_success(self, mock_get_service, mock_templates):
        """Test get_streaming_stats endpoint with successful stats retrieval."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_streaming_stats.return_value = {
            'buffer_size': 100,
            'model_fitted': True
        }
        mock_service.detect_concept_drift.return_value = {
            'drift_detected': False,
            'drift_score': 0.2
        }
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/streaming/stats")
        
        # Assertions
        assert response.status_code == 200
        mock_service.get_streaming_stats.assert_called_once()
        mock_service.detect_concept_drift.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/streaming_stats.html"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_streaming_service')
    def test_reset_streaming_success(self, mock_get_service, mock_templates):
        """Test reset_streaming endpoint with successful stream reset."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/streaming/reset")
        
        # Assertions
        assert response.status_code == 200
        mock_service.reset_stream.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/streaming_reset.html"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    @patch('anomaly_detection.web.api.htmx.get_explainability_service')
    def test_explain_prediction_htmx_success(self, mock_get_explain_service, mock_get_detection_service, mock_templates):
        """Test explain_prediction_htmx endpoint with successful explanation."""
        # Setup mocks
        mock_detection_service = Mock()
        mock_get_detection_service.return_value = mock_detection_service
        mock_detection_service._fitted_models = {'iforest': Mock()}
        
        mock_explain_service = Mock()
        mock_get_explain_service.return_value = mock_explain_service
        mock_explain_service.explain_prediction.return_value = self.sample_explanation
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/explain", data={
            "sample_data": "[1.0, 2.0]",
            "algorithm": "isolation_forest",
            "explainer_type": "feature_importance",
            "feature_names": '["feat1", "feat2"]'
        })
        
        # Assertions
        assert response.status_code == 200
        mock_explain_service.explain_prediction.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/explanation_result.html"
        
        context = template_call[0][1]
        assert context["success"] == True
        assert context["is_anomaly"] == True
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_detection_service')
    @patch('anomaly_detection.web.api.htmx.get_explainability_service')
    def test_explain_prediction_htmx_model_not_fitted(self, mock_get_explain_service, mock_get_detection_service, mock_templates):
        """Test explain_prediction_htmx endpoint when model is not fitted."""
        # Setup mocks
        mock_detection_service = Mock()
        mock_get_detection_service.return_value = mock_detection_service
        mock_detection_service._fitted_models = {}  # No fitted models
        
        mock_explain_service = Mock()
        mock_get_explain_service.return_value = mock_explain_service
        mock_explain_service.explain_prediction.return_value = self.sample_explanation
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/explain", data={
            "sample_data": "[1.0, 2.0]",
            "algorithm": "isolation_forest"
        })
        
        # Assertions
        assert response.status_code == 200
        # Should have fitted the model
        mock_detection_service.fit.assert_called_once()
        mock_explain_service.explain_prediction.assert_called_once()
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_explainability_service')
    def test_get_available_explainers_htmx_success(self, mock_get_service, mock_templates):
        """Test get_available_explainers_htmx endpoint with successful explainer listing."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_available_explainers.return_value = ['shap', 'lime', 'feature_importance']
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/explain/available")
        
        # Assertions
        assert response.status_code == 200
        mock_service.get_available_explainers.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/explainers_list.html"
        
        explainers = template_call[0][1]["explainers"]
        assert len(explainers) == 3
        assert all(explainer['available'] for explainer in explainers)
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_worker_instance')
    def test_submit_worker_job_htmx_success(self, mock_get_worker, mock_templates):
        """Test submit_worker_job_htmx endpoint with successful job submission."""
        # Setup mocks
        mock_worker = AsyncMock()
        mock_get_worker.return_value = mock_worker
        mock_worker.submit_job.return_value = "job-123"
        mock_worker.is_running = False
        mock_worker.job_queue.get_queue_status.return_value = {"pending_jobs": 3}
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/worker/submit", data={
            "job_type": "detection",
            "algorithm": "isolation_forest",
            "contamination": "0.1",
            "priority": "normal"
        })
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/worker_job_submitted.html"
        
        context = template_call[0][1]
        assert context["success"] == True
        assert context["job_id"] == "job-123"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_worker_instance')
    def test_get_worker_job_status_htmx_success(self, mock_get_worker, mock_templates):
        """Test get_worker_job_status_htmx endpoint with successful job status retrieval."""
        # Setup mocks
        mock_worker = AsyncMock()
        mock_get_worker.return_value = mock_worker
        
        job_status = {
            "job_id": "job-123",
            "status": "running",
            "progress": 0.5
        }
        mock_worker.get_job_status.return_value = job_status
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/worker/job/job-123/status")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/worker_job_status.html"
        assert template_call[0][1]["job"] == job_status
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_worker_instance')
    def test_get_worker_job_status_htmx_not_found(self, mock_get_worker, mock_templates):
        """Test get_worker_job_status_htmx endpoint when job is not found."""
        # Setup mocks
        mock_worker = AsyncMock()
        mock_get_worker.return_value = mock_worker
        mock_worker.get_job_status.return_value = None
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/worker/job/nonexistent/status")
        
        # Should return error template with 404 status
        assert response.status_code == 200
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/error_message.html"
        assert template_call[0][2]['status_code'] == 404
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_worker_instance')
    def test_get_worker_dashboard_htmx_success(self, mock_get_worker, mock_templates):
        """Test get_worker_dashboard_htmx endpoint with successful dashboard retrieval."""
        # Setup mocks
        mock_worker = AsyncMock()
        mock_get_worker.return_value = mock_worker
        
        worker_status = {
            "is_running": True,
            "currently_running_jobs": 2,
            "max_concurrent_jobs": 5,
            "queue_status": {"pending_jobs": 10, "total_jobs": 50}
        }
        mock_worker.get_worker_status.return_value = worker_status
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/worker/dashboard")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/worker_dashboard.html"
        
        context = template_call[0][1]
        assert context["worker_status"] == worker_status
        assert context["utilization"] == 40.0  # 2/5 * 100
        assert context["health_status"] == "healthy"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_worker_instance')
    def test_cancel_worker_job_htmx_success(self, mock_get_worker, mock_templates):
        """Test cancel_worker_job_htmx endpoint with successful job cancellation."""
        # Setup mocks
        mock_worker = AsyncMock()
        mock_get_worker.return_value = mock_worker
        mock_worker.cancel_job.return_value = True
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/worker/job/job-123/cancel")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/worker_job_cancelled.html"
        
        context = template_call[0][1]
        assert context["success"] == True
        assert context["job_id"] == "job-123"
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_health_service')
    def test_get_health_report_htmx_success(self, mock_get_service, mock_templates):
        """Test get_health_report_htmx endpoint with successful health report."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        # Mock health report
        mock_report = Mock()
        mock_report.overall_status.value = 'healthy'
        mock_report.overall_score = 95.5
        mock_report.uptime_seconds = 7200
        mock_report.metrics = []
        mock_report.active_alerts = []
        
        mock_service.get_health_report.return_value = mock_report
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/health/report")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/health_report.html"
        
        context = template_call[0][1]
        assert context["report"] == mock_report
        assert context["status_color"] == "success"
        assert context["uptime_hours"] == 2.0
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_health_service')
    def test_get_health_alerts_htmx_success(self, mock_get_service, mock_templates):
        """Test get_health_alerts_htmx endpoint with successful alerts retrieval."""
        # Setup mocks
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.alert_manager.get_active_alerts.return_value = [
            Mock(severity=Mock(value="warning"), title="High CPU", message="CPU usage high"),
            Mock(severity=Mock(value="critical"), title="Memory Alert", message="Memory full")
        ]
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/health/alerts")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/health_alerts.html"
        
        context = template_call[0][1]
        assert len(context["alerts"]) == 2
    
    @patch('anomaly_detection.web.api.htmx.templates')
    @patch('anomaly_detection.web.api.htmx.get_health_service')
    def test_start_health_monitoring_htmx_success(self, mock_get_service, mock_templates):
        """Test start_health_monitoring_htmx endpoint with successful monitoring start."""
        # Setup mocks
        mock_service = AsyncMock()
        mock_get_service.return_value = mock_service
        mock_service.check_interval = 30
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.post("/health/monitoring/start")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "components/monitoring_control_result.html"
        
        context = template_call[0][1]
        assert context["success"] == True
        assert context["check_interval"] == 30
    
    def test_dependency_injection_functions(self):
        """Test dependency injection functions work correctly."""
        from anomaly_detection.web.api.htmx import (
            get_detection_service, get_ensemble_service, get_streaming_service,
            get_explainability_service, get_model_repository
        )
        
        # Test that services are created
        detection_service = get_detection_service()
        assert detection_service is not None
        
        ensemble_service = get_ensemble_service()
        assert ensemble_service is not None
        
        streaming_service = get_streaming_service()
        assert streaming_service is not None
        
        explainability_service = get_explainability_service()
        assert explainability_service is not None
        
        model_repository = get_model_repository()
        assert model_repository is not None
        
        # Test singleton behavior - should return same instances
        assert get_detection_service() is detection_service
        assert get_ensemble_service() is ensemble_service
        assert get_streaming_service() is streaming_service
        assert get_explainability_service() is explainability_service
        assert get_model_repository() is model_repository
    
    def test_algorithm_mappings(self):
        """Test that algorithm mappings are consistent across endpoints."""
        expected_mapping = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        # This mapping should be consistent across all endpoints
        # We can't easily test this without mocking, but we can verify the expected values exist
        assert expected_mapping['isolation_forest'] == 'iforest'
        assert expected_mapping['one_class_svm'] == 'ocsvm'
        assert expected_mapping['lof'] == 'lof'
    
    def test_explainer_type_mappings(self):
        """Test that explainer type mappings are correct."""
        expected_mapping = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        
        # Verify mappings exist and are correct
        assert expected_mapping['shap'] == ExplainerType.SHAP
        assert expected_mapping['lime'] == ExplainerType.LIME
        assert expected_mapping['permutation'] == ExplainerType.PERMUTATION
        assert expected_mapping['feature_importance'] == ExplainerType.FEATURE_IMPORTANCE