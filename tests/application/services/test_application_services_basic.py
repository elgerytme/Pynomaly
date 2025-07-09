"""
Basic tests for core application services.
Tests service instantiation and basic method signatures.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from pynomaly.application.services import DetectionService, EnsembleService
from pynomaly.application.services.autonomous_service import AutonomousDetectionService
from pynomaly.application.services.training_service import AutomatedTrainingService


class TestApplicationServicesBasic:
    """Basic tests for application services."""

    def test_detection_service_instantiation(self):
        """Test DetectionService can be instantiated."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_result_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = DetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            result_repository=mock_result_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Verify
        assert service is not None
        assert hasattr(service, 'detect_anomalies')
        assert hasattr(service, 'get_detection_results')

    def test_ensemble_service_instantiation(self):
        """Test EnsembleService can be instantiated."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_ensemble_repo = Mock()
        mock_detection_service = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = EnsembleService(
            detector_repository=mock_detector_repo,
            ensemble_repository=mock_ensemble_repo,
            detection_service=mock_detection_service,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Verify
        assert service is not None
        assert hasattr(service, 'create_ensemble')
        assert hasattr(service, 'detect_with_ensemble')

    def test_autonomous_detection_service_instantiation(self):
        """Test AutonomousDetectionService can be instantiated."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_experiment_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = AutonomousDetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            experiment_repository=mock_experiment_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Verify
        assert service is not None
        assert hasattr(service, 'create_autonomous_detector')
        assert hasattr(service, 'run_autonomous_detection')

    def test_automated_training_service_instantiation(self):
        """Test AutomatedTrainingService can be instantiated."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_training_job_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = AutomatedTrainingService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            training_job_repository=mock_training_job_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Verify
        assert service is not None
        assert hasattr(service, 'train_detector')
        assert hasattr(service, 'create_training_job')

    def test_detection_service_method_signatures(self):
        """Test DetectionService has expected method signatures."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_result_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = DetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            result_repository=mock_result_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Check method signatures
        assert callable(getattr(service, 'detect_anomalies', None))
        assert callable(getattr(service, 'get_detection_results', None))
        
        # Check if methods are async
        method = getattr(service, 'detect_anomalies')
        import inspect
        assert inspect.iscoroutinefunction(method)

    def test_ensemble_service_method_signatures(self):
        """Test EnsembleService has expected method signatures."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_ensemble_repo = Mock()
        mock_detection_service = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = EnsembleService(
            detector_repository=mock_detector_repo,
            ensemble_repository=mock_ensemble_repo,
            detection_service=mock_detection_service,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Check method signatures
        assert callable(getattr(service, 'create_ensemble', None))
        assert callable(getattr(service, 'detect_with_ensemble', None))

    def test_autonomous_detection_service_method_signatures(self):
        """Test AutonomousDetectionService has expected method signatures."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_experiment_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = AutonomousDetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            experiment_repository=mock_experiment_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Check method signatures
        assert callable(getattr(service, 'create_autonomous_detector', None))
        assert callable(getattr(service, 'run_autonomous_detection', None))

    def test_automated_training_service_method_signatures(self):
        """Test AutomatedTrainingService has expected method signatures."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_training_job_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = AutomatedTrainingService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            training_job_repository=mock_training_job_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Check method signatures
        assert callable(getattr(service, 'train_detector', None))
        assert callable(getattr(service, 'create_training_job', None))

    def test_service_dependency_injection(self):
        """Test that services properly inject dependencies."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_result_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Instantiate service
        service = DetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            result_repository=mock_result_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Verify dependencies are injected
        assert service.detector_repository == mock_detector_repo
        assert service.dataset_repository == mock_dataset_repo
        assert service.result_repository == mock_result_repo
        assert service.algorithm_registry == mock_algorithm_registry

    def test_service_initialization_errors(self):
        """Test service initialization with missing dependencies."""
        # Test DetectionService with missing dependencies
        with pytest.raises(TypeError):
            DetectionService()
            
        # Test with None dependencies
        with pytest.raises((TypeError, AttributeError)):
            DetectionService(
                detector_repository=None,
                dataset_repository=None,
                result_repository=None,
                algorithm_registry=None,
            )

    def test_service_method_existence(self):
        """Test that all expected methods exist on services."""
        # Create mock dependencies
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_result_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Test DetectionService
        detection_service = DetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            result_repository=mock_result_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        expected_methods = [
            'detect_anomalies',
            'get_detection_results',
        ]
        
        for method_name in expected_methods:
            assert hasattr(detection_service, method_name), f"Missing method: {method_name}"
            method = getattr(detection_service, method_name)
            assert callable(method), f"Method {method_name} is not callable"

    def test_service_configuration(self):
        """Test service configuration and settings."""
        # Create mock dependencies with configuration
        mock_detector_repo = Mock()
        mock_dataset_repo = Mock()
        mock_result_repo = Mock()
        mock_algorithm_registry = Mock()
        
        # Test service with configuration
        service = DetectionService(
            detector_repository=mock_detector_repo,
            dataset_repository=mock_dataset_repo,
            result_repository=mock_result_repo,
            algorithm_registry=mock_algorithm_registry,
        )
        
        # Verify service is properly configured
        assert service is not None
        assert hasattr(service, 'detector_repository')
        assert hasattr(service, 'dataset_repository')
        assert hasattr(service, 'result_repository')
        assert hasattr(service, 'algorithm_registry')

    def test_service_isolation(self):
        """Test that services are properly isolated."""
        # Create different mock dependencies for each service
        mock_detector_repo1 = Mock()
        mock_detector_repo2 = Mock()
        mock_dataset_repo1 = Mock()
        mock_dataset_repo2 = Mock()
        mock_result_repo1 = Mock()
        mock_result_repo2 = Mock()
        mock_algorithm_registry1 = Mock()
        mock_algorithm_registry2 = Mock()
        
        # Create two service instances
        service1 = DetectionService(
            detector_repository=mock_detector_repo1,
            dataset_repository=mock_dataset_repo1,
            result_repository=mock_result_repo1,
            algorithm_registry=mock_algorithm_registry1,
        )
        
        service2 = DetectionService(
            detector_repository=mock_detector_repo2,
            dataset_repository=mock_dataset_repo2,
            result_repository=mock_result_repo2,
            algorithm_registry=mock_algorithm_registry2,
        )
        
        # Verify services are isolated
        assert service1 is not service2
        assert service1.detector_repository is not service2.detector_repository
        assert service1.dataset_repository is not service2.dataset_repository
        assert service1.result_repository is not service2.result_repository
        assert service1.algorithm_registry is not service2.algorithm_registry