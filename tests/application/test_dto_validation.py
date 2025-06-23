"""Simple validation tests for DTOs without pytest."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

def test_imports():
    """Test importing DTOs."""
    results = {}
    
    # Test detector DTOs
    try:
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO, DetectorResponseDTO, DetectionRequestDTO
        results['detector'] = True
        print("✓ Detector DTOs imported successfully")
    except ImportError as e:
        results['detector'] = False
        print(f"✗ Detector DTOs failed: {e}")
    
    # Test dataset DTOs
    try:
        from pynomaly.application.dto.dataset_dto import CreateDatasetDTO, DatasetDTO, DataQualityReportDTO
        results['dataset'] = True
        print("✓ Dataset DTOs imported successfully")
    except ImportError as e:
        results['dataset'] = False
        print(f"✗ Dataset DTOs failed: {e}")
    
    # Test result DTOs
    try:
        from pynomaly.application.dto.result_dto import AnomalyDTO, DetectionResultDTO
        results['result'] = True
        print("✓ Result DTOs imported successfully")
    except ImportError as e:
        results['result'] = False
        print(f"✗ Result DTOs failed: {e}")
    
    # Test experiment DTOs
    try:
        from pynomaly.application.dto.experiment_dto import CreateExperimentDTO, ExperimentResponseDTO
        results['experiment'] = True
        print("✓ Experiment DTOs imported successfully")
    except ImportError as e:
        results['experiment'] = False
        print(f"✗ Experiment DTOs failed: {e}")
    
    # Test automl DTOs
    try:
        from pynomaly.application.dto.automl_dto import AutoMLRequestDTO, AutoMLResponseDTO, DatasetProfileDTO
        results['automl'] = True
        print("✓ AutoML DTOs imported successfully")
    except ImportError as e:
        results['automl'] = False
        print(f"✗ AutoML DTOs failed: {e}")
    
    # Test explainability DTOs
    try:
        from pynomaly.application.dto.explainability_dto import FeatureContributionDTO, LocalExplanationDTO, ExplanationRequestDTO
        results['explainability'] = True
        print("✓ Explainability DTOs imported successfully")
    except ImportError as e:
        results['explainability'] = False
        print(f"✗ Explainability DTOs failed: {e}")
    
    return results

def test_detector_dtos():
    """Test detector DTOs functionality."""
    try:
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO, DetectorResponseDTO, DetectionRequestDTO
        
        # Test CreateDetectorDTO
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1
        )
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        print("✓ CreateDetectorDTO works")
        
        # Test DetectorResponseDTO
        response = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow()
        )
        assert response.status == "active"
        assert response.version == "1.0.0"
        print("✓ DetectorResponseDTO works")
        
        # Test DetectionRequestDTO
        request = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0], [3.0, 4.0]]
        )
        assert len(request.data) == 2
        assert request.return_scores is True
        print("✓ DetectionRequestDTO works")
        
        # Test serialization
        data = dto.model_dump()
        assert data["name"] == "test_detector"
        print("✓ Detector DTO serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Detector DTOs failed: {e}")
        return False

def test_dataset_dtos():
    """Test dataset DTOs functionality."""
    try:
        from pynomaly.application.dto.dataset_dto import CreateDatasetDTO, DatasetDTO, DataQualityReportDTO
        
        # Test CreateDatasetDTO
        create_dto = CreateDatasetDTO(
            name="test_dataset",
            description="A test dataset"
        )
        assert create_dto.name == "test_dataset"
        assert create_dto.description == "A test dataset"
        print("✓ CreateDatasetDTO works")
        
        # Test DatasetDTO
        dataset_dto = DatasetDTO(
            id=uuid4(),
            name="test_dataset",
            shape=(100, 5),
            n_samples=100,
            n_features=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=1.0,
            numeric_features=5,
            categorical_features=0
        )
        assert dataset_dto.n_samples == 100
        assert dataset_dto.n_features == 5
        print("✓ DatasetDTO works")
        
        # Test DataQualityReportDTO
        quality_dto = DataQualityReportDTO(
            quality_score=0.85,
            n_missing_values=10,
            n_duplicates=5,
            n_outliers=15
        )
        assert quality_dto.quality_score == 0.85
        print("✓ DataQualityReportDTO works")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset DTOs failed: {e}")
        return False

def test_result_dtos():
    """Test result DTOs functionality."""
    try:
        from pynomaly.application.dto.result_dto import AnomalyDTO, DetectionResultDTO
        
        # Test AnomalyDTO
        anomaly_dto = AnomalyDTO(
            id=uuid4(),
            score=0.95,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="IsolationForest",
            timestamp=datetime.utcnow(),
            severity="high"
        )
        assert anomaly_dto.score == 0.95
        assert anomaly_dto.severity == "high"
        print("✓ AnomalyDTO works")
        
        # Test DetectionResultDTO
        result_dto = DetectionResultDTO(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            created_at=datetime.utcnow(),
            duration_seconds=1.5,
            anomalies=[anomaly_dto],
            total_samples=100,
            anomaly_count=1,
            contamination_rate=0.01,
            mean_score=0.2,
            max_score=0.95,
            min_score=0.05,
            threshold=0.5
        )
        assert result_dto.total_samples == 100
        assert result_dto.anomaly_count == 1
        assert len(result_dto.anomalies) == 1
        print("✓ DetectionResultDTO works")
        
        return True
        
    except Exception as e:
        print(f"✗ Result DTOs failed: {e}")
        return False

def test_experiment_dtos():
    """Test experiment DTOs functionality."""
    try:
        from pynomaly.application.dto.experiment_dto import CreateExperimentDTO, ExperimentResponseDTO
        
        # Test CreateExperimentDTO
        create_dto = CreateExperimentDTO(
            name="test_experiment",
            description="A test experiment"
        )
        assert create_dto.name == "test_experiment"
        print("✓ CreateExperimentDTO works")
        
        # Test ExperimentResponseDTO
        response_dto = ExperimentResponseDTO(
            id="test_id",
            name="test_experiment",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        assert response_dto.status == "active"
        assert response_dto.total_runs == 0
        print("✓ ExperimentResponseDTO works")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment DTOs failed: {e}")
        return False

def test_automl_dtos():
    """Test AutoML DTOs functionality."""
    try:
        from pynomaly.application.dto.automl_dto import AutoMLRequestDTO, AutoMLResponseDTO, DatasetProfileDTO
        
        # Test AutoMLRequestDTO
        request_dto = AutoMLRequestDTO(
            dataset_id="test_dataset"
        )
        assert request_dto.dataset_id == "test_dataset"
        assert request_dto.objective == "auc"
        print("✓ AutoMLRequestDTO works")
        
        # Test DatasetProfileDTO
        profile_dto = DatasetProfileDTO(
            n_samples=1000,
            n_features=10,
            contamination_estimate=0.1,
            feature_types={"f1": "numeric", "f2": "categorical"},
            missing_values_ratio=0.05,
            sparsity_ratio=0.0,
            dimensionality_ratio=0.01,
            dataset_size_mb=2.5,
            complexity_score=0.3
        )
        assert profile_dto.n_samples == 1000
        assert profile_dto.n_features == 10
        print("✓ DatasetProfileDTO works")
        
        # Test AutoMLResponseDTO
        response_dto = AutoMLResponseDTO(
            success=True,
            message="AutoML completed successfully",
            execution_time=300.0
        )
        assert response_dto.success is True
        print("✓ AutoMLResponseDTO works")
        
        return True
        
    except Exception as e:
        print(f"✗ AutoML DTOs failed: {e}")
        return False

def test_explainability_dtos():
    """Test explainability DTOs functionality."""
    try:
        from pynomaly.application.dto.explainability_dto import FeatureContributionDTO, LocalExplanationDTO, ExplanationRequestDTO
        
        # Test FeatureContributionDTO
        contribution_dto = FeatureContributionDTO(
            feature_name="age",
            value=35.0,
            contribution=0.25,
            importance=0.8,
            rank=1
        )
        assert contribution_dto.feature_name == "age"
        assert contribution_dto.rank == 1
        print("✓ FeatureContributionDTO works")
        
        # Test LocalExplanationDTO
        local_dto = LocalExplanationDTO(
            instance_id="inst_1",
            anomaly_score=0.8,
            prediction="anomaly",
            confidence=0.9,
            feature_contributions=[contribution_dto],
            explanation_method="shap",
            model_name="IsolationForest",
            timestamp="2024-01-01T00:00:00"
        )
        assert local_dto.instance_id == "inst_1"
        assert len(local_dto.feature_contributions) == 1
        print("✓ LocalExplanationDTO works")
        
        # Test ExplanationRequestDTO
        request_dto = ExplanationRequestDTO(
            detector_id="det_1"
        )
        assert request_dto.detector_id == "det_1"
        assert request_dto.explanation_method == "shap"
        print("✓ ExplanationRequestDTO works")
        
        return True
        
    except Exception as e:
        print(f"✗ Explainability DTOs failed: {e}")
        return False

def main():
    """Run all DTO tests."""
    print("Running DTO validation tests...\n")
    
    # Test imports
    print("=== Testing Imports ===")
    import_results = test_imports()
    print()
    
    # Test functionality for each DTO group
    results = {}
    
    if import_results.get('detector', False):
        print("=== Testing Detector DTOs ===")
        results['detector'] = test_detector_dtos()
        print()
    
    if import_results.get('dataset', False):
        print("=== Testing Dataset DTOs ===")
        results['dataset'] = test_dataset_dtos()
        print()
    
    if import_results.get('result', False):
        print("=== Testing Result DTOs ===")
        results['result'] = test_result_dtos()
        print()
    
    if import_results.get('experiment', False):
        print("=== Testing Experiment DTOs ===")
        results['experiment'] = test_experiment_dtos()
        print()
    
    if import_results.get('automl', False):
        print("=== Testing AutoML DTOs ===")
        results['automl'] = test_automl_dtos()
        print()
    
    if import_results.get('explainability', False):
        print("=== Testing Explainability DTOs ===")
        results['explainability'] = test_explainability_dtos()
        print()
    
    # Summary
    print("=== Summary ===")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Total DTO groups tested: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if total_tests > 0:
        print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
    
    for category, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {category.title()} DTOs")

if __name__ == "__main__":
    main()