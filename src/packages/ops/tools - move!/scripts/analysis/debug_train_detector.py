#!/usr/bin/env python3
"""Debug script to understand the training detector issue."""

import asyncio
import pandas as pd
from unittest.mock import Mock, AsyncMock

from monorepo.application.use_cases.train_detector import (
    TrainDetectorRequest,
    TrainDetectorUseCase,
)
from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.detector import Detector
from monorepo.domain.value_objects import ContaminationRate


async def debug_training():
    """Debug the training process."""
    print("Starting training debug...")
    
    # Create test data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 200],
    })
    dataset = Dataset(name="Test Data", data=data)
    
    # Create detector
    detector = Detector(
        name="Test Detector",
        algorithm_name="IsolationForest",
        parameters={"contamination": 0.1, "n_estimators": 100},
    )
    
    print(f"Initial detector.is_fitted: {detector.is_fitted}")
    
    # Create mocks
    mock_repo = Mock()
    mock_repo.find_by_id = AsyncMock(return_value=detector)
    mock_repo.save = AsyncMock()
    
    mock_validator = Mock()
    mock_validator.validate_numeric_features.return_value = ["feature1", "feature2"]
    mock_validator.check_data_quality.return_value = {
        "quality_score": 0.9,
        "missing_values": [],
        "constant_features": [],
    }
    
    mock_adapter = Mock()
    mock_adapter.fit_detector = Mock()  # Simple sync mock
    
    # Create use case
    use_case = TrainDetectorUseCase(
        detector_repository=mock_repo,
        feature_validator=mock_validator,
        adapter_registry=mock_adapter,
    )
    
    # Create request without parameters to avoid update_parameters call
    request = TrainDetectorRequest(
        detector_id=detector.id,
        training_data=dataset,
        parameters={},  # Empty parameters
        validate_data=True,
        save_model=True,
    )
    
    print(f"Before training, detector.is_fitted: {detector.is_fitted}")
    
    # Execute training
    response = await use_case.execute(request)
    
    print(f"After training, detector.is_fitted: {detector.is_fitted}")
    print(f"Response detector.is_fitted: {response.trained_detector.is_fitted}")
    
    # Test with parameters that would trigger update_parameters
    print("\n--- Testing with parameters ---")
    detector2 = Detector(
        name="Test Detector 2",
        algorithm_name="IsolationForest",
        parameters={"contamination": 0.1},
    )
    
    mock_repo.find_by_id = AsyncMock(return_value=detector2)
    
    request2 = TrainDetectorRequest(
        detector_id=detector2.id,
        training_data=dataset,
        parameters={"n_estimators": 150},  # This will trigger update_parameters
        validate_data=True,
        save_model=True,
    )
    
    print(f"Before training with params, detector2.is_fitted: {detector2.is_fitted}")
    
    response2 = await use_case.execute(request2)
    
    print(f"After training with params, detector2.is_fitted: {detector2.is_fitted}")
    print(f"Response2 detector.is_fitted: {response2.trained_detector.is_fitted}")
    
    # Check if update_parameters was called
    print(f"Detector2 parameters: {detector2.parameters}")


if __name__ == "__main__":
    asyncio.run(debug_training())