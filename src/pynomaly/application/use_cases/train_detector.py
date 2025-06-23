"""Use case for training a detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import FittingError, InsufficientDataError
from pynomaly.domain.services import FeatureValidator
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol, DetectorRepositoryProtocol


@dataclass
class TrainDetectorRequest:
    """Request for training a detector."""
    
    detector_id: UUID
    training_data: Dataset
    validation_split: Optional[float] = None
    hyperparameter_grid: Optional[Dict[str, List[Any]]] = None
    cv_folds: Optional[int] = None
    scoring_metric: Optional[str] = None
    save_model: bool = True
    early_stopping: bool = False
    max_training_time: Optional[int] = None
    contamination_rate: Optional[ContaminationRate] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    validate_data: bool = True
    experiment_name: Optional[str] = None

    # Backward compatibility properties
    @property
    def dataset(self) -> Dataset:
        """Backward compatibility for dataset property."""
        return self.training_data


@dataclass
class TrainDetectorResponse:
    """Response from detector training."""
    
    trained_detector: Detector
    training_metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    training_warnings: Optional[List[str]] = None

    # Backward compatibility properties
    @property
    def detector_id(self) -> UUID:
        """Backward compatibility for detector_id property."""
        return self.trained_detector.id

    @property
    def training_time_ms(self) -> float:
        """Backward compatibility for training_time_ms property."""
        if self.training_metrics and "training_time" in self.training_metrics:
            return self.training_metrics["training_time"] * 1000
        return 0.0

    @property
    def dataset_summary(self) -> dict:
        """Backward compatibility for dataset_summary property."""
        if self.training_metrics and "dataset_summary" in self.training_metrics:
            return self.training_metrics["dataset_summary"]
        return {}

    @property
    def parameters_used(self) -> dict:
        """Backward compatibility for parameters_used property."""
        return self.trained_detector.parameters

    @property
    def validation_results(self) -> Optional[dict]:
        """Backward compatibility for validation_results property."""
        if self.training_metrics and "validation_results" in self.training_metrics:
            return self.training_metrics["validation_results"]
        return None


class TrainDetectorUseCase:
    """Use case for training an anomaly detector."""
    
    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        feature_validator: FeatureValidator,
        min_samples: int = 10
    ):
        """Initialize the use case.
        
        Args:
            detector_repository: Repository for detectors
            feature_validator: Service for validating features
            min_samples: Minimum samples required for training
        """
        self.detector_repository = detector_repository
        self.feature_validator = feature_validator
        self.min_samples = min_samples
    
    async def execute(self, request: TrainDetectorRequest) -> TrainDetectorResponse:
        """Execute detector training.
        
        Args:
            request: Training request
            
        Returns:
            Training response with comprehensive metrics
            
        Raises:
            InsufficientDataError: If dataset is too small
            FittingError: If training fails
        """
        start_time = datetime.utcnow()
        warnings = []
        
        # Load detector
        detector = self.detector_repository.find_by_id(request.detector_id)
        if detector is None:
            raise ValueError(f"Detector {request.detector_id} not found")
        
        # Check dataset size
        if request.training_data.n_samples < self.min_samples:
            raise InsufficientDataError(
                dataset_name=request.training_data.name,
                n_samples=request.training_data.n_samples,
                min_required=self.min_samples,
                operation="training"
            )
        
        validation_results = None
        
        # Validate data if requested
        if request.validate_data:
            # Check for numeric features
            numeric_features = self.feature_validator.validate_numeric_features(
                request.training_data
            )
            
            if not numeric_features:
                raise FittingError(
                    detector_name=detector.name,
                    reason="No numeric features found in dataset",
                    dataset_name=request.training_data.name
                )
            
            # Check data quality
            quality_report = self.feature_validator.check_data_quality(
                request.training_data
            )
            validation_results = {
                "numeric_features": len(numeric_features),
                "quality_score": quality_report["quality_score"],
                "issues": []
            }
            
            if quality_report["missing_values"]:
                validation_results["issues"].append(
                    f"Missing values in {len(quality_report['missing_values'])} features"
                )
                warnings.append(f"Missing values detected in {len(quality_report['missing_values'])} features")
            
            if quality_report["constant_features"]:
                validation_results["issues"].append(
                    f"{len(quality_report['constant_features'])} constant features"
                )
                warnings.append(f"{len(quality_report['constant_features'])} constant features detected")
        
        # Handle hyperparameter tuning
        best_parameters = {}
        if request.hyperparameter_grid:
            # For now, we'll simulate parameter selection by taking the first value
            # In a real implementation, this would involve cross-validation
            for param, values in request.hyperparameter_grid.items():
                if values:
                    best_parameters[param] = values[0]
                    warnings.append(f"Using first value for {param}: {values[0]}")
        
        # Update parameters if provided
        final_parameters = {**request.parameters, **best_parameters}
        if final_parameters:
            detector.update_parameters(**final_parameters)
        
        # Update contamination rate if provided
        if request.contamination_rate:
            detector.contamination_rate = request.contamination_rate
        
        # Handle validation split if specified
        training_dataset = request.training_data
        validation_score = None
        if request.validation_split and request.validation_split > 0:
            # For now, we'll just note that validation split was requested
            warnings.append(f"Validation split {request.validation_split} requested but not implemented")
        
        # Train the detector
        try:
            detector.fit(training_dataset)
        except Exception as e:
            raise FittingError(
                detector_name=detector.name,
                reason=str(e),
                dataset_name=training_dataset.name
            ) from e
        
        # Calculate training time
        end_time = datetime.utcnow()
        training_time_seconds = (end_time - start_time).total_seconds()
        
        # Update detector metadata
        detector.trained_at = end_time
        detector.is_fitted = True
        detector.update_metadata("last_training_dataset", training_dataset.name)
        detector.update_metadata("last_training_samples", training_dataset.n_samples)
        
        # Build comprehensive training metrics
        training_metrics = {
            "training_time": training_time_seconds,
            "dataset_summary": training_dataset.summary(),
            "validation_results": validation_results,
            "best_parameters": best_parameters or final_parameters,
            "training_samples": training_dataset.n_samples,
            "features_count": training_dataset.n_features
        }
        
        if validation_score is not None:
            training_metrics["validation_score"] = validation_score
        
        # Generate model path if saving
        model_path = None
        if request.save_model:
            self.detector_repository.save(detector)
            # Model artifact saving would be handled by the adapter
            model_path = f"/models/{detector.name}_{detector.id}.pkl"
        
        # Check for early stopping and max training time
        if request.early_stopping:
            warnings.append("Early stopping was enabled but not implemented")
        
        if request.max_training_time and training_time_seconds > request.max_training_time:
            warnings.append(f"Training time ({training_time_seconds:.2f}s) exceeded max time ({request.max_training_time}s)")
        
        # Create comprehensive response
        return TrainDetectorResponse(
            trained_detector=detector,
            training_metrics=training_metrics,
            model_path=model_path,
            training_warnings=warnings if warnings else None
        )