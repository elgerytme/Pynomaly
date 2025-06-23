"""Application service for ensemble detection."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.entities import Dataset, DetectionResult, Anomaly
from pynomaly.domain.services import EnsembleAggregator, AnomalyScorer
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import (
    DetectorProtocol,
    EnsembleDetectorProtocol,
    DetectorRepositoryProtocol
)


class EnsembleService:
    """Service for managing ensemble anomaly detection."""
    
    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        ensemble_aggregator: EnsembleAggregator,
        anomaly_scorer: AnomalyScorer
    ):
        """Initialize ensemble service.
        
        Args:
            detector_repository: Repository for detectors
            ensemble_aggregator: Service for aggregating results
            anomaly_scorer: Service for score processing
        """
        self.detector_repository = detector_repository
        self.ensemble_aggregator = ensemble_aggregator
        self.anomaly_scorer = anomaly_scorer
    
    async def create_ensemble(
        self,
        name: str,
        detector_ids: List[UUID],
        weights: Optional[Dict[UUID, float]] = None,
        aggregation_method: str = "average"
    ) -> EnsembleDetector:
        """Create an ensemble from existing detectors.
        
        Args:
            name: Name for the ensemble
            detector_ids: IDs of detectors to include
            weights: Optional weights for each detector
            aggregation_method: How to aggregate results
            
        Returns:
            Ensemble detector instance
        """
        # Load detectors
        detectors = []
        for detector_id in detector_ids:
            detector = self.detector_repository.find_by_id(detector_id)
            if detector is None:
                raise ValueError(f"Detector {detector_id} not found")
            detectors.append(detector)
        
        # Create ensemble
        ensemble = EnsembleDetector(
            name=name,
            base_detectors=detectors,
            weights=weights,
            aggregation_method=aggregation_method,
            aggregator=self.ensemble_aggregator,
            scorer=self.anomaly_scorer
        )
        
        return ensemble
    
    async def optimize_ensemble_weights(
        self,
        ensemble: EnsembleDetector,
        validation_dataset: Dataset,
        optimization_metric: str = "f1"
    ) -> Dict[str, float]:
        """Optimize ensemble weights using validation data.
        
        Args:
            ensemble: Ensemble to optimize
            validation_dataset: Dataset with labels
            optimization_metric: Metric to optimize
            
        Returns:
            Optimized weights
        """
        if not validation_dataset.has_target:
            raise ValueError("Validation dataset must have labels")
        
        # Get true labels
        true_labels = validation_dataset.target.values  # type: ignore
        
        # Try different weight combinations
        best_weights = ensemble.get_current_weights()
        best_score = 0.0
        
        # Simple grid search (in practice, use optimization library)
        weight_options = [0.0, 0.25, 0.5, 0.75, 1.0]
        n_detectors = len(ensemble.base_detectors)
        
        # Generate weight combinations
        import itertools
        for weight_combo in itertools.product(weight_options, repeat=n_detectors):
            # Skip if all weights are zero
            if sum(weight_combo) == 0:
                continue
            
            # Set weights
            test_weights = dict(zip(
                [d.name for d in ensemble.base_detectors],
                weight_combo
            ))
            ensemble.update_weights(test_weights)
            
            # Evaluate
            result = ensemble.detect(validation_dataset)
            
            # Calculate metric
            if optimization_metric == "f1":
                from sklearn.metrics import f1_score
                score = f1_score(true_labels, result.labels)
            elif optimization_metric == "precision":
                from sklearn.metrics import precision_score
                score = precision_score(true_labels, result.labels)
            elif optimization_metric == "recall":
                from sklearn.metrics import recall_score
                score = recall_score(true_labels, result.labels)
            else:
                raise ValueError(f"Unknown metric: {optimization_metric}")
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_weights = test_weights.copy()
        
        # Apply best weights
        ensemble.update_weights(best_weights)
        
        return best_weights
    
    async def analyze_ensemble_diversity(
        self,
        ensemble: EnsembleDetector,
        dataset: Dataset
    ) -> Dict[str, Any]:
        """Analyze diversity of ensemble members.
        
        Args:
            ensemble: Ensemble to analyze
            dataset: Dataset to use for analysis
            
        Returns:
            Diversity metrics
        """
        # Get predictions from all members
        all_results = {}
        all_scores = {}
        all_labels = {}
        
        for detector in ensemble.base_detectors:
            result = detector.detect(dataset)
            all_results[detector.name] = result
            all_scores[detector.name] = [s.value for s in result.scores]
            all_labels[detector.name] = result.labels
        
        # Calculate agreement
        agreement_rate, per_sample_agreement = self.ensemble_aggregator.calculate_agreement(
            all_labels
        )
        
        # Calculate correlation between detectors
        import pandas as pd
        score_df = pd.DataFrame(all_scores)
        correlation_matrix = score_df.corr()
        
        # Calculate diversity metrics
        avg_correlation = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].mean()
        
        # Rank detectors by contribution
        detector_rankings = self.ensemble_aggregator.rank_detectors(
            {name: [AnomalyScore(value=v) for v in scores]
             for name, scores in all_scores.items()}
        )
        
        return {
            "agreement_rate": float(agreement_rate),
            "average_correlation": float(avg_correlation),
            "correlation_matrix": correlation_matrix.to_dict(),
            "detector_rankings": detector_rankings,
            "per_sample_agreement_stats": {
                "mean": float(per_sample_agreement.mean()),
                "std": float(per_sample_agreement.std()),
                "min": float(per_sample_agreement.min()),
                "max": float(per_sample_agreement.max())
            }
        }
    
    async def get_ensemble_explanations(
        self,
        ensemble: EnsembleDetector,
        dataset: Dataset,
        anomaly_indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get explanations from ensemble perspective.
        
        Args:
            ensemble: Ensemble detector
            dataset: Dataset containing anomalies
            anomaly_indices: Indices to explain
            
        Returns:
            Explanations for each anomaly
        """
        explanations = {}
        
        # Get individual detector opinions
        for idx in anomaly_indices:
            detector_opinions = {}
            
            for detector in ensemble.base_detectors:
                result = detector.detect(dataset)
                score = result.scores[idx].value
                label = result.labels[idx]
                
                detector_opinions[detector.name] = {
                    "score": score,
                    "label": label,
                    "agrees_with_ensemble": None  # Will be set later
                }
            
            # Get ensemble result
            ensemble_result = ensemble.detect(dataset)
            ensemble_label = ensemble_result.labels[idx]
            
            # Mark agreement
            for name, opinion in detector_opinions.items():
                opinion["agrees_with_ensemble"] = opinion["label"] == ensemble_label
            
            # Calculate consensus
            votes_anomaly = sum(
                1 for op in detector_opinions.values()
                if op["label"] == 1
            )
            consensus = votes_anomaly / len(detector_opinions)
            
            explanations[idx] = {
                "ensemble_score": ensemble_result.scores[idx].value,
                "ensemble_label": int(ensemble_label),
                "consensus": consensus,
                "detector_opinions": detector_opinions,
                "disagreement": consensus not in [0.0, 1.0]
            }
        
        return explanations


class EnsembleDetector:
    """Ensemble detector implementation."""
    
    def __init__(
        self,
        name: str,
        base_detectors: List[DetectorProtocol],
        weights: Optional[Dict[UUID, float]] = None,
        aggregation_method: str = "average",
        aggregator: Optional[EnsembleAggregator] = None,
        scorer: Optional[AnomalyScorer] = None
    ):
        """Initialize ensemble detector.
        
        Args:
            name: Name of the ensemble
            base_detectors: List of base detectors
            weights: Optional weights for detectors
            aggregation_method: Method for aggregation
            aggregator: Aggregator service
            scorer: Scorer service
        """
        self.id = uuid4()
        self.name = name
        self.base_detectors = base_detectors
        self.aggregation_method = aggregation_method
        self.aggregator = aggregator or EnsembleAggregator()
        self.scorer = scorer or AnomalyScorer()
        
        # Initialize weights
        if weights:
            self._weights = weights
        else:
            # Equal weights by default
            self._weights = {d.id: 1.0 for d in base_detectors}
        
        # Check if all base detectors are fitted
        self.is_fitted = all(d.is_fitted for d in base_detectors)
    
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using ensemble."""
        # Get results from all detectors
        all_scores = {}
        all_labels = {}
        
        for detector in self.base_detectors:
            result = detector.detect(dataset)
            all_scores[detector.name] = result.scores
            all_labels[detector.name] = result.labels
        
        # Aggregate scores
        weights_by_name = {
            d.name: self._weights.get(d.id, 1.0)
            for d in self.base_detectors
        }
        
        aggregated_scores = self.aggregator.aggregate_scores(
            all_scores,
            weights=weights_by_name,
            method=self.aggregation_method
        )
        
        # Aggregate labels
        aggregated_labels = self.aggregator.aggregate_labels(
            all_labels,
            weights=weights_by_name,
            method="majority"
        )
        
        # Calculate threshold from aggregated scores
        score_values = [s.value for s in aggregated_scores]
        contamination_rate = np.mean(aggregated_labels)
        threshold = self.scorer.calculate_threshold(
            aggregated_scores,
            contamination_rate
        )
        
        # Create anomalies
        anomalies = []
        anomaly_indices = np.where(aggregated_labels == 1)[0]
        
        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=aggregated_scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self.name
            )
            anomalies.append(anomaly)
        
        return DetectionResult(
            detector_id=self.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=aggregated_scores,
            labels=aggregated_labels,
            threshold=threshold,
            metadata={
                "ensemble_size": len(self.base_detectors),
                "aggregation_method": self.aggregation_method
            }
        )
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights by detector name."""
        return {
            d.name: self._weights.get(d.id, 1.0)
            for d in self.base_detectors
        }
    
    def update_weights(self, weights_by_name: Dict[str, float]) -> None:
        """Update weights using detector names."""
        for detector in self.base_detectors:
            if detector.name in weights_by_name:
                self._weights[detector.id] = weights_by_name[detector.name]