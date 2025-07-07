"""Query strategies for active learning sample selection."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from pynomaly.domain.models.active_learning import (
    ActiveLearningConfig,
    QueryResult,
    QueryStrategy,
    SamplePool,
)


class BaseQueryStrategy(ABC):
    """Base class for active learning query strategies."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def select_samples(
        self,
        sample_pool: SamplePool,
        model_predictions: Dict[str, Any],
        labeled_samples: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QueryResult:
        """Select samples for labeling."""
        pass
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize selection scores to [0, 1]."""
        if not scores:
            return scores
        
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in scores.keys()}
        
        return {
            sample_id: (score - min_val) / (max_val - min_val)
            for sample_id, score in scores.items()
        }
    
    def _sample_from_scores(
        self, 
        scores: Dict[str, float], 
        n_samples: int,
        temperature: float = 1.0
    ) -> List[str]:
        """Sample from scores using temperature-controlled sampling."""
        if not scores or n_samples <= 0:
            return []
        
        sample_ids = list(scores.keys())
        score_values = np.array(list(scores.values()))
        
        # Apply temperature
        if temperature > 0:
            probs = np.exp(score_values / temperature)
            probs = probs / np.sum(probs)
        else:
            # Greedy selection
            top_indices = np.argsort(score_values)[-n_samples:]
            return [sample_ids[i] for i in top_indices]
        
        # Sample without replacement
        n_samples = min(n_samples, len(sample_ids))
        selected_indices = np.random.choice(
            len(sample_ids), size=n_samples, replace=False, p=probs
        )
        
        return [sample_ids[i] for i in selected_indices]


class UncertaintySamplingStrategy(BaseQueryStrategy):
    """Uncertainty-based sampling strategies."""
    
    async def select_samples(
        self,
        sample_pool: SamplePool,
        model_predictions: Dict[str, Any],
        labeled_samples: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QueryResult:
        """Select samples with highest prediction uncertainty."""
        
        self.logger.info("Selecting samples using uncertainty sampling")
        
        scores = {}
        
        for sample_id in sample_pool.samples.keys():
            if sample_id in model_predictions:
                prediction = model_predictions[sample_id]
                uncertainty = self._calculate_uncertainty(prediction)
                scores[sample_id] = uncertainty
        
        # Normalize scores
        normalized_scores = self._normalize_scores(scores)
        
        # Select top uncertain samples
        selected_samples = self._sample_from_scores(
            normalized_scores, 
            self.config.batch_size,
            self.config.temperature
        )
        
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            selected_samples=selected_samples,
            selection_scores=normalized_scores,
            query_time=datetime.utcnow(),
            selection_criteria={
                "uncertainty_threshold": self.config.uncertainty_threshold,
                "temperature": self.config.temperature,
            },
            uncertainty_score=np.mean([scores.get(sid, 0) for sid in selected_samples]),
            dataset_size=sample_pool.get_pool_size(),
        )
        
        return query_result
    
    def _calculate_uncertainty(self, prediction: Any) -> float:
        """Calculate uncertainty from model prediction."""
        if isinstance(prediction, dict):
            # Multi-class prediction with probabilities
            if "probabilities" in prediction:
                probs = np.array(prediction["probabilities"])
                if self.config.query_strategy == QueryStrategy.ENTROPY_SAMPLING:
                    return -np.sum(probs * np.log2(probs + 1e-8))
                elif self.config.query_strategy == QueryStrategy.LEAST_CONFIDENT:
                    return 1 - np.max(probs)
                elif self.config.query_strategy == QueryStrategy.MARGIN_SAMPLING:
                    sorted_probs = np.sort(probs)
                    return 1 - (sorted_probs[-1] - sorted_probs[-2])
            
            # Binary prediction with confidence
            if "confidence" in prediction:
                confidence = prediction["confidence"]
                return 1 - abs(confidence - 0.5) * 2  # Distance from decision boundary
        
        # Default: use prediction variance if available
        if hasattr(prediction, 'var') or isinstance(prediction, dict) and 'variance' in prediction:
            variance = prediction.var() if hasattr(prediction, 'var') else prediction['variance']
            return float(variance)
        
        # Fallback: random uncertainty
        return np.random.random()


class DiversitySamplingStrategy(BaseQueryStrategy):
    """Diversity-based sampling strategies."""
    
    async def select_samples(
        self,
        sample_pool: SamplePool,
        model_predictions: Dict[str, Any],
        labeled_samples: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QueryResult:
        """Select diverse samples to cover feature space."""
        
        self.logger.info("Selecting samples using diversity sampling")
        
        if self.config.query_strategy == QueryStrategy.CLUSTER_BASED:
            selected_samples = await self._cluster_based_selection(sample_pool)
        elif self.config.query_strategy == QueryStrategy.REPRESENTATIVE_SAMPLING:
            selected_samples = await self._representative_sampling(sample_pool, labeled_samples)
        else:
            selected_samples = await self._diversity_sampling(sample_pool, labeled_samples)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(sample_pool, selected_samples)
        
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=self.config.query_strategy,
            selected_samples=selected_samples,
            selection_scores={sid: 1.0 for sid in selected_samples},
            query_time=datetime.utcnow(),
            diversity_score=diversity_score,
            dataset_size=sample_pool.get_pool_size(),
        )
        
        return query_result
    
    async def _cluster_based_selection(self, sample_pool: SamplePool) -> List[str]:
        """Select samples from different clusters."""
        data_matrix = sample_pool.get_data_matrix()
        sample_ids = list(sample_pool.samples.keys())
        
        # Perform clustering
        n_clusters = min(self.config.batch_size, len(sample_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_matrix)
        
        # Select one representative from each cluster
        selected_samples = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select sample closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_data = data_matrix[cluster_indices]
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_samples.append(sample_ids[closest_idx])
        
        return selected_samples
    
    async def _representative_sampling(
        self, 
        sample_pool: SamplePool, 
        labeled_samples: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select representative samples covering the feature space."""
        data_matrix = sample_pool.get_data_matrix()
        sample_ids = list(sample_pool.samples.keys())
        
        if not labeled_samples:
            # Use k-center algorithm for initial selection
            return await self._k_center_selection(sample_pool)
        
        # Select samples far from already labeled ones
        labeled_data = []
        for label_info in labeled_samples.values():
            if "data" in label_info:
                labeled_data.append(label_info["data"])
        
        if not labeled_data:
            return await self._k_center_selection(sample_pool)
        
        labeled_matrix = np.array(labeled_data)
        
        # Calculate distances to labeled samples
        distances = cdist(data_matrix, labeled_matrix)
        min_distances = np.min(distances, axis=1)
        
        # Select samples with largest minimum distance to labeled set
        top_indices = np.argsort(min_distances)[-self.config.batch_size:]
        return [sample_ids[i] for i in top_indices]
    
    async def _diversity_sampling(
        self, 
        sample_pool: SamplePool, 
        labeled_samples: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select diverse samples using greedy algorithm."""
        sample_ids = list(sample_pool.samples.keys())
        data_matrix = sample_pool.get_data_matrix()
        
        selected_indices = []
        remaining_indices = list(range(len(sample_ids)))
        
        # Select first sample randomly or farthest from labeled samples
        if labeled_samples:
            labeled_data = []
            for label_info in labeled_samples.values():
                if "data" in label_info:
                    labeled_data.append(label_info["data"])
            
            if labeled_data:
                labeled_matrix = np.array(labeled_data)
                distances = cdist(data_matrix, labeled_matrix)
                min_distances = np.min(distances, axis=1)
                first_idx = np.argmax(min_distances)
            else:
                first_idx = np.random.choice(remaining_indices)
        else:
            first_idx = np.random.choice(remaining_indices)
        
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Greedily select remaining samples
        for _ in range(min(self.config.batch_size - 1, len(remaining_indices))):
            if not remaining_indices:
                break
            
            # Calculate minimum distance to already selected samples
            selected_data = data_matrix[selected_indices]
            min_distances = []
            
            for idx in remaining_indices:
                distances = np.linalg.norm(data_matrix[idx] - selected_data, axis=1)
                min_distances.append(np.min(distances))
            
            # Select sample with maximum minimum distance
            best_remaining_idx = np.argmax(min_distances)
            best_idx = remaining_indices[best_remaining_idx]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [sample_ids[i] for i in selected_indices]
    
    async def _k_center_selection(self, sample_pool: SamplePool) -> List[str]:
        """Select samples using k-center algorithm."""
        sample_ids = list(sample_pool.samples.keys())
        data_matrix = sample_pool.get_data_matrix()
        
        # Start with random sample
        selected_indices = [np.random.choice(len(sample_ids))]
        
        # Iteratively select farthest samples
        for _ in range(min(self.config.batch_size - 1, len(sample_ids) - 1)):
            selected_data = data_matrix[selected_indices]
            
            # Calculate minimum distances to selected set
            distances = cdist(data_matrix, selected_data)
            min_distances = np.min(distances, axis=1)
            
            # Exclude already selected samples
            min_distances[selected_indices] = -1
            
            # Select farthest sample
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(farthest_idx)
        
        return [sample_ids[i] for i in selected_indices]
    
    def _calculate_diversity_score(self, sample_pool: SamplePool, selected_samples: List[str]) -> float:
        """Calculate diversity score for selected samples."""
        if len(selected_samples) < 2:
            return 0.0
        
        selected_data = np.array([
            sample_pool.samples[sid] for sid in selected_samples
        ])
        
        # Calculate pairwise distances
        distances = pdist(selected_data)
        return float(np.mean(distances))


class HybridQueryStrategy(BaseQueryStrategy):
    """Hybrid strategies combining uncertainty and diversity."""
    
    async def select_samples(
        self,
        sample_pool: SamplePool,
        model_predictions: Dict[str, Any],
        labeled_samples: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QueryResult:
        """Select samples balancing uncertainty and diversity."""
        
        self.logger.info("Selecting samples using hybrid uncertainty-diversity strategy")
        
        # Get uncertainty scores
        uncertainty_scores = {}
        for sample_id in sample_pool.samples.keys():
            if sample_id in model_predictions:
                prediction = model_predictions[sample_id]
                uncertainty_scores[sample_id] = self._calculate_uncertainty(prediction)
        
        # Get diversity scores
        diversity_scores = await self._calculate_diversity_scores(sample_pool, labeled_samples)
        
        # Combine scores
        combined_scores = {}
        alpha = self.config.diversity_weight
        
        for sample_id in sample_pool.samples.keys():
            uncertainty = uncertainty_scores.get(sample_id, 0.0)
            diversity = diversity_scores.get(sample_id, 0.0)
            
            combined_scores[sample_id] = (1 - alpha) * uncertainty + alpha * diversity
        
        # Select samples
        normalized_scores = self._normalize_scores(combined_scores)
        selected_samples = self._sample_from_scores(
            normalized_scores,
            self.config.batch_size,
            self.config.temperature
        )
        
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.UNCERTAINTY_DIVERSITY,
            selected_samples=selected_samples,
            selection_scores=normalized_scores,
            query_time=datetime.utcnow(),
            selection_criteria={
                "diversity_weight": alpha,
                "uncertainty_weight": 1 - alpha,
            },
            uncertainty_score=np.mean([uncertainty_scores.get(sid, 0) for sid in selected_samples]),
            diversity_score=self._calculate_diversity_score(sample_pool, selected_samples),
            dataset_size=sample_pool.get_pool_size(),
        )
        
        return query_result
    
    async def _calculate_diversity_scores(
        self, 
        sample_pool: SamplePool, 
        labeled_samples: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate diversity scores for all samples."""
        sample_ids = list(sample_pool.samples.keys())
        data_matrix = sample_pool.get_data_matrix()
        
        diversity_scores = {}
        
        if labeled_samples:
            # Diversity from labeled samples
            labeled_data = []
            for label_info in labeled_samples.values():
                if "data" in label_info:
                    labeled_data.append(label_info["data"])
            
            if labeled_data:
                labeled_matrix = np.array(labeled_data)
                distances = cdist(data_matrix, labeled_matrix)
                min_distances = np.min(distances, axis=1)
                
                for i, sample_id in enumerate(sample_ids):
                    diversity_scores[sample_id] = min_distances[i]
            else:
                # No labeled data, use random scores
                for sample_id in sample_ids:
                    diversity_scores[sample_id] = np.random.random()
        else:
            # Diversity within unlabeled pool
            distances = pdist(data_matrix)
            avg_distance = np.mean(distances)
            
            for i, sample_id in enumerate(sample_ids):
                sample_distances = np.linalg.norm(
                    data_matrix - data_matrix[i], axis=1
                )
                diversity_scores[sample_id] = np.mean(sample_distances)
        
        return diversity_scores
    
    def _calculate_uncertainty(self, prediction: Any) -> float:
        """Calculate uncertainty from model prediction."""
        # Reuse uncertainty calculation from UncertaintySamplingStrategy
        uncertainty_strategy = UncertaintySamplingStrategy(self.config)
        return uncertainty_strategy._calculate_uncertainty(prediction)
    
    def _calculate_diversity_score(self, sample_pool: SamplePool, selected_samples: List[str]) -> float:
        """Calculate diversity score for selected samples."""
        # Reuse diversity calculation from DiversitySamplingStrategy
        diversity_strategy = DiversitySamplingStrategy(self.config)
        return diversity_strategy._calculate_diversity_score(sample_pool, selected_samples)


class CommitteeBasedStrategy(BaseQueryStrategy):
    """Query-by-committee strategies."""
    
    async def select_samples(
        self,
        sample_pool: SamplePool,
        model_predictions: Dict[str, Any],
        labeled_samples: Optional[Dict[str, Any]] = None,
        committee_predictions: Optional[Dict[str, List[Any]]] = None,
        **kwargs
    ) -> QueryResult:
        """Select samples with highest disagreement among committee members."""
        
        self.logger.info("Selecting samples using query-by-committee")
        
        if not committee_predictions:
            self.logger.warning("No committee predictions provided, falling back to uncertainty sampling")
            uncertainty_strategy = UncertaintySamplingStrategy(self.config)
            return await uncertainty_strategy.select_samples(
                sample_pool, model_predictions, labeled_samples, **kwargs
            )
        
        disagreement_scores = {}
        
        for sample_id in sample_pool.samples.keys():
            if sample_id in committee_predictions:
                predictions = committee_predictions[sample_id]
                disagreement = self._calculate_disagreement(predictions)
                disagreement_scores[sample_id] = disagreement
        
        # Select samples with highest disagreement
        normalized_scores = self._normalize_scores(disagreement_scores)
        selected_samples = self._sample_from_scores(
            normalized_scores,
            self.config.batch_size,
            self.config.temperature
        )
        
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.QUERY_BY_COMMITTEE,
            selected_samples=selected_samples,
            selection_scores=normalized_scores,
            query_time=datetime.utcnow(),
            selection_criteria={
                "committee_size": self.config.committee_size,
                "disagreement_metric": self.config.query_strategy.value,
            },
            uncertainty_score=np.mean([disagreement_scores.get(sid, 0) for sid in selected_samples]),
            dataset_size=sample_pool.get_pool_size(),
        )
        
        return query_result
    
    def _calculate_disagreement(self, predictions: List[Any]) -> float:
        """Calculate disagreement among committee predictions."""
        if len(predictions) < 2:
            return 0.0
        
        if self.config.query_strategy == QueryStrategy.VOTE_ENTROPY:
            return self._vote_entropy(predictions)
        elif self.config.query_strategy == QueryStrategy.KL_DIVERGENCE:
            return self._kl_divergence(predictions)
        else:
            return self._simple_disagreement(predictions)
    
    def _vote_entropy(self, predictions: List[Any]) -> float:
        """Calculate vote entropy among committee."""
        # Extract votes (assuming binary classification)
        votes = []
        for pred in predictions:
            if isinstance(pred, dict) and "prediction" in pred:
                votes.append(pred["prediction"])
            elif isinstance(pred, (int, float, bool)):
                votes.append(int(pred))
            else:
                votes.append(1 if pred > 0.5 else 0)
        
        # Calculate vote distribution
        unique_votes, counts = np.unique(votes, return_counts=True)
        probs = counts / len(votes)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        return entropy
    
    def _kl_divergence(self, predictions: List[Any]) -> float:
        """Calculate average KL divergence among committee predictions."""
        # Extract probability distributions
        prob_distributions = []
        for pred in predictions:
            if isinstance(pred, dict) and "probabilities" in pred:
                prob_distributions.append(np.array(pred["probabilities"]))
            elif isinstance(pred, (list, np.ndarray)):
                prob_distributions.append(np.array(pred))
            else:
                # Convert scalar to binary distribution
                p = float(pred) if isinstance(pred, (int, float)) else 0.5
                prob_distributions.append(np.array([1-p, p]))
        
        if len(prob_distributions) < 2:
            return 0.0
        
        # Calculate pairwise KL divergences
        kl_divergences = []
        for i in range(len(prob_distributions)):
            for j in range(i + 1, len(prob_distributions)):
                p = prob_distributions[i] + 1e-8  # Avoid log(0)
                q = prob_distributions[j] + 1e-8
                
                kl_div = np.sum(p * np.log(p / q))
                kl_divergences.append(kl_div)
        
        return np.mean(kl_divergences)
    
    def _simple_disagreement(self, predictions: List[Any]) -> float:
        """Calculate simple variance-based disagreement."""
        # Extract scalar predictions
        values = []
        for pred in predictions:
            if isinstance(pred, dict):
                if "confidence" in pred:
                    values.append(pred["confidence"])
                elif "prediction" in pred:
                    values.append(float(pred["prediction"]))
                else:
                    values.append(0.5)
            else:
                values.append(float(pred))
        
        return float(np.var(values))


class QueryStrategyFactory:
    """Factory for creating query strategies."""
    
    @staticmethod
    def create_strategy(config: ActiveLearningConfig) -> BaseQueryStrategy:
        """Create appropriate query strategy based on configuration."""
        
        uncertainty_strategies = {
            QueryStrategy.UNCERTAINTY_SAMPLING,
            QueryStrategy.LEAST_CONFIDENT,
            QueryStrategy.MARGIN_SAMPLING,
            QueryStrategy.ENTROPY_SAMPLING,
        }
        
        diversity_strategies = {
            QueryStrategy.DIVERSITY_SAMPLING,
            QueryStrategy.CLUSTER_BASED,
            QueryStrategy.REPRESENTATIVE_SAMPLING,
        }
        
        hybrid_strategies = {
            QueryStrategy.UNCERTAINTY_DIVERSITY,
            QueryStrategy.EXPECTED_MODEL_CHANGE,
            QueryStrategy.EXPECTED_ERROR_REDUCTION,
        }
        
        committee_strategies = {
            QueryStrategy.QUERY_BY_COMMITTEE,
            QueryStrategy.VOTE_ENTROPY,
            QueryStrategy.KL_DIVERGENCE,
        }
        
        if config.query_strategy in uncertainty_strategies:
            return UncertaintySamplingStrategy(config)
        elif config.query_strategy in diversity_strategies:
            return DiversitySamplingStrategy(config)
        elif config.query_strategy in hybrid_strategies:
            return HybridQueryStrategy(config)
        elif config.query_strategy in committee_strategies:
            return CommitteeBasedStrategy(config)
        else:
            # Default to uncertainty sampling
            return UncertaintySamplingStrategy(config)