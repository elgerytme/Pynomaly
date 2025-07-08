"""Repository interface and implementations for model performance metrics."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pynomaly.domain.entities.model_performance import ModelPerformanceMetrics


class ModelPerformanceRepository(ABC):
    """Abstract repository for model performance metrics."""

    @abstractmethod
    def save_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Save model performance metrics.
        
        Args:
            metrics: The performance metrics to save
        """
        pass

    @abstractmethod
    def get_metrics(self, model_id: str, dataset_id: str) -> List[ModelPerformanceMetrics]:
        """Get all metrics for a specific model and dataset.
        
        Args:
            model_id: The model identifier
            dataset_id: The dataset identifier
            
        Returns:
            List of performance metrics
        """
        pass

    @abstractmethod
    def get_recent_metrics(
        self, 
        model_id: str, 
        dataset_id: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ModelPerformanceMetrics]:
        """Get recent metrics for a specific model and dataset.
        
        Args:
            model_id: The model identifier
            dataset_id: The dataset identifier
            since: Only return metrics after this timestamp
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent performance metrics
        """
        pass

    @abstractmethod
    def get_metrics_by_time_range(
        self, 
        model_id: str, 
        dataset_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[ModelPerformanceMetrics]:
        """Get metrics within a specific time range.
        
        Args:
            model_id: The model identifier
            dataset_id: The dataset identifier
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            List of performance metrics within the time range
        """
        pass

    @abstractmethod
    def delete_metrics(self, model_id: str, dataset_id: str) -> None:
        """Delete all metrics for a specific model and dataset.
        
        Args:
            model_id: The model identifier
            dataset_id: The dataset identifier
        """
        pass

    @abstractmethod
    def get_metrics_summary(self, model_id: str, dataset_id: str) -> Dict[str, Any]:
        """Get summary statistics for metrics.
        
        Args:
            model_id: The model identifier
            dataset_id: The dataset identifier
            
        Returns:
            Dictionary containing summary statistics
        """
        pass


class InMemoryModelPerformanceRepository(ModelPerformanceRepository):
    """In-memory implementation of ModelPerformanceRepository for MVP."""

    def __init__(self):
        self._metrics: Dict[str, List[ModelPerformanceMetrics]] = {}

    def _get_key(self, model_id: str, dataset_id: str) -> str:
        """Get storage key for model and dataset combination."""
        return f"{model_id}:{dataset_id}"

    def save_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Save model performance metrics."""
        key = self._get_key(metrics.model_id, metrics.dataset_id)
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(metrics)
        # Keep metrics sorted by timestamp
        self._metrics[key].sort(key=lambda m: m.timestamp)

    def get_metrics(self, model_id: str, dataset_id: str) -> List[ModelPerformanceMetrics]:
        """Get all metrics for a specific model and dataset."""
        key = self._get_key(model_id, dataset_id)
        return self._metrics.get(key, [])

    def get_recent_metrics(
        self, 
        model_id: str, 
        dataset_id: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ModelPerformanceMetrics]:
        """Get recent metrics for a specific model and dataset."""
        all_metrics = self.get_metrics(model_id, dataset_id)
        
        # Filter by time if specified
        if since:
            all_metrics = [m for m in all_metrics if m.timestamp >= since]
        
        # Sort by timestamp descending (most recent first)
        all_metrics.sort(key=lambda m: m.timestamp, reverse=True)
        
        # Apply limit if specified
        if limit:
            all_metrics = all_metrics[:limit]
        
        return all_metrics

    def get_metrics_by_time_range(
        self, 
        model_id: str, 
        dataset_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[ModelPerformanceMetrics]:
        """Get metrics within a specific time range."""
        all_metrics = self.get_metrics(model_id, dataset_id)
        return [
            m for m in all_metrics 
            if start_time <= m.timestamp <= end_time
        ]

    def delete_metrics(self, model_id: str, dataset_id: str) -> None:
        """Delete all metrics for a specific model and dataset."""
        key = self._get_key(model_id, dataset_id)
        if key in self._metrics:
            del self._metrics[key]

    def get_metrics_summary(self, model_id: str, dataset_id: str) -> Dict[str, Any]:
        """Get summary statistics for metrics."""
        metrics = self.get_metrics(model_id, dataset_id)
        
        if not metrics:
            return {
                "count": 0,
                "accuracy": {"min": None, "max": None, "avg": None},
                "precision": {"min": None, "max": None, "avg": None},
                "recall": {"min": None, "max": None, "avg": None},
                "f1": {"min": None, "max": None, "avg": None},
                "first_recorded": None,
                "last_recorded": None
            }
        
        accuracies = [m.accuracy for m in metrics]
        precisions = [m.precision for m in metrics]
        recalls = [m.recall for m in metrics]
        f1_scores = [m.f1 for m in metrics]
        
        return {
            "count": len(metrics),
            "accuracy": {
                "min": min(accuracies),
                "max": max(accuracies),
                "avg": sum(accuracies) / len(accuracies)
            },
            "precision": {
                "min": min(precisions),
                "max": max(precisions),
                "avg": sum(precisions) / len(precisions)
            },
            "recall": {
                "min": min(recalls),
                "max": max(recalls),
                "avg": sum(recalls) / len(recalls)
            },
            "f1": {
                "min": min(f1_scores),
                "max": max(f1_scores),
                "avg": sum(f1_scores) / len(f1_scores)
            },
            "first_recorded": min(m.timestamp for m in metrics),
            "last_recorded": max(m.timestamp for m in metrics)
        }


class SQLAlchemyModelPerformanceRepository(ModelPerformanceRepository):
    """SQLAlchemy implementation of ModelPerformanceRepository (placeholder)."""

    def __init__(self, session_factory):
        """Initialize with SQLAlchemy session factory.
        
        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory

    def save_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Save model performance metrics to database."""
        # TODO: Implement database persistence
        # Example structure:
        # with self.session_factory() as session:
        #     db_metrics = ModelPerformanceMetricsModel(
        #         accuracy=metrics.accuracy,
        #         precision=metrics.precision,
        #         recall=metrics.recall,
        #         f1=metrics.f1,
        #         timestamp=metrics.timestamp,
        #         model_id=metrics.model_id,
        #         dataset_id=metrics.dataset_id,
        #         metadata=metrics.metadata
        #     )
        #     session.add(db_metrics)
        #     session.commit()
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def get_metrics(self, model_id: str, dataset_id: str) -> List[ModelPerformanceMetrics]:
        """Get all metrics for a specific model and dataset from database."""
        # TODO: Implement database retrieval
        # Example structure:
        # with self.session_factory() as session:
        #     db_metrics = session.query(ModelPerformanceMetricsModel).filter_by(
        #         model_id=model_id,
        #         dataset_id=dataset_id
        #     ).order_by(ModelPerformanceMetricsModel.timestamp).all()
        #     return [self._to_domain_entity(m) for m in db_metrics]
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def get_recent_metrics(
        self, 
        model_id: str, 
        dataset_id: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ModelPerformanceMetrics]:
        """Get recent metrics for a specific model and dataset from database."""
        # TODO: Implement database retrieval with time filtering
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def get_metrics_by_time_range(
        self, 
        model_id: str, 
        dataset_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[ModelPerformanceMetrics]:
        """Get metrics within a specific time range from database."""
        # TODO: Implement database retrieval with time range filtering
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def delete_metrics(self, model_id: str, dataset_id: str) -> None:
        """Delete all metrics for a specific model and dataset from database."""
        # TODO: Implement database deletion
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def get_metrics_summary(self, model_id: str, dataset_id: str) -> Dict[str, Any]:
        """Get summary statistics for metrics from database."""
        # TODO: Implement database aggregation queries
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def _to_domain_entity(self, db_model) -> ModelPerformanceMetrics:
        """Convert database model to domain entity."""
        # TODO: Implement conversion from DB model to domain entity
        raise NotImplementedError("SQLAlchemy implementation not yet available")
