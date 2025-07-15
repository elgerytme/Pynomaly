"""
API endpoints for visualization data - provides real anomaly detection data for charts
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pynomaly.application.dto.detection_dto import DetectionRequestDTO, DetectionResponseDTO
from pynomaly.application.services.detection_service import DetectionService
from pynomaly.application.services.visualization_service import VisualizationService
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.infrastructure.config.container import Container

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/visualization", tags=["visualization"])


# Pydantic models for visualization data
class TimeSeriesData(BaseModel):
    """Time series data for visualization"""
    timestamps: List[str]
    values: List[float]
    anomalies: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScatterPlotData(BaseModel):
    """Scatter plot data for 2D/3D visualization"""
    points: List[Dict[str, Any]]
    normal_points: List[Dict[str, float]]
    anomaly_points: List[Dict[str, float]]
    features: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CorrelationMatrixData(BaseModel):
    """Correlation matrix data for heatmap"""
    features: List[str]
    correlations: List[List[float]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureImportanceData(BaseModel):
    """Feature importance data for bar charts"""
    features: List[str]
    importance_scores: List[float]
    detector_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DetectorComparisonData(BaseModel):
    """Detector performance comparison data"""
    detectors: List[str]
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    auc: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnomalyDistributionData(BaseModel):
    """Anomaly score distribution data"""
    bins: List[float]
    normal_counts: List[int]
    anomaly_counts: List[int]
    threshold: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Dependency injection
async def get_detection_service() -> DetectionService:
    """Get detection service from container"""
    container = Container()
    return container.detection_service()


async def get_visualization_service() -> VisualizationService:
    """Get visualization service from container"""
    container = Container()
    return container.visualization_service()


@router.get("/timeseries/{detector_id}", response_model=TimeSeriesData)
async def get_timeseries_data(
    detector_id: UUID,
    start_time: Optional[datetime] = Query(None, description="Start time for data range"),
    end_time: Optional[datetime] = Query(None, description="End time for data range"),
    limit: int = Query(1000, description="Maximum number of points"),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Get time series data for visualization from real anomaly detection results
    """
    try:
        # Default time range to last 24 hours if not specified
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=24)

        # Get detection results from database
        detection_results = await detection_service.get_detection_history(
            detector_id=detector_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        if not detection_results:
            # Generate sample data if no real data available
            logger.warning(f"No detection results found for detector {detector_id}, generating sample data")
            return _generate_sample_timeseries_data(start_time, end_time, limit)

        # Transform detection results to visualization format
        timestamps = []
        values = []
        anomalies = []
        scores = []

        for result in detection_results:
            timestamps.append(result.timestamp.isoformat())
            values.append(float(result.aggregated_score))
            scores.append(float(result.anomaly_score))
            
            if result.is_anomaly:
                anomalies.append({
                    "timestamp": result.timestamp.isoformat(),
                    "value": float(result.aggregated_score),
                    "score": float(result.anomaly_score),
                    "id": str(result.id),
                    "confidence": float(result.confidence) if result.confidence else 0.0
                })

        return TimeSeriesData(
            timestamps=timestamps,
            values=values,
            anomalies=anomalies,
            scores=scores,
            metadata={
                "detector_id": str(detector_id),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "total_points": len(timestamps),
                "anomaly_count": len(anomalies),
                "anomaly_rate": len(anomalies) / len(timestamps) if timestamps else 0
            }
        )

    except Exception as e:
        logger.error(f"Error getting timeseries data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeseries data: {str(e)}")


@router.get("/scatter/{dataset_id}", response_model=ScatterPlotData)
async def get_scatter_data(
    dataset_id: UUID,
    feature_x: str = Query(..., description="X-axis feature name"),
    feature_y: str = Query(..., description="Y-axis feature name"),
    feature_z: Optional[str] = Query(None, description="Z-axis feature name for 3D"),
    limit: int = Query(1000, description="Maximum number of points"),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Get scatter plot data from real dataset with anomaly detection results
    """
    try:
        # Get dataset and detection results
        dataset_with_results = await detection_service.get_dataset_with_detections(
            dataset_id=dataset_id,
            limit=limit
        )

        if not dataset_with_results:
            logger.warning(f"No dataset found for {dataset_id}, generating sample data")
            return _generate_sample_scatter_data(feature_x, feature_y, feature_z, limit)

        dataset = dataset_with_results['dataset']
        detection_results = dataset_with_results['detections']

        # Extract feature data
        if not hasattr(dataset.data, 'columns') or feature_x not in dataset.data.columns:
            raise HTTPException(status_code=400, detail=f"Feature {feature_x} not found in dataset")
        if feature_y not in dataset.data.columns:
            raise HTTPException(status_code=400, detail=f"Feature {feature_y} not found in dataset")
        if feature_z and feature_z not in dataset.data.columns:
            raise HTTPException(status_code=400, detail=f"Feature {feature_z} not found in dataset")

        points = []
        normal_points = []
        anomaly_points = []

        # Create anomaly lookup
        anomaly_lookup = {result.sample_index: result for result in detection_results}

        for idx, row in dataset.data.iterrows():
            point_data = {
                "x": float(row[feature_x]),
                "y": float(row[feature_y]),
                "id": f"sample_{idx}",
                "index": idx
            }
            
            if feature_z:
                point_data["z"] = float(row[feature_z])

            # Check if this point is an anomaly
            if idx in anomaly_lookup:
                result = anomaly_lookup[idx]
                point_data.update({
                    "type": "anomaly",
                    "score": float(result.anomaly_score),
                    "confidence": float(result.confidence) if result.confidence else 0.0
                })
                anomaly_points.append(point_data)
            else:
                point_data.update({
                    "type": "normal",
                    "score": 0.0,
                    "confidence": 1.0
                })
                normal_points.append(point_data)

            points.append(point_data)

        features = [feature_x, feature_y]
        if feature_z:
            features.append(feature_z)

        return ScatterPlotData(
            points=points,
            normal_points=normal_points,
            anomaly_points=anomaly_points,
            features=features,
            metadata={
                "dataset_id": str(dataset_id),
                "total_points": len(points),
                "normal_count": len(normal_points),
                "anomaly_count": len(anomaly_points),
                "anomaly_rate": len(anomaly_points) / len(points) if points else 0,
                "feature_stats": {
                    feature_x: {
                        "min": float(dataset.data[feature_x].min()),
                        "max": float(dataset.data[feature_x].max()),
                        "mean": float(dataset.data[feature_x].mean())
                    },
                    feature_y: {
                        "min": float(dataset.data[feature_y].min()),
                        "max": float(dataset.data[feature_y].max()),
                        "mean": float(dataset.data[feature_y].mean())
                    }
                }
            }
        )

    except Exception as e:
        logger.error(f"Error getting scatter data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scatter data: {str(e)}")


@router.get("/correlation/{dataset_id}", response_model=CorrelationMatrixData)
async def get_correlation_data(
    dataset_id: UUID,
    features: Optional[List[str]] = Query(None, description="Specific features to include"),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Get correlation matrix data from real dataset
    """
    try:
        # Get dataset
        dataset = await detection_service.get_dataset(dataset_id)
        
        if not dataset:
            logger.warning(f"No dataset found for {dataset_id}, generating sample data")
            return _generate_sample_correlation_data(features or ["feature1", "feature2", "feature3"])

        # Select numeric columns only
        numeric_data = dataset.data.select_dtypes(include=[np.number])
        
        if features:
            # Filter to requested features
            available_features = [f for f in features if f in numeric_data.columns]
            if not available_features:
                raise HTTPException(status_code=400, detail="None of the requested features found in dataset")
            numeric_data = numeric_data[available_features]

        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        feature_names = list(correlation_matrix.columns)
        correlations = correlation_matrix.values.tolist()

        return CorrelationMatrixData(
            features=feature_names,
            correlations=correlations,
            metadata={
                "dataset_id": str(dataset_id),
                "feature_count": len(feature_names),
                "sample_count": len(numeric_data),
                "correlation_stats": {
                    "max_correlation": float(np.max(correlation_matrix.values)),
                    "min_correlation": float(np.min(correlation_matrix.values)),
                    "avg_correlation": float(np.mean(correlation_matrix.values))
                }
            }
        )

    except Exception as e:
        logger.error(f"Error getting correlation data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get correlation data: {str(e)}")


@router.get("/feature-importance/{detector_id}", response_model=FeatureImportanceData)
async def get_feature_importance_data(
    detector_id: UUID,
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Get feature importance data from trained detector
    """
    try:
        # Get detector with feature importance
        detector_info = await detection_service.get_detector_feature_importance(detector_id)
        
        if not detector_info:
            logger.warning(f"No detector found for {detector_id}, generating sample data")
            return _generate_sample_feature_importance_data()

        detector = detector_info['detector']
        feature_importance = detector_info['feature_importance']

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_features]
        importance_scores = [float(item[1]) for item in sorted_features]

        return FeatureImportanceData(
            features=features,
            importance_scores=importance_scores,
            detector_name=detector.name,
            metadata={
                "detector_id": str(detector_id),
                "algorithm": detector.algorithm_name,
                "feature_count": len(features),
                "max_importance": max(importance_scores) if importance_scores else 0,
                "min_importance": min(importance_scores) if importance_scores else 0
            }
        )

    except Exception as e:
        logger.error(f"Error getting feature importance data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance data: {str(e)}")


@router.get("/detector-comparison", response_model=DetectorComparisonData)
async def get_detector_comparison_data(
    dataset_id: Optional[UUID] = Query(None, description="Dataset to compare on"),
    detector_ids: Optional[List[UUID]] = Query(None, description="Specific detectors to compare"),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Get detector performance comparison data
    """
    try:
        # Get detector comparison metrics
        comparison_data = await detection_service.compare_detectors(
            dataset_id=dataset_id,
            detector_ids=detector_ids
        )
        
        if not comparison_data:
            logger.warning("No detector comparison data available, generating sample data")
            return _generate_sample_detector_comparison_data()

        detectors = list(comparison_data.keys())
        precision = [float(comparison_data[d]['precision']) for d in detectors]
        recall = [float(comparison_data[d]['recall']) for d in detectors]
        f1_score = [float(comparison_data[d]['f1_score']) for d in detectors]
        auc = [float(comparison_data[d]['auc']) for d in detectors]

        return DetectorComparisonData(
            detectors=detectors,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc=auc,
            metadata={
                "dataset_id": str(dataset_id) if dataset_id else None,
                "detector_count": len(detectors),
                "best_detector": {
                    "by_f1": detectors[np.argmax(f1_score)] if f1_score else None,
                    "by_auc": detectors[np.argmax(auc)] if auc else None,
                    "by_precision": detectors[np.argmax(precision)] if precision else None
                }
            }
        )

    except Exception as e:
        logger.error(f"Error getting detector comparison data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detector comparison data: {str(e)}")


@router.get("/anomaly-distribution/{detector_id}", response_model=AnomalyDistributionData)
async def get_anomaly_distribution_data(
    detector_id: UUID,
    bins: int = Query(50, description="Number of histogram bins"),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Get anomaly score distribution data for histogram visualization
    """
    try:
        # Get detection results with scores
        detection_data = await detection_service.get_detector_score_distribution(
            detector_id=detector_id
        )
        
        if not detection_data:
            logger.warning(f"No detection data found for detector {detector_id}, generating sample data")
            return _generate_sample_distribution_data(bins)

        normal_scores = detection_data['normal_scores']
        anomaly_scores = detection_data['anomaly_scores']
        threshold = detection_data['threshold']

        # Create histogram bins
        all_scores = normal_scores + anomaly_scores
        min_score, max_score = min(all_scores), max(all_scores)
        bin_edges = np.linspace(min_score, max_score, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate histograms
        normal_counts, _ = np.histogram(normal_scores, bins=bin_edges)
        anomaly_counts, _ = np.histogram(anomaly_scores, bins=bin_edges)

        return AnomalyDistributionData(
            bins=bin_centers.tolist(),
            normal_counts=normal_counts.tolist(),
            anomaly_counts=anomaly_counts.tolist(),
            threshold=float(threshold),
            metadata={
                "detector_id": str(detector_id),
                "total_normal": len(normal_scores),
                "total_anomalies": len(anomaly_scores),
                "score_range": {
                    "min": float(min_score),
                    "max": float(max_score)
                }
            }
        )

    except Exception as e:
        logger.error(f"Error getting anomaly distribution data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get anomaly distribution data: {str(e)}")


# Sample data generation functions for fallback
def _generate_sample_timeseries_data(start_time: datetime, end_time: datetime, limit: int) -> TimeSeriesData:
    """Generate sample time series data when real data is unavailable"""
    time_delta = (end_time - start_time) / limit
    timestamps = []
    values = []
    scores = []
    anomalies = []

    for i in range(limit):
        current_time = start_time + (time_delta * i)
        timestamp_str = current_time.isoformat()
        
        # Generate synthetic time series with trend and noise
        trend_value = 50 + 10 * np.sin(i * 0.1) + np.random.normal(0, 5)
        anomaly_score = abs(np.random.normal(0.2, 0.15))
        
        timestamps.append(timestamp_str)
        values.append(float(trend_value))
        scores.append(float(anomaly_score))
        
        # Randomly mark some points as anomalies
        if np.random.random() < 0.05:  # 5% anomaly rate
            anomalies.append({
                "timestamp": timestamp_str,
                "value": float(trend_value),
                "score": float(anomaly_score),
                "id": f"sample_{i}",
                "confidence": float(np.random.uniform(0.7, 0.95))
            })

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        anomalies=anomalies,
        scores=scores,
        metadata={
            "type": "sample_data",
            "total_points": len(timestamps),
            "anomaly_count": len(anomalies),
            "anomaly_rate": len(anomalies) / len(timestamps)
        }
    )


def _generate_sample_scatter_data(feature_x: str, feature_y: str, feature_z: Optional[str], limit: int) -> ScatterPlotData:
    """Generate sample scatter plot data"""
    points = []
    normal_points = []
    anomaly_points = []

    for i in range(limit):
        # Generate normal cluster
        if np.random.random() < 0.9:  # 90% normal points
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 1)
            point_type = "normal"
        else:  # 10% anomalies
            x = np.random.normal(3, 0.5)
            y = np.random.normal(3, 0.5)
            point_type = "anomaly"

        point_data = {
            "x": float(x),
            "y": float(y),
            "id": f"sample_{i}",
            "index": i,
            "type": point_type,
            "score": float(np.random.uniform(0.1, 0.9)) if point_type == "anomaly" else float(np.random.uniform(0.0, 0.3)),
            "confidence": float(np.random.uniform(0.7, 0.95))
        }

        if feature_z:
            point_data["z"] = float(np.random.normal(0, 1))

        points.append(point_data)
        
        if point_type == "normal":
            normal_points.append(point_data)
        else:
            anomaly_points.append(point_data)

    features = [feature_x, feature_y]
    if feature_z:
        features.append(feature_z)

    return ScatterPlotData(
        points=points,
        normal_points=normal_points,
        anomaly_points=anomaly_points,
        features=features,
        metadata={
            "type": "sample_data",
            "total_points": len(points),
            "normal_count": len(normal_points),
            "anomaly_count": len(anomaly_points),
            "anomaly_rate": len(anomaly_points) / len(points)
        }
    )


def _generate_sample_correlation_data(features: List[str]) -> CorrelationMatrixData:
    """Generate sample correlation matrix data"""
    n_features = len(features)
    # Generate random correlation matrix
    random_matrix = np.random.randn(n_features, n_features)
    correlation_matrix = np.corrcoef(random_matrix)
    
    return CorrelationMatrixData(
        features=features,
        correlations=correlation_matrix.tolist(),
        metadata={
            "type": "sample_data",
            "feature_count": n_features
        }
    )


def _generate_sample_feature_importance_data() -> FeatureImportanceData:
    """Generate sample feature importance data"""
    features = [f"Feature_{i+1}" for i in range(10)]
    importance_scores = sorted([float(np.random.exponential(0.3)) for _ in range(10)], reverse=True)
    
    return FeatureImportanceData(
        features=features,
        importance_scores=importance_scores,
        detector_name="SampleDetector",
        metadata={
            "type": "sample_data",
            "feature_count": len(features)
        }
    )


def _generate_sample_detector_comparison_data() -> DetectorComparisonData:
    """Generate sample detector comparison data"""
    detectors = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "EllipticEnvelope"]
    precision = [float(np.random.uniform(0.7, 0.95)) for _ in detectors]
    recall = [float(np.random.uniform(0.6, 0.9)) for _ in detectors]
    f1_score = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
    auc = [float(np.random.uniform(0.75, 0.95)) for _ in detectors]
    
    return DetectorComparisonData(
        detectors=detectors,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc=auc,
        metadata={"type": "sample_data"}
    )


def _generate_sample_distribution_data(bins: int) -> AnomalyDistributionData:
    """Generate sample anomaly score distribution data"""
    # Generate sample scores
    normal_scores = np.random.beta(2, 5, 1000).tolist()  # Lower scores for normal
    anomaly_scores = np.random.beta(5, 2, 100).tolist()  # Higher scores for anomalies
    
    all_scores = normal_scores + anomaly_scores
    min_score, max_score = min(all_scores), max(all_scores)
    bin_edges = np.linspace(min_score, max_score, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    normal_counts, _ = np.histogram(normal_scores, bins=bin_edges)
    anomaly_counts, _ = np.histogram(anomaly_scores, bins=bin_edges)
    
    return AnomalyDistributionData(
        bins=bin_centers.tolist(),
        normal_counts=normal_counts.tolist(),
        anomaly_counts=anomaly_counts.tolist(),
        threshold=0.5,
        metadata={
            "type": "sample_data",
            "total_normal": len(normal_scores),
            "total_anomalies": len(anomaly_scores)
        }
    )