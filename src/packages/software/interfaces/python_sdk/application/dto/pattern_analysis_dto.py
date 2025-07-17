"""
Pattern Analysis Data Transfer Objects

DTOs for transferring pattern analysis-related data between layers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class PatternAnalysisRequestDTO:
    """
    Data Transfer Object for pattern analysis requests.
    
    Used to transfer pattern analysis request data between the presentation
    layer (CLI/API) and the application layer.
    """
    
    data: List[float]
    algorithm_config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """
        Basic validation of the DTO structure.
        
        Returns:
            bool: True if the DTO is structurally valid.
        """
        if not isinstance(self.data, list):
            return False
            
        if not self.data:
            return False
            
        if not isinstance(self.algorithm_config, dict):
            return False
            
        if "algorithm_type" not in self.algorithm_config:
            return False
            
        return True


@dataclass
class PatternAnalysisResponseDTO:
    """
    Data Transfer Object for pattern analysis responses.
    
    Used to transfer pattern analysis response data from the application
    layer back to the presentation layer.
    """
    
    request_id: str
    status: str
    message: str
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    patterns: Optional[List[int]] = None
    scores: Optional[List[float]] = None
    algorithm_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DTO to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the response.
        """
        result = {
            "request_id": self.request_id,
            "status": self.status,
            "message": self.message
        }
        
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
            
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
            
        if self.patterns is not None:
            result["patterns"] = self.patterns
            
        if self.scores is not None:
            result["scores"] = self.scores
            
        if self.algorithm_used:
            result["algorithm_used"] = self.algorithm_used
            
        if self.processing_time_ms is not None:
            result["processing_time_ms"] = self.processing_time_ms
            
        if self.error_details:
            result["error_details"] = self.error_details
            
        return result


@dataclass
class AlgorithmRecommendationDTO:
    """
    Data Transfer Object for algorithm recommendations.
    
    Used to transfer algorithm recommendation data between layers.
    """
    
    algorithm_type: str
    confidence_score: float
    reason: str
    estimated_processing_time: Optional[int] = None
    memory_requirements: Optional[str] = None
    performance_characteristics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DTO to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the recommendation.
        """
        result = {
            "algorithm_type": self.algorithm_type,
            "confidence_score": self.confidence_score,
            "reason": self.reason
        }
        
        if self.estimated_processing_time is not None:
            result["estimated_processing_time"] = self.estimated_processing_time
            
        if self.memory_requirements:
            result["memory_requirements"] = self.memory_requirements
            
        if self.performance_characteristics:
            result["performance_characteristics"] = self.performance_characteristics
            
        return result


@dataclass
class BatchPatternAnalysisRequestDTO:
    """
    Data Transfer Object for batch pattern analysis requests.
    
    Used to handle multiple pattern analysis requests in a single operation.
    """
    
    requests: List[PatternAnalysisRequestDTO]
    batch_metadata: Optional[Dict[str, Any]] = None
    priority: str = "normal"
    
    def validate(self) -> bool:
        """
        Validate the batch request structure.
        
        Returns:
            bool: True if the batch request is valid.
        """
        if not isinstance(self.requests, list):
            return False
            
        if not self.requests:
            return False
            
        if len(self.requests) > 100:  # Limit batch size
            return False
            
        return all(request.validate() for request in self.requests)


@dataclass
class BatchPatternAnalysisResponseDTO:
    """
    Data Transfer Object for batch pattern analysis responses.
    
    Used to return results from batch pattern analysis operations.
    """
    
    batch_id: str
    total_requests: int
    completed_requests: int
    failed_requests: int
    status: str
    responses: List[PatternAnalysisResponseDTO]
    batch_processing_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the batch response to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the batch response.
        """
        return {
            "batch_id": self.batch_id,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "status": self.status,
            "responses": [response.to_dict() for response in self.responses],
            "batch_processing_time_ms": self.batch_processing_time_ms
        }