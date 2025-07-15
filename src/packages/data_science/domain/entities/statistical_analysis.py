"""Statistical Analysis entity for data science operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class StatisticalAnalysisId:
    """Statistical Analysis identifier."""
    value: UUID = Field(default_factory=uuid4)


@dataclass(frozen=True)
class DatasetId:
    """Dataset identifier."""
    value: UUID = Field(default_factory=uuid4)


@dataclass(frozen=True)
class UserId:
    """User identifier."""
    value: UUID = Field(default_factory=uuid4)


class AnalysisType(BaseModel):
    """Analysis type value object."""
    name: str
    description: str
    requires_target: bool = False
    
    class Config:
        frozen = True


class StatisticalTest(BaseModel):
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_level: float = 0.95
    interpretation: str
    
    class Config:
        frozen = True


class StatisticalMetrics(BaseModel):
    """Statistical metrics collection."""
    descriptive_stats: Dict[str, float]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    distribution_params: Optional[Dict[str, Any]] = None
    outlier_scores: Optional[List[float]] = None
    
    class Config:
        frozen = True


class StatisticalAnalysis(BaseModel):
    """Statistical analysis aggregate root."""
    
    analysis_id: StatisticalAnalysisId
    dataset_id: DatasetId
    user_id: UserId
    analysis_type: AnalysisType
    status: str = "pending"
    
    # Analysis configuration
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    analysis_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    metrics: Optional[StatisticalMetrics] = None
    statistical_tests: List[StatisticalTest] = Field(default_factory=list)
    visualizations: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    
    # Metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    
    class Config:
        use_enum_values = True
        
    def start_analysis(self) -> None:
        """Start the statistical analysis."""
        self.status = "running"
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
    def complete_analysis(self, metrics: StatisticalMetrics, 
                         tests: List[StatisticalTest],
                         insights: List[str]) -> None:
        """Complete the analysis with results."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.metrics = metrics
        self.statistical_tests = tests
        self.insights = insights
        
        if self.started_at:
            self.execution_time_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()
        
        self.updated_at = datetime.utcnow()
        self.version += 1
        
    def fail_analysis(self, error_message: str) -> None:
        """Mark analysis as failed."""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.version += 1
        
    def add_statistical_test(self, test: StatisticalTest) -> None:
        """Add a statistical test result."""
        self.statistical_tests.append(test)
        self.updated_at = datetime.utcnow()
        self.version += 1
        
    def is_completed(self) -> bool:
        """Check if analysis is completed."""
        return self.status == "completed"
        
    def is_failed(self) -> bool:
        """Check if analysis failed."""
        return self.status == "failed"
        
    def is_running(self) -> bool:
        """Check if analysis is running."""
        return self.status == "running"