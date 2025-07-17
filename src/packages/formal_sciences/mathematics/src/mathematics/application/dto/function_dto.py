"""Data Transfer Objects for mathematical functions."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class FunctionDTO:
    """Data Transfer Object for mathematical functions."""
    id: Optional[str] = None
    expression: str = ""
    variables: List[str] = None
    name: str = ""
    description: str = ""
    function_type: str = ""
    domain_lower: float = -float('inf')
    domain_upper: float = float('inf')
    include_lower: bool = True
    include_upper: bool = True
    excluded_points: List[float] = None
    is_continuous: bool = True
    is_monotonic: bool = False
    is_periodic: bool = False
    period: Optional[float] = None
    is_even: bool = False
    is_odd: bool = False
    author: str = ""
    version: str = "1.0.0"
    tags: List[str] = None
    computational_complexity: str = "O(1)"
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.excluded_points is None:
            self.excluded_points = []
        if self.tags is None:
            self.tags = []


@dataclass
class EvaluationRequestDTO:
    """Request for function evaluation."""
    function_id: str
    variable_values: Dict[str, float]
    cache_result: bool = True


@dataclass
class EvaluationResponseDTO:
    """Response from function evaluation."""
    result: float
    function_id: str
    variable_values: Dict[str, float]
    evaluation_time: float
    cached: bool = False
    error: Optional[str] = None