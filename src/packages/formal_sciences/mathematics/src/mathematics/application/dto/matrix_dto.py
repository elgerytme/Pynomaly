"""Data Transfer Objects for matrices."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


@dataclass
class MatrixDTO:
    """Data Transfer Object for matrices."""
    id: Optional[str] = None
    data: Optional[List[List[float]]] = None
    shape: Optional[Tuple[int, int]] = None
    matrix_type: str = "general"
    is_square: bool = False
    is_symmetric: bool = False
    is_invertible: bool = False
    rank: int = 0
    determinant: Optional[complex] = None
    trace: Optional[complex] = None
    condition_number: Optional[float] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is not None and self.shape is None:
            self.shape = (len(self.data), len(self.data[0]) if self.data else 0)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MatrixOperationDTO:
    """Data Transfer Object for matrix operations."""
    operation: str
    matrix_a_id: str
    matrix_b_id: Optional[str] = None
    scalar_value: Optional[float] = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class MatrixOperationResponseDTO:
    """Response from matrix operation."""
    result_matrix: MatrixDTO
    operation: str
    operand_ids: List[str]
    computation_time: float
    error: Optional[str] = None