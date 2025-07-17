"""Value objects for matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
from enum import Enum
import numpy as np


class MatrixType(Enum):
    """Types of matrices."""
    GENERAL = "general"
    SYMMETRIC = "symmetric"
    ANTISYMMETRIC = "antisymmetric"
    ORTHOGONAL = "orthogonal"
    UNITARY = "unitary"
    DIAGONAL = "diagonal"
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"
    BIDIAGONAL = "bidiagonal"
    TRIDIAGONAL = "tridiagonal"
    SPARSE = "sparse"
    POSITIVE_DEFINITE = "positive_definite"
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"
    HERMITIAN = "hermitian"
    TOEPLITZ = "toeplitz"
    HANKEL = "hankel"
    CIRCULANT = "circulant"


class DecompositionType(Enum):
    """Types of matrix decompositions."""
    LU = "lu"
    QR = "qr"
    CHOLESKY = "cholesky"
    SVD = "svd"
    EIGENVALUE = "eigenvalue"
    SCHUR = "schur"
    HESSENBERG = "hessenberg"
    JORDAN = "jordan"
    POLAR = "polar"


@dataclass(frozen=True)
class MatrixId:
    """Unique identifier for matrices."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class MatrixProperties:
    """Properties of a matrix."""
    matrix_type: MatrixType
    is_square: bool
    is_singular: bool
    is_invertible: bool
    is_symmetric: bool
    is_positive_definite: bool
    is_orthogonal: bool
    is_unitary: bool
    is_hermitian: bool
    rank: int
    determinant: Optional[complex] = None
    trace: Optional[complex] = None
    condition_number: Optional[float] = None
    frobenius_norm: Optional[float] = None
    spectral_norm: Optional[float] = None
    nuclear_norm: Optional[float] = None
    eigenvalue_count: Optional[int] = None
    sparsity_ratio: float = 0.0
    
    def validate(self, shape: Tuple[int, int]) -> bool:
        """Validate matrix properties consistency."""
        rows, cols = shape
        
        # Square matrix checks
        if self.is_square != (rows == cols):
            return False
            
        # Non-square matrices cannot have certain properties
        if not self.is_square:
            if any([self.is_symmetric, self.is_positive_definite, 
                   self.is_orthogonal, self.is_unitary, self.is_hermitian]):
                return False
                
        # Singular matrices cannot be invertible
        if self.is_singular and self.is_invertible:
            return False
            
        # Rank validation
        if self.rank < 0 or self.rank > min(rows, cols):
            return False
            
        # Positive definite implies symmetric
        if self.is_positive_definite and not self.is_symmetric:
            return False
            
        return True


@dataclass(frozen=True)
class MatrixDecomposition:
    """Container for matrix decomposition results."""
    decomposition_type: DecompositionType
    factors: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.utcnow)
    computation_time: float = 0.0
    numerical_rank: Optional[int] = None
    condition_number: Optional[float] = None
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct original matrix from decomposition factors."""
        if self.decomposition_type == DecompositionType.LU:
            P = self.factors.get('P', np.eye(self.factors['L'].shape[0]))
            L = self.factors['L']
            U = self.factors['U']
            return P @ L @ U
            
        elif self.decomposition_type == DecompositionType.QR:
            Q = self.factors['Q']
            R = self.factors['R']
            return Q @ R
            
        elif self.decomposition_type == DecompositionType.CHOLESKY:
            L = self.factors['L']
            return L @ L.T.conj()
            
        elif self.decomposition_type == DecompositionType.SVD:
            U = self.factors['U']
            s = self.factors['s']
            Vh = self.factors['Vh']
            return U @ np.diag(s) @ Vh
            
        elif self.decomposition_type == DecompositionType.EIGENVALUE:
            eigenvalues = self.factors['eigenvalues']
            eigenvectors = self.factors['eigenvectors']
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T.conj()
            
        else:
            raise NotImplementedError(f"Reconstruction not implemented for {self.decomposition_type}")