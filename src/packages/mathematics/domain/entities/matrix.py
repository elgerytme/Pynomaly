"""Matrix domain entity for linear algebra operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4
from enum import Enum
import numpy as np
from scipy import linalg


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
    sparsity_ratio: float = 0.0  # Ratio of zero elements
    
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


@dataclass(frozen=True)
class Matrix:
    """
    Domain entity representing a mathematical matrix.
    
    A matrix is a rectangular array of numbers, symbols, or expressions,
    arranged in rows and columns, used in linear algebra operations.
    """
    matrix_id: MatrixId
    data: np.ndarray
    properties: MatrixProperties
    decompositions: Dict[DecompositionType, MatrixDecomposition] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate matrix after initialization."""
        if self.data.ndim != 2:
            raise ValueError("Matrix data must be 2-dimensional")
            
        if not self.properties.validate(self.data.shape):
            raise ValueError("Matrix properties are inconsistent with data")
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix shape (rows, columns)."""
        return self.data.shape
    
    @property
    def rows(self) -> int:
        """Get number of rows."""
        return self.data.shape[0]
    
    @property
    def cols(self) -> int:
        """Get number of columns."""
        return self.data.shape[1]
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.data.size
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type of matrix elements."""
        return self.data.dtype
    
    @property
    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.rows == self.cols
    
    @property
    def is_empty(self) -> bool:
        """Check if matrix is empty."""
        return self.size == 0
    
    def add(self, other: Matrix) -> Matrix:
        """
        Add another matrix to this matrix.
        
        Args:
            other: Matrix to add
            
        Returns:
            New Matrix representing the sum
            
        Raises:
            ValueError: If matrices have incompatible shapes
        """
        if self.shape != other.shape:
            raise ValueError(f"Cannot add matrices with shapes {self.shape} and {other.shape}")
        
        result_data = self.data + other.data
        
        # Determine result properties (simplified)
        result_properties = MatrixProperties(
            matrix_type=MatrixType.GENERAL,
            is_square=self.is_square,
            is_singular=False,  # Would need recomputation
            is_invertible=False,  # Would need recomputation
            is_symmetric=self.properties.is_symmetric and other.properties.is_symmetric,
            is_positive_definite=False,  # Would need recomputation
            is_orthogonal=False,  # Would need recomputation
            is_unitary=False,  # Would need recomputation
            is_hermitian=self.properties.is_hermitian and other.properties.is_hermitian,
            rank=0,  # Would need recomputation
        )
        
        return Matrix(
            matrix_id=MatrixId(),
            data=result_data,
            properties=result_properties,
            metadata={
                "operation": "addition",
                "operand_ids": [str(self.matrix_id), str(other.matrix_id)]
            }
        )
    
    def subtract(self, other: Matrix) -> Matrix:
        """
        Subtract another matrix from this matrix.
        
        Args:
            other: Matrix to subtract
            
        Returns:
            New Matrix representing the difference
        """
        if self.shape != other.shape:
            raise ValueError(f"Cannot subtract matrices with shapes {self.shape} and {other.shape}")
        
        result_data = self.data - other.data
        
        # Determine result properties (simplified)
        result_properties = MatrixProperties(
            matrix_type=MatrixType.GENERAL,
            is_square=self.is_square,
            is_singular=False,  # Would need recomputation
            is_invertible=False,  # Would need recomputation
            is_symmetric=self.properties.is_symmetric and other.properties.is_symmetric,
            is_positive_definite=False,  # Would need recomputation
            is_orthogonal=False,  # Would need recomputation
            is_unitary=False,  # Would need recomputation
            is_hermitian=self.properties.is_hermitian and other.properties.is_hermitian,
            rank=0,  # Would need recomputation
        )
        
        return Matrix(
            matrix_id=MatrixId(),
            data=result_data,
            properties=result_properties,
            metadata={
                "operation": "subtraction",
                "operand_ids": [str(self.matrix_id), str(other.matrix_id)]
            }
        )
    
    def multiply(self, other: Union[Matrix, float, complex]) -> Matrix:
        """
        Multiply this matrix by another matrix or scalar.
        
        Args:
            other: Matrix or scalar to multiply by
            
        Returns:
            New Matrix representing the product
        """
        if isinstance(other, (int, float, complex)):
            # Scalar multiplication
            result_data = self.data * other
            
            # Properties are preserved for scalar multiplication
            result_properties = self.properties
            
            return Matrix(
                matrix_id=MatrixId(),
                data=result_data,
                properties=result_properties,
                metadata={
                    "operation": "scalar_multiplication",
                    "scalar": other,
                    "operand_id": str(self.matrix_id)
                }
            )
        
        elif isinstance(other, Matrix):
            # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError(f"Cannot multiply matrices with shapes {self.shape} and {other.shape}")
            
            result_data = self.data @ other.data
            
            # Determine result properties
            result_properties = MatrixProperties(
                matrix_type=MatrixType.GENERAL,
                is_square=self.rows == other.cols,
                is_singular=False,  # Would need recomputation
                is_invertible=False,  # Would need recomputation
                is_symmetric=False,  # Would need recomputation
                is_positive_definite=False,  # Would need recomputation
                is_orthogonal=(self.properties.is_orthogonal and other.properties.is_orthogonal 
                              and self.is_square and other.is_square),
                is_unitary=(self.properties.is_unitary and other.properties.is_unitary 
                           and self.is_square and other.is_square),
                is_hermitian=False,  # Would need recomputation
                rank=0,  # Would need recomputation
            )
            
            return Matrix(
                matrix_id=MatrixId(),
                data=result_data,
                properties=result_properties,
                metadata={
                    "operation": "matrix_multiplication",
                    "operand_ids": [str(self.matrix_id), str(other.matrix_id)]
                }
            )
        
        else:
            raise TypeError(f"Cannot multiply matrix by {type(other)}")
    
    def transpose(self) -> Matrix:
        """
        Compute the transpose of the matrix.
        
        Returns:
            New Matrix representing the transpose
        """
        result_data = self.data.T
        
        # Transpose properties
        result_properties = MatrixProperties(
            matrix_type=self.properties.matrix_type,
            is_square=self.is_square,
            is_singular=self.properties.is_singular,
            is_invertible=self.properties.is_invertible,
            is_symmetric=self.properties.is_symmetric,
            is_positive_definite=self.properties.is_positive_definite,
            is_orthogonal=self.properties.is_orthogonal,
            is_unitary=self.properties.is_unitary,
            is_hermitian=self.properties.is_hermitian,
            rank=self.properties.rank,
        )
        
        return Matrix(
            matrix_id=MatrixId(),
            data=result_data,
            properties=result_properties,
            metadata={
                "operation": "transpose",
                "operand_id": str(self.matrix_id)
            }
        )
    
    def conjugate_transpose(self) -> Matrix:
        """
        Compute the conjugate transpose (Hermitian transpose) of the matrix.
        
        Returns:
            New Matrix representing the conjugate transpose
        """
        result_data = self.data.T.conj()
        
        # Conjugate transpose properties
        result_properties = MatrixProperties(
            matrix_type=self.properties.matrix_type,
            is_square=self.is_square,
            is_singular=self.properties.is_singular,
            is_invertible=self.properties.is_invertible,
            is_symmetric=self.properties.is_hermitian,  # A† symmetric iff A Hermitian
            is_positive_definite=self.properties.is_positive_definite,
            is_orthogonal=False,  # Complex case
            is_unitary=self.properties.is_unitary,
            is_hermitian=self.properties.is_symmetric,  # A† Hermitian iff A symmetric
            rank=self.properties.rank,
        )
        
        return Matrix(
            matrix_id=MatrixId(),
            data=result_data,
            properties=result_properties,
            metadata={
                "operation": "conjugate_transpose",
                "operand_id": str(self.matrix_id)
            }
        )
    
    def inverse(self) -> Matrix:
        """
        Compute the inverse of the matrix.
        
        Returns:
            New Matrix representing the inverse
            
        Raises:
            ValueError: If matrix is not invertible
        """
        if not self.is_square:
            raise ValueError("Only square matrices can be inverted")
        
        if self.properties.is_singular:
            raise ValueError("Singular matrices cannot be inverted")
        
        try:
            result_data = linalg.inv(self.data)
        except linalg.LinAlgError as e:
            raise ValueError(f"Matrix inversion failed: {e}")
        
        # Inverse properties
        result_properties = MatrixProperties(
            matrix_type=MatrixType.GENERAL,
            is_square=True,
            is_singular=False,
            is_invertible=True,
            is_symmetric=self.properties.is_symmetric,
            is_positive_definite=self.properties.is_positive_definite,
            is_orthogonal=self.properties.is_orthogonal,
            is_unitary=self.properties.is_unitary,
            is_hermitian=self.properties.is_hermitian,
            rank=self.properties.rank,
        )
        
        return Matrix(
            matrix_id=MatrixId(),
            data=result_data,
            properties=result_properties,
            metadata={
                "operation": "inverse",
                "operand_id": str(self.matrix_id)
            }
        )
    
    def determinant(self) -> complex:
        """
        Compute the determinant of the matrix.
        
        Returns:
            Determinant value
            
        Raises:
            ValueError: If matrix is not square
        """
        if not self.is_square:
            raise ValueError("Determinant is only defined for square matrices")
        
        return linalg.det(self.data)
    
    def trace(self) -> complex:
        """
        Compute the trace (sum of diagonal elements) of the matrix.
        
        Returns:
            Trace value
        """
        return np.trace(self.data)
    
    def rank(self) -> int:
        """
        Compute the rank of the matrix.
        
        Returns:
            Matrix rank
        """
        return np.linalg.matrix_rank(self.data)
    
    def condition_number(self, p: Union[None, int, str] = None) -> float:
        """
        Compute the condition number of the matrix.
        
        Args:
            p: Order of the norm (None, 1, -1, 2, -2, inf, -inf, 'fro')
            
        Returns:
            Condition number
        """
        return np.linalg.cond(self.data, p)
    
    def norm(self, ord: Union[None, int, str] = None) -> float:
        """
        Compute a matrix norm.
        
        Args:
            ord: Order of the norm
            
        Returns:
            Norm value
        """
        return np.linalg.norm(self.data, ord)
    
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors)
            
        Raises:
            ValueError: If matrix is not square
        """
        if not self.is_square:
            raise ValueError("Eigenvalues are only defined for square matrices")
        
        eigenvals, eigenvecs = linalg.eig(self.data)
        return eigenvals, eigenvecs
    
    def lu_decomposition(self) -> MatrixDecomposition:
        """
        Compute LU decomposition with partial pivoting.
        
        Returns:
            MatrixDecomposition containing L, U, and P matrices
        """
        start_time = datetime.utcnow()
        P, L, U = linalg.lu(self.data)
        end_time = datetime.utcnow()
        
        computation_time = (end_time - start_time).total_seconds()
        
        decomposition = MatrixDecomposition(
            decomposition_type=DecompositionType.LU,
            factors={'P': P, 'L': L, 'U': U},
            computed_at=start_time,
            computation_time=computation_time,
            metadata={
                "pivoting": "partial",
                "numerical_rank": np.linalg.matrix_rank(U)
            }
        )
        
        # Cache the decomposition
        new_decompositions = dict(self.decompositions)
        new_decompositions[DecompositionType.LU] = decomposition
        
        return decomposition
    
    def qr_decomposition(self, mode: str = 'reduced') -> MatrixDecomposition:
        """
        Compute QR decomposition.
        
        Args:
            mode: 'reduced' or 'complete'
            
        Returns:
            MatrixDecomposition containing Q and R matrices
        """
        start_time = datetime.utcnow()
        Q, R = linalg.qr(self.data, mode=mode)
        end_time = datetime.utcnow()
        
        computation_time = (end_time - start_time).total_seconds()
        
        decomposition = MatrixDecomposition(
            decomposition_type=DecompositionType.QR,
            factors={'Q': Q, 'R': R},
            computed_at=start_time,
            computation_time=computation_time,
            metadata={
                "mode": mode,
                "numerical_rank": np.linalg.matrix_rank(R)
            }
        )
        
        return decomposition
    
    def svd_decomposition(self, full_matrices: bool = True) -> MatrixDecomposition:
        """
        Compute Singular Value Decomposition.
        
        Args:
            full_matrices: Whether to compute full-sized U and Vh matrices
            
        Returns:
            MatrixDecomposition containing U, s, and Vh matrices
        """
        start_time = datetime.utcnow()
        U, s, Vh = linalg.svd(self.data, full_matrices=full_matrices)
        end_time = datetime.utcnow()
        
        computation_time = (end_time - start_time).total_seconds()
        
        decomposition = MatrixDecomposition(
            decomposition_type=DecompositionType.SVD,
            factors={'U': U, 's': s, 'Vh': Vh},
            computed_at=start_time,
            computation_time=computation_time,
            numerical_rank=np.sum(s > np.finfo(float).eps * max(self.shape) * s[0]),
            condition_number=s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf,
            metadata={
                "full_matrices": full_matrices,
                "singular_values": s.tolist()
            }
        )
        
        return decomposition
    
    def get_decomposition(self, decomposition_type: DecompositionType) -> Optional[MatrixDecomposition]:
        """Get cached decomposition if available."""
        return self.decompositions.get(decomposition_type)
    
    def cache_decomposition(self, decomposition: MatrixDecomposition) -> Matrix:
        """Cache a decomposition result."""
        new_decompositions = dict(self.decompositions)
        new_decompositions[decomposition.decomposition_type] = decomposition
        
        return dataclass.replace(
            self,
            decompositions=new_decompositions,
            last_modified=datetime.utcnow()
        )
    
    def is_equal(self, other: Matrix, tolerance: float = 1e-10) -> bool:
        """
        Check if this matrix is equal to another matrix within tolerance.
        
        Args:
            other: Matrix to compare with
            tolerance: Numerical tolerance
            
        Returns:
            True if matrices are equal within tolerance
        """
        if self.shape != other.shape:
            return False
        
        return np.allclose(self.data, other.data, atol=tolerance, rtol=tolerance)
    
    def to_array(self) -> np.ndarray:
        """Get matrix data as numpy array."""
        return self.data.copy()
    
    def update_metadata(self, **kwargs) -> Matrix:
        """Update matrix metadata."""
        new_metadata = dict(self.metadata)
        new_metadata.update(kwargs)
        
        return dataclass.replace(
            self,
            metadata=new_metadata,
            last_modified=datetime.utcnow()
        )
    
    @classmethod
    def from_array(cls, data: np.ndarray, matrix_type: MatrixType = MatrixType.GENERAL) -> Matrix:
        """
        Create a Matrix from a numpy array.
        
        Args:
            data: Numpy array containing matrix data
            matrix_type: Type of matrix
            
        Returns:
            New Matrix instance
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
        
        # Analyze matrix properties
        rows, cols = data.shape
        is_square = rows == cols
        
        # Basic property detection
        is_symmetric = is_square and np.allclose(data, data.T)
        is_hermitian = is_square and np.allclose(data, data.T.conj())
        
        # Compute rank and determine if singular
        rank = np.linalg.matrix_rank(data)
        is_singular = is_square and rank < rows
        is_invertible = is_square and not is_singular
        
        # Sparsity ratio
        sparsity_ratio = np.count_nonzero(data == 0) / data.size
        
        properties = MatrixProperties(
            matrix_type=matrix_type,
            is_square=is_square,
            is_singular=is_singular,
            is_invertible=is_invertible,
            is_symmetric=is_symmetric,
            is_positive_definite=False,  # Would need eigenvalue analysis
            is_orthogonal=False,  # Would need detailed analysis
            is_unitary=False,  # Would need detailed analysis
            is_hermitian=is_hermitian,
            rank=rank,
            sparsity_ratio=sparsity_ratio,
        )
        
        return cls(
            matrix_id=MatrixId(),
            data=data,
            properties=properties
        )
    
    @classmethod
    def identity(cls, size: int, dtype: np.dtype = np.float64) -> Matrix:
        """Create an identity matrix."""
        data = np.eye(size, dtype=dtype)
        
        properties = MatrixProperties(
            matrix_type=MatrixType.DIAGONAL,
            is_square=True,
            is_singular=False,
            is_invertible=True,
            is_symmetric=True,
            is_positive_definite=True,
            is_orthogonal=True,
            is_unitary=True,
            is_hermitian=True,
            rank=size,
            determinant=1.0,
            trace=float(size),
        )
        
        return cls(
            matrix_id=MatrixId(),
            data=data,
            properties=properties,
            metadata={"special_matrix": "identity", "size": size}
        )
    
    @classmethod
    def zeros(cls, shape: Tuple[int, int], dtype: np.dtype = np.float64) -> Matrix:
        """Create a zero matrix."""
        data = np.zeros(shape, dtype=dtype)
        rows, cols = shape
        
        properties = MatrixProperties(
            matrix_type=MatrixType.GENERAL,
            is_square=rows == cols,
            is_singular=True,
            is_invertible=False,
            is_symmetric=rows == cols,
            is_positive_definite=False,
            is_orthogonal=False,
            is_unitary=False,
            is_hermitian=rows == cols,
            rank=0,
            determinant=0.0 if rows == cols else None,
            trace=0.0,
            sparsity_ratio=1.0,
        )
        
        return cls(
            matrix_id=MatrixId(),
            data=data,
            properties=properties,
            metadata={"special_matrix": "zeros", "shape": shape}
        )
    
    def __str__(self) -> str:
        """String representation of the matrix."""
        return f"Matrix({self.shape[0]}×{self.shape[1]}, {self.properties.matrix_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Matrix(id={self.matrix_id}, shape={self.shape}, "
                f"type={self.properties.matrix_type.value}, "
                f"rank={self.properties.rank})")