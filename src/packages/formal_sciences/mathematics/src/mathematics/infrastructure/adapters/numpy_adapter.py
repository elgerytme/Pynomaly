"""NumPy adapter for numerical mathematics."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import linalg, optimize, integrate

from ...domain.entities.matrix import Matrix
from ...domain.value_objects.matrix_value_objects import MatrixType, MatrixProperties


class NumPyAdapter:
    """Adapter for NumPy numerical mathematics library."""
    
    def __init__(self):
        self._default_dtype = np.float64
    
    def create_matrix(self, data: List[List[float]], 
                     matrix_type: MatrixType = MatrixType.GENERAL) -> Matrix:
        """Create matrix from data."""
        np_array = np.array(data, dtype=self._default_dtype)
        return Matrix.from_array(np_array, matrix_type)
    
    def create_identity_matrix(self, size: int) -> Matrix:
        """Create identity matrix."""
        return Matrix.identity(size, self._default_dtype)
    
    def create_zero_matrix(self, shape: Tuple[int, int]) -> Matrix:
        """Create zero matrix."""
        return Matrix.zeros(shape, self._default_dtype)
    
    def create_random_matrix(self, shape: Tuple[int, int], 
                           distribution: str = "uniform") -> Matrix:
        """Create random matrix."""
        if distribution == "uniform":
            data = np.random.uniform(0, 1, shape)
        elif distribution == "normal":
            data = np.random.normal(0, 1, shape)
        elif distribution == "integer":
            data = np.random.randint(0, 10, shape).astype(self._default_dtype)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return Matrix.from_array(data)
    
    def analyze_matrix_properties(self, matrix: Matrix) -> MatrixProperties:
        """Analyze matrix properties using NumPy."""
        data = matrix.data
        rows, cols = data.shape
        
        # Basic properties
        is_square = rows == cols
        rank = np.linalg.matrix_rank(data)
        is_singular = is_square and rank < rows
        is_invertible = is_square and not is_singular
        
        # Symmetry checks
        is_symmetric = is_square and np.allclose(data, data.T)
        is_hermitian = is_square and np.allclose(data, data.T.conj())
        
        # Orthogonality check
        is_orthogonal = False
        if is_square:
            try:
                product = data @ data.T
                identity = np.eye(rows)
                is_orthogonal = np.allclose(product, identity)
            except:
                pass
        
        # Unitary check (for complex matrices)
        is_unitary = False
        if is_square and np.iscomplexobj(data):
            try:
                product = data @ data.T.conj()
                identity = np.eye(rows)
                is_unitary = np.allclose(product, identity)
            except:
                pass
        
        # Positive definite check
        is_positive_definite = False
        if is_square and is_symmetric:
            try:
                eigenvals = np.linalg.eigvals(data)
                is_positive_definite = np.all(eigenvals > 0)
            except:
                pass
        
        # Compute norms
        frobenius_norm = np.linalg.norm(data, 'fro')
        spectral_norm = np.linalg.norm(data, 2)
        nuclear_norm = np.linalg.norm(data, 'nuc')
        
        # Compute other properties
        determinant = None
        trace = None
        condition_number = None
        
        if is_square:
            try:
                determinant = np.linalg.det(data)
                trace = np.trace(data)
                condition_number = np.linalg.cond(data)
            except:
                pass
        
        # Sparsity ratio
        sparsity_ratio = np.count_nonzero(data == 0) / data.size
        
        return MatrixProperties(
            matrix_type=self._determine_matrix_type(data),
            is_square=is_square,
            is_singular=is_singular,
            is_invertible=is_invertible,
            is_symmetric=is_symmetric,
            is_positive_definite=is_positive_definite,
            is_orthogonal=is_orthogonal,
            is_unitary=is_unitary,
            is_hermitian=is_hermitian,
            rank=rank,
            determinant=determinant,
            trace=trace,
            condition_number=condition_number,
            frobenius_norm=frobenius_norm,
            spectral_norm=spectral_norm,
            nuclear_norm=nuclear_norm,
            sparsity_ratio=sparsity_ratio,
        )
    
    def _determine_matrix_type(self, data: np.ndarray) -> MatrixType:
        """Determine matrix type from data."""
        rows, cols = data.shape
        
        if rows != cols:
            return MatrixType.GENERAL
        
        # Check for diagonal matrix
        if np.allclose(data, np.diag(np.diag(data))):
            return MatrixType.DIAGONAL
        
        # Check for upper triangular
        if np.allclose(data, np.triu(data)):
            return MatrixType.UPPER_TRIANGULAR
        
        # Check for lower triangular
        if np.allclose(data, np.tril(data)):
            return MatrixType.LOWER_TRIANGULAR
        
        # Check for symmetric
        if np.allclose(data, data.T):
            return MatrixType.SYMMETRIC
        
        # Check for antisymmetric
        if np.allclose(data, -data.T):
            return MatrixType.ANTISYMMETRIC
        
        # Check for orthogonal
        try:
            if np.allclose(data @ data.T, np.eye(rows)):
                return MatrixType.ORTHOGONAL
        except:
            pass
        
        # Check for positive definite
        try:
            if np.allclose(data, data.T) and np.all(np.linalg.eigvals(data) > 0):
                return MatrixType.POSITIVE_DEFINITE
        except:
            pass
        
        # Check for sparsity
        sparsity_ratio = np.count_nonzero(data == 0) / data.size
        if sparsity_ratio > 0.5:
            return MatrixType.SPARSE
        
        return MatrixType.GENERAL
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b."""
        return linalg.solve(A, b)
    
    def compute_eigenvalues(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors."""
        return linalg.eig(matrix)
    
    def compute_svd(self, matrix: np.ndarray, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute SVD decomposition."""
        return linalg.svd(matrix, full_matrices=full_matrices)
    
    def compute_lu_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute LU decomposition."""
        return linalg.lu(matrix)
    
    def compute_qr_decomposition(self, matrix: np.ndarray, mode: str = 'reduced') -> Tuple[np.ndarray, np.ndarray]:
        """Compute QR decomposition."""
        return linalg.qr(matrix, mode=mode)
    
    def compute_cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
        """Compute Cholesky decomposition."""
        return linalg.cholesky(matrix)
    
    def matrix_power(self, matrix: np.ndarray, power: int) -> np.ndarray:
        """Compute matrix power."""
        return np.linalg.matrix_power(matrix, power)
    
    def matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential."""
        return linalg.expm(matrix)
    
    def matrix_logarithm(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix logarithm."""
        return linalg.logm(matrix)
    
    def matrix_square_root(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        return linalg.sqrtm(matrix)
    
    def pseudo_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Compute Moore-Penrose pseudoinverse."""
        return linalg.pinv(matrix)
    
    def compute_condition_number(self, matrix: np.ndarray, p: str = "2") -> float:
        """Compute condition number."""
        return np.linalg.cond(matrix, p)
    
    def compute_matrix_norm(self, matrix: np.ndarray, ord: str = 'fro') -> float:
        """Compute matrix norm."""
        return np.linalg.norm(matrix, ord)
    
    def numerical_integration(self, func, a: float, b: float, 
                            method: str = "quad") -> Tuple[float, float]:
        """Numerical integration."""
        if method == "quad":
            return integrate.quad(func, a, b)
        elif method == "simpson":
            # Create points for Simpson's rule
            n = 1000  # Number of intervals
            x = np.linspace(a, b, n + 1)
            y = np.array([func(xi) for xi in x])
            return integrate.simpson(y, x), 0.0
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def find_roots(self, func, initial_guess: float, 
                  method: str = "brentq") -> float:
        """Find roots of a function."""
        if method == "brentq":
            # Need to find bracket first
            return optimize.brentq(func, initial_guess - 1, initial_guess + 1)
        elif method == "fsolve":
            return optimize.fsolve(func, initial_guess)[0]
        else:
            raise ValueError(f"Unknown root finding method: {method}")
    
    def optimize_function(self, func, initial_guess: float, 
                         method: str = "minimize") -> Dict[str, Any]:
        """Optimize a function."""
        if method == "minimize":
            result = optimize.minimize(func, initial_guess)
            return {
                "x": result.x,
                "fun": result.fun,
                "success": result.success,
                "message": result.message,
            }
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def interpolate(self, x_data: np.ndarray, y_data: np.ndarray, 
                   x_new: np.ndarray, method: str = "linear") -> np.ndarray:
        """Interpolate data points."""
        from scipy.interpolate import interp1d
        
        if method in ["linear", "quadratic", "cubic"]:
            f = interp1d(x_data, y_data, kind=method)
            return f(x_new)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def fit_polynomial(self, x_data: np.ndarray, y_data: np.ndarray, 
                      degree: int) -> np.ndarray:
        """Fit polynomial to data."""
        return np.polyfit(x_data, y_data, degree)
    
    def evaluate_polynomial(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at given points."""
        return np.polyval(coefficients, x)