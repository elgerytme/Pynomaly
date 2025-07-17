"""Domain service for matrix operations."""

from typing import List, Optional, Tuple
import numpy as np
from scipy import linalg

from ..entities.matrix import Matrix
from ..value_objects.matrix_value_objects import MatrixType, MatrixProperties, DecompositionType


class MatrixOperationsService:
    """Domain service for matrix operations."""
    
    def solve_linear_system(self, A: Matrix, b: Matrix) -> Matrix:
        """
        Solve linear system Ax = b.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector/matrix
            
        Returns:
            Solution matrix x
        """
        if not A.is_square:
            raise ValueError("Coefficient matrix must be square")
        
        if A.rows != b.rows:
            raise ValueError("Incompatible dimensions for linear system")
        
        if A.properties.is_singular:
            raise ValueError("Cannot solve system with singular matrix")
        
        # Solve using appropriate method based on matrix properties
        if A.properties.is_positive_definite:
            # Use Cholesky decomposition
            try:
                solution_data = linalg.solve(A.data, b.data, assume_a='pos')
            except linalg.LinAlgError:
                # Fallback to general solver
                solution_data = linalg.solve(A.data, b.data)
        elif A.properties.is_symmetric:
            # Use symmetric solver
            solution_data = linalg.solve(A.data, b.data, assume_a='sym')
        else:
            # General solver
            solution_data = linalg.solve(A.data, b.data)
        
        return Matrix.from_array(solution_data)
    
    def compute_eigenvalues(self, matrix: Matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if not matrix.is_square:
            raise ValueError("Eigenvalues only defined for square matrices")
        
        # Use appropriate eigenvalue solver based on matrix properties
        if matrix.properties.is_symmetric:
            eigenvals, eigenvecs = linalg.eigh(matrix.data)
        elif matrix.properties.is_hermitian:
            eigenvals, eigenvecs = linalg.eigh(matrix.data)
        else:
            eigenvals, eigenvecs = linalg.eig(matrix.data)
        
        return eigenvals, eigenvecs
    
    def matrix_power(self, matrix: Matrix, power: int) -> Matrix:
        """
        Compute matrix power A^n.
        
        Args:
            matrix: Base matrix
            power: Power to raise matrix to
            
        Returns:
            Matrix power
        """
        if not matrix.is_square:
            raise ValueError("Matrix power only defined for square matrices")
        
        if power == 0:
            return Matrix.identity(matrix.rows, matrix.dtype)
        elif power == 1:
            return matrix
        elif power == -1:
            return matrix.inverse()
        else:
            # Use eigenvalue decomposition for efficient computation
            eigenvals, eigenvecs = self.compute_eigenvalues(matrix)
            
            # Compute eigenvalues to the power
            powered_eigenvals = eigenvals ** power
            
            # Reconstruct matrix
            result_data = eigenvecs @ np.diag(powered_eigenvals) @ eigenvecs.T.conj()
            
            return Matrix.from_array(result_data)
    
    def matrix_exponential(self, matrix: Matrix) -> Matrix:
        """
        Compute matrix exponential exp(A).
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix exponential
        """
        if not matrix.is_square:
            raise ValueError("Matrix exponential only defined for square matrices")
        
        result_data = linalg.expm(matrix.data)
        return Matrix.from_array(result_data)
    
    def matrix_logarithm(self, matrix: Matrix) -> Matrix:
        """
        Compute matrix logarithm log(A).
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix logarithm
        """
        if not matrix.is_square:
            raise ValueError("Matrix logarithm only defined for square matrices")
        
        result_data = linalg.logm(matrix.data)
        return Matrix.from_array(result_data)
    
    def kronecker_product(self, A: Matrix, B: Matrix) -> Matrix:
        """
        Compute Kronecker product A ⊗ B.
        
        Args:
            A: First matrix
            B: Second matrix
            
        Returns:
            Kronecker product
        """
        result_data = np.kron(A.data, B.data)
        return Matrix.from_array(result_data)
    
    def hadamard_product(self, A: Matrix, B: Matrix) -> Matrix:
        """
        Compute Hadamard (element-wise) product A ∘ B.
        
        Args:
            A: First matrix
            B: Second matrix
            
        Returns:
            Hadamard product
        """
        if A.shape != B.shape:
            raise ValueError("Matrices must have the same shape for Hadamard product")
        
        result_data = A.data * B.data
        return Matrix.from_array(result_data)
    
    def matrix_square_root(self, matrix: Matrix) -> Matrix:
        """
        Compute matrix square root.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix square root
        """
        if not matrix.is_square:
            raise ValueError("Matrix square root only defined for square matrices")
        
        result_data = linalg.sqrtm(matrix.data)
        return Matrix.from_array(result_data)
    
    def pseudo_inverse(self, matrix: Matrix) -> Matrix:
        """
        Compute Moore-Penrose pseudoinverse.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Pseudoinverse matrix
        """
        result_data = linalg.pinv(matrix.data)
        return Matrix.from_array(result_data)
    
    def compute_condition_number(self, matrix: Matrix, norm_type: str = "2") -> float:
        """
        Compute condition number of matrix.
        
        Args:
            matrix: Input matrix
            norm_type: Type of norm to use
            
        Returns:
            Condition number
        """
        return matrix.condition_number(norm_type)
    
    def is_positive_definite(self, matrix: Matrix) -> bool:
        """
        Check if matrix is positive definite.
        
        Args:
            matrix: Input matrix
            
        Returns:
            True if positive definite
        """
        if not matrix.is_square:
            return False
        
        if not matrix.properties.is_symmetric:
            return False
        
        # Check if all eigenvalues are positive
        eigenvals, _ = self.compute_eigenvalues(matrix)
        return np.all(eigenvals > 0)
    
    def is_orthogonal(self, matrix: Matrix, tolerance: float = 1e-10) -> bool:
        """
        Check if matrix is orthogonal.
        
        Args:
            matrix: Input matrix
            tolerance: Numerical tolerance
            
        Returns:
            True if orthogonal
        """
        if not matrix.is_square:
            return False
        
        # Check if A^T * A = I
        transpose = matrix.transpose()
        product = matrix.multiply(transpose)
        identity = Matrix.identity(matrix.rows, matrix.dtype)
        
        return product.is_equal(identity, tolerance)
    
    def spectral_radius(self, matrix: Matrix) -> float:
        """
        Compute spectral radius (largest eigenvalue magnitude).
        
        Args:
            matrix: Input matrix
            
        Returns:
            Spectral radius
        """
        if not matrix.is_square:
            raise ValueError("Spectral radius only defined for square matrices")
        
        eigenvals, _ = self.compute_eigenvalues(matrix)
        return np.max(np.abs(eigenvals))
    
    def matrix_norm(self, matrix: Matrix, norm_type: str = "fro") -> float:
        """
        Compute matrix norm.
        
        Args:
            matrix: Input matrix
            norm_type: Type of norm ('fro', '2', '1', 'inf', 'nuc')
            
        Returns:
            Matrix norm
        """
        if norm_type == "fro":
            return np.linalg.norm(matrix.data, 'fro')
        elif norm_type == "2":
            return np.linalg.norm(matrix.data, 2)
        elif norm_type == "1":
            return np.linalg.norm(matrix.data, 1)
        elif norm_type == "inf":
            return np.linalg.norm(matrix.data, np.inf)
        elif norm_type == "nuc":
            return np.linalg.norm(matrix.data, 'nuc')
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def diagonalize(self, matrix: Matrix) -> Tuple[Matrix, Matrix]:
        """
        Diagonalize matrix if possible.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (eigenvalue matrix, eigenvector matrix)
        """
        if not matrix.is_square:
            raise ValueError("Only square matrices can be diagonalized")
        
        eigenvals, eigenvecs = self.compute_eigenvalues(matrix)
        
        # Check if matrix is diagonalizable (has full set of eigenvectors)
        if eigenvecs.shape[1] != matrix.rows:
            raise ValueError("Matrix is not diagonalizable")
        
        eigenval_matrix = Matrix.from_array(np.diag(eigenvals))
        eigenvec_matrix = Matrix.from_array(eigenvecs)
        
        return eigenval_matrix, eigenvec_matrix