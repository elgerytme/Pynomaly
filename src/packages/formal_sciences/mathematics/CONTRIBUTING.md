# Contributing to Mathematics Package

Thank you for your interest in contributing to the Mathematics package! This package provides comprehensive mathematical computation capabilities serving as the foundational layer for all mathematical operations across the Pynomaly platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Mathematical Development Guidelines](#mathematical-development-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Mathematical Accuracy and Validation](#mathematical-accuracy-and-validation)
- [Community](#community)

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Strong mathematical background and understanding of numerical methods
- Knowledge of linear algebra, calculus, and numerical analysis
- Familiarity with NumPy, SciPy, and SymPy
- Understanding of floating-point arithmetic and numerical stability

### Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-org/monorepo.git
cd monorepo/src/packages/formal_sciences/mathematics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,symbolic,optimization,visualization]"

# Install pre-commit hooks
pre-commit install
```

### Mathematical Libraries Setup

```bash
# Core mathematical dependencies
pip install numpy scipy sympy

# High-performance computing libraries
pip install numba numexpr

# Optimization libraries
pip install cvxpy cvxopt

# Symbolic mathematics
pip install sympy sage-math

# Visualization
pip install matplotlib plotly

# Testing libraries
pip install pytest hypothesis mpmath
```

## Development Environment

### IDE Configuration

Recommended VS Code extensions:
- Python
- Jupyter
- LaTeX Workshop (for mathematical documentation)
- Python Docstring Generator

### Environment Variables

Create a `.env` file for local development:

```bash
# Mathematical Computation Settings
MATH_DEFAULT_PRECISION=15
MATH_MAX_ITERATIONS=10000
MATH_CONVERGENCE_TOLERANCE=1e-12

# Performance Settings
MATH_USE_PARALLEL=true
MATH_MAX_THREADS=8
MATH_CACHE_SIZE=1000

# Numerical Stability Settings
MATH_CHECK_CONDITIONING=true
MATH_WARN_ILL_CONDITIONED=true
MATH_PIVOTING_THRESHOLD=1e-14

# Testing Settings
MATH_RUN_SLOW_TESTS=false
MATH_REFERENCE_PRECISION=50
MATH_GENERATE_PLOTS=false

# Library Integration
MATH_USE_BLAS=true
MATH_USE_LAPACK=true
MATH_USE_GPU=false
```

### Development Dependencies

```bash
# Install with all mathematical features
pip install -e ".[dev,test,symbolic,optimization,gpu,parallel]"

# For high-precision arithmetic
pip install mpmath decimal

# For computer algebra
pip install sympy sage-math

# For optimization
pip install scipy cvxpy pulp

# For parallel computing
pip install numba dask ray

# For testing mathematical accuracy
pip install hypothesis mpmath
```

## Mathematical Development Guidelines

### Core Principles

1. **Mathematical Rigor**: All implementations must be mathematically correct and well-documented
2. **Numerical Stability**: Algorithms must be numerically stable and handle edge cases
3. **Performance**: Efficient algorithms with optimal computational complexity
4. **Accuracy**: Maintain specified numerical precision and provide error estimates
5. **Generality**: Support wide ranges of input types and mathematical structures

### Mathematical Component Development

**Function Implementation Pattern:**
```python
from typing import Union, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class MathematicalResult:
    """Standard result container for mathematical operations."""
    value: Any
    error_estimate: Optional[float]
    iterations: Optional[int]
    convergence_info: Optional[Dict[str, Any]]
    computation_time: float
    method_used: str
    precision_achieved: float

class MathematicalFunction:
    """Base class for mathematical function implementations."""
    
    def __init__(self, precision: int = 15, max_iterations: int = 1000):
        self.precision = precision
        self.max_iterations = max_iterations
        self.tolerance = 10 ** (-precision)
    
    def validate_input(self, *args, **kwargs) -> None:
        """Validate mathematical input parameters."""
        raise NotImplementedError
    
    def estimate_error(self, result: Any, *args, **kwargs) -> float:
        """Estimate numerical error in the result."""
        raise NotImplementedError
    
    def check_convergence(self, current: Any, previous: Any, iteration: int) -> bool:
        """Check convergence criteria."""
        if iteration >= self.max_iterations:
            return True
        
        # Default convergence check
        if hasattr(current, '__sub__') and hasattr(current, '__abs__'):
            return abs(current - previous) < self.tolerance
        
        return False

class LinearAlgebraOperations:
    """Linear algebra operations with numerical stability."""
    
    @staticmethod
    def solve_linear_system(
        A: np.ndarray, 
        b: np.ndarray,
        method: str = "auto",
        check_conditioning: bool = True
    ) -> MathematicalResult:
        """
        Solve linear system Ax = b with comprehensive error analysis.
        
        Args:
            A: Coefficient matrix (n x n)
            b: Right-hand side vector (n,) or matrix (n x m)
            method: Solution method ('lu', 'qr', 'svd', 'auto')
            check_conditioning: Whether to check matrix conditioning
            
        Returns:
            MathematicalResult containing solution and analysis
            
        Raises:
            ValueError: If inputs are invalid
            LinAlgError: If system cannot be solved
            
        Mathematical Background:
            For a linear system Ax = b, we need to ensure:
            1. A is non-singular (det(A) ≠ 0)
            2. System is well-conditioned (cond(A) < 1/ε)
            3. Numerical stability is maintained
            
        Algorithm Selection:
            - Well-conditioned: LU decomposition with partial pivoting
            - Ill-conditioned: SVD with regularization
            - Overdetermined: QR decomposition (least squares)
            - Sparse: Iterative methods (CG, GMRES)
        """
        start_time = time.perf_counter()
        
        # Input validation
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        
        if A.shape[0] != b.shape[0]:
            raise ValueError("Incompatible dimensions: A and b")
        
        n = A.shape[0]
        
        # Conditioning analysis
        condition_number = None
        if check_conditioning:
            condition_number = np.linalg.cond(A)
            if condition_number > 1e12:
                warnings.warn(f"Matrix is ill-conditioned (cond = {condition_number:.2e})")
        
        # Method selection
        if method == "auto":
            if condition_number and condition_number > 1e8:
                method = "svd"  # Use SVD for ill-conditioned systems
            elif n > 1000:
                method = "lu"   # LU for large well-conditioned systems
            else:
                method = "qr"   # QR for moderate-sized systems
        
        # Solve system
        try:
            if method == "lu":
                solution = scipy.linalg.solve(A, b, assume_a='gen')
                error_estimate = self._estimate_lu_error(A, b, solution)
                
            elif method == "qr":
                Q, R = scipy.linalg.qr(A)
                solution = scipy.linalg.solve_triangular(R, Q.T @ b)
                error_estimate = self._estimate_qr_error(A, b, solution)
                
            elif method == "svd":
                U, s, Vt = scipy.linalg.svd(A)
                # Regularization for small singular values
                s_inv = np.where(s > 1e-12, 1/s, 0)
                solution = Vt.T @ np.diag(s_inv) @ U.T @ b
                error_estimate = self._estimate_svd_error(A, b, solution, s)
                
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except np.linalg.LinAlgError as e:
            raise LinAlgError(f"Failed to solve linear system: {e}")
        
        # Compute residual and backward error
        residual = A @ solution - b
        backward_error = np.linalg.norm(residual) / np.linalg.norm(b)
        
        computation_time = time.perf_counter() - start_time
        
        return MathematicalResult(
            value=solution.flatten() if solution.shape[1] == 1 else solution,
            error_estimate=error_estimate,
            iterations=None,
            convergence_info={
                'method': method,
                'condition_number': condition_number,
                'backward_error': backward_error,
                'residual_norm': np.linalg.norm(residual)
            },
            computation_time=computation_time,
            method_used=method,
            precision_achieved=max(0, -np.log10(backward_error))
        )
    
    @staticmethod
    def _estimate_lu_error(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        """Estimate error for LU decomposition solution."""
        # Use condition number and machine epsilon
        cond_A = np.linalg.cond(A)
        machine_eps = np.finfo(A.dtype).eps
        return cond_A * machine_eps * np.linalg.norm(x)
```

**Numerical Integration Example:**
```python
class NumericalIntegration:
    """Numerical integration with adaptive algorithms."""
    
    @staticmethod
    def adaptive_quadrature(
        func: Callable[[float], float],
        a: float,
        b: float,
        tolerance: float = 1e-10,
        max_subdivisions: int = 1000
    ) -> MathematicalResult:
        """
        Adaptive Gaussian quadrature integration.
        
        Mathematical Foundation:
            Uses Gauss-Legendre quadrature with adaptive subdivision:
            ∫[a,b] f(x)dx ≈ Σᵢ wᵢ f(xᵢ)
            
            Where xᵢ are Legendre polynomial roots and wᵢ are weights.
            Error estimation using higher-order rule comparison.
            
        Algorithm:
            1. Apply Gauss-Legendre rules of different orders
            2. Estimate error from rule differences
            3. Subdivide intervals where error exceeds tolerance
            4. Recursively apply until convergence
        """
        start_time = time.perf_counter()
        
        # Gauss-Legendre quadrature points and weights
        def gauss_legendre_7():
            """7-point Gauss-Legendre quadrature."""
            points = np.array([
                -0.9491079123427585, -0.7415311855993944, -0.4058451513773972,
                0.0, 0.4058451513773972, 0.7415311855993944, 0.9491079123427585
            ])
            weights = np.array([
                0.1294849661688697, 0.2797053914892766, 0.3818300505051189,
                0.4179591836734694, 0.3818300505051189, 0.2797053914892766,
                0.1294849661688697
            ])
            return points, weights
        
        def gauss_legendre_15():
            """15-point Gauss-Legendre quadrature for error estimation."""
            # Implementation of 15-point rule
            # (points and weights omitted for brevity)
            pass
        
        def integrate_interval(a: float, b: float) -> Tuple[float, float]:
            """Integrate over single interval with error estimate."""
            # Transform to [-1, 1]
            midpoint = (a + b) / 2
            half_width = (b - a) / 2
            
            points_7, weights_7 = gauss_legendre_7()
            
            # 7-point rule
            x_transformed = midpoint + half_width * points_7
            f_values = np.array([func(x) for x in x_transformed])
            integral_7 = half_width * np.sum(weights_7 * f_values)
            
            # 15-point rule for error estimation
            # (implementation details omitted)
            integral_15 = integral_7  # Placeholder
            
            error_estimate = abs(integral_15 - integral_7)
            
            return integral_7, error_estimate
        
        # Adaptive subdivision algorithm
        result = 0.0
        total_error = 0.0
        subdivisions = 0
        
        # Priority queue for intervals needing refinement
        intervals = [(a, b)]
        
        while intervals and subdivisions < max_subdivisions:
            current_a, current_b = intervals.pop(0)
            integral, error = integrate_interval(current_a, current_b)
            
            if error <= tolerance * (current_b - current_a) / (b - a):
                # Accept this interval
                result += integral
                total_error += error
            else:
                # Subdivide interval
                midpoint = (current_a + current_b) / 2
                intervals.extend([(current_a, midpoint), (midpoint, current_b)])
                subdivisions += 1
        
        computation_time = time.perf_counter() - start_time
        
        return MathematicalResult(
            value=result,
            error_estimate=total_error,
            iterations=subdivisions,
            convergence_info={
                'subdivisions_used': subdivisions,
                'tolerance_achieved': total_error,
                'intervals_processed': subdivisions + 1
            },
            computation_time=computation_time,
            method_used="adaptive_gauss_legendre",
            precision_achieved=max(0, -np.log10(total_error / abs(result))) if result != 0 else float('inf')
        )
```

### Symbolic Mathematics Integration

```python
import sympy as sp
from sympy import symbols, Function, diff, integrate, solve

class SymbolicMathematics:
    """Integration with SymPy for symbolic computation."""
    
    def __init__(self):
        self.symbol_cache = {}
    
    def create_symbolic_function(
        self,
        expression: str,
        variables: List[str]
    ) -> sp.Expr:
        """Create symbolic function from string expression."""
        # Create symbols
        symbol_dict = {}
        for var in variables:
            if var not in self.symbol_cache:
                self.symbol_cache[var] = symbols(var, real=True)
            symbol_dict[var] = self.symbol_cache[var]
        
        # Parse expression
        try:
            expr = sp.sympify(expression, locals=symbol_dict)
            return expr
        except (sp.SympifyError, ValueError) as e:
            raise ValueError(f"Invalid symbolic expression: {e}")
    
    def symbolic_derivative(
        self,
        expression: str,
        variable: str,
        order: int = 1
    ) -> Tuple[str, Optional[Callable]]:
        """Compute symbolic derivative."""
        expr = self.create_symbolic_function(expression, [variable])
        var_symbol = self.symbol_cache[variable]
        
        # Compute derivative
        derivative = diff(expr, var_symbol, order)
        
        # Convert back to string
        derivative_str = str(derivative)
        
        # Create numerical function
        try:
            numerical_func = sp.lambdify(var_symbol, derivative, 'numpy')
        except Exception:
            numerical_func = None
        
        return derivative_str, numerical_func
    
    def symbolic_integration(
        self,
        expression: str,
        variable: str,
        bounds: Optional[Tuple[float, float]] = None
    ) -> MathematicalResult:
        """Compute symbolic or numerical integration."""
        start_time = time.perf_counter()
        
        expr = self.create_symbolic_function(expression, [variable])
        var_symbol = self.symbol_cache[variable]
        
        try:
            if bounds is None:
                # Indefinite integral
                integral = integrate(expr, var_symbol)
                result_expr = str(integral)
                numerical_value = None
            else:
                # Definite integral
                integral = integrate(expr, (var_symbol, bounds[0], bounds[1]))
                result_expr = str(integral)
                
                # Try to evaluate numerically
                try:
                    numerical_value = float(integral.evalf())
                except Exception:
                    numerical_value = None
            
            computation_time = time.perf_counter() - start_time
            
            return MathematicalResult(
                value=numerical_value if numerical_value is not None else result_expr,
                error_estimate=None,  # Symbolic computation
                iterations=None,
                convergence_info={
                    'method': 'symbolic',
                    'symbolic_result': result_expr
                },
                computation_time=computation_time,
                method_used="sympy_integration",
                precision_achieved=float('inf')  # Exact symbolic result
            )
            
        except Exception as e:
            # Fall back to numerical integration
            numerical_func = sp.lambdify(var_symbol, expr, 'numpy')
            if bounds is not None:
                # Use numerical integration as fallback
                return self.numerical_integration(numerical_func, bounds[0], bounds[1])
            else:
                raise ValueError(f"Cannot compute indefinite integral: {e}")
```

## Testing Requirements

### Test Categories

1. **Mathematical Accuracy Tests**: Verify correctness against known solutions
2. **Numerical Stability Tests**: Test behavior with ill-conditioned problems
3. **Performance Tests**: Benchmark computational efficiency
4. **Property-Based Tests**: Test mathematical properties and invariants
5. **Edge Case Tests**: Handle boundary conditions and special values

### Test Structure

```bash
tests/
├── unit/                    # Unit tests for individual functions
│   ├── linear_algebra/     # Linear algebra operations
│   ├── calculus/          # Calculus operations
│   ├── numerical/         # Numerical methods
│   └── symbolic/          # Symbolic mathematics
├── integration/           # Integration tests
├── accuracy/             # Mathematical accuracy validation
├── performance/          # Performance benchmarks
├── property/            # Property-based tests
└── fixtures/            # Test data and reference solutions
```

### Mathematical Testing Patterns

```python
import pytest
import numpy as np
import mpmath
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

class TestLinearAlgebra:
    """Test linear algebra operations with mathematical rigor."""
    
    def test_matrix_multiplication_associativity(self):
        """Test (AB)C = A(BC) for matrix multiplication."""
        A = np.random.rand(5, 4)
        B = np.random.rand(4, 6)
        C = np.random.rand(6, 3)
        
        left = (A @ B) @ C
        right = A @ (B @ C)
        
        np.testing.assert_allclose(left, right, rtol=1e-12)
    
    def test_matrix_inverse_accuracy(self):
        """Test that A * A^(-1) = I for well-conditioned matrices."""
        # Generate well-conditioned matrix
        n = 10
        A = np.random.rand(n, n)
        A = A @ A.T + n * np.eye(n)  # Ensure positive definite
        
        A_inv = np.linalg.inv(A)
        product = A @ A_inv
        identity = np.eye(n)
        
        # Check accuracy
        error = np.linalg.norm(product - identity, 'fro')
        assert error < 1e-12, f"Inverse accuracy error: {error}"
    
    @pytest.mark.parametrize("condition_number", [1e2, 1e6, 1e10])
    def test_linear_system_conditioning(self, condition_number):
        """Test linear system solution with varying condition numbers."""
        n = 20
        
        # Create matrix with specified condition number
        U, _, Vt = np.linalg.svd(np.random.rand(n, n))
        singular_values = np.logspace(0, -np.log10(condition_number), n)
        A = U @ np.diag(singular_values) @ Vt
        
        # Create solution and right-hand side
        x_true = np.random.rand(n)
        b = A @ x_true
        
        # Solve system
        result = LinearAlgebraOperations.solve_linear_system(
            A, b, check_conditioning=True
        )
        
        # Check solution accuracy
        relative_error = np.linalg.norm(result.value - x_true) / np.linalg.norm(x_true)
        
        # Expected error should scale with condition number
        expected_error = condition_number * np.finfo(float).eps
        assert relative_error < 100 * expected_error

@composite
def well_conditioned_matrices(draw, min_size=2, max_size=10):
    """Generate well-conditioned matrices for testing."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate random matrix and make it well-conditioned
    A = draw(st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=n*n, max_size=n*n
    ))
    A = np.array(A).reshape(n, n)
    
    # Ensure well-conditioned by adding to diagonal
    A = A + (np.max(np.abs(A)) + 1) * np.eye(n)
    
    return A

@given(A=well_conditioned_matrices())
def test_matrix_properties(A):
    """Property-based testing of matrix operations."""
    n = A.shape[0]
    
    # Test determinant properties
    det_A = np.linalg.det(A)
    
    # det(A^T) = det(A)
    det_AT = np.linalg.det(A.T)
    assert abs(det_A - det_AT) < 1e-10
    
    # det(kA) = k^n * det(A) for scalar k
    k = 2.0
    kA = k * A
    det_kA = np.linalg.det(kA)
    expected = (k ** n) * det_A
    assert abs(det_kA - expected) < 1e-10 * abs(expected)

class TestNumericalAccuracy:
    """Test numerical accuracy against high-precision references."""
    
    def test_gamma_function_accuracy(self):
        """Test gamma function against mpmath reference."""
        test_values = [0.5, 1.0, 1.5, 2.0, 3.5, 10.5]
        
        for x in test_values:
            # Our implementation
            result = math_functions.gamma(x)
            
            # High-precision reference
            mpmath.mp.dps = 50  # 50 decimal places
            reference = float(mpmath.gamma(x))
            
            relative_error = abs(result - reference) / abs(reference)
            assert relative_error < 1e-14, f"Gamma({x}) accuracy error: {relative_error}"
    
    def test_bessel_function_accuracy(self):
        """Test Bessel functions against reference implementations."""
        orders = [0, 1, 2, 0.5, 1.5]
        x_values = [0.1, 1.0, 5.0, 10.0]
        
        for n in orders:
            for x in x_values:
                # Our implementation
                result = special_functions.bessel_j(n, x)
                
                # SciPy reference
                reference = scipy.special.jv(n, x)
                
                relative_error = abs(result - reference) / abs(reference)
                assert relative_error < 1e-12

class TestNumericalStability:
    """Test numerical stability with challenging problems."""
    
    def test_hilbert_matrix_solution(self):
        """Test linear system solution with Hilbert matrix."""
        for n in range(3, 8):
            # Create Hilbert matrix (notoriously ill-conditioned)
            H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
            
            # Known solution
            x_true = np.ones(n)
            b = H @ x_true
            
            # Solve with our implementation
            result = LinearAlgebraOperations.solve_linear_system(
                H, b, method="svd"  # Use SVD for stability
            )
            
            # Check that we detect ill-conditioning
            assert result.convergence_info['condition_number'] > 1e6
            
            # Solution should still be reasonably accurate
            relative_error = np.linalg.norm(result.value - x_true) / np.linalg.norm(x_true)
            assert relative_error < 1e-6  # Reasonable accuracy given ill-conditioning
    
    def test_catastrophic_cancellation(self):
        """Test handling of catastrophic cancellation."""
        # Function that suffers from cancellation: (1 - cos(x)) / x^2
        # Should be computed as sin(x/2)^2 / (x/2)^2 for small x
        
        small_values = [1e-8, 1e-10, 1e-12]
        
        for x in small_values:
            # Naive computation (prone to cancellation)
            naive_result = (1 - np.cos(x)) / (x**2)
            
            # Stable computation
            half_x = x / 2
            stable_result = (np.sin(half_x) / half_x) ** 2
            
            # Our implementation should use stable form
            our_result = numerical_functions.one_minus_cos_over_x_squared(x)
            
            # Our result should match stable computation
            assert abs(our_result - stable_result) < 1e-15
            
            # And be much more accurate than naive computation for small x
            if x < 1e-6:
                naive_error = abs(naive_result - stable_result) / stable_result
                our_error = abs(our_result - stable_result) / stable_result
                assert our_error < 0.01 * naive_error
```

### Performance Benchmarks

```python
import timeit
import pytest

class TestPerformance:
    """Performance benchmarks for mathematical operations."""
    
    @pytest.mark.performance
    def test_matrix_multiplication_performance(self):
        """Benchmark matrix multiplication performance."""
        sizes = [100, 500, 1000, 2000]
        
        for n in sizes:
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            
            # Time our implementation
            start_time = time.perf_counter()
            result = linear_algebra.matrix_multiply(A, B)
            our_time = time.perf_counter() - start_time
            
            # Time NumPy reference
            start_time = time.perf_counter()
            reference = A @ B
            numpy_time = time.perf_counter() - start_time
            
            # Our implementation should be within 2x of NumPy
            assert our_time < 2 * numpy_time, f"Performance regression for size {n}"
            
            # Results should match
            np.testing.assert_allclose(result, reference, rtol=1e-12)
    
    @pytest.mark.performance  
    def test_eigenvalue_performance(self):
        """Benchmark eigenvalue computation performance."""
        matrix_sizes = [50, 100, 200, 500]
        
        for n in matrix_sizes:
            # Create symmetric matrix for eigenvalues
            A = np.random.rand(n, n)
            A = A @ A.T  # Make symmetric positive definite
            
            # Benchmark our implementation
            start_time = time.perf_counter()
            eigenvals, eigenvecs = linear_algebra.eigenvalues(A)
            our_time = time.perf_counter() - start_time
            
            # Compare with SciPy
            start_time = time.perf_counter()
            ref_vals, ref_vecs = scipy.linalg.eigh(A)
            scipy_time = time.perf_counter() - start_time
            
            # Performance should be competitive
            assert our_time < 3 * scipy_time
            
            # Accuracy check
            np.testing.assert_allclose(
                np.sort(eigenvals), np.sort(ref_vals), rtol=1e-10
            )
```

## Documentation Standards

### Mathematical Documentation

All mathematical functions must include:

```python
def newton_raphson_method(
    func: Callable[[float], float],
    derivative: Callable[[float], float], 
    initial_guess: float,
    tolerance: float = 1e-12,
    max_iterations: int = 100
) -> MathematicalResult:
    """
    Find root of function using Newton-Raphson method.
    
    Mathematical Background:
        The Newton-Raphson method uses the iterative formula:
        x_{n+1} = x_n - f(x_n)/f'(x_n)
        
        Convergence is quadratic near simple roots, provided:
        1. f'(x*) ≠ 0 (simple root)
        2. f and f' are sufficiently smooth
        3. Initial guess is sufficiently close to root
        
    Convergence Analysis:
        - Rate: Quadratic (error roughly squares each iteration)
        - Basin of attraction: Depends on function properties
        - Failure modes: Multiple roots, inflection points, poor initial guess
        
    Algorithm:
        1. Start with initial guess x₀
        2. Iterate: x_{n+1} = x_n - f(x_n)/f'(x_n)
        3. Continue until |x_{n+1} - x_n| < tolerance
        4. Check convergence and validate result
        
    Args:
        func: Function f(x) for which to find root
        derivative: Derivative f'(x) of the function
        initial_guess: Starting point for iteration
        tolerance: Convergence tolerance (default: 1e-12)
        max_iterations: Maximum number of iterations (default: 100)
        
    Returns:
        MathematicalResult containing:
            - value: Root location (or best approximation)
            - error_estimate: Estimated error in root location
            - iterations: Number of iterations performed
            - convergence_info: Convergence analysis data
            - computation_time: Wall-clock time used
            - method_used: "newton_raphson"
            - precision_achieved: Estimated precision of result
            
    Raises:
        ValueError: If inputs are invalid
        ConvergenceError: If method fails to converge
        NumericalError: If derivative is zero or numerical issues occur
        
    Examples:
        Find root of x² - 2 = 0 (should give √2 ≈ 1.414):
        
        >>> def f(x): return x**2 - 2
        >>> def df(x): return 2*x
        >>> result = newton_raphson_method(f, df, 1.0)
        >>> print(f"Root: {result.value:.10f}")
        Root: 1.4142135624
        >>> print(f"Iterations: {result.iterations}")
        Iterations: 4
        
        Find root of cos(x) = 0 (should give π/2):
        
        >>> import math
        >>> def f(x): return math.cos(x)
        >>> def df(x): return -math.sin(x)
        >>> result = newton_raphson_method(f, df, 1.0)
        >>> print(f"Root: {result.value:.10f}")
        Root: 1.5707963268
        
    References:
        - Burden, R.L. & Faires, J.D. "Numerical Analysis", Chapter 2
        - Press, W.H. et al. "Numerical Recipes", Section 9.4
        - Atkinson, K.E. "An Introduction to Numerical Analysis", Chapter 2
        
    See Also:
        secant_method: Alternative method not requiring derivative
        bisection_method: Slower but more robust bracketing method
        brent_method: Hybrid method combining robustness and speed
        
    Notes:
        - Method may fail for functions with multiple roots in vicinity
        - Poor initial guess can lead to divergence or wrong root
        - Derivative must be non-zero in neighborhood of root
        - For polynomials, consider specialized polynomial root finders
        - For systems of equations, use multidimensional Newton method
    """
    # Implementation here
    pass
```

## Pull Request Process

### Before Submitting

1. **Mathematical Validation**: Verify correctness against known solutions
2. **Numerical Testing**: Test stability and accuracy
3. **Performance Benchmarks**: Ensure no performance regressions
4. **Documentation**: Update mathematical documentation and references
5. **Reference Comparisons**: Compare against established libraries

### Pull Request Template

```markdown
## Description
Brief description of mathematical changes and enhancements.

## Type of Change
- [ ] New mathematical algorithm
- [ ] Numerical method improvement
- [ ] Performance optimization
- [ ] Bug fix in mathematical computation
- [ ] Documentation update
- [ ] Test coverage improvement

## Mathematical Components Affected
- [ ] Linear Algebra Operations
- [ ] Calculus (Differentiation/Integration)
- [ ] Numerical Methods
- [ ] Optimization Algorithms
- [ ] Symbolic Mathematics
- [ ] Special Functions

## Mathematical Validation
- [ ] Verified against analytical solutions
- [ ] Compared with reference implementations (SciPy, NumPy, etc.)
- [ ] Tested numerical stability with ill-conditioned problems
- [ ] Validated error estimates and convergence analysis
- [ ] Property-based testing completed

## Performance Impact
- [ ] Benchmarked against existing implementation
- [ ] No performance regression
- [ ] Performance improvement measured
- [ ] Memory usage optimized
- [ ] Complexity analysis provided

## Testing
- [ ] Unit tests for mathematical correctness
- [ ] Property-based tests for mathematical properties
- [ ] Edge case testing (special values, boundary conditions)
- [ ] Numerical stability tests
- [ ] Performance benchmarks

## Documentation
- [ ] Mathematical background documented
- [ ] Algorithm description included
- [ ] Convergence analysis provided
- [ ] References to mathematical literature
- [ ] Usage examples included

## Accuracy Analysis
- [ ] Error bounds established
- [ ] Convergence rate analyzed
- [ ] Numerical precision documented
- [ ] Conditioning analysis performed
```

## Mathematical Accuracy and Validation

### Reference Testing

```python
def validate_against_references():
    """Validate implementations against multiple reference sources."""
    
    # Test against Wolfram Alpha, MATLAB, Mathematica results
    reference_data = load_reference_solutions()
    
    for test_case in reference_data:
        our_result = compute_our_implementation(test_case.input)
        reference_result = test_case.expected_output
        
        # Check relative accuracy
        relative_error = abs(our_result - reference_result) / abs(reference_result)
        assert relative_error < test_case.tolerance
```

### Convergence Analysis

```python
def analyze_convergence_rate(method, test_function):
    """Analyze convergence rate of numerical methods."""
    
    errors = []
    for precision in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        result = method(test_function, tolerance=precision)
        error = abs(result.value - test_function.exact_solution)
        errors.append(error)
    
    # Analyze convergence order
    convergence_order = estimate_convergence_order(errors)
    
    # Verify expected theoretical convergence rate
    assert abs(convergence_order - method.theoretical_order) < 0.1
```

## Community

### Communication Channels

- **Issues**: GitHub Issues for mathematical bugs and enhancement requests
- **Discussions**: GitHub Discussions for mathematical algorithms and theory
- **Slack**: #mathematics-dev channel for real-time mathematical discussions
- **Email**: mathematics-team@yourorg.com for mathematical consultations

### Mathematical Expertise Areas

- **Numerical Analysis**: Numerical methods, stability, and error analysis
- **Linear Algebra**: Matrix computations, decompositions, and eigenvalue problems
- **Optimization**: Constrained/unconstrained optimization and algorithmic development
- **Symbolic Mathematics**: Computer algebra systems and symbolic computation
- **Special Functions**: Mathematical special functions and their implementations

### Getting Help

1. **Mathematical Questions**: Post in GitHub Discussions with "theory" label
2. **Algorithm Issues**: Create GitHub Issues with mathematical details
3. **Performance Problems**: Include benchmarks and complexity analysis
4. **Numerical Stability**: Provide test cases demonstrating instability

### Code Review Guidelines

- **Mathematical Correctness**: Verify algorithms against mathematical literature
- **Numerical Stability**: Check handling of edge cases and ill-conditioned problems
- **Performance**: Ensure optimal algorithmic complexity
- **Documentation**: Require comprehensive mathematical documentation
- **Testing**: Mandate rigorous mathematical validation testing

Thank you for contributing to the Mathematics package! Your contributions help advance the mathematical foundations of the entire Pynomaly platform.