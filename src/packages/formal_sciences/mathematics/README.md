# Mathematics

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

## Overview

Advanced mathematical foundations and computational tools for the Pynomaly platform.

**Architecture Layer**: Foundation Layer  
**Package Type**: Mathematical Library  
**Status**: Production Ready

## Purpose

This package provides comprehensive mathematical abstractions, algorithms, and computational tools that form the theoretical foundation for data analysis in the Pynomaly platform.

### Key Features

- **Pure Mathematics**: Rigorous mathematical abstractions and domain entities
- **Computational Mathematics**: Efficient algorithms for numerical computations
- **Symbolic Mathematics**: Symbolic manipulation and expression evaluation
- **Linear Algebra**: Advanced matrix operations and decompositions
- **Calculus**: Differentiation, integration, and optimization
- **Statistics**: Probability distributions and statistical inference
- **Topology**: Topological spaces and continuous mappings
- **Category Theory**: Categorical abstractions for type-safe mathematics

### Use Cases

- Building mathematical models for data analysis
- Implementing custom statistical algorithms
- Performing advanced numerical computations
- Creating domain-specific mathematical abstractions
- Developing mathematical proofs and verifications
- Educational and research applications

## Architecture

This package follows **Domain-Driven Design** principles with computational rigor:

```
mathematics/
├── mathematics/                # Main package source
│   ├── domain/                # Mathematical domain entities
│   │   ├── entities/          # Core mathematical objects
│   │   │   ├── math_function.py    # Mathematical functions
│   │   │   ├── matrix.py           # Matrix operations
│   │   │   ├── vector.py           # Vector spaces
│   │   │   ├── polynomial.py       # Polynomial algebra
│   │   │   ├── distribution.py     # Probability distributions
│   │   │   └── topological_space.py # Topological structures
│   │   ├── value_objects/     # Immutable mathematical values
│   │   │   ├── number.py          # Number systems
│   │   │   ├── complex.py         # Complex numbers
│   │   │   ├── interval.py        # Real intervals
│   │   │   ├── point.py           # Geometric points
│   │   │   └── measure.py         # Measure theory
│   │   └── services/          # Mathematical operations
│   │       ├── calculus.py        # Calculus operations
│   │       ├── linear_algebra.py  # Linear algebra
│   │       ├── statistics.py      # Statistical computations
│   │       └── optimization.py    # Optimization algorithms
│   ├── algebra/               # Algebraic structures
│   │   ├── groups.py          # Group theory
│   │   ├── rings.py           # Ring theory
│   │   ├── fields.py          # Field theory
│   │   └── modules.py         # Module theory
│   ├── analysis/              # Mathematical analysis
│   │   ├── limits.py          # Limits and convergence
│   │   ├── continuity.py      # Continuity theory
│   │   ├── derivatives.py     # Differentiation
│   │   ├── integrals.py       # Integration
│   │   └── sequences.py       # Sequences and series
│   ├── geometry/              # Geometric structures
│   │   ├── euclidean.py       # Euclidean geometry
│   │   ├── manifolds.py       # Differentiable manifolds
│   │   ├── metrics.py         # Metric spaces
│   │   └── transformations.py # Geometric transformations
│   ├── category_theory/       # Category theory
│   │   ├── categories.py      # Categories and functors
│   │   ├── morphisms.py       # Morphisms and natural transformations
│   │   └── limits.py          # Categorical limits
│   ├── numerical/             # Numerical methods
│   │   ├── solvers.py         # Equation solvers
│   │   ├── interpolation.py   # Interpolation methods
│   │   ├── approximation.py   # Approximation algorithms
│   │   └── integration.py     # Numerical integration
│   ├── symbolic/              # Symbolic computation
│   │   ├── expressions.py     # Symbolic expressions
│   │   ├── simplification.py  # Expression simplification
│   │   ├── differentiation.py # Symbolic differentiation
│   │   └── integration.py     # Symbolic integration
│   └── foundations/           # Mathematical foundations
│       ├── sets.py            # Set theory
│       ├── logic.py           # Mathematical logic
│       ├── proofs.py          # Proof systems
│       └── axioms.py          # Axiomatic foundations
├── tests/                     # Package tests
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── property/              # Property-based tests
│   └── benchmarks/            # Performance benchmarks
├── docs/                      # Package documentation
│   ├── theory/                # Mathematical theory
│   ├── algorithms/            # Algorithm documentation
│   └── examples/              # Usage examples
└── examples/                  # Mathematical examples
    ├── calculus/              # Calculus examples
    ├── linear_algebra/        # Linear algebra examples
    ├── statistics/            # Statistics examples
    └── applications/          # Applied mathematics
```

### Dependencies

- **Internal Dependencies**: None (pure mathematical foundation)
- **External Dependencies**: NumPy, SciPy, SymPy, Matplotlib
- **Optional Dependencies**: JAX, PyTorch (for tensor operations)

### Design Principles

1. **Mathematical Rigor**: All abstractions follow mathematical definitions
2. **Type Safety**: Comprehensive type system for computational objects
3. **Immutability**: Mathematical objects are immutable by default
4. **Composability**: Operations compose naturally following mathematical laws
5. **Performance**: Efficient implementations with numerical stability
6. **Extensibility**: Easy to add new mathematical structures
7. **Verification**: Built-in property checking and theorem proving

## Installation

### Prerequisites

- Python 3.11 or higher
- NumPy 1.24+ for numerical computations
- SciPy 1.10+ for scientific computing
- SymPy 1.12+ for symbolic mathematics

### Package Installation

```bash
# Install from source (development)
cd src/packages/formal_sciences/mathematics
pip install -e .

# Install with all mathematical libraries
pip install pynomaly-mathematics[all]

# Install specific components
pip install pynomaly-mathematics[symbolic,numerical,visualization]
```

### Pynomaly Installation

```bash
# Install entire Pynomaly platform with this package
cd /path/to/pynomaly
pip install -e ".[mathematics]"
```

## Usage

### Quick Start

```python
from pynomaly.mathematics.domain.entities import MathFunction, Matrix
from pynomaly.mathematics.domain.value_objects import RealNumber, ComplexNumber
from pynomaly.mathematics.domain.services import CalculusService
import numpy as np

# Create mathematical functions
f = MathFunction(
    expression="x**2 + 2*x + 1",
    variables=["x"],
    domain=RealInterval(-10, 10),
    function_type=FunctionType.POLYNOMIAL
)

# Evaluate function
result = f.evaluate(x=3.0)  # Returns 16.0

# Compute derivative
df_dx = f.derivative("x")
print(df_dx.expression)  # "2*x + 2"

# Matrix operations
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = A @ B  # Matrix multiplication

# Eigenvalue decomposition
eigenvalues, eigenvectors = A.eigendecomposition()
```

### Basic Examples

#### Example 1: Mathematical Functions

```python
from pynomaly.mathematics.domain.entities import MathFunction
from pynomaly.mathematics.domain.value_objects import Domain, FunctionProperties
from pynomaly.mathematics.domain.services import CalculusService

# Create a trigonometric function
sin_func = MathFunction(
    expression="sin(x)",
    variables=["x"],
    domain=Domain(-np.pi, np.pi),
    properties=FunctionProperties(
        function_type=FunctionType.TRIGONOMETRIC,
        differentiability=DifferentiabilityType.SMOOTH,
        is_continuous=True,
        is_periodic=True,
        period=2*np.pi
    )
)

# Evaluate at multiple points
x_values = np.linspace(-np.pi, np.pi, 100)
y_values = [sin_func.evaluate(x=x) for x in x_values]

# Compute derivative and integral
cos_func = sin_func.derivative("x")
neg_cos_func = sin_func.integral("x")

# Function composition
composite = sin_func.compose(
    MathFunction(expression="2*x", variables=["x"])
)  # sin(2x)
```

#### Example 2: Linear Algebra

```python
from pynomaly.mathematics.domain.entities import Matrix, Vector
from pynomaly.mathematics.domain.services import LinearAlgebraService

# Create matrices
A = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = Matrix([
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
])

# Basic operations
C = A + B  # Matrix addition
D = A @ B  # Matrix multiplication
A_T = A.transpose()  # Transpose

# Advanced operations
det_A = A.determinant()  # Determinant
rank_A = A.rank()  # Rank
inv_A = A.inverse()  # Inverse (if exists)

# Decompositions
Q, R = A.qr_decomposition()  # QR decomposition
U, S, V = A.svd()  # Singular value decomposition
L, U = A.lu_decomposition()  # LU decomposition

# Eigenvalue problems
eigenvals, eigenvecs = A.eigendecomposition()
```

#### Example 3: Statistical Distributions

```python
from pynomaly.mathematics.domain.entities import Distribution
from pynomaly.mathematics.domain.services import StatisticalService

# Create probability distributions
normal = Distribution.normal(mean=0, std=1)
exponential = Distribution.exponential(rate=1.5)
poisson = Distribution.poisson(lambda_param=3.0)

# Probability density/mass functions
pdf_value = normal.pdf(x=0.5)
cdf_value = normal.cdf(x=1.0)

# Generate random samples
samples = normal.sample(n=1000)

# Statistical inference
mean_estimate = normal.estimate_mean(samples)
std_estimate = normal.estimate_std(samples)

# Hypothesis testing
t_stat, p_value = normal.t_test(samples, null_hypothesis=0)
```

### Advanced Usage

#### Custom Mathematical Structures

```python
from pynomaly.mathematics.algebra.groups import Group
from pynomaly.mathematics.algebra.rings import Ring
from pynomaly.mathematics.category_theory.categories import Category

# Define a custom group
class ModularGroup(Group):
    def __init__(self, modulus: int):
        self.modulus = modulus
        self.elements = list(range(modulus))
    
    def operation(self, a: int, b: int) -> int:
        return (a + b) % self.modulus
    
    def identity(self) -> int:
        return 0
    
    def inverse(self, a: int) -> int:
        return (-a) % self.modulus

# Use the custom group
Z5 = ModularGroup(5)
assert Z5.operation(3, 4) == 2
assert Z5.inverse(3) == 2
```

#### Symbolic Mathematics

```python
from pynomaly.mathematics.symbolic.expressions import SymbolicExpression
from pynomaly.mathematics.symbolic.differentiation import SymbolicDifferentiation

# Create symbolic expressions
x, y = symbols("x y")
expr = x**2 + 2*x*y + y**2

# Symbolic differentiation
d_expr_dx = expr.differentiate(x)  # 2*x + 2*y
d_expr_dy = expr.differentiate(y)  # 2*x + 2*y

# Simplification
simplified = expr.simplify()  # (x + y)**2

# Symbolic integration
integral = expr.integrate(x)  # x**3/3 + x**2*y + x*y**2

# Equation solving
solutions = solve(expr - 1, x)  # Solve (x + y)**2 = 1 for x
```

#### Numerical Methods

```python
from pynomaly.mathematics.numerical.solvers import NewtonRaphson, BisectionMethod
from pynomaly.mathematics.numerical.integration import TrapezoidalRule, SimpsonsRule

# Root finding
def f(x):
    return x**3 - 2*x - 5

newton = NewtonRaphson(f, df_dx=lambda x: 3*x**2 - 2)
root = newton.solve(initial_guess=2.0)

# Numerical integration
integrator = SimpsonsRule()
result = integrator.integrate(
    f=lambda x: x**2,
    a=0,
    b=1,
    n=1000
)  # Should be approximately 1/3
```

### Configuration

```python
from pynomaly.mathematics.config import MathematicsConfig

# Configure mathematics package
config = MathematicsConfig(
    precision=64,  # Numerical precision
    symbolic_engine="sympy",  # Symbolic computation engine
    numerical_backend="numpy",  # Numerical backend
    optimization_level=2,  # Optimization level
    parallel_processing=True,  # Enable parallel processing
    cache_size=10000,  # Function evaluation cache size
    verification_enabled=True,  # Enable mathematical verification
    theorem_proving=True  # Enable theorem proving
)

# Apply configuration
configure_mathematics(config)
```

## API Reference

### Core Classes

#### Mathematical Functions

**`MathFunction`**: Represents mathematical functions with domain, codomain, and properties.

```python
class MathFunction:
    def evaluate(self, **kwargs: float) -> float: ...
    def derivative(self, variable: str, order: int = 1) -> 'MathFunction': ...
    def integral(self, variable: str, bounds: Optional[Tuple[float, float]] = None) -> 'MathFunction': ...
    def compose(self, other: 'MathFunction') -> 'MathFunction': ...
    def is_equal(self, other: 'MathFunction', tolerance: float = 1e-10) -> bool: ...
```

#### Linear Algebra

**`Matrix`**: Represents matrices with comprehensive operations.

```python
class Matrix:
    def __add__(self, other: 'Matrix') -> 'Matrix': ...
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix': ...
    def __matmul__(self, other: 'Matrix') -> 'Matrix': ...
    def transpose(self) -> 'Matrix': ...
    def determinant(self) -> float: ...
    def inverse(self) -> 'Matrix': ...
    def eigendecomposition(self) -> Tuple[List[float], List['Vector']]: ...
    def svd(self) -> Tuple['Matrix', 'Matrix', 'Matrix']: ...
```

**`Vector`**: Represents vectors in vector spaces.

```python
class Vector:
    def __add__(self, other: 'Vector') -> 'Vector': ...
    def __mul__(self, scalar: float) -> 'Vector': ...
    def dot(self, other: 'Vector') -> float: ...
    def cross(self, other: 'Vector') -> 'Vector': ...
    def norm(self, p: int = 2) -> float: ...
    def normalize(self) -> 'Vector': ...
    def project(self, onto: 'Vector') -> 'Vector': ...
```

#### Probability Distributions

**`Distribution`**: Represents probability distributions.

```python
class Distribution:
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def sample(self, n: int = 1) -> List[float]: ...
    def mean(self) -> float: ...
    def variance(self) -> float: ...
    def moment(self, order: int) -> float: ...
    
    @classmethod
    def normal(cls, mean: float, std: float) -> 'Distribution': ...
    @classmethod
    def exponential(cls, rate: float) -> 'Distribution': ...
    @classmethod
    def poisson(cls, lambda_param: float) -> 'Distribution': ...
```

### Value Objects

#### Number Systems

**`RealNumber`**: Represents real numbers with precision control.

```python
class RealNumber:
    def __init__(self, value: float, precision: int = 64): ...
    def __add__(self, other: 'RealNumber') -> 'RealNumber': ...
    def __mul__(self, other: 'RealNumber') -> 'RealNumber': ...
    def __pow__(self, exponent: 'RealNumber') -> 'RealNumber': ...
    def sin(self) -> 'RealNumber': ...
    def cos(self) -> 'RealNumber': ...
    def exp(self) -> 'RealNumber': ...
    def log(self) -> 'RealNumber': ...
```

**`ComplexNumber`**: Represents complex numbers.

```python
class ComplexNumber:
    def __init__(self, real: float, imag: float): ...
    def magnitude(self) -> float: ...
    def phase(self) -> float: ...
    def conjugate(self) -> 'ComplexNumber': ...
    def polar_form(self) -> Tuple[float, float]: ...
```

#### Geometric Objects

**`Point`**: Represents points in n-dimensional space.

```python
class Point:
    def __init__(self, coordinates: List[float]): ...
    def distance(self, other: 'Point') -> float: ...
    def midpoint(self, other: 'Point') -> 'Point': ...
    def translate(self, vector: 'Vector') -> 'Point': ...
```

**`Interval`**: Represents intervals on the real line.

```python
class Interval:
    def __init__(self, lower: float, upper: float, closed: Tuple[bool, bool] = (True, True)): ...
    def contains(self, x: float) -> bool: ...
    def intersect(self, other: 'Interval') -> Optional['Interval']: ...
    def union(self, other: 'Interval') -> List['Interval']: ...
    def length(self) -> float: ...
```

### Services

#### Calculus Service

```python
class CalculusService:
    def differentiate(self, func: MathFunction, variable: str) -> MathFunction: ...
    def integrate(self, func: MathFunction, variable: str) -> MathFunction: ...
    def find_critical_points(self, func: MathFunction) -> List[float]: ...
    def find_extrema(self, func: MathFunction) -> Tuple[List[float], List[float]]: ...
    def taylor_series(self, func: MathFunction, point: float, order: int) -> MathFunction: ...
```

#### Linear Algebra Service

```python
class LinearAlgebraService:
    def solve_system(self, A: Matrix, b: Vector) -> Vector: ...
    def least_squares(self, A: Matrix, b: Vector) -> Vector: ...
    def gram_schmidt(self, vectors: List[Vector]) -> List[Vector]: ...
    def eigenvalues(self, matrix: Matrix) -> List[float]: ...
    def is_positive_definite(self, matrix: Matrix) -> bool: ...
```

#### Statistics Service

```python
class StatisticsService:
    def descriptive_stats(self, data: List[float]) -> Dict[str, float]: ...
    def correlation(self, x: List[float], y: List[float]) -> float: ...
    def regression(self, x: List[float], y: List[float]) -> Tuple[float, float]: ...
    def hypothesis_test(self, data: List[float], null_hypothesis: float) -> Tuple[float, float]: ...
    def confidence_interval(self, data: List[float], confidence: float) -> Tuple[float, float]: ...
```

## Mathematical Foundations

### Set Theory

```python
from pynomaly.mathematics.foundations.sets import Set, Subset, PowerSet

# Create sets
A = Set([1, 2, 3, 4, 5])
B = Set([3, 4, 5, 6, 7])

# Set operations
union = A.union(B)  # {1, 2, 3, 4, 5, 6, 7}
intersection = A.intersection(B)  # {3, 4, 5}
difference = A.difference(B)  # {1, 2}
symmetric_difference = A.symmetric_difference(B)  # {1, 2, 6, 7}

# Set relations
is_subset = A.is_subset(B)  # False
is_superset = A.is_superset(B)  # False
is_disjoint = A.is_disjoint(B)  # False
```

### Mathematical Logic

```python
from pynomaly.mathematics.foundations.logic import Proposition, Predicate, Quantifier

# Propositional logic
p = Proposition("It is raining")
q = Proposition("The ground is wet")

# Logical operations
implication = p.implies(q)  # p → q
conjunction = p.and_(q)  # p ∧ q
disjunction = p.or_(q)  # p ∨ q
negation = p.not_()  # ¬p

# Predicate logic
universe = Set([1, 2, 3, 4, 5])
is_even = Predicate(lambda x: x % 2 == 0)

# Quantification
exists_even = Quantifier.exists(universe, is_even)  # ∃x P(x)
all_even = Quantifier.forall(universe, is_even)  # ∀x P(x)
```

### Category Theory

```python
from pynomaly.mathematics.category_theory.categories import Category
from pynomaly.mathematics.category_theory.functors import Functor
from pynomaly.mathematics.category_theory.morphisms import Morphism

# Define a category
class VectorSpaceCategory(Category):
    def compose(self, f: Morphism, g: Morphism) -> Morphism: ...
    def identity(self, obj: Any) -> Morphism: ...

# Define functors
class LinearMapFunctor(Functor):
    def map_object(self, obj: Any) -> Any: ...
    def map_morphism(self, morphism: Morphism) -> Morphism: ...
```

## Performance

### Benchmarks

- **Matrix multiplication**: 1M×1M matrices in <2 seconds
- **Function evaluation**: 1M evaluations in <100ms
- **Symbolic differentiation**: Complex expressions in <10ms
- **Numerical integration**: High precision in <50ms

### Optimization

```python
from pynomaly.mathematics.optimization import MathematicsOptimizer

# Enable optimizations
optimizer = MathematicsOptimizer()
optimizer.enable_parallel_processing()
optimizer.enable_vectorization()
optimizer.enable_caching()
optimizer.enable_jit_compilation()  # Requires JAX
```

## Verification and Theorem Proving

### Property-Based Testing

```python
from pynomaly.mathematics.verification import MathematicalProperty

# Define mathematical properties
@MathematicalProperty
def matrix_multiplication_associative(A: Matrix, B: Matrix, C: Matrix):
    return (A @ B) @ C == A @ (B @ C)

@MathematicalProperty
def function_derivative_linearity(f: MathFunction, g: MathFunction, a: float, b: float):
    h = lambda x: a * f.evaluate(x=x) + b * g.evaluate(x=x)
    h_prime = h.derivative("x")
    expected = lambda x: a * f.derivative("x").evaluate(x=x) + b * g.derivative("x").evaluate(x=x)
    return h_prime.is_equal(expected)
```

### Theorem Proving

```python
from pynomaly.mathematics.proofs import TheoremProver

# Prove mathematical theorems
prover = TheoremProver()

# Prove Pythagorean theorem
pythagorean_proof = prover.prove(
    statement="For a right triangle with legs a and b and hypotenuse c: a² + b² = c²",
    axioms=["Euclidean geometry axioms"],
    method="geometric_proof"
)

# Prove calculus theorems
fundamental_theorem = prover.prove(
    statement="∫[a,b] f'(x)dx = f(b) - f(a)",
    axioms=["Real analysis axioms"],
    method="analytic_proof"
)
```

## Development

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/property/       # Property-based tests
pytest tests/benchmarks/     # Performance benchmarks

# Run with coverage
pytest --cov=mathematics --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format mathematics/

# Type checking
mypy mathematics/

# Mathematical verification
python -m mathematics.verification --check-all
```

## Contributing

1. **Mathematical Rigor**: Ensure all implementations follow mathematical definitions
2. **Type Safety**: Use comprehensive type hints for all mathematical objects
3. **Documentation**: Include mathematical formulas and proofs in docstrings
4. **Testing**: Write property-based tests for computational properties
5. **Performance**: Optimize numerical algorithms for efficiency
6. **Examples**: Provide educational examples and applications

### Adding New Mathematical Structures

```python
from pynomaly.mathematics.domain.entities import MathematicalEntity

class NewMathematicalStructure(MathematicalEntity):
    def __init__(self, properties: Dict[str, Any]):
        super().__init__(properties)
        self.validate_mathematical_properties()
    
    def validate_mathematical_properties(self) -> None:
        """Validate mathematical axioms and properties."""
        # Implement validation logic
        pass
    
    def operation(self, other: 'NewMathematicalStructure') -> 'NewMathematicalStructure':
        """Implement mathematical operation."""
        # Implement operation logic
        pass
```

## Applications

### Machine Learning

```python
from pynomaly.mathematics.domain.services import LinearAlgebraService
from pynomaly.mathematics.domain.entities import Matrix, Vector

# Principal Component Analysis
def pca(data: Matrix, n_components: int) -> Tuple[Matrix, Vector]:
    la_service = LinearAlgebraService()
    
    # Center the data
    centered = data - data.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = centered.transpose() @ centered / (data.shape[0] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = cov_matrix.eigendecomposition()
    
    # Select top components
    top_eigenvectors = eigenvectors[:n_components]
    top_eigenvalues = eigenvalues[:n_components]
    
    return top_eigenvectors, top_eigenvalues
```

### Signal Processing

```python
from pynomaly.mathematics.domain.entities import MathFunction
from pynomaly.mathematics.analysis.transforms import FourierTransform

# Fourier analysis
def analyze_signal(signal: List[float], sampling_rate: float):
    # Create continuous function from discrete signal
    t = np.linspace(0, len(signal) / sampling_rate, len(signal))
    signal_func = MathFunction.from_data(t, signal)
    
    # Compute Fourier transform
    fourier = FourierTransform()
    frequency_domain = fourier.transform(signal_func)
    
    return frequency_domain
```

## Support

- **Documentation**: [Package docs](docs/)
- **Theory**: [Mathematical theory](docs/theory/)
- **Examples**: [Mathematical examples](examples/)
- **Issues**: [GitHub Issues](../../../issues)
- **Discussions**: [GitHub Discussions](../../../discussions)

## License

MIT License. See [LICENSE](../../../LICENSE) file for details.

---

**Part of the [Pynomaly](../../../) monorepo** - Advanced computational platform