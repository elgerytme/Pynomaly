# Security Policy - Mathematics Package

## Overview

The Mathematics package provides comprehensive mathematical computation capabilities that form the foundation for all mathematical operations across the Pynomaly platform. Security is critical to prevent computational attacks, ensure numerical accuracy, and protect against mathematical vulnerabilities that could compromise system integrity.

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | End of Life    |
| ------- | ------------------ | -------------- |
| 3.x.x   | :white_check_mark: | -              |
| 2.9.x   | :white_check_mark: | 2025-06-01     |
| 2.8.x   | :warning:          | 2024-12-31     |
| < 2.8   | :x:                | Ended          |

## Security Model

### Mathematical Security Domains

Our security model addresses these critical areas:

**1. Computational Security**
- Expression parsing and evaluation security
- Prevention of arbitrary code execution
- Resource consumption limits and DoS protection
- Numerical stability and accuracy validation

**2. Mathematical Integrity**
- Algorithm correctness verification
- Numerical precision and error bounds
- Mathematical property validation
- Convergence and stability guarantees

**3. Input Validation**
- Mathematical expression sanitization
- Parameter bounds checking
- Data type validation and conversion
- Malicious input detection and prevention

**4. Resource Management**
- Memory usage limits for large computations
- CPU time limits for iterative algorithms
- Stack overflow prevention in recursive functions
- Parallel computation resource control

## Threat Model

### High-Risk Scenarios

**Computational Attacks**
- Malicious mathematical expressions designed to consume excessive resources
- Algorithmic complexity attacks targeting expensive operations
- Memory exhaustion through large matrix operations
- Stack overflow attacks via deep recursion

**Numerical Attacks**
- Precision loss attacks exploiting floating-point vulnerabilities
- Catastrophic cancellation manipulation
- Overflow/underflow exploitation
- NaN and infinity injection attacks

**Input Validation Bypass**
- Code injection through mathematical expressions
- Parameter manipulation to cause undefined behavior
- Type confusion attacks
- Buffer overflow in numerical routines

**Side-Channel Attacks**
- Timing attacks on mathematical operations
- Power analysis of cryptographic mathematical functions
- Cache timing attacks on algorithm implementations
- Information leakage through error messages

## Security Features

### Input Validation and Sanitization

**Expression Parser Security**
```python
import ast
import re
from typing import Any, Dict, List, Set, Union
from decimal import Decimal

class SecureMathematicalParser:
    """Secure parser for mathematical expressions."""
    
    # Allowed mathematical functions
    SAFE_FUNCTIONS = {
        'abs', 'min', 'max', 'sum', 'round',
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p',
        'sqrt', 'cbrt', 'pow', 'hypot',
        'ceil', 'floor', 'trunc', 'fmod', 'remainder',
        'gamma', 'lgamma', 'erf', 'erfc'
    }
    
    # Allowed mathematical constants
    SAFE_CONSTANTS = {
        'pi', 'e', 'tau', 'inf', 'nan'
    }
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'__\w+__',  # Dunder methods
        r'exec\s*\(',  # exec function
        r'eval\s*\(',  # eval function
        r'compile\s*\(',  # compile function
        r'import\s+',  # import statements
        r'from\s+\w+\s+import',  # from-import statements
        r'open\s*\(',  # file operations
        r'input\s*\(',  # input function
        r'getattr\s*\(',  # attribute access
        r'setattr\s*\(',  # attribute setting
        r'delattr\s*\(',  # attribute deletion
        r'globals\s*\(',  # globals access
        r'locals\s*\(',  # locals access
        r'vars\s*\(',  # vars access
        r'dir\s*\(',  # directory listing
    ]
    
    def __init__(self, max_expression_length: int = 10000):
        self.max_expression_length = max_expression_length
        self._compile_dangerous_patterns()
    
    def _compile_dangerous_patterns(self):
        """Compile dangerous patterns for efficient matching."""
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.DANGEROUS_PATTERNS
        ]
    
    def validate_expression(self, expression: str) -> str:
        """
        Validate and sanitize mathematical expression.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Sanitized expression
            
        Raises:
            SecurityError: If expression contains dangerous patterns
            ValueError: If expression is invalid
        """
        # Length check
        if len(expression) > self.max_expression_length:
            raise SecurityError(f"Expression too long: {len(expression)} > {self.max_expression_length}")
        
        # Remove comments and normalize whitespace
        expression = self._normalize_expression(expression)
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(expression)
        
        # Parse and validate AST
        try:
            tree = ast.parse(expression, mode='eval')
            self._validate_ast(tree)
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        
        return expression
    
    def _normalize_expression(self, expression: str) -> str:
        """Normalize expression by removing comments and extra whitespace."""
        # Remove Python-style comments
        expression = re.sub(r'#.*$', '', expression, flags=re.MULTILINE)
        
        # Normalize whitespace
        expression = ' '.join(expression.split())
        
        return expression.strip()
    
    def _check_dangerous_patterns(self, expression: str) -> None:
        """Check for dangerous patterns in expression."""
        for pattern in self.compiled_patterns:
            if pattern.search(expression):
                raise SecurityError(f"Dangerous pattern detected: {pattern.pattern}")
    
    def _validate_ast(self, node: ast.AST) -> None:
        """Validate AST nodes for security."""
        if isinstance(node, ast.Expression):
            self._validate_ast(node.body)
        
        elif isinstance(node, (ast.Constant, ast.Num)):
            # Constants are safe
            pass
        
        elif isinstance(node, ast.Name):
            # Check if name is in allowed constants or variables
            if node.id not in self.SAFE_CONSTANTS:
                # Variable names should only contain alphanumeric and underscore
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', node.id):
                    raise SecurityError(f"Invalid variable name: {node.id}")
        
        elif isinstance(node, ast.Call):
            # Validate function calls
            if isinstance(node.func, ast.Name):
                if node.func.id not in self.SAFE_FUNCTIONS:
                    raise SecurityError(f"Unsafe function: {node.func.id}")
            else:
                raise SecurityError("Complex function calls not allowed")
            
            # Validate arguments
            for arg in node.args:
                self._validate_ast(arg)
            for keyword in node.keywords:
                self._validate_ast(keyword.value)
        
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
            # Binary and unary operations are generally safe
            for child in ast.iter_child_nodes(node):
                self._validate_ast(child)
        
        elif isinstance(node, ast.Compare):
            # Comparison operations
            self._validate_ast(node.left)
            for comparator in node.comparators:
                self._validate_ast(comparator)
        
        elif isinstance(node, (ast.List, ast.Tuple)):
            # Collections
            for elt in node.elts:
                self._validate_ast(elt)
        
        elif isinstance(node, ast.Subscript):
            # Array/matrix indexing
            self._validate_ast(node.value)
            self._validate_ast(node.slice)
        
        else:
            raise SecurityError(f"Unsafe AST node type: {type(node).__name__}")

class SecureParameterValidator:
    """Validate mathematical function parameters."""
    
    @staticmethod
    def validate_matrix_dimensions(matrix: Any, max_size: int = 10000) -> None:
        """Validate matrix dimensions to prevent memory attacks."""
        if hasattr(matrix, 'shape'):
            total_elements = 1
            for dim in matrix.shape:
                if dim < 0:
                    raise ValueError("Matrix dimensions must be non-negative")
                total_elements *= dim
                
            if total_elements > max_size * max_size:
                raise SecurityError(f"Matrix too large: {total_elements} > {max_size * max_size}")
    
    @staticmethod
    def validate_numerical_range(
        value: Union[float, int], 
        min_val: float = -1e100, 
        max_val: float = 1e100,
        parameter_name: str = "value"
    ) -> None:
        """Validate numerical values are within safe ranges."""
        if not isinstance(value, (int, float, Decimal)):
            raise TypeError(f"{parameter_name} must be numeric")
        
        if value < min_val or value > max_val:
            raise ValueError(f"{parameter_name} out of range: {value}")
        
        # Check for NaN and infinity
        if hasattr(value, '__float__'):
            float_val = float(value)
            if math.isnan(float_val):
                raise ValueError(f"{parameter_name} cannot be NaN")
            if math.isinf(float_val) and not (min_val <= float_val <= max_val):
                raise ValueError(f"{parameter_name} cannot be infinite")
    
    @staticmethod
    def validate_iteration_count(iterations: int, max_iterations: int = 100000) -> None:
        """Validate iteration counts to prevent DoS attacks."""
        if not isinstance(iterations, int):
            raise TypeError("Iteration count must be integer")
        
        if iterations < 0:
            raise ValueError("Iteration count must be non-negative")
        
        if iterations > max_iterations:
            raise SecurityError(f"Too many iterations: {iterations} > {max_iterations}")
```

**Resource Management and DoS Prevention**
```python
import time
import threading
from contextlib import contextmanager
from typing import Optional
import psutil
import signal

class ComputationResourceManager:
    """Manage computational resources to prevent DoS attacks."""
    
    def __init__(
        self,
        max_memory_mb: int = 1000,
        max_cpu_time_seconds: int = 60,
        max_wall_time_seconds: int = 120
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time_seconds = max_cpu_time_seconds
        self.max_wall_time_seconds = max_wall_time_seconds
    
    @contextmanager
    def managed_computation(self, operation_name: str = "computation"):
        """Context manager for resource-limited computation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        timer = None
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Computation timeout: {operation_name}")
        
        try:
            # Set up timeout alarm
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.max_wall_time_seconds)
            else:
                # For Windows, use threading timer
                timer = threading.Timer(
                    self.max_wall_time_seconds,
                    lambda: (_ for _ in ()).throw(TimeoutError(f"Computation timeout: {operation_name}"))
                )
                timer.start()
            
            yield self
            
        finally:
            # Clean up timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            elif timer:
                timer.cancel()
            
            # Check resource usage
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            wall_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            if wall_time > self.max_wall_time_seconds:
                raise SecurityError(f"Wall time limit exceeded: {wall_time:.2f}s")
            
            if memory_used > self.max_memory_mb:
                raise SecurityError(f"Memory limit exceeded: {memory_used:.2f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def check_memory_limit(self) -> None:
        """Check if memory usage exceeds limits."""
        current_memory = self._get_memory_usage()
        if current_memory > self.max_memory_mb:
            raise SecurityError(f"Memory limit exceeded: {current_memory:.2f}MB")

class SecureMathematicalFunction:
    """Base class for secure mathematical functions."""
    
    def __init__(self, resource_manager: ComputationResourceManager):
        self.resource_manager = resource_manager
        self.parser = SecureMathematicalParser()
        self.validator = SecureParameterValidator()
    
    def secure_evaluate(
        self,
        expression: str,
        variables: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Any:
        """Securely evaluate mathematical expression."""
        
        with self.resource_manager.managed_computation("expression_evaluation"):
            # Validate and sanitize expression
            safe_expression = self.parser.validate_expression(expression)
            
            # Validate variables
            if variables:
                for name, value in variables.items():
                    self.validator.validate_numerical_range(value, parameter_name=name)
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                        raise SecurityError(f"Invalid variable name: {name}")
            
            # Create secure evaluation environment
            safe_globals = self._create_safe_globals()
            safe_locals = variables.copy() if variables else {}
            
            try:
                # Compile and evaluate expression
                compiled_expr = compile(safe_expression, '<secure_eval>', 'eval')
                result = eval(compiled_expr, safe_globals, safe_locals)
                
                # Validate result
                self._validate_result(result)
                
                return result
                
            except Exception as e:
                raise ComputationError(f"Evaluation failed: {e}")
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global environment for expression evaluation."""
        import math
        
        safe_globals = {
            '__builtins__': {},  # Remove all builtins
            
            # Mathematical functions
            'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'atan2': math.atan2, 'sinh': math.sinh, 'cosh': math.cosh,
            'tanh': math.tanh, 'asinh': math.asinh, 'acosh': math.acosh,
            'atanh': math.atanh, 'exp': math.exp, 'log': math.log,
            'log10': math.log10, 'log2': math.log2, 'sqrt': math.sqrt,
            'pow': pow, 'ceil': math.ceil, 'floor': math.floor,
            
            # Mathematical constants
            'pi': math.pi, 'e': math.e, 'tau': math.tau,
            'inf': math.inf, 'nan': math.nan,
        }
        
        return safe_globals
    
    def _validate_result(self, result: Any) -> None:
        """Validate computation result for safety."""
        if isinstance(result, (int, float)):
            if math.isnan(result):
                raise ComputationError("Result is NaN")
            if math.isinf(result):
                raise ComputationError("Result is infinite")
        
        elif hasattr(result, '__len__'):
            # Check collection size
            if len(result) > 1000000:  # 1M elements max
                raise SecurityError("Result collection too large")
```

### Numerical Stability and Accuracy

**Precision Control and Error Detection**
```python
import numpy as np
from decimal import Decimal, getcontext
from typing import Union, Tuple

class NumericalSecurityManager:
    """Manage numerical security and stability."""
    
    def __init__(self, default_precision: int = 15):
        self.default_precision = default_precision
        getcontext().prec = max(28, default_precision + 10)  # Extra precision for intermediate calculations
    
    def secure_matrix_operation(
        self,
        operation: str,
        *matrices,
        check_conditioning: bool = True,
        max_condition_number: float = 1e12
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform matrix operation with security checks.
        
        Args:
            operation: Matrix operation name
            matrices: Input matrices
            check_conditioning: Whether to check matrix conditioning
            max_condition_number: Maximum allowed condition number
            
        Returns:
            Result matrix and security metadata
        """
        # Validate input matrices
        for i, matrix in enumerate(matrices):
            self._validate_matrix_security(matrix, f"matrix_{i}")
        
        security_info = {}
        
        if check_conditioning and len(matrices) > 0:
            # Check conditioning of first matrix (typically coefficient matrix)
            matrix = matrices[0]
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                condition_number = np.linalg.cond(matrix)
                security_info['condition_number'] = condition_number
                
                if condition_number > max_condition_number:
                    raise NumericalSecurityError(
                        f"Matrix is ill-conditioned: cond = {condition_number:.2e}"
                    )
        
        # Perform operation with monitoring
        try:
            if operation == "multiply":
                result = self._secure_matrix_multiply(*matrices)
            elif operation == "solve":
                result = self._secure_matrix_solve(*matrices)
            elif operation == "invert":
                result = self._secure_matrix_invert(matrices[0])
            elif operation == "eigenvalues":
                result = self._secure_eigenvalues(matrices[0])
            else:
                raise ValueError(f"Unknown matrix operation: {operation}")
            
            # Validate result
            self._validate_matrix_result(result)
            
            return result, security_info
            
        except np.linalg.LinAlgError as e:
            raise NumericalSecurityError(f"Matrix operation failed: {e}")
    
    def _validate_matrix_security(self, matrix: np.ndarray, name: str) -> None:
        """Validate matrix for security issues."""
        # Check for NaN or infinity
        if np.any(np.isnan(matrix)):
            raise NumericalSecurityError(f"{name} contains NaN values")
        
        if np.any(np.isinf(matrix)):
            raise NumericalSecurityError(f"{name} contains infinite values")
        
        # Check for extremely large values
        max_abs_value = np.max(np.abs(matrix))
        if max_abs_value > 1e100:
            raise NumericalSecurityError(f"{name} contains extremely large values: {max_abs_value}")
        
        # Check for extremely small values that might cause underflow
        min_nonzero = np.min(np.abs(matrix[matrix != 0]))
        if min_nonzero < 1e-300:
            raise NumericalSecurityError(f"{name} contains extremely small values: {min_nonzero}")
    
    def _secure_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Secure matrix multiplication with overflow checking."""
        # Check dimensions
        if A.shape[1] != B.shape[0]:
            raise ValueError("Incompatible matrix dimensions for multiplication")
        
        # Estimate result magnitude to prevent overflow
        max_A = np.max(np.abs(A))
        max_B = np.max(np.abs(B))
        estimated_max_result = max_A * max_B * A.shape[1]
        
        if estimated_max_result > 1e100:
            raise NumericalSecurityError("Matrix multiplication may cause overflow")
        
        # Perform multiplication
        result = A @ B
        
        return result
    
    def _validate_matrix_result(self, result: np.ndarray) -> None:
        """Validate matrix computation result."""
        if np.any(np.isnan(result)):
            raise NumericalSecurityError("Computation result contains NaN")
        
        if np.any(np.isinf(result)):
            raise NumericalSecurityError("Computation result contains infinity")
        
        # Check for suspicious patterns that might indicate numerical instability
        if result.size > 0:
            result_range = np.ptp(result)  # Peak-to-peak range
            result_mean = np.mean(np.abs(result))
            
            if result_range > 1e6 * result_mean:
                raise NumericalSecurityError("Result shows signs of numerical instability")

class FloatingPointSecurityAnalyzer:
    """Analyze floating-point security issues."""
    
    @staticmethod
    def detect_catastrophic_cancellation(
        operand1: float, 
        operand2: float, 
        operation: str = "subtract"
    ) -> Dict[str, Any]:
        """
        Detect potential catastrophic cancellation.
        
        Args:
            operand1: First operand
            operand2: Second operand  
            operation: Operation type ("subtract" or "add")
            
        Returns:
            Analysis results
        """
        analysis = {
            'cancellation_detected': False,
            'relative_error_amplification': 1.0,
            'recommendation': None
        }
        
        if operation == "subtract" and operand1 * operand2 > 0:
            # Same sign subtraction - check for near cancellation
            if abs(operand1) > 0 and abs(operand2) > 0:
                ratio = min(abs(operand1), abs(operand2)) / max(abs(operand1), abs(operand2))
                
                if ratio > 0.9:  # Very close values
                    analysis['cancellation_detected'] = True
                    
                    # Estimate error amplification
                    result = abs(operand1 - operand2)
                    if result > 0:
                        amplification = max(abs(operand1), abs(operand2)) / result
                        analysis['relative_error_amplification'] = amplification
                        
                        if amplification > 1e6:
                            analysis['recommendation'] = "Use higher precision arithmetic"
                        elif amplification > 1000:
                            analysis['recommendation'] = "Consider alternative algorithm"
        
        return analysis
    
    @staticmethod
    def check_precision_loss(original: float, computed: float) -> Dict[str, Any]:
        """Check for precision loss in computation."""
        if original == 0:
            return {
                'precision_loss': abs(computed) > 1e-15,
                'lost_digits': 0 if computed == 0 else 15
            }
        
        relative_error = abs(computed - original) / abs(original)
        lost_digits = max(0, -math.log10(relative_error)) if relative_error > 0 else 15
        
        return {
            'precision_loss': relative_error > 1e-12,
            'relative_error': relative_error,
            'lost_digits': lost_digits,
            'significant_digits_remaining': max(0, 15 - lost_digits)
        }
```

## Security Best Practices

### Development

**Secure Mathematical Development**
- Validate all mathematical inputs and parameters
- Implement resource limits for computational operations
- Use numerically stable algorithms for all implementations
- Regular security code reviews focusing on mathematical vulnerabilities
- Automated testing for edge cases and boundary conditions

**Algorithm Security**
- Choose algorithms based on numerical stability and security properties
- Implement proper error handling for mathematical exceptions
- Use appropriate precision for different computational contexts
- Monitor for algorithmic complexity attacks
- Validate convergence and termination conditions

### Deployment

**Production Security Configuration**
- Configure appropriate resource limits for mathematical computations
- Enable comprehensive monitoring of computational performance
- Implement proper logging for mathematical operations
- Set up alerting for unusual computational patterns
- Regular security assessments of mathematical components

**Mathematical Infrastructure Security**
- Secure mathematical library dependencies
- Use verified mathematical constants and functions
- Implement proper numerical precision management
- Monitor for numerical stability issues
- Regular updates of mathematical libraries

## Vulnerability Reporting

### Reporting Process

Mathematical security vulnerabilities require special attention due to their potential for subtle but critical impact.

**1. Critical Mathematical Vulnerabilities**
- Code injection through mathematical expressions
- DoS attacks via computational complexity
- Numerical stability issues causing incorrect results
- Precision attacks affecting mathematical accuracy

**2. Contact Security Team**
- Email: math-security@yourorg.com
- PGP Key: [Mathematics security PGP key]
- Include "Mathematics Security Vulnerability" in subject

**3. Provide Detailed Information**
```
Subject: Mathematics Security Vulnerability - [Brief Description]

Vulnerability Details:
- Mathematical component: [e.g., expression parser, linear algebra, optimization]
- Vulnerability type: [e.g., code injection, DoS, numerical instability]
- Severity level: [Critical/High/Medium/Low]
- Attack vector: [How vulnerability can be exploited]
- Mathematical impact: [Effect on computational accuracy or performance]
- Reproduction steps: [Detailed mathematical test case]
- Proof of concept: [Mathematical expression or code to reproduce]
- Suggested fix: [If you have mathematical recommendations]

Mathematical Context:
- Algorithm affected: [Specific mathematical algorithm]
- Numerical precision requirements: [Expected vs actual precision]
- Performance characteristics: [Time/space complexity]
- Mathematical properties violated: [Convergence, stability, etc.]
```

### Response Timeline

**Critical Mathematical Vulnerabilities**
- **Acknowledgment**: Within 2 hours
- **Initial Assessment**: Within 6 hours
- **Mathematical Analysis**: Within 12 hours
- **Resolution Timeline**: 24-72 hours depending on algorithmic complexity

**High/Medium Severity**
- **Acknowledgment**: Within 8 hours
- **Initial Assessment**: Within 24 hours
- **Detailed Analysis**: Within 72 hours
- **Resolution Timeline**: 1-2 weeks depending on mathematical complexity

## Contact Information

**Mathematics Security Team**
- Email: math-security@yourorg.com
- Emergency Phone: [Emergency contact for critical mathematical vulnerabilities]
- PGP Key: [Mathematics security PGP key fingerprint]

**Escalation Contacts**
- Mathematical Security Lead: [Contact information]
- Numerical Analysis Expert: [Contact information]
- Mathematical Review Board: [Contact information]

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Next Review**: March 2025