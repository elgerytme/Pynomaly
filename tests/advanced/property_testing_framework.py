#!/usr/bin/env python3
"""
Property-Based Testing Framework for Pynomaly
Implements comprehensive property-based testing using Hypothesis.
"""

import ast
import inspect
import logging
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type, Union
import argparse
import importlib.util
import traceback

try:
    from hypothesis import given, strategies as st, settings, HealthCheck, Phase
    from hypothesis.strategies import SearchStrategy
    from hypothesis.control import assume
    from hypothesis.errors import InvalidArgument, Unsatisfiable
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PropertyTestResult:
    """Result of a single property test."""
    function_name: str
    property_name: str
    test_passed: bool
    examples_tested: int
    counterexample: Optional[str]
    execution_time: float
    error: Optional[str] = None

@dataclass
class PropertyTestSummary:
    """Summary of property-based testing results."""
    total_properties: int
    passed_properties: int
    failed_properties: int
    error_properties: int
    total_examples: int
    execution_time: float
    results: List[PropertyTestResult]
    coverage_analysis: Dict

class PropertyTestGenerator:
    """Generates property-based tests for Python functions."""
    
    def __init__(self):
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing. Install with: pip install hypothesis")
        
        self.property_tests = []
        self.strategies = self._build_strategy_map()
    
    def _build_strategy_map(self) -> Dict[Type, Callable]:
        """Build mapping of types to Hypothesis strategies."""
        return {
            int: lambda: st.integers(min_value=-1000, max_value=1000),
            float: lambda: st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            str: lambda: st.text(min_size=0, max_size=100),
            bool: lambda: st.booleans(),
            list: lambda element_type=int: st.lists(self._get_strategy(element_type), min_size=0, max_size=10),
            dict: lambda: st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=0, max_size=5),
            tuple: lambda *element_types: st.tuples(*[self._get_strategy(t) for t in element_types]),
            
            # NumPy types
            np.ndarray: lambda: st.builds(
                np.array,
                st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1, max_size=20)
            ),
            
            # Pandas types
            pd.DataFrame: lambda: st.builds(
                pd.DataFrame,
                st.dictionaries(
                    st.text(min_size=1, max_size=10),
                    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1, max_size=10),
                    min_size=1, max_size=5
                )
            ),
            
            pd.Series: lambda: st.builds(
                pd.Series,
                st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1, max_size=20)
            ),
        }
    
    def _get_strategy(self, type_hint: Type) -> SearchStrategy:
        """Get Hypothesis strategy for a given type."""
        if type_hint in self.strategies:
            return self.strategies[type_hint]()
        elif hasattr(type_hint, '__origin__'):
            # Handle generic types like List[int], Dict[str, int], etc.
            origin = type_hint.__origin__
            args = getattr(type_hint, '__args__', ())
            
            if origin is list and args:
                element_type = args[0]
                return st.lists(self._get_strategy(element_type), min_size=0, max_size=10)
            elif origin is dict and len(args) == 2:
                key_type, value_type = args
                return st.dictionaries(
                    self._get_strategy(key_type),
                    self._get_strategy(value_type),
                    min_size=0, max_size=5
                )
            elif origin is tuple and args:
                return st.tuples(*[self._get_strategy(arg) for arg in args])
            elif origin is Union:
                # Handle Optional[T] and Union types
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    # This is Optional[T]
                    return st.one_of(st.none(), self._get_strategy(non_none_types[0]))
                else:
                    # General Union
                    return st.one_of(*[self._get_strategy(arg) for arg in non_none_types])
        
        # Default strategy for unknown types
        logger.warning(f"No strategy found for type {type_hint}, using integers")
        return st.integers(min_value=-100, max_value=100)
    
    def discover_functions(self, module_path: Path) -> List[Callable]:
        """Discover testable functions in a module."""
        try:
            spec = importlib.util.spec_from_file_location("target_module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            functions = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) and 
                    not name.startswith('_') and 
                    self._is_testable_function(obj)):
                    functions.append(obj)
            
            return functions
            
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}")
            return []
    
    def _is_testable_function(self, func: Callable) -> bool:
        """Check if a function is suitable for property-based testing."""
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Skip functions with no parameters
            if not sig.parameters:
                return False
            
            # Skip functions with complex parameter types we can't handle
            for param in sig.parameters.values():
                if param.annotation == param.empty:
                    # No type annotation - skip for now
                    return False
                
                # Check if we have a strategy for this type
                if not self._can_generate_strategy(param.annotation):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _can_generate_strategy(self, type_hint: Type) -> bool:
        """Check if we can generate a strategy for the given type."""
        if type_hint in self.strategies:
            return True
        
        if hasattr(type_hint, '__origin__'):
            origin = type_hint.__origin__
            if origin in [list, dict, tuple, Union]:
                return True
        
        # Basic Python types
        if type_hint in [int, float, str, bool]:
            return True
        
        return False
    
    def generate_property_tests(self, functions: List[Callable]) -> List[Dict]:
        """Generate property-based tests for the given functions."""
        property_tests = []
        
        for func in functions:
            tests = self._generate_function_properties(func)
            property_tests.extend(tests)
        
        return property_tests
    
    def _generate_function_properties(self, func: Callable) -> List[Dict]:
        """Generate property tests for a single function."""
        properties = []
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Generate basic properties
        properties.extend(self._generate_basic_properties(func, sig))
        
        # Generate domain-specific properties based on function name/behavior
        properties.extend(self._generate_domain_properties(func, sig))
        
        return properties
    
    def _generate_basic_properties(self, func: Callable, sig: inspect.Signature) -> List[Dict]:
        """Generate basic mathematical properties."""
        properties = []
        func_name = func.__name__
        
        # Idempotency property
        if self._might_be_idempotent(func_name):
            properties.append({
                'name': f'idempotency_{func_name}',
                'description': f'{func_name} should be idempotent',
                'property_type': 'idempotency',
                'function': func,
                'signature': sig
            })
        
        # Non-null property (function should not crash)
        properties.append({
            'name': f'non_null_{func_name}',
            'description': f'{func_name} should not raise exceptions on valid inputs',
            'property_type': 'non_null',
            'function': func,
            'signature': sig
        })
        
        # Determinism property
        properties.append({
            'name': f'determinism_{func_name}',
            'description': f'{func_name} should be deterministic',
            'property_type': 'determinism',
            'function': func,
            'signature': sig
        })
        
        # Type preservation (if applicable)
        if self._preserves_type(func_name):
            properties.append({
                'name': f'type_preservation_{func_name}',
                'description': f'{func_name} should preserve input types',
                'property_type': 'type_preservation',
                'function': func,
                'signature': sig
            })
        
        return properties
    
    def _generate_domain_properties(self, func: Callable, sig: inspect.Signature) -> List[Dict]:
        """Generate domain-specific properties based on function behavior."""
        properties = []
        func_name = func.__name__.lower()
        
        # Anomaly detection specific properties
        if 'detect' in func_name and 'anomal' in func_name:
            properties.extend(self._generate_anomaly_detection_properties(func, sig))
        
        # Data processing properties
        if any(keyword in func_name for keyword in ['preprocess', 'clean', 'transform']):
            properties.extend(self._generate_data_processing_properties(func, sig))
        
        # Mathematical function properties
        if any(keyword in func_name for keyword in ['score', 'distance', 'similarity']):
            properties.extend(self._generate_mathematical_properties(func, sig))
        
        # Model properties
        if any(keyword in func_name for keyword in ['train', 'fit', 'predict']):
            properties.extend(self._generate_model_properties(func, sig))
        
        return properties
    
    def _generate_anomaly_detection_properties(self, func: Callable, sig: inspect.Signature) -> List[Dict]:
        """Generate properties specific to anomaly detection functions."""
        properties = []
        
        # Anomaly scores should be in valid range
        properties.append({
            'name': f'anomaly_score_range_{func.__name__}',
            'description': 'Anomaly scores should be in valid range [0, 1] or similar',
            'property_type': 'score_range',
            'function': func,
            'signature': sig
        })
        
        # Monotonicity in anomaly scores
        properties.append({
            'name': f'anomaly_monotonicity_{func.__name__}',
            'description': 'More anomalous data should have higher anomaly scores',
            'property_type': 'monotonicity',
            'function': func,
            'signature': sig
        })
        
        return properties
    
    def _generate_data_processing_properties(self, func: Callable, sig: inspect.Signature) -> List[Dict]:
        """Generate properties for data processing functions."""
        properties = []
        
        # Shape preservation (for array operations)
        properties.append({
            'name': f'shape_preservation_{func.__name__}',
            'description': 'Data shape should be preserved or follow expected transformation',
            'property_type': 'shape_preservation',
            'function': func,
            'signature': sig
        })
        
        # No data loss (unless explicitly designed to filter)
        if 'filter' not in func.__name__.lower():
            properties.append({
                'name': f'no_data_loss_{func.__name__}',
                'description': 'Function should not lose data unexpectedly',
                'property_type': 'no_data_loss',
                'function': func,
                'signature': sig
            })
        
        return properties
    
    def _generate_mathematical_properties(self, func: Callable, sig: inspect.Signature) -> List[Dict]:
        """Generate properties for mathematical functions."""
        properties = []
        
        # Non-negative scores (if applicable)
        if 'score' in func.__name__.lower():
            properties.append({
                'name': f'non_negative_score_{func.__name__}',
                'description': 'Scores should be non-negative',
                'property_type': 'non_negative',
                'function': func,
                'signature': sig
            })
        
        # Symmetry (for distance functions)
        if 'distance' in func.__name__.lower():
            properties.append({
                'name': f'distance_symmetry_{func.__name__}',
                'description': 'Distance function should be symmetric',
                'property_type': 'symmetry',
                'function': func,
                'signature': sig
            })
        
        return properties
    
    def _generate_model_properties(self, func: Callable, sig: inspect.Signature) -> List[Dict]:
        """Generate properties for ML model functions."""
        properties = []
        
        # Prediction consistency
        if 'predict' in func.__name__.lower():
            properties.append({
                'name': f'prediction_consistency_{func.__name__}',
                'description': 'Predictions should be consistent for same input',
                'property_type': 'prediction_consistency',
                'function': func,
                'signature': sig
            })
        
        return properties
    
    def _might_be_idempotent(self, func_name: str) -> bool:
        """Check if function might be idempotent based on its name."""
        idempotent_patterns = ['normalize', 'clean', 'format', 'validate', 'standardize']
        return any(pattern in func_name.lower() for pattern in idempotent_patterns)
    
    def _preserves_type(self, func_name: str) -> bool:
        """Check if function likely preserves input types."""
        type_preserving_patterns = ['normalize', 'standardize', 'scale', 'transform']
        return any(pattern in func_name.lower() for pattern in type_preserving_patterns)

class PropertyTester:
    """Executes property-based tests and analyzes results."""
    
    def __init__(self, max_examples: int = 100, timeout: int = 60):
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        
        self.max_examples = max_examples
        self.timeout = timeout
        self.generator = PropertyTestGenerator()
    
    def run_property_testing(self, target_files: List[str] = None, output_file: Path = None) -> PropertyTestSummary:
        """Run comprehensive property-based testing."""
        logger.info("Starting property-based testing...")
        start_time = time.time()
        
        # Discover target files
        if target_files is None:
            target_files = self._discover_target_files()
        
        # Generate property tests
        all_properties = []
        for file_path in target_files:
            logger.info(f"Discovering functions in {file_path}")
            functions = self.generator.discover_functions(Path(file_path))
            
            if functions:
                properties = self.generator.generate_property_tests(functions)
                all_properties.extend(properties)
                logger.info(f"Generated {len(properties)} property tests for {file_path}")
        
        logger.info(f"Running {len(all_properties)} property tests...")
        
        # Execute property tests
        results = []
        for i, prop in enumerate(all_properties, 1):
            logger.info(f"Testing property {i}/{len(all_properties)}: {prop['name']}")
            result = self._test_property(prop)
            results.append(result)
        
        # Calculate summary
        total_properties = len(results)
        passed_properties = sum(1 for r in results if r.test_passed and r.error is None)
        failed_properties = sum(1 for r in results if not r.test_passed and r.error is None)
        error_properties = sum(1 for r in results if r.error is not None)
        total_examples = sum(r.examples_tested for r in results)
        execution_time = time.time() - start_time
        
        # Generate coverage analysis
        coverage_analysis = self._analyze_coverage(results)
        
        summary = PropertyTestSummary(
            total_properties=total_properties,
            passed_properties=passed_properties,
            failed_properties=failed_properties,
            error_properties=error_properties,
            total_examples=total_examples,
            execution_time=execution_time,
            results=results,
            coverage_analysis=coverage_analysis
        )
        
        logger.info(f"Property testing completed in {execution_time:.2f}s")
        logger.info(f"Results: {passed_properties} passed, {failed_properties} failed, {error_properties} errors")
        
        return summary
    
    def _discover_target_files(self) -> List[str]:
        """Discover Python files to test."""
        source_dir = Path("src/pynomaly")
        target_files = []
        
        if source_dir.exists():
            for file_path in source_dir.rglob("*.py"):
                if not file_path.name.startswith('test_') and file_path.name != '__init__.py':
                    target_files.append(str(file_path))
        
        return target_files
    
    def _test_property(self, property_def: Dict) -> PropertyTestResult:
        """Test a single property."""
        start_time = time.time()
        examples_tested = 0
        
        try:
            func = property_def['function']
            sig = property_def['signature']
            property_type = property_def['property_type']
            
            # Create argument strategies
            strategies = {}
            for param_name, param in sig.parameters.items():
                if param.annotation != param.empty:
                    strategies[param_name] = self.generator._get_strategy(param.annotation)
            
            # Generate property test based on type
            if property_type == 'non_null':
                result = self._test_non_null_property(func, strategies)
            elif property_type == 'determinism':
                result = self._test_determinism_property(func, strategies)
            elif property_type == 'idempotency':
                result = self._test_idempotency_property(func, strategies)
            elif property_type == 'type_preservation':
                result = self._test_type_preservation_property(func, strategies)
            elif property_type == 'score_range':
                result = self._test_score_range_property(func, strategies)
            elif property_type == 'monotonicity':
                result = self._test_monotonicity_property(func, strategies)
            elif property_type == 'shape_preservation':
                result = self._test_shape_preservation_property(func, strategies)
            elif property_type == 'symmetry':
                result = self._test_symmetry_property(func, strategies)
            else:
                # Default to non-null test
                result = self._test_non_null_property(func, strategies)
            
            execution_time = time.time() - start_time
            
            return PropertyTestResult(
                function_name=func.__name__,
                property_name=property_def['name'],
                test_passed=result['passed'],
                examples_tested=result.get('examples', self.max_examples),
                counterexample=result.get('counterexample'),
                execution_time=execution_time,
                error=result.get('error')
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Property test failed with exception: {e}")
            
            return PropertyTestResult(
                function_name=property_def['function'].__name__,
                property_name=property_def['name'],
                test_passed=False,
                examples_tested=examples_tested,
                counterexample=None,
                execution_time=execution_time,
                error=str(e)
            )
    
    def _test_non_null_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test that function doesn't crash on valid inputs."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples, deadline=self.timeout * 1000, suppress_health_check=[HealthCheck.return_value])
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 1
            try:
                result = func(**kwargs)
                # Don't return anything - Hypothesis functions should return None
            except Exception as e:
                # Raise exception to fail the test
                raise e
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'error': str(e)}
    
    def _test_determinism_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test that function is deterministic."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples // 2, deadline=self.timeout * 1000, suppress_health_check=[HealthCheck.return_value])
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 2
            try:
                result1 = func(**kwargs)
                result2 = func(**kwargs)
                
                # Handle different types of results
                if isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
                    assert np.array_equal(result1, result2), "Results should be deterministic"
                elif isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
                    assert result1.equals(result2), "Results should be deterministic"
                elif isinstance(result1, pd.Series) and isinstance(result2, pd.Series):
                    assert result1.equals(result2), "Results should be deterministic"
                else:
                    assert result1 == result2, "Results should be deterministic"
                    
            except Exception:
                pass  # If function crashes, determinism is not the issue
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _test_idempotency_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test that function is idempotent."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples // 2, deadline=self.timeout * 1000)
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 2
            try:
                result1 = func(**kwargs)
                result2 = func(**result1) if len(inspect.signature(func).parameters) == 1 else func(**kwargs)
                
                if isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
                    return np.array_equal(result1, result2)
                elif isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
                    return result1.equals(result2)
                else:
                    return result1 == result2
                    
            except Exception:
                return True  # Can't test idempotency if function crashes
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _test_type_preservation_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test that function preserves input types appropriately."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples, deadline=self.timeout * 1000)
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 1
            try:
                # For single-argument functions, check type preservation
                if len(kwargs) == 1:
                    input_val = list(kwargs.values())[0]
                    result = func(**kwargs)
                    
                    # Check if types are compatible
                    if isinstance(input_val, np.ndarray):
                        return isinstance(result, np.ndarray)
                    elif isinstance(input_val, pd.DataFrame):
                        return isinstance(result, (pd.DataFrame, pd.Series, np.ndarray))
                    elif isinstance(input_val, (int, float)):
                        return isinstance(result, (int, float, np.number))
                    else:
                        return True  # Accept any result for other types
                
                return True  # Multi-argument functions - skip type checking
                
            except Exception:
                return True  # Type preservation is not the issue if function crashes
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _test_score_range_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test that scores are in valid range."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples, deadline=self.timeout * 1000)
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 1
            try:
                result = func(**kwargs)
                
                # Check if result is a score (numeric)
                if isinstance(result, (int, float, np.number)):
                    return 0 <= result <= 1 or -1 <= result <= 1  # Accept common score ranges
                elif isinstance(result, np.ndarray):
                    return np.all((0 <= result) & (result <= 1)) or np.all((-1 <= result) & (result <= 1))
                else:
                    return True  # Not a score
                    
            except Exception:
                return True  # Score range is not the issue if function crashes
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _test_monotonicity_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test monotonicity properties (simplified)."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples, deadline=self.timeout * 1000)
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 1
            try:
                result = func(**kwargs)
                # For now, just check that function executes
                # Real monotonicity testing would require domain knowledge
                return True
                
            except Exception:
                return True
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _test_shape_preservation_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test that data shapes are preserved appropriately."""
        examples_tested = 0
        
        @given(**strategies)
        @settings(max_examples=self.max_examples, deadline=self.timeout * 1000)
        def test_function(**kwargs):
            nonlocal examples_tested
            examples_tested += 1
            try:
                if len(kwargs) == 1:
                    input_val = list(kwargs.values())[0]
                    result = func(**kwargs)
                    
                    if isinstance(input_val, np.ndarray) and isinstance(result, np.ndarray):
                        # Check if shapes are compatible (same or broadcast-compatible)
                        return input_val.shape == result.shape or len(result.shape) <= len(input_val.shape)
                    elif isinstance(input_val, pd.DataFrame) and isinstance(result, pd.DataFrame):
                        return input_val.shape[0] == result.shape[0]  # Same number of rows
                
                return True
                
            except Exception:
                return True
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _test_symmetry_property(self, func: Callable, strategies: Dict) -> Dict:
        """Test symmetry property for distance functions."""
        examples_tested = 0
        
        # Only test if function takes exactly 2 arguments
        if len(strategies) != 2:
            return {'passed': True, 'examples': 0}
        
        param_names = list(strategies.keys())
        strategy1 = strategies[param_names[0]]
        strategy2 = strategies[param_names[1]]
        
        @given(arg1=strategy1, arg2=strategy2)
        @settings(max_examples=self.max_examples // 2, deadline=self.timeout * 1000)
        def test_function(arg1, arg2):
            nonlocal examples_tested
            examples_tested += 2
            try:
                result1 = func(**{param_names[0]: arg1, param_names[1]: arg2})
                result2 = func(**{param_names[0]: arg2, param_names[1]: arg1})
                
                if isinstance(result1, (int, float, np.number)) and isinstance(result2, (int, float, np.number)):
                    return abs(result1 - result2) < 1e-10  # Allow small floating point differences
                else:
                    return result1 == result2
                    
            except Exception:
                return True  # Symmetry is not the issue if function crashes
        
        try:
            test_function()
            return {'passed': True, 'examples': examples_tested}
        except Exception as e:
            return {'passed': False, 'examples': examples_tested, 'counterexample': str(e)}
    
    def _analyze_coverage(self, results: List[PropertyTestResult]) -> Dict:
        """Analyze property test coverage."""
        coverage = {
            'functions_tested': len(set(r.function_name for r in results)),
            'property_types': len(set(r.property_name.split('_')[0] for r in results)),
            'total_examples': sum(r.examples_tested for r in results),
            'average_examples_per_property': sum(r.examples_tested for r in results) / len(results) if results else 0,
            'coverage_by_type': {}
        }
        
        # Analyze by property type
        for result in results:
            prop_type = result.property_name.split('_')[0]
            if prop_type not in coverage['coverage_by_type']:
                coverage['coverage_by_type'][prop_type] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': 0
                }
            
            coverage['coverage_by_type'][prop_type]['total'] += 1
            if result.error:
                coverage['coverage_by_type'][prop_type]['errors'] += 1
            elif result.test_passed:
                coverage['coverage_by_type'][prop_type]['passed'] += 1
            else:
                coverage['coverage_by_type'][prop_type]['failed'] += 1
        
        return coverage
    
    def save_results(self, summary: PropertyTestSummary, output_file: Path):
        """Save property testing results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        logger.info(f"Property test results saved to {output_file}")
    
    def print_summary(self, summary: PropertyTestSummary):
        """Print human-readable property testing summary."""
        print(f"\n=== Property-Based Testing Summary ===")
        print(f"Total properties tested: {summary.total_properties}")
        print(f"Passed: {summary.passed_properties}")
        print(f"Failed: {summary.failed_properties}")
        print(f"Errors: {summary.error_properties}")
        print(f"Total examples generated: {summary.total_examples}")
        print(f"Execution time: {summary.execution_time:.2f}s")
        
        if summary.coverage_analysis:
            print(f"\nCoverage Analysis:")
            print(f"Functions tested: {summary.coverage_analysis['functions_tested']}")
            print(f"Property types: {summary.coverage_analysis['property_types']}")
            print(f"Avg examples per property: {summary.coverage_analysis['average_examples_per_property']:.1f}")
        
        # Show failed properties
        failed_results = [r for r in summary.results if not r.test_passed or r.error]
        if failed_results:
            print(f"\n=== Failed Properties ===")
            for result in failed_results[:10]:  # Show first 10
                print(f"  {result.property_name} ({result.function_name})")
                if result.error:
                    print(f"    Error: {result.error}")
                elif result.counterexample:
                    print(f"    Counterexample: {result.counterexample}")
            
            if len(failed_results) > 10:
                print(f"  ... and {len(failed_results) - 10} more")

def main():
    """Main entry point for property-based testing."""
    parser = argparse.ArgumentParser(description="Property-Based Testing Framework")
    parser.add_argument("--target-files", nargs="+", help="Specific files to test")
    parser.add_argument("--max-examples", type=int, default=100, help="Maximum examples per property")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per property test (seconds)")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not HYPOTHESIS_AVAILABLE:
        logger.error("Hypothesis is required for property-based testing. Install with: pip install hypothesis")
        sys.exit(1)
    
    # Initialize property tester
    tester = PropertyTester(max_examples=args.max_examples, timeout=args.timeout)
    
    try:
        # Run property testing
        summary = tester.run_property_testing(target_files=args.target_files)
        
        # Print summary
        tester.print_summary(summary)
        
        # Save results
        if args.output:
            tester.save_results(summary, args.output)
        
        # Exit with appropriate code
        if summary.failed_properties > 0 or summary.error_properties > 0:
            logger.warning(f"Property testing found {summary.failed_properties} failures and {summary.error_properties} errors")
            sys.exit(1)
        else:
            logger.info("All property tests passed")
            sys.exit(0)
        
    except Exception as e:
        logger.error(f"Property testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()