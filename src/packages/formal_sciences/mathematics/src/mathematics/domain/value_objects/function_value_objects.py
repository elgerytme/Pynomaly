"""Value objects for mathematical functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum


class FunctionType(Enum):
    """Types of mathematical functions."""
    POLYNOMIAL = "polynomial"
    TRIGONOMETRIC = "trigonometric"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    RATIONAL = "rational"
    COMPOSITE = "composite"
    PIECEWISE = "piecewise"
    IMPLICIT = "implicit"
    PARAMETRIC = "parametric"
    VECTOR_VALUED = "vector_valued"


class DifferentiabilityType(Enum):
    """Function differentiability classification."""
    NOT_DIFFERENTIABLE = "not_differentiable"
    CONTINUOUS = "continuous"
    DIFFERENTIABLE = "differentiable"
    SMOOTH = "smooth"
    ANALYTIC = "analytic"


@dataclass(frozen=True)
class FunctionId:
    """Unique identifier for mathematical functions."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Domain:
    """Mathematical domain specification."""
    lower_bound: float
    upper_bound: float
    include_lower: bool = True
    include_upper: bool = True
    excluded_points: List[float] = field(default_factory=list)
    
    def contains(self, x: float) -> bool:
        """Check if value is in domain."""
        if x in self.excluded_points:
            return False
            
        if x < self.lower_bound or x > self.upper_bound:
            return False
            
        if x == self.lower_bound and not self.include_lower:
            return False
            
        if x == self.upper_bound and not self.include_upper:
            return False
            
        return True
    
    def intersect(self, other: Domain) -> Optional[Domain]:
        """Compute intersection with another domain."""
        lower = max(self.lower_bound, other.lower_bound)
        upper = min(self.upper_bound, other.upper_bound)
        
        if lower > upper:
            return None
            
        include_lower = (
            self.include_lower if lower == self.lower_bound else other.include_lower
        )
        include_upper = (
            self.include_upper if upper == self.upper_bound else other.include_upper
        )
        
        excluded = list(set(self.excluded_points + other.excluded_points))
        excluded = [x for x in excluded if lower <= x <= upper]
        
        return Domain(
            lower_bound=lower,
            upper_bound=upper,
            include_lower=include_lower,
            include_upper=include_upper,
            excluded_points=excluded
        )


@dataclass(frozen=True)
class FunctionProperties:
    """Properties of a mathematical function."""
    function_type: FunctionType
    differentiability: DifferentiabilityType
    is_continuous: bool
    is_monotonic: bool
    is_periodic: bool
    period: Optional[float] = None
    is_even: bool = False
    is_odd: bool = False
    is_bounded: bool = False
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    has_asymptotes: bool = False
    vertical_asymptotes: List[float] = field(default_factory=list)
    horizontal_asymptotes: List[float] = field(default_factory=list)
    oblique_asymptotes: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate function properties consistency."""
        if self.is_even and self.is_odd:
            return False
            
        if self.is_periodic and self.period is None:
            return False
            
        if self.is_bounded and (self.upper_bound is None or self.lower_bound is None):
            return False
            
        return True


@dataclass(frozen=True)
class FunctionMetadata:
    """Metadata for mathematical functions."""
    name: str
    description: str
    author: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    computational_complexity: str = "O(1)"
    numerical_stability: str = "stable"
    
    def add_tag(self, tag: str) -> FunctionMetadata:
        """Add a tag to the function metadata."""
        if tag not in self.tags:
            new_tags = list(self.tags) + [tag]
            return dataclass.replace(self, tags=new_tags)
        return self
    
    def add_reference(self, reference: str) -> FunctionMetadata:
        """Add a reference to the function metadata."""
        if reference not in self.references:
            new_refs = list(self.references) + [reference]
            return dataclass.replace(self, references=new_refs)
        return self


@dataclass(frozen=True)
class EvaluationCache:
    """Cache for function evaluations."""
    cache: Dict[Tuple[float, ...], float] = field(default_factory=dict)
    max_size: int = 1000
    hit_count: int = 0
    miss_count: int = 0
    
    def get(self, inputs: Tuple[float, ...]) -> Optional[float]:
        """Get cached evaluation result."""
        result = self.cache.get(inputs)
        if result is not None:
            return result
        return None
    
    def put(self, inputs: Tuple[float, ...], result: float) -> EvaluationCache:
        """Add evaluation result to cache."""
        if len(self.cache) >= self.max_size:
            # Simple LRU eviction - remove first item
            cache_copy = dict(self.cache)
            cache_copy.pop(next(iter(cache_copy)))
            cache_copy[inputs] = result
            return dataclass.replace(self, cache=cache_copy)
        else:
            cache_copy = dict(self.cache)
            cache_copy[inputs] = result
            return dataclass.replace(self, cache=cache_copy)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0