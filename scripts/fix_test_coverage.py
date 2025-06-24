#!/usr/bin/env python3
"""
Script to systematically fix test coverage issues.
This script addresses the main categories of test failures.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def fix_anomaly_score():
    """Fix AnomalyScore value object to match test expectations."""
    content = '''"""Anomaly score value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AnomalyScore:
    """Immutable value object representing an anomaly score.
    
    Attributes:
        value: The anomaly score value (higher means more anomalous)
        confidence_interval: Confidence interval for the score (optional)
        method: The scoring method used (optional)
    """
    
    value: float
    confidence_interval: Optional['ConfidenceInterval'] = None
    method: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate score after initialization."""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Score value must be numeric, got {type(self.value)}")
        
        if self.confidence_interval is not None:
            if not self.confidence_interval.contains(self.value):
                raise ValueError(
                    f"Score value ({self.value}) must be within confidence interval "
                    f"[{self.confidence_interval.lower}, {self.confidence_interval.upper}]"
                )
    
    def is_valid(self) -> bool:
        """Check if the score is valid."""
        return isinstance(self.value, (int, float)) and not (
            hasattr(self.value, '__isnan__') and self.value.__isnan__()
        )
    
    @property
    def is_confident(self) -> bool:
        """Check if score has confidence intervals."""
        return self.confidence_interval is not None
    
    @property
    def confidence_width(self) -> Optional[float]:
        """Calculate width of confidence interval."""
        if self.is_confident and self.confidence_interval:
            return self.confidence_interval.width
        return None
    
    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold."""
        return self.value > threshold
    
    def __str__(self) -> str:
        """String representation of the score."""
        return str(self.value)
    
    def __lt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value < other.value
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value <= other.value
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented
    
    def __gt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value > other.value
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented
    
    def __ge__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value >= other.value
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented


# Import here to avoid circular imports
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
'''
    
    with open('src/pynomaly/domain/value_objects/anomaly_score.py', 'w') as f:
        f.write(content)
    print("FIXED: AnomalyScore value object")


def fix_contamination_rate():
    """Fix ContaminationRate value object."""
    content = '''"""Contamination rate value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ContaminationRate:
    """Immutable value object representing contamination rate.
    
    Attributes:
        value: The contamination rate (0.0 to 1.0)
    """
    
    value: float
    
    # Class constants for common rates
    AUTO: ClassVar['ContaminationRate']
    LOW: ClassVar['ContaminationRate'] 
    MEDIUM: ClassVar['ContaminationRate']
    HIGH: ClassVar['ContaminationRate']
    
    def __post_init__(self) -> None:
        """Validate contamination rate after initialization."""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Contamination rate must be numeric, got {type(self.value)}")
        
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Contamination rate must be between 0 and 1, got {self.value}")
    
    def is_valid(self) -> bool:
        """Check if the contamination rate is valid."""
        return isinstance(self.value, (int, float)) and 0.0 <= self.value <= 1.0
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value:.2%}"
    
    @classmethod
    def auto(cls) -> 'ContaminationRate':
        """Create auto contamination rate (typically 0.1)."""
        return cls(0.1)
    
    @classmethod
    def low(cls) -> 'ContaminationRate':
        """Create low contamination rate."""
        return cls(0.05)
    
    @classmethod
    def medium(cls) -> 'ContaminationRate':
        """Create medium contamination rate."""
        return cls(0.1)
    
    @classmethod
    def high(cls) -> 'ContaminationRate':
        """Create high contamination rate."""
        return cls(0.2)


# Initialize class constants
ContaminationRate.AUTO = ContaminationRate(0.1)
ContaminationRate.LOW = ContaminationRate(0.05)
ContaminationRate.MEDIUM = ContaminationRate(0.1)
ContaminationRate.HIGH = ContaminationRate(0.2)
'''
    
    with open('src/pynomaly/domain/value_objects/contamination_rate.py', 'w') as f:
        f.write(content)
    print("FIXED: ContaminationRate value object")


def fix_confidence_interval():
    """Fix ConfidenceInterval value object."""
    content = '''"""Confidence interval value object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """Immutable value object representing a confidence interval.
    
    Attributes:
        lower: Lower bound of the interval
        upper: Upper bound of the interval
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    """
    
    lower: float
    upper: float
    confidence_level: float = 0.95
    
    def __post_init__(self) -> None:
        """Validate confidence interval after initialization."""
        if not isinstance(self.lower, (int, float)):
            raise ValueError(f"Lower bound must be numeric, got {type(self.lower)}")
        
        if not isinstance(self.upper, (int, float)):
            raise ValueError(f"Upper bound must be numeric, got {type(self.upper)}")
            
        if self.lower > self.upper:
            raise ValueError(
                f"Lower bound ({self.lower}) cannot be greater than upper bound ({self.upper})"
            )
        
        if not (0.0 <= self.confidence_level <= 1.0):
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )
    
    def is_valid(self) -> bool:
        """Check if the confidence interval is valid."""
        return (
            isinstance(self.lower, (int, float)) and
            isinstance(self.upper, (int, float)) and
            self.lower <= self.upper and
            0.0 <= self.confidence_level <= 1.0
        )
    
    @property
    def width(self) -> float:
        """Calculate width of the interval."""
        return self.upper - self.lower
    
    @property
    def center(self) -> float:
        """Calculate center of the interval."""
        return (self.lower + self.upper) / 2
    
    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower <= value <= self.upper
    
    def __str__(self) -> str:
        """String representation."""
        return f"[{self.lower:.3f}, {self.upper:.3f}] ({self.confidence_level:.0%})"
'''
    
    with open('src/pynomaly/domain/value_objects/confidence_interval.py', 'w') as f:
        f.write(content)
    print("FIXED: ConfidenceInterval value object")


def run_tests_and_show_progress():
    """Run tests and show current progress."""
    print("\nRunning domain tests to check progress...")
    success, stdout, stderr = run_command(
        ".venv/Scripts/python.exe -m pytest tests/domain/ --cov=pynomaly/domain --cov-report=term-missing -q"
    )
    
    if success:
        # Extract coverage percentage
        coverage_match = re.search(r'TOTAL.*?(\d+)%', stdout)
        if coverage_match:
            coverage = coverage_match.group(1)
            print(f"Domain layer coverage: {coverage}%")
        
        # Count passing/failing tests
        test_match = re.search(r'(\d+) failed.*?(\d+) passed', stdout)
        if test_match:
            failed, passed = test_match.groups()
            print(f"Tests: {passed} passed, {failed} failed")
        elif 'passed' in stdout:
            passed_match = re.search(r'(\d+) passed', stdout)
            if passed_match:
                print(f"All {passed_match.group(1)} domain tests passing!")
    else:
        print("Tests still failing, need more fixes")
        print(stderr[:500])  # Show first 500 chars of error
    
    return success


def main():
    """Main execution function."""
    print("Starting test coverage improvement script...")
    print("=" * 60)
    
    # Fix value objects
    print("\nFixing value objects...")
    fix_anomaly_score()
    fix_contamination_rate() 
    fix_confidence_interval()
    
    # Run tests to check progress
    success = run_tests_and_show_progress()
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Domain layer tests are now passing!")
        print("Next steps:")
        print("   1. Fix infrastructure adapter tests")
        print("   2. Fix application use case tests")
        print("   3. Fix presentation layer tests")
        print("   4. Run full test suite with coverage")
    else:
        print("WARNING: Still some issues to resolve in domain layer")
        print("Check the test output above for specific failures")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())