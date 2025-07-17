"""
Base class for all repository governance checkers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class BaseChecker(ABC):
    """Base class for all repository checkers."""
    
    def __init__(self, root_path: Path):
        """Initialize the base checker."""
        self.root_path = root_path
        self.name = self.__class__.__name__
    
    @abstractmethod
    def check(self) -> Dict:
        """
        Run the check and return results.
        
        Returns:
            Dict containing:
            - violations: List of violations found
            - total_violations: Number of violations
            - score: Quality score (0-100)
            - recommendations: List of recommendations
        """
        pass
    
    def get_python_files(self) -> list[Path]:
        """Get all Python files in the repository."""
        return list(self.root_path.rglob("*.py"))
    
    def get_package_directories(self) -> list[Path]:
        """Get all package directories."""
        packages = []
        src_packages = self.root_path / "src" / "packages"
        
        if src_packages.exists():
            for item in src_packages.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    packages.append(item)
        
        return packages
    
    def is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file."""
        return (
            file_path.name.startswith("test_") or
            file_path.name.endswith("_test.py") or
            "test" in file_path.parts
        )
    
    def calculate_score(self, violations: list, max_score: int = 100, penalty_per_violation: int = 5) -> int:
        """Calculate a quality score based on violations."""
        return max(0, max_score - len(violations) * penalty_per_violation)