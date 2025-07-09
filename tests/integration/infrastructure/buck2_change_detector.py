"""Mock Buck2 change detector for testing."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ChangeAnalysis:
    """Mock change analysis result."""
    
    affected_files: list[str]
    test_targets: list[str]
    changed_since: str
    analysis_time: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "affected_files": self.affected_files,
            "test_targets": self.test_targets,
            "changed_since": self.changed_since,
            "analysis_time": self.analysis_time,
        }


class Buck2ChangeDetector:
    """Mock Buck2 change detector."""
    
    def __init__(self, root_path: str = "."):
        """Initialize detector."""
        self.root_path = root_path
        
    def analyze_changes(self, since: str = "HEAD~1") -> ChangeAnalysis:
        """Analyze changes since given commit."""
        return ChangeAnalysis(
            affected_files=["src/pynomaly/test_file.py"],
            test_targets=["//tests:unit_tests"],
            changed_since=since,
            analysis_time=0.1,
        )
        
    def get_affected_targets(self, files: list[str]) -> list[str]:
        """Get Buck2 targets affected by file changes."""
        return ["//tests:unit_tests", "//tests:integration_tests"]
        
    def should_run_target(self, target: str, changed_files: list[str]) -> bool:
        """Check if target should be run based on changed files."""
        return True  # Always run for mock