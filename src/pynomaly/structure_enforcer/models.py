"""
Data models for structure enforcer package.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from datetime import datetime


class ViolationType(Enum):
    """Types of structure violations."""
    STRAY_FILE = "stray_file"
    STRAY_DIRECTORY = "stray_directory"
    MISSING_INIT = "missing_init"
    INVALID_DEPENDENCY = "invalid_dependency"
    NAMING_CONVENTION = "naming_convention"
    CIRCULAR_IMPORT = "circular_import"
    DOMAIN_PURITY = "domain_purity"
    MISSING_LAYER = "missing_layer"
    EMPTY_DIRECTORY = "empty_directory"


class Severity(Enum):
    """Severity levels for violations."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FixType(Enum):
    """Types of fixes that can be applied."""
    MOVE_FILE = "move_file"
    MOVE_DIRECTORY = "move_directory"
    DELETE_FILE = "delete_file"
    DELETE_DIRECTORY = "delete_directory"
    CREATE_FILE = "create_file"
    CREATE_DIRECTORY = "create_directory"
    MODIFY_FILE = "modify_file"


@dataclass
class FileNode:
    """Represents a file in the repository model."""
    path: Path
    name: str
    size: int
    is_python: bool
    is_test: bool
    is_script: bool
    is_config: bool
    is_docs: bool
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.modified_at is None:
            self.modified_at = datetime.now()


@dataclass
class DirectoryNode:
    """Represents a directory in the repository model."""
    path: Path
    name: str
    files: List[FileNode]
    subdirectories: List['DirectoryNode']
    is_package: bool
    is_layer: bool
    layer_name: Optional[str] = None
    
    def get_python_files(self) -> List[FileNode]:
        """Get all Python files in this directory."""
        return [f for f in self.files if f.is_python]
    
    def get_file_count(self) -> int:
        """Get total file count including subdirectories."""
        count = len(self.files)
        for subdir in self.subdirectories:
            count += subdir.get_file_count()
        return count
    
    def has_init_file(self) -> bool:
        """Check if directory has __init__.py file."""
        return any(f.name == "__init__.py" for f in self.files)


@dataclass
class Model:
    """Repository structure model."""
    root_path: Path
    root_directory: DirectoryNode
    total_files: int
    total_directories: int
    max_depth: int
    layers: Dict[str, DirectoryNode]
    dependencies: Dict[str, Set[str]]
    scan_timestamp: datetime
    
    def get_layer_metrics(self) -> Dict[str, Dict[str, int]]:
        """Get metrics for each layer."""
        metrics = {}
        for layer_name, layer_dir in self.layers.items():
            metrics[layer_name] = {
                "files": layer_dir.get_file_count(),
                "directories": len(layer_dir.subdirectories) + 1,
                "python_files": len(layer_dir.get_python_files()),
            }
        return metrics
    
    def get_file_by_path(self, path: Union[str, Path]) -> Optional[FileNode]:
        """Find a file by its path."""
        target_path = Path(path)
        
        def search_directory(directory: DirectoryNode) -> Optional[FileNode]:
            for file in directory.files:
                if file.path == target_path:
                    return file
            for subdir in directory.subdirectories:
                result = search_directory(subdir)
                if result:
                    return result
            return None
        
        return search_directory(self.root_directory)


@dataclass
class Violation:
    """Represents a structure violation."""
    type: ViolationType
    severity: Severity
    message: str
    file_path: Optional[Path] = None
    directory_path: Optional[Path] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    rule_id: Optional[str] = None
    help_url: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.rule_id is None:
            self.rule_id = f"structure-{self.type.value}"
        if self.help_url is None:
            self.help_url = "https://github.com/pynomaly/pynomaly/blob/main/docs/development/FILE_ORGANIZATION_STANDARDS.md"
    
    def get_location(self) -> str:
        """Get human-readable location string."""
        if self.file_path:
            return str(self.file_path)
        elif self.directory_path:
            return str(self.directory_path)
        else:
            return "unknown"


@dataclass
class Fix:
    """Represents a fix for a violation."""
    type: FixType
    description: str
    source_path: Optional[Path] = None
    target_path: Optional[Path] = None
    content: Optional[str] = None
    backup_required: bool = True
    risk_level: str = "low"  # low, medium, high
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.type in [FixType.DELETE_FILE, FixType.DELETE_DIRECTORY]:
            self.risk_level = "high"
        elif self.type in [FixType.MOVE_FILE, FixType.MOVE_DIRECTORY]:
            self.risk_level = "medium"
    
    def get_preview(self) -> str:
        """Get a preview of what this fix will do."""
        if self.type == FixType.MOVE_FILE:
            return f"Move file from {self.source_path} to {self.target_path}"
        elif self.type == FixType.MOVE_DIRECTORY:
            return f"Move directory from {self.source_path} to {self.target_path}"
        elif self.type == FixType.DELETE_FILE:
            return f"Delete file {self.source_path}"
        elif self.type == FixType.DELETE_DIRECTORY:
            return f"Delete directory {self.source_path}"
        elif self.type == FixType.CREATE_FILE:
            return f"Create file {self.target_path}"
        elif self.type == FixType.CREATE_DIRECTORY:
            return f"Create directory {self.target_path}"
        elif self.type == FixType.MODIFY_FILE:
            return f"Modify file {self.source_path}"
        else:
            return self.description


@dataclass
class ValidationResult:
    """Results of validation process."""
    is_valid: bool
    violations: List[Violation]
    model: Model
    timestamp: datetime
    duration_seconds: float
    
    def get_summary(self) -> Dict[str, Union[int, str]]:
        """Get validation summary."""
        return {
            "is_valid": self.is_valid,
            "total_violations": len(self.violations),
            "error_count": len([v for v in self.violations if v.severity == Severity.ERROR]),
            "warning_count": len([v for v in self.violations if v.severity == Severity.WARNING]),
            "info_count": len([v for v in self.violations if v.severity == Severity.INFO]),
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FixResult:
    """Results of applying fixes."""
    applied_fixes: List[Fix]
    failed_fixes: List[tuple[Fix, str]]  # Fix and error message
    dry_run: bool
    timestamp: datetime
    
    def get_summary(self) -> Dict[str, Union[int, str]]:
        """Get fix application summary."""
        return {
            "applied_count": len(self.applied_fixes),
            "failed_count": len(self.failed_fixes),
            "dry_run": self.dry_run,
            "timestamp": self.timestamp.isoformat(),
        }
