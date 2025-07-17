"""Base classes for tool adapters."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """Represents a single analysis issue."""
    
    file: Path
    line: int
    column: int
    severity: str  # error, warning, info
    message: str
    rule: str
    tool: str
    fixable: bool = False
    category: str = ""
    suggestion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if isinstance(self.file, str):
            self.file = Path(self.file)
        
        # Validate severity
        valid_severities = ["error", "warning", "info"]
        if self.severity not in valid_severities:
            self.severity = "info"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary representation."""
        return {
            "file": str(self.file),
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "message": self.message,
            "rule": self.rule,
            "tool": self.tool,
            "fixable": self.fixable,
            "category": self.category,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Issue":
        """Create issue from dictionary representation."""
        return cls(
            file=Path(data["file"]),
            line=data["line"],
            column=data["column"],
            severity=data["severity"],
            message=data["message"],
            rule=data["rule"],
            tool=data["tool"],
            fixable=data.get("fixable", False),
            category=data.get("category", ""),
            suggestion=data.get("suggestion", ""),
            metadata=data.get("metadata", {}),
        )
    
    def merge_with(self, other: "Issue") -> None:
        """Merge this issue with another similar issue."""
        if other.tool != self.tool:
            self.metadata["merged_tools"] = self.metadata.get("merged_tools", [self.tool]) + [other.tool]
        
        if other.suggestion and not self.suggestion:
            self.suggestion = other.suggestion
        
        # Merge metadata
        self.metadata.update(other.metadata)


@dataclass
class AnalysisResult:
    """Represents the result of running an analysis tool."""
    
    tool: str
    issues: List[Issue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "tool": self.tool,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create result from dictionary representation."""
        return cls(
            tool=data["tool"],
            issues=[Issue.from_dict(issue_data) for issue_data in data["issues"]],
            metadata=data.get("metadata", {}),
            execution_time=data.get("execution_time", 0.0),
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
        )


class ToolAdapter(ABC):
    """Base class for tool adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.replace("Adapter", "").lower()
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run analysis on files and return results."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if tool is available in environment."""
        pass
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the tool with given settings."""
        self.config.update(config)
    
    async def _run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                          timeout: float = 300.0) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        try:
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore')
            )
            
            self.logger.debug(f"Command completed with return code: {result.returncode}")
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
            raise
        except Exception as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}: {e}")
            raise
    
    def _filter_files(self, files: List[Path]) -> List[Path]:
        """Filter files to only include supported extensions."""
        supported_extensions = self.get_supported_extensions()
        if not supported_extensions:
            return files
        
        return [
            f for f in files 
            if f.suffix in supported_extensions or f.suffix.lower() in supported_extensions
        ]
    
    def _create_temp_config(self, config_content: str, suffix: str = ".tmp") -> Path:
        """Create a temporary configuration file."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(config_content)
            return Path(f.name)
    
    def _map_severity(self, tool_severity: str) -> str:
        """Map tool-specific severity to standard severity."""
        severity_map = {
            # Common mappings
            "error": "error",
            "warning": "warning",
            "info": "info",
            "note": "info",
            # MyPy
            "error": "error",
            # Ruff
            "E": "error",
            "W": "warning",
            "F": "error",
            "C": "warning",
            "N": "info",
            # Bandit
            "high": "error",
            "medium": "warning",
            "low": "info",
        }
        
        return severity_map.get(tool_severity.lower(), "info")
    
    def _calculate_confidence(self, issue: Issue) -> int:
        """Calculate confidence score for an issue (0-100)."""
        base_confidence = 80
        
        # Adjust based on severity
        if issue.severity == "error":
            base_confidence += 10
        elif issue.severity == "info":
            base_confidence -= 10
        
        # Adjust based on tool
        if issue.tool in ["mypy", "pyright"]:
            base_confidence += 10
        elif issue.tool in ["ruff", "black"]:
            base_confidence += 5
        
        # Adjust based on category
        if issue.category == "security":
            base_confidence += 15
        elif issue.category == "performance":
            base_confidence += 5
        
        return min(100, max(0, base_confidence))
    
    def _create_issue(self, file: Path, line: int, column: int, message: str, 
                     rule: str, severity: str = "info", fixable: bool = False,
                     category: str = "", suggestion: str = "", 
                     metadata: Optional[Dict[str, Any]] = None) -> Issue:
        """Create a standardized issue."""
        issue = Issue(
            file=file,
            line=line,
            column=column,
            severity=self._map_severity(severity),
            message=message,
            rule=rule,
            tool=self.name,
            fixable=fixable,
            category=category,
            suggestion=suggestion,
            metadata=metadata or {},
        )
        
        # Add confidence score
        issue.metadata["confidence"] = self._calculate_confidence(issue)
        
        return issue