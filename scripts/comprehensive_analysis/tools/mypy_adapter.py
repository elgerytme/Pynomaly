"""MyPy adapter for type checking."""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class MyPyAdapter(ToolAdapter):
    """Adapter for MyPy type checking."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mypy_config = self._create_mypy_config()
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py", ".pyi"]
    
    def is_available(self) -> bool:
        """Check if MyPy is available in environment."""
        return shutil.which("mypy") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run MyPy analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="MyPy is not available. Install with: pip install mypy"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = [
            "mypy",
            "--config-file", str(self.mypy_config),
            "--show-error-codes",
            "--show-column-numbers",
            "--show-error-end",
            "--no-error-summary",
            *[str(f) for f in python_files]
        ]
        
        # Add JSON output if available
        if self._supports_json_output():
            cmd.extend(["--output", "json"])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse output
            if "--output" in cmd and "json" in cmd:
                issues = self._parse_json_output(result.stdout)
            else:
                issues = self._parse_text_output(result.stdout)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "files_analyzed": len(python_files),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                }
            )
            
        except Exception as e:
            self.logger.error(f"MyPy analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _create_mypy_config(self) -> Path:
        """Create MyPy configuration file."""
        config_content = self._generate_mypy_config()
        return self._create_temp_config(config_content, ".ini")
    
    def _generate_mypy_config(self) -> str:
        """Generate MyPy configuration content."""
        config = self.config.get("mypy", {})
        
        # Base configuration
        config_lines = [
            "[mypy]",
            f"python_version = {config.get('python_version', '3.11')}",
            f"strict = {config.get('strict', False)}",
            "warn_return_any = True",
            "warn_unused_configs = True",
            f"disallow_untyped_defs = {config.get('disallow_untyped_defs', False)}",
            f"disallow_incomplete_defs = {config.get('disallow_incomplete_defs', False)}",
            f"check_untyped_defs = {config.get('check_untyped_defs', True)}",
            f"disallow_untyped_decorators = {config.get('disallow_untyped_decorators', False)}",
            f"no_implicit_optional = {config.get('no_implicit_optional', True)}",
            f"warn_redundant_casts = {config.get('warn_redundant_casts', True)}",
            f"warn_unused_ignores = {config.get('warn_unused_ignores', True)}",
            f"warn_no_return = {config.get('warn_no_return', True)}",
            f"warn_unreachable = {config.get('warn_unreachable', True)}",
            f"strict_equality = {config.get('strict_equality', True)}",
        ]
        
        # Add plugins
        plugins = config.get("plugins", [])
        if plugins:
            config_lines.append(f"plugins = {', '.join(plugins)}")
        
        # Add ignore patterns
        ignore_patterns = config.get("ignore_patterns", [])
        for pattern in ignore_patterns:
            config_lines.append(f"[mypy-{pattern}]")
            config_lines.append("ignore_errors = True")
        
        return "\n".join(config_lines)
    
    def _supports_json_output(self) -> bool:
        """Check if MyPy supports JSON output."""
        try:
            import subprocess
            result = subprocess.run(
                ["mypy", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "--output" in result.stdout
        except:
            return False
    
    def _parse_json_output(self, output: str) -> List[Issue]:
        """Parse MyPy JSON output."""
        issues = []
        
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                issue = self._create_issue(
                    file=Path(data["file"]),
                    line=data["line"],
                    column=data.get("column", 0),
                    message=data["message"],
                    rule=data.get("error_code", "mypy"),
                    severity=self._map_mypy_severity(data.get("severity", "error")),
                    fixable=self._is_fixable(data),
                    category="type-checking",
                    suggestion=self._get_suggestion(data),
                    metadata={
                        "end_line": data.get("end_line"),
                        "end_column": data.get("end_column"),
                        "hint": data.get("hint"),
                    }
                )
                issues.append(issue)
                
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse MyPy JSON output: {line}")
                continue
        
        return issues
    
    def _parse_text_output(self, output: str) -> List[Issue]:
        """Parse MyPy text output."""
        issues = []
        
        # MyPy output format: file:line:column: severity: message [error-code]
        pattern = r"^(.+?):(\d+):(\d+): (\w+): (.+?)(?:\s+\[([^\]]+)\])?$"
        
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            
            match = re.match(pattern, line)
            if match:
                file_path, line_no, column, severity, message, error_code = match.groups()
                
                issue = self._create_issue(
                    file=Path(file_path),
                    line=int(line_no),
                    column=int(column),
                    message=message,
                    rule=error_code or "mypy",
                    severity=self._map_mypy_severity(severity),
                    fixable=self._is_message_fixable(message),
                    category="type-checking",
                    suggestion=self._get_message_suggestion(message),
                )
                issues.append(issue)
        
        return issues
    
    def _map_mypy_severity(self, severity: str) -> str:
        """Map MyPy severity to standard severity."""
        severity_map = {
            "error": "error",
            "warning": "warning",
            "note": "info",
        }
        return severity_map.get(severity.lower(), "error")
    
    def _is_fixable(self, data: Dict[str, Any]) -> bool:
        """Check if issue is automatically fixable."""
        error_code = data.get("error_code", "")
        message = data.get("message", "")
        
        # Common fixable issues
        fixable_codes = [
            "import-untyped",
            "unused-ignore",
            "redundant-cast",
        ]
        
        fixable_patterns = [
            "Missing type annotation",
            "Unused 'type: ignore' comment",
            "Redundant cast",
        ]
        
        return (
            error_code in fixable_codes or
            any(pattern in message for pattern in fixable_patterns)
        )
    
    def _is_message_fixable(self, message: str) -> bool:
        """Check if message indicates a fixable issue."""
        fixable_patterns = [
            "Missing type annotation",
            "Unused 'type: ignore' comment",
            "Redundant cast",
            "Import cycle",
        ]
        
        return any(pattern in message for pattern in fixable_patterns)
    
    def _get_suggestion(self, data: Dict[str, Any]) -> str:
        """Get fix suggestion for issue."""
        error_code = data.get("error_code", "")
        message = data.get("message", "")
        
        if "Missing type annotation" in message:
            return "Add type annotation to function/variable"
        elif "Unused 'type: ignore' comment" in message:
            return "Remove unnecessary type: ignore comment"
        elif "Redundant cast" in message:
            return "Remove redundant cast"
        elif error_code == "import-untyped":
            return "Install type stubs or add type: ignore comment"
        
        return ""
    
    def _get_message_suggestion(self, message: str) -> str:
        """Get fix suggestion from message text."""
        if "Missing type annotation" in message:
            return "Add type annotation to function/variable"
        elif "Unused 'type: ignore'" in message:
            return "Remove unnecessary type: ignore comment"
        elif "Redundant cast" in message:
            return "Remove redundant cast"
        elif "has no attribute" in message:
            return "Check object type or add type assertion"
        elif "Incompatible types" in message:
            return "Fix type mismatch or add type conversion"
        
        return ""