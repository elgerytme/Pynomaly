"""Ruff adapter for code quality analysis."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class RuffAdapter(ToolAdapter):
    """Adapter for Ruff linting and formatting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ruff_config = self._create_ruff_config()
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py", ".pyi"]
    
    def is_available(self) -> bool:
        """Check if Ruff is available in environment."""
        return shutil.which("ruff") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run Ruff analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="Ruff is not available. Install with: pip install ruff"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = [
            "ruff", "check",
            "--config", str(self.ruff_config),
            "--output-format", "json",
            "--no-cache",
            *[str(f) for f in python_files]
        ]
        
        # Add rule selection
        ruff_config = self.config.get("ruff", {})
        if select_rules := ruff_config.get("select"):
            cmd.extend(["--select", ",".join(select_rules)])
        
        if ignore_rules := ruff_config.get("ignore"):
            cmd.extend(["--ignore", ",".join(ignore_rules)])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse JSON output
            issues = self._parse_ruff_output(result.stdout)
            
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
            self.logger.error(f"Ruff analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _create_ruff_config(self) -> Path:
        """Create Ruff configuration file."""
        config_content = self._generate_ruff_config()
        return self._create_temp_config(config_content, ".toml")
    
    def _generate_ruff_config(self) -> str:
        """Generate Ruff configuration content."""
        ruff_config = self.config.get("ruff", {})
        
        config_dict = {
            "line-length": ruff_config.get("line_length", 88),
            "indent-width": ruff_config.get("indent_width", 4),
            "target-version": ruff_config.get("target_version", "py311"),
            "exclude": ruff_config.get("exclude", [
                "__pycache__",
                ".venv",
                "venv",
                ".git",
                "build",
                "dist",
                "node_modules",
            ]),
        }
        
        # Add rule selection
        if select_rules := ruff_config.get("select"):
            config_dict["select"] = select_rules
        
        if ignore_rules := ruff_config.get("ignore"):
            config_dict["ignore"] = ignore_rules
        
        if per_file_ignores := ruff_config.get("per_file_ignores"):
            config_dict["per-file-ignores"] = per_file_ignores
        
        # Convert to TOML format
        import toml
        return toml.dumps(config_dict)
    
    def _parse_ruff_output(self, output: str) -> List[Issue]:
        """Parse Ruff JSON output."""
        issues = []
        
        if not output.strip():
            return issues
        
        try:
            violations = json.loads(output)
            
            for violation in violations:
                issue = self._create_issue(
                    file=Path(violation["filename"]),
                    line=violation["location"]["row"],
                    column=violation["location"]["column"],
                    message=violation["message"],
                    rule=violation["code"],
                    severity=self._map_ruff_severity(violation["code"]),
                    fixable=(violation.get("fix") or {}).get("applicable", False),
                    category=self._get_rule_category(violation["code"]),
                    suggestion=self._get_ruff_suggestion(violation),
                    metadata={
                        "end_location": violation.get("end_location"),
                        "fix": violation.get("fix"),
                        "url": violation.get("url"),
                    }
                )
                issues.append(issue)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Ruff JSON output: {e}")
            self.logger.debug(f"Output was: {output}")
        
        return issues
    
    def _map_ruff_severity(self, rule_code: str) -> str:
        """Map Ruff rule code to severity."""
        if rule_code.startswith("E"):
            # Error codes
            return "error"
        elif rule_code.startswith("W"):
            # Warning codes
            return "warning"
        elif rule_code.startswith("F"):
            # Pyflakes codes (errors)
            return "error"
        elif rule_code.startswith("C"):
            # Convention codes
            return "warning"
        elif rule_code.startswith("N"):
            # Naming codes
            return "info"
        elif rule_code.startswith("I"):
            # Import codes
            return "info"
        elif rule_code.startswith("B"):
            # Bugbear codes
            return "warning"
        elif rule_code.startswith("S"):
            # Security codes
            return "error"
        else:
            return "info"
    
    def _get_rule_category(self, rule_code: str) -> str:
        """Get category for rule code."""
        category_map = {
            "E": "style",
            "W": "style",
            "F": "logic",
            "C": "complexity",
            "N": "naming",
            "I": "import",
            "B": "bugbear",
            "S": "security",
            "T": "testing",
            "D": "documentation",
            "UP": "upgrade",
            "SIM": "simplify",
            "PL": "pylint",
            "RUF": "ruff",
        }
        
        prefix = rule_code[:1]
        if prefix in category_map:
            return category_map[prefix]
        
        # Check for longer prefixes
        for prefix_len in [3, 2]:
            prefix = rule_code[:prefix_len]
            if prefix in category_map:
                return category_map[prefix]
        
        return "other"
    
    def _get_ruff_suggestion(self, violation: Dict[str, Any]) -> str:
        """Get fix suggestion for Ruff violation."""
        fix_info = violation.get("fix") or {}
        
        if fix_info.get("applicable"):
            message = fix_info.get("message", "")
            if message:
                return f"Auto-fix available: {message}"
        
        # Rule-specific suggestions
        rule_code = violation["code"]
        message = violation["message"]
        
        if rule_code.startswith("E"):
            return "Fix code style issue"
        elif rule_code.startswith("F"):
            return "Fix logic error"
        elif rule_code.startswith("I"):
            return "Fix import organization"
        elif rule_code.startswith("N"):
            return "Fix naming convention"
        elif rule_code == "B006":
            return "Use default factory for mutable default arguments"
        elif rule_code == "B007":
            return "Use enumerate() instead of manual counter"
        elif rule_code.startswith("S"):
            return "Fix security vulnerability"
        
        return ""