"""Pyright adapter for enhanced type checking."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class PyrightAdapter(ToolAdapter):
    """Adapter for Pyright type checking."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pyright_config = self._create_pyright_config()
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py", ".pyi"]
    
    def is_available(self) -> bool:
        """Check if Pyright is available in environment."""
        return shutil.which("pyright") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run Pyright type analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="Pyright is not available. Install with: npm install -g pyright"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = [
            "pyright",
            "--outputformat", "json",
            "--project", str(self.pyright_config.parent),
        ]
        
        # Add configuration
        pyright_config = self.config.get("pyright", {})
        
        if type_checking_mode := pyright_config.get("typeCheckingMode"):
            cmd.extend(["--typecheckingmode", type_checking_mode])
        
        if python_version := pyright_config.get("pythonVersion"):
            cmd.extend(["--pythonversion", python_version])
        
        if venv_path := pyright_config.get("venvPath"):
            cmd.extend(["--venvpath", str(venv_path)])
        
        if pyright_config.get("verbose"):
            cmd.append("--verbose")
        
        # Add file paths
        cmd.extend([str(f) for f in python_files])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse JSON output
            issues = self._parse_pyright_output(result.stdout)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "files_analyzed": len(python_files),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "type_checking_metrics": self._calculate_type_checking_metrics(issues),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pyright analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _create_pyright_config(self) -> Path:
        """Create Pyright configuration file."""
        config_content = self._generate_pyright_config()
        return self._create_temp_config(config_content, ".json", "pyrightconfig")
    
    def _generate_pyright_config(self) -> str:
        """Generate Pyright configuration content."""
        pyright_config = self.config.get("pyright", {})
        
        config_dict = {
            "typeCheckingMode": pyright_config.get("typeCheckingMode", "standard"),
            "pythonVersion": pyright_config.get("pythonVersion", "3.11"),
            "pythonPlatform": pyright_config.get("pythonPlatform", "Linux"),
            "executionEnvironments": [
                {
                    "root": ".",
                    "pythonVersion": pyright_config.get("pythonVersion", "3.11"),
                    "pythonPlatform": pyright_config.get("pythonPlatform", "Linux"),
                    "extraPaths": pyright_config.get("extraPaths", [])
                }
            ],
            "include": pyright_config.get("include", ["**/*.py"]),
            "exclude": pyright_config.get("exclude", [
                "**/__pycache__",
                "**/.git",
                "**/.venv",
                "**/venv",
                "**/node_modules",
                "**/build",
                "**/dist",
                "**/.pytest_cache",
                "**/.mypy_cache",
            ]),
            "ignore": pyright_config.get("ignore", []),
            "defineConstant": pyright_config.get("defineConstant", {}),
            "stubPath": pyright_config.get("stubPath", ""),
            "venvPath": pyright_config.get("venvPath", ""),
            "venv": pyright_config.get("venv", ""),
            "verboseOutput": pyright_config.get("verboseOutput", False),
            "reportGeneralTypeIssues": pyright_config.get("reportGeneralTypeIssues", "error"),
            "reportOptionalSubscript": pyright_config.get("reportOptionalSubscript", "error"),
            "reportOptionalMemberAccess": pyright_config.get("reportOptionalMemberAccess", "error"),
            "reportOptionalCall": pyright_config.get("reportOptionalCall", "error"),
            "reportOptionalIterable": pyright_config.get("reportOptionalIterable", "error"),
            "reportOptionalContextManager": pyright_config.get("reportOptionalContextManager", "error"),
            "reportOptionalOperand": pyright_config.get("reportOptionalOperand", "error"),
            "reportTypedDictNotRequiredAccess": pyright_config.get("reportTypedDictNotRequiredAccess", "error"),
            "reportUntypedFunctionDecorator": pyright_config.get("reportUntypedFunctionDecorator", "error"),
            "reportUntypedClassDecorator": pyright_config.get("reportUntypedClassDecorator", "error"),
            "reportUntypedBaseClass": pyright_config.get("reportUntypedBaseClass", "error"),
            "reportUntypedNamedTuple": pyright_config.get("reportUntypedNamedTuple", "error"),
            "reportPrivateUsage": pyright_config.get("reportPrivateUsage", "error"),
            "reportTypeCommentUsage": pyright_config.get("reportTypeCommentUsage", "error"),
            "reportPrivateImportUsage": pyright_config.get("reportPrivateImportUsage", "error"),
            "reportConstantRedefinition": pyright_config.get("reportConstantRedefinition", "error"),
            "reportIncompatibleMethodOverride": pyright_config.get("reportIncompatibleMethodOverride", "error"),
            "reportIncompatibleVariableOverride": pyright_config.get("reportIncompatibleVariableOverride", "error"),
            "reportInconsistentConstructor": pyright_config.get("reportInconsistentConstructor", "error"),
            "reportOverlappingOverload": pyright_config.get("reportOverlappingOverload", "error"),
            "reportMissingSuperCall": pyright_config.get("reportMissingSuperCall", "error"),
            "reportPropertyTypeMismatch": pyright_config.get("reportPropertyTypeMismatch", "error"),
            "reportFunctionMemberAccess": pyright_config.get("reportFunctionMemberAccess", "error"),
            "reportMissingParameterType": pyright_config.get("reportMissingParameterType", "error"),
            "reportMissingTypeArgument": pyright_config.get("reportMissingTypeArgument", "error"),
            "reportInvalidTypeVarUse": pyright_config.get("reportInvalidTypeVarUse", "error"),
            "reportCallInDefaultInitializer": pyright_config.get("reportCallInDefaultInitializer", "error"),
            "reportUnnecessaryIsInstance": pyright_config.get("reportUnnecessaryIsInstance", "error"),
            "reportUnnecessaryCast": pyright_config.get("reportUnnecessaryCast", "error"),
            "reportUnnecessaryComparison": pyright_config.get("reportUnnecessaryComparison", "error"),
            "reportUnnecessaryContains": pyright_config.get("reportUnnecessaryContains", "error"),
            "reportImplicitStringConcatenation": pyright_config.get("reportImplicitStringConcatenation", "error"),
            "reportUnusedCallResult": pyright_config.get("reportUnusedCallResult", "error"),
            "reportUnusedCoroutine": pyright_config.get("reportUnusedCoroutine", "error"),
            "reportUnusedExcept": pyright_config.get("reportUnusedExcept", "error"),
            "reportUnusedExpression": pyright_config.get("reportUnusedExpression", "error"),
            "reportUnnecessaryTypeIgnoreComment": pyright_config.get("reportUnnecessaryTypeIgnoreComment", "error"),
            "reportMatchNotExhaustive": pyright_config.get("reportMatchNotExhaustive", "error"),
        }
        
        return json.dumps(config_dict, indent=2)
    
    def _parse_pyright_output(self, output: str) -> List[Issue]:
        """Parse Pyright JSON output."""
        issues = []
        
        if not output.strip():
            return issues
        
        try:
            pyright_data = json.loads(output)
            
            # Parse diagnostics
            diagnostics = pyright_data.get("generalDiagnostics", [])
            
            for diagnostic in diagnostics:
                # Extract file and position info
                file_path = diagnostic.get("file", "")
                range_info = diagnostic.get("range", {})
                start_pos = range_info.get("start", {})
                
                issue = self._create_issue(
                    file=Path(file_path) if file_path else Path("unknown"),
                    line=start_pos.get("line", 0) + 1,  # Pyright uses 0-based line numbers
                    column=start_pos.get("character", 0) + 1,  # Pyright uses 0-based column numbers
                    message=diagnostic.get("message", ""),
                    rule=diagnostic.get("rule", "pyright"),
                    severity=self._map_pyright_severity(diagnostic.get("severity", "error")),
                    fixable=False,  # Pyright doesn't provide auto-fixes
                    category="type_checking",
                    suggestion=self._get_pyright_suggestion(diagnostic),
                    metadata={
                        "severity": diagnostic.get("severity"),
                        "rule": diagnostic.get("rule"),
                        "range": range_info,
                        "related_information": diagnostic.get("relatedInformation", []),
                    }
                )
                issues.append(issue)
            
            # Parse file-level diagnostics
            for file_diagnostic in pyright_data.get("diagnostics", []):
                file_path = file_diagnostic.get("file", "")
                for diagnostic in file_diagnostic.get("diagnostics", []):
                    range_info = diagnostic.get("range", {})
                    start_pos = range_info.get("start", {})
                    
                    issue = self._create_issue(
                        file=Path(file_path) if file_path else Path("unknown"),
                        line=start_pos.get("line", 0) + 1,
                        column=start_pos.get("character", 0) + 1,
                        message=diagnostic.get("message", ""),
                        rule=diagnostic.get("rule", "pyright"),
                        severity=self._map_pyright_severity(diagnostic.get("severity", "error")),
                        fixable=False,
                        category="type_checking",
                        suggestion=self._get_pyright_suggestion(diagnostic),
                        metadata={
                            "severity": diagnostic.get("severity"),
                            "rule": diagnostic.get("rule"),
                            "range": range_info,
                            "related_information": diagnostic.get("relatedInformation", []),
                        }
                    )
                    issues.append(issue)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Pyright JSON output: {e}")
            self.logger.debug(f"Output was: {output}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing Pyright output: {e}")
            self.logger.debug(f"Output was: {output}")
        
        return issues
    
    def _map_pyright_severity(self, pyright_severity: str) -> str:
        """Map Pyright severity to standard severity."""
        severity_map = {
            "error": "error",
            "warning": "warning",
            "information": "info",
            "hint": "info"
        }
        return severity_map.get(pyright_severity.lower(), "error")
    
    def _get_pyright_suggestion(self, diagnostic: Dict[str, Any]) -> str:
        """Get type checking suggestion for Pyright diagnostic."""
        message = diagnostic.get("message", "")
        rule = diagnostic.get("rule", "")
        
        # Common type checking suggestions
        if "is not defined" in message:
            return "Import the missing module or define the variable"
        elif "has no attribute" in message:
            return "Check the object type or add the missing attribute"
        elif "Cannot assign" in message:
            return "Check type compatibility or use type casting"
        elif "Argument of type" in message:
            return "Check argument types match function signature"
        elif "not callable" in message:
            return "Ensure the object is callable or check the type"
        elif "incompatible" in message.lower():
            return "Fix type incompatibility by adjusting types"
        elif "missing" in message.lower():
            return "Add the missing type annotation or import"
        elif "undefined" in message.lower():
            return "Define the variable or import the module"
        elif "unbound" in message.lower():
            return "Initialize the variable before use"
        elif "unreachable" in message.lower():
            return "Remove unreachable code or fix control flow"
        elif "unused" in message.lower():
            return "Remove unused variable or use it in the code"
        elif "override" in message.lower():
            return "Fix method signature to match parent class"
        elif "abstract" in message.lower():
            return "Implement abstract methods or make class abstract"
        elif "final" in message.lower():
            return "Remove final modifier or avoid overriding"
        elif "generic" in message.lower():
            return "Add type parameters or specify generic types"
        elif "protocol" in message.lower():
            return "Implement protocol methods or check type compatibility"
        elif "literal" in message.lower():
            return "Use literal types or check value constraints"
        elif "union" in message.lower():
            return "Handle all union type cases or narrow the type"
        elif "optional" in message.lower():
            return "Handle None case or use non-optional type"
        elif "return" in message.lower():
            return "Add return statement or fix return type"
        else:
            return "Review type annotations and fix type-related issues"
    
    def _calculate_type_checking_metrics(self, issues: List[Issue]) -> Dict[str, Any]:
        """Calculate type checking metrics from issues."""
        metrics = {
            "total_type_issues": len(issues),
            "type_errors": 0,
            "type_warnings": 0,
            "type_info": 0,
            "files_with_type_issues": set(),
            "issues_by_category": {},
            "issues_by_file": {},
            "most_common_rules": {},
            "type_coverage_estimate": 0,
        }
        
        for issue in issues:
            # Count by severity
            if issue.severity == "error":
                metrics["type_errors"] += 1
            elif issue.severity == "warning":
                metrics["type_warnings"] += 1
            else:
                metrics["type_info"] += 1
            
            # Track files with issues
            metrics["files_with_type_issues"].add(str(issue.file))
            
            # Count issues by file
            file_str = str(issue.file)
            metrics["issues_by_file"][file_str] = (
                metrics["issues_by_file"].get(file_str, 0) + 1
            )
            
            # Count by rule
            rule = issue.rule
            metrics["most_common_rules"][rule] = (
                metrics["most_common_rules"].get(rule, 0) + 1
            )
            
            # Categorize issues
            category = self._categorize_type_issue(issue.message)
            metrics["issues_by_category"][category] = (
                metrics["issues_by_category"].get(category, 0) + 1
            )
        
        # Convert set to count
        metrics["files_with_type_issues"] = len(metrics["files_with_type_issues"])
        
        # Calculate type safety score (higher is better)
        metrics["type_safety_score"] = max(0, 100 - (
            metrics["type_errors"] * 10 +
            metrics["type_warnings"] * 5 +
            metrics["type_info"] * 1
        ))
        
        # Estimate type coverage (rough approximation)
        if metrics["files_with_type_issues"] > 0:
            metrics["type_coverage_estimate"] = max(0, 100 - (
                metrics["total_type_issues"] / metrics["files_with_type_issues"] * 10
            ))
        else:
            metrics["type_coverage_estimate"] = 100
        
        return metrics
    
    def _categorize_type_issue(self, message: str) -> str:
        """Categorize type issue based on message."""
        message_lower = message.lower()
        
        if "not defined" in message_lower or "undefined" in message_lower:
            return "undefined_reference"
        elif "has no attribute" in message_lower:
            return "attribute_error"
        elif "cannot assign" in message_lower or "incompatible" in message_lower:
            return "type_incompatibility"
        elif "argument" in message_lower:
            return "argument_type"
        elif "return" in message_lower:
            return "return_type"
        elif "callable" in message_lower:
            return "callable_type"
        elif "missing" in message_lower:
            return "missing_annotation"
        elif "unreachable" in message_lower:
            return "unreachable_code"
        elif "unused" in message_lower:
            return "unused_code"
        elif "override" in message_lower:
            return "method_override"
        elif "abstract" in message_lower:
            return "abstract_method"
        elif "final" in message_lower:
            return "final_type"
        elif "generic" in message_lower:
            return "generic_type"
        elif "protocol" in message_lower:
            return "protocol_compliance"
        elif "literal" in message_lower:
            return "literal_type"
        elif "union" in message_lower:
            return "union_type"
        elif "optional" in message_lower:
            return "optional_type"
        else:
            return "other"