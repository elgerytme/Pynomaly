"""Bandit adapter for security vulnerability scanning."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class BanditAdapter(ToolAdapter):
    """Adapter for Bandit security vulnerability scanning."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bandit_config = self._create_bandit_config()
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py"]
    
    def is_available(self) -> bool:
        """Check if Bandit is available in environment."""
        return shutil.which("bandit") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run Bandit security analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="Bandit is not available. Install with: pip install bandit"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = [
            "bandit",
            "--format", "json",
            "--recursive",
            "--aggregate", "file",
            "--severity-level", "low",
            "--confidence-level", "low",
        ]
        
        # Add configuration
        bandit_config = self.config.get("bandit", {})
        
        if skip_tests := bandit_config.get("skip_tests"):
            cmd.extend(["--skip", ",".join(skip_tests)])
        
        if exclude_dirs := bandit_config.get("exclude_dirs"):
            cmd.extend(["--exclude", ",".join(exclude_dirs)])
        
        if baseline_file := bandit_config.get("baseline"):
            cmd.extend(["--baseline", str(baseline_file)])
        
        # Add file paths
        cmd.extend([str(f) for f in python_files])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse JSON output
            issues = self._parse_bandit_output(result.stdout)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "files_analyzed": len(python_files),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "security_metrics": self._calculate_security_metrics(issues),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Bandit analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _create_bandit_config(self) -> Path:
        """Create Bandit configuration file."""
        config_content = self._generate_bandit_config()
        return self._create_temp_config(config_content, ".yaml")
    
    def _generate_bandit_config(self) -> str:
        """Generate Bandit configuration content."""
        bandit_config = self.config.get("bandit", {})
        
        config_dict = {
            "tests": bandit_config.get("tests", []),
            "skips": bandit_config.get("skips", []),
            "exclude_dirs": bandit_config.get("exclude_dirs", [
                "/__pycache__/",
                "/.git/",
                "/.venv/",
                "/venv/",
                "/build/",
                "/dist/",
                "/node_modules/",
                "/.pytest_cache/",
                "/.mypy_cache/",
            ]),
            "exclude": bandit_config.get("exclude", []),
            "assert_used": {
                "skips": ["*_test.py", "*/test_*.py", "*/tests/*"],
                "word_list": ["assert"]
            }
        }
        
        # Convert to YAML format
        import yaml
        return yaml.dump(config_dict)
    
    def _parse_bandit_output(self, output: str) -> List[Issue]:
        """Parse Bandit JSON output."""
        issues = []
        
        if not output.strip():
            return issues
        
        try:
            bandit_data = json.loads(output)
            
            # Parse results
            results = bandit_data.get("results", [])
            
            for result in results:
                issue = self._create_issue(
                    file=Path(result["filename"]),
                    line=result["line_number"],
                    column=result.get("col_offset", 0),
                    message=result["issue_text"],
                    rule=result["test_id"],
                    severity=self._map_bandit_severity(result["issue_severity"]),
                    fixable=False,  # Bandit doesn't provide auto-fixes
                    category="security",
                    suggestion=self._get_bandit_suggestion(result),
                    metadata={
                        "confidence": result["issue_confidence"],
                        "cwe": result.get("issue_cwe"),
                        "more_info": result.get("more_info"),
                        "line_range": result.get("line_range"),
                        "test_name": result.get("test_name"),
                        "code": result.get("code"),
                    }
                )
                issues.append(issue)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Bandit JSON output: {e}")
            self.logger.debug(f"Output was: {output}")
        except KeyError as e:
            self.logger.error(f"Unexpected Bandit output format: {e}")
            self.logger.debug(f"Output was: {output}")
        
        return issues
    
    def _map_bandit_severity(self, bandit_severity: str) -> str:
        """Map Bandit severity to standard severity."""
        severity_map = {
            "HIGH": "error",
            "MEDIUM": "warning", 
            "LOW": "info"
        }
        return severity_map.get(bandit_severity.upper(), "info")
    
    def _get_bandit_suggestion(self, result: Dict[str, Any]) -> str:
        """Get security suggestion for Bandit result."""
        test_id = result.get("test_id", "")
        test_name = result.get("test_name", "")
        more_info = result.get("more_info", "")
        
        # Common security suggestions
        suggestions = {
            "B101": "Use assert only in tests, not in production code",
            "B102": "Avoid exec() calls as they can execute arbitrary code",
            "B103": "Set umask to a restrictive value (0o077)",
            "B104": "Binding to all interfaces (0.0.0.0) can be a security risk",
            "B105": "Use secrets module for password/token generation",
            "B106": "Use secure random number generation",
            "B107": "Use HTTPS instead of HTTP for sensitive operations",
            "B108": "Use secure temp file creation",
            "B110": "Use try/except/pass can hide security issues",
            "B112": "Use try/except/continue can hide security issues",
            "B201": "Use of Flask debug mode in production",
            "B301": "Use pickle only with trusted data",
            "B302": "Use marshal only with trusted data",
            "B303": "Use MD5 only for non-cryptographic purposes",
            "B304": "Use SHA1 only for non-cryptographic purposes",
            "B305": "Use cryptographically secure hash functions",
            "B306": "Use mktemp_* functions securely",
            "B307": "Use eval() with extreme caution",
            "B308": "Use mark_safe() carefully in templates",
            "B309": "Use HTTPSConnection instead of HTTPConnection",
            "B310": "Use urllib.parse.quote() for URL encoding",
            "B311": "Use secrets module for random values",
            "B312": "Use secure random number generation",
            "B313": "Use secure XML parsing",
            "B314": "Use secure XML parsing",
            "B315": "Use secure XML parsing",
            "B316": "Use secure XML parsing",
            "B317": "Use secure XML parsing",
            "B318": "Use secure XML parsing",
            "B319": "Use secure XML parsing",
            "B320": "Use secure XML parsing",
            "B321": "Use secure FTP connection",
            "B322": "Use secure input validation",
            "B323": "Use secure random number generation",
            "B324": "Use secure hash functions",
            "B325": "Use secure temp file creation",
            "B401": "Use secure import practices",
            "B402": "Use secure import practices",
            "B403": "Use secure import practices",
            "B404": "Use secure subprocess calls",
            "B405": "Use secure import practices",
            "B406": "Use secure import practices",
            "B407": "Use secure XML parsing",
            "B408": "Use secure XML parsing",
            "B409": "Use secure XML parsing",
            "B410": "Use secure XML parsing",
            "B411": "Use secure XML parsing",
            "B412": "Use secure telnet connection",
            "B413": "Use secure encryption",
            "B501": "Use secure SSL/TLS configuration",
            "B502": "Use secure SSL/TLS configuration",
            "B503": "Use secure SSL/TLS configuration",
            "B504": "Use secure SSL/TLS configuration",
            "B505": "Use secure SSL/TLS configuration",
            "B506": "Use secure YAML loading",
            "B507": "Use secure SSH configuration",
            "B508": "Use secure SQL queries",
            "B509": "Use secure SQL queries",
            "B601": "Use secure shell escaping",
            "B602": "Use secure subprocess calls",
            "B603": "Use secure subprocess calls",
            "B604": "Use secure subprocess calls",
            "B605": "Use secure subprocess calls",
            "B606": "Use secure subprocess calls",
            "B607": "Use secure subprocess calls",
            "B608": "Use secure SQL queries",
            "B609": "Use secure SQL queries",
            "B610": "Use secure SQL queries",
            "B701": "Use secure Jinja2 templates",
            "B702": "Use secure test methods",
            "B703": "Use secure Django settings",
        }
        
        suggestion = suggestions.get(test_id, "")
        
        if not suggestion and test_name:
            if "password" in test_name.lower():
                suggestion = "Use secure password handling"
            elif "random" in test_name.lower():
                suggestion = "Use cryptographically secure random generation"
            elif "hash" in test_name.lower():
                suggestion = "Use secure hash functions"
            elif "sql" in test_name.lower():
                suggestion = "Use parameterized queries to prevent SQL injection"
            elif "xml" in test_name.lower():
                suggestion = "Use secure XML parsing to prevent XXE attacks"
            elif "ssl" in test_name.lower() or "tls" in test_name.lower():
                suggestion = "Use secure SSL/TLS configuration"
            else:
                suggestion = "Review security implications and apply secure coding practices"
        
        if more_info:
            suggestion += f" (More info: {more_info})"
        
        return suggestion
    
    def _calculate_security_metrics(self, issues: List[Issue]) -> Dict[str, Any]:
        """Calculate security metrics from issues."""
        metrics = {
            "total_vulnerabilities": len(issues),
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "categories": {},
            "confidence_levels": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "cwe_counts": {},
            "most_common_issues": {},
        }
        
        for issue in issues:
            # Count by severity
            if issue.severity == "error":
                metrics["high_severity"] += 1
            elif issue.severity == "warning":
                metrics["medium_severity"] += 1
            else:
                metrics["low_severity"] += 1
            
            # Count by test ID
            test_id = issue.rule
            metrics["most_common_issues"][test_id] = metrics["most_common_issues"].get(test_id, 0) + 1
            
            # Count by confidence
            if issue.metadata and "confidence" in issue.metadata:
                confidence = issue.metadata["confidence"]
                metrics["confidence_levels"][confidence] = metrics["confidence_levels"].get(confidence, 0) + 1
            
            # Count by CWE
            if issue.metadata and "cwe" in issue.metadata and issue.metadata["cwe"]:
                cwe = issue.metadata["cwe"]
                metrics["cwe_counts"][cwe] = metrics["cwe_counts"].get(cwe, 0) + 1
        
        # Calculate risk score
        metrics["risk_score"] = (
            metrics["high_severity"] * 10 +
            metrics["medium_severity"] * 5 +
            metrics["low_severity"] * 1
        )
        
        return metrics