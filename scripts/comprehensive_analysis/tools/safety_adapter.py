"""Safety adapter for dependency vulnerability scanning."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class SafetyAdapter(ToolAdapter):
    """Adapter for Safety dependency vulnerability scanning."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".txt", ".in", ".pip", ".py"]  # requirements files and setup.py
    
    def is_available(self) -> bool:
        """Check if Safety is available in environment."""
        return shutil.which("safety") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run Safety dependency analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="Safety is not available. Install with: pip install safety"
            )
        
        # Find requirements files and setup.py
        dependency_files = self._find_dependency_files(files)
        if not dependency_files:
            # If no dependency files found, scan installed packages
            return await self._scan_installed_packages()
        
        # Analyze each dependency file
        all_issues = []
        total_execution_time = 0
        files_analyzed = 0
        
        for dep_file in dependency_files:
            try:
                result = await self._analyze_dependency_file(dep_file)
                all_issues.extend(result.issues)
                total_execution_time += result.execution_time
                files_analyzed += 1
            except Exception as e:
                self.logger.error(f"Failed to analyze {dep_file}: {e}")
        
        return AnalysisResult(
            tool=self.name,
            issues=all_issues,
            execution_time=total_execution_time,
            success=True,
            metadata={
                "files_analyzed": files_analyzed,
                "dependency_files": [str(f) for f in dependency_files],
                "vulnerability_metrics": self._calculate_vulnerability_metrics(all_issues),
            }
        )
    
    def _find_dependency_files(self, files: List[Path]) -> List[Path]:
        """Find dependency files in the provided files."""
        dependency_files = []
        
        for file in files:
            if file.name in ["requirements.txt", "requirements-dev.txt", "requirements-test.txt", 
                           "requirements.in", "setup.py", "pyproject.toml", "Pipfile", "poetry.lock"]:
                dependency_files.append(file)
            elif file.name.endswith((".txt", ".in")) and "requirements" in file.name:
                dependency_files.append(file)
        
        return dependency_files
    
    async def _scan_installed_packages(self) -> AnalysisResult:
        """Scan installed packages when no requirements files are found."""
        cmd = ["safety", "check", "--json", "--full-report"]
        
        safety_config = self.config.get("safety", {})
        
        if ignore_ids := safety_config.get("ignore"):
            cmd.extend(["--ignore", ",".join(ignore_ids)])
        
        if output_format := safety_config.get("output"):
            cmd.extend(["--output", output_format])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse JSON output
            issues = self._parse_safety_output(result.stdout)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "scan_type": "installed_packages",
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "vulnerability_metrics": self._calculate_vulnerability_metrics(issues),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Safety installed packages scan failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_dependency_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a specific dependency file."""
        if file_path.name == "setup.py":
            return await self._analyze_setup_py(file_path)
        elif file_path.name == "pyproject.toml":
            return await self._analyze_pyproject_toml(file_path)
        else:
            return await self._analyze_requirements_file(file_path)
    
    async def _analyze_requirements_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a requirements file."""
        cmd = ["safety", "check", "--json", "--file", str(file_path)]
        
        safety_config = self.config.get("safety", {})
        
        if ignore_ids := safety_config.get("ignore"):
            cmd.extend(["--ignore", ",".join(ignore_ids)])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse JSON output
            issues = self._parse_safety_output(result.stdout, file_path)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "scan_type": "requirements_file",
                    "file": str(file_path),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Safety requirements analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_setup_py(self, file_path: Path) -> AnalysisResult:
        """Analyze setup.py file by extracting requirements."""
        # For setup.py, we need to extract requirements first
        # This is a simplified approach - in production, you'd want more robust parsing
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for install_requires or requirements
            issues = []
            
            # Create a temporary requirements file from setup.py
            temp_req_file = file_path.parent / "temp_requirements.txt"
            
            # Extract requirements (simplified)
            requirements = self._extract_requirements_from_setup_py(content)
            
            if requirements:
                with open(temp_req_file, 'w') as f:
                    f.write('\n'.join(requirements))
                
                # Analyze the temporary requirements file
                result = await self._analyze_requirements_file(temp_req_file)
                
                # Clean up
                temp_req_file.unlink()
                
                # Update file references in issues
                for issue in result.issues:
                    issue.file = file_path
                
                return result
            else:
                return AnalysisResult(
                    tool=self.name,
                    success=True,
                    metadata={"message": "No requirements found in setup.py"}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to analyze setup.py: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_pyproject_toml(self, file_path: Path) -> AnalysisResult:
        """Analyze pyproject.toml file."""
        try:
            import toml
            
            with open(file_path, 'r', encoding='utf-8') as f:
                pyproject_data = toml.load(f)
            
            # Extract dependencies
            dependencies = []
            
            # Poetry dependencies
            if 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
                poetry_deps = pyproject_data['tool']['poetry'].get('dependencies', {})
                dependencies.extend([f"{name}>={version}" if version != "*" else name 
                                   for name, version in poetry_deps.items() if name != "python"])
            
            # PEP 621 dependencies
            if 'project' in pyproject_data:
                project_deps = pyproject_data['project'].get('dependencies', [])
                dependencies.extend(project_deps)
            
            if dependencies:
                # Create temporary requirements file
                temp_req_file = file_path.parent / "temp_requirements.txt"
                
                with open(temp_req_file, 'w') as f:
                    f.write('\n'.join(dependencies))
                
                # Analyze the temporary requirements file
                result = await self._analyze_requirements_file(temp_req_file)
                
                # Clean up
                temp_req_file.unlink()
                
                # Update file references in issues
                for issue in result.issues:
                    issue.file = file_path
                
                return result
            else:
                return AnalysisResult(
                    tool=self.name,
                    success=True,
                    metadata={"message": "No dependencies found in pyproject.toml"}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to analyze pyproject.toml: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _extract_requirements_from_setup_py(self, content: str) -> List[str]:
        """Extract requirements from setup.py content."""
        requirements = []
        
        # Simple regex-based extraction (could be improved)
        import re
        
        # Look for install_requires
        install_requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if install_requires_match:
            req_text = install_requires_match.group(1)
            # Extract quoted strings
            req_matches = re.findall(r'["\']([^"\']+)["\']', req_text)
            requirements.extend(req_matches)
        
        # Look for requirements.txt references
        req_file_match = re.search(r'open\(["\']([^"\']*requirements[^"\']*\.txt)["\']', content)
        if req_file_match:
            req_file = req_file_match.group(1)
            try:
                with open(req_file, 'r') as f:
                    requirements.extend(f.read().splitlines())
            except FileNotFoundError:
                pass
        
        return [req.strip() for req in requirements if req.strip() and not req.strip().startswith('#')]
    
    def _parse_safety_output(self, output: str, source_file: Path = None) -> List[Issue]:
        """Parse Safety JSON output."""
        issues = []
        
        if not output.strip():
            return issues
        
        try:
            # Safety output can be a list of vulnerabilities or error message
            if output.strip().startswith('['):
                vulnerabilities = json.loads(output)
            else:
                # Try to parse as single object
                data = json.loads(output)
                vulnerabilities = data.get('vulnerabilities', [])
            
            for vuln in vulnerabilities:
                issue = self._create_issue(
                    file=source_file or Path("requirements.txt"),
                    line=1,  # Safety doesn't provide line numbers
                    column=1,
                    message=self._format_vulnerability_message(vuln),
                    rule=vuln.get("id", "SAFETY"),
                    severity=self._map_safety_severity(vuln.get("severity", "UNKNOWN")),
                    fixable=True,  # Dependencies can usually be updated
                    category="security",
                    suggestion=self._get_safety_suggestion(vuln),
                    metadata={
                        "package_name": vuln.get("package_name"),
                        "installed_version": vuln.get("installed_version"),
                        "affected_versions": vuln.get("affected_versions"),
                        "fixed_versions": vuln.get("fixed_versions"),
                        "cve": vuln.get("cve"),
                        "severity": vuln.get("severity"),
                        "more_info_url": vuln.get("more_info_url"),
                    }
                )
                issues.append(issue)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Safety JSON output: {e}")
            self.logger.debug(f"Output was: {output}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing Safety output: {e}")
            self.logger.debug(f"Output was: {output}")
        
        return issues
    
    def _format_vulnerability_message(self, vuln: Dict[str, Any]) -> str:
        """Format vulnerability message."""
        package = vuln.get("package_name", "unknown")
        version = vuln.get("installed_version", "unknown")
        vuln_id = vuln.get("id", "")
        
        message = f"Vulnerability {vuln_id} in {package} {version}"
        
        if advisory := vuln.get("advisory"):
            message += f": {advisory}"
        
        return message
    
    def _map_safety_severity(self, safety_severity: str) -> str:
        """Map Safety severity to standard severity."""
        severity_map = {
            "CRITICAL": "error",
            "HIGH": "error",
            "MEDIUM": "warning",
            "LOW": "info",
            "UNKNOWN": "info"
        }
        return severity_map.get(safety_severity.upper(), "info")
    
    def _get_safety_suggestion(self, vuln: Dict[str, Any]) -> str:
        """Get security suggestion for Safety vulnerability."""
        package = vuln.get("package_name", "unknown")
        current_version = vuln.get("installed_version", "unknown")
        fixed_versions = vuln.get("fixed_versions", [])
        
        if fixed_versions:
            # Get the latest fixed version
            latest_fix = max(fixed_versions) if fixed_versions else None
            if latest_fix:
                return f"Update {package} from {current_version} to {latest_fix} or later"
        
        return f"Update {package} to a version that fixes this vulnerability"
    
    def _calculate_vulnerability_metrics(self, issues: List[Issue]) -> Dict[str, Any]:
        """Calculate vulnerability metrics from issues."""
        metrics = {
            "total_vulnerabilities": len(issues),
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "low_vulnerabilities": 0,
            "packages_affected": set(),
            "cve_counts": {},
            "severity_distribution": {},
            "most_vulnerable_packages": {},
        }
        
        for issue in issues:
            # Count by severity
            if issue.severity == "error":
                if issue.metadata and issue.metadata.get("severity") == "CRITICAL":
                    metrics["critical_vulnerabilities"] += 1
                else:
                    metrics["high_vulnerabilities"] += 1
            elif issue.severity == "warning":
                metrics["medium_vulnerabilities"] += 1
            else:
                metrics["low_vulnerabilities"] += 1
            
            # Track affected packages
            if issue.metadata and "package_name" in issue.metadata:
                package_name = issue.metadata["package_name"]
                metrics["packages_affected"].add(package_name)
                metrics["most_vulnerable_packages"][package_name] = (
                    metrics["most_vulnerable_packages"].get(package_name, 0) + 1
                )
            
            # Count CVEs
            if issue.metadata and "cve" in issue.metadata and issue.metadata["cve"]:
                cve = issue.metadata["cve"]
                metrics["cve_counts"][cve] = metrics["cve_counts"].get(cve, 0) + 1
            
            # Severity distribution
            severity = issue.metadata.get("severity", "UNKNOWN") if issue.metadata else "UNKNOWN"
            metrics["severity_distribution"][severity] = (
                metrics["severity_distribution"].get(severity, 0) + 1
            )
        
        # Convert set to count
        metrics["packages_affected"] = len(metrics["packages_affected"])
        
        # Calculate risk score
        metrics["risk_score"] = (
            metrics["critical_vulnerabilities"] * 50 +
            metrics["high_vulnerabilities"] * 25 +
            metrics["medium_vulnerabilities"] * 10 +
            metrics["low_vulnerabilities"] * 5
        )
        
        return metrics