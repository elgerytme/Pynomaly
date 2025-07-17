"""isort adapter for import sorting analysis."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class IsortAdapter(ToolAdapter):
    """Adapter for isort import sorting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py", ".pyi"]
    
    def is_available(self) -> bool:
        """Check if isort is available in environment."""
        return shutil.which("isort") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run isort import analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="isort is not available. Install with: pip install isort"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = ["isort", "--check-only", "--diff"]
        
        # Add configuration
        isort_config = self.config.get("isort", {})
        
        if profile := isort_config.get("profile"):
            cmd.extend(["--profile", profile])
        
        if line_length := isort_config.get("line_length"):
            cmd.extend(["--line-length", str(line_length)])
        
        if multi_line := isort_config.get("multi_line_output"):
            cmd.extend(["--multi-line", str(multi_line)])
        
        if force_single_line := isort_config.get("force_single_line"):
            cmd.append("--force-single-line")
        
        if force_sort_within_sections := isort_config.get("force_sort_within_sections"):
            cmd.append("--force-sort-within-sections")
        
        if combine_as := isort_config.get("combine_as"):
            cmd.append("--combine-as")
        
        if combine_star := isort_config.get("combine_star"):
            cmd.append("--combine-star")
        
        if skip_gitignore := isort_config.get("skip_gitignore", True):
            cmd.append("--skip-gitignore")
        
        # Add file paths
        cmd.extend([str(f) for f in python_files])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse isort output
            issues = self._parse_isort_output(result.stdout, python_files)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "files_analyzed": len(python_files),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "import_metrics": self._calculate_import_metrics(issues),
                }
            )
            
        except Exception as e:
            self.logger.error(f"isort analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _parse_isort_output(self, output: str, files: List[Path]) -> List[Issue]:
        """Parse isort diff output."""
        issues = []
        
        if not output.strip():
            return issues
        
        # isort outputs diff format when imports need sorting
        # Split by file sections
        sections = output.split("--- ")
        
        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            # Extract file path from header
            file_line = lines[0]
            if '\t' in file_line:
                file_path_str = file_line.split('\t')[0]
            else:
                file_path_str = file_line.split()[0]
            
            # Find the matching file path
            file_path = None
            for f in files:
                if str(f).endswith(file_path_str) or file_path_str.endswith(str(f)):
                    file_path = f
                    break
            
            if not file_path:
                continue
            
            # Count import changes
            added_lines = 0
            removed_lines = 0
            import_changes = []
            
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    added_lines += 1
                    if 'import' in line:
                        import_changes.append(f"Added: {line[1:].strip()}")
                elif line.startswith('-') and not line.startswith('---'):
                    removed_lines += 1
                    if 'import' in line:
                        import_changes.append(f"Removed: {line[1:].strip()}")
            
            if added_lines > 0 or removed_lines > 0:
                issue = self._create_issue(
                    file=file_path,
                    line=1,
                    column=1,
                    message=f"Imports would be sorted by isort (+{added_lines} -{removed_lines} lines)",
                    rule="ISORT001",
                    severity="info",
                    fixable=True,
                    category="import_sorting",
                    suggestion="Run 'isort <file>' to sort imports",
                    metadata={
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                        "import_changes": import_changes,
                        "diff_section": section[:500]  # Truncate for storage
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _calculate_import_metrics(self, issues: List[Issue]) -> Dict[str, Any]:
        """Calculate import metrics from issues."""
        metrics = {
            "files_needing_sorting": len(issues),
            "total_import_changes": 0,
            "total_added_lines": 0,
            "total_removed_lines": 0,
            "import_organization_score": 0,
            "common_import_issues": {},
        }
        
        for issue in issues:
            if issue.metadata:
                metrics["total_added_lines"] += issue.metadata.get("added_lines", 0)
                metrics["total_removed_lines"] += issue.metadata.get("removed_lines", 0)
                
                # Analyze import changes
                import_changes = issue.metadata.get("import_changes", [])
                for change in import_changes:
                    if "from" in change and "import" in change:
                        metrics["common_import_issues"]["from_import"] = (
                            metrics["common_import_issues"].get("from_import", 0) + 1
                        )
                    elif "import" in change:
                        metrics["common_import_issues"]["direct_import"] = (
                            metrics["common_import_issues"].get("direct_import", 0) + 1
                        )
        
        metrics["total_import_changes"] = (
            metrics["total_added_lines"] + metrics["total_removed_lines"]
        )
        
        # Calculate import organization score (higher is better)
        if issues:
            # Score based on how many files need sorting
            total_files = len(set(issue.file for issue in issues))
            metrics["import_organization_score"] = max(0, 100 - (total_files * 15))
        else:
            metrics["import_organization_score"] = 100
        
        return metrics