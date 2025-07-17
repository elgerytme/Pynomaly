"""Black adapter for code formatting analysis."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class BlackAdapter(ToolAdapter):
    """Adapter for Black code formatting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py", ".pyi"]
    
    def is_available(self) -> bool:
        """Check if Black is available in environment."""
        return shutil.which("black") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run Black formatting analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="Black is not available. Install with: pip install black"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = ["black", "--check", "--diff"]
        
        # Add configuration
        black_config = self.config.get("black", {})
        
        if line_length := black_config.get("line_length"):
            cmd.extend(["--line-length", str(line_length)])
        
        if target_version := black_config.get("target_version"):
            cmd.extend(["--target-version", target_version])
        
        if skip_string_normalization := black_config.get("skip_string_normalization"):
            cmd.append("--skip-string-normalization")
        
        if skip_magic_trailing_comma := black_config.get("skip_magic_trailing_comma"):
            cmd.append("--skip-magic-trailing-comma")
        
        if preview := black_config.get("preview"):
            cmd.append("--preview")
        
        # Add file paths
        cmd.extend([str(f) for f in python_files])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse Black output
            issues = self._parse_black_output(result.stdout, python_files)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "files_analyzed": len(python_files),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "formatting_metrics": self._calculate_formatting_metrics(issues),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Black analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _parse_black_output(self, output: str, files: List[Path]) -> List[Issue]:
        """Parse Black diff output."""
        issues = []
        
        if not output.strip():
            return issues
        
        # Black outputs diff format when files need formatting
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
            
            # Count formatting changes
            added_lines = 0
            removed_lines = 0
            
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    added_lines += 1
                elif line.startswith('-') and not line.startswith('---'):
                    removed_lines += 1
            
            if added_lines > 0 or removed_lines > 0:
                issue = self._create_issue(
                    file=file_path,
                    line=1,
                    column=1,
                    message=f"File would be reformatted by Black (+{added_lines} -{removed_lines} lines)",
                    rule="BLACK001",
                    severity="info",
                    fixable=True,
                    category="formatting",
                    suggestion="Run 'black <file>' to format the file",
                    metadata={
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                        "diff_section": section[:500]  # Truncate for storage
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _calculate_formatting_metrics(self, issues: List[Issue]) -> Dict[str, Any]:
        """Calculate formatting metrics from issues."""
        metrics = {
            "files_needing_formatting": len(issues),
            "total_line_changes": 0,
            "total_added_lines": 0,
            "total_removed_lines": 0,
            "formatting_consistency_score": 0,
        }
        
        for issue in issues:
            if issue.metadata:
                metrics["total_added_lines"] += issue.metadata.get("added_lines", 0)
                metrics["total_removed_lines"] += issue.metadata.get("removed_lines", 0)
        
        metrics["total_line_changes"] = (
            metrics["total_added_lines"] + metrics["total_removed_lines"]
        )
        
        # Calculate formatting consistency score (higher is better)
        if issues:
            # Score based on how many files need formatting
            total_files = len(set(issue.file for issue in issues))
            metrics["formatting_consistency_score"] = max(0, 100 - (total_files * 10))
        else:
            metrics["formatting_consistency_score"] = 100
        
        return metrics