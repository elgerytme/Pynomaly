"""Vulture adapter for dead code detection."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import re

from .adapter_base import ToolAdapter, AnalysisResult, Issue


class VultureAdapter(ToolAdapter):
    """Adapter for Vulture dead code detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".py"]
    
    def is_available(self) -> bool:
        """Check if Vulture is available in environment."""
        return shutil.which("vulture") is not None
    
    async def analyze(self, files: List[Path]) -> AnalysisResult:
        """Run Vulture dead code analysis on files."""
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message="Vulture is not available. Install with: pip install vulture"
            )
        
        # Filter to supported files
        python_files = self._filter_files(files)
        if not python_files:
            return AnalysisResult(tool=self.name, success=True)
        
        # Build command
        cmd = ["vulture", "--min-confidence", "60"]
        
        # Add configuration
        vulture_config = self.config.get("vulture", {})
        
        if min_confidence := vulture_config.get("min_confidence"):
            cmd.extend(["--min-confidence", str(min_confidence)])
        
        if exclude_patterns := vulture_config.get("exclude"):
            cmd.extend(["--exclude", ",".join(exclude_patterns)])
        
        if sort_by := vulture_config.get("sort_by"):
            cmd.extend(["--sort-by", sort_by])
        
        if vulture_config.get("make_whitelist"):
            cmd.append("--make-whitelist")
        
        if vulture_config.get("verbose"):
            cmd.append("--verbose")
        
        # Add file paths
        cmd.extend([str(f) for f in python_files])
        
        try:
            import time
            start_time = time.time()
            
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse Vulture output
            issues = self._parse_vulture_output(result.stdout)
            
            return AnalysisResult(
                tool=self.name,
                issues=issues,
                execution_time=execution_time,
                success=True,
                metadata={
                    "files_analyzed": len(python_files),
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "dead_code_metrics": self._calculate_dead_code_metrics(issues),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Vulture analysis failed: {e}")
            return AnalysisResult(
                tool=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _parse_vulture_output(self, output: str) -> List[Issue]:
        """Parse Vulture output."""
        issues = []
        
        if not output.strip():
            return issues
        
        # Vulture output format: file:line: message (confidence%)
        # Example: app.py:42: unused function 'calculate' (90% confidence)
        
        for line in output.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Parse the line
            match = re.match(r'(.+?):(\d+):\s*(.+?)\s*\((\d+)%\s*confidence\)', line)
            if match:
                file_path, line_num, message, confidence = match.groups()
                
                # Extract item type and name from message
                item_type, item_name = self._parse_message(message)
                
                issue = self._create_issue(
                    file=Path(file_path),
                    line=int(line_num),
                    column=1,  # Vulture doesn't provide column info
                    message=message,
                    rule=f"VULTURE_{item_type.upper()}",
                    severity=self._map_vulture_severity(item_type, int(confidence)),
                    fixable=True,  # Dead code can usually be removed
                    category="dead_code",
                    suggestion=self._get_vulture_suggestion(item_type, item_name),
                    metadata={
                        "confidence": int(confidence),
                        "item_type": item_type,
                        "item_name": item_name,
                        "dead_code_category": self._categorize_dead_code(item_type),
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _parse_message(self, message: str) -> tuple[str, str]:
        """Parse Vulture message to extract item type and name."""
        # Common patterns:
        # "unused function 'calculate'"
        # "unused variable 'x'"
        # "unused import 'sys'"
        # "unused class 'MyClass'"
        # "unused method 'process'"
        # "unused attribute 'value'"
        # "unused property 'name'"
        
        # Extract using regex
        patterns = [
            r'unused (function|method|class|variable|import|attribute|property) [\'"]([^\'"]+)[\'"]',
            r'unused (function|method|class|variable|import|attribute|property) ([a-zA-Z_][a-zA-Z0-9_]*)',
            r'unused ([a-zA-Z_][a-zA-Z0-9_]*)',  # fallback
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                if len(match.groups()) == 2:
                    return match.group(1), match.group(2)
                else:
                    return "unknown", match.group(1)
        
        # If no pattern matches, try to extract the item type
        if "function" in message:
            return "function", self._extract_name_from_message(message)
        elif "method" in message:
            return "method", self._extract_name_from_message(message)
        elif "class" in message:
            return "class", self._extract_name_from_message(message)
        elif "variable" in message:
            return "variable", self._extract_name_from_message(message)
        elif "import" in message:
            return "import", self._extract_name_from_message(message)
        elif "attribute" in message:
            return "attribute", self._extract_name_from_message(message)
        elif "property" in message:
            return "property", self._extract_name_from_message(message)
        else:
            return "unknown", "unknown"
    
    def _extract_name_from_message(self, message: str) -> str:
        """Extract name from message using quotes or last word."""
        # Look for quoted name
        quote_match = re.search(r'[\'"]([^\'"]+)[\'"]', message)
        if quote_match:
            return quote_match.group(1)
        
        # Fallback to last word
        words = message.split()
        if words:
            return words[-1]
        
        return "unknown"
    
    def _map_vulture_severity(self, item_type: str, confidence: int) -> str:
        """Map Vulture findings to severity based on type and confidence."""
        # High confidence dead code is more severe
        if confidence >= 90:
            return "warning"
        elif confidence >= 70:
            return "info"
        else:
            return "info"
    
    def _categorize_dead_code(self, item_type: str) -> str:
        """Categorize dead code by type."""
        categories = {
            "function": "unused_function",
            "method": "unused_method",
            "class": "unused_class",
            "variable": "unused_variable",
            "import": "unused_import",
            "attribute": "unused_attribute",
            "property": "unused_property",
        }
        return categories.get(item_type, "unknown")
    
    def _get_vulture_suggestion(self, item_type: str, item_name: str) -> str:
        """Get suggestion for dead code removal."""
        suggestions = {
            "function": f"Remove unused function '{item_name}' or mark as private with leading underscore",
            "method": f"Remove unused method '{item_name}' or mark as private with leading underscore",
            "class": f"Remove unused class '{item_name}' or ensure it's used in the codebase",
            "variable": f"Remove unused variable '{item_name}' or use it in the code",
            "import": f"Remove unused import '{item_name}' to reduce dependencies",
            "attribute": f"Remove unused attribute '{item_name}' or use it in the class",
            "property": f"Remove unused property '{item_name}' or use it in the class",
        }
        
        return suggestions.get(item_type, f"Review and remove unused {item_type} '{item_name}' if not needed")
    
    def _calculate_dead_code_metrics(self, issues: List[Issue]) -> Dict[str, Any]:
        """Calculate dead code metrics from issues."""
        metrics = {
            "total_dead_code_items": len(issues),
            "unused_functions": 0,
            "unused_methods": 0,
            "unused_classes": 0,
            "unused_variables": 0,
            "unused_imports": 0,
            "unused_attributes": 0,
            "unused_properties": 0,
            "high_confidence_items": 0,
            "medium_confidence_items": 0,
            "low_confidence_items": 0,
            "files_with_dead_code": set(),
            "dead_code_by_file": {},
            "confidence_distribution": {},
            "most_common_dead_code": {},
        }
        
        for issue in issues:
            # Count by type
            if issue.metadata and "item_type" in issue.metadata:
                item_type = issue.metadata["item_type"]
                key = f"unused_{item_type}s"
                if key in metrics:
                    metrics[key] += 1
            
            # Count by confidence
            if issue.metadata and "confidence" in issue.metadata:
                confidence = issue.metadata["confidence"]
                if confidence >= 90:
                    metrics["high_confidence_items"] += 1
                elif confidence >= 70:
                    metrics["medium_confidence_items"] += 1
                else:
                    metrics["low_confidence_items"] += 1
                
                # Confidence distribution
                conf_range = f"{confidence//10*10}-{confidence//10*10+9}%"
                metrics["confidence_distribution"][conf_range] = (
                    metrics["confidence_distribution"].get(conf_range, 0) + 1
                )
            
            # Track files with dead code
            metrics["files_with_dead_code"].add(str(issue.file))
            
            # Count dead code by file
            file_str = str(issue.file)
            metrics["dead_code_by_file"][file_str] = (
                metrics["dead_code_by_file"].get(file_str, 0) + 1
            )
            
            # Count most common dead code types
            if issue.metadata and "dead_code_category" in issue.metadata:
                category = issue.metadata["dead_code_category"]
                metrics["most_common_dead_code"][category] = (
                    metrics["most_common_dead_code"].get(category, 0) + 1
                )
        
        # Convert set to count
        metrics["files_with_dead_code"] = len(metrics["files_with_dead_code"])
        
        # Calculate dead code percentage (rough estimate)
        total_items = (
            metrics["unused_functions"] +
            metrics["unused_methods"] +
            metrics["unused_classes"] +
            metrics["unused_variables"] +
            metrics["unused_imports"] +
            metrics["unused_attributes"] +
            metrics["unused_properties"]
        )
        
        metrics["dead_code_percentage"] = (
            round(total_items / max(1, total_items * 10) * 100, 2)  # Rough estimate
        )
        
        # Calculate cleanliness score (higher is better)
        metrics["code_cleanliness_score"] = max(0, 100 - (
            metrics["high_confidence_items"] * 10 +
            metrics["medium_confidence_items"] * 5 +
            metrics["low_confidence_items"] * 2
        ))
        
        return metrics