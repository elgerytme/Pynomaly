"""
Fix suggester implementation.
"""

from pathlib import Path
from typing import List

from .config import get_suggested_location
from .models import Fix, FixType, Violation, ViolationType


class FixSuggester:
    """Suggests fixes for structure violations."""
    
    def __init__(self, violations: List[Violation]):
        self.violations = violations
    
    def suggest(self) -> List[Fix]:
        """Suggest fixes for all violations."""
        fixes = []
        
        for violation in self.violations:
            fix = self._suggest_fix_for_violation(violation)
            if fix:
                fixes.append(fix)
        
        return fixes
    
    def _suggest_fix_for_violation(self, violation: Violation) -> Fix:
        """Suggest a fix for a specific violation."""
        if violation.type == ViolationType.STRAY_FILE:
            return self._suggest_file_relocation_fix(violation)
        
        elif violation.type == ViolationType.STRAY_DIRECTORY:
            return self._suggest_directory_relocation_fix(violation)
        
        elif violation.type == ViolationType.MISSING_INIT:
            return self._suggest_init_file_creation_fix(violation)
        
        elif violation.type == ViolationType.MISSING_LAYER:
            return self._suggest_layer_creation_fix(violation)
        
        elif violation.type == ViolationType.EMPTY_DIRECTORY:
            return self._suggest_empty_directory_fix(violation)
        
        elif violation.type in [
            ViolationType.NAMING_CONVENTION,
            ViolationType.INVALID_DEPENDENCY,
            ViolationType.DOMAIN_PURITY,
            ViolationType.CIRCULAR_IMPORT,
        ]:
            # These require manual intervention
            return self._suggest_manual_fix(violation)
        
        return None
    
    def _suggest_file_relocation_fix(self, violation: Violation) -> Fix:
        """Suggest relocation fix for a stray file."""
        if not violation.file_path:
            return None
        
        filename = violation.file_path.name
        suggested_location = get_suggested_location(filename)
        
        if suggested_location == "DELETE":
            return Fix(
                type=FixType.DELETE_FILE,
                description=f"Delete stray file: {filename}",
                source_path=violation.file_path,
                risk_level="high",
            )
        
        elif suggested_location.startswith("REVIEW"):
            return Fix(
                type=FixType.MOVE_FILE,
                description=f"Move {filename} to appropriate location (manual review needed)",
                source_path=violation.file_path,
                target_path=None,  # Requires manual decision
                risk_level="medium",
            )
        
        else:
            # Find the actual target directory
            root_path = violation.file_path.parent
            while root_path.name != violation.file_path.parts[0]:
                root_path = root_path.parent
            
            target_path = root_path / suggested_location / filename
            
            return Fix(
                type=FixType.MOVE_FILE,
                description=f"Move {filename} to {suggested_location}",
                source_path=violation.file_path,
                target_path=target_path,
                risk_level="medium",
            )
    
    def _suggest_directory_relocation_fix(self, violation: Violation) -> Fix:
        """Suggest relocation fix for a stray directory."""
        if not violation.directory_path:
            return None
        
        dir_name = violation.directory_path.name
        
        # Suggest based on common patterns
        if any(pattern in dir_name.lower() for pattern in ["test", "testing"]):
            # Find project root
            root_path = violation.directory_path.parent
            while root_path.name != violation.directory_path.parts[0]:
                root_path = root_path.parent
            
            target_path = root_path / "tests" / dir_name
            
            return Fix(
                type=FixType.MOVE_DIRECTORY,
                description=f"Move {dir_name}/ to tests/",
                source_path=violation.directory_path,
                target_path=target_path,
                risk_level="medium",
            )
        
        elif any(pattern in dir_name.lower() for pattern in ["temp", "tmp", "env", "venv"]):
            return Fix(
                type=FixType.DELETE_DIRECTORY,
                description=f"Delete temporary directory: {dir_name}/",
                source_path=violation.directory_path,
                risk_level="high",
            )
        
        else:
            return Fix(
                type=FixType.MOVE_DIRECTORY,
                description=f"Move {dir_name}/ to appropriate location (manual review needed)",
                source_path=violation.directory_path,
                target_path=None,  # Requires manual decision
                risk_level="medium",
            )
    
    def _suggest_init_file_creation_fix(self, violation: Violation) -> Fix:
        """Suggest creation of __init__.py file."""
        if not violation.directory_path:
            return None
        
        target_path = violation.directory_path / "__init__.py"
        
        return Fix(
            type=FixType.CREATE_FILE,
            description=f"Create __init__.py in {violation.directory_path.name}",
            target_path=target_path,
            content='"""\\n{} package.\\n"""\\n'.format(violation.directory_path.name.replace('_', ' ').title()),
            risk_level="low",
        )
    
    def _suggest_layer_creation_fix(self, violation: Violation) -> Fix:
        """Suggest creation of missing layer directory."""
        if not violation.directory_path:
            return None
        
        return Fix(
            type=FixType.CREATE_DIRECTORY,
            description=f"Create missing layer directory: {violation.directory_path.name}",
            target_path=violation.directory_path,
            risk_level="low",
        )
    
    def _suggest_empty_directory_fix(self, violation: Violation) -> Fix:
        """Suggest fix for empty directory."""
        if not violation.directory_path:
            return None
        
        return Fix(
            type=FixType.DELETE_DIRECTORY,
            description=f"Delete empty directory: {violation.directory_path.name}",
            source_path=violation.directory_path,
            risk_level="medium",
        )
    
    def _suggest_manual_fix(self, violation: Violation) -> Fix:
        """Suggest manual fix for violations that require human intervention."""
        if violation.type == ViolationType.NAMING_CONVENTION:
            return Fix(
                type=FixType.MODIFY_FILE,
                description=f"Fix naming convention violation: {violation.message}",
                source_path=violation.file_path,
                risk_level="low",
            )
        
        elif violation.type == ViolationType.INVALID_DEPENDENCY:
            return Fix(
                type=FixType.MODIFY_FILE,
                description=f"Fix dependency violation: {violation.message}",
                source_path=violation.file_path,
                risk_level="medium",
            )
        
        elif violation.type == ViolationType.DOMAIN_PURITY:
            return Fix(
                type=FixType.MODIFY_FILE,
                description=f"Fix domain purity violation: {violation.message}",
                source_path=violation.file_path,
                risk_level="medium",
            )
        
        elif violation.type == ViolationType.CIRCULAR_IMPORT:
            return Fix(
                type=FixType.MODIFY_FILE,
                description=f"Fix circular import: {violation.message}",
                source_path=violation.file_path,
                risk_level="high",
            )
        
        return None


def suggest_fixes(violations: List[Violation]) -> List[Fix]:
    """
    Suggest fixes for the detected violations.
    
    Args:
        violations: List of detected violations.
    
    Returns:
        List[Fix]: List of suggested fixes.
    """
    suggester = FixSuggester(violations)
    return suggester.suggest()
