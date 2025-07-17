"""
Base class for automated fixes in repository governance.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FixResult:
    """Result of an automated fix operation."""
    
    def __init__(self, success: bool, message: str, files_changed: List[str] = None, 
                 details: Dict[str, Any] = None):
        self.success = success
        self.message = message
        self.files_changed = files_changed or []
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "files_changed": self.files_changed,
            "details": self.details
        }


class AutoFixer(ABC):
    """Base class for automated repository fixes."""
    
    def __init__(self, root_path: Path, dry_run: bool = False):
        """Initialize the auto fixer."""
        self.root_path = root_path
        self.dry_run = dry_run
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_fix(self, violation: Dict[str, Any]) -> bool:
        """Check if this fixer can handle the given violation."""
        pass
    
    @abstractmethod
    def fix(self, violation: Dict[str, Any]) -> FixResult:
        """Apply the fix for the given violation."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the fixer."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this fixer does."""
        pass
    
    def get_python_files(self, directory: Path = None) -> List[Path]:
        """Get all Python files in the repository."""
        search_path = directory or self.root_path
        return list(search_path.rglob("*.py"))
    
    def get_all_files(self, directory: Path = None) -> List[Path]:
        """Get all files in the repository."""
        search_path = directory or self.root_path
        return [p for p in search_path.rglob("*") if p.is_file()]
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of a file before modification."""
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        if not self.dry_run:
            backup_path.write_bytes(file_path.read_bytes())
        return backup_path
    
    def remove_backup(self, backup_path: Path) -> None:
        """Remove a backup file."""
        if not self.dry_run and backup_path.exists():
            backup_path.unlink()
    
    def restore_backup(self, backup_path: Path) -> None:
        """Restore a file from backup."""
        if not self.dry_run and backup_path.exists():
            original_path = backup_path.with_suffix(backup_path.suffix.replace('.backup', ''))
            original_path.write_bytes(backup_path.read_bytes())
            backup_path.unlink()
    
    def safe_write_file(self, file_path: Path, content: str) -> bool:
        """Safely write content to a file with backup."""
        try:
            if not self.dry_run:
                backup_path = self.backup_file(file_path)
                try:
                    file_path.write_text(content, encoding='utf-8')
                    self.remove_backup(backup_path)
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to write {file_path}: {e}")
                    self.restore_backup(backup_path)
                    return False
            else:
                self.logger.info(f"DRY RUN: Would write {len(content)} characters to {file_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to backup {file_path}: {e}")
            return False
    
    def safe_delete_file(self, file_path: Path) -> bool:
        """Safely delete a file."""
        try:
            if not self.dry_run:
                if file_path.exists():
                    file_path.unlink()
                    return True
                else:
                    self.logger.warning(f"File not found for deletion: {file_path}")
                    return False
            else:
                self.logger.info(f"DRY RUN: Would delete {file_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    def safe_move_file(self, src_path: Path, dst_path: Path) -> bool:
        """Safely move a file."""
        try:
            if not self.dry_run:
                if src_path.exists():
                    # Ensure destination directory exists
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    src_path.rename(dst_path)
                    return True
                else:
                    self.logger.warning(f"Source file not found: {src_path}")
                    return False
            else:
                self.logger.info(f"DRY RUN: Would move {src_path} to {dst_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to move {src_path} to {dst_path}: {e}")
            return False
    
    def apply_fixes(self, violations: List[Dict[str, Any]]) -> List[FixResult]:
        """Apply fixes to a list of violations."""
        results = []
        
        for violation in violations:
            if self.can_fix(violation):
                try:
                    result = self.fix(violation)
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"Successfully fixed: {result.message}")
                    else:
                        self.logger.warning(f"Failed to fix: {result.message}")
                        
                except Exception as e:
                    self.logger.error(f"Error applying fix: {e}")
                    results.append(FixResult(
                        success=False,
                        message=f"Error applying fix: {str(e)}",
                        details={"violation": violation}
                    ))
            else:
                self.logger.debug(f"Cannot fix violation: {violation}")
        
        return results
    
    def get_fix_summary(self, results: List[FixResult]) -> Dict[str, Any]:
        """Get a summary of fix results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        all_files_changed = set()
        for result in successful:
            all_files_changed.update(result.files_changed)
        
        return {
            "fixer": self.name,
            "total_fixes_attempted": len(results),
            "successful_fixes": len(successful),
            "failed_fixes": len(failed),
            "files_changed": len(all_files_changed),
            "changed_files": list(all_files_changed),
            "dry_run": self.dry_run
        }