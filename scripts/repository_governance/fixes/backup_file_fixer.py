"""
Automated fixer for backup file cleanup.
"""

import re
from pathlib import Path
from typing import Dict, List, Any

from .auto_fixer import AutoFixer, FixResult


class BackupFileFixer(AutoFixer):
    """Fixer for cleaning up backup files and temporary artifacts."""
    
    def __init__(self, root_path: Path, dry_run: bool = False):
        """Initialize the backup file fixer."""
        super().__init__(root_path, dry_run)
        
        # Common backup file patterns
        self.backup_patterns = [
            r'\.bak$',
            r'\.backup$',
            r'\.orig$',
            r'\.old$',
            r'~$',
            r'\.tmp$',
            r'\.temp$',
            r'\.swp$',
            r'\.swo$',
            r'\.DS_Store$',
            r'Thumbs\.db$',
            r'\.pyc$',
            r'__pycache__',
            r'\.git/.*\.lock$',
            r'\.coverage$',
            r'\.pytest_cache',
            r'\.mypy_cache',
            r'\.ruff_cache',
            r'\.tox',
            r'\.venv',
            r'venv/',
            r'env/',
            r'\.env$',
            r'node_modules/',
            r'build/',
            r'dist/',
            r'\.egg-info/',
            r'\.wheel$',
            r'\.tar\.gz$',
            r'\.zip$',
            r'\.7z$',
            r'\.log$',
            r'\.out$',
            r'\.err$',
            r'core\.\d+$',
            r'\.pid$',
            r'\.lock$',
            r'\.lockfile$'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.backup_patterns]
    
    @property
    def name(self) -> str:
        """Name of the fixer."""
        return "BackupFileFixer"
    
    @property
    def description(self) -> str:
        """Description of what this fixer does."""
        return "Removes backup files, temporary files, and build artifacts"
    
    def can_fix(self, violation: Dict[str, Any]) -> bool:
        """Check if this fixer can handle the given violation."""
        return violation.get("type") in ["backup_files", "temp_files", "build_artifacts"]
    
    def fix(self, violation: Dict[str, Any]) -> FixResult:
        """Apply the fix for the given violation."""
        violation_type = violation.get("type")
        
        if violation_type == "backup_files":
            return self._fix_backup_files(violation)
        elif violation_type == "temp_files":
            return self._fix_temp_files(violation)
        elif violation_type == "build_artifacts":
            return self._fix_build_artifacts(violation)
        else:
            return FixResult(
                success=False,
                message=f"Unknown violation type: {violation_type}"
            )
    
    def _fix_backup_files(self, violation: Dict[str, Any]) -> FixResult:
        """Fix backup file violations."""
        files_to_remove = violation.get("files", [])
        removed_files = []
        failed_files = []
        
        for file_path_str in files_to_remove:
            file_path = Path(file_path_str)
            
            # Safety check - make sure it's actually a backup file
            if self._is_backup_file(file_path):
                if self.safe_delete_file(file_path):
                    removed_files.append(file_path_str)
                else:
                    failed_files.append(file_path_str)
            else:
                self.logger.warning(f"Skipping non-backup file: {file_path}")
                failed_files.append(file_path_str)
        
        success = len(failed_files) == 0
        message = f"Removed {len(removed_files)} backup files"
        if failed_files:
            message += f", failed to remove {len(failed_files)} files"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=removed_files,
            details={
                "removed_files": removed_files,
                "failed_files": failed_files
            }
        )
    
    def _fix_temp_files(self, violation: Dict[str, Any]) -> FixResult:
        """Fix temporary file violations."""
        files_to_remove = violation.get("files", [])
        removed_files = []
        failed_files = []
        
        for file_path_str in files_to_remove:
            file_path = Path(file_path_str)
            
            # Safety check - make sure it's actually a temp file
            if self._is_temp_file(file_path):
                if self.safe_delete_file(file_path):
                    removed_files.append(file_path_str)
                else:
                    failed_files.append(file_path_str)
            else:
                self.logger.warning(f"Skipping non-temp file: {file_path}")
                failed_files.append(file_path_str)
        
        success = len(failed_files) == 0
        message = f"Removed {len(removed_files)} temporary files"
        if failed_files:
            message += f", failed to remove {len(failed_files)} files"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=removed_files,
            details={
                "removed_files": removed_files,
                "failed_files": failed_files
            }
        )
    
    def _fix_build_artifacts(self, violation: Dict[str, Any]) -> FixResult:
        """Fix build artifact violations."""
        directories_to_remove = violation.get("directories", [])
        files_to_remove = violation.get("files", [])
        removed_items = []
        failed_items = []
        
        # Remove directories
        for dir_path_str in directories_to_remove:
            dir_path = Path(dir_path_str)
            
            if self._is_build_artifact_dir(dir_path):
                if self._remove_directory(dir_path):
                    removed_items.append(dir_path_str)
                else:
                    failed_items.append(dir_path_str)
            else:
                self.logger.warning(f"Skipping non-build directory: {dir_path}")
                failed_items.append(dir_path_str)
        
        # Remove files
        for file_path_str in files_to_remove:
            file_path = Path(file_path_str)
            
            if self._is_build_artifact_file(file_path):
                if self.safe_delete_file(file_path):
                    removed_items.append(file_path_str)
                else:
                    failed_items.append(file_path_str)
            else:
                self.logger.warning(f"Skipping non-build file: {file_path}")
                failed_items.append(file_path_str)
        
        success = len(failed_items) == 0
        message = f"Removed {len(removed_items)} build artifacts"
        if failed_items:
            message += f", failed to remove {len(failed_items)} items"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=removed_items,
            details={
                "removed_items": removed_items,
                "failed_items": failed_items
            }
        )
    
    def _is_backup_file(self, file_path: Path) -> bool:
        """Check if a file is a backup file."""
        filename = file_path.name
        return any(pattern.search(filename) for pattern in self.compiled_patterns[:6])  # First 6 patterns are backup patterns
    
    def _is_temp_file(self, file_path: Path) -> bool:
        """Check if a file is a temporary file."""
        filename = file_path.name
        return any(pattern.search(filename) for pattern in self.compiled_patterns[6:12])  # Temp file patterns
    
    def _is_build_artifact_dir(self, dir_path: Path) -> bool:
        """Check if a directory is a build artifact directory."""
        dirname = dir_path.name
        build_dir_patterns = [
            r'__pycache__',
            r'\.pytest_cache',
            r'\.mypy_cache',
            r'\.ruff_cache',
            r'\.tox',
            r'\.venv',
            r'venv',
            r'env',
            r'node_modules',
            r'build',
            r'dist',
            r'\.egg-info'
        ]
        return any(re.search(pattern, dirname) for pattern in build_dir_patterns)
    
    def _is_build_artifact_file(self, file_path: Path) -> bool:
        """Check if a file is a build artifact file."""
        filename = file_path.name
        return any(pattern.search(filename) for pattern in self.compiled_patterns[12:])  # Build artifact patterns
    
    def _remove_directory(self, dir_path: Path) -> bool:
        """Remove a directory and all its contents."""
        try:
            if not self.dry_run:
                if dir_path.exists():
                    import shutil
                    shutil.rmtree(dir_path)
                    return True
                else:
                    self.logger.warning(f"Directory not found for removal: {dir_path}")
                    return False
            else:
                self.logger.info(f"DRY RUN: Would remove directory {dir_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to remove directory {dir_path}: {e}")
            return False
    
    def clean_all_backup_files(self) -> FixResult:
        """Clean all backup files in the repository."""
        all_files = self.get_all_files()
        backup_files = []
        
        for file_path in all_files:
            if self._is_backup_file(file_path) or self._is_temp_file(file_path) or self._is_build_artifact_file(file_path):
                backup_files.append(str(file_path))
        
        if not backup_files:
            return FixResult(
                success=True,
                message="No backup files found to clean",
                files_changed=[]
            )
        
        # Create a synthetic violation for cleanup
        violation = {
            "type": "backup_files",
            "files": backup_files
        }
        
        return self._fix_backup_files(violation)
    
    def clean_all_build_artifacts(self) -> FixResult:
        """Clean all build artifacts in the repository."""
        all_paths = list(self.root_path.rglob("*"))
        
        artifact_dirs = []
        artifact_files = []
        
        for path in all_paths:
            if path.is_dir() and self._is_build_artifact_dir(path):
                artifact_dirs.append(str(path))
            elif path.is_file() and self._is_build_artifact_file(path):
                artifact_files.append(str(path))
        
        if not artifact_dirs and not artifact_files:
            return FixResult(
                success=True,
                message="No build artifacts found to clean",
                files_changed=[]
            )
        
        # Create a synthetic violation for cleanup
        violation = {
            "type": "build_artifacts",
            "directories": artifact_dirs,
            "files": artifact_files
        }
        
        return self._fix_build_artifacts(violation)