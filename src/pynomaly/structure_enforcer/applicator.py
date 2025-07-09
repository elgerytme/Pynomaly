"""
Fix applicator implementation.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .models import Fix, FixResult, FixType


class FixApplicator:
    """Applies fixes to the repository."""
    
    def __init__(self, fixes: List[Fix], dry_run: bool = True):
        self.fixes = fixes
        self.dry_run = dry_run
        self.applied_fixes: List[Fix] = []
        self.failed_fixes: List[Tuple[Fix, str]] = []
    
    def apply(self) -> FixResult:
        """Apply all fixes."""
        self.applied_fixes = []
        self.failed_fixes = []
        
        for fix in self.fixes:
            try:
                success = self._apply_fix(fix)
                if success:
                    self.applied_fixes.append(fix)
                else:
                    self.failed_fixes.append((fix, "Unknown error"))
            except Exception as e:
                self.failed_fixes.append((fix, str(e)))
        
        return FixResult(
            applied_fixes=self.applied_fixes,
            failed_fixes=self.failed_fixes,
            dry_run=self.dry_run,
            timestamp=datetime.now(),
        )
    
    def _apply_fix(self, fix: Fix) -> bool:
        """Apply a single fix."""
        if fix.type == FixType.MOVE_FILE:
            return self._apply_move_file_fix(fix)
        
        elif fix.type == FixType.MOVE_DIRECTORY:
            return self._apply_move_directory_fix(fix)
        
        elif fix.type == FixType.DELETE_FILE:
            return self._apply_delete_file_fix(fix)
        
        elif fix.type == FixType.DELETE_DIRECTORY:
            return self._apply_delete_directory_fix(fix)
        
        elif fix.type == FixType.CREATE_FILE:
            return self._apply_create_file_fix(fix)
        
        elif fix.type == FixType.CREATE_DIRECTORY:
            return self._apply_create_directory_fix(fix)
        
        elif fix.type == FixType.MODIFY_FILE:
            return self._apply_modify_file_fix(fix)
        
        return False
    
    def _apply_move_file_fix(self, fix: Fix) -> bool:
        """Apply a move file fix."""
        if not fix.source_path or not fix.target_path:
            return False
        
        if not fix.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {fix.source_path}")
        
        if fix.target_path.exists():
            raise FileExistsError(f"Target file already exists: {fix.target_path}")
        
        if self.dry_run:
            print(f"[DRY RUN] Would move file: {fix.source_path} -> {fix.target_path}")
            return True
        
        # Create backup if required
        if fix.backup_required:
            backup_path = fix.source_path.with_suffix(fix.source_path.suffix + '.backup')
            shutil.copy2(fix.source_path, backup_path)
        
        # Create target directory if it doesn't exist
        fix.target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        shutil.move(str(fix.source_path), str(fix.target_path))
        
        print(f"Moved file: {fix.source_path} -> {fix.target_path}")
        return True
    
    def _apply_move_directory_fix(self, fix: Fix) -> bool:
        """Apply a move directory fix."""
        if not fix.source_path or not fix.target_path:
            return False
        
        if not fix.source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {fix.source_path}")
        
        if fix.target_path.exists():
            raise FileExistsError(f"Target directory already exists: {fix.target_path}")
        
        if self.dry_run:
            print(f"[DRY RUN] Would move directory: {fix.source_path} -> {fix.target_path}")
            return True
        
        # Create backup if required
        if fix.backup_required:
            backup_path = fix.source_path.with_suffix('.backup')
            shutil.copytree(fix.source_path, backup_path)
        
        # Create target parent directory if it doesn't exist
        fix.target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the directory
        shutil.move(str(fix.source_path), str(fix.target_path))
        
        print(f"Moved directory: {fix.source_path} -> {fix.target_path}")
        return True
    
    def _apply_delete_file_fix(self, fix: Fix) -> bool:
        """Apply a delete file fix."""
        if not fix.source_path:
            return False
        
        if not fix.source_path.exists():
            # File already deleted, consider it success
            return True
        
        if self.dry_run:
            print(f"[DRY RUN] Would delete file: {fix.source_path}")
            return True
        
        # Create backup if required
        if fix.backup_required:
            backup_path = fix.source_path.with_suffix(fix.source_path.suffix + '.backup')
            shutil.copy2(fix.source_path, backup_path)
        
        # Delete the file
        fix.source_path.unlink()
        
        print(f"Deleted file: {fix.source_path}")
        return True
    
    def _apply_delete_directory_fix(self, fix: Fix) -> bool:
        """Apply a delete directory fix."""
        if not fix.source_path:
            return False
        
        if not fix.source_path.exists():
            # Directory already deleted, consider it success
            return True
        
        if self.dry_run:
            print(f"[DRY RUN] Would delete directory: {fix.source_path}")
            return True
        
        # Create backup if required
        if fix.backup_required:
            backup_path = fix.source_path.with_suffix('.backup')
            shutil.copytree(fix.source_path, backup_path)
        
        # Delete the directory
        shutil.rmtree(fix.source_path)
        
        print(f"Deleted directory: {fix.source_path}")
        return True
    
    def _apply_create_file_fix(self, fix: Fix) -> bool:
        """Apply a create file fix."""
        if not fix.target_path:
            return False
        
        if fix.target_path.exists():
            raise FileExistsError(f"Target file already exists: {fix.target_path}")
        
        if self.dry_run:
            print(f"[DRY RUN] Would create file: {fix.target_path}")
            return True
        
        # Create target directory if it doesn't exist
        fix.target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file
        content = fix.content or ""
        with open(fix.target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created file: {fix.target_path}")
        return True
    
    def _apply_create_directory_fix(self, fix: Fix) -> bool:
        """Apply a create directory fix."""
        if not fix.target_path:
            return False
        
        if fix.target_path.exists():
            # Directory already exists, consider it success
            return True
        
        if self.dry_run:
            print(f"[DRY RUN] Would create directory: {fix.target_path}")
            return True
        
        # Create the directory
        fix.target_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directory: {fix.target_path}")
        return True
    
    def _apply_modify_file_fix(self, fix: Fix) -> bool:
        """Apply a modify file fix."""
        if not fix.source_path:
            return False
        
        if not fix.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {fix.source_path}")
        
        if self.dry_run:
            print(f"[DRY RUN] Would modify file: {fix.source_path}")
            print(f"[DRY RUN] Description: {fix.description}")
            return True
        
        # For now, just print the description as this requires manual intervention
        print(f"Manual modification required for: {fix.source_path}")
        print(f"Description: {fix.description}")
        
        # Create backup if required
        if fix.backup_required:
            backup_path = fix.source_path.with_suffix(fix.source_path.suffix + '.backup')
            shutil.copy2(fix.source_path, backup_path)
        
        # Note: Actual modification would require specific logic based on the violation type
        # For now, we just flag it as needing manual intervention
        return True


def apply_fixes(fixes: List[Fix], dry_run: bool = True) -> FixResult:
    """
    Apply the suggested fixes to the repository.
    
    Args:
        fixes: List of fixes to apply.
        dry_run: If True, only simulate the changes without actually applying them.
    
    Returns:
        FixResult: Results of applying the fixes.
    """
    applicator = FixApplicator(fixes, dry_run)
    return applicator.apply()
