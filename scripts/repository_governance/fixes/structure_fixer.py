"""
Automated fixer for repository structure and organization issues.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Any, Set

from .auto_fixer import AutoFixer, FixResult


class StructureFixer(AutoFixer):
    """Fixer for repository structure and organization issues."""
    
    def __init__(self, root_path: Path, dry_run: bool = False):
        """Initialize the structure fixer."""
        super().__init__(root_path, dry_run)
        
        # Standard directory structure for packages
        self.standard_structure = {
            "domain": ["entities", "value_objects", "services", "repositories"],
            "application": ["dto", "services", "use_cases"],
            "infrastructure": ["repositories", "adapters", "config"],
            "interfaces": ["api", "cli", "python_sdk"]
        }
        
        # Files that should be present in each package
        self.required_files = {
            "root": ["__init__.py", "README.md", "pyproject.toml"],
            "domain": ["__init__.py"],
            "application": ["__init__.py"],
            "infrastructure": ["__init__.py"],
            "interfaces": ["__init__.py"]
        }
    
    @property
    def name(self) -> str:
        """Name of the fixer."""
        return "StructureFixer"
    
    @property
    def description(self) -> str:
        """Description of what this fixer does."""
        return "Fixes repository structure and organization issues"
    
    def can_fix(self, violation: Dict[str, Any]) -> bool:
        """Check if this fixer can handle the given violation."""
        return violation.get("type") in [
            "missing_init_files",
            "misplaced_files",
            "missing_directories",
            "inconsistent_structure",
            "empty_directories"
        ]
    
    def fix(self, violation: Dict[str, Any]) -> FixResult:
        """Apply the fix for the given violation."""
        violation_type = violation.get("type")
        
        if violation_type == "missing_init_files":
            return self._fix_missing_init_files(violation)
        elif violation_type == "misplaced_files":
            return self._fix_misplaced_files(violation)
        elif violation_type == "missing_directories":
            return self._fix_missing_directories(violation)
        elif violation_type == "inconsistent_structure":
            return self._fix_inconsistent_structure(violation)
        elif violation_type == "empty_directories":
            return self._fix_empty_directories(violation)
        else:
            return FixResult(
                success=False,
                message=f"Unknown violation type: {violation_type}"
            )
    
    def _fix_missing_init_files(self, violation: Dict[str, Any]) -> FixResult:
        """Fix missing __init__.py files."""
        missing_files = violation.get("missing_files", [])
        created_files = []
        failed_files = []
        
        for file_path_str in missing_files:
            file_path = Path(file_path_str)
            
            # Ensure the directory exists
            if not file_path.parent.exists():
                if not self.dry_run:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    self.logger.info(f"DRY RUN: Would create directory {file_path.parent}")
            
            # Create the __init__.py file
            init_content = self._generate_init_content(file_path)
            
            if not self.dry_run:
                try:
                    file_path.write_text(init_content, encoding='utf-8')
                    created_files.append(file_path_str)
                except Exception as e:
                    self.logger.error(f"Failed to create {file_path}: {e}")
                    failed_files.append(file_path_str)
            else:
                self.logger.info(f"DRY RUN: Would create {file_path}")
                created_files.append(file_path_str)
        
        success = len(failed_files) == 0
        message = f"Created {len(created_files)} __init__.py files"
        if failed_files:
            message += f", failed to create {len(failed_files)} files"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=created_files,
            details={
                "created_files": created_files,
                "failed_files": failed_files
            }
        )
    
    def _fix_misplaced_files(self, violation: Dict[str, Any]) -> FixResult:
        """Fix misplaced files by moving them to correct locations."""
        misplaced_files = violation.get("misplaced_files", [])
        moved_files = []
        failed_files = []
        
        for file_info in misplaced_files:
            current_path = Path(file_info.get("current_path", ""))
            suggested_path = Path(file_info.get("suggested_path", ""))
            
            if not current_path.exists():
                failed_files.append(str(current_path))
                continue
            
            # Ensure destination directory exists
            if not suggested_path.parent.exists():
                if not self.dry_run:
                    suggested_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    self.logger.info(f"DRY RUN: Would create directory {suggested_path.parent}")
            
            # Move the file
            if self.safe_move_file(current_path, suggested_path):
                moved_files.append({
                    "from": str(current_path),
                    "to": str(suggested_path)
                })
            else:
                failed_files.append(str(current_path))
        
        success = len(failed_files) == 0
        message = f"Moved {len(moved_files)} files to correct locations"
        if failed_files:
            message += f", failed to move {len(failed_files)} files"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=[move["to"] for move in moved_files],
            details={
                "moved_files": moved_files,
                "failed_files": failed_files
            }
        )
    
    def _fix_missing_directories(self, violation: Dict[str, Any]) -> FixResult:
        """Fix missing directories by creating them."""
        missing_dirs = violation.get("missing_directories", [])
        created_dirs = []
        failed_dirs = []
        
        for dir_path_str in missing_dirs:
            dir_path = Path(dir_path_str)
            
            if not self.dry_run:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path_str)
                    
                    # Create __init__.py if it's a Python package directory
                    if self._should_have_init_file(dir_path):
                        init_file = dir_path / "__init__.py"
                        init_content = self._generate_init_content(init_file)
                        init_file.write_text(init_content, encoding='utf-8')
                        
                except Exception as e:
                    self.logger.error(f"Failed to create directory {dir_path}: {e}")
                    failed_dirs.append(dir_path_str)
            else:
                self.logger.info(f"DRY RUN: Would create directory {dir_path}")
                created_dirs.append(dir_path_str)
        
        success = len(failed_dirs) == 0
        message = f"Created {len(created_dirs)} directories"
        if failed_dirs:
            message += f", failed to create {len(failed_dirs)} directories"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=[],  # Directories don't count as file changes
            details={
                "created_directories": created_dirs,
                "failed_directories": failed_dirs
            }
        )
    
    def _fix_inconsistent_structure(self, violation: Dict[str, Any]) -> FixResult:
        """Fix inconsistent package structure."""
        package_path = Path(violation.get("package_path", ""))
        inconsistencies = violation.get("inconsistencies", [])
        
        if not package_path.exists():
            return FixResult(
                success=False,
                message=f"Package path not found: {package_path}"
            )
        
        fixes_applied = []
        failed_fixes = []
        
        for inconsistency in inconsistencies:
            fix_type = inconsistency.get("type", "")
            
            if fix_type == "missing_layer":
                layer = inconsistency.get("layer", "")
                layer_path = package_path / layer
                
                if not layer_path.exists():
                    if not self.dry_run:
                        try:
                            layer_path.mkdir(parents=True, exist_ok=True)
                            
                            # Create __init__.py
                            init_file = layer_path / "__init__.py"
                            init_content = self._generate_init_content(init_file)
                            init_file.write_text(init_content, encoding='utf-8')
                            
                            # Create standard subdirectories for this layer
                            if layer in self.standard_structure:
                                for subdir in self.standard_structure[layer]:
                                    subdir_path = layer_path / subdir
                                    subdir_path.mkdir(exist_ok=True)
                                    
                                    # Create __init__.py in subdirectory
                                    subdir_init = subdir_path / "__init__.py"
                                    subdir_init.write_text(f'"""{subdir.title()} module."""\n', encoding='utf-8')
                            
                            fixes_applied.append(f"Created {layer} layer")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to create layer {layer}: {e}")
                            failed_fixes.append(f"Failed to create {layer} layer")
                    else:
                        self.logger.info(f"DRY RUN: Would create {layer} layer")
                        fixes_applied.append(f"Would create {layer} layer")
        
        success = len(failed_fixes) == 0
        message = f"Applied {len(fixes_applied)} structure fixes"
        if failed_fixes:
            message += f", {len(failed_fixes)} fixes failed"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=[],  # Structure changes don't count as file changes
            details={
                "fixes_applied": fixes_applied,
                "failed_fixes": failed_fixes,
                "package_path": str(package_path)
            }
        )
    
    def _fix_empty_directories(self, violation: Dict[str, Any]) -> FixResult:
        """Fix empty directories by either removing them or adding placeholder files."""
        empty_dirs = violation.get("empty_directories", [])
        action = violation.get("action", "remove")  # "remove" or "populate"
        
        processed_dirs = []
        failed_dirs = []
        
        for dir_path_str in empty_dirs:
            dir_path = Path(dir_path_str)
            
            if not dir_path.exists():
                continue
            
            if action == "remove":
                if self._remove_directory(dir_path):
                    processed_dirs.append(dir_path_str)
                else:
                    failed_dirs.append(dir_path_str)
            elif action == "populate":
                # Add a placeholder file or __init__.py
                if self._should_have_init_file(dir_path):
                    placeholder_file = dir_path / "__init__.py"
                    placeholder_content = self._generate_init_content(placeholder_file)
                else:
                    placeholder_file = dir_path / ".gitkeep"
                    placeholder_content = "# This file ensures the directory is tracked by Git\\n"
                
                if not self.dry_run:
                    try:
                        placeholder_file.write_text(placeholder_content, encoding='utf-8')
                        processed_dirs.append(dir_path_str)
                    except Exception as e:
                        self.logger.error(f"Failed to populate {dir_path}: {e}")
                        failed_dirs.append(dir_path_str)
                else:
                    self.logger.info(f"DRY RUN: Would populate {dir_path}")
                    processed_dirs.append(dir_path_str)
        
        success = len(failed_dirs) == 0
        action_verb = "Removed" if action == "remove" else "Populated"
        message = f"{action_verb} {len(processed_dirs)} empty directories"
        if failed_dirs:
            message += f", failed to {action} {len(failed_dirs)} directories"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=[],  # Directory operations don't count as file changes
            details={
                "processed_directories": processed_dirs,
                "failed_directories": failed_dirs,
                "action": action
            }
        )
    
    def _generate_init_content(self, init_file_path: Path) -> str:
        """Generate appropriate content for an __init__.py file."""
        parent_dir = init_file_path.parent
        parent_name = parent_dir.name
        
        # Generate content based on the directory type
        if parent_name in ["entities", "value_objects"]:
            return f'"""{parent_name.replace("_", " ").title()} module."""\n'
        elif parent_name in ["services", "repositories"]:
            return f'"""{parent_name.replace("_", " ").title()} module."""\n'
        elif parent_name in ["dto", "use_cases"]:
            return f'"""{parent_name.replace("_", " ").title()} module."""\n'
        elif parent_name in ["api", "cli", "python_sdk"]:
            return f'"""{parent_name.replace("_", " ").title()} interface module."""\n'
        else:
            return f'"""{parent_name.replace("_", " ").title()} module."""\n'
    
    def _should_have_init_file(self, dir_path: Path) -> bool:
        """Check if a directory should have an __init__.py file."""
        # Python package directories should have __init__.py
        parent_parts = dir_path.parts
        
        # Check if it's in a Python package structure
        python_indicators = ["src", "packages", "domain", "application", "infrastructure", "interfaces"]
        return any(indicator in parent_parts for indicator in python_indicators)
    
    def _remove_directory(self, dir_path: Path) -> bool:
        """Remove a directory and all its contents."""
        try:
            if not self.dry_run:
                if dir_path.exists():
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
    
    def create_standard_package_structure(self, package_path: Path) -> FixResult:
        """Create a standard package structure."""
        if not package_path.exists():
            return FixResult(
                success=False,
                message=f"Package path not found: {package_path}"
            )
        
        created_items = []
        failed_items = []
        
        # Create standard layers
        for layer, subdirs in self.standard_structure.items():
            layer_path = package_path / layer
            
            if not layer_path.exists():
                if not self.dry_run:
                    try:
                        layer_path.mkdir(parents=True, exist_ok=True)
                        
                        # Create __init__.py for the layer
                        init_file = layer_path / "__init__.py"
                        init_content = self._generate_init_content(init_file)
                        init_file.write_text(init_content, encoding='utf-8')
                        
                        created_items.append(f"Layer: {layer}")
                        
                        # Create subdirectories
                        for subdir in subdirs:
                            subdir_path = layer_path / subdir
                            subdir_path.mkdir(exist_ok=True)
                            
                            # Create __init__.py in subdirectory
                            subdir_init = subdir_path / "__init__.py"
                            subdir_init.write_text(f'"""{subdir.title()} module."""\n', encoding='utf-8')
                            
                            created_items.append(f"Subdirectory: {layer}/{subdir}")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to create layer {layer}: {e}")
                        failed_items.append(f"Layer: {layer}")
                else:
                    self.logger.info(f"DRY RUN: Would create layer {layer}")
                    created_items.append(f"Would create layer: {layer}")
        
        success = len(failed_items) == 0
        message = f"Created standard package structure with {len(created_items)} items"
        if failed_items:
            message += f", failed to create {len(failed_items)} items"
        
        return FixResult(
            success=success,
            message=message,
            files_changed=[],  # Structure creation doesn't count as file changes
            details={
                "created_items": created_items,
                "failed_items": failed_items,
                "package_path": str(package_path)
            }
        )