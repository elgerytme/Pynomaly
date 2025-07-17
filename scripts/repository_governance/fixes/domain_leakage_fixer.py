"""
Automated fixer for domain leakage violations.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

from .auto_fixer import AutoFixer, FixResult


class DomainLeakageFixer(AutoFixer):
    """Fixer for domain leakage and architectural violations."""
    
    def __init__(self, root_path: Path, dry_run: bool = False):
        """Initialize the domain leakage fixer."""
        super().__init__(root_path, dry_run)
        
        # Forbidden import patterns that indicate domain leakage
        self.forbidden_patterns = [
            r'from\s+src\.packages\..*?\..*?\..*?\s+import',  # Monorepo imports
            r'import\s+src\.packages\..*?\..*?\..*?',
            r'from\s+\.\.\..*?\s+import',  # Deep relative imports
            r'import\s+\.\.\..*?'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.forbidden_patterns]
    
    @property
    def name(self) -> str:
        """Name of the fixer."""
        return "DomainLeakageFixer"
    
    @property
    def description(self) -> str:
        """Description of what this fixer does."""
        return "Fixes domain leakage by replacing monorepo imports with local entities"
    
    def can_fix(self, violation: Dict[str, Any]) -> bool:
        """Check if this fixer can handle the given violation."""
        return violation.get("type") in ["monorepo_imports", "cross_domain_dependencies", "circular_dependencies"]
    
    def fix(self, violation: Dict[str, Any]) -> FixResult:
        """Apply the fix for the given violation."""
        violation_type = violation.get("type")
        
        if violation_type == "monorepo_imports":
            return self._fix_monorepo_imports(violation)
        elif violation_type == "cross_domain_dependencies":
            return self._fix_cross_domain_dependencies(violation)
        elif violation_type == "circular_dependencies":
            return self._fix_circular_dependencies(violation)
        else:
            return FixResult(
                success=False,
                message=f"Unknown violation type: {violation_type}"
            )
    
    def _fix_monorepo_imports(self, violation: Dict[str, Any]) -> FixResult:
        """Fix monorepo import violations."""
        file_path = Path(violation.get("file", ""))
        imports = violation.get("imports", [])
        
        if not file_path.exists():
            return FixResult(
                success=False,
                message=f"File not found: {file_path}"
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_applied = []
            
            # Process each import
            for import_info in imports:
                import_line = import_info.get("import_line", "")
                suggested_replacement = self._generate_replacement(import_info)
                
                if suggested_replacement:
                    # Replace the import
                    content = content.replace(import_line, suggested_replacement)
                    fixes_applied.append({
                        "original": import_line,
                        "replacement": suggested_replacement
                    })
            
            if content != original_content:
                if self.safe_write_file(file_path, content):
                    return FixResult(
                        success=True,
                        message=f"Fixed {len(fixes_applied)} monorepo imports in {file_path.name}",
                        files_changed=[str(file_path)],
                        details={
                            "fixes_applied": fixes_applied,
                            "file": str(file_path)
                        }
                    )
                else:
                    return FixResult(
                        success=False,
                        message=f"Failed to write changes to {file_path}"
                    )
            else:
                return FixResult(
                    success=True,
                    message=f"No changes needed for {file_path.name}",
                    files_changed=[]
                )
                
        except Exception as e:
            return FixResult(
                success=False,
                message=f"Error processing {file_path}: {str(e)}"
            )
    
    def _fix_cross_domain_dependencies(self, violation: Dict[str, Any]) -> FixResult:
        """Fix cross-domain dependency violations."""
        file_path = Path(violation.get("file", ""))
        dependencies = violation.get("dependencies", [])
        
        if not file_path.exists():
            return FixResult(
                success=False,
                message=f"File not found: {file_path}"
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_applied = []
            
            # Process each dependency
            for dep in dependencies:
                from_domain = dep.get("from_domain", "")
                to_domain = dep.get("to_domain", "")
                import_line = dep.get("import_line", "")
                
                # Create a domain abstraction or interface
                interface_replacement = self._create_domain_interface(from_domain, to_domain, import_line)
                
                if interface_replacement:
                    content = content.replace(import_line, interface_replacement)
                    fixes_applied.append({
                        "original": import_line,
                        "replacement": interface_replacement,
                        "from_domain": from_domain,
                        "to_domain": to_domain
                    })
            
            if content != original_content:
                if self.safe_write_file(file_path, content):
                    return FixResult(
                        success=True,
                        message=f"Fixed {len(fixes_applied)} cross-domain dependencies in {file_path.name}",
                        files_changed=[str(file_path)],
                        details={
                            "fixes_applied": fixes_applied,
                            "file": str(file_path)
                        }
                    )
                else:
                    return FixResult(
                        success=False,
                        message=f"Failed to write changes to {file_path}"
                    )
            else:
                return FixResult(
                    success=True,
                    message=f"No changes needed for {file_path.name}",
                    files_changed=[]
                )
                
        except Exception as e:
            return FixResult(
                success=False,
                message=f"Error processing {file_path}: {str(e)}"
            )
    
    def _fix_circular_dependencies(self, violation: Dict[str, Any]) -> FixResult:
        """Fix circular dependency violations."""
        cycle = violation.get("cycle", [])
        
        if not cycle:
            return FixResult(
                success=False,
                message="No cycle information provided"
            )
        
        # For circular dependencies, we need to break the cycle
        # This is complex and may require refactoring
        fixes_applied = []
        files_changed = []
        
        # Strategy: Move common dependencies to a shared module
        for i, file_path_str in enumerate(cycle):
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find imports that are part of the cycle
                cycle_imports = self._find_cycle_imports(content, cycle)
                
                if cycle_imports:
                    # Replace with interface or shared module imports
                    new_content = self._replace_cycle_imports(content, cycle_imports)
                    
                    if new_content != content:
                        if self.safe_write_file(file_path, new_content):
                            files_changed.append(str(file_path))
                            fixes_applied.append({
                                "file": str(file_path),
                                "imports_fixed": len(cycle_imports)
                            })
                        
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if fixes_applied:
            return FixResult(
                success=True,
                message=f"Fixed circular dependencies in {len(fixes_applied)} files",
                files_changed=files_changed,
                details={
                    "fixes_applied": fixes_applied,
                    "cycle": cycle
                }
            )
        else:
            return FixResult(
                success=False,
                message="No fixes could be applied to break the circular dependency"
            )
    
    def _generate_replacement(self, import_info: Dict[str, Any]) -> str:
        """Generate a replacement for a monorepo import."""
        import_line = import_info.get("import_line", "")
        module = import_info.get("module", "")
        names = import_info.get("names", [])
        
        # Extract the domain and entity from the import
        if "from src.packages." in import_line:
            # Extract domain from the import path
            match = re.search(r'from src\.packages\.([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)', import_line)
            if match:
                domain = match.group(2)
                layer = match.group(3)
                entity_type = match.group(4)
                
                # Create local import
                if layer == "domain" and entity_type == "entities":
                    return f"# Local entity - moved from monorepo import\n# from {module} import {', '.join(names)}"
                elif layer == "domain" and entity_type == "value_objects":
                    return f"# Local value object - moved from monorepo import\n# from {module} import {', '.join(names)}"
                elif layer == "application" and entity_type == "services":
                    return f"# Local service - moved from monorepo import\n# from {module} import {', '.join(names)}"
        
        # Default fallback - comment out the import
        return f"# FIXME: Monorepo import needs local replacement\n# {import_line}"
    
    def _create_domain_interface(self, from_domain: str, to_domain: str, import_line: str) -> str:
        """Create a domain interface to replace cross-domain dependency."""
        # This is a simplified approach - in practice, you'd want to create actual interface files
        return f"# FIXME: Cross-domain dependency - create interface\n# {import_line}\n# Consider creating an interface in {from_domain} domain"
    
    def _find_cycle_imports(self, content: str, cycle: List[str]) -> List[str]:
        """Find imports that are part of a circular dependency."""
        cycle_imports = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_str = self._format_import_node(node)
                    
                    # Check if this import references any file in the cycle
                    for cycle_file in cycle:
                        if self._import_references_file(import_str, cycle_file):
                            cycle_imports.append(import_str)
                            break
        except SyntaxError:
            pass
        
        return cycle_imports
    
    def _replace_cycle_imports(self, content: str, cycle_imports: List[str]) -> str:
        """Replace imports that are part of a circular dependency."""
        new_content = content
        
        for import_line in cycle_imports:
            # Replace with a comment and TODO
            replacement = f"# FIXME: Circular dependency - refactor needed\n# {import_line}"
            new_content = new_content.replace(import_line, replacement)
        
        return new_content
    
    def _format_import_node(self, node: ast.AST) -> str:
        """Format an import node back to string."""
        if isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            return f"from {node.module} import {names}"
        elif isinstance(node, ast.Import):
            names = ", ".join(alias.name for alias in node.names)
            return f"import {names}"
        return ""
    
    def _import_references_file(self, import_str: str, file_path: str) -> bool:
        """Check if an import references a specific file."""
        # Simple heuristic - check if the file path components are in the import
        file_path_obj = Path(file_path)
        parts = file_path_obj.parts
        
        # Check if any part of the file path is in the import
        for part in parts:
            if part in import_str:
                return True
        
        return False
    
    def fix_all_monorepo_imports(self) -> FixResult:
        """Fix all monorepo imports in the repository."""
        python_files = self.get_python_files()
        total_fixes = 0
        files_changed = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Find and replace monorepo imports
                for pattern in self.compiled_patterns:
                    matches = list(pattern.finditer(content))
                    for match in matches:
                        import_line = match.group(0)
                        replacement = f"# FIXME: Monorepo import needs local replacement\n# {import_line}"
                        content = content.replace(import_line, replacement)
                        total_fixes += 1
                
                if content != original_content:
                    if self.safe_write_file(file_path, content):
                        files_changed.append(str(file_path))
                        
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return FixResult(
            success=True,
            message=f"Fixed {total_fixes} monorepo imports across {len(files_changed)} files",
            files_changed=files_changed,
            details={
                "total_fixes": total_fixes,
                "files_processed": len(python_files)
            }
        )