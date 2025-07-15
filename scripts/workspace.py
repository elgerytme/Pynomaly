#!/usr/bin/env python3
"""
Pynomaly Workspace Management Tool

Unified workspace management for the Pynomaly monorepo.
Provides commands for package management, dependency tracking, testing, and building.
"""

import json
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional
import tomllib


class WorkspaceManager:
    """Manages workspace operations for Pynomaly monorepo."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.src_path = root_path / "src"
        self.packages_path = self.src_path / "packages"
        self.config_path = root_path / "workspace.json"
        self._packages_cache = None
    
    @property
    def packages(self) -> Dict[str, Dict]:
        """Get all workspace packages with metadata."""
        if self._packages_cache is None:
            self._packages_cache = self._discover_packages()
        return self._packages_cache
    
    def _discover_packages(self) -> Dict[str, Dict]:
        """Discover all packages in the workspace."""
        packages = {}
        
        # Main pynomaly package
        packages["pynomaly"] = {
            "path": str(self.src_path / "pynomaly"),
            "type": "main",
            "dependencies": [],
            "dependents": []
        }
        
        # Discover packages in src/packages/
        if self.packages_path.exists():
            for package_dir in self.packages_path.iterdir():
                if package_dir.is_dir() and (package_dir / "__init__.py").exists():
                    packages[package_dir.name] = {
                        "path": str(package_dir),
                        "type": "package",
                        "dependencies": self._get_package_dependencies(package_dir),
                        "dependents": []
                    }
        
        # Build dependency graph
        self._build_dependency_graph(packages)
        return packages
    
    def _get_package_dependencies(self, package_path: Path) -> List[str]:
        """Extract dependencies from a package's pyproject.toml or __init__.py."""
        dependencies = []
        
        # Check pyproject.toml
        pyproject_path = package_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    pyproject = tomllib.load(f)
                    deps = pyproject.get("project", {}).get("dependencies", [])
                    for dep in deps:
                        if dep.startswith("pynomaly"):
                            dependencies.append("pynomaly")
            except Exception:
                pass
        
        return dependencies
    
    def _build_dependency_graph(self, packages: Dict[str, Dict]):
        """Build the dependency graph by updating dependents."""
        for pkg_name, pkg_info in packages.items():
            for dep in pkg_info["dependencies"]:
                if dep in packages:
                    packages[dep]["dependents"].append(pkg_name)
    
    def list_packages(self, package_type: Optional[str] = None) -> List[str]:
        """List all packages, optionally filtered by type."""
        if package_type:
            return [name for name, info in self.packages.items() 
                   if info["type"] == package_type]
        return list(self.packages.keys())
    
    def get_affected_packages(self, changed_package: str) -> Set[str]:
        """Get all packages that depend on the changed package."""
        affected = {changed_package}
        queue = [changed_package]
        
        while queue:
            current = queue.pop(0)
            if current in self.packages:
                for dependent in self.packages[current]["dependents"]:
                    if dependent not in affected:
                        affected.add(dependent)
                        queue.append(dependent)
        
        return affected
    
    def install_package(self, package_name: str) -> bool:
        """Install dependencies for a specific package."""
        if package_name not in self.packages:
            print(f"Error: Package '{package_name}' not found")
            return False
        
        package_path = Path(self.packages[package_name]["path"])
        
        # Try to install with pip if pyproject.toml exists
        pyproject_path = package_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-e", str(package_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Successfully installed {package_name}")
                    return True
                else:
                    print(f"❌ Failed to install {package_name}: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error installing {package_name}: {e}")
                return False
        
        print(f"⚠️  No pyproject.toml found for {package_name}")
        return True
    
    def test_package(self, package_name: str, args: List[str] = None) -> bool:
        """Run tests for a specific package."""
        if package_name not in self.packages:
            print(f"Error: Package '{package_name}' not found")
            return False
        
        package_path = Path(self.packages[package_name]["path"])
        test_args = args or []
        
        # Find test directory
        test_dirs = []
        if (package_path / "tests").exists():
            test_dirs.append(str(package_path / "tests"))
        if (self.root_path / "tests" / package_name).exists():
            test_dirs.append(str(self.root_path / "tests" / package_name))
        
        if not test_dirs:
            print(f"⚠️  No tests found for {package_name}")
            return True
        
        try:
            cmd = [sys.executable, "-m", "pytest"] + test_dirs + test_args
            result = subprocess.run(cmd, cwd=str(self.root_path))
            
            if result.returncode == 0:
                print(f"✅ Tests passed for {package_name}")
                return True
            else:
                print(f"❌ Tests failed for {package_name}")
                return False
        except Exception as e:
            print(f"❌ Error running tests for {package_name}: {e}")
            return False
    
    def build_package(self, package_name: str) -> bool:
        """Build a specific package."""
        if package_name not in self.packages:
            print(f"Error: Package '{package_name}' not found")
            return False
        
        package_path = Path(self.packages[package_name]["path"])
        
        # Check if pyproject.toml exists
        pyproject_path = package_path / "pyproject.toml"
        if not pyproject_path.exists():
            print(f"⚠️  No pyproject.toml found for {package_name}, skipping build")
            return True
        
        try:
            # Use hatch for building
            result = subprocess.run([
                "hatch", "build"
            ], cwd=str(package_path), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully built {package_name}")
                return True
            else:
                print(f"❌ Failed to build {package_name}: {result.stderr}")
                return False
        except FileNotFoundError:
            # Fallback to python -m build
            try:
                result = subprocess.run([
                    sys.executable, "-m", "build"
                ], cwd=str(package_path), capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Successfully built {package_name}")
                    return True
                else:
                    print(f"❌ Failed to build {package_name}: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error building {package_name}: {e}")
                return False
    
    def print_dependency_graph(self):
        """Print the workspace dependency graph."""
        print("📦 Workspace Dependency Graph:")
        print("=" * 50)
        
        for package_name, package_info in self.packages.items():
            deps = package_info["dependencies"]
            dependents = package_info["dependents"]
            
            print(f"\n📦 {package_name} ({package_info['type']})")
            print(f"   Path: {package_info['path']}")
            
            if deps:
                print(f"   Dependencies: {', '.join(deps)}")
            else:
                print("   Dependencies: None")
            
            if dependents:
                print(f"   Dependents: {', '.join(dependents)}")
            else:
                print("   Dependents: None")
    
    def save_config(self):
        """Save workspace configuration to workspace.json."""
        config = {
            "packages": self.packages,
            "workspace_root": str(self.root_path),
            "generated_at": str(subprocess.check_output(["date"], text=True).strip())
        }
        
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Workspace configuration saved to {self.config_path}")


def main():
    parser = argparse.ArgumentParser(description="Pynomaly Workspace Management Tool")
    parser.add_argument("--root", type=Path, default=Path.cwd(), 
                       help="Workspace root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List workspace packages")
    list_parser.add_argument("--type", choices=["main", "package"], 
                           help="Filter by package type")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install package dependencies")
    install_parser.add_argument("package", nargs="?", help="Package name (or 'all')")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run package tests")
    test_parser.add_argument("package", nargs="?", help="Package name (or 'all')")
    test_parser.add_argument("--affected", action="store_true", 
                           help="Test affected packages only")
    test_parser.add_argument("--args", nargs="*", help="Additional pytest arguments")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build packages")
    build_parser.add_argument("package", nargs="?", help="Package name (or 'all')")
    build_parser.add_argument("--affected", action="store_true", 
                            help="Build affected packages only")
    
    # Deps command
    deps_parser = subparsers.add_parser("deps", help="Show dependency information")
    deps_parser.add_argument("--graph", action="store_true", 
                           help="Show full dependency graph")
    deps_parser.add_argument("--affected", metavar="PACKAGE", 
                           help="Show packages affected by PACKAGE")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Workspace configuration")
    config_parser.add_argument("--save", action="store_true", 
                             help="Save current workspace config")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize workspace manager
    workspace = WorkspaceManager(args.root)
    
    if args.command == "list":
        packages = workspace.list_packages(args.type)
        print(f"📦 Workspace Packages ({len(packages)}):")
        for package in packages:
            package_info = workspace.packages[package]
            print(f"  • {package} ({package_info['type']}) - {package_info['path']}")
    
    elif args.command == "install":
        if args.package == "all":
            success = True
            for package in workspace.list_packages():
                success &= workspace.install_package(package)
            return 0 if success else 1
        elif args.package:
            return 0 if workspace.install_package(args.package) else 1
        else:
            print("Error: Please specify a package name or 'all'")
            return 1
    
    elif args.command == "test":
        if args.package == "all":
            success = True
            for package in workspace.list_packages():
                success &= workspace.test_package(package, args.args)
            return 0 if success else 1
        elif args.package:
            return 0 if workspace.test_package(args.package, args.args) else 1
        else:
            print("Error: Please specify a package name or 'all'")
            return 1
    
    elif args.command == "build":
        if args.package == "all":
            success = True
            for package in workspace.list_packages():
                success &= workspace.build_package(package)
            return 0 if success else 1
        elif args.package:
            return 0 if workspace.build_package(args.package) else 1
        else:
            print("Error: Please specify a package name or 'all'")
            return 1
    
    elif args.command == "deps":
        if args.graph:
            workspace.print_dependency_graph()
        elif args.affected:
            affected = workspace.get_affected_packages(args.affected)
            print(f"📦 Packages affected by changes to '{args.affected}':")
            for package in sorted(affected):
                print(f"  • {package}")
        else:
            print("Error: Please specify --graph or --affected PACKAGE")
            return 1
    
    elif args.command == "config":
        if args.save:
            workspace.save_config()
        else:
            print("Error: Please specify --save")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())