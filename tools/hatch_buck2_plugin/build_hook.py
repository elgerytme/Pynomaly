"""Hatch build hook for Buck2 integration.

This plugin integrates Buck2 builds into the Hatch build process,
allowing Buck2 to handle compilation and optimization while Hatch
manages packaging and distribution.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.plugin import hookimpl


class Buck2BuildHook(BuildHookInterface):
    """Custom build hook for Buck2 integration."""

    PLUGIN_NAME = "buck2"

    def __init__(self, root: str, config: Dict[str, Any]) -> None:
        super().__init__(root, config)
        self.root_path = Path(root)
        self.config = config
        
        # Buck2 configuration
        self.buck2_executable = config.get("executable", "buck2")
        self.build_targets = config.get("targets", [])
        self.web_assets_enabled = config.get("web_assets", True)
        self.artifacts_dir = self.root_path / config.get("artifacts_dir", "buck-out")
        
    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """Initialize the build hook."""
        self.app.display_info("Initializing Buck2 build hook")
        
        # Verify Buck2 is available
        if not self._check_buck2_available():
            self.app.display_warning(
                "Buck2 not found. Falling back to standard build process."
            )
            return
            
        # Clean previous artifacts
        self._clean_artifacts()
        
    def clean(self, versions: List[str]) -> None:
        """Clean Buck2 build artifacts."""
        self.app.display_info("Cleaning Buck2 artifacts")
        self._clean_artifacts()
        
    def finalize(self, version: str, build_data: Dict[str, Any], artifact_path: str) -> None:
        """Run Buck2 builds and copy artifacts."""
        # TEMPORARY: Skip Buck2 builds for editable installs to prevent issues
        # This is a temporary workaround to fix editable install problems
        if True:  # Disabled Buck2 integration temporarily
            self.app.display_info("Buck2 integration temporarily disabled for editable installs")
            return
            
        if not self._check_buck2_available():
            return
            
        self.app.display_info("Running Buck2 builds")
        
        try:
            # Run Buck2 builds
            self._run_buck2_builds()
            
            # Copy artifacts to build directory
            self._copy_artifacts_to_build(artifact_path)
            
            self.app.display_success("Buck2 build completed successfully")
            
        except subprocess.CalledProcessError as e:
            self.app.display_error(f"Buck2 build failed: {e}")
            raise
        except Exception as e:
            self.app.display_error(f"Buck2 integration error: {e}")
            raise
    
    def _check_buck2_available(self) -> bool:
        """Check if Buck2 is available in the system."""
        try:
            result = subprocess.run(
                [self.buck2_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _clean_artifacts(self) -> None:
        """Clean Buck2 build artifacts."""
        if self.artifacts_dir.exists():
            shutil.rmtree(self.artifacts_dir, ignore_errors=True)
        
        # Clean Buck2 cache
        try:
            subprocess.run(
                [self.buck2_executable, "clean"],
                cwd=self.root_path,
                check=False,
                capture_output=True
            )
        except Exception:
            pass  # Ignore cleanup errors
    
    def _run_buck2_builds(self) -> None:
        """Execute Buck2 build targets."""
        # Default targets if none specified
        default_targets = [
            "//:pynomaly-lib",
            "//:pynomaly-cli", 
            "//:pynomaly-api",
            "//:pynomaly-web"
        ]
        
        if self.web_assets_enabled:
            default_targets.extend([
                "//:web-assets",
                "//:tailwind-build"
            ])
        
        targets_to_build = self.build_targets if self.build_targets else default_targets
        
        self.app.display_info(f"Building Buck2 targets: {targets_to_build}")
        
        # Build each target
        for target in targets_to_build:
            self.app.display_info(f"Building target: {target}")
            
            try:
                result = subprocess.run(
                    [self.buck2_executable, "build", target],
                    cwd=self.root_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                self.app.display_debug(f"Buck2 output for {target}:\n{result.stdout}")
                
            except subprocess.CalledProcessError as e:
                self.app.display_error(f"Failed to build {target}:")
                self.app.display_error(f"stdout: {e.stdout}")
                self.app.display_error(f"stderr: {e.stderr}")
                raise
    
    def _copy_artifacts_to_build(self, artifact_path: str) -> None:
        """Copy Buck2 build artifacts to Hatch build directory."""
        build_dir = Path(artifact_path).parent
        
        # Create buck2-artifacts directory in build
        buck2_artifacts_dir = build_dir / "buck2-artifacts"
        buck2_artifacts_dir.mkdir(exist_ok=True)
        
        # Copy Buck2 outputs
        buck_out_dir = self.root_path / "buck-out"
        if buck_out_dir.exists():
            self.app.display_info("Copying Buck2 artifacts to build directory")
            
            # Copy binaries
            self._copy_buck2_binaries(buck_out_dir, buck2_artifacts_dir)
            
            # Copy web assets if enabled
            if self.web_assets_enabled:
                self._copy_web_assets(buck_out_dir, buck2_artifacts_dir)
            
            # Copy any additional artifacts
            self._copy_additional_artifacts(buck_out_dir, buck2_artifacts_dir)
    
    def _copy_buck2_binaries(self, buck_out_dir: Path, dest_dir: Path) -> None:
        """Copy Buck2 binary artifacts."""
        binaries_dir = dest_dir / "bin"
        binaries_dir.mkdir(exist_ok=True)
        
        # Look for common binary output patterns
        binary_patterns = [
            "**/pynomaly-cli*",
            "**/pynomaly-api*", 
            "**/pynomaly-web*",
            "**/*.pex"  # Python executables
        ]
        
        for pattern in binary_patterns:
            for artifact in buck_out_dir.rglob(pattern):
                if artifact.is_file():
                    dest_file = binaries_dir / artifact.name
                    shutil.copy2(artifact, dest_file)
                    self.app.display_debug(f"Copied binary: {artifact.name}")
    
    def _copy_web_assets(self, buck_out_dir: Path, dest_dir: Path) -> None:
        """Copy web assets from Buck2 build."""
        web_assets_dir = dest_dir / "web"
        web_assets_dir.mkdir(exist_ok=True)
        
        # Look for web asset patterns
        web_patterns = [
            "**/tailwind*.css",
            "**/web-assets*",
            "**/*.js",
            "**/*.css",
            "**/*.html"
        ]
        
        for pattern in web_patterns:
            for asset in buck_out_dir.rglob(pattern):
                if asset.is_file():
                    # Preserve directory structure for web assets
                    rel_path = asset.relative_to(buck_out_dir)
                    dest_file = web_assets_dir / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(asset, dest_file)
                    self.app.display_debug(f"Copied web asset: {rel_path}")
    
    def _copy_additional_artifacts(self, buck_out_dir: Path, dest_dir: Path) -> None:
        """Copy any additional Buck2 artifacts."""
        # Copy documentation if built
        docs_pattern = "**/docs-site*"
        for docs_artifact in buck_out_dir.rglob(docs_pattern):
            if docs_artifact.is_file():
                docs_dir = dest_dir / "docs"
                docs_dir.mkdir(exist_ok=True)
                shutil.copy2(docs_artifact, docs_dir / docs_artifact.name)
                self.app.display_debug(f"Copied docs: {docs_artifact.name}")
        
        # Copy test results if available
        test_pattern = "**/test-results*"
        for test_artifact in buck_out_dir.rglob(test_pattern):
            if test_artifact.is_file():
                test_dir = dest_dir / "test-results"
                test_dir.mkdir(exist_ok=True)
                shutil.copy2(test_artifact, test_dir / test_artifact.name)
                self.app.display_debug(f"Copied test results: {test_artifact.name}")


@hookimpl
def hatch_register_build_hook():
    """Register the Buck2 build hook with Hatch."""
    return Buck2BuildHook