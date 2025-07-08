"""Build hooks for Pynomaly project.

This module provides custom build hooks for the Hatch build system,
including automatic dashboard export during the build process.
"""

import os
import subprocess
import sys
from pathlib import Path

from hatchling.plugin import hookimpl


class DashboardExportBuildHook:
    """Build hook to export Grafana dashboards during the build process."""
    
    def __init__(self, root: str, config: dict):
        self.root = Path(root)
        self.config = config
    
    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize the build hook and export dashboards."""
        print("📊 Exporting Grafana dashboards...")
        
        # Path to the dashboard export script
        export_script = self.root / "scripts" / "export_dashboards.py"
        
        if not export_script.exists():
            print("⚠️  Warning: Dashboard export script not found, skipping...")
            return
        
        try:
            # Run the dashboard export script
            result = subprocess.run(
                [sys.executable, str(export_script)],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            if result.stderr:
                print("Dashboard export warnings:", result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Dashboard export failed: {e}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            # Don't fail the build for dashboard export failures
        except Exception as e:
            print(f"❌ Unexpected error during dashboard export: {e}")


@hookimpl
def hatch_build_hook(root: str, config: dict):
    """Hatch build hook entry point."""
    return DashboardExportBuildHook(root, config)
