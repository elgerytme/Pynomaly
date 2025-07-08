#!/usr/bin/env python3
"""Buck2 Simulator for testing Buck2 + Hatch integration without requiring Buck2 binary.

This simulator provides a mock Buck2 environment that mimics Buck2's behavior
for testing the integration framework, build hooks, and workflow validation.
"""

import sys
import time
from pathlib import Path
from typing import Any


class Buck2Simulator:
    """Simulator for Buck2 build system."""

    def __init__(self):
        self.root_path = Path.cwd()
        self.buck_out_dir = self.root_path / "buck-out"
        self.build_cache = {}

    def simulate_command(self, args: list[str]) -> dict[str, Any]:
        """Simulate Buck2 command execution."""
        if not args:
            return {"error": "No command provided"}

        cmd = args[0]

        if cmd == "--version":
            return {"success": True, "output": "buck2 v0.1.0-simulator", "stderr": ""}
        elif cmd == "build":
            return self._simulate_build(args[1:])
        elif cmd == "test":
            return self._simulate_test(args[1:])
        elif cmd == "clean":
            return self._simulate_clean()
        elif cmd == "query":
            return self._simulate_query(args[1:])
        else:
            return {"error": f"Unknown command: {cmd}"}

    def _simulate_build(self, targets: list[str]) -> dict[str, Any]:
        """Simulate build command."""
        if not targets:
            return {"error": "No targets specified"}

        print(f"🔨 Buck2 Simulator: Building targets {targets}")

        # Create buck-out directory structure
        self.buck_out_dir.mkdir(exist_ok=True)

        build_results = {}

        for target in targets:
            print(f"  📦 Building {target}...")
            time.sleep(0.1)  # Simulate build time

            # Create mock artifacts
            artifact_path = self._create_mock_artifact(target)
            build_results[target] = {
                "status": "success",
                "artifact": str(artifact_path),
                "build_time": 0.1,
            }

        return {
            "success": True,
            "targets": build_results,
            "total_time": len(targets) * 0.1,
        }

    def _simulate_test(self, targets: list[str]) -> dict[str, Any]:
        """Simulate test command."""
        print(f"🧪 Buck2 Simulator: Running tests for {targets}")

        test_results = {}
        for target in targets:
            print(f"  ✅ Testing {target}...")
            time.sleep(0.05)  # Simulate test time

            test_results[target] = {
                "status": "passed",
                "tests_run": 10,
                "tests_passed": 10,
                "tests_failed": 0,
            }

        return {
            "success": True,
            "test_results": test_results,
            "total_tests": len(targets) * 10,
        }

    def _simulate_clean(self) -> dict[str, Any]:
        """Simulate clean command."""
        print("🧹 Buck2 Simulator: Cleaning build artifacts...")

        # Remove buck-out directory
        import shutil

        if self.buck_out_dir.exists():
            shutil.rmtree(self.buck_out_dir)

        return {"success": True, "message": "Build artifacts cleaned"}

    def _simulate_query(self, args: list[str]) -> dict[str, Any]:
        """Simulate query command."""
        return {
            "success": True,
            "targets": [
                "//:pynomaly-lib",
                "//:pynomaly-cli",
                "//:pynomaly-api",
                "//:pynomaly-web",
                "//:web-assets",
                "//:test-all",
            ],
        }

    def _create_mock_artifact(self, target: str) -> Path:
        """Create mock build artifact."""
        # Extract target name
        target_name = target.split(":")[-1]

        # Create appropriate artifact based on target type
        if "test" in target_name:
            artifact_dir = self.buck_out_dir / "test-results"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{target_name}-results.xml"
            artifact_path.write_text(
                f'<testsuites><testsuite name="{target_name}"/></testsuites>'
            )
        elif "web-assets" in target_name:
            artifact_dir = self.buck_out_dir / "web"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "assets.tar.gz"
            artifact_path.write_text("mock web assets bundle")
        elif any(x in target_name for x in ["cli", "api", "web"]):
            artifact_dir = self.buck_out_dir / "bin"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{target_name}.pex"
            artifact_path.write_text(
                f"#!/usr/bin/env python3\n# Mock {target_name} binary"
            )
        else:
            artifact_dir = self.buck_out_dir / "lib"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{target_name}.tar"
            artifact_path.write_text(f"mock {target_name} library")

        return artifact_path


def main():
    """Main entry point for Buck2 simulator."""
    simulator = Buck2Simulator()

    # Parse command line arguments
    args = sys.argv[1:]

    # Run simulation
    result = simulator.simulate_command(args)

    if result.get("success"):
        if result.get("output"):
            print(result["output"])
        sys.exit(0)
    else:
        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
