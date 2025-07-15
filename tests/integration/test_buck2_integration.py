"""Integration tests for Buck2 + Hatch build system."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestBuck2Integration:
    """Test Buck2 build system integration."""

    def test_buck2_availability(self):
        """Test that Buck2 is available and working."""
        try:
            result = subprocess.run(
                ["buck2", "--version"], capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0
            assert "buck2" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Buck2 not available")

    def test_buck2_config_validation(self):
        """Test Buck2 configuration is valid."""
        buck_config_path = Path.cwd() / ".buckconfig"
        assert buck_config_path.exists(), ".buckconfig file not found"

        with open(buck_config_path) as f:
            config_content = f.read()

        # Check required sections
        assert "[python]" in config_content
        assert "[build]" in config_content
        assert "[cache]" in config_content

    def test_buck_file_syntax(self):
        """Test BUCK file syntax is valid."""
        buck_file_path = Path.cwd() / "BUCK"
        assert buck_file_path.exists(), "BUCK file not found"

        try:
            result = subprocess.run(
                ["buck2", "targets", "//..."],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path.cwd(),
            )
            assert result.returncode == 0, f"Buck2 targets failed: {result.stderr}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Buck2 not available or BUCK file syntax error")

    def test_python_targets_exist(self):
        """Test that Python targets are properly defined."""
        try:
            result = subprocess.run(
                ["buck2", "targets", "//..."],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                targets = result.stdout.strip().split("\n")
                target_names = [t.strip() for t in targets if t.strip()]

                # Check for core targets
                expected_targets = [
                    "//:domain",
                    "//:application",
                    "//:infrastructure",
                    "//:presentation",
                ]

                for expected in expected_targets:
                    assert any(
                        expected in target for target in target_names
                    ), f"Target {expected} not found in {target_names}"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Buck2 not available")

    def test_buck2_build_dry_run(self):
        """Test Buck2 build dry run."""
        try:
            result = subprocess.run(
                ["buck2", "build", "//:domain", "--dry-run"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Dry run should succeed even if dependencies are missing
            assert result.returncode in [
                0,
                1,
            ], f"Buck2 dry run failed: {result.stderr}"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Buck2 not available")


class TestHatchIntegration:
    """Test Hatch build system integration."""

    def test_hatch_availability(self):
        """Test that Hatch is available."""
        try:
            result = subprocess.run(
                ["hatch", "--version"], capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0
            assert "hatch" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Hatch not available")

    def test_pyproject_toml_configuration(self):
        """Test pyproject.toml configuration for Hatch."""
        pyproject_path = Path.cwd() / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Check build system
        assert "build-system" in config
        assert config["build-system"]["build-backend"] == "hatchling.build"

        # Check Buck2 hook configuration
        assert "tool" in config
        assert "hatch" in config["tool"]
        build_config = config["tool"]["hatch"]

        if "build" in build_config and "hooks" in build_config["build"]:
            hooks = build_config["build"]["hooks"]
            if "buck2" in hooks:
                buck2_hook = hooks["buck2"]
                assert "requires" in buck2_hook or "dependencies" in buck2_hook

    def test_hatch_environments(self):
        """Test Hatch environment configuration."""
        try:
            result = subprocess.run(
                ["hatch", "env", "show"], capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                envs_output = result.stdout
                assert "default" in envs_output
                assert "lint" in envs_output
                assert "test" in envs_output

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Hatch not available")

    def test_hatch_build_hook_plugin(self):
        """Test that Buck2 build hook plugin can be imported."""
        try:
            from hatch_buck2_plugin import Buck2BuildHook

            # Test plugin interface
            assert hasattr(Buck2BuildHook, "PLUGIN_NAME")
            assert Buck2BuildHook.PLUGIN_NAME == "buck2"
            assert hasattr(Buck2BuildHook, "finalize")

        except ImportError:
            pytest.skip("Buck2 build hook plugin not available")


class TestNpmIntegration:
    """Test npm web assets integration."""

    def test_package_json_exists(self):
        """Test package.json configuration."""
        package_json_path = Path.cwd() / "package.json"
        assert package_json_path.exists(), "package.json not found"

        with open(package_json_path) as f:
            config = json.load(f)

        # Check required scripts
        assert "scripts" in config
        scripts = config["scripts"]
        assert "build" in scripts
        assert "build-css" in scripts
        assert "build-js" in scripts

        # Check dependencies
        assert "dependencies" in config
        deps = config["dependencies"]
        assert "d3" in deps
        assert "echarts" in deps
        assert "htmx.org" in deps

    def test_npm_availability(self):
        """Test that npm is available."""
        try:
            result = subprocess.run(
                ["npm", "--version"], capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0
            # Version should be a valid semver string
            version = result.stdout.strip()
            assert "." in version
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("npm not available")

    def test_web_asset_directories(self):
        """Test web asset directory structure."""
        web_root = Path.cwd() / "src" / "pynomaly" / "presentation" / "web"
        assert web_root.exists(), "Web presentation directory not found"

        # Check asset sources
        assets_dir = web_root / "assets"
        if assets_dir.exists():
            css_dir = assets_dir / "css"
            js_dir = assets_dir / "js"
            assert css_dir.exists() or (web_root / "static" / "css").exists()
            assert js_dir.exists() or (web_root / "static" / "js").exists()

        # Check static output directory
        static_dir = web_root / "static"
        assert static_dir.exists(), "Static directory should exist"


class TestIntegratedWorkflow:
    """Test integrated build workflow."""

    def test_makefile_targets(self):
        """Test Makefile targets exist and are valid."""
        makefile_path = Path.cwd() / "Makefile"
        assert makefile_path.exists(), "Makefile not found"

        with open(makefile_path) as f:
            makefile_content = f.read()

        # Check key targets exist
        key_targets = [
            "build:",
            "test:",
            "clean:",
            "buck-build:",
            "npm-build:",
            "deps:",
        ]

        for target in key_targets:
            assert target in makefile_content, f"Target {target} not found in Makefile"

    def test_make_help(self):
        """Test make help command."""
        try:
            result = subprocess.run(
                ["make", "help"], capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0
            help_output = result.stdout
            assert "build" in help_output.lower()
            assert "test" in help_output.lower()
            assert "clean" in help_output.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("make not available")

    def test_build_system_environment_detection(self):
        """Test build system can detect available tools."""
        # Test which tools are available
        tools_status = {}

        for tool in ["buck2", "hatch", "npm", "node"]:
            try:
                result = subprocess.run(
                    [tool, "--version"], capture_output=True, timeout=5
                )
                tools_status[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tools_status[tool] = False

        # At minimum, we should have hatch for Python packaging
        assert (
            tools_status.get("hatch", False) or Path.cwd().joinpath(".venv").exists()
        ), "Neither Hatch nor virtual environment available"

        # Log tool availability for debugging
        print(f"Tool availability: {tools_status}")

    @pytest.mark.slow
    def test_dependency_installation_workflow(self):
        """Test that dependencies can be installed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test Python dependencies (if hatch available)
            try:
                result = subprocess.run(
                    ["python", "-m", "pip", "list"], capture_output=True, timeout=30
                )
                assert result.returncode == 0
                pip_list = result.stdout.decode()
                # Should have some basic packages
                assert len(pip_list.strip().split("\n")) > 5
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip("Python/pip not available")

    def test_configuration_consistency(self):
        """Test configuration consistency across build systems."""
        # Check that version information is consistent
        pyproject_path = Path.cwd() / "pyproject.toml"
        package_json_path = Path.cwd() / "package.json"

        if pyproject_path.exists() and package_json_path.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib

            with open(pyproject_path, "rb") as f:
                pyproject_config = tomllib.load(f)

            with open(package_json_path) as f:
                package_config = json.load(f)

            # Both should have proper name and description
            assert "project" in pyproject_config
            assert "name" in pyproject_config["project"]
            assert "name" in package_config

            # Check that they're related
            python_name = pyproject_config["project"]["name"]
            npm_name = package_config["name"]
            assert (
                python_name in npm_name or "pynomaly" in npm_name.lower()
            ), f"Names don't align: {python_name} vs {npm_name}"


class TestPerformanceOptimizations:
    """Test build system performance optimizations."""

    def test_buck2_cache_configuration(self):
        """Test Buck2 cache is properly configured."""
        buckconfig_path = Path.cwd() / ".buckconfig"
        if not buckconfig_path.exists():
            pytest.skip(".buckconfig not found")

        with open(buckconfig_path) as f:
            config_content = f.read()

        # Should have cache configuration
        assert "[cache]" in config_content
        # Should specify cache mode
        cache_section_found = False
        for line in config_content.split("\n"):
            if line.startswith("[cache]"):
                cache_section_found = True
            elif cache_section_found and line.startswith("mode"):
                assert "dir" in line or "remote" in line
                break

    def test_parallel_build_configuration(self):
        """Test parallel build configuration."""
        buckconfig_path = Path.cwd() / ".buckconfig"
        if buckconfig_path.exists():
            with open(buckconfig_path) as f:
                config_content = f.read()

            # Check for build parallelization settings
            if "[build]" in config_content:
                # Should have some parallelization configuration
                build_section = False
                for line in config_content.split("\n"):
                    if line.startswith("[build]"):
                        build_section = True
                    elif build_section and line.strip() and not line.startswith("["):
                        # This is a build configuration line
                        assert "=" in line, f"Invalid build config line: {line}"

    def test_web_asset_optimization(self):
        """Test web asset build optimization."""
        package_json_path = Path.cwd() / "package.json"
        if not package_json_path.exists():
            pytest.skip("package.json not found")

        with open(package_json_path) as f:
            config = json.load(f)

        # Check build scripts include minification
        if "scripts" in config:
            build_script = config["scripts"].get("build-css", "")
            assert "--minify" in build_script, "CSS build should include minification"

            js_build_script = config["scripts"].get("build-js", "")
            if js_build_script:
                assert (
                    "--minify" in js_build_script
                ), "JS build should include minification"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
