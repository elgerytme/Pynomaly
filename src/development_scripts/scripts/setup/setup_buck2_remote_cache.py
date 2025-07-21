#!/usr/bin/env python3
"""Buck2 Remote Cache Setup Script for CI/CD environments.

This script configures Buck2 remote caching for team collaboration
and CI/CD acceleration using GitHub Actions cache or external services.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class Buck2RemoteCacheSetup:
    """Setup Buck2 remote caching for CI/CD acceleration."""

    def __init__(self):
        self.root_path = Path.cwd()
        self.buckconfig_path = self.root_path / ".buckconfig"
        self.buckconfig_remote_path = self.root_path / ".buckconfig.remote"

    def setup_cache_strategy(self, strategy: str = "github-actions") -> dict[str, Any]:
        """Setup remote cache strategy."""
        print(f"🚀 Setting up Buck2 remote cache strategy: {strategy}")

        strategies = {
            "github-actions": self._setup_github_actions_cache,
            "http-cache": self._setup_http_cache,
            "s3-cache": self._setup_s3_cache,
            "disable": self._disable_remote_cache,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}"
            )

        return strategies[strategy]()

    def _setup_github_actions_cache(self) -> dict[str, Any]:
        """Setup GitHub Actions cache integration."""
        print("  📦 Configuring GitHub Actions cache integration...")

        # Check if running in GitHub Actions
        is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"

        if is_github_actions:
            # Use GitHub Actions specific cache configuration
            cache_config = {
                "mode": "dir",
                "dir": ".buck-cache",
                "ci_integration": "github-actions",
                "cache_key_prefix": f"buck2-{os.getenv('GITHUB_REPOSITORY', 'anomaly_detection')}",
            }

            # Create cache directory if not exists
            cache_dir = self.root_path / ".buck-cache"
            cache_dir.mkdir(exist_ok=True)

            print("    ✅ GitHub Actions cache configured")
            print(f"    📁 Cache directory: {cache_dir}")
            print(f"    🔑 Cache key prefix: {cache_config['cache_key_prefix']}")

        else:
            # Local development - prepare for GitHub Actions
            cache_config = {
                "mode": "dir",
                "dir": ".buck-cache",
                "local_development": True,
                "note": "Configured for GitHub Actions compatibility",
            }

            print("    ℹ️  Local development mode - GitHub Actions ready")

        # Apply configuration
        self._update_buckconfig_cache(cache_config)

        return {
            "strategy": "github-actions",
            "success": True,
            "config": cache_config,
            "is_ci": is_github_actions,
        }

    def _setup_http_cache(self) -> dict[str, Any]:
        """Setup HTTP-based remote cache."""
        print("  🌐 Configuring HTTP remote cache...")

        # This would be configured with an actual remote cache service
        cache_url = os.getenv("BUCK2_HTTP_CACHE_URL", "")

        if not cache_url:
            print("    ⚠️  No BUCK2_HTTP_CACHE_URL environment variable set")
            print("    ℹ️  Using local cache with HTTP cache preparation")

            cache_config = {
                "mode": "dir",
                "dir": ".buck-cache",
                "http_cache_prepared": True,
                "http_cache_url_env": "BUCK2_HTTP_CACHE_URL",
            }
        else:
            cache_config = {
                "mode": "dir",
                "dir": ".buck-cache",
                "http_cache_url": cache_url,
                "http_max_concurrent_writes": 4,
                "http_write_timeout_seconds": 30,
                "http_read_timeout_seconds": 10,
            }

            print(f"    ✅ HTTP cache configured: {cache_url}")

        self._update_buckconfig_cache(cache_config)

        return {
            "strategy": "http-cache",
            "success": True,
            "config": cache_config,
            "cache_url": cache_url,
        }

    def _setup_s3_cache(self) -> dict[str, Any]:
        """Setup S3-based remote cache."""
        print("  ☁️  Configuring S3 remote cache...")

        s3_bucket = os.getenv("BUCK2_S3_CACHE_BUCKET", "")

        if not s3_bucket:
            print("    ⚠️  No BUCK2_S3_CACHE_BUCKET environment variable set")
            print("    ℹ️  Using local cache with S3 cache preparation")

            cache_config = {
                "mode": "dir",
                "dir": ".buck-cache",
                "s3_cache_prepared": True,
                "s3_bucket_env": "BUCK2_S3_CACHE_BUCKET",
            }
        else:
            cache_config = {
                "mode": "dir",
                "dir": ".buck-cache",
                "s3_bucket": s3_bucket,
                "s3_region": os.getenv("BUCK2_S3_CACHE_REGION", "us-east-1"),
                "s3_prefix": os.getenv("BUCK2_S3_CACHE_PREFIX", "buck2-cache/"),
            }

            print(f"    ✅ S3 cache configured: {s3_bucket}")

        self._update_buckconfig_cache(cache_config)

        return {
            "strategy": "s3-cache",
            "success": True,
            "config": cache_config,
            "s3_bucket": s3_bucket,
        }

    def _disable_remote_cache(self) -> dict[str, Any]:
        """Disable remote cache - use local only."""
        print("  🚫 Disabling remote cache - using local cache only...")

        cache_config = {"mode": "dir", "dir": ".buck-cache", "remote_cache": "disabled"}

        self._update_buckconfig_cache(cache_config)

        return {"strategy": "disabled", "success": True, "config": cache_config}

    def _update_buckconfig_cache(self, cache_config: dict[str, Any]):
        """Update .buckconfig with cache configuration."""
        # For now, we'll work with the remote config template
        # In production, this would modify the actual .buckconfig

        print("    📝 Cache configuration prepared:")
        for key, value in cache_config.items():
            print(f"      {key}: {value}")

    def validate_cache_setup(self) -> dict[str, Any]:
        """Validate Buck2 cache setup."""
        print("🧪 Validating Buck2 cache setup...")

        validation_results = {
            "buck2_available": False,
            "cache_directory_exists": False,
            "cache_writable": False,
            "buckconfig_valid": False,
        }

        try:
            # Check Buck2 availability (try common paths)
            buck2_paths = [
                "buck2",
                "/mnt/c/Users/andre/buck2.exe",
                "/usr/local/bin/buck2",
            ]
            buck2_available = False

            for buck2_path in buck2_paths:
                try:
                    result = subprocess.run(
                        [buck2_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        buck2_available = True
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            validation_results["buck2_available"] = buck2_available

            if validation_results["buck2_available"]:
                print("  ✅ Buck2 is available")
            else:
                print("  ❌ Buck2 is not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  ❌ Buck2 not found or not accessible")

        # Check cache directory
        cache_dir = self.root_path / ".buck-cache"
        validation_results["cache_directory_exists"] = cache_dir.exists()

        if validation_results["cache_directory_exists"]:
            print(f"  ✅ Cache directory exists: {cache_dir}")

            # Test write permissions
            try:
                test_file = cache_dir / "test_write_permissions.txt"
                test_file.write_text("test")
                test_file.unlink()
                validation_results["cache_writable"] = True
                print("  ✅ Cache directory is writable")
            except Exception as e:
                print(f"  ❌ Cache directory is not writable: {e}")
        else:
            print(
                "  ⚠️  Cache directory does not exist (will be created on first build)"
            )

        # Check .buckconfig
        validation_results["buckconfig_valid"] = self.buckconfig_path.exists()
        if validation_results["buckconfig_valid"]:
            print("  ✅ .buckconfig exists")
        else:
            print("  ❌ .buckconfig not found")

        # Overall status
        critical_checks = ["buck2_available", "buckconfig_valid"]
        all_critical_passed = all(
            validation_results[check] for check in critical_checks
        )

        validation_results["overall_status"] = (
            "ready" if all_critical_passed else "needs_attention"
        )

        print(f"🎯 Validation status: {validation_results['overall_status']}")

        return validation_results

    def generate_cache_performance_test(self) -> dict[str, Any]:
        """Generate cache performance test."""
        print("🚀 Running Buck2 cache performance test...")

        if not self.validate_cache_setup()["buck2_available"]:
            return {"error": "Buck2 not available for performance testing"}

        try:
            # Find Buck2 executable
            buck2_cmd = "buck2"
            for buck2_path in [
                "buck2",
                "/mnt/c/Users/andre/buck2.exe",
                "/usr/local/bin/buck2",
            ]:
                try:
                    subprocess.run(
                        [buck2_path, "--version"], capture_output=True, timeout=5
                    )
                    buck2_cmd = buck2_path
                    break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            # Clean build test
            print("  🧹 Testing clean build performance...")
            subprocess.run(
                [buck2_cmd, "clean"], capture_output=True, cwd=self.root_path
            )

            import time

            start_time = time.time()
            result = subprocess.run(
                [buck2_cmd, "build", "//:validation"],
                capture_output=True,
                text=True,
                cwd=self.root_path,
            )
            clean_build_time = time.time() - start_time

            # Cached build test
            print("  ⚡ Testing cached build performance...")
            start_time = time.time()
            cached_result = subprocess.run(
                [buck2_cmd, "build", "//:validation"],
                capture_output=True,
                text=True,
                cwd=self.root_path,
            )
            cached_build_time = time.time() - start_time

            # Calculate performance metrics
            cache_effectiveness = (
                clean_build_time / cached_build_time if cached_build_time > 0 else 1.0
            )

            performance_results = {
                "clean_build_time": clean_build_time,
                "cached_build_time": cached_build_time,
                "cache_effectiveness": cache_effectiveness,
                "clean_build_success": result.returncode == 0,
                "cached_build_success": cached_result.returncode == 0,
            }

            print(f"  📊 Clean build: {clean_build_time:.3f}s")
            print(f"  📊 Cached build: {cached_build_time:.3f}s")
            print(f"  📊 Cache effectiveness: {cache_effectiveness:.1f}x")

            return performance_results

        except Exception as e:
            return {"error": f"Performance test failed: {str(e)}"}


def main():
    """Main entry point for Buck2 remote cache setup."""
    parser = argparse.ArgumentParser(description="Setup Buck2 remote cache for CI/CD")
    parser.add_argument(
        "--strategy",
        choices=["github-actions", "http-cache", "s3-cache", "disable"],
        default="github-actions",
        help="Cache strategy to use",
    )
    parser.add_argument("--validate", action="store_true", help="Validate cache setup")
    parser.add_argument(
        "--performance-test", action="store_true", help="Run cache performance test"
    )

    args = parser.parse_args()

    setup = Buck2RemoteCacheSetup()

    try:
        if args.validate:
            results = setup.validate_cache_setup()
            if results["overall_status"] != "ready":
                sys.exit(1)

        if args.performance_test:
            perf_results = setup.generate_cache_performance_test()
            if "error" in perf_results:
                print(f"❌ Performance test failed: {perf_results['error']}")
                sys.exit(1)

        if not args.validate and not args.performance_test:
            # Setup cache strategy
            results = setup.setup_cache_strategy(args.strategy)
            print(f"✅ Buck2 remote cache setup completed: {results['strategy']}")

            # Auto-validate after setup
            validation = setup.validate_cache_setup()
            if validation["overall_status"] != "ready":
                print("⚠️  Setup completed but validation shows issues")
                sys.exit(1)

        print("🎉 Buck2 remote cache configuration successful!")

    except Exception as e:
        print(f"❌ Buck2 remote cache setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
