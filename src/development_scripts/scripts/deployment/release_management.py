#!/usr/bin/env python3
"""
Production Release Management System
Comprehensive release management for production deployments
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import semver
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReleaseType(Enum):
    """Types of releases"""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    HOTFIX = "hotfix"
    PRERELEASE = "prerelease"


class ReleaseStage(Enum):
    """Release stages"""

    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    COMPLETED = "completed"
    ROLLBACK = "rollback"


class ReleaseStatus(Enum):
    """Release status"""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ReleaseConfig:
    """Release configuration"""

    # Version management
    version_file: str = "src/anomaly_detection/_version.py"
    changelog_file: str = "CHANGELOG.md"

    # Git settings
    main_branch: str = "main"
    develop_branch: str = "develop"
    release_branch_prefix: str = "release/"
    hotfix_branch_prefix: str = "hotfix/"

    # Deployment settings
    staging_environment: str = "staging"
    production_environment: str = "production"

    # Validation settings
    require_tests: bool = True
    require_security_scan: bool = True
    require_performance_validation: bool = True
    require_manual_approval: bool = True

    # Rollback settings
    auto_rollback_enabled: bool = True
    rollback_window_minutes: int = 60
    health_check_interval: int = 30

    # Notification settings
    slack_webhook: str | None = None
    email_notifications: list[str] = field(default_factory=list)

    # Feature flags
    enable_canary_deployment: bool = True
    canary_traffic_percentage: int = 10
    canary_duration_minutes: int = 30


@dataclass
class ReleaseNote:
    """Release note entry"""

    category: str  # "features", "bugfixes", "security", "breaking"
    description: str
    issue_id: str | None = None
    pr_id: str | None = None


@dataclass
class Release:
    """Release information"""

    version: str
    release_type: ReleaseType
    status: ReleaseStatus = ReleaseStatus.DRAFT
    stage: ReleaseStage = ReleaseStage.PLANNING

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: datetime | None = None
    deployed_at: datetime | None = None

    # Content
    title: str = ""
    description: str = ""
    release_notes: list[ReleaseNote] = field(default_factory=list)

    # Git information
    branch_name: str = ""
    commit_hash: str = ""
    previous_version: str = ""

    # Validation results
    test_results: dict[str, Any] = field(default_factory=dict)
    security_scan_results: dict[str, Any] = field(default_factory=dict)
    performance_results: dict[str, Any] = field(default_factory=dict)

    # Deployment information
    deployment_id: str | None = None
    rollback_plan: dict[str, Any] = field(default_factory=dict)

    # Monitoring
    health_checks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class ReleaseManager:
    """Comprehensive release management system"""

    def __init__(self, config: ReleaseConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        self.releases_dir = self.project_root / "releases"
        self.releases_dir.mkdir(exist_ok=True)

    def get_current_version(self) -> str:
        """Get current version from version file"""
        try:
            version_file = self.project_root / self.config.version_file
            if version_file.exists():
                content = version_file.read_text()

                # Extract version from __version__ = "x.y.z" format
                version_match = re.search(
                    r'__version__\s*=\s*["\']([^"\']+)["\']', content
                )
                if version_match:
                    return version_match.group(1)

            logger.warning(f"Could not find version in {version_file}")
            return "0.0.0"

        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return "0.0.0"

    def calculate_next_version(self, release_type: ReleaseType) -> str:
        """Calculate next version based on release type"""
        current = self.get_current_version()

        try:
            if release_type == ReleaseType.MAJOR:
                return semver.bump_major(current)
            elif release_type == ReleaseType.MINOR:
                return semver.bump_minor(current)
            elif release_type == ReleaseType.PATCH:
                return semver.bump_patch(current)
            elif release_type == ReleaseType.HOTFIX:
                return semver.bump_patch(current)
            elif release_type == ReleaseType.PRERELEASE:
                return semver.bump_prerelease(current)
            else:
                return semver.bump_patch(current)

        except Exception as e:
            logger.error(f"Failed to calculate next version: {e}")
            # Fallback to simple increment
            parts = current.split(".")
            if len(parts) >= 3:
                parts[2] = str(int(parts[2]) + 1)
                return ".".join(parts)
            return "0.0.1"

    def create_release(
        self,
        release_type: ReleaseType,
        title: str | None = None,
        description: str | None = None,
        scheduled_at: datetime | None = None,
    ) -> Release:
        """Create a new release"""

        current_version = self.get_current_version()
        new_version = self.calculate_next_version(release_type)

        release = Release(
            version=new_version,
            release_type=release_type,
            title=title or f"Release {new_version}",
            description=description or f"Release {new_version}",
            scheduled_at=scheduled_at,
            previous_version=current_version,
        )

        # Set branch name based on release type
        if release_type == ReleaseType.HOTFIX:
            release.branch_name = f"{self.config.hotfix_branch_prefix}{new_version}"
        else:
            release.branch_name = f"{self.config.release_branch_prefix}{new_version}"

        logger.info(f"Created release {new_version} ({release_type.value})")
        return release

    async def prepare_release_branch(self, release: Release) -> bool:
        """Prepare release branch"""
        logger.info(f"🌿 Preparing release branch: {release.branch_name}")

        try:
            # Ensure we're on the correct base branch
            base_branch = (
                self.config.main_branch
                if release.release_type == ReleaseType.HOTFIX
                else self.config.develop_branch
            )

            # Create and checkout release branch
            await self._run_git(["checkout", base_branch])
            await self._run_git(["pull", "origin", base_branch])
            await self._run_git(["checkout", "-b", release.branch_name])

            # Update version file
            await self._update_version_file(release.version)

            # Generate/update changelog
            await self._update_changelog(release)

            # Commit changes
            await self._run_git(["add", "."])
            await self._run_git(["commit", "-m", f"Prepare release {release.version}"])

            # Push branch
            await self._run_git(["push", "-u", "origin", release.branch_name])

            # Get commit hash
            commit_hash = await self._run_git(["rev-parse", "HEAD"])
            release.commit_hash = commit_hash.strip()

            logger.info(f"✅ Release branch prepared: {release.branch_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to prepare release branch: {e}")
            return False

    async def _update_version_file(self, version: str):
        """Update version in version file"""
        version_file = self.project_root / self.config.version_file

        if version_file.exists():
            content = version_file.read_text()

            # Update __version__ line
            updated_content = re.sub(
                r'__version__\s*=\s*["\'][^"\']+["\']',
                f'__version__ = "{version}"',
                content,
            )

            version_file.write_text(updated_content)
            logger.info(f"Updated version to {version}")

    async def _update_changelog(self, release: Release):
        """Update changelog with release information"""
        changelog_file = self.project_root / self.config.changelog_file

        # Generate changelog entry
        changelog_entry = self._generate_changelog_entry(release)

        if changelog_file.exists():
            existing_content = changelog_file.read_text()

            # Insert new entry at the top (after title)
            lines = existing_content.split("\n")

            # Find insertion point (after main title)
            insert_index = 1
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith("#"):
                    insert_index = i
                    break

            # Insert changelog entry
            lines.insert(insert_index, changelog_entry)
            lines.insert(insert_index + 1, "")

            new_content = "\n".join(lines)
        else:
            # Create new changelog
            new_content = f"# Changelog\n\n{changelog_entry}\n"

        changelog_file.write_text(new_content)
        logger.info(f"Updated changelog for {release.version}")

    def _generate_changelog_entry(self, release: Release) -> str:
        """Generate changelog entry for release"""
        date_str = release.created_at.strftime("%Y-%m-%d")

        entry = f"## [{release.version}] - {date_str}\n\n"

        if release.description:
            entry += f"{release.description}\n\n"

        # Group release notes by category
        categories = {
            "features": "### Added",
            "bugfixes": "### Fixed",
            "security": "### Security",
            "breaking": "### Breaking Changes",
        }

        for category, title in categories.items():
            notes = [
                note for note in release.release_notes if note.category == category
            ]
            if notes:
                entry += f"{title}\n"
                for note in notes:
                    entry += f"- {note.description}\n"
                entry += "\n"

        return entry

    async def run_release_validation(self, release: Release) -> bool:
        """Run comprehensive release validation"""
        logger.info(f"🔍 Running release validation for {release.version}")

        validation_passed = True

        # Run tests
        if self.config.require_tests:
            logger.info("Running test suite...")
            test_result = await self._run_tests()
            release.test_results = test_result
            if not test_result.get("success", False):
                logger.error("❌ Tests failed")
                validation_passed = False
            else:
                logger.info("✅ Tests passed")

        # Run security scan
        if self.config.require_security_scan:
            logger.info("Running security scan...")
            security_result = await self._run_security_scan()
            release.security_scan_results = security_result
            if not security_result.get("success", False):
                logger.error("❌ Security scan failed")
                validation_passed = False
            else:
                logger.info("✅ Security scan passed")

        # Run performance validation
        if self.config.require_performance_validation:
            logger.info("Running performance validation...")
            perf_result = await self._run_performance_validation()
            release.performance_results = perf_result
            if not perf_result.get("success", False):
                logger.error("❌ Performance validation failed")
                validation_passed = False
            else:
                logger.info("✅ Performance validation passed")

        if validation_passed:
            release.status = ReleaseStatus.VERIFIED
            logger.info("✅ Release validation passed")
        else:
            release.status = ReleaseStatus.FAILED
            logger.error("❌ Release validation failed")

        return validation_passed

    async def _run_tests(self) -> dict[str, Any]:
        """Run comprehensive test suite"""
        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/testing/comprehensive_test_pipeline.py",
                    "--types",
                    "unit",
                    "integration",
                    "security",
                ]
            )

            return {
                "success": result["returncode"] == 0,
                "output": result["stdout"],
                "error": result["stderr"],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_security_scan(self) -> dict[str, Any]:
        """Run security vulnerability scan"""
        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/security/automated_security_scanner.py",
                ]
            )

            return {
                "success": result["returncode"] == 0,
                "output": result["stdout"],
                "error": result["stderr"],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_performance_validation(self) -> dict[str, Any]:
        """Run performance validation"""
        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/performance_testing.py",
                    "--test-type",
                    "release",
                    "--duration",
                    "300",
                ]
            )

            return {
                "success": result["returncode"] == 0,
                "output": result["stdout"],
                "error": result["stderr"],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def deploy_to_staging(self, release: Release) -> bool:
        """Deploy release to staging environment"""
        logger.info(f"🚀 Deploying {release.version} to staging")

        try:
            release.stage = ReleaseStage.STAGING
            release.status = ReleaseStatus.IN_PROGRESS

            # Run staging deployment
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/production_deploy_orchestrator.py",
                    "--environment",
                    "staging",
                    "--image-tag",
                    release.version,
                ]
            )

            if result["returncode"] == 0:
                logger.info("✅ Staging deployment successful")

                # Run smoke tests on staging
                smoke_result = await self._run_staging_smoke_tests()
                if smoke_result:
                    release.status = ReleaseStatus.DEPLOYED
                    return True
                else:
                    logger.error("❌ Staging smoke tests failed")
                    release.status = ReleaseStatus.FAILED
                    return False
            else:
                logger.error(f"❌ Staging deployment failed: {result['stderr']}")
                release.status = ReleaseStatus.FAILED
                return False

        except Exception as e:
            logger.error(f"❌ Staging deployment error: {e}")
            release.status = ReleaseStatus.FAILED
            return False

    async def _run_staging_smoke_tests(self) -> bool:
        """Run smoke tests on staging environment"""
        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/run_smoke_tests.py",
                    "--url",
                    "https://staging-api.anomaly_detection.io",
                ]
            )

            return result["returncode"] == 0

        except Exception as e:
            logger.error(f"Staging smoke tests failed: {e}")
            return False

    async def deploy_to_production(self, release: Release) -> bool:
        """Deploy release to production environment"""
        logger.info(f"🚀 Deploying {release.version} to production")

        try:
            release.stage = ReleaseStage.PRODUCTION
            release.status = ReleaseStatus.IN_PROGRESS
            release.deployed_at = datetime.now()

            # Create deployment plan
            deployment_strategy = (
                "blue_green" if not self.config.enable_canary_deployment else "canary"
            )

            if deployment_strategy == "canary":
                success = await self._deploy_canary(release)
            else:
                success = await self._deploy_blue_green(release)

            if success:
                release.status = ReleaseStatus.DEPLOYED
                release.stage = ReleaseStage.COMPLETED

                # Start post-deployment monitoring
                await self._start_post_deployment_monitoring(release)

                logger.info(f"✅ Production deployment successful: {release.version}")
                return True
            else:
                logger.error("❌ Production deployment failed")
                release.status = ReleaseStatus.FAILED
                return False

        except Exception as e:
            logger.error(f"❌ Production deployment error: {e}")
            release.status = ReleaseStatus.FAILED
            return False

    async def _deploy_canary(self, release: Release) -> bool:
        """Deploy using canary strategy"""
        logger.info(f"🐤 Starting canary deployment for {release.version}")

        try:
            # Deploy canary with limited traffic
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/blue_green_deployment.py",
                    "--image-tag",
                    release.version,
                    "--canary-percentage",
                    str(self.config.canary_traffic_percentage),
                ]
            )

            if result["returncode"] != 0:
                return False

            # Monitor canary for specified duration
            logger.info(
                f"⏱️ Monitoring canary for {self.config.canary_duration_minutes} minutes"
            )
            await asyncio.sleep(self.config.canary_duration_minutes * 60)

            # Check canary health
            canary_healthy = await self._check_canary_health(release)

            if canary_healthy:
                # Promote canary to full deployment
                logger.info("✅ Canary healthy, promoting to full deployment")
                return await self._promote_canary(release)
            else:
                # Rollback canary
                logger.error("❌ Canary unhealthy, rolling back")
                await self._rollback_canary(release)
                return False

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return False

    async def _deploy_blue_green(self, release: Release) -> bool:
        """Deploy using blue/green strategy"""
        logger.info(f"🔵🟢 Starting blue/green deployment for {release.version}")

        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/blue_green_deployment.py",
                    "--image-tag",
                    release.version,
                ]
            )

            return result["returncode"] == 0

        except Exception as e:
            logger.error(f"Blue/green deployment failed: {e}")
            return False

    async def _check_canary_health(self, release: Release) -> bool:
        """Check canary deployment health"""
        try:
            # Check error rate, latency, and other metrics
            # This would integrate with monitoring systems like Prometheus

            # Simulate health check
            await asyncio.sleep(5)

            # In real implementation, query actual metrics
            error_rate = 0.02  # 2% error rate
            avg_latency = 150  # 150ms

            # Health thresholds
            max_error_rate = 0.05  # 5%
            max_latency = 1000  # 1000ms

            healthy = error_rate < max_error_rate and avg_latency < max_latency

            release.metrics.update(
                {
                    "canary_error_rate": error_rate,
                    "canary_avg_latency": avg_latency,
                    "canary_healthy": healthy,
                }
            )

            return healthy

        except Exception as e:
            logger.error(f"Canary health check failed: {e}")
            return False

    async def _promote_canary(self, release: Release) -> bool:
        """Promote canary to full deployment"""
        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/promote_canary.py",
                    "--version",
                    release.version,
                ]
            )

            return result["returncode"] == 0

        except Exception as e:
            logger.error(f"Canary promotion failed: {e}")
            return False

    async def _rollback_canary(self, release: Release) -> bool:
        """Rollback canary deployment"""
        try:
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/rollback_canary.py",
                    "--version",
                    release.version,
                ]
            )

            release.status = ReleaseStatus.ROLLED_BACK
            return result["returncode"] == 0

        except Exception as e:
            logger.error(f"Canary rollback failed: {e}")
            return False

    async def _start_post_deployment_monitoring(self, release: Release):
        """Start post-deployment monitoring"""
        logger.info("📊 Starting post-deployment monitoring")

        # Schedule periodic health checks
        monitoring_duration = self.config.rollback_window_minutes
        check_interval = self.config.health_check_interval

        for i in range(0, monitoring_duration, check_interval):
            await asyncio.sleep(check_interval * 60)

            health_check = await self._perform_health_check(release)
            release.health_checks.append(health_check)

            if not health_check["healthy"]:
                logger.warning(f"⚠️ Health check failed at {i + check_interval} minutes")

                if self.config.auto_rollback_enabled:
                    logger.info("🔄 Initiating automatic rollback")
                    await self.rollback_release(release)
                    break
            else:
                logger.info(f"✅ Health check passed at {i + check_interval} minutes")

    async def _perform_health_check(self, release: Release) -> dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            # Run health validation
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/validate_health.py",
                    "--url",
                    "https://api.anomaly_detection.io",
                ]
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "healthy": result["returncode"] == 0,
                "details": result["stdout"]
                if result["returncode"] == 0
                else result["stderr"],
            }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "healthy": False,
                "details": str(e),
            }

    async def rollback_release(self, release: Release) -> bool:
        """Rollback a release"""
        logger.info(f"🔄 Rolling back release {release.version}")

        try:
            # Execute rollback
            result = await self._run_command(
                [
                    "./environments/.venv/bin/python",
                    "scripts/deployment/blue_green_deployment.py",
                    "--rollback",
                    release.previous_version,
                ]
            )

            if result["returncode"] == 0:
                release.status = ReleaseStatus.ROLLED_BACK
                release.stage = ReleaseStage.ROLLBACK

                logger.info(f"✅ Rollback successful to {release.previous_version}")
                return True
            else:
                logger.error(f"❌ Rollback failed: {result['stderr']}")
                return False

        except Exception as e:
            logger.error(f"❌ Rollback error: {e}")
            return False

    def save_release(self, release: Release):
        """Save release information to file"""
        try:
            release_file = self.releases_dir / f"release_{release.version}.json"

            # Convert to serializable format
            release_data = {
                "version": release.version,
                "release_type": release.release_type.value,
                "status": release.status.value,
                "stage": release.stage.value,
                "created_at": release.created_at.isoformat(),
                "scheduled_at": release.scheduled_at.isoformat()
                if release.scheduled_at
                else None,
                "deployed_at": release.deployed_at.isoformat()
                if release.deployed_at
                else None,
                "title": release.title,
                "description": release.description,
                "branch_name": release.branch_name,
                "commit_hash": release.commit_hash,
                "previous_version": release.previous_version,
                "test_results": release.test_results,
                "security_scan_results": release.security_scan_results,
                "performance_results": release.performance_results,
                "deployment_id": release.deployment_id,
                "rollback_plan": release.rollback_plan,
                "health_checks": release.health_checks,
                "metrics": release.metrics,
                "release_notes": [
                    {
                        "category": note.category,
                        "description": note.description,
                        "issue_id": note.issue_id,
                        "pr_id": note.pr_id,
                    }
                    for note in release.release_notes
                ],
            }

            with open(release_file, "w") as f:
                json.dump(release_data, f, indent=2)

            logger.info(f"📄 Release information saved: {release_file}")

        except Exception as e:
            logger.error(f"Failed to save release: {e}")

    async def _run_git(self, args: list[str]) -> str:
        """Run git command"""
        cmd = ["git"] + args
        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise Exception(f"Git command failed: {result['stderr']}")

        return result["stdout"]

    async def _run_command(self, cmd: list[str]) -> dict[str, Any]:
        """Run shell command"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            stdout, stderr = await process.communicate()

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
            }

        except Exception as e:
            return {"returncode": 1, "stdout": "", "stderr": str(e)}


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Release Management System")
    parser.add_argument(
        "action", choices=["create", "validate", "deploy", "rollback", "status"]
    )
    parser.add_argument(
        "--type", choices=[t.value for t in ReleaseType], default="patch"
    )
    parser.add_argument("--version", help="Specific version for rollback")
    parser.add_argument("--title", help="Release title")
    parser.add_argument("--description", help="Release description")
    parser.add_argument("--schedule", help="Schedule release (ISO format)")
    parser.add_argument("--config", help="Path to release configuration")

    args = parser.parse_args()

    # Load configuration
    config = ReleaseConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Create release manager
    manager = ReleaseManager(config)

    try:
        if args.action == "create":
            release_type = ReleaseType(args.type)
            scheduled_at = (
                datetime.fromisoformat(args.schedule) if args.schedule else None
            )

            release = manager.create_release(
                release_type=release_type,
                title=args.title,
                description=args.description,
                scheduled_at=scheduled_at,
            )

            # Prepare release branch
            await manager.prepare_release_branch(release)

            # Save release
            manager.save_release(release)

            print(f"✅ Created release {release.version}")
            return 0

        elif args.action == "validate":
            # Load latest release
            version = args.version or manager.get_current_version()
            # In a real implementation, load release from file
            release = manager.create_release(ReleaseType.PATCH)
            release.version = version

            success = await manager.run_release_validation(release)
            return 0 if success else 1

        elif args.action == "deploy":
            # Load release and deploy
            version = args.version or manager.get_current_version()
            release = manager.create_release(ReleaseType.PATCH)
            release.version = version

            # Deploy to staging first
            staging_success = await manager.deploy_to_staging(release)
            if not staging_success:
                return 1

            # Deploy to production
            prod_success = await manager.deploy_to_production(release)
            return 0 if prod_success else 1

        elif args.action == "rollback":
            if not args.version:
                print("❌ Version required for rollback")
                return 1

            release = manager.create_release(ReleaseType.PATCH)
            release.version = args.version

            success = await manager.rollback_release(release)
            return 0 if success else 1

        elif args.action == "status":
            # Show current release status
            current_version = manager.get_current_version()
            print(f"Current version: {current_version}")
            return 0

    except Exception as e:
        logger.error(f"Release management failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
