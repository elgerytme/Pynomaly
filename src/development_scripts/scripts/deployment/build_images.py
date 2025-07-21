#!/usr/bin/env python3
"""
Docker Image Builder for Production Deployment
Builds optimized Docker images for all anomaly_detection components
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DockerImageBuilder:
    """Builds Docker images for production deployment"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.docker_registry = os.getenv("DOCKER_REGISTRY", "anomaly_detection")
        self.image_tag = os.getenv("IMAGE_TAG", "latest")
        self.build_args = self._get_build_args()

    def _get_build_args(self) -> dict[str, str]:
        """Get build arguments from environment"""
        return {
            "BUILD_DATE": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "VCS_REF": self._get_git_commit(),
            "VERSION": self._get_version(),
            "PYTHON_VERSION": "3.11",
        }

    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.stdout.strip()[:8]
        except Exception:
            return "unknown"

    def _get_version(self) -> str:
        """Get application version"""
        try:
            version_file = self.project_root / "src" / "anomaly_detection" / "_version.py"
            if version_file.exists():
                with open(version_file) as f:
                    content = f.read()
                    # Extract version from __version__ = "x.y.z"
                    for line in content.split("\n"):
                        if "__version__" in line:
                            return line.split('"')[1]
            return "0.1.0"
        except Exception:
            return "0.1.0"

    def build_image(
        self, dockerfile: str, image_name: str, context: str | None = None
    ) -> bool:
        """Build a single Docker image"""
        try:
            dockerfile_path = self.project_root / dockerfile
            if not dockerfile_path.exists():
                logger.error(f"Dockerfile not found: {dockerfile_path}")
                return False

            # Construct image tag
            full_image_name = f"{self.docker_registry}/{image_name}:{self.image_tag}"

            # Build command
            cmd = [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                full_image_name,
                "--label",
                f"org.opencontainers.image.created={self.build_args['BUILD_DATE']}",
                "--label",
                f"org.opencontainers.image.revision={self.build_args['VCS_REF']}",
                "--label",
                f"org.opencontainers.image.version={self.build_args['VERSION']}",
            ]

            # Add build args
            for key, value in self.build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add context
            build_context = context or str(self.project_root)
            cmd.append(build_context)

            logger.info(f"Building image: {full_image_name}")
            logger.info(f"Command: {' '.join(cmd)}")

            # Execute build
            result = subprocess.run(cmd, cwd=self.project_root)

            if result.returncode == 0:
                logger.info(f"Successfully built: {full_image_name}")
                return True
            else:
                logger.error(f"Failed to build: {full_image_name}")
                return False

        except Exception as e:
            logger.error(f"Error building image {image_name}: {e}")
            return False

    def build_all_images(self) -> bool:
        """Build all required Docker images"""
        images_to_build = [
            {
                "dockerfile": "Dockerfile.production",
                "image_name": "anomaly_detection-api",
                "context": None,
            },
            {
                "dockerfile": "deploy/docker/Dockerfile.web",
                "image_name": "anomaly_detection-web",
                "context": None,
            },
            {
                "dockerfile": "deploy/docker/Dockerfile.worker",
                "image_name": "anomaly_detection-worker",
                "context": None,
            },
            {
                "dockerfile": "deploy/docker/Dockerfile.monitoring",
                "image_name": "anomaly_detection-monitoring",
                "context": None,
            },
        ]

        built_images = []
        failed_images = []

        for image_config in images_to_build:
            dockerfile = image_config["dockerfile"]
            image_name = image_config["image_name"]
            context = image_config.get("context")

            logger.info(f"Building {image_name}...")

            if self.build_image(dockerfile, image_name, context):
                built_images.append(image_name)
            else:
                failed_images.append(image_name)

        # Report results
        logger.info(f"Built images: {len(built_images)}")
        for image in built_images:
            logger.info(f"  ✓ {image}")

        if failed_images:
            logger.error(f"Failed images: {len(failed_images)}")
            for image in failed_images:
                logger.error(f"  ✗ {image}")
            return False

        # Tag images for different environments
        self._tag_images_for_environments(built_images)

        return True

    def _tag_images_for_environments(self, images: list[str]):
        """Tag images for different environments"""
        environments = ["staging", "production"]

        for image_name in images:
            source_tag = f"{self.docker_registry}/{image_name}:{self.image_tag}"

            for env in environments:
                env_tag = f"{self.docker_registry}/{image_name}:{env}-{self.image_tag}"

                try:
                    subprocess.run(["docker", "tag", source_tag, env_tag], check=True)
                    logger.info(f"Tagged {source_tag} as {env_tag}")
                except subprocess.CalledProcessError:
                    logger.warning(f"Failed to tag {source_tag} as {env_tag}")

    def create_image_manifest(self, output_path: str | None = None) -> bool:
        """Create manifest of built images"""
        try:
            # Get list of images
            result = subprocess.run(
                ["docker", "images", f"{self.docker_registry}/*", "--format", "json"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error("Failed to get docker images list")
                return False

            images = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        images.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            manifest = {
                "build_date": self.build_args["BUILD_DATE"],
                "version": self.build_args["VERSION"],
                "git_commit": self.build_args["VCS_REF"],
                "registry": self.docker_registry,
                "tag": self.image_tag,
                "images": images,
            }

            # Save manifest
            manifest_path = (
                output_path
                or self.project_root
                / "artifacts"
                / "deployment"
                / "image_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"Image manifest saved: {manifest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create image manifest: {e}")
            return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Build Docker images for production")
    parser.add_argument("--registry", help="Docker registry prefix")
    parser.add_argument("--tag", help="Image tag")
    parser.add_argument("--manifest", help="Output path for image manifest")

    args = parser.parse_args()

    # Override defaults with command line arguments
    if args.registry:
        os.environ["DOCKER_REGISTRY"] = args.registry
    if args.tag:
        os.environ["IMAGE_TAG"] = args.tag

    builder = DockerImageBuilder()

    if builder.build_all_images():
        builder.create_image_manifest(args.manifest)
        logger.info("🎉 All images built successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some images failed to build!")
        sys.exit(1)


if __name__ == "__main__":
    main()
