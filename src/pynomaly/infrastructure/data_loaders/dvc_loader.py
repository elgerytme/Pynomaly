"""DVC integration for data versioning and management."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataLoadingError
from pynomaly.shared.protocols import DataLoaderProtocol


class DVCConfig(BaseModel):
    """Configuration for DVC integration."""

    repo_path: Path = Field(default_factory=lambda: Path.cwd())
    remote_name: str | None = Field(default=None, description="DVC remote name")
    cache_dir: Path | None = Field(
        default=None, description="Custom DVC cache directory"
    )
    auto_stage: bool = Field(
        default=True, description="Automatically stage files to DVC"
    )


class DVCLoader(DataLoaderProtocol):
    """Data loader with DVC integration for versioned datasets."""

    def __init__(self, config: DVCConfig | None = None):
        """Initialize DVC loader.

        Args:
            config: DVC configuration settings
        """
        self.config = config or DVCConfig()
        self._ensure_dvc_available()
        self._initialize_dvc()

    def _ensure_dvc_available(self) -> None:
        """Check if DVC is available."""
        try:
            import dvc  # noqa: F401
        except ImportError as e:
            raise DataLoadingError(
                "DVC not available. Install with: pip install 'pynomaly[data-formats]'"
            ) from e

    def _initialize_dvc(self) -> None:
        """Initialize DVC repository if needed."""
        dvc_dir = self.config.repo_path / ".dvc"
        if not dvc_dir.exists():
            # Initialize DVC repo
            try:
                subprocess.run(
                    ["dvc", "init"],
                    cwd=self.config.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise DataLoadingError(f"Failed to initialize DVC: {e.stderr}") from e

    def load(self, path: str | Path, **kwargs: Any) -> Dataset:
        """Load dataset with DVC tracking.

        Args:
            path: Path to the data file
            **kwargs: Additional arguments for pandas loading

        Returns:
            Dataset object with DVC metadata
        """
        path = Path(path)

        # Check if file is already tracked by DVC
        dvc_file = path.with_suffix(path.suffix + ".dvc")

        if dvc_file.exists():
            # File is tracked by DVC, pull if needed
            self._dvc_pull(str(path))
        elif self.config.auto_stage:
            # Add file to DVC tracking
            self._dvc_add(str(path))

        # Load the actual data
        if not path.exists():
            raise DataLoadingError(f"Data file not found: {path}")

        try:
            # Determine file type and load accordingly
            if path.suffix.lower() in [".csv"]:
                data = pd.read_csv(path, **kwargs)
            elif path.suffix.lower() in [".parquet"]:
                data = pd.read_parquet(path, **kwargs)
            elif path.suffix.lower() in [".json"]:
                data = pd.read_json(path, **kwargs)
            else:
                raise DataLoadingError(f"Unsupported file format: {path.suffix}")

            # Create dataset with DVC metadata
            metadata = self._get_dvc_metadata(path)
            return Dataset(
                name=path.stem,
                data=data,
                metadata={
                    "dvc_tracked": dvc_file.exists(),
                    "dvc_metadata": metadata,
                    "source_path": str(path),
                    "loader": "dvc",
                },
            )

        except Exception as e:
            raise DataLoadingError(f"Failed to load data from {path}: {str(e)}") from e

    def save(self, dataset: Dataset, path: str | Path, **kwargs: Any) -> None:
        """Save dataset with DVC tracking.

        Args:
            dataset: Dataset to save
            path: Output path
            **kwargs: Additional arguments for saving
        """
        path = Path(path)

        try:
            # Save the data
            if path.suffix.lower() == ".csv":
                dataset.data.to_csv(path, index=False, **kwargs)
            elif path.suffix.lower() == ".parquet":
                dataset.data.to_parquet(path, index=False, **kwargs)
            elif path.suffix.lower() == ".json":
                dataset.data.to_json(path, **kwargs)
            else:
                raise DataLoadingError(f"Unsupported output format: {path.suffix}")

            # Add to DVC tracking
            if self.config.auto_stage:
                self._dvc_add(str(path))

        except Exception as e:
            raise DataLoadingError(f"Failed to save dataset to {path}: {str(e)}") from e

    def _dvc_add(self, file_path: str) -> None:
        """Add file to DVC tracking."""
        try:
            subprocess.run(
                ["dvc", "add", file_path],
                cwd=self.config.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Don't fail if file is already tracked
            if "already tracked" not in e.stderr:
                raise DataLoadingError(f"Failed to add file to DVC: {e.stderr}") from e

    def _dvc_pull(self, file_path: str) -> None:
        """Pull file from DVC remote."""
        try:
            subprocess.run(
                ["dvc", "pull", file_path + ".dvc"],
                cwd=self.config.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Don't fail if no remote configured or file not in remote
            if (
                "no remote" not in e.stderr.lower()
                and "not found" not in e.stderr.lower()
            ):
                raise DataLoadingError(
                    f"Failed to pull file from DVC: {e.stderr}"
                ) from e

    def _get_dvc_metadata(self, file_path: Path) -> dict[str, Any]:
        """Get DVC metadata for a file."""
        dvc_file = file_path.with_suffix(file_path.suffix + ".dvc")

        if not dvc_file.exists():
            return {}

        try:
            import yaml

            with open(dvc_file) as f:
                metadata = yaml.safe_load(f)
            return metadata
        except Exception:
            return {}

    def push_to_remote(self, file_path: str) -> None:
        """Push file to DVC remote storage.

        Args:
            file_path: Path to the file to push
        """
        if not self.config.remote_name:
            raise DataLoadingError("No DVC remote configured")

        try:
            subprocess.run(
                ["dvc", "push", file_path + ".dvc"],
                cwd=self.config.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise DataLoadingError(f"Failed to push to DVC remote: {e.stderr}") from e

    def list_versions(self, file_path: str) -> list[str]:
        """List available versions of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of available versions (git commits)
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--", file_path + ".dvc"],
                cwd=self.config.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            versions = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    commit_hash = line.split()[0]
                    versions.append(commit_hash)
            return versions
        except subprocess.CalledProcessError:
            return []

    def checkout_version(self, file_path: str, version: str) -> None:
        """Checkout a specific version of a file.

        Args:
            file_path: Path to the file
            version: Git commit hash or tag
        """
        try:
            # Checkout the DVC file at specific version
            subprocess.run(
                ["git", "checkout", version, "--", file_path + ".dvc"],
                cwd=self.config.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Pull the data for that version
            self._dvc_pull(file_path)

        except subprocess.CalledProcessError as e:
            raise DataLoadingError(
                f"Failed to checkout version {version}: {e.stderr}"
            ) from e


def create_dvc_loader(repo_path: Path | None = None, **kwargs: Any) -> DVCLoader:
    """Factory function to create DVC loader.

    Args:
        repo_path: Path to DVC repository
        **kwargs: Additional configuration options

    Returns:
        Configured DVC loader
    """
    config = DVCConfig(repo_path=repo_path or Path.cwd(), **kwargs)
    return DVCLoader(config)
