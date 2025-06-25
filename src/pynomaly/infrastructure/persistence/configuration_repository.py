"""Configuration repository for persistent storage of experiment configurations."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly.application.dto.configuration_dto import (
    ConfigurationCollectionDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    ConfigurationStatus,
    ConfigurationTemplateDTO,
    ExperimentConfigurationDTO,
)

logger = logging.getLogger(__name__)


class ConfigurationRepository:
    """Repository for configuration persistence with file-based storage."""

    def __init__(
        self,
        storage_path: Path,
        enable_versioning: bool = True,
        enable_compression: bool = False,
        backup_enabled: bool = True,
    ):
        """Initialize configuration repository.

        Args:
            storage_path: Base path for configuration storage
            enable_versioning: Enable configuration versioning
            enable_compression: Enable compression for storage
            backup_enabled: Enable automatic backups
        """
        self.storage_path = Path(storage_path)
        self.enable_versioning = enable_versioning
        self.enable_compression = enable_compression
        self.backup_enabled = backup_enabled

        # Initialize directory structure
        self._initialize_directory_structure()

        # Initialize metadata tracking
        self.metadata_file = self.storage_path / "repository_metadata.json"
        self.metadata = self._load_repository_metadata()

    def _initialize_directory_structure(self) -> None:
        """Initialize the repository directory structure."""
        directories = [
            self.storage_path,
            self.storage_path / "configurations",
            self.storage_path / "collections",
            self.storage_path / "templates",
            self.storage_path / "backups",
            self.storage_path / "indexes",
            self.storage_path / "exports",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized configuration repository at {self.storage_path}")

    async def save_configuration(
        self, configuration: ExperimentConfigurationDTO
    ) -> bool:
        """Save configuration to persistent storage.

        Args:
            configuration: Configuration to save

        Returns:
            True if saved successfully
        """
        try:
            config_file = self._get_configuration_file_path(configuration.id)

            # Create backup if configuration exists
            if config_file.exists() and self.backup_enabled:
                await self._create_backup(configuration.id)

            # Save configuration
            config_data = configuration.model_dump()

            # Add repository metadata
            config_data["_repository_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "file_version": "1.0",
                "compression_enabled": self.enable_compression,
            }

            if self.enable_compression:
                await self._save_compressed(config_file, config_data)
            else:
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, default=str)

            # Update indexes
            await self._update_indexes(configuration)

            # Update repository metadata
            await self._update_repository_metadata(configuration)

            logger.info(
                f"Saved configuration {configuration.name} ({configuration.id})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration {configuration.id}: {e}")
            return False

    async def load_configuration(
        self, config_id: UUID
    ) -> ExperimentConfigurationDTO | None:
        """Load configuration by ID.

        Args:
            config_id: Configuration ID

        Returns:
            Configuration if found, None otherwise
        """
        try:
            config_file = self._get_configuration_file_path(config_id)

            if not config_file.exists():
                return None

            if self.enable_compression:
                config_data = await self._load_compressed(config_file)
            else:
                with open(config_file, encoding="utf-8") as f:
                    config_data = json.load(f)

            # Remove repository metadata
            config_data.pop("_repository_metadata", None)

            configuration = ExperimentConfigurationDTO(**config_data)

            # Update last accessed timestamp
            await self._update_access_timestamp(config_id)

            return configuration

        except Exception as e:
            logger.error(f"Failed to load configuration {config_id}: {e}")
            return None

    async def delete_configuration(
        self, config_id: UUID, create_backup: bool = True
    ) -> bool:
        """Delete configuration from storage.

        Args:
            config_id: Configuration ID to delete
            create_backup: Whether to create backup before deletion

        Returns:
            True if deleted successfully
        """
        try:
            config_file = self._get_configuration_file_path(config_id)

            if not config_file.exists():
                return False

            # Create backup if requested
            if create_backup and self.backup_enabled:
                await self._create_backup(config_id)

            # Delete configuration file
            config_file.unlink()

            # Remove from indexes
            await self._remove_from_indexes(config_id)

            # Update repository metadata
            self.metadata["total_configurations"] -= 1
            self.metadata["last_modified"] = datetime.now().isoformat()
            await self._save_repository_metadata()

            logger.info(f"Deleted configuration {config_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete configuration {config_id}: {e}")
            return False

    async def search_configurations(
        self, request: ConfigurationSearchRequestDTO
    ) -> list[ExperimentConfigurationDTO]:
        """Search configurations based on criteria.

        Args:
            request: Search request parameters

        Returns:
            List of matching configurations
        """
        try:
            # Load all configurations (can be optimized with indexes)
            all_configs = await self._load_all_configurations()

            # Apply filters
            filtered_configs = self._apply_search_filters(all_configs, request)

            # Sort results
            sorted_configs = self._sort_configurations(filtered_configs, request)

            # Apply pagination
            start_idx = request.offset
            end_idx = start_idx + request.limit

            return sorted_configs[start_idx:end_idx]

        except Exception as e:
            logger.error(f"Configuration search failed: {e}")
            return []

    async def list_configurations(
        self,
        source: ConfigurationSource | None = None,
        status: ConfigurationStatus | None = None,
        limit: int = 100,
    ) -> list[ExperimentConfigurationDTO]:
        """List configurations with optional filtering.

        Args:
            source: Filter by configuration source
            status: Filter by configuration status
            limit: Maximum number of configurations to return

        Returns:
            List of configurations
        """
        try:
            all_configs = await self._load_all_configurations()

            # Apply filters
            filtered_configs = all_configs

            if source:
                filtered_configs = [
                    config
                    for config in filtered_configs
                    if config.metadata.source == source
                ]

            if status:
                filtered_configs = [
                    config for config in filtered_configs if config.status == status
                ]

            # Sort by creation date (newest first)
            filtered_configs.sort(key=lambda x: x.metadata.created_at, reverse=True)

            return filtered_configs[:limit]

        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return []

    async def save_collection(self, collection: ConfigurationCollectionDTO) -> bool:
        """Save configuration collection.

        Args:
            collection: Collection to save

        Returns:
            True if saved successfully
        """
        try:
            collection_file = (
                self.storage_path / "collections" / f"{collection.id}.json"
            )

            collection_data = collection.model_dump()
            with open(collection_file, "w", encoding="utf-8") as f:
                json.dump(collection_data, f, indent=2, default=str)

            logger.info(f"Saved collection {collection.name} ({collection.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to save collection {collection.id}: {e}")
            return False

    async def load_collection(
        self, collection_id: UUID
    ) -> ConfigurationCollectionDTO | None:
        """Load configuration collection by ID.

        Args:
            collection_id: Collection ID

        Returns:
            Collection if found, None otherwise
        """
        try:
            collection_file = (
                self.storage_path / "collections" / f"{collection_id}.json"
            )

            if not collection_file.exists():
                return None

            with open(collection_file, encoding="utf-8") as f:
                collection_data = json.load(f)

            return ConfigurationCollectionDTO(**collection_data)

        except Exception as e:
            logger.error(f"Failed to load collection {collection_id}: {e}")
            return None

    async def save_template(self, template: ConfigurationTemplateDTO) -> bool:
        """Save configuration template.

        Args:
            template: Template to save

        Returns:
            True if saved successfully
        """
        try:
            template_file = self.storage_path / "templates" / f"{template.id}.json"

            template_data = template.model_dump()
            with open(template_file, "w", encoding="utf-8") as f:
                json.dump(template_data, f, indent=2, default=str)

            logger.info(f"Saved template {template.name} ({template.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to save template {template.id}: {e}")
            return False

    async def load_template(self, template_id: UUID) -> ConfigurationTemplateDTO | None:
        """Load configuration template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template if found, None otherwise
        """
        try:
            template_file = self.storage_path / "templates" / f"{template_id}.json"

            if not template_file.exists():
                return None

            with open(template_file, encoding="utf-8") as f:
                template_data = json.load(f)

            return ConfigurationTemplateDTO(**template_data)

        except Exception as e:
            logger.error(f"Failed to load template {template_id}: {e}")
            return None

    async def export_configurations(
        self, config_ids: list[UUID], export_path: Path, include_metadata: bool = True
    ) -> bool:
        """Export configurations to specified path.

        Args:
            config_ids: Configuration IDs to export
            export_path: Path to export to
            include_metadata: Whether to include repository metadata

        Returns:
            True if exported successfully
        """
        try:
            export_data = {
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "total_configurations": len(config_ids),
                    "repository_path": str(self.storage_path),
                    "include_metadata": include_metadata,
                },
                "configurations": [],
            }

            for config_id in config_ids:
                config = await self.load_configuration(config_id)
                if config:
                    config_data = config.model_dump()
                    if not include_metadata:
                        config_data.pop("metadata", None)
                    export_data["configurations"].append(config_data)

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(
                f"Exported {len(export_data['configurations'])} configurations to {export_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to export configurations: {e}")
            return False

    async def import_configurations(
        self, import_path: Path, overwrite_existing: bool = False
    ) -> int:
        """Import configurations from specified path.

        Args:
            import_path: Path to import from
            overwrite_existing: Whether to overwrite existing configurations

        Returns:
            Number of configurations imported
        """
        try:
            with open(import_path, encoding="utf-8") as f:
                import_data = json.load(f)

            imported_count = 0
            configurations = import_data.get("configurations", [])

            for config_data in configurations:
                try:
                    config = ExperimentConfigurationDTO(**config_data)

                    # Check if configuration already exists
                    existing = await self.load_configuration(config.id)
                    if existing and not overwrite_existing:
                        logger.warning(
                            f"Configuration {config.id} already exists, skipping"
                        )
                        continue

                    # Save configuration
                    if await self.save_configuration(config):
                        imported_count += 1

                except Exception as e:
                    logger.warning(f"Failed to import configuration: {e}")
                    continue

            logger.info(f"Imported {imported_count} configurations from {import_path}")
            return imported_count

        except Exception as e:
            logger.error(f"Failed to import configurations: {e}")
            return 0

    def get_repository_statistics(self) -> dict[str, Any]:
        """Get repository statistics.

        Returns:
            Repository statistics dictionary
        """
        try:
            # Count files in each directory
            config_count = len(
                list((self.storage_path / "configurations").glob("*.json"))
            )
            collection_count = len(
                list((self.storage_path / "collections").glob("*.json"))
            )
            template_count = len(list((self.storage_path / "templates").glob("*.json")))
            backup_count = len(list((self.storage_path / "backups").glob("*.json")))

            # Calculate storage size
            total_size = sum(
                f.stat().st_size
                for f in self.storage_path.rglob("*.json")
                if f.is_file()
            )

            return {
                "total_configurations": config_count,
                "total_collections": collection_count,
                "total_templates": template_count,
                "total_backups": backup_count,
                "storage_size_bytes": total_size,
                "storage_path": str(self.storage_path),
                "versioning_enabled": self.enable_versioning,
                "compression_enabled": self.enable_compression,
                "backup_enabled": self.backup_enabled,
                "last_modified": self.metadata.get("last_modified"),
                "repository_created": self.metadata.get("created_at"),
            }

        except Exception as e:
            logger.error(f"Failed to get repository statistics: {e}")
            return {}

    # Private methods

    def _get_configuration_file_path(self, config_id: UUID) -> Path:
        """Get file path for configuration."""
        return self.storage_path / "configurations" / f"{config_id}.json"

    async def _load_all_configurations(self) -> list[ExperimentConfigurationDTO]:
        """Load all configurations from storage."""
        configurations = []
        config_dir = self.storage_path / "configurations"

        for config_file in config_dir.glob("*.json"):
            try:
                config_id = UUID(config_file.stem)
                config = await self.load_configuration(config_id)
                if config:
                    configurations.append(config)
            except Exception as e:
                logger.warning(f"Failed to load configuration {config_file}: {e}")
                continue

        return configurations

    def _apply_search_filters(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationSearchRequestDTO,
    ) -> list[ExperimentConfigurationDTO]:
        """Apply search filters to configuration list."""
        filtered = configurations

        # Filter by query
        if request.query:
            query_lower = request.query.lower()
            filtered = [
                config
                for config in filtered
                if (
                    query_lower in config.name.lower()
                    or query_lower in config.algorithm_config.algorithm_name.lower()
                    or (
                        config.metadata.description
                        and query_lower in config.metadata.description.lower()
                    )
                )
            ]

        # Filter by tags
        if request.tags:
            filtered = [
                config
                for config in filtered
                if all(tag in config.metadata.tags for tag in request.tags)
            ]

        # Filter by source
        if request.source:
            filtered = [
                config
                for config in filtered
                if config.metadata.source == request.source
            ]

        # Filter by algorithm
        if request.algorithm:
            filtered = [
                config
                for config in filtered
                if config.algorithm_config.algorithm_name == request.algorithm
            ]

        # Filter by date range
        if request.created_after:
            filtered = [
                config
                for config in filtered
                if config.metadata.created_at >= request.created_after
            ]

        if request.created_before:
            filtered = [
                config
                for config in filtered
                if config.metadata.created_at <= request.created_before
            ]

        # Filter by performance
        if request.min_accuracy and request.min_accuracy > 0:
            filtered = [
                config
                for config in filtered
                if (
                    config.performance_results
                    and config.performance_results.accuracy
                    and config.performance_results.accuracy >= request.min_accuracy
                )
            ]

        return filtered

    def _sort_configurations(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationSearchRequestDTO,
    ) -> list[ExperimentConfigurationDTO]:
        """Sort configurations based on request parameters."""
        reverse = request.sort_order.lower() == "desc"

        if request.sort_by == "created_at":
            return sorted(
                configurations, key=lambda x: x.metadata.created_at, reverse=reverse
            )
        elif request.sort_by == "name":
            return sorted(configurations, key=lambda x: x.name.lower(), reverse=reverse)
        elif request.sort_by == "algorithm":
            return sorted(
                configurations,
                key=lambda x: x.algorithm_config.algorithm_name,
                reverse=reverse,
            )
        elif request.sort_by == "accuracy" and any(
            c.performance_results for c in configurations
        ):
            return sorted(
                configurations,
                key=lambda x: (
                    x.performance_results.accuracy
                    if x.performance_results and x.performance_results.accuracy
                    else 0
                ),
                reverse=reverse,
            )
        else:
            # Default to created_at
            return sorted(
                configurations, key=lambda x: x.metadata.created_at, reverse=reverse
            )

    async def _create_backup(self, config_id: UUID) -> None:
        """Create backup of configuration."""
        try:
            config_file = self._get_configuration_file_path(config_id)
            if config_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = (
                    self.storage_path / "backups" / f"{config_id}_{timestamp}.json"
                )
                shutil.copy2(config_file, backup_file)
                logger.debug(f"Created backup for configuration {config_id}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {config_id}: {e}")

    async def _save_compressed(self, file_path: Path, data: dict[str, Any]) -> None:
        """Save data with compression."""
        import gzip

        json_str = json.dumps(data, indent=2, default=str)
        with gzip.open(f"{file_path}.gz", "wt", encoding="utf-8") as f:
            f.write(json_str)

    async def _load_compressed(self, file_path: Path) -> dict[str, Any]:
        """Load compressed data."""
        import gzip

        compressed_file = Path(f"{file_path}.gz")
        if compressed_file.exists():
            with gzip.open(compressed_file, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Fallback to uncompressed
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)

    async def _update_indexes(self, configuration: ExperimentConfigurationDTO) -> None:
        """Update search indexes for configuration."""
        # Simplified index update - in production would use proper indexing
        index_file = self.storage_path / "indexes" / "configurations.json"

        try:
            if index_file.exists():
                with open(index_file, encoding="utf-8") as f:
                    index_data = json.load(f)
            else:
                index_data = {"configurations": {}}

            # Add configuration to index
            index_data["configurations"][str(configuration.id)] = {
                "name": configuration.name,
                "algorithm": configuration.algorithm_config.algorithm_name,
                "source": configuration.metadata.source,
                "tags": configuration.metadata.tags,
                "created_at": configuration.metadata.created_at.isoformat(),
                "status": configuration.status,
            }

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to update indexes: {e}")

    async def _remove_from_indexes(self, config_id: UUID) -> None:
        """Remove configuration from indexes."""
        index_file = self.storage_path / "indexes" / "configurations.json"

        try:
            if index_file.exists():
                with open(index_file, encoding="utf-8") as f:
                    index_data = json.load(f)

                index_data["configurations"].pop(str(config_id), None)

                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to remove from indexes: {e}")

    async def _update_access_timestamp(self, config_id: UUID) -> None:
        """Update last accessed timestamp for configuration."""
        try:
            access_file = self.storage_path / "indexes" / "access_log.json"

            if access_file.exists():
                with open(access_file, encoding="utf-8") as f:
                    access_data = json.load(f)
            else:
                access_data = {}

            access_data[str(config_id)] = datetime.now().isoformat()

            with open(access_file, "w", encoding="utf-8") as f:
                json.dump(access_data, f, indent=2, default=str)

        except Exception as e:
            logger.debug(f"Failed to update access timestamp: {e}")

    def _load_repository_metadata(self) -> dict[str, Any]:
        """Load repository metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load repository metadata: {e}")

        # Return default metadata
        return {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "total_configurations": 0,
            "last_modified": datetime.now().isoformat(),
        }

    async def _update_repository_metadata(
        self, configuration: ExperimentConfigurationDTO
    ) -> None:
        """Update repository metadata after configuration change."""
        self.metadata["last_modified"] = datetime.now().isoformat()
        self.metadata["total_configurations"] = len(
            list((self.storage_path / "configurations").glob("*.json"))
        )
        await self._save_repository_metadata()

    async def _save_repository_metadata(self) -> None:
        """Save repository metadata."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save repository metadata: {e}")
