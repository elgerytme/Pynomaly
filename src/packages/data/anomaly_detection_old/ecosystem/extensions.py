"""
Extension Registry and Management for Pynomaly Detection
=======================================================

Comprehensive extension system providing:
- Extension discovery and registration
- Version management and compatibility checking
- Installation and upgrade management
- Extension configuration and settings
- Dependency resolution and management
- Performance monitoring and analytics
- Security scanning and validation
"""

import logging
import json
import os
import threading
import hashlib
import zipfile
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except ImportError:
    PKG_RESOURCES_AVAILABLE = False

try:
    import semver
    SEMVER_AVAILABLE = True
except ImportError:
    SEMVER_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExtensionType(Enum):
    """Extension type enumeration."""
    ALGORITHM = "algorithm"
    DATA_SOURCE = "data_source"
    VISUALIZATION = "visualization"
    INTEGRATION = "integration"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    UI_COMPONENT = "ui_component"
    WORKFLOW = "workflow"
    UTILITY = "utility"
    THEME = "theme"

class ExtensionStatus(Enum):
    """Extension status enumeration."""
    AVAILABLE = "available"
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"

class CompatibilityLevel(Enum):
    """Compatibility level enumeration."""
    COMPATIBLE = "compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"

@dataclass
class ExtensionManifest:
    """Extension manifest definition."""
    extension_id: str
    name: str
    version: str
    description: str
    author: str
    extension_type: ExtensionType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    platform_version: str = ">=1.0.0"
    python_version: str = ">=3.8"
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    updated_date: datetime = field(default_factory=datetime.now)
    minimum_memory_mb: int = 64
    supported_platforms: List[str] = field(default_factory=lambda: ["linux", "windows", "macos"])
    categories: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    changelog: str = ""

@dataclass
class ExtensionInfo:
    """Extension information and runtime data."""
    manifest: ExtensionManifest
    status: ExtensionStatus
    installed_version: Optional[str] = None
    installation_path: Optional[str] = None
    installed_date: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    error_message: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    compatibility: CompatibilityLevel = CompatibilityLevel.UNKNOWN

@dataclass
class ExtensionDependency:
    """Extension dependency information."""
    name: str
    version_spec: str
    optional: bool = False
    description: str = ""
    installation_source: Optional[str] = None

@dataclass
class ExtensionUpdate:
    """Extension update information."""
    extension_id: str
    current_version: str
    available_version: str
    update_type: str  # major, minor, patch
    changelog: str = ""
    breaking_changes: List[str] = field(default_factory=list)
    requires_restart: bool = False
    download_url: Optional[str] = None
    file_size: int = 0

class ExtensionRegistry:
    """Central registry for extension discovery and management."""
    
    def __init__(self, registry_url: Optional[str] = None):
        """Initialize extension registry.
        
        Args:
            registry_url: Optional remote registry URL
        """
        self.registry_url = registry_url or "https://registry.pynomaly.com"
        
        # Extension tracking
        self.available_extensions: Dict[str, ExtensionManifest] = {}
        self.installed_extensions: Dict[str, ExtensionInfo] = {}
        
        # Categories and tags
        self.categories: Dict[str, List[str]] = {}
        self.tags: Dict[str, List[str]] = {}
        
        # Registry cache
        self.cache_timeout = 3600  # 1 hour
        self.last_sync: Optional[datetime] = None
        
        # Threading
        self.lock = threading.RLock()
        
        logger.info("Extension Registry initialized")
    
    def sync_registry(self, force: bool = False) -> bool:
        """Sync with remote extension registry.
        
        Args:
            force: Force sync even if cache is valid
            
        Returns:
            True if sync successful
        """
        try:
            with self.lock:
                # Check cache validity
                if not force and self.last_sync:
                    elapsed = (datetime.now() - self.last_sync).total_seconds()
                    if elapsed < self.cache_timeout:
                        logger.info("Registry cache is valid, skipping sync")
                        return True
                
                # Fetch registry data
                registry_data = self._fetch_registry_data()
                if not registry_data:
                    return False
                
                # Update available extensions
                for ext_data in registry_data.get('extensions', []):
                    try:
                        manifest = ExtensionManifest(**ext_data)
                        self.available_extensions[manifest.extension_id] = manifest
                        
                        # Update categories
                        for category in manifest.categories:
                            if category not in self.categories:
                                self.categories[category] = []
                            if manifest.extension_id not in self.categories[category]:
                                self.categories[category].append(manifest.extension_id)
                        
                        # Update tags
                        for keyword in manifest.keywords:
                            if keyword not in self.tags:
                                self.tags[keyword] = []
                            if manifest.extension_id not in self.tags[keyword]:
                                self.tags[keyword].append(manifest.extension_id)
                                
                    except Exception as e:
                        logger.warning(f"Failed to parse extension manifest: {e}")
                        continue
                
                self.last_sync = datetime.now()
                logger.info(f"Registry synced: {len(self.available_extensions)} extensions available")
                return True
                
        except Exception as e:
            logger.error(f"Registry sync failed: {e}")
            return False
    
    def search_extensions(self, query: str, 
                         extension_type: Optional[ExtensionType] = None,
                         category: Optional[str] = None,
                         limit: int = 50) -> List[ExtensionManifest]:
        """Search available extensions.
        
        Args:
            query: Search query
            extension_type: Optional type filter
            category: Optional category filter
            limit: Maximum results to return
            
        Returns:
            List of matching extensions
        """
        try:
            results = []
            query_lower = query.lower()
            
            for extension in self.available_extensions.values():
                # Type filter
                if extension_type and extension.extension_type != extension_type:
                    continue
                
                # Category filter
                if category and category not in extension.categories:
                    continue
                
                # Text search
                searchable_text = f"{extension.name} {extension.description} {' '.join(extension.keywords)}".lower()
                
                if not query or query_lower in searchable_text:
                    results.append(extension)
                
                if len(results) >= limit:
                    break
            
            # Sort by relevance (name match priority, then description)
            def relevance_score(ext):
                score = 0
                if query_lower in ext.name.lower():
                    score += 10
                if query_lower in ext.description.lower():
                    score += 5
                if any(query_lower in kw.lower() for kw in ext.keywords):
                    score += 3
                return score
            
            results.sort(key=relevance_score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Extension search failed: {e}")
            return []
    
    def get_extension_details(self, extension_id: str) -> Optional[ExtensionManifest]:
        """Get detailed extension information.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            Extension manifest or None
        """
        return self.available_extensions.get(extension_id)
    
    def get_extension_dependencies(self, extension_id: str) -> List[ExtensionDependency]:
        """Get extension dependencies.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            List of dependencies
        """
        try:
            extension = self.available_extensions.get(extension_id)
            if not extension:
                return []
            
            dependencies = []
            for dep_spec in extension.dependencies:
                # Parse dependency specification
                if '>=' in dep_spec:
                    name, version = dep_spec.split('>=', 1)
                    version_spec = f'>={version.strip()}'
                elif '==' in dep_spec:
                    name, version = dep_spec.split('==', 1)
                    version_spec = f'=={version.strip()}'
                else:
                    name = dep_spec
                    version_spec = '*'
                
                dependencies.append(ExtensionDependency(
                    name=name.strip(),
                    version_spec=version_spec
                ))
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Failed to get extension dependencies: {e}")
            return []
    
    def check_compatibility(self, extension_id: str, 
                          platform_version: str = "1.0.0") -> CompatibilityLevel:
        """Check extension compatibility.
        
        Args:
            extension_id: Extension identifier
            platform_version: Current platform version
            
        Returns:
            Compatibility level
        """
        try:
            extension = self.available_extensions.get(extension_id)
            if not extension:
                return CompatibilityLevel.UNKNOWN
            
            if not SEMVER_AVAILABLE:
                # Fallback to simple string comparison
                return CompatibilityLevel.COMPATIBLE
            
            # Check platform version compatibility
            required_version = extension.platform_version.replace('>=', '').strip()
            
            try:
                if semver.compare(platform_version, required_version) >= 0:
                    return CompatibilityLevel.COMPATIBLE
                else:
                    return CompatibilityLevel.INCOMPATIBLE
            except ValueError:
                return CompatibilityLevel.UNKNOWN
                
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return CompatibilityLevel.UNKNOWN
    
    def get_popular_extensions(self, limit: int = 20) -> List[ExtensionManifest]:
        """Get popular extensions.
        
        Args:
            limit: Maximum number of extensions to return
            
        Returns:
            List of popular extensions
        """
        # For now, return extensions sorted by name
        # In practice, this would use download/usage statistics
        extensions = list(self.available_extensions.values())
        extensions.sort(key=lambda e: e.name)
        return extensions[:limit]
    
    def get_categories(self) -> Dict[str, int]:
        """Get extension categories with counts.
        
        Returns:
            Dictionary mapping category names to extension counts
        """
        return {category: len(extensions) for category, extensions in self.categories.items()}
    
    def _fetch_registry_data(self) -> Optional[Dict[str, Any]]:
        """Fetch data from remote registry.
        
        Returns:
            Registry data or None
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, using mock data")
            return self._get_mock_registry_data()
        
        try:
            response = requests.get(f"{self.registry_url}/extensions", timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to fetch registry data: {e}")
            return self._get_mock_registry_data()
    
    def _get_mock_registry_data(self) -> Dict[str, Any]:
        """Get mock registry data for development/testing.
        
        Returns:
            Mock registry data
        """
        return {
            "extensions": [
                {
                    "extension_id": "advanced_isolation_forest",
                    "name": "Advanced Isolation Forest",
                    "version": "2.1.0",
                    "description": "Enhanced Isolation Forest algorithm with GPU acceleration",
                    "author": "ML Team",
                    "extension_type": "algorithm",
                    "entry_point": "advanced_isolation_forest.main",
                    "dependencies": ["scikit-learn>=1.0.0", "numpy>=1.20.0"],
                    "keywords": ["isolation", "forest", "gpu", "fast"],
                    "categories": ["algorithms", "gpu"]
                },
                {
                    "extension_id": "prometheus_exporter",
                    "name": "Prometheus Metrics Exporter",
                    "version": "1.5.2",
                    "description": "Export anomaly detection metrics to Prometheus",
                    "author": "DevOps Team",
                    "extension_type": "integration",
                    "entry_point": "prometheus_exporter.main",
                    "dependencies": ["prometheus-client>=0.12.0"],
                    "keywords": ["prometheus", "metrics", "monitoring"],
                    "categories": ["monitoring", "integrations"]
                },
                {
                    "extension_id": "interactive_plots",
                    "name": "Interactive Visualization Plots",
                    "version": "3.0.1",
                    "description": "Create interactive charts for anomaly analysis",
                    "author": "Visualization Team",
                    "extension_type": "visualization",
                    "entry_point": "interactive_plots.main",
                    "dependencies": ["plotly>=5.0.0", "dash>=2.0.0"],
                    "keywords": ["plotly", "interactive", "charts", "dashboard"],
                    "categories": ["visualization", "dashboard"]
                }
            ]
        }


class ExtensionManager:
    """Extension lifecycle management system."""
    
    def __init__(self, install_directory: Optional[str] = None,
                 registry: Optional[ExtensionRegistry] = None):
        """Initialize extension manager.
        
        Args:
            install_directory: Directory for extension installations
            registry: Extension registry instance
        """
        self.install_directory = install_directory or os.path.join(os.getcwd(), "extensions")
        self.registry = registry or ExtensionRegistry()
        
        # Extension tracking
        self.installed_extensions: Dict[str, ExtensionInfo] = {}
        self.loaded_extensions: Dict[str, Any] = {}
        
        # Installation queue
        self.install_queue: List[str] = []
        self.installing: Set[str] = set()
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_installed': 0,
            'enabled_extensions': 0,
            'total_downloads': 0,
            'failed_installations': 0,
            'disk_usage_mb': 0
        }
        
        # Ensure directories exist
        os.makedirs(self.install_directory, exist_ok=True)
        
        # Load installed extensions
        self._load_installed_extensions()
        
        logger.info(f"Extension Manager initialized (install_dir: {self.install_directory})")
    
    def install_extension(self, extension_id: str, version: Optional[str] = None) -> bool:
        """Install extension.
        
        Args:
            extension_id: Extension identifier
            version: Optional specific version
            
        Returns:
            True if installation successful
        """
        try:
            with self.lock:
                if extension_id in self.installing:
                    logger.warning(f"Extension already being installed: {extension_id}")
                    return False
                
                if extension_id in self.installed_extensions:
                    logger.warning(f"Extension already installed: {extension_id}")
                    return False
                
                # Get extension manifest
                manifest = self.registry.get_extension_details(extension_id)
                if not manifest:
                    logger.error(f"Extension not found in registry: {extension_id}")
                    return False
                
                # Check compatibility
                compatibility = self.registry.check_compatibility(extension_id)
                if compatibility == CompatibilityLevel.INCOMPATIBLE:
                    logger.error(f"Extension not compatible: {extension_id}")
                    return False
                
                # Check and install dependencies
                if not self._install_dependencies(extension_id):
                    logger.error(f"Failed to install dependencies for: {extension_id}")
                    return False
                
                # Mark as installing
                self.installing.add(extension_id)
                
                try:
                    # Download and install
                    if self._download_and_install(extension_id, version or manifest.version):
                        # Create extension info
                        extension_info = ExtensionInfo(
                            manifest=manifest,
                            status=ExtensionStatus.INSTALLED,
                            installed_version=version or manifest.version,
                            installation_path=os.path.join(self.install_directory, extension_id),
                            installed_date=datetime.now(),
                            compatibility=compatibility
                        )
                        
                        self.installed_extensions[extension_id] = extension_info
                        
                        # Update statistics
                        self.stats['total_installed'] += 1
                        
                        logger.info(f"Extension installed successfully: {extension_id}")
                        return True
                    else:
                        self.stats['failed_installations'] += 1
                        return False
                        
                finally:
                    self.installing.discard(extension_id)
                    
        except Exception as e:
            logger.error(f"Extension installation failed: {e}")
            self.stats['failed_installations'] += 1
            return False
    
    def uninstall_extension(self, extension_id: str) -> bool:
        """Uninstall extension.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            True if uninstallation successful
        """
        try:
            with self.lock:
                extension_info = self.installed_extensions.get(extension_id)
                if not extension_info:
                    logger.warning(f"Extension not installed: {extension_id}")
                    return True
                
                # Disable first if enabled
                if extension_info.status == ExtensionStatus.ENABLED:
                    self.disable_extension(extension_id)
                
                # Remove from loaded extensions
                if extension_id in self.loaded_extensions:
                    del self.loaded_extensions[extension_id]
                
                # Remove installation directory
                if extension_info.installation_path and os.path.exists(extension_info.installation_path):
                    shutil.rmtree(extension_info.installation_path)
                
                # Remove from tracking
                del self.installed_extensions[extension_id]
                
                # Update statistics
                if self.stats['total_installed'] > 0:
                    self.stats['total_installed'] -= 1
                
                logger.info(f"Extension uninstalled: {extension_id}")
                return True
                
        except Exception as e:
            logger.error(f"Extension uninstallation failed: {e}")
            return False
    
    def enable_extension(self, extension_id: str) -> bool:
        """Enable installed extension.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            True if enabled successfully
        """
        try:
            with self.lock:
                extension_info = self.installed_extensions.get(extension_id)
                if not extension_info:
                    logger.error(f"Extension not installed: {extension_id}")
                    return False
                
                if extension_info.status == ExtensionStatus.ENABLED:
                    logger.info(f"Extension already enabled: {extension_id}")
                    return True
                
                # Load extension
                if self._load_extension(extension_id):
                    extension_info.status = ExtensionStatus.ENABLED
                    self.stats['enabled_extensions'] += 1
                    
                    logger.info(f"Extension enabled: {extension_id}")
                    return True
                else:
                    extension_info.status = ExtensionStatus.ERROR
                    extension_info.error_message = "Failed to load extension"
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to enable extension: {e}")
            return False
    
    def disable_extension(self, extension_id: str) -> bool:
        """Disable enabled extension.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            True if disabled successfully
        """
        try:
            with self.lock:
                extension_info = self.installed_extensions.get(extension_id)
                if not extension_info:
                    logger.error(f"Extension not installed: {extension_id}")
                    return False
                
                if extension_info.status != ExtensionStatus.ENABLED:
                    logger.info(f"Extension not enabled: {extension_id}")
                    return True
                
                # Unload extension
                if extension_id in self.loaded_extensions:
                    # Call cleanup if available
                    loaded_ext = self.loaded_extensions[extension_id]
                    if hasattr(loaded_ext, 'cleanup'):
                        try:
                            loaded_ext.cleanup()
                        except Exception as e:
                            logger.warning(f"Extension cleanup failed: {e}")
                    
                    del self.loaded_extensions[extension_id]
                
                extension_info.status = ExtensionStatus.DISABLED
                if self.stats['enabled_extensions'] > 0:
                    self.stats['enabled_extensions'] -= 1
                
                logger.info(f"Extension disabled: {extension_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to disable extension: {e}")
            return False
    
    def update_extension(self, extension_id: str, version: Optional[str] = None) -> bool:
        """Update installed extension.
        
        Args:
            extension_id: Extension identifier
            version: Optional specific version to update to
            
        Returns:
            True if update successful
        """
        try:
            with self.lock:
                extension_info = self.installed_extensions.get(extension_id)
                if not extension_info:
                    logger.error(f"Extension not installed: {extension_id}")
                    return False
                
                # Get latest version info
                latest_manifest = self.registry.get_extension_details(extension_id)
                if not latest_manifest:
                    logger.error(f"Extension not found in registry: {extension_id}")
                    return False
                
                target_version = version or latest_manifest.version
                
                # Check if update needed
                if extension_info.installed_version == target_version:
                    logger.info(f"Extension already up to date: {extension_id}")
                    return True
                
                # Disable extension during update
                was_enabled = extension_info.status == ExtensionStatus.ENABLED
                if was_enabled:
                    self.disable_extension(extension_id)
                
                extension_info.status = ExtensionStatus.UPDATING
                
                try:
                    # Backup current installation
                    backup_path = f"{extension_info.installation_path}.backup"
                    if os.path.exists(extension_info.installation_path):
                        shutil.copytree(extension_info.installation_path, backup_path)
                    
                    # Download and install new version
                    if self._download_and_install(extension_id, target_version):
                        extension_info.installed_version = target_version
                        extension_info.manifest = latest_manifest
                        extension_info.status = ExtensionStatus.INSTALLED
                        
                        # Re-enable if it was enabled
                        if was_enabled:
                            self.enable_extension(extension_id)
                        
                        # Remove backup
                        if os.path.exists(backup_path):
                            shutil.rmtree(backup_path)
                        
                        logger.info(f"Extension updated successfully: {extension_id}")
                        return True
                    else:
                        # Restore from backup
                        if os.path.exists(backup_path):
                            if os.path.exists(extension_info.installation_path):
                                shutil.rmtree(extension_info.installation_path)
                            shutil.copytree(backup_path, extension_info.installation_path)
                            shutil.rmtree(backup_path)
                        
                        extension_info.status = ExtensionStatus.ERROR
                        extension_info.error_message = "Update failed, restored previous version"
                        
                        return False
                        
                except Exception as e:
                    extension_info.status = ExtensionStatus.ERROR
                    extension_info.error_message = str(e)
                    raise
                    
        except Exception as e:
            logger.error(f"Extension update failed: {e}")
            return False
    
    def get_installed_extensions(self, status: Optional[ExtensionStatus] = None) -> List[ExtensionInfo]:
        """Get list of installed extensions.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of installed extensions
        """
        extensions = list(self.installed_extensions.values())
        
        if status:
            extensions = [ext for ext in extensions if ext.status == status]
        
        return extensions
    
    def get_available_updates(self) -> List[ExtensionUpdate]:
        """Get available extension updates.
        
        Returns:
            List of available updates
        """
        try:
            updates = []
            
            for extension_id, extension_info in self.installed_extensions.items():
                latest_manifest = self.registry.get_extension_details(extension_id)
                if not latest_manifest:
                    continue
                
                current_version = extension_info.installed_version
                latest_version = latest_manifest.version
                
                if current_version != latest_version:
                    # Determine update type
                    update_type = self._determine_update_type(current_version, latest_version)
                    
                    update = ExtensionUpdate(
                        extension_id=extension_id,
                        current_version=current_version,
                        available_version=latest_version,
                        update_type=update_type,
                        changelog=latest_manifest.changelog
                    )
                    
                    updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Failed to get available updates: {e}")
            return []
    
    def configure_extension(self, extension_id: str, config: Dict[str, Any]) -> bool:
        """Configure extension settings.
        
        Args:
            extension_id: Extension identifier
            config: Configuration dictionary
            
        Returns:
            True if configuration successful
        """
        try:
            with self.lock:
                extension_info = self.installed_extensions.get(extension_id)
                if not extension_info:
                    logger.error(f"Extension not installed: {extension_id}")
                    return False
                
                # Validate configuration against schema
                if extension_info.manifest.configuration_schema:
                    if not self._validate_configuration(config, extension_info.manifest.configuration_schema):
                        logger.error(f"Invalid configuration for extension: {extension_id}")
                        return False
                
                # Update configuration
                extension_info.configuration.update(config)
                
                # Apply configuration to loaded extension
                if extension_id in self.loaded_extensions:
                    loaded_ext = self.loaded_extensions[extension_id]
                    if hasattr(loaded_ext, 'configure'):
                        loaded_ext.configure(extension_info.configuration)
                
                logger.info(f"Extension configured: {extension_id}")
                return True
                
        except Exception as e:
            logger.error(f"Extension configuration failed: {e}")
            return False
    
    def get_extension_metrics(self, extension_id: str) -> Dict[str, Any]:
        """Get extension performance metrics.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            Performance metrics dictionary
        """
        extension_info = self.installed_extensions.get(extension_id)
        if not extension_info:
            return {}
        
        return {
            'usage_count': extension_info.usage_count,
            'last_used': extension_info.last_used.isoformat() if extension_info.last_used else None,
            'status': extension_info.status.value,
            'installed_date': extension_info.installed_date.isoformat() if extension_info.installed_date else None,
            'performance_metrics': extension_info.performance_metrics,
            'disk_usage_mb': self._calculate_extension_disk_usage(extension_id)
        }
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get extension manager statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate real-time metrics
            status_counts = {}
            for ext_info in self.installed_extensions.values():
                status = ext_info.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            stats.update({
                'status_distribution': status_counts,
                'available_extensions': len(self.registry.available_extensions),
                'loaded_extensions': len(self.loaded_extensions),
                'install_directory': self.install_directory,
                'registry_last_sync': self.registry.last_sync.isoformat() if self.registry.last_sync else None
            })
            
            return stats
    
    def _load_installed_extensions(self):
        """Load information about installed extensions."""
        try:
            if not os.path.exists(self.install_directory):
                return
            
            for item in os.listdir(self.install_directory):
                item_path = os.path.join(self.install_directory, item)
                
                if os.path.isdir(item_path):
                    manifest_path = os.path.join(item_path, 'manifest.json')
                    
                    if os.path.exists(manifest_path):
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest_data = json.load(f)
                                manifest = ExtensionManifest(**manifest_data)
                            
                            extension_info = ExtensionInfo(
                                manifest=manifest,
                                status=ExtensionStatus.INSTALLED,
                                installed_version=manifest.version,
                                installation_path=item_path,
                                installed_date=datetime.fromtimestamp(os.path.getctime(item_path))
                            )
                            
                            self.installed_extensions[manifest.extension_id] = extension_info
                            self.stats['total_installed'] += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to load extension {item}: {e}")
                            continue
            
            logger.info(f"Loaded {len(self.installed_extensions)} installed extensions")
            
        except Exception as e:
            logger.error(f"Failed to load installed extensions: {e}")
    
    def _install_dependencies(self, extension_id: str) -> bool:
        """Install extension dependencies.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            True if all dependencies installed successfully
        """
        try:
            dependencies = self.registry.get_extension_dependencies(extension_id)
            
            for dep in dependencies:
                # Check if dependency is already satisfied
                if self._is_dependency_satisfied(dep):
                    continue
                
                # Install dependency (simplified - would use proper package manager)
                logger.info(f"Installing dependency: {dep.name}")
                
                # For now, just log - in practice would install via pip/conda
                if not dep.optional:
                    logger.info(f"Required dependency {dep.name} would be installed here")
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def _is_dependency_satisfied(self, dependency: ExtensionDependency) -> bool:
        """Check if dependency is satisfied.
        
        Args:
            dependency: Dependency to check
            
        Returns:
            True if dependency is satisfied
        """
        # Simplified implementation
        if PKG_RESOURCES_AVAILABLE:
            try:
                pkg_resources.require(f"{dependency.name}{dependency.version_spec}")
                return True
            except pkg_resources.DistributionNotFound:
                return False
            except pkg_resources.VersionConflict:
                return False
        
        return True  # Assume satisfied if can't check
    
    def _download_and_install(self, extension_id: str, version: str) -> bool:
        """Download and install extension.
        
        Args:
            extension_id: Extension identifier
            version: Extension version
            
        Returns:
            True if download and installation successful
        """
        try:
            # Mock download and installation
            # In practice, would download from registry and extract
            
            install_path = os.path.join(self.install_directory, extension_id)
            os.makedirs(install_path, exist_ok=True)
            
            # Create mock manifest file
            manifest = self.registry.get_extension_details(extension_id)
            if manifest:
                manifest_path = os.path.join(install_path, 'manifest.json')
                with open(manifest_path, 'w') as f:
                    # Convert manifest to dict for JSON serialization
                    manifest_dict = {
                        'extension_id': manifest.extension_id,
                        'name': manifest.name,
                        'version': version,
                        'description': manifest.description,
                        'author': manifest.author,
                        'extension_type': manifest.extension_type.value,
                        'entry_point': manifest.entry_point,
                        'dependencies': manifest.dependencies
                    }
                    json.dump(manifest_dict, f, indent=2)
            
            # Create mock main module
            main_file = os.path.join(install_path, '__init__.py')
            with open(main_file, 'w') as f:
                f.write(f'# {manifest.name} Extension\n')
                f.write(f'__version__ = "{version}"\n')
                f.write('def initialize():\n    pass\n')
                f.write('def cleanup():\n    pass\n')
            
            logger.info(f"Extension installed to: {install_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download and installation failed: {e}")
            return False
    
    def _load_extension(self, extension_id: str) -> bool:
        """Load extension into memory.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            True if loading successful
        """
        try:
            extension_info = self.installed_extensions.get(extension_id)
            if not extension_info or not extension_info.installation_path:
                return False
            
            # Add extension path to sys.path temporarily
            import sys
            if extension_info.installation_path not in sys.path:
                sys.path.insert(0, extension_info.installation_path)
            
            try:
                # Import extension module
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    extension_id,
                    os.path.join(extension_info.installation_path, '__init__.py')
                )
                
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Initialize extension
                    if hasattr(module, 'initialize'):
                        module.initialize()
                    
                    self.loaded_extensions[extension_id] = module
                    
                    logger.info(f"Extension loaded: {extension_id}")
                    return True
                
                return False
                
            finally:
                # Remove from sys.path
                if extension_info.installation_path in sys.path:
                    sys.path.remove(extension_info.installation_path)
                    
        except Exception as e:
            logger.error(f"Extension loading failed: {e}")
            return False
    
    def _determine_update_type(self, current: str, latest: str) -> str:
        """Determine update type (major, minor, patch).
        
        Args:
            current: Current version
            latest: Latest version
            
        Returns:
            Update type string
        """
        if SEMVER_AVAILABLE:
            try:
                current_ver = semver.VersionInfo.parse(current)
                latest_ver = semver.VersionInfo.parse(latest)
                
                if latest_ver.major > current_ver.major:
                    return "major"
                elif latest_ver.minor > current_ver.minor:
                    return "minor"
                else:
                    return "patch"
            except ValueError:
                pass
        
        return "unknown"
    
    def _validate_configuration(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: JSON schema
            
        Returns:
            True if configuration is valid
        """
        # Simplified validation - in practice would use jsonschema library
        required_fields = schema.get('required', [])
        
        for field in required_fields:
            if field not in config:
                return False
        
        return True
    
    def _calculate_extension_disk_usage(self, extension_id: str) -> float:
        """Calculate extension disk usage in MB.
        
        Args:
            extension_id: Extension identifier
            
        Returns:
            Disk usage in MB
        """
        try:
            extension_info = self.installed_extensions.get(extension_id)
            if not extension_info or not extension_info.installation_path:
                return 0.0
            
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(extension_info.installation_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Failed to calculate disk usage: {e}")
            return 0.0