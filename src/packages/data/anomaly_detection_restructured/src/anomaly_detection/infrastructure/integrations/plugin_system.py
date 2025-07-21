"""
Plugin Architecture and Marketplace for Pynomaly Detection
==========================================================

Comprehensive plugin system providing:
- Extensible plugin architecture with discovery and loading
- Plugin marketplace with ratings and reviews
- Secure plugin execution environment
- Plugin versioning and dependency management
- Developer tools for plugin creation
"""

import logging
import json
import os
import sys
import importlib
import inspect
import threading
import hashlib
import zipfile
import tempfile
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid

try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except ImportError:
    PKG_RESOURCES_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Plugin type enumeration."""
    DETECTOR = "detector"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    VISUALIZER = "visualizer"
    EXPORTER = "exporter"
    INTEGRATION = "integration"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    NOTIFIER = "notifier"
    CUSTOM = "custom"

class PluginStatus(Enum):
    """Plugin status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"
    UNLOADING = "unloading"
    DISABLED = "disabled"

class MarketplaceStatus(Enum):
    """Marketplace plugin status."""
    PUBLISHED = "published"
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"

@dataclass
class PluginManifest:
    """Plugin manifest definition."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    pynomaly_version: str = ">=1.0.0"
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    updated_date: datetime = field(default_factory=datetime.now)

@dataclass
class PluginInfo:
    """Plugin information."""
    manifest: PluginManifest
    status: PluginStatus
    loaded_instance: Optional[Any] = None
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    file_path: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class MarketplacePlugin:
    """Marketplace plugin information."""
    plugin_id: str
    manifest: PluginManifest
    status: MarketplaceStatus
    download_count: int = 0
    rating: float = 0.0
    review_count: int = 0
    file_url: str = ""
    file_size: int = 0
    checksum: str = ""
    screenshots: List[str] = field(default_factory=list)
    changelog: str = ""
    support_url: Optional[str] = None
    published_date: Optional[datetime] = None

class PluginBase(ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup plugin resources.
        
        Returns:
            True if cleanup successful
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information.
        
        Returns:
            Plugin information dictionary
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        return True
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._is_initialized

class DetectorPlugin(PluginBase):
    """Base class for detector plugins."""
    
    @abstractmethod
    def detect(self, data: Any) -> Dict[str, Any]:
        """Perform anomaly detection.
        
        Args:
            data: Input data for detection
            
        Returns:
            Detection results
        """
        pass

class PreprocessorPlugin(PluginBase):
    """Base class for preprocessor plugins."""
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data.
        
        Args:
            data: Input data to preprocess
            
        Returns:
            Preprocessed data
        """
        pass

class VisualizerPlugin(PluginBase):
    """Base class for visualizer plugins."""
    
    @abstractmethod
    def visualize(self, data: Any, config: Dict[str, Any] = None) -> str:
        """Generate visualization.
        
        Args:
            data: Data to visualize
            config: Visualization configuration
            
        Returns:
            Visualization (HTML, SVG, etc.)
        """
        pass

class PluginManager:
    """Comprehensive plugin management system."""
    
    def __init__(self, plugin_directory: Optional[str] = None):
        """Initialize plugin manager.
        
        Args:
            plugin_directory: Directory containing plugins
        """
        self.plugin_directory = plugin_directory or os.path.join(os.getcwd(), "plugins")
        
        # Plugin storage
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_instances: Dict[str, Any] = {}
        
        # Plugin discovery
        self.discovered_plugins: Dict[str, str] = {}
        
        # Security and sandboxing
        self.allowed_permissions = {
            "file_read", "file_write", "network_access", 
            "database_access", "system_info"
        }
        self.sandbox_enabled = True
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_plugins': 0,
            'active_plugins': 0,
            'failed_loads': 0,
            'total_usage': 0
        }
        
        # Ensure plugin directory exists
        os.makedirs(self.plugin_directory, exist_ok=True)
        
        logger.info(f"Plugin Manager initialized (directory: {self.plugin_directory})")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins.
        
        Returns:
            List of discovered plugin IDs
        """
        try:
            discovered = []
            
            # Scan plugin directory
            for item in os.listdir(self.plugin_directory):
                item_path = os.path.join(self.plugin_directory, item)
                
                # Check for plugin files (.py) or packages (directories with __init__.py)
                if os.path.isfile(item_path) and item.endswith('.py'):
                    plugin_id = item[:-3]  # Remove .py extension
                    self.discovered_plugins[plugin_id] = item_path
                    discovered.append(plugin_id)
                    
                elif os.path.isdir(item_path):
                    manifest_path = os.path.join(item_path, 'manifest.json')
                    init_path = os.path.join(item_path, '__init__.py')
                    
                    if os.path.exists(manifest_path) and os.path.exists(init_path):
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest_data = json.load(f)
                                plugin_id = manifest_data.get('plugin_id', item)
                                self.discovered_plugins[plugin_id] = item_path
                                discovered.append(plugin_id)
                        except Exception as e:
                            logger.warning(f"Failed to read manifest for {item}: {e}")
            
            logger.info(f"Discovered {len(discovered)} plugins")
            return discovered
            
        except Exception as e:
            logger.error(f"Plugin discovery failed: {e}")
            return []
    
    def load_plugin(self, plugin_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin.
        
        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration
            
        Returns:
            True if plugin loaded successfully
        """
        try:
            with self.lock:
                if plugin_id in self.plugins:
                    logger.warning(f"Plugin already loaded: {plugin_id}")
                    return True
                
                # Check if plugin was discovered
                if plugin_id not in self.discovered_plugins:
                    logger.error(f"Plugin not found: {plugin_id}")
                    return False
                
                plugin_path = self.discovered_plugins[plugin_id]
                
                # Load plugin manifest
                manifest = self._load_manifest(plugin_path, plugin_id)
                if not manifest:
                    return False
                
                # Check dependencies
                if not self._check_dependencies(manifest):
                    logger.error(f"Plugin dependencies not satisfied: {plugin_id}")
                    return False
                
                # Check permissions
                if not self._check_permissions(manifest):
                    logger.error(f"Plugin permissions not allowed: {plugin_id}")
                    return False
                
                # Load plugin code
                plugin_instance = self._load_plugin_code(plugin_path, manifest, config)
                if not plugin_instance:
                    return False
                
                # Initialize plugin
                if not plugin_instance.initialize():
                    logger.error(f"Plugin initialization failed: {plugin_id}")
                    return False
                
                # Store plugin info
                plugin_info = PluginInfo(
                    manifest=manifest,
                    status=PluginStatus.ACTIVE,
                    loaded_instance=plugin_instance,
                    load_time=datetime.now(),
                    file_path=plugin_path,
                    checksum=self._calculate_checksum(plugin_path)
                )
                
                self.plugins[plugin_id] = plugin_info
                self.plugin_instances[plugin_id] = plugin_instance
                
                # Update statistics
                self.stats['total_plugins'] += 1
                self.stats['active_plugins'] += 1
                
                logger.info(f"Plugin loaded successfully: {plugin_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            
            # Update plugin status if partially loaded
            if plugin_id in self.plugins:
                self.plugins[plugin_id].status = PluginStatus.ERROR
                self.plugins[plugin_id].error_message = str(e)
                self.stats['failed_loads'] += 1
            
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if plugin unloaded successfully
        """
        try:
            with self.lock:
                if plugin_id not in self.plugins:
                    logger.warning(f"Plugin not loaded: {plugin_id}")
                    return True
                
                plugin_info = self.plugins[plugin_id]
                plugin_info.status = PluginStatus.UNLOADING
                
                # Cleanup plugin
                if plugin_info.loaded_instance:
                    try:
                        plugin_info.loaded_instance.cleanup()
                    except Exception as e:
                        logger.warning(f"Plugin cleanup failed for {plugin_id}: {e}")
                
                # Remove from collections
                del self.plugins[plugin_id]
                if plugin_id in self.plugin_instances:
                    del self.plugin_instances[plugin_id]
                
                # Update statistics
                self.stats['active_plugins'] -= 1
                
                logger.info(f"Plugin unloaded: {plugin_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False
    
    def execute_plugin(self, plugin_id: str, method: str, *args, **kwargs) -> Any:
        """Execute a plugin method.
        
        Args:
            plugin_id: Plugin identifier
            method: Method name to execute
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        try:
            with self.lock:
                if plugin_id not in self.plugin_instances:
                    raise ValueError(f"Plugin not loaded: {plugin_id}")
                
                plugin_instance = self.plugin_instances[plugin_id]
                plugin_info = self.plugins[plugin_id]
                
                if plugin_info.status != PluginStatus.ACTIVE:
                    raise RuntimeError(f"Plugin not active: {plugin_id}")
                
                # Check if method exists
                if not hasattr(plugin_instance, method):
                    raise AttributeError(f"Plugin {plugin_id} does not have method: {method}")
                
                # Execute method
                result = getattr(plugin_instance, method)(*args, **kwargs)
                
                # Update usage statistics
                plugin_info.usage_count += 1
                plugin_info.last_used = datetime.now()
                self.stats['total_usage'] += 1
                
                return result
                
        except Exception as e:
            logger.error(f"Plugin execution failed for {plugin_id}.{method}: {e}")
            raise
    
    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin information.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin information or None
        """
        return self.plugins.get(plugin_id)
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None,
                    status: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """List loaded plugins.
        
        Args:
            plugin_type: Optional plugin type filter
            status: Optional status filter
            
        Returns:
            List of plugin information
        """
        plugins = list(self.plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.manifest.plugin_type == plugin_type]
        
        if status:
            plugins = [p for p in plugins if p.status == status]
        
        return plugins
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'discovered_plugins': len(self.discovered_plugins),
                'plugin_types': {},
                'plugin_statuses': {}
            })
            
            # Count by type and status
            for plugin_info in self.plugins.values():
                plugin_type = plugin_info.manifest.plugin_type.value
                status = plugin_info.status.value
                
                stats['plugin_types'][plugin_type] = stats['plugin_types'].get(plugin_type, 0) + 1
                stats['plugin_statuses'][status] = stats['plugin_statuses'].get(status, 0) + 1
            
            return stats
    
    def install_plugin_from_file(self, file_path: str, force: bool = False) -> bool:
        """Install plugin from file.
        
        Args:
            file_path: Path to plugin file (.zip or .py)
            force: Force installation even if plugin exists
            
        Returns:
            True if installation successful
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Plugin file not found: {file_path}")
                return False
            
            # Handle different file types
            if file_path.endswith('.zip'):
                return self._install_plugin_zip(file_path, force)
            elif file_path.endswith('.py'):
                return self._install_plugin_py(file_path, force)
            else:
                logger.error(f"Unsupported plugin file type: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Plugin installation failed: {e}")
            return False
    
    def _load_manifest(self, plugin_path: str, plugin_id: str) -> Optional[PluginManifest]:
        """Load plugin manifest."""
        try:
            if os.path.isdir(plugin_path):
                # Load from manifest.json
                manifest_path = os.path.join(plugin_path, 'manifest.json')
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest_data = json.load(f)
                        return PluginManifest(**manifest_data)
            
            # Generate basic manifest for .py files
            return PluginManifest(
                plugin_id=plugin_id,
                name=plugin_id.replace('_', ' ').title(),
                version="1.0.0",
                description=f"Plugin {plugin_id}",
                author="Unknown",
                plugin_type=PluginType.CUSTOM,
                entry_point=f"{plugin_id}.main"
            )
            
        except Exception as e:
            logger.error(f"Failed to load manifest for {plugin_id}: {e}")
            return None
    
    def _load_plugin_code(self, plugin_path: str, manifest: PluginManifest,
                         config: Optional[Dict[str, Any]]) -> Optional[PluginBase]:
        """Load plugin code and create instance."""
        try:
            # Add plugin path to sys.path if it's a directory
            if os.path.isdir(plugin_path):
                if plugin_path not in sys.path:
                    sys.path.insert(0, plugin_path)
                
                # Import the plugin module
                module_name = os.path.basename(plugin_path)
                spec = importlib.util.spec_from_file_location(
                    module_name, 
                    os.path.join(plugin_path, '__init__.py')
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            
            elif plugin_path.endswith('.py'):
                # Import single Python file
                module_name = os.path.splitext(os.path.basename(plugin_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            
            else:
                raise ValueError(f"Unsupported plugin path: {plugin_path}")
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module, manifest.plugin_type)
            if not plugin_class:
                raise ValueError(f"No suitable plugin class found in {plugin_path}")
            
            # Create plugin instance
            plugin_instance = plugin_class(config)
            
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin code: {e}")
            return None
    
    def _find_plugin_class(self, module: Any, plugin_type: PluginType) -> Optional[Type[PluginBase]]:
        """Find appropriate plugin class in module."""
        try:
            # Look for specific plugin base classes
            base_class_map = {
                PluginType.DETECTOR: DetectorPlugin,
                PluginType.PREPROCESSOR: PreprocessorPlugin,
                PluginType.VISUALIZER: VisualizerPlugin
            }
            
            target_base = base_class_map.get(plugin_type, PluginBase)
            
            # Find classes that inherit from the target base
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, target_base) and 
                    obj != target_base and 
                    obj.__module__ == module.__name__):
                    return obj
            
            # Fallback: look for any PluginBase subclass
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginBase) and 
                    obj != PluginBase and 
                    obj.__module__ == module.__name__):
                    return obj
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find plugin class: {e}")
            return None
    
    def _check_dependencies(self, manifest: PluginManifest) -> bool:
        """Check if plugin dependencies are satisfied."""
        try:
            if not PKG_RESOURCES_AVAILABLE:
                logger.warning("pkg_resources not available, skipping dependency check")
                return True
            
            for dependency in manifest.dependencies:
                try:
                    pkg_resources.require(dependency)
                except pkg_resources.DistributionNotFound:
                    logger.error(f"Dependency not found: {dependency}")
                    return False
                except pkg_resources.VersionConflict as e:
                    logger.error(f"Dependency version conflict: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    def _check_permissions(self, manifest: PluginManifest) -> bool:
        """Check if plugin permissions are allowed."""
        try:
            for permission in manifest.permissions:
                if permission not in self.allowed_permissions:
                    logger.error(f"Permission not allowed: {permission}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        try:
            hash_md5 = hashlib.md5()
            
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
            elif os.path.isdir(file_path):
                # Calculate checksum for all files in directory
                for root, dirs, files in os.walk(file_path):
                    for file in sorted(files):
                        file_full_path = os.path.join(root, file)
                        with open(file_full_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _install_plugin_zip(self, zip_path: str, force: bool) -> bool:
        """Install plugin from ZIP file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to temporary directory first
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(temp_dir)
                    
                    # Find manifest file
                    manifest_path = None
                    for root, dirs, files in os.walk(temp_dir):
                        if 'manifest.json' in files:
                            manifest_path = os.path.join(root, 'manifest.json')
                            break
                    
                    if not manifest_path:
                        logger.error("No manifest.json found in plugin ZIP")
                        return False
                    
                    # Load manifest
                    with open(manifest_path, 'r') as f:
                        manifest_data = json.load(f)
                        manifest = PluginManifest(**manifest_data)
                    
                    # Check if plugin already exists
                    target_path = os.path.join(self.plugin_directory, manifest.plugin_id)
                    if os.path.exists(target_path) and not force:
                        logger.error(f"Plugin already exists: {manifest.plugin_id}")
                        return False
                    
                    # Copy plugin to plugins directory
                    import shutil
                    if os.path.exists(target_path):
                        shutil.rmtree(target_path)
                    
                    plugin_source = os.path.dirname(manifest_path)
                    shutil.copytree(plugin_source, target_path)
                    
                    logger.info(f"Plugin installed: {manifest.plugin_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"ZIP plugin installation failed: {e}")
            return False
    
    def _install_plugin_py(self, py_path: str, force: bool) -> bool:
        """Install plugin from Python file."""
        try:
            filename = os.path.basename(py_path)
            plugin_id = os.path.splitext(filename)[0]
            
            target_path = os.path.join(self.plugin_directory, filename)
            
            # Check if plugin already exists
            if os.path.exists(target_path) and not force:
                logger.error(f"Plugin already exists: {plugin_id}")
                return False
            
            # Copy plugin file
            import shutil
            shutil.copy2(py_path, target_path)
            
            logger.info(f"Plugin installed: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Python plugin installation failed: {e}")
            return False


class PluginRegistry:
    """Plugin registry for managing plugin metadata and discovery."""
    
    def __init__(self, registry_url: Optional[str] = None):
        """Initialize plugin registry.
        
        Args:
            registry_url: Optional registry service URL
        """
        self.registry_url = registry_url
        self.registered_plugins: Dict[str, MarketplacePlugin] = {}
        self.plugin_categories: Dict[str, List[str]] = {}
        
        logger.info("Plugin Registry initialized")
    
    def register_plugin(self, marketplace_plugin: MarketplacePlugin) -> bool:
        """Register plugin in registry.
        
        Args:
            marketplace_plugin: Plugin to register
            
        Returns:
            True if registration successful
        """
        try:
            plugin_id = marketplace_plugin.plugin_id
            
            # Validate plugin
            if not self._validate_plugin(marketplace_plugin):
                return False
            
            # Store plugin
            self.registered_plugins[plugin_id] = marketplace_plugin
            
            # Add to categories
            plugin_type = marketplace_plugin.manifest.plugin_type.value
            if plugin_type not in self.plugin_categories:
                self.plugin_categories[plugin_type] = []
            
            if plugin_id not in self.plugin_categories[plugin_type]:
                self.plugin_categories[plugin_type].append(plugin_id)
            
            logger.info(f"Plugin registered: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Plugin registration failed: {e}")
            return False
    
    def search_plugins(self, query: str, plugin_type: Optional[PluginType] = None,
                      limit: int = 50) -> List[MarketplacePlugin]:
        """Search for plugins.
        
        Args:
            query: Search query
            plugin_type: Optional plugin type filter
            limit: Maximum results to return
            
        Returns:
            List of matching plugins
        """
        try:
            results = []
            query_lower = query.lower()
            
            for plugin in self.registered_plugins.values():
                # Type filter
                if plugin_type and plugin.manifest.plugin_type != plugin_type:
                    continue
                
                # Status filter (only published/approved)
                if plugin.status not in [MarketplaceStatus.PUBLISHED, MarketplaceStatus.APPROVED]:
                    continue
                
                # Text search
                searchable_text = f"{plugin.manifest.name} {plugin.manifest.description} {' '.join(plugin.manifest.keywords)}".lower()
                
                if query_lower in searchable_text:
                    results.append(plugin)
                
                if len(results) >= limit:
                    break
            
            # Sort by rating and download count
            results.sort(key=lambda p: (p.rating, p.download_count), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Plugin search failed: {e}")
            return []
    
    def get_plugin_details(self, plugin_id: str) -> Optional[MarketplacePlugin]:
        """Get detailed plugin information.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin details or None
        """
        return self.registered_plugins.get(plugin_id)
    
    def _validate_plugin(self, plugin: MarketplacePlugin) -> bool:
        """Validate plugin for registry."""
        try:
            # Check required fields
            if not plugin.plugin_id or not plugin.manifest.name:
                return False
            
            # Check manifest validity
            manifest = plugin.manifest
            if not manifest.entry_point or not manifest.version:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            return False


class PluginMarketplace:
    """Plugin marketplace for discovering and downloading plugins."""
    
    def __init__(self, marketplace_url: Optional[str] = None,
                 plugin_manager: Optional[PluginManager] = None):
        """Initialize plugin marketplace.
        
        Args:
            marketplace_url: Marketplace service URL
            plugin_manager: Plugin manager instance
        """
        self.marketplace_url = marketplace_url or "https://marketplace.pynomaly.com"
        self.plugin_manager = plugin_manager
        self.cache: Dict[str, Any] = {}
        self.cache_timeout = 3600  # 1 hour
        
        logger.info("Plugin Marketplace initialized")
    
    def browse_plugins(self, category: Optional[str] = None,
                      sort_by: str = "popularity",
                      limit: int = 20) -> List[Dict[str, Any]]:
        """Browse marketplace plugins.
        
        Args:
            category: Optional category filter
            sort_by: Sort criteria (popularity, rating, recent)
            limit: Maximum results to return
            
        Returns:
            List of plugin summaries
        """
        try:
            # Check cache
            cache_key = f"browse_{category}_{sort_by}_{limit}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Fetch from marketplace API
            plugins = self._fetch_marketplace_plugins(category, sort_by, limit)
            
            # Cache results
            self.cache[cache_key] = {
                'data': plugins,
                'timestamp': datetime.now()
            }
            
            return plugins
            
        except Exception as e:
            logger.error(f"Failed to browse plugins: {e}")
            return []
    
    def search_marketplace(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search marketplace for plugins.
        
        Args:
            query: Search query
            filters: Optional search filters
            
        Returns:
            List of search results
        """
        try:
            # Check cache
            cache_key = f"search_{query}_{hash(str(filters))}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Search marketplace
            results = self._search_marketplace_api(query, filters)
            
            # Cache results
            self.cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Marketplace search failed: {e}")
            return []
    
    def download_plugin(self, plugin_id: str, version: Optional[str] = None) -> bool:
        """Download and install plugin from marketplace.
        
        Args:
            plugin_id: Plugin identifier
            version: Optional specific version
            
        Returns:
            True if download and installation successful
        """
        try:
            if not self.plugin_manager:
                logger.error("No plugin manager available for installation")
                return False
            
            # Get plugin details
            plugin_info = self._get_marketplace_plugin(plugin_id, version)
            if not plugin_info:
                logger.error(f"Plugin not found in marketplace: {plugin_id}")
                return False
            
            # Download plugin file
            download_url = plugin_info.get('download_url')
            if not download_url:
                logger.error(f"No download URL for plugin: {plugin_id}")
                return False
            
            temp_file = self._download_file(download_url)
            if not temp_file:
                return False
            
            try:
                # Install plugin
                success = self.plugin_manager.install_plugin_from_file(temp_file)
                
                if success:
                    # Update download count
                    self._increment_download_count(plugin_id)
                
                return success
                
            finally:
                # Cleanup temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            logger.error(f"Plugin download failed: {e}")
            return False
    
    def rate_plugin(self, plugin_id: str, rating: float, review: Optional[str] = None) -> bool:
        """Rate a plugin.
        
        Args:
            plugin_id: Plugin identifier
            rating: Rating (1-5)
            review: Optional review text
            
        Returns:
            True if rating submitted successfully
        """
        try:
            if not 1 <= rating <= 5:
                logger.error("Rating must be between 1 and 5")
                return False
            
            # Submit rating to marketplace API
            return self._submit_rating(plugin_id, rating, review)
            
        except Exception as e:
            logger.error(f"Plugin rating failed: {e}")
            return False
    
    def _fetch_marketplace_plugins(self, category: Optional[str], sort_by: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch plugins from marketplace API."""
        # Mock implementation
        return [
            {
                'plugin_id': 'advanced_detector',
                'name': 'Advanced Anomaly Detector',
                'description': 'Advanced ML-based anomaly detection',
                'author': 'ML Team',
                'version': '2.1.0',
                'rating': 4.8,
                'downloads': 1520,
                'category': 'detector'
            },
            {
                'plugin_id': 'data_visualizer',
                'name': 'Interactive Data Visualizer',
                'description': 'Create interactive charts and plots',
                'author': 'Viz Team',
                'version': '1.5.2',
                'rating': 4.6,
                'downloads': 890,
                'category': 'visualizer'
            }
        ]
    
    def _search_marketplace_api(self, query: str, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search marketplace API."""
        # Mock implementation
        return self._fetch_marketplace_plugins(None, "popularity", 10)
    
    def _get_marketplace_plugin(self, plugin_id: str, version: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get plugin details from marketplace."""
        # Mock implementation
        return {
            'plugin_id': plugin_id,
            'version': version or '1.0.0',
            'download_url': f'https://marketplace.pynomaly.com/download/{plugin_id}'
        }
    
    def _download_file(self, url: str) -> Optional[str]:
        """Download file from URL."""
        try:
            if not REQUESTS_AVAILABLE:
                logger.error("Requests library not available for download")
                return None
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with temp_file as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"File download failed: {e}")
            return None
    
    def _increment_download_count(self, plugin_id: str):
        """Increment download count for plugin."""
        # Mock implementation
        pass
    
    def _submit_rating(self, plugin_id: str, rating: float, review: Optional[str]) -> bool:
        """Submit rating to marketplace."""
        # Mock implementation
        return True
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.cache:
            return False
        
        entry = self.cache[cache_key]
        age = (datetime.now() - entry['timestamp']).total_seconds()
        
        return age < self.cache_timeout