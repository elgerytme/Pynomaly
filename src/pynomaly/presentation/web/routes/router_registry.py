"""Router registration layer for web UI components."""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from ..models.dashboard_models import DashboardLayout, NavigationItem
from ..models.ui_models import MountConfig, RouteGuard

logger = logging.getLogger(__name__)


class RouterRegistry:
    """Registry for managing FastAPI routers and their mounting."""
    
    def __init__(self):
        """Initialize router registry."""
        self.routers: Dict[str, APIRouter] = {}
        self.mount_configs: Dict[str, MountConfig] = {}
        self.navigation_items: List[NavigationItem] = []
        self.is_registered = False
    
    def register_router(
        self,
        name: str,
        router: APIRouter,
        mount_config: Optional[MountConfig] = None
    ) -> None:
        """Register a router with optional mount configuration.
        
        Args:
            name: Router name
            router: FastAPI router instance
            mount_config: Optional mount configuration
        """
        if self.is_registered:
            logger.warning(f"Router registry already registered, skipping {name}")
            return
            
        self.routers[name] = router
        
        if mount_config:
            self.mount_configs[name] = mount_config
        
        logger.debug(f"Registered router: {name}")
    
    def register_navigation_item(self, item: NavigationItem) -> None:
        """Register a navigation item.
        
        Args:
            item: Navigation item to register
        """
        if not any(nav.id == item.id for nav in self.navigation_items):
            self.navigation_items.append(item)
            logger.debug(f"Registered navigation item: {item.label}")
    
    def mount_all_routers(self, app: FastAPI) -> bool:
        """Mount all registered routers to the FastAPI app.
        
        Args:
            app: FastAPI application instance
            
        Returns:
            True if all routers were mounted successfully
        """
        if hasattr(app.state, 'routers_mounted') and app.state.routers_mounted:
            logger.info("Routers already mounted, skipping...")
            return True
        
        success_count = 0
        total_routers = len(self.routers)
        
        for name, router in self.routers.items():
            try:
                # Get mount configuration
                mount_config = self.mount_configs.get(name)
                
                if mount_config:
                    # Mount static files if configured
                    if mount_config.static_files_path:
                        try:
                            app.mount(
                                f"/static/{name}",
                                StaticFiles(directory=mount_config.static_files_path),
                                name=f"static_{name}"
                            )
                            logger.debug(f"Mounted static files for {name}")
                        except Exception as e:
                            logger.warning(f"Failed to mount static files for {name}: {e}")
                    
                    # Mount router with path prefix
                    prefix = mount_config.mount_path
                    tags = [name]
                else:
                    # Default mounting
                    prefix = f"/{name}"
                    tags = [name]
                
                # Include router
                app.include_router(router, prefix=prefix, tags=tags)
                
                # Mark mount config as mounted
                if mount_config:
                    mount_config.mark_as_mounted()
                
                success_count += 1
                logger.info(f"Successfully mounted router: {name}")
                
            except Exception as e:
                logger.error(f"Failed to mount router {name}: {e}")
        
        # Mark routers as mounted
        app.state.routers_mounted = True
        app.state.mounted_router_count = success_count
        
        if success_count == total_routers:
            logger.info(f"Successfully mounted all {total_routers} routers")
            return True
        else:
            logger.warning(f"Mounted {success_count}/{total_routers} routers")
            return False
    
    def get_navigation_items(
        self,
        user_permissions: Optional[List[str]] = None
    ) -> List[NavigationItem]:
        """Get navigation items filtered by user permissions.
        
        Args:
            user_permissions: Optional list of user permissions
            
        Returns:
            Filtered list of navigation items
        """
        if not user_permissions:
            return [item for item in self.navigation_items if not item.requires_auth]
        
        filtered_items = []
        for item in self.navigation_items:
            # Check if user has required permissions
            if item.required_permissions:
                if not all(perm in user_permissions for perm in item.required_permissions):
                    continue
            
            filtered_items.append(item)
        
        # Sort by order
        return sorted(filtered_items, key=lambda x: x.order)
    
    def get_mount_status(self) -> Dict[str, bool]:
        """Get mount status for all registered routers.
        
        Returns:
            Dictionary mapping router names to mount status
        """
        status = {}
        for name, mount_config in self.mount_configs.items():
            status[name] = mount_config.is_mounted
        
        # Add routers without mount config
        for name in self.routers:
            if name not in status:
                status[name] = self.is_registered
        
        return status
    
    def finalize_registration(self) -> None:
        """Finalize router registration to prevent further changes."""
        self.is_registered = True
        logger.info(f"Finalized router registry with {len(self.routers)} routers")


# Global router registry instance
_router_registry: Optional[RouterRegistry] = None


def get_router_registry() -> RouterRegistry:
    """Get the global router registry instance.
    
    Returns:
        Router registry instance
    """
    global _router_registry
    if _router_registry is None:
        _router_registry = RouterRegistry()
    return _router_registry


def register_dashboard_router() -> None:
    """Register dashboard router with the registry."""
    from ..routes.dashboard_routes import create_dashboard_router
    
    registry = get_router_registry()
    
    # Create dashboard router
    router = create_dashboard_router()
    
    # Create mount configuration
    mount_config = MountConfig(
        component_name="dashboard",
        mount_path="/dashboard",
        static_files_path="src/pynomaly/presentation/web/static",
        templates_path="src/pynomaly/presentation/web/templates",
        route_guards={
            "/dashboard": RouteGuard(requires_auth=False),
            "/dashboard/admin": RouteGuard(
                requires_auth=True,
                required_roles=["admin"],
                required_permissions=["dashboard:admin"]
            )
        }
    )
    
    # Register router
    registry.register_router("dashboard", router, mount_config)
    
    # Register navigation items
    registry.register_navigation_item(NavigationItem(
        id="dashboard",
        label="Dashboard",
        route="/dashboard",
        icon="dashboard",
        order=1
    ))
    
    logger.info("Dashboard router registered")


def register_websocket_routes() -> None:
    """Register WebSocket routes with the registry."""
    from ..routes.websocket_routes import create_websocket_router
    
    registry = get_router_registry()
    
    # Create WebSocket router
    router = create_websocket_router()
    
    # Create mount configuration
    mount_config = MountConfig(
        component_name="websocket",
        mount_path="/ws",
        static_files_path="",
        templates_path="",
        route_guards={
            "/ws": RouteGuard(requires_auth=True)
        }
    )
    
    # Register router
    registry.register_router("websocket", router, mount_config)
    
    logger.info("WebSocket router registered")


def register_all_routers() -> None:
    """Register all web UI routers."""
    register_dashboard_router()
    register_websocket_routes()
    
    # Finalize registration
    registry = get_router_registry()
    registry.finalize_registration()
    
    logger.info("All web UI routers registered")
