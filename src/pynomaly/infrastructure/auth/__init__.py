"""Infrastructure authentication and authorization."""

from .dependencies import require_api_key, require_role_or_api_key
from .enhanced_dependencies import (
    cli_require_permissions,
    cli_require_roles,
    cli_require_superuser,
    require_permissions,
    require_permissions_or_api_key,
    require_roles,
    require_superuser,
)
from .enhanced_dependencies import (
    get_current_active_user as get_current_active_user_enhanced,
)
from .enhanced_dependencies import get_current_user as get_current_user_enhanced
from .jwt_auth import (
    JWTAuthService,
    TokenPayload,
    TokenResponse,
    UserModel,
    get_auth,
    init_auth,
)
from .jwt_auth_enhanced import (
    EnhancedJWTAuthService,
    JWKSResponse,
)
from .jwt_auth_enhanced import UserModel as EnhancedUserModel
from .jwt_auth_enhanced import get_auth as get_enhanced_auth
from .jwt_auth_enhanced import init_auth as init_enhanced_auth
from .middleware import (
    PermissionChecker,
    RateLimiter,
    RoleChecker,
    create_auth_context,
    default_limiter,
    get_current_active_user,
    get_current_user,
    require_admin,
    require_analyst,
    require_data_scientist,
    require_permission,
    require_read,
    require_role,
    require_super_admin,
    require_tenant_admin,
    require_viewer,
    require_write,
    strict_limiter,
    track_request_metrics,
)
from .websocket_auth import (
    HTMXAuthMiddleware,
    WebSocketAuthMiddleware,
    create_websocket_auth_dependency,
    get_htmx_user,
    require_htmx_auth,
)

__all__ = [
    # JWT Auth
    "JWTAuthService",
    "UserModel",
    "TokenPayload",
    "TokenResponse",
    "init_auth",
    "get_auth",
    # Middleware
    "get_current_user",
    "get_current_active_user",
    "PermissionChecker",
    "RateLimiter",
    "RoleChecker",
    "require_read",
    "require_write",
    "require_admin",
    "require_permission",
    "default_limiter",
    "strict_limiter",
    "track_request_metrics",
    "create_auth_context",
    # Role-based dependencies
    "require_role",
    "require_super_admin",
    "require_tenant_admin",
    "require_data_scientist",
    "require_analyst",
    "require_viewer",
    # Other dependencies
    "require_api_key",
    "require_role_or_api_key",
    # WebSocket & HTMX Auth
    "WebSocketAuthMiddleware",
    "HTMXAuthMiddleware",
    "create_websocket_auth_dependency",
    "get_htmx_user",
    "require_htmx_auth",
]
