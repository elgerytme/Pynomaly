"""
Examples demonstrating how to use RBAC middleware/decorators in FastAPI routes.

This module provides examples of how to integrate role-based access control
into your API endpoints using the newly implemented RBAC middleware.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from pynomaly.infrastructure.auth.middleware import (
    require_analyst,
    require_data_scientist,
    require_role,
    require_super_admin,
    require_tenant_admin,
    require_viewer,
)
from pynomaly.infrastructure.persistence.user_models import UserModel

# Example router demonstrating RBAC usage
rbac_examples_router = APIRouter()


@rbac_examples_router.get("/viewer-only")
async def viewer_only_endpoint(
    user: Annotated[UserModel, Depends(require_viewer)],
) -> dict:
    """
    Endpoint accessible only to users with 'viewer' role.

    The require_viewer dependency automatically:
    1. Authenticates the user
    2. Checks if they have the 'viewer' role
    3. Returns HTTP 403 if they don't have the required role
    """
    return {
        "message": "Hello viewer!",
        "user_id": str(user.id),
        "username": user.username,
        "roles": [role.name for role in user.roles],
    }


@rbac_examples_router.get("/analyst-dashboard")
async def analyst_dashboard(
    user: Annotated[UserModel, Depends(require_analyst)],
) -> dict:
    """
    Analyst-only dashboard endpoint.

    Only users with 'analyst' role can access this endpoint.
    """
    return {
        "message": "Welcome to the analyst dashboard!",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "full_name": f"{user.first_name} {user.last_name}",
        },
        "permissions": ["detection.run", "detection.view", "report.create"],
    }


@rbac_examples_router.post("/data-science/models")
async def create_model(
    user: Annotated[UserModel, Depends(require_data_scientist)],
) -> dict:
    """
    Model creation endpoint for data scientists.

    Only users with 'data_scientist' role can create models.
    """
    return {
        "message": "Model creation initiated",
        "created_by": user.username,
        "user_roles": [role.name for role in user.roles],
    }


@rbac_examples_router.get("/admin/users")
async def list_users(user: Annotated[UserModel, Depends(require_tenant_admin)]) -> dict:
    """
    User management endpoint for tenant administrators.

    Only users with 'tenant_admin' role can access user management.
    """
    return {
        "message": "User management interface",
        "admin_user": user.username,
        "available_actions": ["invite_user", "manage_roles", "view_billing"],
    }


@rbac_examples_router.post("/platform/tenants")
async def create_tenant(
    user: Annotated[UserModel, Depends(require_super_admin)],
) -> dict:
    """
    Platform-level tenant creation for super administrators.

    Only super admins can create new tenants.
    """
    return {
        "message": "Tenant creation interface",
        "super_admin": user.username,
        "platform_permissions": ["tenant.create", "tenant.delete", "user.manage_all"],
    }


@rbac_examples_router.get("/custom-role-example")
async def custom_role_example(
    user: Annotated[UserModel, Depends(require_role("custom_analyst"))],
) -> dict:
    """
    Example of using a custom role with the require_role function.

    This demonstrates how to create role checks for custom roles
    that aren't predefined in the common dependencies.
    """
    return {
        "message": "Custom analyst endpoint",
        "user": user.username,
        "custom_role": "custom_analyst",
    }


@rbac_examples_router.get("/multiple-roles-example")
async def multiple_roles_allowed(
    # Note: This is an example of how you might handle multiple allowed roles
    # For now, we check for one role but you could extend this pattern
    user: Annotated[UserModel, Depends(require_data_scientist)],
) -> dict:
    """
    Example endpoint that could be extended to allow multiple roles.

    Currently shows data scientist access, but the pattern could be extended
    to create a MultiRoleChecker that accepts multiple roles.
    """
    user_roles = [role.name for role in user.roles]

    # Additional role validation could be done here if needed
    allowed_roles = ["data_scientist", "tenant_admin", "super_admin"]
    has_allowed_role = any(role in allowed_roles for role in user_roles)

    if not has_allowed_role:
        # This wouldn't typically be reached due to the dependency,
        # but shows how you might extend for multiple role support
        pass

    return {
        "message": "Multi-role endpoint access granted",
        "user_roles": user_roles,
        "allowed_roles": allowed_roles,
    }


# Example of route integration in your main application
def integrate_rbac_examples(app):
    """
    Example of how to integrate RBAC-protected routes into your main application.

    Args:
        app: FastAPI application instance
    """
    app.include_router(
        rbac_examples_router, prefix="/api/v1/rbac-examples", tags=["RBAC Examples"]
    )


# Usage patterns documentation
RBAC_USAGE_PATTERNS = """
## RBAC Integration Patterns

### 1. Using Predefined Role Dependencies

```python
from pynomaly.infrastructure.auth.middleware import require_analyst

@app.get("/analyst-endpoint")
async def my_endpoint(user: Annotated[UserModel, Depends(require_analyst)]):
    return {"message": "Analyst access granted"}
```

### 2. Using Custom Roles

```python
from pynomaly.infrastructure.auth.middleware import require_role

@app.get("/custom-endpoint")
async def my_endpoint(user: Annotated[UserModel, Depends(require_role("custom_role"))]):
    return {"message": "Custom role access granted"}
```

### 3. Combining with Other Dependencies

```python
@app.post("/create-dataset")
async def create_dataset(
    dataset_data: DatasetCreate,
    user: Annotated[UserModel, Depends(require_data_scientist)],
    db: Session = Depends(get_db)
):
    # Function implementation
    pass
```

### 4. Error Handling

The RBAC middleware automatically handles:
- HTTP 401: Authentication required (no valid token)
- HTTP 403: Access denied (authenticated but insufficient role)

### 5. Available Predefined Roles

- `require_super_admin`: Platform-wide administrator
- `require_tenant_admin`: Tenant administrator
- `require_data_scientist`: Can create/manage models
- `require_analyst`: Can view results and run detection
- `require_viewer`: Read-only access
"""
