"""Authentication and Authorization Service for Hexagonal Architecture."""

from fastapi import FastAPI, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared', 'src'))

from shared.auth import (
    auth_service, authz_service, user_service,
    User, UserRole, AuthToken,
    authenticate_user, create_access_token, verify_token, has_permission
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Authentication & Authorization Service",
    description="Centralized Auth Service for Hexagonal Architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[User]:
    """Get current user from token."""
    if not credentials:
        return None
    
    user = await verify_token(credentials.credentials)
    return user

async def require_auth(user: User = Depends(get_current_user)) -> User:
    """Require authentication."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user

async def require_admin(user: User = Depends(require_auth)) -> User:
    """Require admin role."""
    if UserRole.ADMIN not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user

# Health and status endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "auth-service", "timestamp": datetime.utcnow()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready", "service": "auth-service", "timestamp": datetime.utcnow()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "auth_login_attempts_total": 150,
        "auth_successful_logins_total": 140,
        "auth_failed_logins_total": 10,
        "auth_active_tokens": 45,
        "auth_users_total": len(await user_service.list_users())
    }

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """User login endpoint."""
    try:
        user = await authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        token = await create_access_token(user)
        
        # Update last login
        await user_service.update_user(user.user_id, {"last_login": datetime.utcnow().isoformat()})
        
        return {
            "status": "success",
            "data": {
                "access_token": token.access_token,
                "token_type": token.token_type,
                "expires_in": token.expires_in,
                "refresh_token": token.refresh_token,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": [role.value for role in user.roles]
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/v1/auth/refresh")
async def refresh_token_endpoint(refresh_token: str = Form(...)):
    """Refresh access token."""
    try:
        new_token = await auth_service.refresh_token(refresh_token)
        if not new_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        return {
            "status": "success",
            "data": {
                "access_token": new_token.access_token,
                "token_type": new_token.token_type,
                "expires_in": new_token.expires_in,
                "refresh_token": new_token.refresh_token
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@app.post("/api/v1/auth/verify")
async def verify_token_endpoint(token: str = Form(...)):
    """Verify access token."""
    try:
        user = await verify_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return {
            "status": "success",
            "data": {
                "valid": True,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": [role.value for role in user.roles]
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=500, detail="Token verification failed")

@app.get("/api/v1/auth/me")
async def get_current_user_info(user: User = Depends(require_auth)):
    """Get current user information."""
    permissions = await authz_service.get_user_permissions(user)
    
    return {
        "status": "success",
        "data": {
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "is_active": user.is_active,
                "last_login": user.last_login
            },
            "permissions": permissions
        }
    }

# Authorization endpoints
@app.post("/api/v1/auth/check-permission")
async def check_permission(
    request: Dict[str, str],
    user: User = Depends(require_auth)
):
    """Check if user has specific permission."""
    try:
        resource = request.get("resource")
        action = request.get("action")
        
        if not resource or not action:
            raise HTTPException(status_code=400, detail="resource and action are required")
        
        has_perm = await has_permission(user, resource, action)
        
        return {
            "status": "success",
            "data": {
                "user_id": user.user_id,
                "resource": resource,
                "action": action,
                "has_permission": has_perm
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Permission check failed: {e}")
        raise HTTPException(status_code=500, detail="Permission check failed")

@app.get("/api/v1/auth/permissions")
async def get_user_permissions(user: User = Depends(require_auth)):
    """Get all permissions for current user."""
    try:
        permissions = await authz_service.get_user_permissions(user)
        
        return {
            "status": "success",
            "data": {
                "user_id": user.user_id,
                "permissions": permissions
            }
        }
    except Exception as e:
        logger.error(f"Get permissions failed: {e}")
        raise HTTPException(status_code=500, detail="Get permissions failed")

# User management endpoints
@app.post("/api/v1/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    request: Dict[str, Any],
    admin_user: User = Depends(require_admin)
):
    """Create new user (admin only)."""
    try:
        username = request.get("username")
        password = request.get("password")
        email = request.get("email")
        roles = request.get("roles", ["viewer"])
        
        if not all([username, password, email]):
            raise HTTPException(status_code=400, detail="username, password, and email are required")
        
        # Convert string roles to UserRole enums
        user_roles = []
        for role in roles:
            try:
                user_roles.append(UserRole(role))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
        
        user = await user_service.create_user(username, password, email, user_roles)
        
        return {
            "status": "success",
            "data": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "created_at": user.created_at
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(status_code=500, detail="User creation failed")

@app.get("/api/v1/users")
async def list_users(admin_user: User = Depends(require_admin)):
    """List all users (admin only)."""
    try:
        users = await user_service.list_users()
        
        return {
            "status": "success",
            "data": {
                "users": [
                    {
                        "user_id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "roles": [role.value for role in user.roles],
                        "is_active": user.is_active,
                        "created_at": user.created_at,
                        "last_login": user.last_login
                    }
                    for user in users
                ],
                "total_count": len(users)
            }
        }
    except Exception as e:
        logger.error(f"List users failed: {e}")
        raise HTTPException(status_code=500, detail="List users failed")

@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str, admin_user: User = Depends(require_admin)):
    """Get user by ID (admin only)."""
    try:
        user = await user_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "status": "success",
            "data": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "is_active": user.is_active,
                "created_at": user.created_at,
                "last_login": user.last_login
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user failed: {e}")
        raise HTTPException(status_code=500, detail="Get user failed")

@app.put("/api/v1/users/{user_id}")
async def update_user(
    user_id: str,
    request: Dict[str, Any],
    admin_user: User = Depends(require_admin)
):
    """Update user (admin only)."""
    try:
        success = await user_service.update_user(user_id, request)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "status": "success",
            "data": {
                "user_id": user_id,
                "updated": success
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user failed: {e}")
        raise HTTPException(status_code=500, detail="Update user failed")

@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: str, admin_user: User = Depends(require_admin)):
    """Delete user (admin only)."""
    try:
        success = await user_service.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "status": "success",
            "data": {
                "user_id": user_id,
                "deleted": success
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user failed: {e}")
        raise HTTPException(status_code=500, detail="Delete user failed")

# System endpoints
@app.get("/api/v1/roles")
async def list_roles():
    """List available user roles."""
    return {
        "status": "success",
        "data": {
            "roles": [
                {
                    "name": role.value,
                    "description": f"{role.value.replace('_', ' ').title()} role"
                }
                for role in UserRole
            ]
        }
    }

@app.get("/api/v1/status")
async def get_service_status():
    """Get auth service status."""
    users = await user_service.list_users()
    
    return {
        "status": "running",
        "service": "auth-service",
        "version": "1.0.0",
        "environment": "development",
        "timestamp": datetime.utcnow(),
        "statistics": {
            "total_users": len(users),
            "active_users": len([u for u in users if u.is_active]),
            "roles_available": len(UserRole)
        },
        "capabilities": [
            "authentication",
            "authorization",
            "user_management",
            "role_based_access_control",
            "jwt_tokens"
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8090,
        reload=True,
        log_level="info"
    )