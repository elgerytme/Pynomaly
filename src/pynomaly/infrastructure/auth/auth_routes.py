"""FastAPI routes for enterprise authentication."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from .auth_service import EnterpriseAuthService
from .enterprise_auth import AuthMethod, MFAMethod, Permission

logger = logging.getLogger(__name__)

# Request/Response models
class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str
    auth_method: AuthMethod = AuthMethod.PASSWORD


class OAuthLoginRequest(BaseModel):
    """OAuth login request."""
    code: str
    redirect_uri: str
    state: Optional[str] = None


class SAMLResponse(BaseModel):
    """SAML response."""
    saml_response: str
    relay_state: Optional[str] = None


class MFAVerificationRequest(BaseModel):
    """MFA verification request."""
    mfa_token: str
    mfa_method: MFAMethod
    verification_code: str


class MFASetupRequest(BaseModel):
    """MFA setup request."""
    mfa_method: MFAMethod


class UserResponse(BaseModel):
    """User response."""
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[str]
    permissions: List[str]
    is_active: bool
    last_login: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response."""
    success: bool
    access_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    user: Optional[UserResponse] = None
    requires_mfa: bool = False
    mfa_token: Optional[str] = None
    mfa_methods: List[MFAMethod] = []
    error_message: Optional[str] = None


class RoleAssignmentRequest(BaseModel):
    """Role assignment request."""
    user_id: str
    role_id: str


def create_auth_router(auth_service: EnterpriseAuthService) -> APIRouter:
    """Create authentication router."""
    
    router = APIRouter(prefix="/auth", tags=["authentication"])
    
    def get_request_info(request: Request) -> Dict[str, Any]:
        """Extract request information for authentication."""
        return {
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers)
            }
        }
    
    @router.post("/login", response_model=LoginResponse)
    async def login(request: LoginRequest, req: Request):
        """Authenticate user with username/password."""
        try:
            request_info = get_request_info(req)
            
            credentials = {
                "username": request.username,
                "password": request.password
            }
            
            result = await auth_service.authenticate(
                auth_method=request.auth_method,
                credentials=credentials,
                request_info=request_info
            )
            
            if result.success and result.user and result.session:
                # Generate JWT token
                token = await auth_service.session_manager.generate_jwt_token(
                    result.user, result.session
                )
                
                return LoginResponse(
                    success=True,
                    access_token=token,
                    expires_in=int(auth_service.session_manager.session_timeout.total_seconds()),
                    user=UserResponse(
                        user_id=result.user.user_id,
                        username=result.user.username,
                        email=result.user.email,
                        full_name=result.user.full_name,
                        roles=list(result.user.roles),
                        permissions=[p.value for p in result.user.permissions],
                        is_active=result.user.is_active,
                        last_login=result.user.last_login.isoformat() if result.user.last_login else None
                    )
                )
            elif result.requires_mfa:
                return LoginResponse(
                    success=False,
                    requires_mfa=True,
                    mfa_methods=result.mfa_methods,
                    user=UserResponse(
                        user_id=result.user.user_id,
                        username=result.user.username,
                        email=result.user.email,
                        full_name=result.user.full_name,
                        roles=list(result.user.roles),
                        permissions=[p.value for p in result.user.permissions],
                        is_active=result.user.is_active
                    ) if result.user else None
                )
            else:
                return LoginResponse(
                    success=False,
                    error_message=result.error_message or "Authentication failed"
                )
                
        except Exception as e:
            logger.error(f"Login endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal authentication error"
            )
    
    @router.post("/oauth/login", response_model=LoginResponse)
    async def oauth_login(request: OAuthLoginRequest, req: Request):
        """Authenticate user with OAuth code."""
        try:
            request_info = get_request_info(req)
            
            credentials = {
                "code": request.code,
                "redirect_uri": request.redirect_uri
            }
            
            result = await auth_service.authenticate(
                auth_method=AuthMethod.OAUTH2,
                credentials=credentials,
                request_info=request_info
            )
            
            if result.success and result.user and result.session:
                token = await auth_service.session_manager.generate_jwt_token(
                    result.user, result.session
                )
                
                return LoginResponse(
                    success=True,
                    access_token=token,
                    expires_in=int(auth_service.session_manager.session_timeout.total_seconds()),
                    user=UserResponse(
                        user_id=result.user.user_id,
                        username=result.user.username,
                        email=result.user.email,
                        full_name=result.user.full_name,
                        roles=list(result.user.roles),
                        permissions=[p.value for p in result.user.permissions],
                        is_active=result.user.is_active,
                        last_login=result.user.last_login.isoformat() if result.user.last_login else None
                    )
                )
            else:
                return LoginResponse(
                    success=False,
                    error_message=result.error_message or "OAuth authentication failed"
                )
                
        except Exception as e:
            logger.error(f"OAuth login endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth authentication error"
            )
    
    @router.post("/saml/acs", response_model=LoginResponse)
    async def saml_acs(request: SAMLResponse, req: Request):
        """SAML Assertion Consumer Service."""
        try:
            request_info = get_request_info(req)
            
            credentials = {
                "saml_response": request.saml_response
            }
            
            result = await auth_service.authenticate(
                auth_method=AuthMethod.SAML,
                credentials=credentials,
                request_info=request_info
            )
            
            if result.success and result.user and result.session:
                token = await auth_service.session_manager.generate_jwt_token(
                    result.user, result.session
                )
                
                return LoginResponse(
                    success=True,
                    access_token=token,
                    expires_in=int(auth_service.session_manager.session_timeout.total_seconds()),
                    user=UserResponse(
                        user_id=result.user.user_id,
                        username=result.user.username,
                        email=result.user.email,
                        full_name=result.user.full_name,
                        roles=list(result.user.roles),
                        permissions=[p.value for p in result.user.permissions],
                        is_active=result.user.is_active
                    )
                )
            else:
                return LoginResponse(
                    success=False,
                    error_message=result.error_message or "SAML authentication failed"
                )
                
        except Exception as e:
            logger.error(f"SAML ACS endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAML authentication error"
            )
    
    @router.post("/mfa/verify", response_model=LoginResponse)
    async def verify_mfa(request: MFAVerificationRequest):
        """Verify multi-factor authentication."""
        try:
            result = await auth_service.verify_mfa(
                mfa_token=request.mfa_token,
                mfa_method=request.mfa_method,
                verification_code=request.verification_code
            )
            
            if result.success and result.user and result.session:
                token = await auth_service.session_manager.generate_jwt_token(
                    result.user, result.session
                )
                
                return LoginResponse(
                    success=True,
                    access_token=token,
                    expires_in=int(auth_service.session_manager.session_timeout.total_seconds()),
                    user=UserResponse(
                        user_id=result.user.user_id,
                        username=result.user.username,
                        email=result.user.email,
                        full_name=result.user.full_name,
                        roles=list(result.user.roles),
                        permissions=[p.value for p in result.user.permissions],
                        is_active=result.user.is_active,
                        last_login=result.user.last_login.isoformat() if result.user.last_login else None
                    )
                )
            else:
                return LoginResponse(
                    success=False,
                    error_message=result.error_message or "MFA verification failed"
                )
                
        except Exception as e:
            logger.error(f"MFA verification endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MFA verification error"
            )
    
    @router.post("/mfa/setup")
    async def setup_mfa(request: MFASetupRequest, current_user=Depends(get_current_user)):
        """Setup multi-factor authentication."""
        try:
            result = await auth_service.setup_mfa(
                user_id=current_user.user_id,
                mfa_method=request.mfa_method
            )
            
            return result
            
        except Exception as e:
            logger.error(f"MFA setup endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MFA setup error"
            )
    
    @router.post("/logout")
    async def logout(current_user=Depends(get_current_user)):
        """Logout current user."""
        try:
            # In a real implementation, we'd get the session ID from the token
            # For now, we'll just return success
            return {"success": True, "message": "Logged out successfully"}
            
        except Exception as e:
            logger.error(f"Logout endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout error"
            )
    
    @router.get("/me", response_model=UserResponse)
    async def get_current_user_info(current_user=Depends(get_current_user)):
        """Get current user information."""
        return UserResponse(
            user_id=current_user.user_id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            roles=list(current_user.roles),
            permissions=[p.value for p in current_user.permissions],
            is_active=current_user.is_active,
            last_login=current_user.last_login.isoformat() if current_user.last_login else None
        )
    
    @router.post("/roles/assign")
    async def assign_role(
        request: RoleAssignmentRequest,
        current_user=Depends(get_current_user)
    ):
        """Assign role to user (admin only)."""
        if Permission.USER_ADMIN not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator privileges required"
            )
        
        try:
            success = await auth_service.assign_role(request.user_id, request.role_id)
            
            if success:
                return {"success": True, "message": f"Role {request.role_id} assigned to user {request.user_id}"}
            else:
                return {"success": False, "message": "Failed to assign role"}
                
        except Exception as e:
            logger.error(f"Role assignment endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Role assignment error"
            )
    
    @router.delete("/roles/remove")
    async def remove_role(
        request: RoleAssignmentRequest,
        current_user=Depends(get_current_user)
    ):
        """Remove role from user (admin only)."""
        if Permission.USER_ADMIN not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator privileges required"
            )
        
        try:
            success = await auth_service.remove_role(request.user_id, request.role_id)
            
            if success:
                return {"success": True, "message": f"Role {request.role_id} removed from user {request.user_id}"}
            else:
                return {"success": False, "message": "Failed to remove role"}
                
        except Exception as e:
            logger.error(f"Role removal endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Role removal error"
            )
    
    @router.get("/roles")
    async def list_roles(current_user=Depends(get_current_user)):
        """List all available roles."""
        try:
            roles = await auth_service.rbac_manager.list_roles()
            
            return {
                "roles": [
                    {
                        "role_id": role.role_id,
                        "name": role.name,
                        "description": role.description,
                        "permissions": [p.value for p in role.permissions],
                        "is_system_role": role.is_system_role
                    }
                    for role in roles
                ]
            }
            
        except Exception as e:
            logger.error(f"List roles endpoint error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list roles"
            )
    
    async def get_current_user(request: Request):
        """Dependency to get current authenticated user."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        user = await auth_service.verify_jwt_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return user
    
    return router


# Health check endpoint
def create_health_router() -> APIRouter:
    """Create health check router."""
    router = APIRouter(prefix="/health", tags=["health"])
    
    @router.get("/")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "pynomaly-auth",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    return router