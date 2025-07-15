"""JWKS (JSON Web Key Set) endpoint for JWT token verification."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from pynomaly.infrastructure.auth.enhanced_dependencies import require_superuser
from pynomaly.infrastructure.auth.jwt_auth_enhanced import (
    EnhancedJWTAuthService,
    JWKSResponse,
    UserModel,
    get_auth,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/.well-known/jwks.json", response_model=JWKSResponse)
async def get_jwks(
    auth_service: EnhancedJWTAuthService = Depends(get_auth),
) -> JWKSResponse:
    """Get JSON Web Key Set (JWKS) for JWT token verification.

    This endpoint provides the public keys used to verify JWT tokens.
    It's typically used by clients and other services to validate tokens.

    Returns:
        JWKS response with public keys

    Raises:
        HTTPException: If auth service is not available
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    jwks = auth_service.get_jwks()

    # Return as JSON response with proper headers
    return JSONResponse(
        content=jwks.model_dump(),
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Type": "application/json",
        },
    )


@router.post("/rotate-keys", status_code=status.HTTP_200_OK)
async def rotate_keys(
    current_user: UserModel = Depends(require_superuser),
    auth_service: EnhancedJWTAuthService = Depends(get_auth),
) -> dict:
    """Rotate JWT signing keys.

    This endpoint allows superusers to rotate the JWT signing keys.
    After rotation, new tokens will be signed with the new key,
    but existing tokens can still be verified with the old key.

    Args:
        current_user: Current superuser
        auth_service: Enhanced JWT auth service

    Returns:
        Success message with new key ID

    Raises:
        HTTPException: If auth service is not available
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    new_key_id = auth_service.rotate_keys()

    return {
        "message": "Keys rotated successfully",
        "new_key_id": new_key_id,
        "timestamp": auth_service.settings.app.environment,
    }


@router.post("/cleanup-keys", status_code=status.HTTP_200_OK)
async def cleanup_old_keys(
    keep_count: int = 2,
    current_user: UserModel = Depends(require_superuser),
    auth_service: EnhancedJWTAuthService = Depends(get_auth),
) -> dict:
    """Clean up old JWT signing keys.

    This endpoint allows superusers to clean up old JWT signing keys,
    keeping only the most recent ones. This helps maintain security
    and reduce storage overhead.

    Args:
        keep_count: Number of keys to keep (default: 2)
        current_user: Current superuser
        auth_service: Enhanced JWT auth service

    Returns:
        Success message

    Raises:
        HTTPException: If auth service is not available
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    if keep_count < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Keep count must be at least 1",
        )

    auth_service.cleanup_old_keys(keep_count)

    return {
        "message": f"Old keys cleaned up, keeping {keep_count} most recent keys",
        "current_key_count": len(auth_service.jwks_keys),
    }


@router.get("/keys/info", response_model=dict)
async def get_key_info(
    current_user: UserModel = Depends(require_superuser),
    auth_service: EnhancedJWTAuthService = Depends(get_auth),
) -> dict:
    """Get information about current JWT signing keys.

    This endpoint provides information about the current JWT signing keys
    for monitoring and debugging purposes.

    Args:
        current_user: Current superuser
        auth_service: Enhanced JWT auth service

    Returns:
        Key information

    Raises:
        HTTPException: If auth service is not available
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    return {
        "current_key_id": auth_service.current_key_id,
        "total_keys": len(auth_service.jwks_keys),
        "key_ids": [key.kid for key in auth_service.jwks_keys],
        "algorithm": auth_service.algorithm,
    }
