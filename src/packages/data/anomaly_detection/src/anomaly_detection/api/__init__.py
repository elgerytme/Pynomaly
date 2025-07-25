"""API package for Anomaly Detection."""

# Import migrated health module from shared.observability
try:
    from shared.observability.api import health
except ImportError:
    # Fallback health implementation
    from fastapi import APIRouter

    health = APIRouter()

    @health.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "anomaly-detection"}
