"""Data Lineage FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

logger = structlog.get_logger()


class LineageTrackingRequest(BaseModel):
    """Request model for lineage tracking."""
    source: str
    target: str
    process: Optional[str] = None
    transformation_details: Dict[str, Any] = {}


class LineageTrackingResponse(BaseModel):
    """Response model for lineage tracking."""
    lineage_id: str
    source: str
    target: str
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Data Lineage API server")
    yield
    logger.info("Shutting down Data Lineage API server")


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Data Lineage API",
        description="API for tracking data flow, dependencies, and impact analysis",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-lineage"}


@app.post("/api/v1/lineage/track", response_model=LineageTrackingResponse)
async def track_lineage(request: LineageTrackingRequest) -> LineageTrackingResponse:
    """Track data lineage."""
    logger.info("Tracking lineage", source=request.source, target=request.target)
    
    lineage_id = f"lineage_{hash(request.source + request.target) % 10000}"
    
    return LineageTrackingResponse(
        lineage_id=lineage_id,
        source=request.source,
        target=request.target,
        status="tracked"
    )


@app.get("/api/v1/lineage/{dataset}/impact")
async def analyze_impact(
    dataset: str,
    direction: str = "both"
) -> Dict[str, Any]:
    """Analyze data impact."""
    return {
        "dataset": dataset,
        "direction": direction,
        "analysis_id": "impact_001",
        "affected_systems": 5,
        "impact_score": 0.8,
        "upstream_dependencies": 3,
        "downstream_consumers": 7,
        "status": "analyzed"
    }


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "data_lineage.server:app",
        host="0.0.0.0",
        port=8009,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()