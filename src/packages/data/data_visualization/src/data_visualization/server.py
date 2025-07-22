"""Data Visualization FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

logger = structlog.get_logger()


class ChartRequest(BaseModel):
    """Request model for chart creation."""
    data_source: str
    chart_type: str = "bar"
    x_column: str
    y_column: str
    styling: Dict[str, Any] = {}


class ChartResponse(BaseModel):
    """Response model for chart."""
    chart_id: str
    chart_type: str
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Data Visualization API server")
    yield
    logger.info("Shutting down Data Visualization API server")


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Data Visualization API",
        description="API for charts, dashboards, and interactive visualizations",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-visualization"}


@app.post("/api/v1/charts", response_model=ChartResponse)
async def create_chart(request: ChartRequest) -> ChartResponse:
    """Create chart visualization."""
    logger.info("Creating chart", type=request.chart_type, source=request.data_source)
    
    chart_id = f"chart_{hash(request.data_source + request.chart_type) % 10000}"
    
    return ChartResponse(
        chart_id=chart_id,
        chart_type=request.chart_type,
        status="created"
    )


@app.post("/api/v1/dashboards")
async def create_dashboard(
    name: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create interactive dashboard."""
    dashboard_id = f"dash_{hash(name) % 10000}"
    
    return {
        "dashboard_id": dashboard_id,
        "name": name,
        "url": f"/dashboards/{dashboard_id}",
        "status": "created"
    }


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "data_visualization.server:app",
        host="0.0.0.0",
        port=8012,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()