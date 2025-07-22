"""Data Ingestion FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

logger = structlog.get_logger()


class StreamIngestionRequest(BaseModel):
    """Request model for stream ingestion."""
    source_type: str
    source_config: Dict[str, Any]
    target_config: Dict[str, Any]
    processing_rules: List[Dict[str, Any]] = []


class StreamIngestionResponse(BaseModel):
    """Response model for stream ingestion."""
    stream_id: str
    status: str
    throughput_rate: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Data Ingestion API server")
    yield
    logger.info("Shutting down Data Ingestion API server")


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Data Ingestion API",
        description="API for data collection, streaming, and batch ingestion",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-ingestion"}


@app.post("/api/v1/stream/start", response_model=StreamIngestionResponse)
async def start_stream_ingestion(request: StreamIngestionRequest) -> StreamIngestionResponse:
    """Start streaming ingestion."""
    logger.info("Starting stream ingestion", source=request.source_type)
    
    stream_id = f"stream_{hash(str(request.source_config)) % 10000}"
    
    return StreamIngestionResponse(
        stream_id=stream_id,
        status="started",
        throughput_rate=1000.0
    )


@app.post("/api/v1/batch/ingest")
async def run_batch_ingestion(
    source: str,
    target: str,
    schedule: Optional[str] = None
) -> Dict[str, Any]:
    """Run batch ingestion."""
    return {
        "batch_id": "batch_001",
        "source": source,
        "target": target,
        "records_ingested": 10000,
        "status": "completed"
    }


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "data_ingestion.server:app",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()