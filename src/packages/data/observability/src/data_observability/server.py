"""
Data Observability API Server

Provides REST API server for data observability operations including
catalog management, lineage tracking, and quality monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import UUID
import uvicorn

from .application.facades.observability_facade import DataObservabilityFacade
from .infrastructure.di.container import DataObservabilityContainer


# Pydantic models for API
class AssetRegistration(BaseModel):
    name: str
    asset_type: str
    location: str
    data_format: str
    description: Optional[str] = None
    owner: Optional[str] = None
    domain: Optional[str] = None


class AssetSearch(BaseModel):
    query: str
    limit: int = 20


class AssetResponse(BaseModel):
    id: str
    name: str
    asset_type: str
    location: str
    description: Optional[str] = None
    owner: Optional[str] = None


# FastAPI application
app = FastAPI(
    title="Data Observability API",
    description="REST API for data observability operations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
container = DataObservabilityContainer()
container.wire(modules=[__name__])


def get_facade() -> DataObservabilityFacade:
    """Get observability facade instance."""
    return container.observability_facade()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-observability"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Data Observability API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "assets": "/assets",
            "search": "/assets/search",
            "inspect": "/assets/{asset_id}",
            "dashboard": "/dashboard"
        }
    }


@app.post("/assets", response_model=AssetResponse)
async def register_asset(
    asset: AssetRegistration,
    facade: DataObservabilityFacade = Depends(get_facade)
):
    """Register a new data asset."""
    try:
        registered_asset = facade.register_data_asset(
            name=asset.name,
            asset_type=asset.asset_type,
            location=asset.location,
            data_format=asset.data_format,
            description=asset.description,
            owner=asset.owner,
            domain=asset.domain
        )
        
        return AssetResponse(
            id=str(registered_asset.id),
            name=registered_asset.name,
            asset_type=registered_asset.asset_type,
            location=registered_asset.location,
            description=registered_asset.description,
            owner=registered_asset.owner
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/assets/search")
async def search_assets(
    search: AssetSearch,
    facade: DataObservabilityFacade = Depends(get_facade)
) -> List[AssetResponse]:
    """Search for data assets."""
    try:
        assets = facade.discover_data_assets(search.query, search.limit)
        
        return [
            AssetResponse(
                id=str(asset.id),
                name=asset.name,
                asset_type=asset.asset_type,
                location=asset.location,
                description=asset.description,
                owner=asset.owner
            )
            for asset in assets
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/assets/{asset_id}")
async def get_asset_view(
    asset_id: str,
    facade: DataObservabilityFacade = Depends(get_facade)
) -> Dict[str, Any]:
    """Get comprehensive view of a data asset."""
    try:
        asset_uuid = UUID(asset_id)
        view = facade.get_comprehensive_asset_view(asset_uuid)
        
        if 'error' in view:
            raise HTTPException(status_code=404, detail=view['error'])
        
        return view
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
async def get_dashboard(
    facade: DataObservabilityFacade = Depends(get_facade)
) -> Dict[str, Any]:
    """Get data health dashboard."""
    try:
        return facade.get_data_health_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assets/{asset_id}/investigate")
async def investigate_asset_issue(
    asset_id: str,
    issue_type: str,
    severity: str = "medium",
    facade: DataObservabilityFacade = Depends(get_facade)
) -> Dict[str, Any]:
    """Investigate data issue for an asset."""
    try:
        asset_uuid = UUID(asset_id)
        investigation = facade.investigate_data_issue(
            asset_uuid, issue_type, severity
        )
        return investigation
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "data_observability.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )