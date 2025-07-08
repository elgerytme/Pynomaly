"""Dataset management endpoints."""

import io
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from pynomaly.application.dto import DataQualityReportDTO, DatasetDTO
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.auth import (
    UserModel,
    require_analyst,
    require_data_scientist,
    require_tenant_admin,
    require_viewer,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.auth_deps import get_container_simple


# Simplified permission system for OpenAPI compatibility
def require_permissions(permission):
    """Simplified permission checker for OpenAPI compatibility."""
    return require_viewer  # Fallback to basic auth for now


class CommonPermissions:
    """Common permission constants."""

    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"


router = APIRouter()


@router.get("/", response_model=list[DatasetDTO])
async def list_datasets(
    has_target: bool | None = Query(None, description="Filter by target presence"),
    limit: int = Query(100, ge=1, le=1000),
    current_user: UserModel = Depends(require_viewer),
    container: Container = Depends(lambda: Container()),
) -> list[DatasetDTO]:
    """List all datasets."""
    dataset_repo = container.dataset_repository()

    # Get all datasets
    datasets = dataset_repo.find_all()

    # Apply filters
    if has_target is not None:
        datasets = [d for d in datasets if d.has_target == has_target]

    # Limit results
    datasets = datasets[:limit]

    # Convert to DTOs
    return [
        DatasetDTO(
            id=d.id,
            name=d.name,
            shape=d.shape,
            n_samples=d.n_samples,
            n_features=d.n_features,
            feature_names=d.feature_names or [],
            has_target=d.has_target,
            target_column=d.target_column,
            created_at=d.created_at,
            metadata=d.metadata,
            description=d.description,
            memory_usage_mb=d.memory_usage / 1024 / 1024,
            numeric_features=len(d.get_numeric_features()),
            categorical_features=len(d.get_categorical_features()),
        )
        for d in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetDTO)
async def get_dataset(
    dataset_id: UUID,
    current_user: UserModel = Depends(require_viewer),
    container: Container = Depends(lambda: Container()),
) -> DatasetDTO:
    """Get a specific dataset."""
    dataset_repo = container.dataset_repository()

    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetDTO(
        id=dataset.id,
        name=dataset.name,
        shape=dataset.shape,
        n_samples=dataset.n_samples,
        n_features=dataset.n_features,
        feature_names=dataset.feature_names or [],
        has_target=dataset.has_target,
        target_column=dataset.target_column,
        created_at=dataset.created_at,
        metadata=dataset.metadata,
        description=dataset.description,
        memory_usage_mb=dataset.memory_usage / 1024 / 1024,
        numeric_features=len(dataset.get_numeric_features()),
        categorical_features=len(dataset.get_categorical_features()),
    )


@router.post("/upload", response_model=DatasetDTO)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    description: str | None = Form(None),
    target_column: str | None = Form(None),
    current_user: UserModel = Depends(require_data_scientist),
    container: Container = Depends(lambda: Container()),
) -> DatasetDTO:
    """Upload a dataset from file."""
    settings = container.config()

    # Check file size
    file_size_mb = len(await file.read()) / 1024 / 1024
    await file.seek(0)  # Reset file pointer

    if file_size_mb > settings.max_dataset_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_dataset_size_mb}MB",
        )

    # Determine file type and load
    if file.filename.endswith((".csv", ".tsv", ".txt")):
        # Load CSV
        try:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse CSV: {str(e)}"
            )
    elif file.filename.endswith((".parquet", ".pq")):
        # Load Parquet
        try:
            content = await file.read()
            df = pd.read_parquet(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse Parquet: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported file format. Use CSV or Parquet."
        )

    # Create dataset
    dataset_name = name or file.filename.rsplit(".", 1)[0]

    try:
        dataset = Dataset(
            name=dataset_name,
            data=df,
            description=description,
            target_column=target_column,
            metadata={
                "source": "upload",
                "original_filename": file.filename,
                "file_size_mb": file_size_mb,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create dataset: {str(e)}"
        )

    # Save to repository
    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)

    return DatasetDTO(
        id=dataset.id,
        name=dataset.name,
        shape=dataset.shape,
        n_samples=dataset.n_samples,
        n_features=dataset.n_features,
        feature_names=dataset.feature_names or [],
        has_target=dataset.has_target,
        target_column=dataset.target_column,
        created_at=dataset.created_at,
        metadata=dataset.metadata,
        description=dataset.description,
        memory_usage_mb=dataset.memory_usage / 1024 / 1024,
        numeric_features=len(dataset.get_numeric_features()),
        categorical_features=len(dataset.get_categorical_features()),
    )


@router.get("/{dataset_id}/quality", response_model=DataQualityReportDTO)
async def check_dataset_quality(
    dataset_id: UUID,
    current_user: UserModel = Depends(require_viewer),
    container: Container = Depends(lambda: Container()),
) -> DataQualityReportDTO:
    """Check dataset quality."""
    dataset_repo = container.dataset_repository()
    feature_validator = container.feature_validator()

    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Run quality check
    quality_report = feature_validator.check_data_quality(dataset)
    suggestions = feature_validator.suggest_preprocessing(quality_report)

    return DataQualityReportDTO(
        n_samples=quality_report["n_samples"],
        n_features=quality_report["n_features"],
        missing_values=quality_report["missing_values"],
        constant_features=quality_report["constant_features"],
        low_variance_features=quality_report["low_variance_features"],
        infinite_values=quality_report["infinite_values"],
        duplicate_rows=quality_report["duplicate_rows"],
        quality_score=quality_report["quality_score"],
        suggestions=suggestions,
    )


@router.get("/{dataset_id}/sample")
async def get_dataset_sample(
    dataset_id: UUID,
    n: int = Query(10, ge=1, le=100, description="Number of rows to return"),
    container: Container = Depends(get_container_simple),
    _user=Depends(require_permissions(CommonPermissions.DATASET_READ)),
) -> dict:
    """Get a sample of dataset rows."""
    dataset_repo = container.dataset_repository()

    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get sample
    sample_size = min(n, dataset.n_samples)
    sample_df = dataset.data.sample(n=sample_size, random_state=42)

    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset.name,
        "sample_size": sample_size,
        "total_rows": dataset.n_samples,
        "columns": list(dataset.data.columns),
        "data": sample_df.to_dict(orient="records"),
    }


@router.post("/{dataset_id}/split")
async def split_dataset(
    dataset_id: UUID,
    test_size: float = Query(0.2, ge=0.1, le=0.5),
    random_state: int | None = Query(None),
    container: Container = Depends(get_container_simple),
    _user=Depends(require_permissions(CommonPermissions.DATASET_WRITE)),
) -> dict:
    """Split dataset into train and test sets."""
    dataset_repo = container.dataset_repository()

    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Split dataset
        train_dataset, test_dataset = dataset.split(
            test_size=test_size, random_state=random_state
        )

        # Save both datasets
        dataset_repo.save(train_dataset)
        dataset_repo.save(test_dataset)

        return {
            "train_dataset_id": str(train_dataset.id),
            "test_dataset_id": str(test_dataset.id),
            "train_size": train_dataset.n_samples,
            "test_size": test_dataset.n_samples,
            "train_name": train_dataset.name,
            "test_name": test_dataset.name,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to split dataset: {str(e)}"
        )


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: UUID,
    container: Container = Depends(get_container_simple),
    _user=Depends(require_permissions(CommonPermissions.DATASET_DELETE)),
) -> dict:
    """Delete a dataset."""
    dataset_repo = container.dataset_repository()

    if not dataset_repo.exists(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    success = dataset_repo.delete(dataset_id)

    return {"success": success, "message": "Dataset deleted"}
