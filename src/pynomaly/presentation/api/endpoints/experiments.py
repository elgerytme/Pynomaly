"""Experiment tracking endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from pynomaly.application.dto import (
    CreateExperimentDTO,
    ExperimentDTO,
    LeaderboardEntryDTO,
    RunDTO,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container, get_current_user

router = APIRouter()


@router.post("/", response_model=ExperimentDTO)
async def create_experiment(
    experiment_data: CreateExperimentDTO,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> ExperimentDTO:
    """Create a new experiment."""
    experiment_service = container.experiment_tracking_service()

    try:
        experiment_id = await experiment_service.create_experiment(
            name=experiment_data.name,
            description=experiment_data.description,
            tags=experiment_data.tags,
        )

        # Get the created experiment
        experiments = await experiment_service.list_saved_models()
        experiment = experiments.get(experiment_id, {})

        return ExperimentDTO(
            id=experiment_id,
            name=experiment_data.name,
            description=experiment_data.description,
            tags=experiment_data.tags,
            created_at=experiment.get("created_at"),
            runs=[],
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create experiment: {str(e)}"
        )


@router.get("/", response_model=list[ExperimentDTO])
async def list_experiments(
    tag: str | None = Query(None, description="Filter by tag"),
    limit: int = Query(100, ge=1, le=1000),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> list[ExperimentDTO]:
    """List all experiments."""
    experiment_service = container.experiment_tracking_service()

    # Load experiments
    experiment_service._load_experiments()
    experiments = []

    for exp_id, exp_data in experiment_service.experiments.items():
        # Apply tag filter if provided
        if tag and tag not in exp_data.get("tags", []):
            continue

        # Convert runs
        runs = [RunDTO(**run) for run in exp_data.get("runs", [])]

        experiment = ExperimentDTO(
            id=exp_id,
            name=exp_data["name"],
            description=exp_data.get("description"),
            tags=exp_data.get("tags", []),
            created_at=exp_data["created_at"],
            runs=runs,
        )

        experiments.append(experiment)

    # Sort by creation date and limit
    experiments.sort(key=lambda e: e.created_at, reverse=True)

    return experiments[:limit]


@router.get("/{experiment_id}", response_model=ExperimentDTO)
async def get_experiment(
    experiment_id: str,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> ExperimentDTO:
    """Get a specific experiment."""
    experiment_service = container.experiment_tracking_service()

    # Load experiments
    experiment_service._load_experiments()

    if experiment_id not in experiment_service.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp_data = experiment_service.experiments[experiment_id]

    # Convert runs
    runs = [RunDTO(**run) for run in exp_data.get("runs", [])]

    return ExperimentDTO(
        id=experiment_id,
        name=exp_data["name"],
        description=exp_data.get("description"),
        tags=exp_data.get("tags", []),
        created_at=exp_data["created_at"],
        runs=runs,
    )


@router.post("/{experiment_id}/runs")
async def log_run(
    experiment_id: str,
    detector_name: str,
    dataset_name: str,
    parameters: dict,
    metrics: dict,
    artifacts: dict | None = None,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> dict:
    """Log a run to an experiment."""
    experiment_service = container.experiment_tracking_service()

    try:
        run_id = await experiment_service.log_run(
            experiment_id=experiment_id,
            detector_name=detector_name,
            dataset_name=dataset_name,
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts,
        )

        return {"success": True, "run_id": run_id, "experiment_id": experiment_id}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to log run: {str(e)}")


@router.get("/{experiment_id}/compare")
async def compare_runs(
    experiment_id: str,
    metric: str = Query("f1", description="Metric to sort by"),
    run_ids: list[str] | None = Query(None),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> dict:
    """Compare runs within an experiment."""
    experiment_service = container.experiment_tracking_service()

    try:
        comparison_df = await experiment_service.compare_runs(
            experiment_id=experiment_id, run_ids=run_ids, metric=metric
        )

        # Convert DataFrame to dict for JSON response
        return {
            "experiment_id": experiment_id,
            "metric": metric,
            "comparison": comparison_df.to_dict(orient="records"),
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to compare runs: {str(e)}")


@router.get("/{experiment_id}/best")
async def get_best_run(
    experiment_id: str,
    metric: str = Query("f1", description="Metric to optimize"),
    higher_is_better: bool = Query(True),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> RunDTO:
    """Get the best run from an experiment."""
    experiment_service = container.experiment_tracking_service()

    try:
        best_run = await experiment_service.get_best_run(
            experiment_id=experiment_id,
            metric=metric,
            higher_is_better=higher_is_better,
        )

        return RunDTO(**best_run)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get best run: {str(e)}")


@router.get("/leaderboard", response_model=list[LeaderboardEntryDTO])
async def get_leaderboard(
    metric: str = Query("f1", description="Metric to rank by"),
    experiment_ids: list[str] | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> list[LeaderboardEntryDTO]:
    """Get leaderboard across experiments."""
    experiment_service = container.experiment_tracking_service()

    try:
        leaderboard_df = await experiment_service.create_leaderboard(
            experiment_ids=experiment_ids, metric=metric
        )

        # Convert to DTOs
        leaderboard_data = leaderboard_df.head(limit).to_dict(orient="records")

        return [
            LeaderboardEntryDTO(
                rank=entry["rank"],
                experiment=entry["experiment"],
                run_id=entry["run_id"],
                detector=entry["detector"],
                dataset=entry["dataset"],
                metric_value=entry[metric],
                metric_name=metric,
                timestamp=entry["timestamp"],
            )
            for entry in leaderboard_data
        ]

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create leaderboard: {str(e)}"
        )


@router.post("/{experiment_id}/export")
async def export_experiment(
    experiment_id: str,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> FileResponse:
    """Export experiment data."""
    experiment_service = container.experiment_tracking_service()
    settings = container.config()

    try:
        # Create export directory
        export_path = settings.temp_path / f"export_{experiment_id}"

        # Export experiment
        await experiment_service.export_experiment(
            experiment_id=experiment_id, export_path=export_path
        )

        # Create zip file
        import shutil

        zip_path = settings.temp_path / f"{experiment_id}.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", export_path)

        # Return file
        return FileResponse(
            path=zip_path,
            filename=f"experiment_{experiment_id}.zip",
            media_type="application/zip",
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to export experiment: {str(e)}"
        )
