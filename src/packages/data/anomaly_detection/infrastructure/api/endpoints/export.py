"""
API endpoints for business intelligence export functionality.
"""

import tempfile
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ....application.dto.export_options import ExportFormat, ExportOptions
from ....application.services.export_service import ExportService
from ....domain.entities.detection_result import DetectionResult
from ..deps import get_container

router = APIRouter(prefix="/export", tags=["export"])


class ExportFormatAPI(str, Enum):
    """API enum for export formats (core only)."""

    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class ExportRequest(BaseModel):
    """Request model for export operations."""

    result_id: UUID = Field(..., description="Detection result ID to export")
    format: ExportFormatAPI = Field(..., description="Export format")
    filename: str | None = Field(
        None, description="Output filename (for file-based exports)"
    )

    # Format-specific options
    include_charts: bool = Field(True, description="Include charts and visualizations")
    advanced_formatting: bool = Field(True, description="Use advanced formatting")
    highlight_anomalies: bool = Field(True, description="Highlight anomalous samples")

    # JSON options
    json_indent: int | None = Field(2, description="JSON indentation level")

    # Parquet options
    parquet_compression: str = Field(
        "snappy", description="Parquet compression algorithm"
    )

    # General options
    notify_on_completion: bool = Field(
        False, description="Send notification when complete"
    )
    notification_emails: list[str] | None = Field(
        None, description="Notification email addresses"
    )


class MultiExportRequest(BaseModel):
    """Request model for multi-format export."""

    result_id: UUID = Field(..., description="Detection result ID to export")
    formats: list[ExportFormatAPI] = Field(..., description="Export formats")
    base_filename: str = Field(
        "anomaly_export", description="Base filename for outputs"
    )

    # Format-specific configurations
    excel_options: dict[str, Any] | None = Field(
        None, description="Excel-specific options"
    )
    powerbi_options: dict[str, Any] | None = Field(
        None, description="Power BI-specific options"
    )
    gsheets_options: dict[str, Any] | None = Field(
        None, description="Google Sheets-specific options"
    )
    smartsheet_options: dict[str, Any] | None = Field(
        None, description="Smartsheet-specific options"
    )


class ExportStatusResponse(BaseModel):
    """Response model for export status."""

    export_id: str = Field(..., description="Export operation ID")
    status: str = Field(
        ..., description="Export status: pending, running, completed, failed"
    )
    format: str = Field(..., description="Export format")
    progress: float = Field(..., description="Progress percentage (0-100)")
    message: str | None = Field(None, description="Status message")
    result: dict[str, Any] | None = Field(
        None, description="Export result when completed"
    )
    error: str | None = Field(None, description="Error message if failed")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: str | None = Field(None, description="Completion timestamp")


class SupportedFormatsResponse(BaseModel):
    """Response model for supported formats."""

    total_formats: int = Field(..., description="Total number of supported formats")
    supported_formats: list[str] = Field(
        ..., description="List of supported format names"
    )
    adapters: dict[str, dict[str, Any]] = Field(..., description="Adapter information")


# Store for tracking export operations (in production, use Redis or database)
export_operations: dict[str, dict[str, Any]] = {}


@router.get("/formats", response_model=SupportedFormatsResponse)
async def get_supported_formats(
    container=Depends(get_container),
) -> SupportedFormatsResponse:
    """Get list of supported export formats and their capabilities."""
    try:
        export_service = ExportService()
        stats = export_service.get_export_statistics()

        return SupportedFormatsResponse(
            total_formats=stats["total_formats"],
            supported_formats=stats["supported_formats"],
            adapters=stats["adapters"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get supported formats: {e}"
        )


@router.post("/validate")
async def validate_export_request(
    request: ExportRequest, container=Depends(get_container)
) -> dict[str, Any]:
    """Validate export request configuration."""
    try:
        export_service = ExportService()

        # Map API format to enum
        format_mapping = {
            ExportFormatAPI.EXCEL: ExportFormat.EXCEL,
            ExportFormatAPI.POWERBI: ExportFormat.POWERBI,
            ExportFormatAPI.GSHEETS: ExportFormat.GSHEETS,
            ExportFormatAPI.SMARTSHEET: ExportFormat.SMARTSHEET,
        }

        export_format = format_mapping[request.format]

        # Create export options
        options = _create_export_options(request, export_format)

        # Validate
        file_path = request.filename or f"export.{request.format.value}"
        validation = export_service.validate_export_request(
            export_format, file_path, options
        )

        return validation

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {e}")


@router.post("/start", response_model=ExportStatusResponse)
async def start_export(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    container=Depends(get_container),
) -> ExportStatusResponse:
    """Start an export operation."""
    try:
        # Verify result exists
        result_repo = container.result_repository()
        detection_result = result_repo.find_by_id(request.result_id)

        if not detection_result:
            raise HTTPException(status_code=404, detail="Detection result not found")

        # Generate export ID
        import uuid
        from datetime import datetime

        export_id = str(uuid.uuid4())

        # Create export operation record
        export_operations[export_id] = {
            "export_id": export_id,
            "status": "pending",
            "format": request.format.value,
            "progress": 0.0,
            "message": "Export queued",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }

        # Start background export task
        background_tasks.add_task(
            _perform_export, export_id, request, detection_result, container
        )

        return ExportStatusResponse(**export_operations[export_id])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start export: {e}")


@router.get("/status/{export_id}", response_model=ExportStatusResponse)
async def get_export_status(export_id: str) -> ExportStatusResponse:
    """Get status of an export operation."""
    if export_id not in export_operations:
        raise HTTPException(status_code=404, detail="Export operation not found")

    return ExportStatusResponse(**export_operations[export_id])


@router.get("/download/{export_id}")
async def download_export_file(export_id: str):
    """Download exported file (for file-based exports)."""
    if export_id not in export_operations:
        raise HTTPException(status_code=404, detail="Export operation not found")

    operation = export_operations[export_id]

    if operation["status"] != "completed":
        raise HTTPException(status_code=400, detail="Export not completed")

    if not operation.get("result", {}).get("file_path"):
        raise HTTPException(status_code=400, detail="No file available for download")

    file_path = operation["result"]["file_path"]

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream",
    )


@router.post("/multi", response_model=list[ExportStatusResponse])
async def start_multi_export(
    request: MultiExportRequest,
    background_tasks: BackgroundTasks,
    container=Depends(get_container),
) -> list[ExportStatusResponse]:
    """Start multiple export operations simultaneously."""
    try:
        # Verify result exists
        result_repo = container.result_repository()
        detection_result = result_repo.find_by_id(request.result_id)

        if not detection_result:
            raise HTTPException(status_code=404, detail="Detection result not found")

        responses = []

        for format_api in request.formats:
            # Create individual export request
            export_request = ExportRequest(
                result_id=request.result_id,
                format=format_api,
                filename=f"{request.base_filename}.{format_api.value}",
            )

            # Apply format-specific options
            if format_api == ExportFormatAPI.EXCEL and request.excel_options:
                for key, value in request.excel_options.items():
                    setattr(export_request, key, value)
            elif format_api == ExportFormatAPI.POWERBI and request.powerbi_options:
                for key, value in request.powerbi_options.items():
                    setattr(export_request, key, value)
            elif format_api == ExportFormatAPI.GSHEETS and request.gsheets_options:
                for key, value in request.gsheets_options.items():
                    setattr(export_request, key, value)
            elif (
                format_api == ExportFormatAPI.SMARTSHEET and request.smartsheet_options
            ):
                for key, value in request.smartsheet_options.items():
                    setattr(export_request, key, value)

            # Start individual export
            response = await start_export(export_request, background_tasks, container)
            responses.append(response)

        return responses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start multi-export: {e}"
        )


@router.get("/history")
async def get_export_history(
    limit: int = 50, format_filter: ExportFormatAPI | None = None
) -> list[ExportStatusResponse]:
    """Get export operation history."""
    operations = list(export_operations.values())

    # Filter by format if specified
    if format_filter:
        operations = [op for op in operations if op["format"] == format_filter.value]

    # Sort by creation time (newest first)
    operations.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply limit
    operations = operations[:limit]

    return [ExportStatusResponse(**op) for op in operations]


@router.delete("/cancel/{export_id}")
async def cancel_export(export_id: str) -> dict[str, str]:
    """Cancel a pending export operation."""
    if export_id not in export_operations:
        raise HTTPException(status_code=404, detail="Export operation not found")

    operation = export_operations[export_id]

    if operation["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed operation")

    # Update status
    operation["status"] = "cancelled"
    operation["message"] = "Export cancelled by user"

    return {"message": "Export operation cancelled"}


@router.get("/templates/{format}")
async def get_export_templates(format: ExportFormatAPI) -> dict[str, Any]:
    """Get template configurations for specific export format."""
    templates = {
        ExportFormatAPI.EXCEL: {
            "basic": {
                "include_charts": True,
                "advanced_formatting": True,
                "highlight_anomalies": True,
            },
            "simple": {
                "include_charts": False,
                "advanced_formatting": False,
                "highlight_anomalies": True,
            },
            "full": {
                "include_charts": True,
                "advanced_formatting": True,
                "highlight_anomalies": True,
                "create_multiple_sheets": True,
                "add_conditional_formatting": True,
            },
        },
        ExportFormatAPI.POWERBI: {
            "streaming": {"streaming_dataset": True, "table_name": "AnomalyResults"},
            "batch": {"streaming_dataset": False, "table_name": "AnomalyAnalysis"},
        },
        ExportFormatAPI.GSHEETS: {
            "collaborative": {"permissions": "edit", "include_charts": True},
            "readonly": {"permissions": "view", "include_charts": False},
        },
        ExportFormatAPI.SMARTSHEET: {
            "project": {
                "workspace_name": "Anomaly Detection",
                "access_level": "EDITOR",
            },
            "readonly": {"access_level": "VIEWER"},
        },
    }

    if format not in templates:
        raise HTTPException(status_code=404, detail="Templates not found for format")

    return templates[format]


# Helper functions


def _create_export_options(
    request: ExportRequest, export_format: ExportFormat
) -> ExportOptions:
    """Create export options from API request."""
    options = ExportOptions(format=export_format)

    # Common options
    options.include_charts = request.include_charts
    options.notify_on_completion = request.notify_on_completion
    options.notification_emails = request.notification_emails or []

    # Format-specific options
    if export_format == ExportFormat.EXCEL:
        options.use_advanced_formatting = request.advanced_formatting
        options.highlight_anomalies = request.highlight_anomalies

    elif export_format == ExportFormat.POWERBI:
        options.workspace_id = request.workspace_id
        options.dataset_name = request.dataset_name
        options.streaming_dataset = request.streaming_dataset
        options.table_name = request.table_name

    elif export_format == ExportFormat.GSHEETS:
        options.spreadsheet_id = request.spreadsheet_id
        options.share_with_emails = request.share_with_emails or []
        options.permissions = request.permissions

    elif export_format == ExportFormat.SMARTSHEET:
        options.workspace_name = request.workspace_name
        options.folder_id = request.folder_id
        options.sheet_template_id = request.template_id

    return options


async def _perform_export(
    export_id: str, request: ExportRequest, detection_result: DetectionResult, container
):
    """Perform the actual export operation in background."""
    from datetime import datetime

    try:
        # Update status
        export_operations[export_id]["status"] = "running"
        export_operations[export_id]["message"] = "Starting export..."
        export_operations[export_id]["progress"] = 10.0

        # Map format
        format_mapping = {
            ExportFormatAPI.EXCEL: ExportFormat.EXCEL,
            ExportFormatAPI.POWERBI: ExportFormat.POWERBI,
            ExportFormatAPI.GSHEETS: ExportFormat.GSHEETS,
            ExportFormatAPI.SMARTSHEET: ExportFormat.SMARTSHEET,
        }

        export_format = format_mapping[request.format]
        options = _create_export_options(request, export_format)

        # Update progress
        export_operations[export_id]["progress"] = 30.0
        export_operations[export_id]["message"] = "Configuring export..."

        # Initialize export service
        export_service = ExportService()

        # Update progress
        export_operations[export_id]["progress"] = 50.0
        export_operations[export_id]["message"] = "Exporting data..."

        # Determine output path
        if request.format == ExportFormatAPI.EXCEL:
            # For file-based exports, use temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f".{request.format.value}", delete=False
            ) as tmp_file:
                output_path = tmp_file.name
        else:
            # For cloud services, no file path needed
            output_path = ""

        # Perform export
        result = export_service.export_results(detection_result, output_path, options)

        # Update progress
        export_operations[export_id]["progress"] = 90.0
        export_operations[export_id]["message"] = "Finalizing export..."

        # Complete
        export_operations[export_id]["status"] = "completed"
        export_operations[export_id]["progress"] = 100.0
        export_operations[export_id]["message"] = "Export completed successfully"
        export_operations[export_id]["result"] = result
        export_operations[export_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        # Handle failure
        export_operations[export_id]["status"] = "failed"
        export_operations[export_id]["error"] = str(e)
        export_operations[export_id]["message"] = f"Export failed: {e}"
        export_operations[export_id]["completed_at"] = datetime.now().isoformat()
