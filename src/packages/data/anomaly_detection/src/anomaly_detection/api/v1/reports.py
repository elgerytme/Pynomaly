"""Report generation API endpoints."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Depends
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import structlog
import json
import uuid
import tempfile
import asyncio

from ...infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])


class ReportRequest(BaseModel):
    """Report generation request model."""
    report_type: str = Field(..., description="Type of report (detection, performance, batch, summary)")
    title: Optional[str] = Field(None, description="Custom report title")
    template: str = Field(default="standard", description="Report template (standard, executive, technical)")
    format: str = Field(default="html", description="Output format (html, json, pdf)")
    include_plots: bool = Field(default=True, description="Include visualization plots")
    data_sources: List[str] = Field(..., description="Data sources for the report")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional report parameters")


class DetectionReportRequest(ReportRequest):
    """Detection report specific request model."""
    detection_results: Dict[str, Any] = Field(..., description="Detection results data")
    evaluation_metrics: Optional[Dict[str, float]] = Field(None, description="Evaluation metrics if available")
    dataset_info: Dict[str, Any] = Field(..., description="Dataset information")


class PerformanceReportRequest(ReportRequest):
    """Performance report specific request model."""
    model_ids: List[str] = Field(..., description="Model IDs to include in report")
    time_range_days: int = Field(default=30, description="Time range for performance data")
    include_comparisons: bool = Field(default=True, description="Include model comparisons")


class BatchReportRequest(ReportRequest):
    """Batch report generation request model."""
    input_data_paths: List[str] = Field(..., description="Paths to input data files")
    output_directory: str = Field(..., description="Output directory for generated reports")


class ReportJob(BaseModel):
    """Report generation job model."""
    job_id: str
    status: str = Field(..., description="Job status (pending, running, completed, failed)")
    report_type: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100, description="Completion percentage")


class ReportMetadata(BaseModel):
    """Report metadata model."""
    report_id: str
    title: str
    report_type: str
    format: str
    template: str
    generated_at: datetime
    file_size_bytes: Optional[int] = None
    parameters: Dict[str, Any]


# In-memory storage for demonstration (in production, use proper database/queue)
_report_jobs: Dict[str, ReportJob] = {}
_generated_reports: Dict[str, ReportMetadata] = {}
_report_files: Dict[str, str] = {}  # report_id -> file_path


@router.post("/generate/detection")
async def generate_detection_report(
    request: DetectionReportRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Generate a detection report."""
    try:
        job_id = str(uuid.uuid4())
        
        # Create report job
        job = ReportJob(
            job_id=job_id,
            status="pending",
            report_type="detection",
            created_at=datetime.now()
        )
        
        _report_jobs[job_id] = job
        
        # Schedule background task
        background_tasks.add_task(
            _generate_detection_report_task,
            job_id,
            request
        )
        
        logger.info("Detection report generation started", job_id=job_id)
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Detection report generation started"
        }
        
    except Exception as e:
        logger.error("Failed to start detection report generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start report generation: {str(e)}")


@router.post("/generate/performance")
async def generate_performance_report(
    request: PerformanceReportRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Generate a performance monitoring report."""
    try:
        job_id = str(uuid.uuid4())
        
        job = ReportJob(
            job_id=job_id,
            status="pending",
            report_type="performance",
            created_at=datetime.now()
        )
        
        _report_jobs[job_id] = job
        
        background_tasks.add_task(
            _generate_performance_report_task,
            job_id,
            request
        )
        
        logger.info("Performance report generation started", job_id=job_id)
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Performance report generation started"
        }
        
    except Exception as e:
        logger.error("Failed to start performance report generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start report generation: {str(e)}")


@router.post("/generate/batch")
async def generate_batch_reports(
    request: BatchReportRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Generate batch reports for multiple data sources."""
    try:
        job_id = str(uuid.uuid4())
        
        job = ReportJob(
            job_id=job_id,
            status="pending",
            report_type="batch",
            created_at=datetime.now()
        )
        
        _report_jobs[job_id] = job
        
        background_tasks.add_task(
            _generate_batch_reports_task,
            job_id,
            request
        )
        
        logger.info("Batch report generation started", job_id=job_id, file_count=len(request.input_data_paths))
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"Batch report generation started for {len(request.input_data_paths)} files"
        }
        
    except Exception as e:
        logger.error("Failed to start batch report generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start batch report generation: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_report_job_status(job_id: str) -> ReportJob:
    """Get the status of a report generation job."""
    if job_id not in _report_jobs:
        raise HTTPException(status_code=404, detail=f"Report job {job_id} not found")
    
    return _report_jobs[job_id]


@router.get("/jobs")
async def list_report_jobs(
    status: Optional[str] = None,
    report_type: Optional[str] = None,
    limit: int = 50
) -> List[ReportJob]:
    """List report generation jobs with optional filtering."""
    jobs = list(_report_jobs.values())
    
    # Apply filters
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    if report_type:
        jobs = [job for job in jobs if job.report_type == report_type]
    
    # Sort by creation date (newest first) and apply limit
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return jobs[:limit]


@router.get("/download/{report_id}")
async def download_report(report_id: str) -> FileResponse:
    """Download a generated report."""
    if report_id not in _generated_reports:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    
    if report_id not in _report_files:
        raise HTTPException(status_code=404, detail=f"Report file {report_id} not available")
    
    report_metadata = _generated_reports[report_id]
    file_path = _report_files[report_id]
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Report file no longer exists")
    
    # Determine media type based on format
    media_type_map = {
        "html": "text/html",
        "json": "application/json",
        "pdf": "application/pdf",
        "csv": "text/csv"
    }
    
    media_type = media_type_map.get(report_metadata.format, "application/octet-stream")
    filename = f"{report_metadata.title.replace(' ', '_')}_{report_id}.{report_metadata.format}"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/view/{report_id}")
async def view_report(report_id: str) -> HTMLResponse:
    """View an HTML report in the browser."""
    if report_id not in _generated_reports:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    
    report_metadata = _generated_reports[report_id]
    
    if report_metadata.format != "html":
        raise HTTPException(status_code=400, detail="Only HTML reports can be viewed directly")
    
    if report_id not in _report_files:
        raise HTTPException(status_code=404, detail="Report file not available")
    
    file_path = _report_files[report_id]
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Report file no longer exists")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
        
    except Exception as e:
        logger.error("Failed to read report file", report_id=report_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to read report file")


@router.get("/list")
async def list_generated_reports(
    report_type: Optional[str] = None,
    format: Optional[str] = None,
    limit: int = 50
) -> List[ReportMetadata]:
    """List generated reports with optional filtering."""
    reports = list(_generated_reports.values())
    
    # Apply filters
    if report_type:
        reports = [report for report in reports if report.report_type == report_type]
    
    if format:
        reports = [report for report in reports if report.format == format]
    
    # Sort by generation date (newest first) and apply limit
    reports.sort(key=lambda x: x.generated_at, reverse=True)
    
    return reports[:limit]


@router.delete("/jobs/{job_id}")
async def cancel_report_job(job_id: str) -> Dict[str, str]:
    """Cancel a pending or running report job."""
    if job_id not in _report_jobs:
        raise HTTPException(status_code=404, detail=f"Report job {job_id} not found")
    
    job = _report_jobs[job_id]
    
    if job.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel {job.status} job")
    
    job.status = "cancelled"
    job.completed_at = datetime.now()
    
    logger.info("Report job cancelled", job_id=job_id)
    
    return {"message": f"Report job {job_id} cancelled successfully"}


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str) -> Dict[str, str]:
    """Delete a generated report."""
    if report_id not in _generated_reports:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    
    # Delete file if exists
    if report_id in _report_files:
        file_path = _report_files[report_id]
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to delete report file", file_path=file_path, error=str(e))
        
        del _report_files[report_id]
    
    # Remove from metadata
    del _generated_reports[report_id]
    
    logger.info("Report deleted", report_id=report_id)
    
    return {"message": f"Report {report_id} deleted successfully"}


# Background task functions

async def _generate_detection_report_task(job_id: str, request: DetectionReportRequest):
    """Background task to generate detection report."""
    job = _report_jobs[job_id]
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 10
        
        # Generate report content
        report_content = _create_detection_report_content(request)
        job.progress = 50
        
        # Create report file
        report_id, file_path = await _create_report_file(report_content, request.format, request.template)
        job.progress = 80
        
        # Store report metadata
        metadata = ReportMetadata(
            report_id=report_id,
            title=request.title or "Anomaly Detection Report",
            report_type="detection",
            format=request.format,
            template=request.template,
            generated_at=datetime.now(),
            file_size_bytes=Path(file_path).stat().st_size,
            parameters=request.parameters
        )
        
        _generated_reports[report_id] = metadata
        _report_files[report_id] = file_path
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.now()
        job.output_path = report_id
        job.progress = 100
        
        logger.info("Detection report generated successfully", job_id=job_id, report_id=report_id)
        
    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now()
        job.error_message = str(e)
        
        logger.error("Detection report generation failed", job_id=job_id, error=str(e))


async def _generate_performance_report_task(job_id: str, request: PerformanceReportRequest):
    """Background task to generate performance report."""
    job = _report_jobs[job_id]
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 10
        
        # Collect performance data
        performance_data = await _collect_performance_data(request.model_ids, request.time_range_days)
        job.progress = 40
        
        # Generate report content
        report_content = _create_performance_report_content(performance_data, request)
        job.progress = 70
        
        # Create report file
        report_id, file_path = await _create_report_file(report_content, request.format, request.template)
        job.progress = 90
        
        # Store report metadata
        metadata = ReportMetadata(
            report_id=report_id,
            title=request.title or "Performance Monitoring Report",
            report_type="performance",
            format=request.format,
            template=request.template,
            generated_at=datetime.now(),
            file_size_bytes=Path(file_path).stat().st_size,
            parameters=request.parameters
        )
        
        _generated_reports[report_id] = metadata
        _report_files[report_id] = file_path
        
        job.status = "completed"
        job.completed_at = datetime.now()
        job.output_path = report_id
        job.progress = 100
        
        logger.info("Performance report generated successfully", job_id=job_id, report_id=report_id)
        
    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now()
        job.error_message = str(e)
        
        logger.error("Performance report generation failed", job_id=job_id, error=str(e))


async def _generate_batch_reports_task(job_id: str, request: BatchReportRequest):
    """Background task to generate batch reports."""
    job = _report_jobs[job_id]
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 5
        
        generated_reports = []
        total_files = len(request.input_data_paths)
        
        for i, data_path in enumerate(request.input_data_paths):
            try:
                # Process each file
                # In a real implementation, this would load and process the actual data
                mock_detection_data = {
                    "input": data_path,
                    "algorithm": "isolation_forest",
                    "detection_results": {
                        "anomalies_detected": 42,
                        "total_samples": 1000,
                        "anomaly_rate": 0.042
                    }
                }
                
                # Create individual report
                report_content = _create_detection_report_content_from_dict(mock_detection_data)
                report_id, file_path = await _create_report_file(report_content, request.format, request.template)
                
                # Store individual report
                metadata = ReportMetadata(
                    report_id=report_id,
                    title=f"Batch Report - {Path(data_path).stem}",
                    report_type="detection",
                    format=request.format,
                    template=request.template,
                    generated_at=datetime.now(),
                    file_size_bytes=Path(file_path).stat().st_size,
                    parameters=request.parameters
                )
                
                _generated_reports[report_id] = metadata
                _report_files[report_id] = file_path
                generated_reports.append(report_id)
                
                # Update progress
                job.progress = int(5 + (i + 1) / total_files * 85)
                
            except Exception as e:
                logger.warning("Failed to process file in batch", file_path=data_path, error=str(e))
        
        # Create batch summary report
        summary_content = _create_batch_summary_content(generated_reports, request)
        summary_id, summary_path = await _create_report_file(summary_content, request.format, "summary")
        
        summary_metadata = ReportMetadata(
            report_id=summary_id,
            title="Batch Processing Summary",
            report_type="batch_summary",
            format=request.format,
            template="summary",
            generated_at=datetime.now(),
            file_size_bytes=Path(summary_path).stat().st_size,
            parameters={"individual_reports": generated_reports}
        )
        
        _generated_reports[summary_id] = summary_metadata
        _report_files[summary_id] = summary_path
        
        job.status = "completed"
        job.completed_at = datetime.now()
        job.output_path = summary_id
        job.progress = 100
        
        logger.info("Batch reports generated successfully", 
                   job_id=job_id, 
                   individual_reports=len(generated_reports),
                   summary_report=summary_id)
        
    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now()
        job.error_message = str(e)
        
        logger.error("Batch report generation failed", job_id=job_id, error=str(e))


# Helper functions

def _create_detection_report_content(request: DetectionReportRequest) -> Dict[str, Any]:
    """Create detection report content from request."""
    return {
        "title": request.title or "Anomaly Detection Report",
        "type": "detection",
        "template": request.template,
        "generated_at": datetime.now().isoformat(),
        "detection_results": request.detection_results,
        "evaluation_metrics": request.evaluation_metrics,
        "dataset_info": request.dataset_info,
        "parameters": request.parameters,
        "recommendations": _generate_detection_recommendations(request.detection_results, request.evaluation_metrics)
    }


def _create_detection_report_content_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create detection report content from data dictionary."""
    return {
        "title": f"Detection Report - {Path(data['input']).stem}",
        "type": "detection",
        "template": "standard",
        "generated_at": datetime.now().isoformat(),
        "detection_results": data.get("detection_results", {}),
        "evaluation_metrics": data.get("evaluation_metrics"),
        "dataset_info": data.get("dataset_info", {}),
        "parameters": {"input_file": data["input"], "algorithm": data.get("algorithm", "unknown")}
    }


def _create_performance_report_content(performance_data: Dict[str, Any], request: PerformanceReportRequest) -> Dict[str, Any]:
    """Create performance report content."""
    return {
        "title": request.title or "Performance Monitoring Report",
        "type": "performance",
        "template": request.template,
        "generated_at": datetime.now().isoformat(),
        "performance_data": performance_data,
        "model_ids": request.model_ids,
        "time_range_days": request.time_range_days,
        "include_comparisons": request.include_comparisons,
        "parameters": request.parameters
    }


def _create_batch_summary_content(report_ids: List[str], request: BatchReportRequest) -> Dict[str, Any]:
    """Create batch summary report content."""
    return {
        "title": "Batch Processing Summary",
        "type": "batch_summary",
        "template": "summary",
        "generated_at": datetime.now().isoformat(),
        "total_files_processed": len(request.input_data_paths),
        "successful_reports": len(report_ids),
        "success_rate": len(report_ids) / len(request.input_data_paths) * 100,
        "individual_reports": report_ids,
        "parameters": request.parameters
    }


async def _create_report_file(content: Dict[str, Any], format: str, template: str) -> tuple[str, str]:
    """Create report file and return report_id and file_path."""
    report_id = str(uuid.uuid4())
    
    # Create temporary file
    temp_dir = Path(tempfile.gettempdir()) / "anomaly_reports"
    temp_dir.mkdir(exist_ok=True)
    
    file_extension = format
    file_path = str(temp_dir / f"report_{report_id}.{file_extension}")
    
    if format == "json":
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2, default=str)
    
    elif format == "html":
        html_content = _generate_html_report(content, template)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    else:
        # For other formats, save as JSON for now
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2, default=str)
    
    return report_id, file_path


def _generate_html_report(content: Dict[str, Any], template: str) -> str:
    """Generate HTML report from content."""
    title = content.get("title", "Report")
    generated_at = content.get("generated_at", datetime.now().isoformat())
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {generated_at}</p>
        
        <h2>Report Content</h2>
        <pre style="background: #f8f9fa; padding: 20px; border-radius: 5px; overflow-x: auto;">
{json.dumps(content, indent=2, default=str)}
        </pre>
        
        <div class="footer">
            <p>Generated by Anomaly Detection System</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


async def _collect_performance_data(model_ids: List[str], time_range_days: int) -> Dict[str, Any]:
    """Collect performance data for models (mock implementation)."""
    # In a real implementation, this would query the actual monitoring system
    return {
        "models": model_ids,
        "time_range_days": time_range_days,
        "collected_at": datetime.now().isoformat(),
        "performance_summary": {
            "total_models": len(model_ids),
            "average_accuracy": 0.85,
            "models_with_alerts": 1
        }
    }


def _generate_detection_recommendations(detection_results: Dict[str, Any], evaluation_metrics: Optional[Dict[str, float]]) -> List[str]:
    """Generate recommendations based on detection results."""
    recommendations = []
    
    anomaly_rate = detection_results.get("anomaly_rate", 0)
    
    if anomaly_rate > 0.2:
        recommendations.append("High anomaly rate detected. Consider reviewing data quality or detection parameters.")
    elif anomaly_rate < 0.01:
        recommendations.append("Very low anomaly rate. Consider increasing sensitivity or reviewing detection criteria.")
    
    if evaluation_metrics:
        if evaluation_metrics.get("precision", 1.0) < 0.7:
            recommendations.append("Low precision indicates many false positives. Consider tightening detection criteria.")
        if evaluation_metrics.get("recall", 1.0) < 0.7:
            recommendations.append("Low recall indicates missed anomalies. Consider loosening detection criteria.")
    
    if not recommendations:
        recommendations.append("Detection results appear nominal. Continue monitoring for performance drift.")
    
    return recommendations