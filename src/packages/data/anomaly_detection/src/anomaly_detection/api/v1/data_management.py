"""Data management API endpoints for file upload, validation, and processing."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ...domain.services.batch_processing_service import BatchProcessingService
from ...domain.services.data_validation_service import DataValidationService
from ...domain.services.data_conversion_service import DataConversionService
from ...domain.services.data_profiling_service import DataProfilingService
from ...domain.services.data_sampling_service import DataSamplingService
from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.in_memory_model_repository import InMemoryModelRepository

logger = structlog.get_logger(__name__)

router = APIRouter()

# Background task storage for tracking long-running operations
background_tasks_status = {}

# Pydantic models for request/response
class FileUploadResponse(BaseModel):
    """Response model for file upload."""
    file_id: str
    filename: str
    size_bytes: int
    content_type: Optional[str]
    upload_path: str
    message: str

class ValidationRequest(BaseModel):
    """Request model for data validation."""
    file_paths: List[str]
    schema_file: Optional[str] = None
    check_types: bool = True
    check_missing: bool = True
    check_outliers: bool = True
    check_duplicates: bool = True
    custom_rules: Optional[Dict[str, Any]] = None

class ConversionRequest(BaseModel):
    """Request model for data conversion."""
    file_paths: List[str]
    output_format: str = Field(..., description="Target format: csv, json, parquet, excel, hdf5, pickle")
    compression: Optional[str] = Field(None, description="Compression: gzip, bz2, xz")
    chunk_size: int = 10000
    preserve_dtypes: bool = True
    conversion_options: Optional[Dict[str, Any]] = None

class ProfilingRequest(BaseModel):
    """Request model for data profiling."""
    file_paths: List[str]
    include_correlations: bool = True
    include_distributions: bool = True
    generate_plots: bool = False
    sample_size: Optional[int] = None
    sections: Optional[List[str]] = None

class SamplingRequest(BaseModel):
    """Request model for data sampling."""
    file_paths: List[str]
    sample_size: int
    method: str = Field(default="random", description="Sampling method: random, systematic, stratified, cluster, reservoir")
    stratify_column: Optional[str] = None
    cluster_column: Optional[str] = None
    seed: Optional[int] = None
    replacement: bool = False

class BatchDetectionRequest(BaseModel):
    """Request model for batch anomaly detection."""
    file_paths: List[str]
    algorithms: List[str] = ["isolation_forest"]
    output_format: str = "json"
    save_models: bool = False
    model_name_prefix: Optional[str] = None
    chunk_size: int = 1000
    parallel_jobs: int = 4

class TaskStatusResponse(BaseModel):
    """Response model for background task status."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

# Dependency to get services
def get_batch_processing_service():
    """Get batch processing service instance."""
    detection_service = DetectionService()
    model_repository = InMemoryModelRepository()
    return BatchProcessingService(detection_service, model_repository)

def get_validation_service():
    """Get data validation service instance."""
    return DataValidationService()

def get_conversion_service():
    """Get data conversion service instance."""
    return DataConversionService()

def get_profiling_service():
    """Get data profiling service instance."""
    return DataProfilingService()

def get_sampling_service():
    """Get data sampling service instance."""
    return DataSamplingService()

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> FileUploadResponse:
    """
    Upload a data file for processing.
    
    Supports: CSV, JSON, Parquet, Excel files
    Maximum file size: 100MB
    """
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.json', '.parquet', '.xlsx', '.xls'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (100MB limit)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        file_content = await file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(file_content)} bytes. Maximum: {MAX_FILE_SIZE} bytes"
            )
        
        # Generate unique file ID and save file
        file_id = str(uuid4())
        upload_dir = Path("/tmp/anomaly_detection_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info("File uploaded successfully",
                   file_id=file_id,
                   filename=file.filename,
                   size_bytes=len(file_content),
                   content_type=file.content_type)
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            size_bytes=len(file_content),
            content_type=file.content_type,
            upload_path=str(file_path),
            message="File uploaded successfully"
        )
        
    except Exception as e:
        logger.error("File upload failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/validate")
async def validate_files(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    validation_service: DataValidationService = Depends(get_validation_service)
) -> Dict[str, Any]:
    """
    Validate data files for quality and schema compliance.
    
    Performs comprehensive validation including:
    - Schema validation (if schema provided)
    - Data type consistency checks
    - Missing value analysis
    - Duplicate detection
    - Outlier identification
    - Data consistency checks
    """
    try:
        # Convert string paths to Path objects and validate existence
        file_paths = []
        for path_str in request.file_paths:
            path = Path(path_str)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
            file_paths.append(path)
        
        # Convert schema file path if provided
        schema_file = Path(request.schema_file) if request.schema_file else None
        if schema_file and not schema_file.exists():
            raise HTTPException(status_code=404, detail=f"Schema file not found: {request.schema_file}")
        
        # For single file, run synchronously
        if len(file_paths) == 1:
            result = await validation_service.validate_file(
                file_path=file_paths[0],
                schema_file=schema_file,
                check_types=request.check_types,
                check_missing=request.check_missing,
                check_outliers=request.check_outliers,
                check_duplicates=request.check_duplicates,
                custom_rules=request.custom_rules
            )
            return {"validation_result": result}
        
        # For multiple files, use batch validation
        result = await validation_service.validate_multiple_files(
            file_paths=file_paths,
            schema_file=schema_file,
            check_types=request.check_types,
            check_missing=request.check_missing,
            check_outliers=request.check_outliers,
            check_duplicates=request.check_duplicates,
            custom_rules=request.custom_rules
        )
        
        return {"batch_validation_result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/convert")
async def convert_files(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    conversion_service: DataConversionService = Depends(get_conversion_service)
) -> Dict[str, Any]:
    """
    Convert data files between different formats.
    
    Supported formats: CSV, JSON, Parquet, Excel, HDF5, Pickle
    Supported compressions: gzip, bz2, xz
    """
    try:
        # Validate file paths
        file_paths = []
        for path_str in request.file_paths:
            path = Path(path_str)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
            file_paths.append(path)
        
        # Validate output format
        supported_formats = conversion_service.get_supported_formats()
        if request.output_format not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {request.output_format}. Supported: {list(supported_formats.keys())}"
            )
        
        # Validate compression if specified
        if request.compression:
            supported_compressions = conversion_service.get_supported_compressions()
            if request.compression not in supported_compressions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported compression: {request.compression}. Supported: {list(supported_compressions.keys())}"
                )
        
        # Create output directory
        output_dir = Path("/tmp/anomaly_detection_conversions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert files
        if len(file_paths) == 1:
            # Single file conversion
            output_file = await conversion_service.convert_file(
                input_file=file_paths[0],
                output_format=request.output_format,
                output_dir=output_dir,
                compression=request.compression,
                chunk_size=request.chunk_size,
                preserve_dtypes=request.preserve_dtypes,
                conversion_options=request.conversion_options
            )
            
            return {
                "conversion_result": {
                    "input_file": str(file_paths[0]),
                    "output_file": str(output_file),
                    "format": request.output_format,
                    "compression": request.compression
                }
            }
        else:
            # Batch conversion
            result = await conversion_service.batch_convert(
                input_files=file_paths,
                output_format=request.output_format,
                output_dir=output_dir,
                compression=request.compression,
                chunk_size=request.chunk_size,
                preserve_dtypes=request.preserve_dtypes,
                conversion_options=request.conversion_options
            )
            
            return {"batch_conversion_result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data conversion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@router.post("/profile")
async def profile_files(
    request: ProfilingRequest,
    background_tasks: BackgroundTasks,
    profiling_service: DataProfilingService = Depends(get_profiling_service)
) -> Dict[str, Any]:
    """
    Generate comprehensive data profiles for uploaded files.
    
    Includes:
    - Dataset information (rows, columns, memory usage)
    - Column analysis (types, null values, uniqueness)
    - Data quality metrics (missing data, duplicates)
    - Statistical summaries
    - Correlation analysis
    - Distribution analysis
    - Pattern detection
    """
    try:
        # Validate file paths
        file_paths = []
        for path_str in request.file_paths:
            path = Path(path_str)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
            file_paths.append(path)
        
        # Profile files (single or multiple)
        if len(file_paths) == 1:
            profile = await profiling_service.profile_file(
                file_path=file_paths[0],
                include_correlations=request.include_correlations,
                include_distributions=request.include_distributions,
                generate_plots=request.generate_plots,
                sample_size=request.sample_size,
                sections=request.sections
            )
            
            return {"profile_result": profile}
        else:
            # Profile multiple files
            profiles = []
            for file_path in file_paths:
                try:
                    profile = await profiling_service.profile_file(
                        file_path=file_path,
                        include_correlations=request.include_correlations,
                        include_distributions=request.include_distributions,
                        generate_plots=request.generate_plots,
                        sample_size=request.sample_size,
                        sections=request.sections
                    )
                    profiles.append(profile)
                except Exception as e:
                    logger.error("File profiling failed", file=str(file_path), error=str(e))
                    profiles.append({
                        "error": f"Profiling failed: {str(e)}",
                        "file_path": str(file_path)
                    })
            
            return {"batch_profile_result": {"profiles": profiles}}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data profiling failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@router.post("/sample")
async def sample_files(
    request: SamplingRequest,
    background_tasks: BackgroundTasks,
    sampling_service: DataSamplingService = Depends(get_sampling_service)
) -> Dict[str, Any]:
    """
    Generate statistical samples from data files.
    
    Supported sampling methods:
    - random: Simple random sampling
    - systematic: Systematic sampling with regular intervals
    - stratified: Stratified sampling based on a column
    - cluster: Cluster sampling based on groups
    - reservoir: Reservoir sampling for streaming data
    """
    try:
        # Validate file paths
        file_paths = []
        for path_str in request.file_paths:
            path = Path(path_str)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
            file_paths.append(path)
        
        # Validate sampling method
        available_methods = sampling_service.get_sampling_methods_info()
        if request.method not in available_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported sampling method: {request.method}. Available: {list(available_methods.keys())}"
            )
        
        # Create output directory for samples
        output_dir = Path("/tmp/anomaly_detection_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample files
        sampling_options = {
            'stratify_column': request.stratify_column,
            'cluster_column': request.cluster_column,
            'seed': request.seed,
            'replacement': request.replacement
        }
        
        # Remove None values from sampling options
        sampling_options = {k: v for k, v in sampling_options.items() if v is not None}
        
        if len(file_paths) == 1:
            # Single file sampling
            sample_data = await sampling_service.sample_file(
                file_path=file_paths[0],
                sample_size=request.sample_size,
                method=request.method,
                **sampling_options
            )
            
            # Save sample to file
            output_file = output_dir / f"{file_paths[0].stem}_sample_{request.method}_{request.sample_size}.csv"
            sample_data.to_csv(output_file, index=False)
            
            return {
                "sampling_result": {
                    "input_file": str(file_paths[0]),
                    "output_file": str(output_file),
                    "method": request.method,
                    "original_size": len(sample_data) + request.sample_size,  # Approximation
                    "sample_size": len(sample_data)
                }
            }
        else:
            # Multiple file sampling
            samples = await sampling_service.sample_multiple_files(
                file_paths=file_paths,
                sample_size=request.sample_size,
                method=request.method,
                output_dir=output_dir,
                **sampling_options
            )
            
            results = []
            for i, (file_path, sample_data) in enumerate(zip(file_paths, samples)):
                if not sample_data.empty:
                    output_file = output_dir / f"{file_path.stem}_sample_{request.method}_{request.sample_size}.csv"
                    results.append({
                        "input_file": str(file_path),
                        "output_file": str(output_file),
                        "method": request.method,
                        "sample_size": len(sample_data),
                        "success": True
                    })
                else:
                    results.append({
                        "input_file": str(file_path),
                        "output_file": None,
                        "method": request.method,
                        "sample_size": 0,
                        "success": False,
                        "error": "Sampling failed"
                    })
            
            return {"batch_sampling_result": {"results": results}}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data sampling failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Sampling failed: {str(e)}")

@router.post("/batch-detect")
async def batch_detect_anomalies(
    request: BatchDetectionRequest,
    background_tasks: BackgroundTasks,
    batch_service: BatchProcessingService = Depends(get_batch_processing_service)
) -> Dict[str, Any]:
    """
    Run batch anomaly detection on multiple files.
    
    Supports multiple algorithms:
    - isolation_forest
    - local_outlier_factor  
    - one_class_svm
    - elliptic_envelope
    
    Results can be saved in JSON or CSV format.
    """
    try:
        # Validate file paths
        file_paths = []
        for path_str in request.file_paths:
            path = Path(path_str)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
            file_paths.append(path)
        
        # Create output directory
        output_dir = Path("/tmp/anomaly_detection_batch_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run batch detection
        results = await batch_service.batch_detect_anomalies(
            file_paths=file_paths,
            output_dir=output_dir,
            algorithms=request.algorithms,
            output_format=request.output_format,
            save_models=request.save_models,
            model_name_prefix=request.model_name_prefix,
            chunk_size=request.chunk_size,
            parallel_jobs=request.parallel_jobs
        )
        
        return {"batch_detection_result": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch anomaly detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@router.get("/download/{file_path:path}")
async def download_file(file_path: str) -> FileResponse:
    """
    Download a processed file.
    
    Security note: Only files in the temp processing directories are accessible.
    """
    try:
        # Validate file path is in allowed directories
        allowed_dirs = {
            "/tmp/anomaly_detection_uploads",
            "/tmp/anomaly_detection_conversions", 
            "/tmp/anomaly_detection_samples",
            "/tmp/anomaly_detection_batch_results"
        }
        
        full_path = Path(file_path)
        
        # Check if path is under allowed directories
        path_allowed = any(
            str(full_path).startswith(allowed_dir) 
            for allowed_dir in allowed_dirs
        )
        
        if not path_allowed:
            raise HTTPException(status_code=403, detail="Access to this file path is not allowed")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File download failed", file_path=file_path, error=str(e))
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/formats/supported")
async def get_supported_formats(
    conversion_service: DataConversionService = Depends(get_conversion_service)
) -> Dict[str, Any]:
    """Get information about supported file formats and compression methods."""
    return {
        "supported_formats": conversion_service.get_supported_formats(),
        "supported_compressions": conversion_service.get_supported_compressions()
    }

@router.get("/sampling/methods")
async def get_sampling_methods(
    sampling_service: DataSamplingService = Depends(get_sampling_service)
) -> Dict[str, Any]:
    """Get information about available sampling methods."""
    return {
        "sampling_methods": sampling_service.get_sampling_methods_info()
    }

@router.delete("/cleanup")
async def cleanup_temp_files() -> Dict[str, str]:
    """
    Clean up temporary files older than 24 hours.
    
    This endpoint helps manage disk space by removing old processed files.
    """
    try:
        import time
        import shutil
        
        temp_dirs = [
            Path("/tmp/anomaly_detection_uploads"),
            Path("/tmp/anomaly_detection_conversions"),
            Path("/tmp/anomaly_detection_samples"), 
            Path("/tmp/anomaly_detection_batch_results")
        ]
        
        cleaned_files = 0
        cleaned_size = 0
        
        # Clean files older than 24 hours
        cutoff_time = time.time() - (24 * 60 * 60)
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        if file_path.stat().st_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleaned_files += 1
                            cleaned_size += file_size
        
        logger.info("Temp file cleanup completed",
                   cleaned_files=cleaned_files,
                   cleaned_size_mb=cleaned_size / (1024 * 1024))
        
        return {
            "message": f"Cleanup completed: {cleaned_files} files removed, {cleaned_size / (1024 * 1024):.2f} MB freed"
        }
        
    except Exception as e:
        logger.error("Temp file cleanup failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")