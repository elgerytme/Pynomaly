"""API endpoints for data profiling service."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import logging
import tempfile
import os
from datetime import datetime
import asyncio

from ...application.use_cases.profile_dataset import ProfileDatasetUseCase
from ...application.services.performance_optimizer import PerformanceOptimizer
from ...infrastructure.adapters.data_source_adapter import DataSourceFactory, DataSourceType
from ...infrastructure.adapters.cloud_storage_adapter import get_cloud_storage_adapter
from ...infrastructure.adapters.streaming_adapter import get_streaming_adapter
from ...infrastructure.adapters.nosql_adapter import get_nosql_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profiling", tags=["data-profiling"])

# Background task storage (in production, use a proper task queue like Celery)
background_tasks_storage = {}


class FileProfilingRequest(BaseModel):
    """Request model for file profiling."""
    file_path: str = Field(..., description="Path to the file to profile")
    profiling_strategy: str = Field("full", description="Profiling strategy: full, sample, or adaptive")
    sample_size: Optional[int] = Field(None, description="Sample size for sampling strategies")
    sample_percentage: Optional[float] = Field(None, description="Sample percentage for percentage sampling")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional file options")


class DatabaseProfilingRequest(BaseModel):
    """Request model for database profiling."""
    db_type: str = Field(..., description="Database type: postgresql, mysql, sqlite")
    connection: Dict[str, Any] = Field(..., description="Database connection parameters")
    table_name: Optional[str] = Field(None, description="Specific table to profile")
    query: Optional[str] = Field(None, description="Custom SQL query")
    profiling_strategy: str = Field("full", description="Profiling strategy")
    sample_size: Optional[int] = Field(None, description="Sample size for sampling")


class CloudStorageProfilingRequest(BaseModel):
    """Request model for cloud storage profiling."""
    provider: str = Field(..., description="Cloud provider: s3, azure, gcs")
    connection: Dict[str, Any] = Field(..., description="Cloud storage connection parameters")
    object_key: str = Field(..., description="Object key/path to profile")
    profiling_strategy: str = Field("full", description="Profiling strategy")


class StreamingProfilingRequest(BaseModel):
    """Request model for streaming data profiling."""
    provider: str = Field(..., description="Streaming provider: kafka, kinesis")
    connection: Dict[str, Any] = Field(..., description="Streaming connection parameters")
    stream_name: str = Field(..., description="Stream/topic name")
    sample_size: int = Field(1000, description="Number of messages to sample")
    timeout_seconds: int = Field(30, description="Timeout for message consumption")


class NoSQLProfilingRequest(BaseModel):
    """Request model for NoSQL database profiling."""
    db_type: str = Field(..., description="NoSQL database type: mongodb, cassandra, dynamodb")
    connection: Dict[str, Any] = Field(..., description="NoSQL connection parameters")
    collection_name: str = Field(..., description="Collection/table name")
    query: Optional[Dict[str, Any]] = Field(None, description="Query filter (MongoDB style)")
    sample_size: Optional[int] = Field(None, description="Sample size")


class ProfilingResponse(BaseModel):
    """Response model for profiling operations."""
    success: bool
    task_id: Optional[str] = None
    profile_id: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/file", response_model=ProfilingResponse)
async def profile_file(request: FileProfilingRequest, background_tasks: BackgroundTasks):
    """Profile a file dataset."""
    try:
        use_case = ProfileDatasetUseCase()
        
        if request.profiling_strategy == "sample" and request.sample_size:
            profile = use_case.execute_sample(request.file_path, request.sample_size)
        elif request.profiling_strategy == "sample" and request.sample_percentage:
            profile = use_case.execute_percentage_sample(request.file_path, request.sample_percentage)
        else:
            profile = use_case.execute(request.file_path, request.profiling_strategy)
        
        # Get summary
        summary = use_case.get_profiling_summary(profile)
        
        return ProfilingResponse(
            success=True,
            profile_id=str(profile.profile_id.value),
            message="File profiling completed successfully",
            data=summary
        )
        
    except Exception as e:
        logger.error(f"File profiling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file/upload", response_model=ProfilingResponse)
async def profile_uploaded_file(
    file: UploadFile = File(...),
    profiling_strategy: str = Query("full", description="Profiling strategy"),
    sample_size: Optional[int] = Query(None, description="Sample size")
):
    """Profile an uploaded file."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            use_case = ProfileDatasetUseCase()
            
            if profiling_strategy == "sample" and sample_size:
                profile = use_case.execute_sample(temp_path, sample_size)
            else:
                profile = use_case.execute(temp_path, profiling_strategy)
            
            summary = use_case.get_profiling_summary(profile)
            
            return ProfilingResponse(
                success=True,
                profile_id=str(profile.profile_id.value),
                message=f"Uploaded file '{file.filename}' profiling completed successfully",
                data=summary
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"Uploaded file profiling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database", response_model=ProfilingResponse)
async def profile_database(request: DatabaseProfilingRequest, background_tasks: BackgroundTasks):
    """Profile a database table or query."""
    try:
        # Create data source adapter
        source = DataSourceFactory.create_database_source(request.db_type, request.connection)
        
        # Load data
        load_kwargs = {}
        if request.table_name:
            load_kwargs['table_name'] = request.table_name
        elif request.query:
            load_kwargs['query'] = request.query
        else:
            raise ValueError("Either table_name or query must be provided")
        
        if request.sample_size:
            load_kwargs['limit'] = request.sample_size
        
        df = source.load_data(**load_kwargs)
        
        # Create temporary file for profiling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
        
        try:
            use_case = ProfileDatasetUseCase()
            profile = use_case.execute(temp_path, request.profiling_strategy)
            summary = use_case.get_profiling_summary(profile)
            
            return ProfilingResponse(
                success=True,
                profile_id=str(profile.profile_id.value),
                message="Database profiling completed successfully",
                data=summary
            )
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"Database profiling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cloud-storage", response_model=ProfilingResponse)
async def profile_cloud_storage(request: CloudStorageProfilingRequest):
    """Profile a cloud storage object."""
    try:
        # Create cloud storage adapter
        adapter = get_cloud_storage_adapter(request.provider, request.connection)
        
        with adapter:
            # Load object
            df = adapter.load_object(request.object_key)
            
            # Create temporary file for profiling
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name
            
            try:
                use_case = ProfileDatasetUseCase()
                profile = use_case.execute(temp_path, request.profiling_strategy)
                summary = use_case.get_profiling_summary(profile)
                
                return ProfilingResponse(
                    success=True,
                    profile_id=str(profile.profile_id.value),
                    message="Cloud storage profiling completed successfully",
                    data=summary
                )
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Cloud storage profiling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streaming", response_model=ProfilingResponse)
async def profile_streaming_data(request: StreamingProfilingRequest):
    """Profile streaming data."""
    try:
        # Create streaming adapter
        adapter = get_streaming_adapter(request.provider, request.connection)
        
        with adapter:
            from ...infrastructure.adapters.streaming_adapter import StreamingDataProfiler
            profiler = StreamingDataProfiler(adapter)
            
            # Profile stream
            profile_result = profiler.profile_stream(
                request.stream_name,
                sample_size=request.sample_size,
                timeout_seconds=request.timeout_seconds
            )
            
            return ProfilingResponse(
                success=profile_result['success'],
                message="Streaming data profiling completed",
                data=profile_result
            )
    
    except Exception as e:
        logger.error(f"Streaming data profiling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nosql", response_model=ProfilingResponse)
async def profile_nosql_database(request: NoSQLProfilingRequest):
    """Profile a NoSQL database collection."""
    try:
        # Create NoSQL adapter
        adapter = get_nosql_adapter(request.db_type, request.connection)
        
        with adapter:
            # Load collection data
            load_kwargs = {}
            if request.query:
                load_kwargs['query'] = request.query
            if request.sample_size:
                load_kwargs['limit'] = request.sample_size
            
            df = adapter.load_collection(request.collection_name, **load_kwargs)
            
            # Create temporary file for profiling
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name
            
            try:
                use_case = ProfileDatasetUseCase()
                profile = use_case.execute(temp_path, "full")
                summary = use_case.get_profiling_summary(profile)
                
                return ProfilingResponse(
                    success=True,
                    profile_id=str(profile.profile_id.value),
                    message="NoSQL database profiling completed successfully",
                    data=summary
                )
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"NoSQL database profiling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async/file", response_model=ProfilingResponse)
async def profile_file_async(request: FileProfilingRequest, background_tasks: BackgroundTasks):
    """Profile a file dataset asynchronously."""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Store task in background storage
    background_tasks_storage[task_id] = {
        'status': 'pending',
        'created_at': datetime.utcnow(),
        'result': None,
        'error': None
    }
    
    # Add background task
    background_tasks.add_task(
        _profile_file_background,
        task_id,
        request.file_path,
        request.profiling_strategy,
        request.sample_size,
        request.sample_percentage
    )
    
    return ProfilingResponse(
        success=True,
        task_id=task_id,
        message="File profiling task submitted. Use /tasks/{task_id} to check status."
    )


@router.get("/tasks/{task_id}", response_model=ProfilingResponse)
async def get_task_status(task_id: str):
    """Get the status of an async profiling task."""
    if task_id not in background_tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = background_tasks_storage[task_id]
    
    if task_info['status'] == 'completed':
        return ProfilingResponse(
            success=True,
            task_id=task_id,
            message="Task completed successfully",
            data=task_info['result']
        )
    elif task_info['status'] == 'failed':
        return ProfilingResponse(
            success=False,
            task_id=task_id,
            message="Task failed",
            error=task_info['error']
        )
    else:
        return ProfilingResponse(
            success=True,
            task_id=task_id,
            message=f"Task status: {task_info['status']}"
        )


@router.get("/system/resources")
async def get_system_resources():
    """Get current system resource utilization."""
    try:
        optimizer = PerformanceOptimizer()
        resources = optimizer.check_system_resources()
        
        return JSONResponse(content={
            "success": True,
            "data": resources
        })
        
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats/supported")
async def get_supported_formats():
    """Get list of supported file formats and data sources."""
    return JSONResponse(content={
        "success": True,
        "data": {
            "file_formats": {
                "csv": "Comma-Separated Values",
                "json": "JavaScript Object Notation",
                "jsonl": "JSON Lines",
                "parquet": "Apache Parquet",
                "xlsx": "Excel (newer format)",
                "xls": "Excel (legacy format)",
                "tsv": "Tab-Separated Values",
                "feather": "Apache Arrow Feather"
            },
            "databases": {
                "postgresql": "PostgreSQL",
                "mysql": "MySQL",
                "sqlite": "SQLite"
            },
            "nosql_databases": {
                "mongodb": "MongoDB",
                "cassandra": "Apache Cassandra",
                "dynamodb": "AWS DynamoDB"
            },
            "cloud_storage": {
                "s3": "Amazon S3",
                "azure": "Azure Blob Storage",
                "gcs": "Google Cloud Storage"
            },
            "streaming": {
                "kafka": "Apache Kafka",
                "kinesis": "AWS Kinesis"
            }
        }
    })


async def _profile_file_background(task_id: str, file_path: str, 
                                  profiling_strategy: str,
                                  sample_size: Optional[int],
                                  sample_percentage: Optional[float]):
    """Background task for file profiling."""
    try:
        background_tasks_storage[task_id]['status'] = 'running'
        
        use_case = ProfileDatasetUseCase()
        
        if profiling_strategy == "sample" and sample_size:
            profile = use_case.execute_sample(file_path, sample_size)
        elif profiling_strategy == "sample" and sample_percentage:
            profile = use_case.execute_percentage_sample(file_path, sample_percentage)
        else:
            profile = use_case.execute(file_path, profiling_strategy)
        
        summary = use_case.get_profiling_summary(profile)
        
        background_tasks_storage[task_id].update({
            'status': 'completed',
            'result': {
                'profile_id': str(profile.profile_id.value),
                'summary': summary
            }
        })
        
    except Exception as e:
        logger.error(f"Background profiling task {task_id} failed: {e}")
        background_tasks_storage[task_id].update({
            'status': 'failed',
            'error': str(e)
        })


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed or failed task."""
    if task_id not in background_tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del background_tasks_storage[task_id]
    
    return JSONResponse(content={
        "success": True,
        "message": f"Task {task_id} deleted successfully"
    })


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "service": "data-profiling",
        "timestamp": datetime.utcnow().isoformat()
    })