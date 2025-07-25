"""
Demo API for Domain-Driven Monorepo Platform

Provides REST API endpoints for demonstrating platform capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import uuid

app = FastAPI(
    title="Domain-Driven Monorepo Platform Demo",
    description="Interactive demo showcasing package generation and platform capabilities",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://demo-web:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PackageRequest(BaseModel):
    name: str
    domain: str
    description: str
    architecture: str = "hexagonal"
    language: str = "python"
    framework: str = "fastapi"
    database: str = "postgresql"
    include_monitoring: bool = True
    include_security: bool = True

class PackageStatus(BaseModel):
    id: str
    name: str
    status: str  # "generating", "completed", "failed"
    progress: int  # 0-100
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    package_path: Optional[str] = None

class ValidationResult(BaseModel):
    package_name: str
    score: int
    violations: List[Dict[str, Any]]
    recommendations: List[str]

# In-memory storage for demo
package_jobs: Dict[str, PackageStatus] = {}
generated_packages: List[str] = []

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/api/platform/stats")
async def get_platform_stats():
    """Get platform statistics"""
    return {
        "total_packages": len(generated_packages),
        "active_jobs": len([j for j in package_jobs.values() if j.status == "generating"]),
        "completed_jobs": len([j for j in package_jobs.values() if j.status == "completed"]),
        "platform_version": "1.0.0",
        "uptime": "Demo Mode"
    }

@app.get("/api/packages/templates")
async def get_package_templates():
    """Get available package templates"""
    return {
        "domains": [
            {"id": "ecommerce", "name": "E-commerce", "description": "Online retail and marketplace"},
            {"id": "fintech", "name": "Financial Technology", "description": "Payment and banking services"},
            {"id": "healthcare", "name": "Healthcare", "description": "Medical and health services"},
            {"id": "iot", "name": "Internet of Things", "description": "Device connectivity and data"},
            {"id": "media", "name": "Media & Entertainment", "description": "Content and streaming services"}
        ],
        "architectures": [
            {"id": "hexagonal", "name": "Hexagonal Architecture", "description": "Ports and adapters pattern"},
            {"id": "clean", "name": "Clean Architecture", "description": "Layered architecture with dependency inversion"},
            {"id": "microservices", "name": "Microservices", "description": "Distributed service architecture"},
            {"id": "event_driven", "name": "Event-Driven", "description": "Event sourcing and CQRS patterns"}
        ],
        "frameworks": [
            {"id": "fastapi", "name": "FastAPI", "description": "Modern Python web framework"},
            {"id": "django", "name": "Django", "description": "Full-featured Python web framework"},
            {"id": "flask", "name": "Flask", "description": "Lightweight Python web framework"},
            {"id": "express", "name": "Express.js", "description": "Node.js web framework"}
        ]
    }

@app.post("/api/packages/generate")
async def generate_package(request: PackageRequest, background_tasks: BackgroundTasks):
    """Generate a new package"""
    job_id = str(uuid.uuid4())
    
    # Create job status
    job = PackageStatus(
        id=job_id,
        name=request.name,
        status="generating",
        progress=0,
        created_at=datetime.utcnow()
    )
    
    package_jobs[job_id] = job
    
    # Start package generation in background
    background_tasks.add_task(run_package_generation, job_id, request)
    
    return {"job_id": job_id, "status": "started"}

async def run_package_generation(job_id: str, request: PackageRequest):
    """Run package generation process"""
    try:
        job = package_jobs[job_id]
        
        # Simulate progress updates
        for progress in [10, 25, 40, 60, 80, 95]:
            await asyncio.sleep(1)
            job.progress = progress
            package_jobs[job_id] = job
        
        # Create package directory structure (simplified for demo)
        package_path = f"/app/demo-packages/{request.domain}/{request.name}"
        Path(package_path).mkdir(parents=True, exist_ok=True)
        
        # Generate basic package structure
        await create_demo_package_structure(package_path, request)
        
        # Complete the job
        job.status = "completed"
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.package_path = package_path
        package_jobs[job_id] = job
        
        # Add to generated packages list
        generated_packages.append(f"{request.domain}/{request.name}")
        
    except Exception as e:
        job = package_jobs[job_id]
        job.status = "failed"
        job.error_message = str(e)
        package_jobs[job_id] = job

async def create_demo_package_structure(package_path: str, request: PackageRequest):
    """Create demo package structure"""
    
    # Create directory structure
    dirs = [
        "src", "tests/unit", "tests/integration", "tests/security", "tests/performance",
        "docs", "k8s", "monitoring", "infrastructure"
    ]
    
    for dir_name in dirs:
        Path(f"{package_path}/{dir_name}").mkdir(parents=True, exist_ok=True)
    
    # Create pyproject.toml
    pyproject_content = f'''[project]
name = "{request.name}"
version = "0.1.0"
description = "{request.description}"
authors = ["Demo User <demo@example.com>"]
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.28.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "security: Security tests",
    "performance: Performance tests",
]
'''
    
    with open(f"{package_path}/pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    # Create Dockerfile
    dockerfile_content = f'''FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e .

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.{request.name.replace("-", "_")}.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open(f"{package_path}/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Create basic main.py
    main_content = f'''"""
{request.name} - {request.description}

Generated using Domain-Driven Monorepo Platform
Architecture: {request.architecture}
Framework: {request.framework}
"""

from fastapi import FastAPI

app = FastAPI(
    title="{request.name}",
    description="{request.description}",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {{"message": "Welcome to {request.name}", "status": "running"}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "service": "{request.name}"}}
'''
    
    package_name = request.name.replace("-", "_")
    src_dir = Path(f"{package_path}/src/{package_name}")
    src_dir.mkdir(parents=True, exist_ok=True)
    
    with open(f"{src_dir}/main.py", "w") as f:
        f.write(main_content)
    
    # Create __init__.py
    with open(f"{src_dir}/__init__.py", "w") as f:
        f.write(f'"""\\n{request.description}\\n"""\\n\\n__version__ = "0.1.0"\\n')

@app.get("/api/packages/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get package generation job status"""
    if job_id not in package_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return package_jobs[job_id]

@app.get("/api/packages")
async def list_packages():
    """List all generated packages"""
    packages = []
    
    for package_name in generated_packages:
        # Get package info
        domain, name = package_name.split("/")
        package_path = f"/app/demo-packages/{package_name}"
        
        # Check if package exists
        if Path(package_path).exists():
            packages.append({
                "name": name,
                "domain": domain,
                "path": package_name,
                "status": "ready",
                "created_at": datetime.utcnow().isoformat()  # Simplified for demo
            })
    
    return {"packages": packages, "total": len(packages)}

@app.post("/api/packages/{domain}/{name}/validate")
async def validate_package_independence(domain: str, name: str):
    """Validate package independence"""
    package_path = f"/app/demo-packages/{domain}/{name}"
    
    if not Path(package_path).exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Simulate validation (in real implementation, would run actual validator)
    await asyncio.sleep(2)  # Simulate validation time
    
    return ValidationResult(
        package_name=f"{domain}/{name}",
        score=92,
        violations=[
            {
                "type": "import_dependency",
                "severity": "medium",
                "message": "External dependency on shared utility",
                "file": "src/main.py",
                "line": 15
            }
        ],
        recommendations=[
            "Consider internalizing the shared utility",
            "Add explicit interface for external dependency",
            "Document dependency relationship"
        ]
    )

@app.post("/api/packages/{domain}/{name}/deploy")
async def deploy_package(domain: str, name: str):
    """Deploy package to demo environment"""
    package_path = f"/app/demo-packages/{domain}/{name}"
    
    if not Path(package_path).exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Simulate deployment
    await asyncio.sleep(3)
    
    return {
        "status": "deployed",
        "endpoint": f"http://localhost:8080/{domain}/{name}",
        "health_check": f"http://localhost:8080/{domain}/{name}/health",
        "deployment_id": str(uuid.uuid4())
    }

@app.get("/api/examples")
async def get_examples():
    """Get example packages and use cases"""
    return {
        "examples": [
            {
                "name": "E-commerce Order Management",
                "domain": "ecommerce",
                "description": "Complete order processing system with payment integration",
                "architecture": "hexagonal",
                "features": ["Order Creation", "Payment Processing", "Inventory Management", "Event Publishing"],
                "tech_stack": ["FastAPI", "PostgreSQL", "Redis", "RabbitMQ"]
            },
            {
                "name": "User Authentication Service",
                "domain": "identity",
                "description": "JWT-based authentication with OAuth2 support",
                "architecture": "clean",
                "features": ["User Registration", "Login/Logout", "Token Management", "Role-Based Access"],
                "tech_stack": ["FastAPI", "PostgreSQL", "Redis", "JWT"]
            },
            {
                "name": "Real-time Chat System",
                "domain": "communication",
                "description": "WebSocket-based messaging with presence tracking",
                "architecture": "event_driven",
                "features": ["Real-time Messaging", "Presence Tracking", "Message History", "File Sharing"],
                "tech_stack": ["FastAPI", "WebSockets", "MongoDB", "Redis"]
            }
        ]
    }

@app.get("/api/security/scan/{domain}/{name}")
async def security_scan(domain: str, name: str):
    """Run security scan on package"""
    package_path = f"/app/demo-packages/{domain}/{name}"
    
    if not Path(package_path).exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Simulate security scan
    await asyncio.sleep(3)
    
    return {
        "scan_date": datetime.utcnow().isoformat(),
        "overall_score": 85,
        "findings": [
            {
                "severity": "medium",
                "category": "dependency",
                "title": "Outdated dependency version",
                "description": "Package uses an older version of a dependency",
                "recommendation": "Update to latest stable version"
            },
            {
                "severity": "low",
                "category": "configuration",
                "title": "Missing security header",
                "description": "API response missing security headers",
                "recommendation": "Add security middleware"
            }
        ],
        "compliance": {
            "owasp_top10": "passing",
            "cis_benchmarks": "passing",
            "gdpr": "compliant"
        }
    }

@app.get("/api/performance/benchmark/{domain}/{name}")
async def performance_benchmark(domain: str, name: str):
    """Run performance benchmark on package"""
    package_path = f"/app/demo-packages/{domain}/{name}"
    
    if not Path(package_path).exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Simulate performance benchmark
    await asyncio.sleep(4)
    
    return {
        "benchmark_date": datetime.utcnow().isoformat(),
        "metrics": {
            "response_time_avg": 125,  # ms
            "response_time_p95": 250,
            "throughput": 1250,  # requests/second
            "memory_usage": 256,  # MB
            "cpu_usage": 15  # percent
        },
        "load_test_results": {
            "concurrent_users": 100,
            "duration": "5m",
            "success_rate": 99.8,
            "error_rate": 0.2
        },
        "recommendations": [
            "Response times are within acceptable range",
            "Consider connection pooling for database",
            "Memory usage is optimal"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)