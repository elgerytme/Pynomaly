"""
Distributed quality processing service for enterprise-scale performance optimization.

This service implements distributed processing capabilities, auto-scaling,
load balancing, and fault-tolerant processing for data quality operations.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import weakref
import redis
import psutil
from contextlib import asynccontextmanager

from ...domain.entities.quality_job import QualityJob, JobStatus, JobPriority
from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from ...domain.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class ProcessingNodeStatus(Enum):
    """Status of processing nodes."""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


class WorkloadType(Enum):
    """Types of quality processing workloads."""
    VALIDATION = "validation"
    PROFILING = "profiling"
    CLEANSING = "cleansing"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"
    REPORTING = "reporting"


@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed cluster."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    
    # Capacity and status
    max_concurrent_jobs: int = 10
    current_job_count: int = 0
    status: ProcessingNodeStatus = ProcessingNodeStatus.IDLE
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    network_throughput_mbps: float = 0.0
    
    # Specialization
    supported_workloads: Set[WorkloadType] = field(default_factory=set)
    processing_capabilities: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    avg_job_execution_time: float = 0.0
    job_success_rate: float = 1.0
    total_jobs_processed: int = 0
    
    # Metadata
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    @property
    def utilization_score(self) -> float:
        """Calculate node utilization score (0-1)."""
        cpu_factor = self.cpu_usage_percent / 100
        memory_factor = self.memory_usage_percent / 100
        job_factor = self.current_job_count / max(1, self.max_concurrent_jobs)
        
        return (cpu_factor + memory_factor + job_factor) / 3
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy and available."""
        return (
            self.status in [ProcessingNodeStatus.ACTIVE, ProcessingNodeStatus.IDLE, ProcessingNodeStatus.BUSY] and
            self.last_heartbeat > datetime.utcnow() - timedelta(minutes=2) and
            self.cpu_usage_percent < 95 and
            self.memory_usage_percent < 90
        )
    
    @property
    def can_accept_job(self) -> bool:
        """Check if node can accept new jobs."""
        return (
            self.is_healthy and
            self.current_job_count < self.max_concurrent_jobs and
            self.status != ProcessingNodeStatus.OVERLOADED
        )


@dataclass
class WorkloadDistributionStrategy:
    """Strategy for distributing workloads across nodes."""
    strategy_name: str
    priority_weight: float = 1.0
    resource_weight: float = 1.0
    affinity_weight: float = 0.5
    
    # Load balancing parameters
    max_job_queue_size: int = 1000
    job_timeout_minutes: int = 60
    retry_attempts: int = 3
    
    # Auto-scaling parameters
    scale_up_threshold: float = 0.8  # When average utilization exceeds this
    scale_down_threshold: float = 0.3  # When average utilization drops below this
    min_nodes: int = 2
    max_nodes: int = 100
    
    # Fault tolerance
    enable_redundancy: bool = True
    redundancy_factor: int = 2  # Number of backup nodes for critical jobs


@dataclass
class ClusterMetrics:
    """Comprehensive cluster performance metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Node metrics
    total_nodes: int = 0
    active_nodes: int = 0
    failed_nodes: int = 0
    
    # Resource utilization
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    
    # Job processing metrics
    jobs_queued: int = 0
    jobs_processing: int = 0
    jobs_completed_last_hour: int = 0
    jobs_failed_last_hour: int = 0
    avg_job_completion_time: float = 0.0
    
    # Throughput metrics
    records_processed_per_second: float = 0.0
    data_throughput_mbps: float = 0.0
    api_requests_per_second: float = 0.0
    
    # Quality metrics
    quality_operations_per_second: float = 0.0
    validation_success_rate: float = 0.0
    cleansing_throughput: float = 0.0


class DistributedQualityProcessingService:
    """Service for distributed data quality processing with enterprise-scale optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the distributed processing service."""
        self.config = config
        
        # Cluster management
        self.nodes: Dict[str, ProcessingNode] = {}
        self.job_queue: deque = deque()
        self.active_jobs: Dict[str, QualityJob] = {}
        self.completed_jobs: Dict[str, QualityJob] = {}
        
        # Processing pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get("max_thread_workers", 50)
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=config.get("max_process_workers", 8)
        )
        
        # Distribution strategy
        self.distribution_strategy = WorkloadDistributionStrategy(
            strategy_name=config.get("distribution_strategy", "resource_aware"),
            max_job_queue_size=config.get("max_job_queue_size", 10000),
            scale_up_threshold=config.get("scale_up_threshold", 0.8),
            scale_down_threshold=config.get("scale_down_threshold", 0.3),
            min_nodes=config.get("min_nodes", 2),
            max_nodes=config.get("max_nodes", 1000)
        )
        
        # Redis for distributed coordination
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            decode_responses=True
        )
        
        # Performance tracking
        self.cluster_metrics = ClusterMetrics()
        self.performance_history: List[ClusterMetrics] = []
        
        # Service state
        self.is_running = False
        self.node_discovery_enabled = config.get("enable_node_discovery", True)
        
        # Start background tasks
        asyncio.create_task(self._cluster_monitoring_task())
        asyncio.create_task(self._job_scheduling_task())
        asyncio.create_task(self._auto_scaling_task())
        asyncio.create_task(self._health_check_task())
        asyncio.create_task(self._performance_monitoring_task())
    
    async def start(self) -> None:
        """Start the distributed processing service."""
        logger.info("Starting distributed quality processing service...")
        
        self.is_running = True
        
        # Register this node if not in coordinator-only mode
        if self.config.get("enable_local_processing", True):
            await self._register_local_node()
        
        # Discover existing nodes
        if self.node_discovery_enabled:
            await self._discover_existing_nodes()
        
        logger.info(f"Distributed processing service started with {len(self.nodes)} nodes")
    
    async def shutdown(self) -> None:
        """Shutdown the distributed processing service."""
        logger.info("Shutting down distributed processing service...")
        
        self.is_running = False
        
        # Complete active jobs
        await self._complete_active_jobs()
        
        # Shutdown pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Clear state
        self.nodes.clear()
        self.job_queue.clear()
        self.active_jobs.clear()
        
        logger.info("Distributed processing service shutdown complete")
    
    async def _register_local_node(self) -> None:
        """Register the local node in the cluster."""
        # Get system information
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        local_node = ProcessingNode(
            node_id=f"node_{hashlib.sha256(str(psutil.boot_time()).encode()).hexdigest()[:8]}",
            hostname=psutil.hostname(),
            ip_address="localhost",  # Would get actual IP in production
            port=self.config.get("processing_port", 8080),
            max_concurrent_jobs=cpu_count * 2,
            memory_available_gb=memory.total / (1024**3),
            supported_workloads={WorkloadType.VALIDATION, WorkloadType.PROFILING, 
                               WorkloadType.CLEANSING, WorkloadType.MONITORING}
        )
        
        self.nodes[local_node.node_id] = local_node
        
        # Register in Redis for cluster coordination
        await self._register_node_in_redis(local_node)
        
        logger.info(f"Registered local node: {local_node.node_id}")
    
    async def _register_node_in_redis(self, node: ProcessingNode) -> None:
        """Register node in Redis for cluster coordination."""
        try:
            node_data = {
                "hostname": node.hostname,
                "ip_address": node.ip_address,
                "port": node.port,
                "max_concurrent_jobs": node.max_concurrent_jobs,
                "supported_workloads": [w.value for w in node.supported_workloads],
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "status": node.status.value
            }
            
            # Store node information
            self.redis_client.hset(
                f"quality:nodes:{node.node_id}", 
                mapping=node_data
            )
            
            # Add to active nodes set
            self.redis_client.sadd("quality:active_nodes", node.node_id)
            
            # Set expiration for automatic cleanup
            self.redis_client.expire(f"quality:nodes:{node.node_id}", 300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Failed to register node in Redis: {str(e)}")
    
    async def _discover_existing_nodes(self) -> None:
        """Discover existing nodes from Redis."""
        try:
            active_nodes = self.redis_client.smembers("quality:active_nodes")
            
            for node_id in active_nodes:
                if node_id in self.nodes:
                    continue  # Skip if already known
                
                node_data = self.redis_client.hgetall(f"quality:nodes:{node_id}")
                if node_data:
                    node = ProcessingNode(
                        node_id=node_id,
                        hostname=node_data.get("hostname", "unknown"),
                        ip_address=node_data.get("ip_address", "unknown"),
                        port=int(node_data.get("port", 8080)),
                        max_concurrent_jobs=int(node_data.get("max_concurrent_jobs", 10)),
                        supported_workloads={
                            WorkloadType(w) for w in json.loads(
                                node_data.get("supported_workloads", "[]")
                            )
                        }
                    )
                    
                    self.nodes[node_id] = node
                    logger.info(f"Discovered node: {node_id}")
        
        except Exception as e:
            logger.error(f"Failed to discover existing nodes: {str(e)}")
    
    async def _cluster_monitoring_task(self) -> None:
        """Background task for cluster monitoring and coordination."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update cluster metrics
                await self._update_cluster_metrics()
                
                # Sync with Redis
                await self._sync_cluster_state()
                
                # Check for failed nodes
                await self._detect_failed_nodes()
                
                logger.debug(f"Cluster monitoring: {len(self.nodes)} nodes, "
                           f"{self.cluster_metrics.jobs_queued} queued jobs")
                
            except Exception as e:
                logger.error(f"Cluster monitoring error: {str(e)}")
    
    async def _job_scheduling_task(self) -> None:
        """Background task for job scheduling and distribution."""
        while self.is_running:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Process job queue
                if self.job_queue and self._has_available_capacity():
                    job = self.job_queue.popleft()
                    await self._schedule_job(job)
                
                # Check for completed jobs
                await self._check_completed_jobs()
                
                # Rebalance if needed
                if len(self.job_queue) > self.distribution_strategy.max_job_queue_size // 2:
                    await self._rebalance_workload()
                
            except Exception as e:
                logger.error(f"Job scheduling error: {str(e)}")
    
    async def _auto_scaling_task(self) -> None:
        """Background task for auto-scaling cluster resources."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.config.get("enable_auto_scaling", True):
                    continue
                
                # Calculate cluster utilization
                avg_utilization = self._calculate_average_utilization()
                
                # Scale up if needed
                if (avg_utilization > self.distribution_strategy.scale_up_threshold and
                    len(self.nodes) < self.distribution_strategy.max_nodes):
                    await self._request_scale_up()
                
                # Scale down if needed
                elif (avg_utilization < self.distribution_strategy.scale_down_threshold and
                      len(self.nodes) > self.distribution_strategy.min_nodes):
                    await self._request_scale_down()
                
                logger.debug(f"Auto-scaling check: {avg_utilization:.2f} utilization, "
                           f"{len(self.nodes)} nodes")
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {str(e)}")
    
    async def _health_check_task(self) -> None:
        """Background task for node health monitoring."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update node health metrics
                for node in self.nodes.values():
                    await self._update_node_health(node)
                
                # Remove unhealthy nodes
                unhealthy_nodes = [
                    node_id for node_id, node in self.nodes.items()
                    if not node.is_healthy
                ]
                
                for node_id in unhealthy_nodes:
                    await self._remove_unhealthy_node(node_id)
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
    
    async def _performance_monitoring_task(self) -> None:
        """Background task for performance monitoring and optimization."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Record current metrics
                current_metrics = await self._collect_cluster_metrics()
                self.performance_history.append(current_metrics)
                
                # Keep only last 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.performance_history = [
                    m for m in self.performance_history
                    if m.timestamp > cutoff_time
                ]
                
                # Analyze performance trends
                await self._analyze_performance_trends()
                
                # Generate optimization recommendations
                recommendations = await self._generate_optimization_recommendations()
                if recommendations:
                    logger.info(f"Performance recommendations: {recommendations}")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
    
    # Error handling would be managed by interface implementation
    async def submit_job(self, job: QualityJob) -> str:
        """Submit a quality processing job to the cluster."""
        # Assign job ID if not present
        if not job.job_id:
            job.job_id = f"job_{int(time.time())}_{len(self.job_queue)}"
        
        # Set job status
        job.status = JobStatus.QUEUED
        job.queued_at = datetime.utcnow()
        
        # Add to queue
        self.job_queue.append(job)
        
        # Store in Redis for distributed coordination
        await self._store_job_in_redis(job)
        
        logger.info(f"Submitted job {job.job_id} to processing queue")
        return job.job_id
    
    async def _schedule_job(self, job: QualityJob) -> None:
        """Schedule a job for execution on the best available node."""
        # Find best node for this job
        best_node = await self._find_best_node_for_job(job)
        
        if not best_node:
            # No available nodes, put job back in queue
            self.job_queue.appendleft(job)
            logger.warning(f"No available nodes for job {job.job_id}, requeuing")
            return
        
        # Assign job to node
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.assigned_node_id = best_node.node_id
        
        # Update node status
        best_node.current_job_count += 1
        if best_node.current_job_count >= best_node.max_concurrent_jobs:
            best_node.status = ProcessingNodeStatus.BUSY
        
        # Store active job
        self.active_jobs[job.job_id] = job
        
        # Execute job asynchronously
        asyncio.create_task(self._execute_job_on_node(job, best_node))
        
        logger.info(f"Scheduled job {job.job_id} on node {best_node.node_id}")
    
    async def _find_best_node_for_job(self, job: QualityJob) -> Optional[ProcessingNode]:
        """Find the best node to execute a specific job."""
        available_nodes = [
            node for node in self.nodes.values()
            if node.can_accept_job
        ]
        
        if not available_nodes:
            return None
        
        # Score nodes based on various factors
        node_scores = []
        
        for node in available_nodes:
            score = 0.0
            
            # Resource utilization (lower is better)
            utilization_score = 1.0 - node.utilization_score
            score += utilization_score * self.distribution_strategy.resource_weight
            
            # Job priority (higher priority gets better nodes)
            if job.priority == JobPriority.HIGH:
                score += 0.3
            elif job.priority == JobPriority.MEDIUM:
                score += 0.1
            
            # Workload affinity
            job_workload = self._determine_job_workload(job)
            if job_workload in node.supported_workloads:
                score += 0.2 * self.distribution_strategy.affinity_weight
            
            # Performance history
            if node.job_success_rate > 0.95:
                score += 0.1
            if node.avg_job_execution_time < 60:  # Fast execution
                score += 0.1
            
            node_scores.append((node, score))
        
        # Return node with highest score
        node_scores.sort(key=lambda x: x[1], reverse=True)
        return node_scores[0][0]
    
    def _determine_job_workload(self, job: QualityJob) -> WorkloadType:
        """Determine the workload type for a job."""
        job_type = job.job_type.lower()
        
        if "validation" in job_type or "validate" in job_type:
            return WorkloadType.VALIDATION
        elif "profiling" in job_type or "profile" in job_type:
            return WorkloadType.PROFILING
        elif "cleansing" in job_type or "clean" in job_type:
            return WorkloadType.CLEANSING
        elif "monitoring" in job_type or "monitor" in job_type:
            return WorkloadType.MONITORING
        elif "analytics" in job_type or "analyze" in job_type:
            return WorkloadType.ANALYTICS
        elif "report" in job_type:
            return WorkloadType.REPORTING
        else:
            return WorkloadType.VALIDATION  # Default
    
    async def _execute_job_on_node(self, job: QualityJob, node: ProcessingNode) -> None:
        """Execute a job on a specific node."""
        start_time = time.time()
        
        try:
            # Execute job based on type
            if node.node_id == next(iter(self.nodes.keys())):  # Local node
                result = await self._execute_job_locally(job)
            else:
                result = await self._execute_job_remotely(job, node)
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            
            # Update node metrics
            execution_time = time.time() - start_time
            node.avg_job_execution_time = (
                (node.avg_job_execution_time * node.total_jobs_processed + execution_time) /
                (node.total_jobs_processed + 1)
            )
            node.total_jobs_processed += 1
            
            logger.info(f"Job {job.job_id} completed successfully on node {node.node_id}")
            
        except Exception as e:
            # Handle job failure
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            
            # Update node failure rate
            total_jobs = node.total_jobs_processed + 1
            success_jobs = node.total_jobs_processed * node.job_success_rate
            node.job_success_rate = success_jobs / total_jobs
            node.total_jobs_processed = total_jobs
            
            logger.error(f"Job {job.job_id} failed on node {node.node_id}: {str(e)}")
            
            # Retry job if configured
            if job.retry_count < self.distribution_strategy.retry_attempts:
                job.retry_count += 1
                job.status = JobStatus.QUEUED
                self.job_queue.append(job)
                logger.info(f"Retrying job {job.job_id} (attempt {job.retry_count})")
        
        finally:
            # Update node status
            node.current_job_count = max(0, node.current_job_count - 1)
            if node.current_job_count < node.max_concurrent_jobs:
                node.status = ProcessingNodeStatus.ACTIVE if node.current_job_count > 0 else ProcessingNodeStatus.IDLE
            
            # Move job to completed
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            self.completed_jobs[job.job_id] = job
            
            # Update Redis
            await self._update_job_status_in_redis(job)
    
    async def _execute_job_locally(self, job: QualityJob) -> Dict[str, Any]:
        """Execute a job on the local node."""
        # This would contain the actual job execution logic
        # For now, simulate job execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "job_id": job.job_id,
            "result": "Job executed successfully",
            "records_processed": 1000,
            "execution_node": "local",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_job_remotely(self, job: QualityJob, node: ProcessingNode) -> Dict[str, Any]:
        """Execute a job on a remote node."""
        # This would contain logic to send job to remote node
        # For now, simulate remote execution
        await asyncio.sleep(0.2)  # Simulate network + processing time
        
        return {
            "job_id": job.job_id,
            "result": "Job executed successfully on remote node",
            "records_processed": 1000,
            "execution_node": node.node_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _has_available_capacity(self) -> bool:
        """Check if cluster has available capacity for new jobs."""
        return any(node.can_accept_job for node in self.nodes.values())
    
    def _calculate_average_utilization(self) -> float:
        """Calculate average cluster utilization."""
        if not self.nodes:
            return 0.0
        
        total_utilization = sum(node.utilization_score for node in self.nodes.values())
        return total_utilization / len(self.nodes)
    
    async def _store_job_in_redis(self, job: QualityJob) -> None:
        """Store job information in Redis."""
        try:
            job_data = {
                "job_type": job.job_type,
                "priority": job.priority.value,
                "status": job.status.value,
                "queued_at": job.queued_at.isoformat() if job.queued_at else "",
                "dataset_id": job.dataset_id or "",
                "retry_count": job.retry_count
            }
            
            self.redis_client.hset(f"quality:jobs:{job.job_id}", mapping=job_data)
            
        except Exception as e:
            logger.error(f"Failed to store job in Redis: {str(e)}")
    
    async def _update_job_status_in_redis(self, job: QualityJob) -> None:
        """Update job status in Redis."""
        try:
            updates = {
                "status": job.status.value,
                "assigned_node_id": job.assigned_node_id or "",
                "started_at": job.started_at.isoformat() if job.started_at else "",
                "completed_at": job.completed_at.isoformat() if job.completed_at else "",
                "error_message": job.error_message or ""
            }
            
            self.redis_client.hset(f"quality:jobs:{job.job_id}", mapping=updates)
            
        except Exception as e:
            logger.error(f"Failed to update job status in Redis: {str(e)}")
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status information."""
        active_nodes = [node for node in self.nodes.values() if node.is_healthy]
        
        return {
            "cluster_id": self.config.get("cluster_id", "default"),
            "timestamp": datetime.utcnow().isoformat(),
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "average_utilization": self._calculate_average_utilization(),
            "jobs_queued": len(self.job_queue),
            "jobs_active": len(self.active_jobs),
            "jobs_completed": len(self.completed_jobs),
            "performance_metrics": {
                "avg_cpu_utilization": self.cluster_metrics.avg_cpu_utilization,
                "avg_memory_utilization": self.cluster_metrics.avg_memory_utilization,
                "records_processed_per_second": self.cluster_metrics.records_processed_per_second,
                "avg_job_completion_time": self.cluster_metrics.avg_job_completion_time
            },
            "node_details": [
                {
                    "node_id": node.node_id,
                    "hostname": node.hostname,
                    "status": node.status.value,
                    "utilization": node.utilization_score,
                    "current_jobs": node.current_job_count,
                    "max_jobs": node.max_concurrent_jobs
                }
                for node in self.nodes.values()
            ]
        }
    
    # Error handling would be managed by interface implementation
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
        else:
            # Check Redis
            job_data = self.redis_client.hgetall(f"quality:jobs:{job_id}")
            if not job_data:
                return None
            
            return {
                "job_id": job_id,
                "status": job_data.get("status", "unknown"),
                "job_type": job_data.get("job_type", "unknown"),
                "priority": job_data.get("priority", "medium"),
                "assigned_node_id": job_data.get("assigned_node_id"),
                "queued_at": job_data.get("queued_at"),
                "started_at": job_data.get("started_at"),
                "completed_at": job_data.get("completed_at"),
                "error_message": job_data.get("error_message")
            }
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "job_type": job.job_type,
            "priority": job.priority.value,
            "assigned_node_id": job.assigned_node_id,
            "queued_at": job.queued_at.isoformat() if job.queued_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result": job.result,
            "error_message": job.error_message,
            "retry_count": job.retry_count
        }
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on cluster performance."""
        recommendations = []
        
        # Analyze resource utilization
        avg_utilization = self._calculate_average_utilization()
        if avg_utilization > 0.9:
            recommendations.append("Cluster utilization is very high. Consider adding more nodes.")
        elif avg_utilization < 0.3:
            recommendations.append("Cluster utilization is low. Consider removing idle nodes to save costs.")
        
        # Analyze job queue
        if len(self.job_queue) > self.distribution_strategy.max_job_queue_size // 2:
            recommendations.append("Job queue is getting large. Consider optimizing job processing or adding capacity.")
        
        # Analyze node performance
        slow_nodes = [
            node for node in self.nodes.values()
            if node.avg_job_execution_time > 300  # 5 minutes
        ]
        if slow_nodes:
            recommendations.append(f"{len(slow_nodes)} nodes have slow job execution. Consider investigation or replacement.")
        
        # Analyze failure rates
        failing_nodes = [
            node for node in self.nodes.values()
            if node.job_success_rate < 0.9
        ]
        if failing_nodes:
            recommendations.append(f"{len(failing_nodes)} nodes have high failure rates. Consider maintenance or replacement.")
        
        return recommendations