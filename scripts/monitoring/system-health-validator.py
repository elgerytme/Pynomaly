#!/usr/bin/env python3

"""
Comprehensive System Health Validation Framework
This script performs automated health checks and validation for the MLOps platform
"""

import asyncio
import aiohttp
import subprocess
import json
import yaml
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import ssl
import socket
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HealthCheckResult:
    """Individual health check result"""
    check_name: str
    component: str
    status: str  # HEALTHY, DEGRADED, UNHEALTHY
    response_time: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class SystemHealthReport:
    """Complete system health report"""
    environment: str
    timestamp: str
    overall_status: str
    total_checks: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    checks: List[HealthCheckResult]
    performance_summary: Dict[str, Any]
    recommendations: List[str]

class HealthValidator:
    """Main health validation class"""
    
    def __init__(self, environment: str, base_url: str, namespace: str):
        self.environment = environment
        self.base_url = base_url.rstrip('/')
        self.namespace = namespace
        self.checks: List[HealthCheckResult] = []
        
        # Define health check endpoints
        self.endpoints = {
            'api_health': '/health',
            'api_detailed': '/health/detailed',
            'api_metrics': '/metrics',
            'api_ready': '/ready',
            'auth_health': '/api/v1/auth/health',
            'models_health': '/api/v1/models/health'
        }
    
    async def run_comprehensive_health_check(self) -> SystemHealthReport:
        """Run comprehensive health check across all components"""
        logger.info(f"Starting comprehensive health check for {self.environment} environment")
        
        # Run all health checks
        await asyncio.gather(
            self.check_api_health(),
            self.check_database_health(),
            self.check_redis_health(),
            self.check_kubernetes_health(),
            self.check_monitoring_health(),
            self.check_security_health(),
            self.check_storage_health(),
            self.check_network_connectivity(),
            self.check_ssl_certificates(),
            self.check_resource_utilization()
        )
        
        # Generate final report
        return self._generate_report()
    
    async def check_api_health(self):
        """Check API service health"""
        logger.info("Checking API service health...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for check_name, endpoint in self.endpoints.items():
                start_time = time.time()
                try:
                    url = f"{self.base_url}{endpoint}"
                    async with session.get(url, ssl=False) as response:
                        response_time = time.time() - start_time
                        response_text = await response.text()
                        
                        if response.status == 200:
                            status = "HEALTHY"
                            message = f"API endpoint {endpoint} is healthy"
                            
                            # Parse response for additional details
                            details = {}
                            try:
                                if response.headers.get('content-type', '').startswith('application/json'):
                                    response_data = json.loads(response_text)
                                    details = response_data
                            except json.JSONDecodeError:
                                details = {"response_preview": response_text[:100]}
                        
                        elif response.status in [503, 502]:
                            status = "UNHEALTHY"
                            message = f"API endpoint {endpoint} returned {response.status}"
                            details = {"status_code": response.status, "response": response_text[:200]}
                        else:
                            status = "DEGRADED"
                            message = f"API endpoint {endpoint} returned unexpected status {response.status}"
                            details = {"status_code": response.status}
                        
                        self.checks.append(HealthCheckResult(
                            check_name=check_name,
                            component="API Service",
                            status=status,
                            response_time=response_time,
                            message=message,
                            details=details
                        ))
                
                except asyncio.TimeoutError:
                    self.checks.append(HealthCheckResult(
                        check_name=check_name,
                        component="API Service",
                        status="UNHEALTHY",
                        response_time=time.time() - start_time,
                        message=f"Timeout accessing {endpoint}",
                        details={"error": "timeout"}
                    ))
                
                except Exception as e:
                    self.checks.append(HealthCheckResult(
                        check_name=check_name,
                        component="API Service",
                        status="UNHEALTHY",
                        response_time=time.time() - start_time,
                        message=f"Error accessing {endpoint}: {str(e)}",
                        details={"error": str(e)}
                    ))
    
    async def check_database_health(self):
        """Check database health"""
        logger.info("Checking database health...")
        
        start_time = time.time()
        try:
            # Check PostgreSQL pod status
            result = subprocess.run([
                "kubectl", "get", "pods", "-l", "app=postgres",
                "-n", self.namespace, "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                postgres_pods = pods_data.get('items', [])
                
                if not postgres_pods:
                    self.checks.append(HealthCheckResult(
                        check_name="postgres_pods",
                        component="Database",
                        status="UNHEALTHY",
                        response_time=time.time() - start_time,
                        message="No PostgreSQL pods found",
                        details={"pod_count": 0}
                    ))
                    return
                
                healthy_pods = 0
                total_pods = len(postgres_pods)
                
                for pod in postgres_pods:
                    pod_name = pod.get('metadata', {}).get('name', 'unknown')
                    pod_status = pod.get('status', {})
                    pod_phase = pod_status.get('phase', 'Unknown')
                    
                    if pod_phase == 'Running':
                        # Check if all containers are ready
                        container_statuses = pod_status.get('containerStatuses', [])
                        all_ready = all(cs.get('ready', False) for cs in container_statuses)
                        
                        if all_ready:
                            healthy_pods += 1
                
                if healthy_pods == total_pods:
                    status = "HEALTHY"
                    message = f"All {total_pods} PostgreSQL pods are healthy"
                elif healthy_pods > 0:
                    status = "DEGRADED"
                    message = f"{healthy_pods}/{total_pods} PostgreSQL pods are healthy"
                else:
                    status = "UNHEALTHY"
                    message = f"No healthy PostgreSQL pods ({total_pods} total)"
                
                self.checks.append(HealthCheckResult(
                    check_name="postgres_pods",
                    component="Database",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={"healthy_pods": healthy_pods, "total_pods": total_pods}
                ))
                
                # Test database connectivity
                if healthy_pods > 0:
                    await self._test_database_connectivity()
            
            else:
                self.checks.append(HealthCheckResult(
                    check_name="postgres_pods",
                    component="Database",
                    status="UNHEALTHY",
                    response_time=time.time() - start_time,
                    message="Failed to check PostgreSQL pod status",
                    details={"error": result.stderr}
                ))
        
        except subprocess.TimeoutExpired:
            self.checks.append(HealthCheckResult(
                check_name="postgres_pods",
                component="Database",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message="Timeout checking PostgreSQL pod status"
            ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="postgres_pods",
                component="Database",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"Error checking database health: {str(e)}",
                details={"error": str(e)}
            ))
    
    async def _test_database_connectivity(self):
        """Test database connectivity from within the cluster"""
        start_time = time.time()
        try:
            # Test database connection using kubectl exec
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace,
                "deployment/api-server", "--",
                "pg_isready", "-h", "postgres", "-p", "5432"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                status = "HEALTHY"
                message = "Database connectivity test passed"
            else:
                status = "DEGRADED"
                message = "Database connectivity test failed"
            
            self.checks.append(HealthCheckResult(
                check_name="database_connectivity",
                component="Database",
                status=status,
                response_time=time.time() - start_time,
                message=message,
                details={"pg_isready_output": result.stdout.strip()}
            ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="database_connectivity",
                component="Database",
                status="DEGRADED",
                response_time=time.time() - start_time,
                message=f"Could not test database connectivity: {str(e)}"
            ))
    
    async def check_redis_health(self):
        """Check Redis health"""
        logger.info("Checking Redis health...")
        
        start_time = time.time()
        try:
            # Check Redis pods
            result = subprocess.run([
                "kubectl", "get", "pods", "-l", "app=redis-cache",
                "-n", self.namespace, "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                redis_pods = pods_data.get('items', [])
                
                if redis_pods:
                    running_pods = sum(
                        1 for pod in redis_pods 
                        if pod.get('status', {}).get('phase') == 'Running'
                    )
                    
                    total_pods = len(redis_pods)
                    
                    if running_pods == total_pods:
                        status = "HEALTHY"
                        message = f"All {total_pods} Redis pods are running"
                    elif running_pods > 0:
                        status = "DEGRADED"
                        message = f"{running_pods}/{total_pods} Redis pods are running"
                    else:
                        status = "UNHEALTHY"
                        message = "No Redis pods are running"
                    
                    self.checks.append(HealthCheckResult(
                        check_name="redis_pods",
                        component="Cache",
                        status=status,
                        response_time=time.time() - start_time,
                        message=message,
                        details={"running_pods": running_pods, "total_pods": total_pods}
                    ))
                else:
                    self.checks.append(HealthCheckResult(
                        check_name="redis_pods",
                        component="Cache",
                        status="UNHEALTHY",
                        response_time=time.time() - start_time,
                        message="No Redis pods found"
                    ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="redis_pods",
                component="Cache",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"Error checking Redis health: {str(e)}"
            ))
    
    async def check_kubernetes_health(self):
        """Check Kubernetes cluster health"""
        logger.info("Checking Kubernetes cluster health...")
        
        # Check node health
        start_time = time.time()
        try:
            result = subprocess.run([
                "kubectl", "get", "nodes", "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                nodes_data = json.loads(result.stdout)
                nodes = nodes_data.get('items', [])
                
                ready_nodes = 0
                total_nodes = len(nodes)
                node_details = []
                
                for node in nodes:
                    node_name = node.get('metadata', {}).get('name', 'unknown')
                    conditions = node.get('status', {}).get('conditions', [])
                    
                    # Check if node is ready
                    ready_condition = next(
                        (c for c in conditions if c.get('type') == 'Ready'), None
                    )
                    
                    if ready_condition and ready_condition.get('status') == 'True':
                        ready_nodes += 1
                        node_status = "Ready"
                    else:
                        node_status = "NotReady"
                    
                    node_details.append({
                        "name": node_name,
                        "status": node_status
                    })
                
                if ready_nodes == total_nodes:
                    status = "HEALTHY"
                    message = f"All {total_nodes} nodes are ready"
                elif ready_nodes > total_nodes * 0.5:  # More than half nodes ready
                    status = "DEGRADED"
                    message = f"{ready_nodes}/{total_nodes} nodes are ready"
                else:
                    status = "UNHEALTHY"
                    message = f"Only {ready_nodes}/{total_nodes} nodes are ready"
                
                self.checks.append(HealthCheckResult(
                    check_name="kubernetes_nodes",
                    component="Kubernetes",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={"ready_nodes": ready_nodes, "total_nodes": total_nodes, "nodes": node_details}
                ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="kubernetes_nodes",
                component="Kubernetes",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"Error checking Kubernetes nodes: {str(e)}"
            ))
        
        # Check namespace health
        await self._check_namespace_health()
    
    async def _check_namespace_health(self):
        """Check namespace-specific health"""
        start_time = time.time()
        try:
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace, "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = pods_data.get('items', [])
                
                running_pods = 0
                pending_pods = 0
                failed_pods = 0
                
                for pod in pods:
                    phase = pod.get('status', {}).get('phase', 'Unknown')
                    if phase == 'Running':
                        running_pods += 1
                    elif phase == 'Pending':
                        pending_pods += 1
                    elif phase in ['Failed', 'CrashLoopBackOff']:
                        failed_pods += 1
                
                total_pods = len(pods)
                
                if failed_pods == 0 and pending_pods <= 1:
                    status = "HEALTHY"
                    message = f"Namespace health good: {running_pods} running, {pending_pods} pending, {failed_pods} failed"
                elif failed_pods <= total_pods * 0.1:  # Less than 10% failed
                    status = "DEGRADED"
                    message = f"Some issues in namespace: {running_pods} running, {pending_pods} pending, {failed_pods} failed"
                else:
                    status = "UNHEALTHY"
                    message = f"Namespace has issues: {running_pods} running, {pending_pods} pending, {failed_pods} failed"
                
                self.checks.append(HealthCheckResult(
                    check_name="namespace_pods",
                    component="Kubernetes",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={
                        "total_pods": total_pods,
                        "running_pods": running_pods,
                        "pending_pods": pending_pods,
                        "failed_pods": failed_pods
                    }
                ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="namespace_pods",
                component="Kubernetes",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"Error checking namespace health: {str(e)}"
            ))
    
    async def check_monitoring_health(self):
        """Check monitoring stack health"""
        logger.info("Checking monitoring stack health...")
        
        monitoring_services = ['prometheus', 'grafana', 'alertmanager']
        
        for service in monitoring_services:
            start_time = time.time()
            try:
                result = subprocess.run([
                    "kubectl", "get", "pods", "-l", f"app={service}",
                    "-n", self.namespace, "-o", "json"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    pods_data = json.loads(result.stdout)
                    pods = pods_data.get('items', [])
                    
                    if pods:
                        running_pods = sum(
                            1 for pod in pods 
                            if pod.get('status', {}).get('phase') == 'Running'
                        )
                        total_pods = len(pods)
                        
                        if running_pods == total_pods:
                            status = "HEALTHY"
                            message = f"{service.capitalize()} is healthy ({running_pods} pods running)"
                        elif running_pods > 0:
                            status = "DEGRADED"
                            message = f"{service.capitalize()} partially healthy ({running_pods}/{total_pods} pods running)"
                        else:
                            status = "UNHEALTHY"
                            message = f"{service.capitalize()} is unhealthy (no pods running)"
                    else:
                        status = "UNHEALTHY"
                        message = f"No {service} pods found"
                    
                    self.checks.append(HealthCheckResult(
                        check_name=f"{service}_health",
                        component="Monitoring",
                        status=status,
                        response_time=time.time() - start_time,
                        message=message,
                        details={"service": service, "pod_count": len(pods)}
                    ))
            
            except Exception as e:
                self.checks.append(HealthCheckResult(
                    check_name=f"{service}_health",
                    component="Monitoring",
                    status="UNHEALTHY",
                    response_time=time.time() - start_time,
                    message=f"Error checking {service} health: {str(e)}"
                ))
    
    async def check_security_health(self):
        """Check security-related health"""
        logger.info("Checking security health...")
        
        # Check certificate expiry
        await self._check_certificate_expiry()
        
        # Check security policies
        await self._check_security_policies()
    
    async def _check_certificate_expiry(self):
        """Check SSL certificate expiry"""
        start_time = time.time()
        try:
            parsed_url = urlparse(self.base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
            
            if parsed_url.scheme == 'https':
                context = ssl.create_default_context()
                with socket.create_connection((hostname, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()
                        
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days
                        
                        if days_until_expiry > 30:
                            status = "HEALTHY"
                            message = f"SSL certificate expires in {days_until_expiry} days"
                        elif days_until_expiry > 7:
                            status = "DEGRADED"
                            message = f"SSL certificate expires soon ({days_until_expiry} days)"
                        else:
                            status = "UNHEALTHY"
                            message = f"SSL certificate expires very soon ({days_until_expiry} days)"
                        
                        self.checks.append(HealthCheckResult(
                            check_name="ssl_certificate",
                            component="Security",
                            status=status,
                            response_time=time.time() - start_time,
                            message=message,
                            details={
                                "expiry_date": cert['notAfter'],
                                "days_until_expiry": days_until_expiry,
                                "issuer": cert.get('issuer', 'Unknown')
                            }
                        ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="ssl_certificate",
                component="Security",
                status="DEGRADED",
                response_time=time.time() - start_time,
                message=f"Could not check SSL certificate: {str(e)}"
            ))
    
    async def _check_security_policies(self):
        """Check security policies"""
        start_time = time.time()
        try:
            # Check network policies
            result = subprocess.run([
                "kubectl", "get", "networkpolicies", "-n", self.namespace, "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                policies_data = json.loads(result.stdout)
                policies = policies_data.get('items', [])
                
                if policies:
                    status = "HEALTHY"
                    message = f"Network policies are configured ({len(policies)} policies)"
                else:
                    status = "DEGRADED"
                    message = "No network policies found"
                
                self.checks.append(HealthCheckResult(
                    check_name="network_policies",
                    component="Security",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={"policy_count": len(policies)}
                ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="network_policies",
                component="Security",
                status="DEGRADED",
                response_time=time.time() - start_time,
                message=f"Could not check network policies: {str(e)}"
            ))
    
    async def check_storage_health(self):
        """Check storage health"""
        logger.info("Checking storage health...")
        
        start_time = time.time()
        try:
            # Check persistent volumes
            result = subprocess.run([
                "kubectl", "get", "pvc", "-n", self.namespace, "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pvc_data = json.loads(result.stdout)
                pvcs = pvc_data.get('items', [])
                
                bound_pvcs = 0
                total_pvcs = len(pvcs)
                
                for pvc in pvcs:
                    phase = pvc.get('status', {}).get('phase', 'Unknown')
                    if phase == 'Bound':
                        bound_pvcs += 1
                
                if bound_pvcs == total_pvcs and total_pvcs > 0:
                    status = "HEALTHY"
                    message = f"All {total_pvcs} PVCs are bound"
                elif bound_pvcs > 0:
                    status = "DEGRADED"
                    message = f"{bound_pvcs}/{total_pvcs} PVCs are bound"
                else:
                    status = "UNHEALTHY"
                    message = "No PVCs are bound"
                
                self.checks.append(HealthCheckResult(
                    check_name="persistent_volumes",
                    component="Storage",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={"bound_pvcs": bound_pvcs, "total_pvcs": total_pvcs}
                ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="persistent_volumes",
                component="Storage",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"Error checking storage health: {str(e)}"
            ))
    
    async def check_network_connectivity(self):
        """Check network connectivity"""
        logger.info("Checking network connectivity...")
        
        # Check DNS resolution
        start_time = time.time()
        try:
            parsed_url = urlparse(self.base_url)
            hostname = parsed_url.hostname
            
            socket.gethostbyname(hostname)
            
            self.checks.append(HealthCheckResult(
                check_name="dns_resolution",
                component="Network",
                status="HEALTHY",
                response_time=time.time() - start_time,
                message=f"DNS resolution successful for {hostname}"
            ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="dns_resolution",
                component="Network",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"DNS resolution failed: {str(e)}"
            ))
        
        # Check service connectivity
        await self._check_service_connectivity()
    
    async def _check_service_connectivity(self):
        """Check internal service connectivity"""
        start_time = time.time()
        try:
            result = subprocess.run([
                "kubectl", "get", "services", "-n", self.namespace, "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                services_data = json.loads(result.stdout)
                services = services_data.get('items', [])
                
                service_count = len(services)
                
                if service_count > 0:
                    status = "HEALTHY"
                    message = f"Found {service_count} services in namespace"
                else:
                    status = "UNHEALTHY"
                    message = "No services found in namespace"
                
                self.checks.append(HealthCheckResult(
                    check_name="service_connectivity",
                    component="Network",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={"service_count": service_count}
                ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="service_connectivity",
                component="Network",
                status="UNHEALTHY",
                response_time=time.time() - start_time,
                message=f"Error checking service connectivity: {str(e)}"
            ))
    
    async def check_ssl_certificates(self):
        """Check SSL certificate health"""
        # Already covered in security health check
        pass
    
    async def check_resource_utilization(self):
        """Check resource utilization"""
        logger.info("Checking resource utilization...")
        
        start_time = time.time()
        try:
            # Check node resource usage
            result = subprocess.run([
                "kubectl", "top", "nodes"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                high_cpu_nodes = 0
                high_memory_nodes = 0
                total_nodes = len(lines)
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cpu_usage = parts[2]
                        memory_usage = parts[4]
                        
                        # Extract percentage values
                        if cpu_usage.endswith('%'):
                            cpu_percent = int(cpu_usage[:-1])
                            if cpu_percent > 80:
                                high_cpu_nodes += 1
                        
                        if memory_usage.endswith('%'):
                            memory_percent = int(memory_usage[:-1])
                            if memory_percent > 80:
                                high_memory_nodes += 1
                
                if high_cpu_nodes == 0 and high_memory_nodes == 0:
                    status = "HEALTHY"
                    message = "Resource utilization is normal"
                elif high_cpu_nodes <= 1 and high_memory_nodes <= 1:
                    status = "DEGRADED"
                    message = f"Some nodes have high resource usage (CPU: {high_cpu_nodes}, Memory: {high_memory_nodes})"
                else:
                    status = "UNHEALTHY"
                    message = f"Many nodes have high resource usage (CPU: {high_cpu_nodes}, Memory: {high_memory_nodes})"
                
                self.checks.append(HealthCheckResult(
                    check_name="resource_utilization",
                    component="Performance",
                    status=status,
                    response_time=time.time() - start_time,
                    message=message,
                    details={
                        "total_nodes": total_nodes,
                        "high_cpu_nodes": high_cpu_nodes,
                        "high_memory_nodes": high_memory_nodes
                    }
                ))
        
        except Exception as e:
            self.checks.append(HealthCheckResult(
                check_name="resource_utilization",
                component="Performance",
                status="DEGRADED",
                response_time=time.time() - start_time,
                message=f"Could not check resource utilization: {str(e)}"
            ))
    
    def _generate_report(self) -> SystemHealthReport:
        """Generate comprehensive health report"""
        timestamp = datetime.now().isoformat()
        
        # Count status types
        healthy_count = sum(1 for check in self.checks if check.status == "HEALTHY")
        degraded_count = sum(1 for check in self.checks if check.status == "DEGRADED")
        unhealthy_count = sum(1 for check in self.checks if check.status == "UNHEALTHY")
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "UNHEALTHY"
        elif degraded_count > 0:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"
        
        # Calculate performance summary
        response_times = [check.response_time for check in self.checks]
        performance_summary = {
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return SystemHealthReport(
            environment=self.environment,
            timestamp=timestamp,
            overall_status=overall_status,
            total_checks=len(self.checks),
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            checks=self.checks,
            performance_summary=performance_summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Check for unhealthy components
        unhealthy_checks = [check for check in self.checks if check.status == "UNHEALTHY"]
        if unhealthy_checks:
            recommendations.append(f"Immediate attention needed: {len(unhealthy_checks)} unhealthy components detected")
        
        # Check for degraded components
        degraded_checks = [check for check in self.checks if check.status == "DEGRADED"]
        if degraded_checks:
            recommendations.append(f"Monitor closely: {len(degraded_checks)} components in degraded state")
        
        # Specific recommendations
        components_with_issues = set(check.component for check in self.checks if check.status != "HEALTHY")
        
        if "Database" in components_with_issues:
            recommendations.append("Database issues detected. Check connection pools and query performance.")
        
        if "API Service" in components_with_issues:
            recommendations.append("API service issues detected. Review logs and scaling configuration.")
        
        if "Kubernetes" in components_with_issues:
            recommendations.append("Kubernetes issues detected. Check node and pod health.")
        
        if "Security" in components_with_issues:
            recommendations.append("Security issues detected. Review certificates and policies.")
        
        # Performance recommendations
        slow_checks = [check for check in self.checks if check.response_time > 5.0]
        if slow_checks:
            recommendations.append("Some health checks are slow. Investigate performance bottlenecks.")
        
        if not recommendations:
            recommendations.append("System health looks good! Continue regular monitoring.")
        
        return recommendations

class HealthReportGenerator:
    """Generate health reports in various formats"""
    
    @staticmethod
    def generate_json_report(report: SystemHealthReport, output_file: str):
        """Generate JSON report"""
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
    
    @staticmethod
    def generate_html_report(report: SystemHealthReport, output_file: str):
        """Generate HTML report"""
        status_colors = {
            "HEALTHY": "#4caf50",
            "DEGRADED": "#ff9800",
            "UNHEALTHY": "#f44336"
        }
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>System Health Report - {report.environment}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .status-badge {{ padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }}
        .healthy {{ background: #4caf50; }}
        .degraded {{ background: #ff9800; }}
        .unhealthy {{ background: #f44336; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .check {{ margin: 10px 0; padding: 15px; border-left: 4px solid #ccc; background: #fafafa; }}
        .check.healthy {{ border-left-color: #4caf50; }}
        .check.degraded {{ border-left-color: #ff9800; }}
        .check.unhealthy {{ border-left-color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>System Health Report</h1>
        <p><strong>Environment:</strong> {report.environment}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        <p><strong>Overall Status:</strong> 
           <span class="status-badge {report.overall_status.lower()}">{report.overall_status}</span>
        </p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{report.total_checks}</h3>
            <p>Total Checks</p>
        </div>
        <div class="metric">
            <h3 style="color: #4caf50">{report.healthy_count}</h3>
            <p>Healthy</p>
        </div>
        <div class="metric">
            <h3 style="color: #ff9800">{report.degraded_count}</h3>
            <p>Degraded</p>
        </div>
        <div class="metric">
            <h3 style="color: #f44336">{report.unhealthy_count}</h3>
            <p>Unhealthy</p>
        </div>
    </div>
    
    <h2>Health Checks</h2>
"""
        
        # Group checks by component
        components = {}
        for check in report.checks:
            if check.component not in components:
                components[check.component] = []
            components[check.component].append(check)
        
        for component, checks in components.items():
            html_content += f"<h3>{component}</h3>"
            for check in checks:
                html_content += f"""
    <div class="check {check.status.lower()}">
        <h4>{check.check_name} <span class="status-badge {check.status.lower()}">{check.status}</span></h4>
        <p><strong>Message:</strong> {check.message}</p>
        <p><strong>Response Time:</strong> {check.response_time:.3f}s</p>
        {f'<p><strong>Details:</strong> {json.dumps(check.details, indent=2)}</p>' if check.details else ''}
    </div>
"""
        
        if report.recommendations:
            html_content += """
    <h2>Recommendations</h2>
    <ul>
"""
            for rec in report.recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MLOps Platform Health Validator')
    parser.add_argument('--environment', required=True, help='Environment name (staging/production)')
    parser.add_argument('--base-url', required=True, help='Base URL of the application')
    parser.add_argument('--namespace', required=True, help='Kubernetes namespace')
    parser.add_argument('--output-dir', default='./health-reports', help='Output directory for reports')
    parser.add_argument('--format', choices=['json', 'html', 'both'], default='both', help='Report format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run health validation
    validator = HealthValidator(args.environment, args.base_url, args.namespace)
    report = await validator.run_comprehensive_health_check()
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.format in ['json', 'both']:
        json_file = output_dir / f'health-report-{args.environment}-{timestamp}.json'
        HealthReportGenerator.generate_json_report(report, str(json_file))
        logger.info(f"JSON report generated: {json_file}")
    
    if args.format in ['html', 'both']:
        html_file = output_dir / f'health-report-{args.environment}-{timestamp}.html'
        HealthReportGenerator.generate_html_report(report, str(html_file))
        logger.info(f"HTML report generated: {html_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SYSTEM HEALTH SUMMARY")
    print(f"{'='*60}")
    print(f"Environment: {report.environment}")
    print(f"Overall Status: {report.overall_status}")
    print(f"Total Checks: {report.total_checks}")
    print(f"Healthy: {report.healthy_count}")
    print(f"Degraded: {report.degraded_count}")
    print(f"Unhealthy: {report.unhealthy_count}")
    print(f"{'='*60}")
    
    # Print recommendations
    if report.recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        print(f"{'='*60}")
    
    # Exit with appropriate code
    if report.overall_status == "UNHEALTHY":
        logger.error("System health is unhealthy!")
        return 2
    elif report.overall_status == "DEGRADED":
        logger.warning("System health is degraded!")
        return 1
    else:
        logger.info("System health is good!")
        return 0

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)