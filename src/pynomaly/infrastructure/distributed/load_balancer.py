"""Load balancer for distributing requests across multiple Pynomaly instances."""

import asyncio
import logging
import random
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import hashlib

from pynomaly.domain.exceptions import ProcessingError


logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    HEALTH_AWARE = "health_aware"


@dataclass
class ServerInstance:
    """Represents a server instance in the load balancer."""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    
    @property
    def url(self) -> str:
        """Get the full URL for this server."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def load_percentage(self) -> float:
        """Get current load as percentage."""
        return (self.current_connections / self.max_connections) * 100 if self.max_connections > 0 else 100
    
    @property
    def is_available(self) -> bool:
        """Check if server is available for new requests."""
        return (
            self.is_healthy and
            self.current_connections < self.max_connections and
            (self.last_health_check is None or 
             datetime.now() - self.last_health_check < timedelta(minutes=2))
        )
    
    def update_metrics(self, success: bool, response_time: float) -> None:
        """Update server metrics."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        # Update error rate (exponential moving average)
        current_error_rate = self.failed_requests / self.total_requests
        self.error_rate = 0.9 * self.error_rate + 0.1 * current_error_rate
        
        # Update response time (exponential moving average)
        self.response_time_avg = 0.9 * self.response_time_avg + 0.1 * response_time


class LoadBalancer:
    """Load balancer for distributing requests across multiple Pynomaly instances."""
    
    def __init__(self,
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
                 health_check_interval: int = 30,
                 health_check_timeout: int = 10,
                 max_retries: int = 3):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_retries = max_retries
        
        # Server management
        self.servers: Dict[str, ServerInstance] = {}
        self.server_locks: Dict[str, asyncio.Lock] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.consistent_hash_ring: Dict[int, str] = {}
        
        # Background tasks
        self._running = False
        self._health_checker: Optional[asyncio.Task] = None
        
        # HTTP client for health checks and request forwarding
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Load balancer initialized with strategy: {strategy.value}")
    
    async def start(self) -> None:
        """Start the load balancer."""
        if self._running:
            return
        
        self._running = True
        self._session = aiohttp.ClientSession()
        
        # Start health checking
        self._health_checker = asyncio.create_task(self._health_check_loop())
        
        logger.info("Load balancer started")
    
    async def stop(self) -> None:
        """Stop the load balancer."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel health checker
        if self._health_checker:
            self._health_checker.cancel()
            try:
                await self._health_checker
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self._session:
            await self._session.close()
        
        logger.info("Load balancer stopped")
    
    async def add_server(self,
                        server_id: str,
                        host: str,
                        port: int,
                        weight: int = 1,
                        max_connections: int = 100) -> bool:
        """Add a server to the load balancer."""
        if server_id in self.servers:
            logger.warning(f"Server {server_id} already exists")
            return False
        
        server = ServerInstance(
            id=server_id,
            host=host,
            port=port,
            weight=weight,
            max_connections=max_connections
        )
        
        self.servers[server_id] = server
        self.server_locks[server_id] = asyncio.Lock()
        
        # Update consistent hash ring if using that strategy
        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._update_consistent_hash_ring()
        
        logger.info(f"Added server {server_id} at {host}:{port}")
        return True
    
    async def remove_server(self, server_id: str) -> bool:
        """Remove a server from the load balancer."""
        if server_id not in self.servers:
            return False
        
        # Wait for current connections to finish
        server = self.servers[server_id]
        while server.current_connections > 0:
            await asyncio.sleep(0.5)
        
        # Remove server
        del self.servers[server_id]
        del self.server_locks[server_id]
        
        # Update consistent hash ring if using that strategy
        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._update_consistent_hash_ring()
        
        logger.info(f"Removed server {server_id}")
        return True
    
    async def forward_request(self,
                            method: str,
                            path: str,
                            headers: Optional[Dict[str, str]] = None,
                            json_data: Optional[Dict[str, Any]] = None,
                            params: Optional[Dict[str, str]] = None,
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """Forward a request to an appropriate server."""
        for retry in range(self.max_retries):
            try:
                # Select server based on strategy
                server = await self._select_server(session_id)
                if not server:
                    raise ProcessingError("No available servers")
                
                # Forward request
                result = await self._forward_to_server(
                    server=server,
                    method=method,
                    path=path,
                    headers=headers,
                    json_data=json_data,
                    params=params
                )
                
                return result
                
            except Exception as e:
                logger.warning(f"Request failed on retry {retry + 1}: {e}")
                if retry == self.max_retries - 1:
                    raise ProcessingError(f"Request failed after {self.max_retries} retries: {e}")
                
                # Wait before retry
                await asyncio.sleep(min(2 ** retry, 10))
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers."""
        servers_status = {}
        
        for server_id, server in self.servers.items():
            servers_status[server_id] = {
                "id": server.id,
                "host": server.host,
                "port": server.port,
                "weight": server.weight,
                "is_healthy": server.is_healthy,
                "is_available": server.is_available,
                "current_connections": server.current_connections,
                "max_connections": server.max_connections,
                "load_percentage": server.load_percentage,
                "response_time_avg": server.response_time_avg,
                "error_rate": server.error_rate,
                "total_requests": server.total_requests,
                "failed_requests": server.failed_requests,
                "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None
            }
        
        return {
            "strategy": self.strategy.value,
            "total_servers": len(self.servers),
            "healthy_servers": len([s for s in self.servers.values() if s.is_healthy]),
            "available_servers": len([s for s in self.servers.values() if s.is_available]),
            "servers": servers_status
        }
    
    async def _select_server(self, session_id: Optional[str] = None) -> Optional[ServerInstance]:
        """Select a server based on the configured strategy."""
        available_servers = [s for s in self.servers.values() if s.is_available]
        
        if not available_servers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_servers)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(available_servers)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(available_servers)
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_servers)
        
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._select_consistent_hash(available_servers, session_id)
        
        elif self.strategy == LoadBalancingStrategy.HEALTH_AWARE:
            return self._select_health_aware(available_servers)
        
        else:
            return available_servers[0]
    
    def _select_round_robin(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server using round-robin strategy."""
        server = servers[self.round_robin_index % len(servers)]
        self.round_robin_index += 1
        return server
    
    def _select_least_connections(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server with least connections."""
        return min(servers, key=lambda s: s.current_connections)
    
    def _select_weighted_round_robin(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server using weighted round-robin strategy."""
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)
        
        if not weighted_servers:
            return servers[0]
        
        server = weighted_servers[self.round_robin_index % len(weighted_servers)]
        self.round_robin_index += 1
        return server
    
    def _select_consistent_hash(self, servers: List[ServerInstance], session_id: Optional[str]) -> ServerInstance:
        """Select server using consistent hashing."""
        if not session_id:
            return random.choice(servers)
        
        # Hash the session ID
        hash_key = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        
        # Find the server in the consistent hash ring
        if not self.consistent_hash_ring:
            return random.choice(servers)
        
        # Find the first server with a hash >= our key
        for ring_hash in sorted(self.consistent_hash_ring.keys()):
            if ring_hash >= hash_key:
                server_id = self.consistent_hash_ring[ring_hash]
                if server_id in self.servers and self.servers[server_id] in servers:
                    return self.servers[server_id]
        
        # Wrap around to the first server
        first_hash = min(self.consistent_hash_ring.keys())
        server_id = self.consistent_hash_ring[first_hash]
        if server_id in self.servers and self.servers[server_id] in servers:
            return self.servers[server_id]
        
        return random.choice(servers)
    
    def _select_health_aware(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server based on health metrics (response time and error rate)."""
        # Score servers based on response time and error rate
        scored_servers = []
        
        for server in servers:
            # Lower score is better
            response_score = server.response_time_avg / 1000  # Normalize to seconds
            error_score = server.error_rate * 10  # Weight error rate heavily
            load_score = server.load_percentage / 100
            
            total_score = response_score + error_score + load_score
            scored_servers.append((total_score, server))
        
        # Sort by score and return best server
        scored_servers.sort(key=lambda x: x[0])
        return scored_servers[0][1]
    
    def _update_consistent_hash_ring(self) -> None:
        """Update the consistent hash ring for consistent hashing strategy."""
        self.consistent_hash_ring = {}
        
        for server in self.servers.values():
            # Add multiple virtual nodes for better distribution
            for i in range(server.weight * 100):  # 100 virtual nodes per weight
                virtual_key = f"{server.id}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self.consistent_hash_ring[hash_value] = server.id
    
    async def _forward_to_server(self,
                                server: ServerInstance,
                                method: str,
                                path: str,
                                headers: Optional[Dict[str, str]] = None,
                                json_data: Optional[Dict[str, Any]] = None,
                                params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Forward request to a specific server."""
        start_time = time.time()
        success = False
        
        try:
            # Increment connection count
            async with self.server_locks[server.id]:
                server.current_connections += 1
            
            # Build URL
            url = f"{server.url}{path}"
            
            # Forward request
            async with self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 400:
                    raise ProcessingError(f"Server returned status {response.status}")
                
                result = await response.json()
                success = True
                return result
        
        except Exception as e:
            logger.error(f"Request to server {server.id} failed: {e}")
            raise
        
        finally:
            # Update metrics
            response_time = time.time() - start_time
            server.update_metrics(success, response_time)
            
            # Decrement connection count
            async with self.server_locks[server.id]:
                server.current_connections = max(0, server.current_connections - 1)
    
    async def _health_check_loop(self) -> None:
        """Perform periodic health checks on all servers."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all servers."""
        check_tasks = []
        
        for server in self.servers.values():
            task = asyncio.create_task(self._check_server_health(server))
            check_tasks.append(task)
        
        if check_tasks:
            await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def _check_server_health(self, server: ServerInstance) -> None:
        """Check health of a specific server."""
        try:
            start_time = time.time()
            
            async with self._session.get(
                f"{server.url}/health",
                timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    server.is_healthy = True
                    server.last_health_check = datetime.now()
                    
                    # Update response time
                    server.response_time_avg = 0.8 * server.response_time_avg + 0.2 * response_time
                else:
                    server.is_healthy = False
                    logger.warning(f"Health check failed for server {server.id}: status {response.status}")
        
        except Exception as e:
            server.is_healthy = False
            logger.warning(f"Health check failed for server {server.id}: {e}")


async def main():
    """Main function for running load balancer as standalone service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pynomaly Load Balancer")
    parser.add_argument("--strategy", choices=[s.value for s in LoadBalancingStrategy], 
                       default="round_robin", help="Load balancing strategy")
    parser.add_argument("--port", type=int, default=8080, help="Load balancer port")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create load balancer
    strategy = LoadBalancingStrategy(args.strategy)
    lb = LoadBalancer(strategy=strategy)
    
    # Add some example servers
    await lb.add_server("server1", "localhost", 8001)
    await lb.add_server("server2", "localhost", 8002)
    await lb.add_server("server3", "localhost", 8003)
    
    try:
        await lb.start()
        logger.info(f"Load balancer started on port {args.port}")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down load balancer...")
    finally:
        await lb.stop()


if __name__ == "__main__":
    asyncio.run(main())