"""Main entry point for the model server."""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI

from pynomaly.application.services.deployment_orchestration_service import DeploymentOrchestrationService
from pynomaly.application.services.model_registry_service import ModelRegistryService
from pynomaly.infrastructure.serving.model_server import create_model_server
from pynomaly.domain.entities.deployment import Environment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/model_server.log') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class ModelServerManager:
    """Manager for the model server lifecycle."""
    
    def __init__(self):
        self.app: Optional[FastAPI] = None
        self.server_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
    
    async def initialize_services(self) -> tuple[DeploymentOrchestrationService, ModelRegistryService]:
        """Initialize required services."""
        # Get configuration from environment
        storage_path = Path(os.getenv('STORAGE_PATH', '/app/data'))
        container_registry = os.getenv('CONTAINER_REGISTRY_URL', 'pynomaly-registry.local')
        kubernetes_config = os.getenv('KUBERNETES_CONFIG_PATH')
        
        # Initialize model registry service
        model_registry_service = ModelRegistryService(
            storage_path=storage_path / 'registry'
        )
        
        # Initialize deployment orchestration service
        deployment_service = DeploymentOrchestrationService(
            model_registry_service=model_registry_service,
            storage_path=storage_path / 'deployments',
            container_registry_url=container_registry,
            kubernetes_config_path=Path(kubernetes_config) if kubernetes_config else None
        )
        
        return deployment_service, model_registry_service
    
    async def start_server(self):
        """Start the model server."""
        try:
            logger.info("Initializing Pynomaly Model Server...")
            
            # Initialize services
            deployment_service, model_registry_service = await self.initialize_services()
            
            # Get environment configuration
            environment_name = os.getenv('ENVIRONMENT', 'production').lower()
            environment = Environment(environment_name)
            
            # Create model server
            model_server = create_model_server(
                deployment_service=deployment_service,
                model_registry_service=model_registry_service,
                environment=environment
            )
            
            self.app = model_server.app
            
            # Configure server
            host = os.getenv('MODEL_SERVER_HOST', '0.0.0.0')
            port = int(os.getenv('MODEL_SERVER_PORT', '8080'))
            workers = int(os.getenv('WORKERS', '1'))
            log_level = os.getenv('LOG_LEVEL', 'info').lower()
            
            logger.info(f"Starting server on {host}:{port} with {workers} workers")
            logger.info(f"Environment: {environment.value}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                workers=workers if workers > 1 else None,
                log_level=log_level,
                access_log=True,
                reload=False,  # Disable reload in production
                loop="asyncio"
            )
            
            # Start server
            server = uvicorn.Server(config)
            
            # Handle graceful shutdown
            async def shutdown_handler():
                await self.shutdown_event.wait()
                logger.info("Received shutdown signal, stopping server...")
                server.should_exit = True
            
            shutdown_task = asyncio.create_task(shutdown_handler())
            
            # Start server
            try:
                await server.serve()
            except Exception as e:
                logger.error(f"Server error: {e}")
                raise
            finally:
                shutdown_task.cancel()
                try:
                    await shutdown_task
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        # Handle common shutdown signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Handle additional signals on Unix systems
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)
    
    async def shutdown(self):
        """Shutdown the server gracefully."""
        logger.info("Initiating graceful shutdown...")
        self.shutdown_event.set()
        
        # Allow some time for ongoing requests to complete
        await asyncio.sleep(1)
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    try:
        # Create server manager
        server_manager = ModelServerManager()
        
        # Setup signal handlers
        server_manager.setup_signal_handlers()
        
        # Start server
        await server_manager.start_server()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


def run_server():
    """Synchronous entry point for running the server."""
    try:
        # Check for required environment variables
        required_env_vars = []
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            sys.exit(1)
        
        # Run the async main function
        asyncio.run(main())
        
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()