"""
Gunicorn configuration for Pynomaly production deployment.
Optimized for horizontal scaling with multiple workers.
"""

import multiprocessing
import os
from pathlib import Path

# Get configuration from environment variables
def get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

# Server socket configuration
bind = os.getenv('BIND', '0.0.0.0:8000')
backlog = get_env_int('BACKLOG', 2048)

# Worker processes
cpu_count = multiprocessing.cpu_count()
workers = get_env_int('WORKERS', min(max(2, cpu_count * 2 + 1), 12))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
worker_connections = get_env_int('WORKER_CONNECTIONS', 1000)
max_requests = get_env_int('MAX_REQUESTS', 1000)
max_requests_jitter = get_env_int('MAX_REQUESTS_JITTER', 50)
preload_app = get_env_bool('PRELOAD_APP', True)
timeout = get_env_int('TIMEOUT', 30)
keepalive = get_env_int('KEEPALIVE', 2)

# Logging configuration
accesslog = os.getenv('ACCESS_LOG', '-')
errorlog = os.getenv('ERROR_LOG', '-')
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = os.getenv('ACCESS_LOG_FORMAT', 
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s')

# Process naming
proc_name = os.getenv('PROC_NAME', 'pynomaly-api')

# Security settings
limit_request_line = get_env_int('LIMIT_REQUEST_LINE', 4096)
limit_request_fields = get_env_int('LIMIT_REQUEST_FIELDS', 100)
limit_request_field_size = get_env_int('LIMIT_REQUEST_FIELD_SIZE', 8190)

# Performance optimizations
# Use RAM for temporary files if available
if os.path.exists('/dev/shm'):
    worker_tmp_dir = '/dev/shm'
else:
    worker_tmp_dir = None

# SSL configuration (if certificates are provided)
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')
ca_certs = os.getenv('SSL_CA_CERTS')
cert_reqs = get_env_int('SSL_CERT_REQS', 0)  # 0 = ssl.CERT_NONE

# User/Group (for privilege dropping)
user = os.getenv('USER')
group = os.getenv('GROUP')

# Daemon mode (typically False for containerized deployments)
daemon = get_env_bool('DAEMON', False)

# PID file
pidfile = os.getenv('PIDFILE')

# Environment-specific configurations
environment = os.getenv('ENVIRONMENT', 'production')

if environment == 'development':
    # Development settings
    workers = 1
    reload = True
    preload_app = False
    timeout = 120
    loglevel = 'debug'
elif environment == 'production':
    # Production settings
    reload = False
    preload_app = True
    timeout = 30
    loglevel = 'info'
elif environment == 'testing':
    # Testing settings
    workers = 2
    reload = False
    preload_app = True
    timeout = 60
    loglevel = 'warning'

# Hooks for application lifecycle
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Pynomaly API server...")
    server.log.info(f"Environment: {environment}")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Worker class: {worker_class}")
    server.log.info(f"Worker connections: {worker_connections}")
    server.log.info(f"Bind: {bind}")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Pynomaly API server...")

def worker_int(worker):
    """Called just after a worker has been killed by a signal."""
    worker.log.info(f"Worker {worker.pid} killed by signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info(f"Pre-fork worker {worker.age}")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Post-fork worker {worker.pid}")

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info(f"Worker {worker.pid} initialized")

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info(f"Worker {worker.pid} aborted")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forking new master process")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Pynomaly API server is ready to accept connections")

def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker {worker.pid} exited")

def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info(f"Number of workers changed from {old_value} to {new_value}")

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Pynomaly API server is shutting down...")

# Custom application factory
def create_app():
    """Create the FastAPI application."""
    from pynomaly.presentation.api.app import create_app
    return create_app()

# Log configuration details
print(f"Gunicorn Configuration:")
print(f"  Environment: {environment}")
print(f"  Workers: {workers}")
print(f"  Worker class: {worker_class}")
print(f"  Worker connections: {worker_connections}")
print(f"  Bind: {bind}")
print(f"  Timeout: {timeout}")
print(f"  Preload app: {preload_app}")
print(f"  Log level: {loglevel}")
print(f"  Max requests: {max_requests}")
print(f"  Max requests jitter: {max_requests_jitter}")
print(f"  Process name: {proc_name}")

# Export configuration for external tools
__all__ = [
    'bind', 'backlog', 'workers', 'worker_class', 'worker_connections',
    'max_requests', 'max_requests_jitter', 'preload_app', 'timeout',
    'keepalive', 'accesslog', 'errorlog', 'loglevel', 'proc_name',
    'limit_request_line', 'limit_request_fields', 'limit_request_field_size',
    'worker_tmp_dir', 'keyfile', 'certfile', 'ca_certs', 'cert_reqs',
    'user', 'group', 'daemon', 'pidfile'
]
