"""
Production Gunicorn configuration for Pynomaly API
"""

import multiprocessing
import os

# =============================================================================
# SERVER SOCKET
# =============================================================================

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# =============================================================================
# WORKER PROCESSES
# =============================================================================

# Worker count calculation: (2 x CPU cores) + 1
workers = os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1)
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = int(os.getenv('MAX_REQUESTS', '1000'))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', '100'))

# =============================================================================
# TIMEOUTS
# =============================================================================

timeout = int(os.getenv('TIMEOUT', '120'))
keepalive = int(os.getenv('KEEPALIVE', '5'))
graceful_timeout = 30

# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

preload_app = True
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# =============================================================================
# LOGGING
# =============================================================================

# Logging configuration
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
errorlog = '-'  # stderr
accesslog = '-'  # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Custom log format for production
if os.getenv('LOG_FORMAT', 'text') == 'json':
    access_log_format = 'remote_ip=%(h)s request_id=%({X-Request-ID}i)s method=%(m)s url="%(U)s" query="%(q)s" status=%(s)s size=%(b)s duration=%(D)s user_agent="%(a)s"'

# =============================================================================
# SSL/TLS (if terminating SSL at application level)
# =============================================================================

# Uncomment if handling SSL at Gunicorn level
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
# ssl_version = 2  # SSLv23
# ciphers = 'TLSv1'

# =============================================================================
# SECURITY
# =============================================================================

# Limit request line size to prevent attacks
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Worker recycling for memory management
max_requests = 1000
max_requests_jitter = 100

# Pre-load application for better performance
preload_app = True

# =============================================================================
# DEVELOPMENT/DEBUGGING
# =============================================================================

# Only enable in development
if os.getenv('ENVIRONMENT') == 'development':
    reload = True
    reload_extra_files = ['src/']
else:
    reload = False

# =============================================================================
# HOOKS
# =============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Pynomaly API server")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Pynomaly API server")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Pynomaly API server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker has been killed by SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.debug("Worker %s is about to be forked", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.debug("Worker %s has been forked", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker %s initialized", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.error("Worker %s aborted", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forking new master process")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    # Add request ID for tracing
    worker.log.debug("Processing request: %s %s", req.method, req.path)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.debug("Completed request: %s %s - %s", req.method, req.path, resp.status)

def child_exit(server, worker):
    """Called just after a worker has been reaped."""
    server.log.info("Worker %s exited", worker.pid)

def worker_exit(server, worker):
    """Called just after a worker has been reaped."""
    server.log.info("Worker %s has exited", worker.pid)

def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info("Number of workers changed from %s to %s", old_value, new_value)

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down Pynomaly API server")

# =============================================================================
# ERROR HANDLING
# =============================================================================

def worker_connections_handler(worker, req, client, server):
    """Custom handler for worker connection issues."""
    worker.log.warning("Worker connection issue: %s", req.path)

# =============================================================================
# HEALTH CHECK CONFIGURATION
# =============================================================================

# Custom application configuration
raw_env = [
    'GUNICORN_WORKER_ID=%s' % os.getpid(),
    'GUNICORN_WORKER_COUNT=%s' % workers,
]

# =============================================================================
# MONITORING INTEGRATION
# =============================================================================

# Prometheus metrics endpoint configuration
if os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true':
    raw_env.append('PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir')

# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

# Graceful worker restarts
max_requests = 1000
max_requests_jitter = 100
graceful_timeout = 30

# =============================================================================
# CUSTOMIZATION BASED ON ENVIRONMENT
# =============================================================================

# Production optimizations
if os.getenv('ENVIRONMENT') == 'production':
    # More aggressive worker recycling in production
    max_requests = 500
    max_requests_jitter = 50
    
    # Longer timeouts for complex ML operations
    timeout = 300
    
    # More workers for higher throughput
    workers = max(4, multiprocessing.cpu_count() * 2)
    
    # Enhanced logging
    loglevel = 'warning'
    
elif os.getenv('ENVIRONMENT') == 'development':
    # Development-friendly settings
    workers = 1
    reload = True
    loglevel = 'debug'
    timeout = 0  # No timeout in development

# =============================================================================
# CUSTOM MIDDLEWARE CONFIGURATION
# =============================================================================

# Add custom middleware for production monitoring
def application(environ, start_response):
    """Custom WSGI middleware for additional functionality."""
    # Add custom headers, monitoring, etc.
    pass