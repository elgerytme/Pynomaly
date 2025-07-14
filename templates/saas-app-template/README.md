# SaaS Application Template

A comprehensive Software-as-a-Service application template combining HTMX, Tailwind CSS, FastAPI, Typer CLI, authentication, security, and testing - everything needed for a production-ready SaaS platform.

## Features

### 🌐 Frontend Stack
- **HTMX**: Dynamic interactions without heavy JavaScript
- **Tailwind CSS**: Utility-first styling with custom components
- **Jinja2**: Server-side templating with layout inheritance
- **Alpine.js**: Lightweight JavaScript framework for enhanced interactivity
- **Progressive Web App**: Offline capabilities and mobile optimization

### ⚡ Backend Stack  
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Modern ORM with async support
- **Alembic**: Database migrations
- **Redis**: Caching and session storage
- **Celery**: Background task processing

### 🔐 Authentication & Security
- **JWT Authentication**: Secure token-based auth
- **Multi-Factor Authentication**: TOTP, SMS, and email 2FA
- **OAuth2 Integration**: Google, GitHub, Microsoft providers
- **Role-Based Access Control**: Granular permission system
- **Security Middleware**: CSRF, XSS, SQL injection protection
- **Audit Logging**: Comprehensive security event tracking

### 🛠️ CLI & Management
- **Typer CLI**: Rich command-line interface for administration
- **Database Management**: Migration and seeding commands
- **User Management**: Create, update, and manage users
- **Monitoring Commands**: Health checks and system status
- **Deployment Tools**: Automated deployment and configuration

### 📊 SaaS Features
- **Multi-Tenancy**: Tenant isolation and management
- **Subscription Management**: Plans, billing, and payments
- **Usage Analytics**: Metrics and reporting
- **API Management**: Rate limiting and usage tracking
- **Webhook System**: Event-driven integrations
- **Email System**: Transactional and marketing emails

## Directory Structure

```
saas-app-template/
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
│   ├── docker/           # Docker configurations
│   ├── kubernetes/       # K8s manifests
│   ├── terraform/        # Infrastructure as code
│   └── scripts/          # Deployment scripts
├── docs/                  # Documentation
│   ├── api/              # API documentation
│   ├── user/             # User guides
│   └── admin/            # Admin documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── src/                   # Source code
│   └── saas_app/
│       ├── api/          # FastAPI endpoints
│       ├── auth/         # Authentication system
│       ├── cli/          # Typer CLI commands
│       ├── core/         # Core business logic
│       ├── db/           # Database models
│       ├── email/        # Email templates and services
│       ├── integrations/ # Third-party integrations
│       ├── middleware/   # Custom middleware
│       ├── saas/         # SaaS-specific features
│       ├── static/       # Static assets
│       ├── templates/    # Jinja2 templates
│       ├── utils/        # Utility functions
│       └── workers/      # Background tasks
├── tests/                # Test suites
│   ├── api/             # API tests
│   ├── auth/            # Authentication tests
│   ├── e2e/             # End-to-end tests
│   ├── integration/     # Integration tests
│   ├── performance/     # Performance tests
│   └── unit/            # Unit tests
├── migrations/           # Database migrations
├── scripts/              # Automation scripts
├── .github/              # CI/CD workflows
├── pyproject.toml        # Project configuration
├── docker-compose.yml   # Development environment
├── Dockerfile           # Production container
├── README.md            # Documentation
├── SECURITY.md          # Security policy
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-saas-app
   cd my-saas-app
   ```

2. **Setup environment**:
   ```bash
   cp env/development/.env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Or setup locally**:
   ```bash
   pip install -e ".[dev,test]"
   alembic upgrade head
   npm install && npm run build-css
   ```

5. **Initialize the application**:
   ```bash
   saas-cli setup --admin-email admin@example.com
   ```

6. **Start the application**:
   ```bash
   uvicorn saas_app.main:app --reload
   ```

7. **Access the application**:
   - App: http://localhost:8000
   - Admin: http://localhost:8000/admin
   - API Docs: http://localhost:8000/docs
   - CLI Help: `saas-cli --help`

## SaaS Features

### Multi-Tenancy

```python
from saas_app.saas.tenant_manager import TenantManager

# Create tenant
tenant_manager = TenantManager()
tenant = await tenant_manager.create_tenant(
    name="Acme Corp",
    subdomain="acme",
    plan="premium",
    owner_email="admin@acme.com"
)

# Tenant middleware for request isolation
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    subdomain = request.url.hostname.split('.')[0]
    tenant = await tenant_manager.get_tenant_by_subdomain(subdomain)
    request.state.tenant = tenant
    request.state.db = tenant_manager.get_tenant_db(tenant.id)
    return await call_next(request)
```

### Subscription Management

```python
from saas_app.saas.subscription_manager import SubscriptionManager

# Subscription plans
subscription_manager = SubscriptionManager()

# Create subscription
subscription = await subscription_manager.create_subscription(
    tenant_id=tenant.id,
    plan_id="premium",
    billing_cycle="monthly"
)

# Check feature access
has_access = await subscription_manager.check_feature_access(
    tenant_id=tenant.id,
    feature="advanced_analytics"
)

# Track usage
await subscription_manager.track_usage(
    tenant_id=tenant.id,
    metric="api_calls",
    value=1
)
```

### Usage Analytics

```python
from saas_app.saas.analytics import AnalyticsManager

# Track events
analytics = AnalyticsManager()
await analytics.track_event(
    tenant_id=tenant.id,
    user_id=user.id,
    event="feature_used",
    properties={
        "feature": "dashboard",
        "plan": "premium",
        "session_id": session.id
    }
)

# Generate reports
report = await analytics.generate_usage_report(
    tenant_id=tenant.id,
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)
```

### Payment Integration

```python
from saas_app.saas.payment_processor import PaymentProcessor

# Process payment
payment_processor = PaymentProcessor()
payment = await payment_processor.process_payment(
    tenant_id=tenant.id,
    amount=99.00,
    currency="USD",
    payment_method_id="pm_123456789"
)

# Handle webhook
@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    event = payment_processor.verify_webhook(payload, sig_header)
    await payment_processor.handle_webhook_event(event)
    
    return {"status": "success"}
```

## Frontend Architecture

### HTMX Integration

```html
<!-- Dynamic dashboard -->
<div class="dashboard" 
     hx-get="/api/dashboard/stats"
     hx-trigger="load, every 30s"
     hx-swap="innerHTML">
    <div class="loading-spinner">Loading...</div>
</div>

<!-- Infinite scroll -->
<div class="data-table"
     hx-get="/api/data/list?page=1"
     hx-trigger="load"
     hx-swap="innerHTML">
</div>
<div hx-get="/api/data/list"
     hx-trigger="revealed"
     hx-vals='{"page": "2"}'
     hx-swap="afterend">
</div>

<!-- Real-time notifications -->
<div hx-ws="connect:/ws/notifications"
     hx-swap="innerHTML">
    <div id="notifications"></div>
</div>
```

### Tailwind Components

```html
<!-- Subscription plan card -->
<div class="plan-card">
    <div class="plan-header">
        <h3 class="plan-title">Premium Plan</h3>
        <div class="plan-price">
            <span class="currency">$</span>
            <span class="amount">99</span>
            <span class="period">/month</span>
        </div>
    </div>
    <div class="plan-features">
        <ul class="feature-list">
            <li class="feature-item">✓ Unlimited API calls</li>
            <li class="feature-item">✓ Advanced analytics</li>
            <li class="feature-item">✓ Priority support</li>
        </ul>
    </div>
    <button class="btn btn-primary btn-full"
            hx-post="/api/subscriptions/upgrade"
            hx-vals='{"plan": "premium"}'>
        Upgrade Now
    </button>
</div>
```

### Progressive Web App

```javascript
// Service worker for offline support
self.addEventListener('fetch', (event) => {
    if (event.request.url.includes('/api/')) {
        event.respondWith(
            caches.match(event.request)
                .then((response) => {
                    if (response) {
                        return response;
                    }
                    return fetch(event.request);
                })
        );
    }
});

// Push notifications
self.addEventListener('push', (event) => {
    const data = event.data.json();
    self.registration.showNotification(data.title, {
        body: data.body,
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/badge-72x72.png',
        actions: data.actions
    });
});
```

## API Architecture

### Tenant-Aware Endpoints

```python
from fastapi import APIRouter, Depends
from saas_app.auth.dependencies import get_current_user, get_current_tenant

router = APIRouter()

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    tenant: Tenant = Depends(get_current_tenant),
    user: User = Depends(get_current_user)
):
    """Get tenant-specific dashboard statistics."""
    stats = await analytics_service.get_tenant_stats(tenant.id)
    return {
        "users": stats.user_count,
        "revenue": stats.monthly_revenue,
        "api_calls": stats.api_calls_today,
        "storage_used": stats.storage_usage
    }

@router.post("/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    tenant: Tenant = Depends(get_current_tenant),
    user: User = Depends(get_current_user)
):
    """Create new API key for tenant."""
    api_key = await api_key_service.create_key(
        tenant_id=tenant.id,
        name=request.name,
        permissions=request.permissions,
        created_by=user.id
    )
    return {"api_key": api_key.key, "id": api_key.id}
```

### Rate Limiting & Usage Tracking

```python
from saas_app.middleware.rate_limiting import RateLimitMiddleware

# Tenant-based rate limiting
app.add_middleware(
    RateLimitMiddleware,
    key_func=lambda request: f"tenant:{request.state.tenant.id}",
    rate_limit_func=lambda request: request.state.tenant.plan.api_rate_limit,
    usage_tracker=usage_tracker
)

# API usage tracking
@app.middleware("http")
async def usage_tracking_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        await usage_tracker.track_api_call(
            tenant_id=request.state.tenant.id,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    return await call_next(request)
```

## CLI Administration

### User Management Commands

```python
import typer
from saas_app.cli.commands.users import users_app

# Add user
@users_app.command()
def create(
    email: str = typer.Argument(..., help="User email"),
    tenant_id: str = typer.Option(..., help="Tenant ID"),
    role: str = typer.Option("user", help="User role"),
    send_invite: bool = typer.Option(True, help="Send invitation email")
):
    """Create a new user."""
    user = user_service.create_user(
        email=email,
        tenant_id=tenant_id,
        role=role
    )
    
    if send_invite:
        email_service.send_invitation(user.email, user.invite_token)
    
    typer.echo(f"✅ Created user: {user.email}")

# List users
@users_app.command()
def list(
    tenant_id: str = typer.Option(None, help="Filter by tenant"),
    role: str = typer.Option(None, help="Filter by role"),
    limit: int = typer.Option(50, help="Limit results")
):
    """List users."""
    users = user_service.list_users(
        tenant_id=tenant_id,
        role=role,
        limit=limit
    )
    
    table = Table(title="Users")
    table.add_column("ID")
    table.add_column("Email")
    table.add_column("Tenant")
    table.add_column("Role")
    table.add_column("Status")
    table.add_column("Created")
    
    for user in users:
        table.add_row(
            str(user.id),
            user.email,
            user.tenant.name,
            user.role,
            user.status,
            user.created_at.strftime("%Y-%m-%d")
        )
    
    console.print(table)
```

### Tenant Management Commands

```python
from saas_app.cli.commands.tenants import tenants_app

@tenants_app.command()
def create(
    name: str = typer.Argument(..., help="Tenant name"),
    subdomain: str = typer.Argument(..., help="Subdomain"),
    plan: str = typer.Option("starter", help="Subscription plan"),
    owner_email: str = typer.Option(..., help="Owner email")
):
    """Create a new tenant."""
    tenant = tenant_service.create_tenant(
        name=name,
        subdomain=subdomain,
        plan=plan,
        owner_email=owner_email
    )
    
    # Create database schema
    tenant_service.setup_tenant_database(tenant.id)
    
    # Send welcome email
    email_service.send_welcome_email(owner_email, tenant)
    
    typer.echo(f"✅ Created tenant: {tenant.name}")
    typer.echo(f"🌐 URL: https://{subdomain}.yourdomain.com")

@tenants_app.command()
def upgrade(
    tenant_id: str = typer.Argument(..., help="Tenant ID"),
    plan: str = typer.Argument(..., help="New plan")
):
    """Upgrade tenant subscription."""
    tenant = tenant_service.upgrade_plan(tenant_id, plan)
    
    # Update feature access
    feature_service.update_tenant_features(tenant_id, plan)
    
    typer.echo(f"✅ Upgraded {tenant.name} to {plan}")
```

## Background Tasks

### Celery Configuration

```python
from celery import Celery
from saas_app.core.config import settings

celery_app = Celery(
    "saas_app",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "saas_app.workers.email_tasks",
        "saas_app.workers.analytics_tasks",
        "saas_app.workers.billing_tasks",
        "saas_app.workers.cleanup_tasks"
    ]
)

# Task routing
celery_app.conf.task_routes = {
    "saas_app.workers.email_tasks.*": {"queue": "email"},
    "saas_app.workers.analytics_tasks.*": {"queue": "analytics"},
    "saas_app.workers.billing_tasks.*": {"queue": "billing"},
}
```

### Email Tasks

```python
from saas_app.workers.celery_app import celery_app
from saas_app.email.email_service import EmailService

@celery_app.task
def send_welcome_email(user_email: str, tenant_name: str):
    """Send welcome email to new user."""
    email_service = EmailService()
    return email_service.send_template_email(
        to=user_email,
        template="welcome",
        context={"tenant_name": tenant_name}
    )

@celery_app.task
def send_usage_report(tenant_id: str):
    """Send monthly usage report."""
    analytics = AnalyticsManager()
    report = analytics.generate_monthly_report(tenant_id)
    
    email_service = EmailService()
    return email_service.send_template_email(
        to=report.tenant.owner_email,
        template="usage_report",
        context={"report": report}
    )
```

### Analytics Tasks

```python
@celery_app.task
def calculate_daily_metrics():
    """Calculate daily metrics for all tenants."""
    tenants = tenant_service.get_all_tenants()
    
    for tenant in tenants:
        metrics = analytics_service.calculate_daily_metrics(tenant.id)
        analytics_service.store_metrics(tenant.id, metrics)

@celery_app.task
def process_event_stream():
    """Process analytics events from stream."""
    events = event_stream.get_pending_events()
    
    for event in events:
        analytics_service.process_event(event)
        event_stream.mark_processed(event.id)
```

## Testing Strategy

### API Testing

```python
import pytest
from fastapi.testclient import TestClient
from saas_app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def authenticated_client(client, test_user, test_tenant):
    # Login and get token
    response = client.post("/auth/login", json={
        "email": test_user.email,
        "password": "password"
    })
    token = response.json()["access_token"]
    
    # Set authorization header
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client

class TestTenantAPI:
    def test_create_tenant(self, authenticated_client):
        response = authenticated_client.post("/api/tenants", json={
            "name": "Test Corp",
            "subdomain": "testcorp",
            "plan": "premium"
        })
        assert response.status_code == 201
        assert response.json()["name"] == "Test Corp"
    
    def test_tenant_isolation(self, authenticated_client):
        # Create two tenants
        tenant1 = authenticated_client.post("/api/tenants", json={
            "name": "Tenant 1", "subdomain": "tenant1"
        }).json()
        
        tenant2 = authenticated_client.post("/api/tenants", json={
            "name": "Tenant 2", "subdomain": "tenant2"
        }).json()
        
        # Verify data isolation
        with client_for_tenant(tenant1["id"]) as client1:
            with client_for_tenant(tenant2["id"]) as client2:
                # Create data in tenant1
                data1 = client1.post("/api/data", json={"name": "tenant1-data"})
                
                # Verify tenant2 cannot see tenant1's data
                data2 = client2.get("/api/data")
                assert len(data2.json()) == 0
```

### E2E Testing

```python
from playwright.async_api import async_playwright

class TestSaaSWorkflow:
    async def test_complete_signup_flow(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Navigate to signup
            await page.goto("http://localhost:8000/signup")
            
            # Fill signup form
            await page.fill("#company-name", "Test Company")
            await page.fill("#subdomain", "testco")
            await page.fill("#email", "admin@testco.com")
            await page.fill("#password", "SecurePass123!")
            
            # Submit form
            await page.click("#signup-button")
            
            # Wait for tenant creation
            await page.wait_for_selector(".success-message")
            
            # Verify redirect to dashboard
            await page.wait_for_url("**/dashboard")
            
            # Verify tenant setup
            tenant_name = await page.text_content(".tenant-name")
            assert tenant_name == "Test Company"
            
            await browser.close()
```

## Deployment

### Docker Configuration

```dockerfile
# Multi-stage build
FROM node:18-slim AS css-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY tailwind.config.js ./
COPY src/saas_app/static/css/input.css ./src/saas_app/static/css/
COPY src/saas_app/templates/ ./src/saas_app/templates/
RUN npm run build-css:prod

FROM python:3.11-slim AS python-builder
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e .[production]

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin
COPY --from=css-builder /app/src/saas_app/static/css/output.css ./src/saas_app/static/css/
COPY src/ ./src/
COPY migrations/ ./migrations/
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["gunicorn", "saas_app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Kubernetes Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saas-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: saas-app
  template:
    metadata:
      labels:
        app: saas-app
    spec:
      containers:
      - name: app
        image: saas-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi" 
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Terraform Infrastructure

```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.24"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    main = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 1
      
      instance_types = ["t3.medium"]
      
      k8s_labels = {
        Environment = var.environment
        Application = "saas-app"
      }
    }
  }
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier = "${var.app_name}-db"
  
  engine         = "postgres"
  engine_version = "14.6"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment == "dev"
  
  tags = {
    Name        = "${var.app_name}-db"
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.app_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "main" {
  cluster_id           = "${var.app_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Name        = "${var.app_name}-redis"
    Environment = var.environment
  }
}
```

## Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
from saas_app.monitoring.metrics import MetricsCollector

# Custom metrics
tenant_requests = Counter(
    'saas_tenant_requests_total',
    'Total requests per tenant',
    ['tenant_id', 'endpoint', 'method', 'status']
)

api_duration = Histogram(
    'saas_api_duration_seconds',
    'API request duration',
    ['tenant_id', 'endpoint']
)

active_subscriptions = Gauge(
    'saas_active_subscriptions',
    'Number of active subscriptions',
    ['plan']
)

# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    tenant_id = getattr(request.state, 'tenant', {}).get('id', 'unknown')
    
    tenant_requests.labels(
        tenant_id=tenant_id,
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    api_duration.labels(
        tenant_id=tenant_id,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

### Health Checks

```python
from saas_app.monitoring.health import HealthChecker

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_checker = HealthChecker()
    
    checks = await health_checker.run_all_checks()
    
    status = "healthy" if all(c["status"] == "ok" for c in checks) else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow(),
        "checks": checks,
        "version": app.version
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    try:
        # Check database connectivity
        await database.execute("SELECT 1")
        
        # Check Redis connectivity
        await redis.ping()
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")
```

## Performance Optimization

### Caching Strategy

```python
from saas_app.core.cache import CacheManager

class TenantDataService:
    def __init__(self):
        self.cache = CacheManager()
    
    @cache_result(ttl=300)  # 5 minutes
    async def get_tenant_stats(self, tenant_id: str) -> dict:
        """Get tenant statistics with caching."""
        return await self._calculate_tenant_stats(tenant_id)
    
    @cache_result(ttl=3600, key_func=lambda self, tenant_id: f"features:{tenant_id}")
    async def get_tenant_features(self, tenant_id: str) -> list:
        """Get tenant features with longer cache."""
        subscription = await self.get_tenant_subscription(tenant_id)
        return subscription.plan.features
```

### Database Optimization

```python
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload

class OptimizedQueries:
    async def get_tenant_with_users(self, tenant_id: str):
        """Optimized query with eager loading."""
        query = (
            select(Tenant)
            .options(
                selectinload(Tenant.users),
                joinedload(Tenant.subscription),
                selectinload(Tenant.api_keys)
            )
            .where(Tenant.id == tenant_id)
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_usage_analytics(self, tenant_id: str, days: int = 30):
        """Optimized analytics query."""
        # Use read replica for analytics
        query = text("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as api_calls,
                AVG(duration) as avg_duration
            FROM api_usage_logs 
            WHERE tenant_id = :tenant_id 
            AND created_at >= NOW() - INTERVAL :days DAY
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """)
        
        result = await self.read_db.execute(
            query, 
            {"tenant_id": tenant_id, "days": days}
        )
        return result.fetchall()
```

## Security Implementation

### Multi-Tenant Security

```python
from saas_app.security.tenant_isolation import TenantIsolationService

class TenantSecurityMiddleware:
    def __init__(self):
        self.isolation_service = TenantIsolationService()
    
    async def __call__(self, request: Request, call_next):
        # Extract tenant from subdomain or header
        tenant = await self.get_tenant_from_request(request)
        
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        # Verify tenant is active
        if not tenant.is_active:
            raise HTTPException(status_code=403, detail="Tenant suspended")
        
        # Set tenant context
        request.state.tenant = tenant
        request.state.db = self.isolation_service.get_tenant_db(tenant.id)
        
        # Row-level security
        await self.isolation_service.set_tenant_context(tenant.id)
        
        return await call_next(request)
```

### API Security

```python
from saas_app.security.api_security import APISecurityManager

class APIKeyMiddleware:
    def __init__(self):
        self.security_manager = APISecurityManager()
    
    async def authenticate_api_key(self, request: Request):
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        # Validate API key
        key_info = await self.security_manager.validate_api_key(api_key)
        
        if not key_info:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check rate limits
        await self.security_manager.check_rate_limit(key_info)
        
        # Track usage
        await self.security_manager.track_api_usage(key_info, request)
        
        return key_info
```

## Best Practices

### Code Organization
1. **Clean Architecture**: Separation of concerns with clear layers
2. **Domain-Driven Design**: Business logic in domain layer
3. **Dependency Injection**: Loose coupling between components
4. **Type Safety**: Comprehensive type hints throughout

### Security
1. **Defense in Depth**: Multiple security layers
2. **Tenant Isolation**: Complete data separation
3. **API Security**: Rate limiting and authentication
4. **Data Protection**: Encryption and secure storage

### Performance
1. **Caching Strategy**: Multi-level caching
2. **Database Optimization**: Efficient queries and indexing
3. **Async Processing**: Non-blocking operations
4. **Resource Management**: Connection pooling and limits

### Monitoring
1. **Comprehensive Metrics**: Business and technical metrics
2. **Health Checks**: Proactive monitoring
3. **Logging**: Structured and searchable logs
4. **Alerting**: Automated incident detection

### Testing
1. **Test Pyramid**: Unit, integration, and E2E tests
2. **Test Isolation**: Independent test execution
3. **Mock Strategy**: External service mocking
4. **Performance Testing**: Load and stress testing

## License

MIT License - see LICENSE file for details