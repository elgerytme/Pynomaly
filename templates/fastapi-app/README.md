# FastAPI App Template

A production-ready FastAPI application template with modern Python practices and comprehensive features.

## Features

- **FastAPI Framework**: High-performance async web framework
- **Modern Python**: Python 3.11+ with type hints and async/await
- **Authentication**: JWT-based auth with RBAC
- **Database**: SQLAlchemy with async support
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: Auto-generated OpenAPI docs
- **Monitoring**: Prometheus metrics and health checks
- **Security**: CORS, rate limiting, input validation
- **Docker**: Production-ready containerization

## Directory Structure

```
fastapi-app/
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
├── docs/                  # Documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── src/                   # Source code
│   └── fastapi_app/
│       ├── api/          # API endpoints
│       ├── auth/         # Authentication
│       ├── core/         # Core logic
│       ├── db/           # Database models
│       ├── schemas/      # Pydantic models
│       ├── services/     # Business logic
│       └── utils/        # Utilities
├── pkg/                  # Package metadata
├── examples/             # Usage examples
├── tests/                # Test suites
├── .github/              # GitHub workflows
├── scripts/              # Automation scripts
├── pyproject.toml        # Project configuration
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local development
├── README.md            # Project documentation
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> fastapi-app
   cd fastapi-app
   ```

2. **Setup environment**:
   ```bash
   cp env/development/.env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker**:
   ```bash
   docker-compose up -d
   ```

4. **Or run locally**:
   ```bash
   pip install -e .
   uvicorn fastapi_app.main:app --reload
   ```

5. **Access the API**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## Development

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Running the Application

```bash
# Development server
uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
gunicorn fastapi_app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Database Management

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Run migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/fastapi_app

# Run specific test file
pytest tests/test_auth.py

# Run integration tests
pytest tests/integration/
```

## API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Users
- `GET /users/me` - Current user profile
- `PUT /users/me` - Update profile
- `GET /users/{id}` - Get user by ID
- `GET /users/` - List users (admin only)

### Health & Monitoring
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

## Configuration

The application uses environment variables for configuration:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis
REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

## Features

### Authentication & Authorization
- JWT token-based authentication
- Role-based access control (RBAC)
- Password hashing with bcrypt
- Token refresh mechanism
- User registration and login

### Database Integration
- SQLAlchemy ORM with async support
- Alembic migrations
- Database connection pooling
- Automatic table creation
- PostgreSQL/MySQL support

### API Documentation
- Automatic OpenAPI/Swagger documentation
- Interactive API explorer
- Request/response examples
- Schema validation

### Security Features
- Input validation with Pydantic
- CORS middleware
- Rate limiting
- Security headers
- SQL injection prevention

### Monitoring & Logging
- Prometheus metrics
- Health check endpoints
- Structured logging
- Request/response logging
- Error tracking

### Testing
- Unit tests with pytest
- Integration tests
- Test fixtures
- Database test isolation
- Mock external services

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t fastapi-app:latest .

# Run container
docker run -p 8000:8000 fastapi-app:latest

# Use docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Check deployment
kubectl get pods -l app=fastapi-app
```

### Environment Variables

Create appropriate `.env` files for each environment:

- `env/development/.env` - Development settings
- `env/staging/.env` - Staging settings
- `env/production/.env` - Production settings

## License

MIT License - see LICENSE file for details