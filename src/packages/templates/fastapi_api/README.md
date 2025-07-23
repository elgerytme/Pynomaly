# FastAPI API Template

A production-ready FastAPI template with async SQLAlchemy, Pydantic, and structured project layout.

## Features

- FastAPI with async support
- SQLAlchemy ORM with async PostgreSQL
- Pydantic for data validation
- Structured project layout
- Docker support
- Environment configuration
- Structured logging with structlog
- CRUD operations for Users and Items
- API versioning

## Usage

This template uses placeholders that should be replaced when creating a new project:

- `{{package_name}}` - Your package name (e.g., "my_api")
- `{{description}}` - Your project description
- `{{author}}` - Author name
- `{{secret_key}}` - Secret key for JWT tokens
- `{{db_user}}`, `{{db_password}}`, `{{db_host}}`, `{{db_port}}`, `{{db_name}}` - Database configuration
- `{{redis_host}}`, `{{redis_port}}` - Redis configuration

## Project Structure

```
fastapi_api/
├── src/
│   └── {{package_name}}/
│       ├── api/          # API endpoints
│       ├── core/         # Core configuration
│       ├── db/           # Database configuration
│       ├── models/       # SQLAlchemy models
│       ├── schemas/      # Pydantic schemas
│       └── services/     # Business logic
├── .env.template
├── Dockerfile.template
├── pyproject.toml.template
└── README.md
```

## Getting Started

1. Replace all template variables
2. Install dependencies: `poetry install`
3. Set up environment variables in `.env`
4. Run migrations: `alembic upgrade head`
5. Start the server: `uvicorn {{package_name}}.main:app --reload`