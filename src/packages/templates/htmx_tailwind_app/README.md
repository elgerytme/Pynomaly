# HTMX + Tailwind CSS Web App Template

A modern web application template using FastAPI, HTMX, and Tailwind CSS for server-side rendering with dynamic UI updates.

## Features

- FastAPI backend with async support
- HTMX for dynamic UI updates without JavaScript
- Tailwind CSS for beautiful, responsive design
- Server-side rendering with Jinja2 templates
- Alpine.js for lightweight client-side interactions
- SQLAlchemy ORM with async PostgreSQL
- Structured project layout
- Docker support

## Usage

This template uses placeholders that should be replaced when creating a new project:

- `{{package_name}}` - Your package name (e.g., "my_app")
- `{{description}}` - Your project description
- `{{author}}` - Author name
- `{{secret_key}}` - Secret key for sessions
- `{{db_user}}`, `{{db_password}}`, `{{db_host}}`, `{{db_port}}`, `{{db_name}}` - Database configuration
- `{{redis_host}}`, `{{redis_port}}` - Redis configuration

## Project Structure

```
htmx_tailwind_app/
├── src/
│   └── {{package_name}}/
│       ├── api/          # API endpoints
│       ├── core/         # Core configuration
│       ├── db/           # Database configuration
│       ├── models/       # SQLAlchemy models
│       ├── schemas/      # Pydantic schemas
│       ├── services/     # Business logic
│       ├── static/       # Static files (CSS, JS, images)
│       └── templates/    # Jinja2 templates
│           ├── components/  # Reusable components
│           ├── layouts/     # Layout templates
│           └── pages/       # Page templates
├── .env.template
├── Dockerfile.template
├── pyproject.toml.template
└── README.md
```

## Key Technologies

### Frontend
- **HTMX** - Access modern browser features directly from HTML
- **Tailwind CSS** - Utility-first CSS framework
- **Alpine.js** - Minimal framework for composing JavaScript behavior
- **Jinja2** - Modern templating engine

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **SQLAlchemy** - SQL toolkit and ORM
- **Pydantic** - Data validation using Python type annotations
- **Uvicorn** - Lightning-fast ASGI server

## HTMX Features Demonstrated

- Live search with debounce
- Form submissions without page reload
- Dynamic content updates
- Delete confirmations
- Inline editing
- Real-time notifications
- Periodic polling for updates

## Getting Started

1. Replace all template variables
2. Install dependencies: `poetry install`
3. Set up environment variables in `.env`
4. Run database migrations: `alembic upgrade head`
5. Start the development server: `uvicorn {{package_name}}.main:app --reload`
6. Open http://localhost:8000 in your browser

## Development

For hot-reloading during development:
```bash
uvicorn {{package_name}}.main:app --reload --host 0.0.0.0 --port 8000
```

## Building for Production

```bash
docker build -t {{package_name}} .
docker run -p 8000:8000 --env-file .env {{package_name}}
```