.PHONY: help install dev-install test lint format clean build docs serve run-api run-cli run-web docker-build docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
POETRY := poetry
PROJECT := pynomaly
SRC := src
TESTS := tests

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	$(POETRY) install

dev-install: ## Install all dependencies including dev
	$(POETRY) install --with dev
	pre-commit install

test: ## Run tests
	$(POETRY) run pytest

test-cov: ## Run tests with coverage
	$(POETRY) run pytest --cov=$(PROJECT) --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	$(POETRY) run ptw

lint: ## Run all linters
	$(POETRY) run black --check $(SRC) $(TESTS)
	$(POETRY) run isort --check-only $(SRC) $(TESTS)
	$(POETRY) run mypy $(SRC)
	$(POETRY) run ruff $(SRC) $(TESTS)
	$(POETRY) run bandit -r $(SRC)
	$(POETRY) run safety check

format: ## Format code
	$(POETRY) run black $(SRC) $(TESTS)
	$(POETRY) run isort $(SRC) $(TESTS)

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	$(POETRY) build

build-css: ## Build Tailwind CSS
	npm run build-css

watch-css: ## Watch and rebuild CSS
	npm run dev

docs: ## Build documentation
	$(POETRY) run mkdocs build

serve-docs: ## Serve documentation locally
	$(POETRY) run mkdocs serve

run-api: ## Run API server
	$(POETRY) run uvicorn pynomaly.presentation.api.app:app --reload

run-cli: ## Run CLI
	$(POETRY) run pynomaly

run-web: ## Run web server with CSS watch
	make -j2 run-api watch-css

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

migrate: ## Run database migrations
	$(POETRY) run alembic upgrade head

migration: ## Create new migration
	$(POETRY) run alembic revision --autogenerate -m "$(MSG)"

shell: ## Open Python shell with app context
	$(POETRY) run ipython

check: lint test ## Run all checks

ci: check build ## Run CI pipeline

release: ## Create a new release
	@echo "Creating release..."
	@read -p "Version (current: $$($(POETRY) version -s)): " version; \
	$(POETRY) version $$version; \
	git add pyproject.toml; \
	git commit -m "chore: bump version to $$version"; \
	git tag -a v$$version -m "Release version $$version"; \
	echo "Created release v$$version"

publish: build ## Publish to PyPI
	$(POETRY) publish

setup-pre-commit: ## Setup pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit run --all-files

update-deps: ## Update dependencies
	$(POETRY) update
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	npm update

security-check: ## Run security checks
	$(POETRY) run bandit -r $(SRC)
	$(POETRY) run safety check
	$(POETRY) run pip-audit