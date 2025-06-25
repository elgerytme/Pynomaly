# Makefile for Pynomaly project
# Provides convenient rules for testing, development, and deployment

.DEFAULT_GOAL := help
.PHONY: help test test-current test-fresh test-unit test-integration test-performance test-security test-fast test-verbose clean install lint format type-check docs build deploy

# Colors for output
CYAN = \033[36m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
RESET = \033[0m

# Project configuration
PROJECT_NAME := pynomaly
PYTHON := python3
PIP := pip
SCRIPTS_DIR := scripts
TEST_DIR := tests
SRC_DIR := src
DOCS_DIR := docs
REPORTS_DIR := test-reports

# Help target
help: ## Show this help message
	@echo "$(CYAN)Pynomaly Development Makefile$(RESET)"
	@echo "=============================="
	@echo ""
	@echo "$(GREEN)Testing Rules:$(RESET)"
	@grep -E '^test.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Rules:$(RESET)"
	@grep -E '^(install|lint|format|type-check|clean).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Build & Deploy Rules:$(RESET)"
	@grep -E '^(docs|build|deploy).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  make test              # Run all tests in current environment"
	@echo "  make test-fresh        # Run all tests in fresh environment"
	@echo "  make test-unit         # Run only unit tests"
	@echo "  make test-fast         # Run fast tests only"
	@echo "  make test-verbose      # Run tests with verbose output"
	@echo ""

# =======================
# Testing Rules
# =======================

test: ## Run all tests in current environment
	@echo "$(GREEN)Running all tests in current environment...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh

test-current: ## Run all tests in current environment (alias for test)
	@$(MAKE) test

test-fresh: ## Run all tests in fresh virtual environment
	@echo "$(GREEN)Running all tests in fresh environment...$(RESET)"
	@$(SCRIPTS_DIR)/test-fresh.sh

test-unit: ## Run only unit tests
	@echo "$(GREEN)Running unit tests...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --unit-only

test-integration: ## Run only integration tests
	@echo "$(GREEN)Running integration tests...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --integration-only

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --performance

test-security: ## Run security tests
	@echo "$(GREEN)Running security tests...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --security

test-fast: ## Run fast tests only (skip slow integration tests)
	@echo "$(GREEN)Running fast tests...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --fast

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests with verbose output...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --verbose

test-coverage: ## Run tests with coverage reporting
	@echo "$(GREEN)Running tests with coverage...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --coverage

test-no-coverage: ## Run tests without coverage reporting
	@echo "$(GREEN)Running tests without coverage...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --no-coverage

test-parallel: ## Run tests in parallel
	@echo "$(GREEN)Running tests in parallel...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --parallel

test-sequential: ## Run tests sequentially (no parallel)
	@echo "$(GREEN)Running tests sequentially...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --no-parallel

test-fail-fast: ## Run tests with fail-fast (stop on first failure)
	@echo "$(GREEN)Running tests with fail-fast...$(RESET)"
	@$(SCRIPTS_DIR)/test-current.sh --fail-fast

# Fresh environment test variants
test-fresh-clean: ## Run tests in fresh environment (clean existing venv)
	@echo "$(GREEN)Running tests in clean fresh environment...$(RESET)"
	@$(SCRIPTS_DIR)/test-fresh.sh --clean-venv

test-fresh-keep: ## Run tests in fresh environment (keep venv after)
	@echo "$(GREEN)Running tests in fresh environment (keeping venv)...$(RESET)"
	@$(SCRIPTS_DIR)/test-fresh.sh --keep-venv

test-fresh-unit: ## Run unit tests in fresh environment
	@echo "$(GREEN)Running unit tests in fresh environment...$(RESET)"
	@$(SCRIPTS_DIR)/test-fresh.sh --unit-only

test-fresh-fast: ## Run fast tests in fresh environment
	@echo "$(GREEN)Running fast tests in fresh environment...$(RESET)"
	@$(SCRIPTS_DIR)/test-fresh.sh --fast

# =======================
# PowerShell Equivalents
# =======================

test-ps: ## Run tests using PowerShell script (Windows)
	@echo "$(GREEN)Running tests with PowerShell...$(RESET)"
	@powershell -ExecutionPolicy Bypass -File $(SCRIPTS_DIR)/test-current.ps1

test-fresh-ps: ## Run tests in fresh environment using PowerShell (Windows)
	@echo "$(GREEN)Running tests in fresh environment with PowerShell...$(RESET)"
	@powershell -ExecutionPolicy Bypass -File $(SCRIPTS_DIR)/test-fresh.ps1

# =======================
# Quick Commands
# =======================

t: test ## Quick alias for test
tf: test-fresh ## Quick alias for test-fresh
tu: test-unit ## Quick alias for test-unit
ti: test-integration ## Quick alias for test-integration
tv: test-verbose ## Quick alias for test-verbose
tff: test-fast ## Quick alias for test-fast

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