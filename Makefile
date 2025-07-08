# Pynomaly Makefile with Buck2 + Hatch integration
# Production-ready build and development workflow
#
# Prerequisites:
#   - Hatch installed (pip install hatch)
#   - Buck2 installed (optional, for accelerated builds)
#   - Node.js 18+ and npm (for web assets)
#   - Git repository initialized
#
# Quick start:
#   make deps     - Install all dependencies
#   make build    - Build entire project (Buck2 + Hatch + npm)
#   make test     - Run all tests with Buck2 acceleration
#   make dev      - Start development environment
#   make build    - Build package
#   make clean    - Clean up artifacts

.PHONY: help setup install dev-install lint format test test-cov build clean docker pre-commit ci status release docs git-hooks branch-new branch-switch branch-validate

# Default target
help: ## Show this help message
	@echo "🚀 Pynomaly Development Commands (Hatch-based)"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Initial project setup (install Hatch, create environments)"
	@echo "  make install        - Install package in current environment"
	@echo "  make dev-install    - Install package in development mode with all dependencies"
	@echo ""
	@echo "Development & Quality:"
	@echo "  make dev            - Start Docker development environment (recommended)"
	@echo "  make dev-storage    - Start development with storage services (PostgreSQL, Redis, MinIO)"
	@echo "  make dev-test       - Run tests in Docker environment"
	@echo "  make dev-clean      - Clean Docker development environment"
	@echo "  make dev-legacy     - Start legacy npm watch mode (deprecated)"
	@echo "  make lint           - Run all code quality checks (style, type, format)"
	@echo "  make format         - Auto-format code (ruff, black, isort)"
	@echo "  make style          - Check code style without fixing"
	@echo "  make typing         - Run type checking with mypy"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run core tests (domain + application)"
	@echo "  make test-all       - Run all tests including integration"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-unit      - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo ""
	@echo "Build & Package:"
	@echo "  make build          - Build wheel and source distribution"
	@echo "  make version        - Show current version"
	@echo "  make clean          - Clean build artifacts and cache"
	@echo ""
	@echo "Pre-commit & CI:"
	@echo "  make pre-commit     - Install and run pre-commit hooks"
	@echo "  make ci             - Run full CI pipeline locally"
	@echo ""
	@echo "Security:"
	@echo "  make security-scan  - Run security scan locally with non-zero exit on high severity issues"
	@echo "  make aggregate-sarif - Aggregate SARIF files into combined report"
	@echo ""
	@echo "Documentation & Deployment:"
	@echo "  make docs           - Build documentation"
	@echo "  make docs-serve     - Serve documentation locally"
	@echo "  make docker         - Build Docker image"
	@echo ""
	@echo "Utilities:"
	@echo "  make status         - Show project status and environment info"
	@echo "  make env-show       - Show all Hatch environments"
	@echo "  make env-clean      - Clean and recreate environments"
	@echo ""
	@echo "Branching & Git:"
	@echo "  make git-hooks            - Install Git hooks (pre-commit, pre-push, post-checkout)"
	@echo "  make branch-new TYPE NAME - Create new branch with validation (e.g., feature my-feature)"
	@echo "  make branch-switch NAME   - Switch branches with safety checks"
	@echo "  make branch-validate      - Validate current branch name for CI"
	@echo "  Note: Cross-platform scripts in scripts/git/ (bash & PowerShell)"
	@echo ""
	@echo "For detailed help: make help-detailed"

help-detailed: ## Show detailed help with examples
	@echo "🔧 Detailed Pynomaly Development Guide"
	@echo ""
	@echo "=== INITIAL SETUP ==="
	@echo "1. Clone repository and navigate to directory"
	@echo "2. Run: make setup"
	@echo "3. Run: make dev-install"
	@echo "4. Run: make pre-commit"
	@echo ""
	@echo "=== DAILY DEVELOPMENT WORKFLOW ==="
	@echo "1. make dev           # Start Docker development environment"
	@echo "2. make format        # Auto-fix code style"
	@echo "3. make dev-test      # Run tests in Docker"
	@echo "4. make lint          # Check quality"
	@echo "5. git add . && git commit -m 'feat: your changes'"
	@echo "6. make ci            # Full CI check before push"
	@echo "7. git push"
	@echo ""
	@echo "=== HATCH COMMANDS USED ==="
	@echo "• hatch version       → Git-based version management"
	@echo "• hatch build         → Package building"
	@echo "• hatch env run       → Environment-specific commands"
	@echo "• hatch env show      → List environments"
	@echo ""
	@echo "=== ENVIRONMENTS AVAILABLE ==="
	@echo "• default: Basic development (path: .venv)"
	@echo "• test: Full test suite with matrix (py3.11, py3.12)"
	@echo "• lint: Code quality tools (detached)"
	@echo "• docs: Documentation building"
	@echo "• dev: Development tools and pre-commit"
	@echo "• prod: Production environment"
	@echo "• cli: CLI-specific testing"

# === SETUP & INSTALLATION ===

setup: ## Initial project setup - install Hatch and create environments
	@echo "🚀 Setting up Pynomaly development environment..."
	@command -v hatch >/dev/null 2>&1 || (echo "Installing Hatch..." && pip install hatch)
	@echo "✅ Hatch installed: $$(hatch --version)"
	@echo "📦 Creating Hatch environments..."
	hatch env create
	@echo "📋 Available environments:"
	hatch env show
	@echo "✅ Setup complete! Run 'make dev-install' next."

install: ## Install package in current environment
	pip install -e .

dev-install: ## Install package in development mode with all dependencies
	@echo "📦 Installing Pynomaly in development mode..."
	hatch env run dev:setup
	@echo "✅ Development installation complete!"

# === CODE QUALITY ===

lint: ## Run all code quality checks
	@echo "🔍 Running code quality checks..."
	@echo "1️⃣ Style checking..."
	hatch env run lint:style
	@echo "2️⃣ Type checking..."
	hatch env run lint:typing
	@echo "✅ All quality checks passed!"

format: ## Auto-format code
	@echo "🎨 Auto-formatting code..."
	hatch env run lint:fmt
	@echo "✅ Code formatting complete!"

style: ## Check code style without fixing
	@echo "🎨 Checking code style..."
	hatch env run lint:style

typing: ## Run type checking
	@echo "🔎 Running type checking..."
	hatch env run lint:typing

# === TESTING ===

test: ## Run core tests (domain + application)
	@echo "🧪 Running core tests..."
	hatch env run test:run tests/domain/ tests/application/ -v

test-all: ## Run all tests including integration
	@echo "🧪 Running all tests..."
	hatch env run test:run -v

test-cov: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	hatch env run test:run-cov
	@echo "📊 Coverage report generated in htmlcov/"

test-unit: ## Run only unit tests
	@echo "🧪 Running unit tests..."
	hatch env run test:run tests/domain/ tests/application/ -v

test-integration: ## Run only integration tests
	@echo "🧪 Running integration tests..."
	hatch env run test:run tests/infrastructure/ -v --ignore=tests/infrastructure/test_*_performance*

test-parallel: ## Run tests in parallel
	@echo "🧪 Running tests in parallel..."
	hatch env run test:run-parallel

# === BUILD & PACKAGE ===

build: ## Build wheel and source distribution
	@echo "📦 Building package..."
	hatch build --clean
	@echo "📋 Build artifacts:"
	@ls -la dist/

version: ## Show current version
	@echo "📋 Current version: $$(hatch version)"

clean: ## Clean build artifacts and cache
	@echo "🧹 Cleaning build artifacts..."
	hatch env run dev:clean
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# === ENVIRONMENTS ===

env-show: ## Show all Hatch environments
	@echo "📋 Hatch environments:"
	hatch env show

env-clean: ## Clean and recreate environments
	@echo "🧹 Cleaning Hatch environments..."
	hatch env prune
	hatch env create
	@echo "✅ Environments recreated!"

# === PRE-COMMIT & CI ===

pre-commit: ## Install and run pre-commit hooks
	@echo "🔗 Setting up pre-commit hooks..."
	pip install pre-commit
	pre-commit install --install-hooks
	@echo "🧪 Running pre-commit on all files..."
	pre-commit run --all-files
	@echo "✅ Pre-commit setup complete!"

ci: ## Run full CI pipeline locally
	@echo "🚀 Running full CI pipeline locally..."
	@echo "1️⃣ Version check..."
	hatch version
	@echo "2️⃣ Code quality..."
	$(MAKE) lint
	@echo "3️⃣ Core tests..."
	$(MAKE) test
	@echo "4️⃣ Integration tests..."
	$(MAKE) test-integration
	@echo "5️⃣ Build package..."
	$(MAKE) build
	@echo "6️⃣ CLI test..."
	hatch env run cli:test-cli
	@echo "7️⃣ Core imports..."
	python -c "import sys; sys.path.insert(0, 'src'); from pynomaly.domain.entities import Dataset; print('✅ Core imports successful')"
	@echo "✅ Full CI pipeline completed successfully!"

# === DOCUMENTATION ===

docs: ## Build documentation
	@echo "📖 Building documentation..."
	hatch env run docs:build
	@echo "✅ Documentation built in site/"

docs-serve: ## Serve documentation locally
	@echo "📖 Serving documentation at http://localhost:8080"
	hatch env run docs:serve

# === DOCKER ===

docker: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -f deploy/docker/Dockerfile -t pynomaly:latest .
	@echo "✅ Docker image built: pynomaly:latest"

# === UTILITIES ===

status: ## Show project status and environment info
	@echo "📊 Pynomaly Project Status"
	@echo "=========================="
	@echo "Version: $$(hatch version)"
	@echo "Hatch: $$(hatch --version)"
	@echo "Python: $$(python --version)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $$(git status --porcelain 2>/dev/null | wc -l || echo '0') files changed"
	@echo ""
	@echo "📋 Environments:"
	@hatch env show --ascii 2>/dev/null || echo "Run 'make setup' to create environments"
	@echo ""
	@echo "📦 Build artifacts:"
	@ls -la dist/ 2>/dev/null || echo "No build artifacts (run 'make build')"

# === BRANCHING & GIT ===

# OS Detection
OS := $(shell uname -s 2>/dev/null || echo Windows_NT)

git-hooks: ## Install Git hooks from scripts/git/hooks/
	@echo "🔗 Installing Git hooks..."
	@echo "Setting Git hooks path to scripts/git/hooks/"
	git config core.hooksPath scripts/git/hooks
	@echo "✅ Git hooks installed successfully!"
	@echo "Hooks available:"
	@echo "  - pre-commit  → branch naming lint + partial linting"
	@echo "  - pre-push    → run unit tests"
	@echo "  - post-checkout → remind to restart long-running services"
	@echo "🎯 Cross-platform installation complete!"
	@echo ""
	@echo "Alternative installation methods (if make is not available):"
	@echo "  Windows: powershell -ExecutionPolicy Bypass -File scripts/git/install-hooks.ps1"
	@echo "  Unix:    bash scripts/git/install-hooks.sh"

branch-new: ## Create new branch with validation (usage: make branch-new TYPE=feature NAME=my-feature)
	@echo "🌿 Creating new branch..."
	@if [ -z "$(TYPE)" ] || [ -z "$(NAME)" ]; then \
		echo "❌ Error: Both TYPE and NAME are required"; \
		echo "Usage: make branch-new TYPE=<type> NAME=<name>"; \
		echo "Valid types: feature, bugfix, hotfix, release, chore, docs"; \
		echo "Example: make branch-new TYPE=feature NAME=anomaly-detection"; \
		exit 1; \
	fi
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File scripts/git/git_new_branch.ps1 -Type "$(TYPE)" -Name "$(NAME)"
else
	@hatch env run git:git_new_branch.sh $(TYPE) $(NAME)
endif

branch-switch: ## Switch branches with safety checks (usage: make branch-switch NAME=feature/my-feature)
	@echo "🔄 Switching branches with safety checks..."
	@if [ -z "$(NAME)" ]; then \
		echo "❌ Error: NAME is required"; \
		echo "Usage: make branch-switch NAME=<branch-name>"; \
		echo "Example: make branch-switch NAME=feature/anomaly-detection"; \
		exit 1; \
	fi
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File scripts/git/git_switch_safe.ps1 -Name "$(NAME)"
else
	@hatch env run git:git_switch_safe.sh $(NAME)
endif

branch-validate: ## Validate current branch name for CI compliance
	@echo "🔍 Validating current branch name..."
	@CURRENT_BRANCH=$$(git branch --show-current 2>/dev/null); \
	if [ -z "$$CURRENT_BRANCH" ]; then \
		echo "❌ Error: Not in a git repository or no current branch"; \
		exit 1; \
	fi; \
	echo "Current branch: $$CURRENT_BRANCH"; \
	if [ "$$CURRENT_BRANCH" = "main" ] || [ "$$CURRENT_BRANCH" = "master" ] || [ "$$CURRENT_BRANCH" = "develop" ]; then \
		echo "✅ Main branch '$$CURRENT_BRANCH' - validation passed"; \
		exit 0; \
	fi; \
	if echo "$$CURRENT_BRANCH" | grep -qE '^(feature|bugfix|hotfix|release|chore|docs)/[a-z0-9-]+$$'; then \
		echo "✅ Branch name '$$CURRENT_BRANCH' follows naming convention"; \
		echo "📋 Branch type: $$(echo $$CURRENT_BRANCH | cut -d'/' -f1)"; \
		echo "📋 Branch name: $$(echo $$CURRENT_BRANCH | cut -d'/' -f2)"; \
	else \
		echo "❌ Error: Branch name '$$CURRENT_BRANCH' does not follow naming convention"; \
		echo "Expected format: <type>/<name>"; \
		echo "Valid types: feature, bugfix, hotfix, release, chore, docs"; \
		echo "Name must contain only lowercase letters, numbers, and hyphens"; \
		echo "Examples: feature/anomaly-detection, bugfix/memory-leak, docs/api-updates"; \
		exit 1; \
	fi

# === SECURITY COMMANDS ===

security-scan: ## Run security scan locally with non-zero exit on high severity issues
	@echo "🔒 Running security scan..."
	python test_security_scan.py

aggregate-sarif: ## Aggregate SARIF files into combined report
	@echo "📊 Aggregating SARIF files..."
	@echo "Converting bandit JSON to SARIF..."
	@if [ -f "bandit-report.json" ]; then \
		python scripts/convert_to_sarif.py bandit-report.json bandit-report.sarif; \
		echo "✅ Bandit SARIF converted"; \
	else \
		echo "⚠️  No bandit-report.json found"; \
	fi
	@echo "Aggregating all SARIF files..."
	@SARIF_FILES=$$(find . -name "*.sarif" -type f 2>/dev/null | head -10); \
	if [ -n "$$SARIF_FILES" ]; then \
		python scripts/aggregate_sarif.py $$SARIF_FILES combined-security-report.sarif; \
		echo "✅ Combined SARIF report created: combined-security-report.sarif"; \
	else \
		echo "⚠️  No SARIF files found to aggregate"; \
	fi

# === PRODUCTION COMMANDS ===

prod-api: ## Start production API server
	@echo "🚀 Starting production API server..."
	hatch env run prod:serve-api-prod

prod-api-dev: ## Start development API server
	@echo "🚀 Starting development API server..."
	hatch env run prod:serve-api

cli-help: ## Show CLI help
	hatch env run cli:run --help

# === BUCK2 + NPM INTEGRATION ===

# Buck2 configuration
ENABLE_BUCK2 ?= $(shell command -v buck2 >/dev/null 2>&1 && echo true || echo false)
BUCK2_CACHE_DIR ?= .buck-cache

deps: ## Install all dependencies (Python + npm)
	@echo "📦 Installing all dependencies..."
	@echo "1️⃣ Installing Python dependencies..."
	hatch env create
	@echo "2️⃣ Installing npm dependencies..."
	npm install
	@echo "✅ All dependencies installed!"

buck-build: ## Build with Buck2 (if available)
ifeq ($(ENABLE_BUCK2), true)
	@echo "🚀 Building with Buck2..."
	@mkdir -p $(BUCK2_CACHE_DIR)
	buck2 build //... --local-cache $(BUCK2_CACHE_DIR)
	@echo "✅ Buck2 build completed!"
else
	@echo "⚠️  Buck2 not available, falling back to Hatch build..."
	$(MAKE) build
endif

buck-test: ## Run tests with Buck2 (if available)
ifeq ($(ENABLE_BUCK2), true)
	@echo "🧪 Running tests with Buck2..."
	buck2 test //tests:unit_tests //tests:integration_tests --verbose
else
	@echo "⚠️  Buck2 not available, falling back to pytest..."
	$(MAKE) test-all
endif

buck-clean: ## Clean Buck2 artifacts
ifeq ($(ENABLE_BUCK2), true)
	@echo "🧹 Cleaning Buck2 artifacts..."
	-buck2 clean
	-rm -rf $(BUCK2_CACHE_DIR)
	-rm -rf buck-out/
	@echo "✅ Buck2 artifacts cleaned!"
endif

npm-install: ## Install npm dependencies
	@echo "📦 Installing npm dependencies..."
	npm install
	@echo "✅ npm dependencies installed!"

npm-build: npm-install ## Build web assets
	@echo "🎨 Building web assets..."
	npm run build
	@echo "✅ Web assets built successfully!"

npm-watch: npm-install ## Watch and rebuild web assets
	@echo "👀 Starting web asset watch mode..."
	npm run watch

npm-clean: ## Clean npm artifacts
	@echo "🧹 Cleaning npm artifacts..."
	-rm -rf node_modules/
	-rm -rf src/pynomaly/presentation/web/static/css/styles.css
	-rm -rf src/pynomaly/presentation/web/static/js/app.js
	@echo "✅ npm artifacts cleaned!"

dev: ## Start development environment with Docker (recommended)
	@echo "🚀 Starting Docker development environment..."
	@echo "This will start the full development stack with hot-reload"
	@echo "API will be available at http://localhost:8000"
	@echo "Press Ctrl+C to stop"
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File ./scripts/docker/dev/run-dev.ps1 -build
else
	@./scripts/docker/dev/run-dev.sh --build
endif

dev-legacy: ## Start legacy development environment with watch mode (deprecated)
	@echo "⚠️  Starting legacy development environment..."
	@echo "This will start web asset watch mode only"
	@echo "Press Ctrl+C to stop"
	@trap 'kill 0' EXIT; npm run watch

dev-storage: ## Start development environment with storage services
	@echo "🚀 Starting Docker development environment with storage services..."
	@echo "This will start PostgreSQL, Redis, and MinIO services"
	@echo "API will be available at http://localhost:8000"
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File ./scripts/docker/dev/run-dev-with-storage.ps1 -storage all
else
	@./scripts/docker/dev/run-dev-with-storage.sh --storage all
endif

dev-test: ## Run tests in Docker environment
	@echo "🧪 Running tests in Docker environment..."
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File ./scripts/docker/test/run-test.ps1 -type all
else
	@./scripts/docker/test/run-test.sh --type all
endif

dev-clean: ## Clean Docker development environment
	@echo "🧺 Cleaning Docker development environment..."
	@echo "Stopping and removing development containers..."
	@docker ps -q --filter "name=pynomaly-dev" | xargs -r docker stop
	@docker ps -aq --filter "name=pynomaly-dev" | xargs -r docker rm
	@echo "✅ Development containers cleaned!"

# Update existing targets to include Buck2 + npm integration
build: npm-build ## Build package and web assets
	@echo "🏗️  Building complete project..."
	@echo "1️⃣ Building web assets..."
	@$(MAKE) npm-build
	@echo "2️⃣ Building Python package..."
ifeq ($(ENABLE_BUCK2), true)
	@$(MAKE) buck-build
else
	hatch build --clean
endif
	@echo "✅ Complete build finished!"

clean: npm-clean buck-clean ## Clean all build artifacts
	@echo "🧹 Cleaning all build artifacts..."
	hatch clean
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	@echo "✅ All artifacts cleaned!"

# === QUICK ALIASES ===

l: lint     ## Alias for lint
f: format   ## Alias for format
t: test     ## Alias for test
b: build    ## Alias for build
c: clean    ## Alias for clean
s: status   ## Alias for status

# Make sure all targets are treated as commands, not files
.DEFAULT_GOAL := help
