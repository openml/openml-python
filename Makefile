# ============================================================================
# Professional Makefile for OpenML Python
# ============================================================================
# This Makefile automates common development tasks including:
# - Virtual environment management
# - Dependency installation
# - Code formatting (Black, Isort, Ruff)
# - Linting (Ruff)
# - Type checking (Mypy)
# - Testing with coverage (Pytest)
# - Pre-commit hooks
# - Build artifact cleaning
# ============================================================================

.PHONY: help
.DEFAULT_GOAL := help

# ============================================================================
# Configuration Variables
# ============================================================================
PYTHON_VERSION ?= 3.8
PYTHON ?= python
VENV_NAME ?= .venv
VENV_BIN := $(VENV_NAME)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
VENV_UV := $(VENV_BIN)/uv
PYTEST := $(VENV_BIN)/pytest
MYPY := $(VENV_BIN)/mypy
RUFF := $(VENV_BIN)/ruff
BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
PRE_COMMIT := $(VENV_BIN)/pre-commit
MKDOCS := $(VENV_BIN)/mkdocs

# Project directories
SRC_DIR := openml
TEST_DIR := tests
EXAMPLES_DIR := examples
DOCS_DIR := docs

# Coverage settings
COVERAGE_MIN ?= 80
COVERAGE_REPORT := htmlcov
COVERAGE_FILE := .coverage

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m

# ============================================================================
# Helper Functions
# ============================================================================
define print_header
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)==> $(1)$(COLOR_RESET)"
endef

define print_success
	@echo "$(COLOR_GREEN)✓ $(1)$(COLOR_RESET)"
endef

define print_warning
	@echo "$(COLOR_YELLOW)⚠ $(1)$(COLOR_RESET)"
endef

# ============================================================================
# Help Target
# ============================================================================
help: ## Show this help message
	@echo "$(COLOR_BOLD)OpenML Python - Available Commands$(COLOR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(COLOR_BLUE)<target>$(COLOR_RESET)\n\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2 } \
		/^##@/ { printf "\n$(COLOR_BOLD)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# ============================================================================
##@ Environment Setup
# ============================================================================

venv: $(VENV_BIN)/activate ## Create virtual environment
$(VENV_BIN)/activate:
	$(call print_header,Creating virtual environment)
	@test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME)
	@$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	@$(VENV_PIP) install uv
	$(call print_success,Virtual environment created at $(VENV_NAME))

install: venv ## Install package with core dependencies
	$(call print_header,Installing core dependencies)
	@$(VENV_UV) pip install -e .
	$(call print_success,Core dependencies installed)

install-dev: venv ## Install package with development dependencies
	$(call print_header,Installing development dependencies)
	@$(VENV_UV) pip install -e .[test,examples,docs]
	@$(PRE_COMMIT) install
	$(call print_success,Development dependencies installed)

install-test: venv ## Install package with test dependencies
	$(call print_header,Installing test dependencies)
	@$(VENV_UV) pip install -e .[test]
	$(call print_success,Test dependencies installed)

install-docs: venv ## Install package with documentation dependencies
	$(call print_header,Installing documentation dependencies)
	@$(VENV_UV) pip install -e .[docs]
	$(call print_success,Documentation dependencies installed)

install-all: install-dev ## Install all dependencies (alias for install-dev)

# ============================================================================
##@ Code Quality
# ============================================================================

format: venv ## Format code with Ruff (recommended)
	$(call print_header,Formatting code with Ruff)
	@$(RUFF) format $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(RUFF) check --fix --unsafe-fixes $(SRC_DIR) $(EXAMPLES_DIR)
	$(call print_success,Code formatted successfully)

format-black: venv ## Format code with Black + Isort (alternative)
	$(call print_header,Formatting code with Black and Isort)
	@$(ISORT) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(BLACK) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	$(call print_success,Code formatted successfully with Black and Isort)

format-all: venv ## Format code with both Ruff and Black + Isort
	$(call print_header,Formatting code with all formatters)
	@$(ISORT) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(BLACK) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(RUFF) format $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(RUFF) check --fix --unsafe-fixes $(SRC_DIR) $(EXAMPLES_DIR)
	$(call print_success,Code formatted successfully with all tools)

format-check: venv ## Check code formatting without making changes
	$(call print_header,Checking code formatting)
	@$(RUFF) format --check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	$(call print_success,Code formatting is correct)

format-check-black: venv ## Check code formatting with Black + Isort
	$(call print_header,Checking code formatting with Black and Isort)
	@$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(BLACK) --check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	$(call print_success,Code formatting is correct)

lint: venv ## Lint code with Ruff
	$(call print_header,Linting code with Ruff)
	@$(RUFF) check $(SRC_DIR) $(EXAMPLES_DIR)
	$(call print_success,Linting completed successfully)

lint-fix: venv ## Lint and auto-fix issues with Ruff
	$(call print_header,Linting and fixing code with Ruff)
	@$(RUFF) check --fix $(SRC_DIR) $(EXAMPLES_DIR)
	$(call print_success,Linting and fixes completed)

typecheck: venv ## Run type checking with Mypy
	$(call print_header,Running type checking with Mypy)
	@$(MYPY) $(SRC_DIR) $(TEST_DIR)
	$(call print_success,Type checking completed successfully)

check: venv ## Run pre-commit checks on all files
	$(call print_header,Running pre-commit checks)
	@$(PRE_COMMIT) run --all-files
	$(call print_success,Pre-commit checks passed)

quality: format-check lint typecheck ## Run all quality checks (format, lint, typecheck)
	$(call print_success,All quality checks passed!)

# ============================================================================
##@ Testing
# ============================================================================

test: venv ## Run tests with pytest
	$(call print_header,Running tests)
	@$(PYTEST) -v $(TEST_DIR)
	$(call print_success,Tests completed)

test-fast: venv ## Run tests in parallel with pytest-xdist
	$(call print_header,Running tests in parallel)
	@$(PYTEST) -n auto -v $(TEST_DIR)
	$(call print_success,Tests completed)

test-verbose: venv ## Run tests with verbose output
	$(call print_header,Running tests with verbose output)
	@$(PYTEST) -vv -s $(TEST_DIR)

test-coverage: venv ## Run tests with coverage report
	$(call print_header,Running tests with coverage)
	@$(PYTEST) --cov=$(SRC_DIR) --cov-report=html --cov-report=term --cov-report=xml $(TEST_DIR)
	$(call print_success,Coverage report generated in $(COVERAGE_REPORT)/)
	@echo "$(COLOR_BLUE)Open $(COVERAGE_REPORT)/index.html to view detailed report$(COLOR_RESET)"

test-coverage-report: test-coverage ## Run tests with coverage and open HTML report
	@$(PYTHON) -m webbrowser -t $(COVERAGE_REPORT)/index.html 2>/dev/null || \
		echo "$(COLOR_YELLOW)Open $(COVERAGE_REPORT)/index.html in your browser$(COLOR_RESET)"

test-failed: venv ## Re-run only failed tests
	$(call print_header,Re-running failed tests)
	@$(PYTEST) --lf -v $(TEST_DIR)

test-watch: venv ## Run tests in watch mode (requires pytest-watch)
	$(call print_header,Running tests in watch mode)
	@$(VENV_BIN)/ptw $(TEST_DIR)

# ============================================================================
##@ Documentation
# ============================================================================

docs: venv install-docs ## Build documentation
	$(call print_header,Building documentation)
	@$(MKDOCS) build
	$(call print_success,Documentation built in site/)

docs-serve: venv install-docs ## Serve documentation locally
	$(call print_header,Serving documentation)
	@echo "$(COLOR_BLUE)Documentation available at http://127.0.0.1:8000$(COLOR_RESET)"
	@$(MKDOCS) serve

docs-deploy: venv install-docs ## Deploy documentation to GitHub Pages
	$(call print_header,Deploying documentation)
	@$(MKDOCS) gh-deploy --force
	$(call print_success,Documentation deployed)

# ============================================================================
##@ Build & Distribution
# ============================================================================

build: clean ## Build source and wheel distribution
	$(call print_header,Building distribution packages)
	@$(VENV_PYTHON) -m pip install --upgrade build
	@$(VENV_PYTHON) -m build
	$(call print_success,Distribution packages built in dist/)

release: clean build ## Build and upload to PyPI (requires credentials)
	$(call print_header,Uploading to PyPI)
	@$(VENV_PYTHON) -m pip install --upgrade twine
	@$(VENV_PYTHON) -m twine upload dist/*
	$(call print_success,Package uploaded to PyPI)

release-test: clean build ## Build and upload to Test PyPI
	$(call print_header,Uploading to Test PyPI)
	@$(VENV_PYTHON) -m pip install --upgrade twine
	@$(VENV_PYTHON) -m twine upload --repository testpypi dist/*
	$(call print_success,Package uploaded to Test PyPI)

# ============================================================================
##@ Cleaning
# ============================================================================

clean: ## Remove build artifacts and cached files
	$(call print_header,Cleaning build artifacts)
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf .eggs/
	@rm -rf site/
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete
	@find . -type f -name '*.pyo' -delete
	@find . -type f -name '*~' -delete
	@find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	$(call print_success,Build artifacts cleaned)

clean-test: ## Remove test and coverage artifacts
	$(call print_header,Cleaning test artifacts)
	@rm -rf .pytest_cache/
	@rm -rf .tox/
	@rm -rf $(COVERAGE_FILE)
	@rm -rf $(COVERAGE_REPORT)/
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	$(call print_success,Test artifacts cleaned)

clean-docs: ## Remove documentation build artifacts
	$(call print_header,Cleaning documentation artifacts)
	@rm -rf site/
	@rm -rf docs/_build/
	$(call print_success,Documentation artifacts cleaned)

clean-venv: ## Remove virtual environment
	$(call print_header,Removing virtual environment)
	@rm -rf $(VENV_NAME)
	$(call print_success,Virtual environment removed)

clean-all: clean clean-test clean-docs ## Remove all artifacts and caches
	$(call print_header,Performing deep clean)
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf .eggs/
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
	$(call print_success,All artifacts cleaned)

distclean: clean-all clean-venv ## Remove everything including virtual environment
	$(call print_success,Complete clean finished)

# ============================================================================
##@ Development Workflow
# ============================================================================

dev-setup: install-dev ## Complete development environment setup
	$(call print_success,Development environment ready!)
	@echo "$(COLOR_BLUE)Next steps:$(COLOR_RESET)"
	@echo "  1. Activate the virtual environment: source $(VENV_NAME)/bin/activate"
	@echo "  2. Start coding!"
	@echo "  3. Run 'make test' to verify your changes"

ci: clean install-test quality test-coverage ## Run full CI pipeline locally
	$(call print_success,CI pipeline completed successfully!)

pre-push: quality test ## Run checks before pushing (format, lint, typecheck, test)
	$(call print_success,Ready to push!)

quick-check: format lint ## Quick format and lint check
	$(call print_success,Quick check completed!)

# ============================================================================
##@ Maintenance
# ============================================================================

update-deps: venv ## Update all dependencies
	$(call print_header,Updating dependencies)
	@$(VENV_UV) pip install --upgrade -e .[test,examples,docs]
	@$(PRE_COMMIT) autoupdate
	$(call print_success,Dependencies updated)

show-deps: venv ## Show installed dependencies
	$(call print_header,Installed dependencies)
	@$(VENV_PIP) list

show-outdated: venv ## Show outdated dependencies
	$(call print_header,Outdated dependencies)
	@$(VENV_PIP) list --outdated

freeze: venv ## Freeze current dependencies
	$(call print_header,Freezing dependencies)
	@$(VENV_PIP) freeze > requirements-frozen.txt
	$(call print_success,Dependencies frozen to requirements-frozen.txt)

# ============================================================================
##@ Utility
# ============================================================================

shell: venv ## Open Python shell with project context
	$(call print_header,Opening Python shell)
	@$(VENV_PYTHON)

info: ## Show project information
	@echo "$(COLOR_BOLD)Project Information$(COLOR_RESET)"
	@echo "  Python version:    $(PYTHON_VERSION)"
	@echo "  Virtual env:       $(VENV_NAME)"
	@echo "  Source directory:  $(SRC_DIR)"
	@echo "  Test directory:    $(TEST_DIR)"
	@echo "  Coverage minimum:  $(COVERAGE_MIN)%"

validate: ## Validate project structure and configuration
	$(call print_header,Validating project structure)
	@test -f pyproject.toml || (echo "$(COLOR_YELLOW)Missing pyproject.toml$(COLOR_RESET)" && exit 1)
	@test -d $(SRC_DIR) || (echo "$(COLOR_YELLOW)Missing source directory$(COLOR_RESET)" && exit 1)
	@test -d $(TEST_DIR) || (echo "$(COLOR_YELLOW)Missing test directory$(COLOR_RESET)" && exit 1)
	$(call print_success,Project structure is valid)

# ============================================================================
##@ Aliases & Shortcuts
# ============================================================================

all: clean install-dev test ## Default target: clean, install, and test
fmt: format ## Alias for format
check-format: format-check ## Alias for format-check
t: test ## Alias for test
tc: test-coverage ## Alias for test-coverage
l: lint ## Alias for lint
lf: lint-fix ## Alias for lint-fix
type: typecheck ## Alias for typecheck
