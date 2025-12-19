.PHONY: test test-fast test-slow test-all clean install

# Run fast tests (excludes slow tests like full downloads)
test-fast:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v -m "not slow and not integration"

# Run only slow tests
test-slow:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v -m "slow"

# Run all tests
test-all:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v

# Run tests with coverage
test-cov:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v --cov=colliderml --cov-report=term-missing --cov-report=xml

# Clean up cache and build artifacts
clean:
	rm -rf .pytest_cache __pycache__ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install package in development mode
install:
	pip install -e ".[dev]"
