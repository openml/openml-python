"""Pytest plugin for configuring OpenML test server URL.

This plugin allows tests to use a local test server instead of the remote
test.openml.org server. This helps avoid race conditions and server load issues.

Usage:
    pytest --local-server  # Use local Docker server at http://localhost:8080
    pytest                 # Use remote test.openml.org (default)
"""
from __future__ import annotations

import os
import pytest
import openml


def pytest_addoption(parser):
    """Add command-line options for test server configuration."""
    parser.addoption(
        "--local-server",
        action="store_true",
        default=False,
        help="Use local Docker-based test server instead of test.openml.org",
    )
    parser.addoption(
        "--local-server-url",
        action="store",
        default="http://localhost:8080/api/v1/xml",
        help="URL of local test server (default: http://localhost:8080/api/v1/xml)",
    )


def pytest_configure(config):
    """Configure test server URL based on command-line options."""
    config.addinivalue_line(
        "markers",
        "uses_test_server: mark test as using the OpenML test server",
    )
    
    # If local server is enabled, configure OpenML to use it
    if config.getoption("--local-server"):
        local_url = config.getoption("--local-server-url")
        # Store original config to restore later if needed
        config._original_test_server = openml.config.server
        openml.config.server = local_url
        print(f"\n[pytest-openml] Using local test server: {local_url}")


def pytest_unconfigure(config):
    """Restore original server configuration after tests."""
    if hasattr(config, "_original_test_server"):
        openml.config.server = config._original_test_server


@pytest.fixture(scope="session", autouse=True)
def configure_test_server(request):
    """Session-level fixture to configure test server.
    
    This ensures the test server URL is properly set for all tests
    that use the @pytest.mark.uses_test_server decorator.
    """
    config = request.config
    if config.getoption("--local-server"):
        # Verify local server is accessible
        local_url = config.getoption("--local-server-url")
        print(f"[pytest-openml] Test server configured: {local_url}")
    else:
        print("[pytest-openml] Using remote test server: https://test.openml.org")
    
    yield
    
    # Cleanup after all tests
    if hasattr(config, "_original_test_server"):
        print("[pytest-openml] Restored original server configuration")
