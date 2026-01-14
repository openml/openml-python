# Local Test Server Setup for OpenML-Python

This document explains how to use the local test server infrastructure to run tests without relying on the remote test.openml.org server.

## Problem Statement

Previously, tests relied on the remote test server (`test.openml.org`), which led to several issues:

1. **Race Conditions**: Multiple CI jobs running in parallel could create conflicts in the shared database
2. **Server Load**: High server load causing timeouts and 500 errors
3. **Flaky Tests**: Sporadic failures unrelated to code changes, making CI unreliable
4. **Network Issues**: Timeouts when fetching data from remote server

## Solution

We've implemented a **local Docker-based test infrastructure** that allows tests to run against a local server instance, eliminating race conditions and server dependencies.

### Architecture

The local test setup consists of three Docker services:

1. **test-database**: MySQL database for storing OpenML data
2. **php-api-v1**: PHP-based OpenML API v1 (current production API)
3. **python-api-v2**: Python-based OpenML API v2 (future migration target, see #1575)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ with openml-python installed

### Running Tests Locally

```bash
# Start the local test server
./docker/test-server.sh start

# Run tests against local server
pytest --local-server

# Run only tests that use the test server
pytest --local-server -m uses_test_server

# Stop the local server
./docker/test-server.sh stop
```

### Test Server Management

The `test-server.sh` script provides easy management of local services:

```bash
# Start services
./docker/test-server.sh start

# Check status
./docker/test-server.sh status

# View logs
./docker/test-server.sh logs

# Restart services
./docker/test-server.sh restart

# Stop services
./docker/test-server.sh stop
```

## Configuration

### Pytest Options

- `--local-server`: Use local Docker server instead of test.openml.org
- `--local-server-url URL`: Specify custom local server URL (default: http://localhost:8080/api/v1/xml)

### Environment Variables

Tests automatically configure the OpenML client when using `--local-server`:

```python
# In tests/pytest_openml_server.py
openml.config.server = "http://localhost:8080/api/v1/xml"
```

## CI Integration

The GitHub Actions workflow `.github/workflows/test.yml` includes a `test-local-server` job that:

1. Sets up MySQL database service
2. Starts a mock API server (will use official images in production)
3. Runs tests marked with `@pytest.mark.uses_test_server`

### Current CI Behavior

- **Standard tests**: Run with `-m "not uses_test_server"` (skips server tests)
- **Production tests**: Run against production server (`www.openml.org`)
- **Local server tests**: Run with `--local-server` flag (new!)

## Migration Path

### Short-term (Current Implementation)

✅ Docker Compose configuration for local services  
✅ Pytest plugin for server configuration  
✅ CI workflow with local server job  
✅ Management scripts for local development  

### Mid-term (Next Steps)

- [ ] Replace mock server with official OpenML PHP API Docker image
- [ ] Add database initialization scripts with test data
- [ ] Remove `xfail` markers from server tests
- [ ] Update CI to run all server tests with local instance

### Long-term (Future Goals)

- [ ] Migrate to Python API v2 (see #1575)
- [ ] Separate server API tests from SDK tests
- [ ] CRON-based server stress testing
- [ ] Production-like test environment with realistic data

## Development

### Adding New Server Tests

When writing tests that interact with the OpenML server:

```python
import pytest

@pytest.mark.uses_test_server()
def test_my_feature():
    # This test will run against local server when using --local-server
    dataset = openml.datasets.get_dataset(1)
    assert dataset is not None
```

### Debugging

```bash
# Start server and view logs in real-time
./docker/test-server.sh start
./docker/test-server.sh logs

# Run specific test with verbose output
pytest --local-server -sv tests/test_datasets/test_dataset.py::test_specific_test

# Check service health
./docker/test-server.sh status
```

### Local Server URLs

When services are running:

- MySQL Database: `localhost:3307`
- PHP API v1: `http://localhost:8080`
- Python API v2: `http://localhost:8000`

## Troubleshooting

### Services won't start

```bash
# Check if ports are already in use
lsof -i :3307
lsof -i :8080
lsof -i :8000

# Stop any conflicting services
./docker/test-server.sh stop

# Remove all containers and volumes
docker-compose -f docker/docker-compose.test.yml down -v
```

### Tests fail with local server

The current implementation uses a mock server for demonstration. Some tests may fail until official OpenML server images are integrated.

```bash
# View detailed error logs
pytest --local-server -sv -vv --tb=long

# Check server logs
./docker/test-server.sh logs
```

### Database connection issues

```bash
# Verify MySQL service is healthy
docker ps | grep openml-test-db

# Check MySQL logs
docker logs openml-test-db

# Test connection manually
mysql -h 127.0.0.1 -P 3307 -u openml -popenml openml_test
```

## Related Issues and PRs

- #1586: Main issue for flaky tests and race conditions
- #1587: Temporary fix with xfail markers (to be removed)
- #1613: Additional xfail markers (to be removed)
- #1614: Test plan for local server setup
- #1575: V1 → V2 API migration

## Contributing

To contribute to the test infrastructure:

1. Test changes locally with `./docker/test-server.sh start`
2. Ensure tests pass with both `--local-server` and remote server
3. Update documentation if adding new features
4. Submit PR with clear description of changes

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Pytest Plugins](https://docs.pytest.org/en/stable/writing_plugins.html)
- [OpenML API Documentation](https://www.openml.org/apis)
