# OpenML Local Development Environment Setup

This guide outlines the standard procedures for setting up a local development environment for the OpenML ecosystem. It covers the configuration of the backend servers (API v1 and API v2) and the Python Client SDK.

OpenML currently has two backend architecture:

* **API v1**: The PHP-based server currently serving production traffic.
* **API v2**: The Python-based server (FastAPI) currently under active development.

> Note on Migration: API v1 is projected to remain operational through at least 2026. API v2 is the target architecture for future development.

## 1. API v1 Setup (PHP Backend)

This section details the deployment of the legacy PHP backend.

### Prerequisites

* **Docker**: Docker Desktop (Ensure the daemon is running).
* **Version Control**: Git.

### Installation Steps

#### 1. Clone the Repository

Retrieve the OpenML services source code:

```bash
git clone https://github.com/openml/services
cd services
```

#### 2. Configure File Permissions

To ensure the containerized PHP service can write to the local filesystem, initialize the data directory permissions.

From the repository root:

```bash
chown -R www-data:www-data data/php
```

If the `www-data` user does not exist on the host system, grant full permissions as a fallback:

```bash
chmod -R 777 data/php
```

#### 3. Launch Services

Initialize the container stack:

```bash
docker compose --profile all up -d
```

#### Warning: Container Conflicts

If API v2 (Python backend) containers are present on the system, name conflicts may occur. To resolve this, stop and remove existing containers before launching API v1:

```bash
docker compose --profile all down
docker compose --profile all up -d
```

#### 4. Verification

Validate the deployment by accessing the flow endpoint. A successful response will return structured JSON data.

* **Endpoint**: http://localhost:8080/api/v1/json/flow/181

### Client Configuration

To direct the `openml-python` client to the local API v1 instance, modify the configuration as shown below. The API key corresponds to the default key located in `services/config/php/.env`.

```python
import openml
from openml_sklearn.extension import SklearnExtension
from sklearn.neighbors import KNeighborsClassifier

# Configure client to use local Docker instance
openml.config.server = "http://localhost:8080/api/v1/xml"
openml.config.apikey = "AD000000000000000000000000000000"

# Test flow publication
clf = KNeighborsClassifier(n_neighbors=3)
extension = SklearnExtension()
knn_flow = extension.model_to_flow(clf)

knn_flow.publish()
```

## 2. API v2 Setup (Python Backend)

This section details the deployment of the FastAPI backend.

### Prerequisites

* **Docker**: Docker Desktop (Ensure the daemon is running).
* **Version Control**: Git.

### Installation Steps

#### 1. Clone the Repository

Retrieve the API v2 source code:

```bash
git clone https://github.com/openml/server-api
cd server-api
```

#### 2. Launch Services

Build and start the container stack:

```bash
docker compose --profile all up
```

#### 3. Verification

Validate the deployment using the following endpoints:

* **Task Endpoint**: http://localhost:8001/tasks/31
* **Swagger UI (Documentation)**: http://localhost:8001/docs

## 3. Python SDK (`openml-python`) Setup

This section outlines the environment setup for contributing to the OpenML Python client.

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/openml/openml-python
cd openml-python
```

#### 2. Environment Initialization

Create an isolated virtual environment (example using Conda):

```bash
conda create -n openml-python-dev python=3.12
conda activate openml-python-dev
```

#### 3. Install Dependencies

Install the package in editable mode, including development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

#### 4. Configure Quality Gates

Install pre-commit hooks to enforce coding standards:

```bash
pre-commit install
pre-commit run --all-files
```

## 4. Testing Guidelines

The OpenML Python SDK utilizes `pytest` markers to categorize tests based on dependencies and execution context.

| Marker            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `sklearn`          | Tests requiring `scikit-learn`. Skipped if the library is missing.          |
| `production_server`| Tests that interact with the live OpenML server (real API calls).         |
| `test_server`     | Tests requiring the OpenML test server environment.                       |

### Execution Examples

Run the full test suite:

```bash
pytest
```

Run a specific subset (e.g., `scikit-learn` tests):

```bash
pytest -m sklearn
```

Exclude production tests (local only):

```bash
pytest -m "not production_server"
```

### Running Tests Against Local Services

The CI pipeline runs tests against a local OpenML server using Docker, instead of the remote
test server. You can replicate this setup locally to avoid depending on the remote server.

#### Prerequisites

* **Docker**: Docker Desktop (ensure the daemon is running).
* **Git**: To clone the services repository.

#### 1. Clone the Services Repository

```bash
git clone --depth 1 https://github.com/openml/services.git
cd services
```

#### 2. Configure File Permissions

The containerized services need write access to the data and log directories:

```bash
chmod -R a+rw ./data
chmod -R a+rw ./logs
```

#### 3. Start the Docker Services

Launch the required service profiles (REST API, MinIO, and Evaluation Engine):

```bash
docker compose --profile rest-api --profile minio --profile evaluation-engine up -d
```

Wait for the PHP API to become healthy:

```bash
# Wait up to 60 seconds for the API to boot
timeout 60s bash -c 'until [ "$(docker inspect -f {{.State.Health.Status}} openml-php-rest-api)" == "healthy" ]; do sleep 5; done'
```

Verify the services are running:

```bash
curl -sSfL http://localhost:8000/api/v1/xml/data/1 | head -n 15
```

#### 4. Run Tests Against Local Services

Set the `OPENML_USE_LOCAL_SERVICES` environment variable and run pytest:

=== "Linux/macOS"

    ```bash
    export OPENML_USE_LOCAL_SERVICES="true"
    pytest -n 4 --durations=20 --dist load -sv -m "not production_server"
    ```

=== "Windows (PowerShell)"

    ```shell
    $env:OPENML_USE_LOCAL_SERVICES = "true"
    pytest -n 4 --durations=20 --dist load -sv -m "not production_server"
    ```

When this environment variable is set, the test framework automatically redirects API calls
to `http://localhost:8000` instead of the remote test server.

#### 5. Cleanup

After you're done testing, stop and remove the Docker containers:

```bash
cd services
docker compose --profile rest-api --profile minio --profile evaluation-engine down
```

### Admin Privilege Tests

Certain tests require administrative privileges on the test server. These are skipped automatically unless an admin API key is provided via environment variables.

#### Windows (PowerShell):

```shell
$env:OPENML_TEST_SERVER_ADMIN_KEY = "admin-key"
```

#### Linux/macOS:

```bash
export OPENML_TEST_SERVER_ADMIN_KEY="admin-key"
```

