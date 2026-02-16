# OpenML Development Environment Setup Guide

This document describes the working and reproducible approach to setting up a local development environment for OpenML. While multiple configurations are possible, the steps below reflect a setup that has been tested and verified in practice.

OpenML currently provides two backend implementations:

* **API v1**: The legacy PHP-based server (currently in production)
* **API v2**: The newer Python-based server built with FastAPI

According to the OpenML maintainers, API v1 will remain operational until at least the end of 2026, while development and migration efforts continue on API v2.

This guide covers:
* Local setup of API v1 (PHP backend)
* Local setup of API v2 (Python backend)
* Development setup for the Python SDK (openml-python)
___

## API v1 (PHP Backend) Setup

### Prerequisites

* Docker Desktop
* Git

### Step 1: Clone the Services Repository

Fork and clone the OpenML services repository:

```bash
git clone https://github.com/openml/services
cd services
```

### Step 2: Install Docker Desktop

Download and install Docker Desktop from: https://www.docker.com/products/docker-desktop/

Ensure Docker is running before proceeding.

### Step 3: Initialize File Permissions

On first use, you must ensure that the PHP data directory has the correct permissions. From the repository root, run:

```bash
chown -R www-data:www-data data/php
```

If this fails (for example, if www-data does not exist on your system), use:

```bash
chmod -R 777 data/php
```

### Step 4: Start the API v1 Services

With Docker running in the background, start all services using:

```bash
docker compose --profile all up -d
```

**Handling Container Name Conflicts**

If you have previously set up the API v2 (Python backend), you may encounter container name conflicts, as some service names overlap.

To resolve this:

Remove or rename the conflicting containers (easiest via Docker Desktop), then restart:

```bash
docker compose --profile all down
docker compose --profile all up -d
```

### Step 5: Verify the API v1 Server

Confirm that the server is running correctly by opening: http://localhost:8080/api/v1/json/flow/181

A successful setup will return structured JSON data describing an OpenML flow.

### Step 6: Configure `openml-python` to Use the Local API v1 Server

To interact with your local API v1 instance, configure the OpenML client to point to the local server and use the API key defined in:

`services/config/php/.env`

**Example usage:**

```python
import openml

openml.config.server = "http://localhost:8080/api/v1/xml"
openml.config.apikey = "AD000000000000000000000000000000"

from openml_sklearn.extension import SklearnExtension
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
extension = SklearnExtension()
knn_flow = extension.model_to_flow(clf)

knn_flow.publish()
```

If successful, the flow will be uploaded to your local OpenML server and you should see output similar to:

```bash
OpenML Flow
===========
Flow ID.........: 182 (version 1)
Flow URL........: http://localhost:8080/f/182
Flow Name.......: sklearn.neighbors._classification.KNeighborsClassifier
Flow Description: Classifier implementing the k-nearest neighbors vote.
Upload Date.....: 2025-12-30 09:11:17
Dependencies....:
  - sklearn==1.8.0
  - numpy>=1.24.1
  - scipy>=1.10.0
  - joblib>=1.3.0
  - threadpoolctl>=3.2.0
```
---

## API v2 (Python Backend) Setup

### Prerequisites

* Docker Desktop
* Git

### Step 1: Clone the Server API Repository

Fork and clone the API v2 repository:

```bash
git clone https://github.com/openml/server-api
cd server-api
```

### Step 2: Install Docker Desktop

If not already installed, download Docker Desktop from:

https://www.docker.com/products/docker-desktop/

Ensure Docker is running.

### Step 3: Build and Start the Services

From the repository root, run:

```bash
docker compose --profile all up
```

This will build and start all containers and expose the services on your local machine.

### Step 4: Verify the API v2 Server

Once the containers are running, verify the setup using the following endpoints:

* FastAPI backend (v2): http://localhost:8001/tasks/31
* Swagger UI documentation: http://localhost:8001/docs


Both endpoints should return meaningful responses if the setup is successful.

---

## Python SDK (openml-python) Development Setup

### Prerequisites

* Python
* Git
* A virtual environment manager (conda, venv, uv, etc.)

### Step 1: Clone the Python SDK Repository

Fork and clone the OpenML Python client repository:

```bash
git clone https://github.com/openml/openml-python
cd openml-python
```

### Step 2: Create and Activate a Virtual Environment

You may use any environment manager. Below is an example using conda:

```bash
conda create -n openml-python-dev python=3.12
conda activate openml-python-dev
```

### Step 3: Install Development Dependencies

Install the package in editable mode along with development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

### Step 4: Enable Pre-commit Hooks

Install and run the pre-commit hooks to ensure code quality and formatting:

```bash
pre-commit install
pre-commit run --all-files
```
---

### Running Tests with Pytest Markers

We are using pytest markers in OpenML Python SDK to categorize tests based on their dependencies and execution requirements. This allows developers to selectively include or skip certain groups of tests depending on their local setup.

#### Available Markers

* `sklearn`: Marks tests that require scikit-learn. These tests are skipped if scikit-learn is not installed.

* `production`: Marks tests that interact with the production OpenML server. These typically involve real API calls.

* `uses_test_server`: Marks tests that require OpenML test server.

#### Run full test suite:

```python
pytest
```

#### Run only tests with a specific marker (for example, `sklearn`):

```python
pytest -m sklearn
```

#### Run multiple markers using logical expressions:

```python
pytest -m "sklearn and not production"
```

### Skip tests that require the production server:

```python
pytest -m "not production"
```

### To list all registered pytest markers in the repository, run:

```python
pytest --markers
```

### Running Tests That Require Admin Privileges

Some tests require admin privileges on the test server and will be automatically skipped unless you provide an admin API key. For regular contributors, the tests will skip gracefully. For core contributors who need to run these tests locally:

Set up the key by exporting the variable:
run this in the terminal before running the tests:

```bash
# For windows
$env:OPENML_TEST_SERVER_ADMIN_KEY = "admin-key"
# For linux/mac
export OPENML_TEST_SERVER_ADMIN_KEY="admin-key"
```


## Notes and Recommendations

API v1 and API v2 can be run side-by-side, but container name conflicts must be resolved manually.

API v1 is required for many existing workflows (e.g. flow publishing).

API v2 is under active development and is the future direction of the OpenML platform.
