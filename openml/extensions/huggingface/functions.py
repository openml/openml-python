"""Core functions for HuggingFace Hub integration."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openml.flows import OpenMLFlow

try:
    from huggingface_hub import HfApi, create_repo, hf_hub_download

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from openml.exceptions import PyOpenMLError

from .config import get_config


def _check_huggingface_available() -> None:
    """Check if huggingface_hub is installed."""
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "HuggingFace Hub integration requires 'huggingface_hub'. "
            "Install with: pip install huggingface_hub"
        )


def upload_flow_to_huggingface(
    flow: OpenMLFlow,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """Upload an OpenML flow to HuggingFace Hub.

    This function creates a model repository on HuggingFace Hub and uploads:
    1. The serialized model (pickle format)
    2. OpenML flow metadata (JSON)
    3. A model card with documentation

    Parameters
    ----------
    flow : OpenMLFlow
        OpenML flow to upload. Must have a valid flow_id (i.e., published to OpenML).
    repo_id : str
        Repository name in format 'username/repo-name' or 'organization/repo-name'.
    token : str
        HuggingFace API token with write access.
    private : bool, default=False
        Whether to create a private repository.
    commit_message : str, optional
        Custom commit message. If None, uses default from config.

    Returns
    -------
    str
        URL of the uploaded model on HuggingFace Hub.

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    PyOpenMLError
        If the flow has no flow_id or model.

    Examples
    --------
    >>> import openml
    >>> from openml.extensions.huggingface import upload_flow_to_huggingface
    >>>
    >>> # Get a flow from OpenML
    >>> flow = openml.flows.get_flow(12345, reinstantiate=True)
    >>>
    >>> # Upload to HuggingFace
    >>> url = upload_flow_to_huggingface(
    ...     flow=flow,
    ...     repo_id="my-username/my-sklearn-model",
    ...     token="hf_xxxxx",
    ...     private=False,
    ... )
    >>> print(f"Model uploaded to: {url}")
    """
    _check_huggingface_available()

    config = get_config()

    if flow.flow_id is None:
        raise PyOpenMLError(
            "Flow must be published to OpenML before uploading to HuggingFace. "
            "Use flow.publish() first."
        )

    if flow.model is None:
        raise PyOpenMLError(
            "Flow must have a model instance. "
            "Use openml.flows.get_flow(flow_id, reinstantiate=True)."
        )

    # Create repository
    api = HfApi()
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
    except Exception as e:
        raise PyOpenMLError(f"Failed to create HuggingFace repository: {e}") from e

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. Save the model
        model_path = tmpdir_path / config.model_filename
        with model_path.open("wb") as f:
            pickle.dump(flow.model, f)

        # 2. Save OpenML metadata
        metadata = {
            "openml_flow_id": flow.flow_id,
            "openml_flow_name": flow.name,
            "openml_url": f"https://www.openml.org/f/{flow.flow_id}",
            "flow_description": flow.description,
            "dependencies": flow.dependencies,
            "parameters": flow.parameters,
            "external_version": flow.external_version,
        }
        metadata_path = tmpdir_path / config.metadata_filename
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        # 3. Create model card
        model_card = _create_model_card(flow)
        card_path = tmpdir_path / config.readme_filename
        with card_path.open("w") as f:
            f.write(model_card)

        # Upload files
        commit_msg = commit_message or config.default_commit_message

        try:
            for file_path in [model_path, metadata_path, card_path]:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_msg,
                )
        except Exception as e:
            raise PyOpenMLError(f"Failed to upload files to HuggingFace: {e}") from e

    return f"https://huggingface.co/{repo_id}"


def download_flow_from_huggingface(
    repo_id: str,
    token: str | None = None,
    local_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Download a model and its OpenML metadata from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        Repository name in format 'username/repo-name'.
    token : str, optional
        HuggingFace API token (required for private repos).
    local_dir : str or Path, optional
        Directory to save downloaded files. If None, uses cache directory from config.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': The deserialized model object
        - 'metadata': OpenML flow metadata (dict)
        - 'model_path': Path to downloaded model file
        - 'metadata_path': Path to metadata file

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    FileNotFoundError
        If required files are not found in the repository.

    Examples
    --------
    >>> from openml.extensions.huggingface import download_flow_from_huggingface
    >>>
    >>> # Download model and metadata
    >>> result = download_flow_from_huggingface("my-username/my-sklearn-model")
    >>> model = result['model']
    >>> metadata = result['metadata']
    >>>
    >>> print(f"Original OpenML Flow ID: {metadata['openml_flow_id']}")
    """
    _check_huggingface_available()

    config = get_config()

    if local_dir is None:
        cache_dir = config.cache_dir
        if cache_dir is None:
            raise RuntimeError("Cache directory is not configured")
        local_dir = cache_dir / repo_id.replace("/", "_")

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download model
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=config.model_filename,
            token=token,
            local_dir=str(local_dir),
        )

        # Download metadata
        metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename=config.metadata_filename,
            token=token,
            local_dir=str(local_dir),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download model from {repo_id}. "
            f"Make sure the repository exists and contains the required files. "
            f"Error: {e}"
        ) from e

    # Load model
    # Note: pickle.load can be unsafe with untrusted data.
    # Only use with models from trusted sources.
    model_path_obj = Path(model_path)
    with model_path_obj.open("rb") as f:
        model = pickle.load(f)  # noqa: S301

    # Load metadata
    metadata_path_obj = Path(metadata_path)
    with metadata_path_obj.open() as f:
        metadata = json.load(f)

    return {
        "model": model,
        "metadata": metadata,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


def _create_model_card(flow: OpenMLFlow) -> str:
    """Create a HuggingFace model card for an OpenML flow."""
    card = f"""---
tags:
- openml
- scikit-learn
- machine-learning
library_name: sklearn
---

# {flow.name}

This model was uploaded from [OpenML](https://www.openml.org/f/{flow.flow_id}).

## Model Description

{flow.description or "No description provided."}

## OpenML Information

- **Flow ID**: {flow.flow_id}
- **Flow Name**: {flow.name}
- **External Version**: {flow.external_version}
- **OpenML URL**: https://www.openml.org/f/{flow.flow_id}

## Dependencies

```
{flow.dependencies or "No dependencies listed."}
```

## Parameters

"""

    if flow.parameters:
        for param_name, param_value in flow.parameters.items():
            card += f"- `{param_name}`: {param_value}\n"
    else:
        card += "No parameters defined.\n"

    card += """
## Usage

```python
import pickle
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id="REPO_ID", filename="model.pkl")

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Use the model
# predictions = model.predict(X)
```

## Citation

```bibtex
@misc{openml,
  author = {OpenML},
  title = {OpenML: Open Machine Learning},
  year = {2023},
  url = {https://www.openml.org}
}
```
"""
    return card