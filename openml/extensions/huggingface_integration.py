from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import openml

if TYPE_CHECKING:
    from openml.runs import OpenMLRun

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi
    from transformers import AutoModel, PreTrainedModel

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    PreTrainedModel = object  # type: ignore


def is_hf_transformer(model: Any) -> bool:
    """Check if a model is a Hugging Face Transformers model."""
    if not _HF_AVAILABLE:
        return False
    return isinstance(model, PreTrainedModel)


def push_model_to_hub_for_run(
    model: Any,
    run: OpenMLRun,
    repo_id: str,
    token: str | None = None,
) -> OpenMLRun:
    """
    Push a Hugging Face model to the Hub and link it to an OpenML run.

    If the model is not a Hugging Face model, the run is returned unchanged.
    If the model is a Hugging Face model, it is pushed to the Hub, and a tag
    referencing the commit is added to the run.

    Parameters
    ----------
    model : Any
        The model to push.
    run : OpenMLRun
        The OpenML run to link.
    repo_id : str
        The ID of the repository to push to (e.g. "username/repo_name").
    token : str, optional
        The Hugging Face authentication token.

    Returns
    -------
    OpenMLRun
        The updated OpenML run.
    """
    if not is_hf_transformer(model):
        return run

    if not _HF_AVAILABLE:
        # Should be unreachable if is_hf_transformer works correctly,
        # but good for safety if logic changes.
        logger.warning("Hugging Face integration dependencies not found. Skipping push.")
        return run

    # 1. Push to Hub
    model.push_to_hub(repo_id, commit_message=f"OpenML Run {run.run_id}", token=token)

    # 2. Get latest commit
    api = HfApi(token=token)
    commit_sha = api.list_repo_commits(repo_id)[0].commit_id

    # 3. Construct URI
    # Format: hf://{user_or_org}/{repo_name}@{commit_sha}
    hf_uri = f"hf://{repo_id}@{commit_sha}"

    # 4. Store URI in tags
    run.tags.append(f"hf_uri={hf_uri}")
    run.tags.append("hf-integrated")

    return run


def load_model_from_run(
    run_id: int,
    token: str | None = None,
) -> Any:
    """
    Load a Hugging Face model linked to an OpenML run.

    Parameters
    ----------
    run_id : int
        The ID of the OpenML run.
    token : str, optional
        The Hugging Face authentication token.

    Returns
    -------
    Any
        The loaded Hugging Face model.

    Raises
    ------
    ImportError
        If Hugging Face dependencies are not installed.
    ValueError
        If the run does not have a linked Hugging Face model.
    """
    if not _HF_AVAILABLE:
        raise ImportError("Hugging Face integration requires 'huggingface_hub' and 'transformers'.")

    run = openml.runs.get_run(run_id)

    hf_uri = None
    for tag in run.tags:
        if tag.startswith("hf_uri="):
            hf_uri = tag.split("=", 1)[1]
            break

    if not hf_uri:
        raise ValueError(
            f"Run {run_id} does not have a linked Hugging Face model (no 'hf_uri' tag)."
        )

    # Parse URI: hf://{repo_id}@{commit_sha}
    # Remove hf://
    uri_path = hf_uri[5:]
    if "@" not in uri_path:
        raise ValueError(f"Invalid HF URI format: {hf_uri}")

    repo_id, commit_sha = uri_path.split("@", 1)

    # Load model
    return AutoModel.from_pretrained(repo_id, revision=commit_sha, token=token)


def run_task_with_hf_sync(
    model: Any,
    task_id: int,
    repo_id: str,
    hf_token: str | None = None,
) -> OpenMLRun:
    """
    Run a task and sync the model to Hugging Face Hub.

    Parameters
    ----------
    model : Any
        The model to run.
    task_id : int
        The ID of the task to run.
    repo_id : str
        The Hugging Face repository ID to push to.
    hf_token : str, optional
        The Hugging Face authentication token.

    Returns
    -------
    OpenMLRun
        The published OpenML run.
    """
    task = openml.tasks.get_task(task_id)
    run = openml.runs.run_model_on_task(model, task)
    run = push_model_to_hub_for_run(model, run, repo_id=repo_id, token=hf_token)
    run.publish()
    return run
