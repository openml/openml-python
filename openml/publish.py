# License: BSD 3-Clause
from __future__ import annotations

from typing import Any, Sequence

from . import extensions
from .base import OpenMLBase


def publish(obj: Any, *, name: str | None = None, tags: Sequence[str] | None = None) -> Any:
    """Publish a common object (flow/model/run/dataset) with minimal friction.

    This function provides a unified entry point for publishing various OpenML objects.
    It automatically detects the object type and routes to the appropriate publishing
    mechanism:

    - For OpenML objects (``OpenMLDataset``, ``OpenMLFlow``, ``OpenMLRun``, etc.),
      it directly calls their ``publish()`` method.
    - For external estimators (e.g., scikit-learn estimators), it uses registered
      extensions to convert them to ``OpenMLFlow`` objects before publishing.

    Parameters
    ----------
    obj : Any
        The object to publish. Can be:
        - An OpenML object (OpenMLDataset, OpenMLFlow, OpenMLRun, OpenMLTask)
        - An estimator instance from a supported framework (e.g., scikit-learn)
    name : str, optional
        Override the default name for the published object.
        If not provided, uses the object's default naming convention.
    tags : Sequence[str], optional
        Additional tags to attach to the published object.
        Will be merged with any existing tags, removing duplicates while
        preserving order.

    Returns
    -------
    Any
        The published object (typically with updated ID and metadata).

    Raises
    ------
    ValueError
        If no extension is registered to handle the provided estimator type.

    Examples
    --------
    Publishing an OpenML dataset:

    >>> dataset = openml.datasets.get_dataset(61)
    >>> openml.publish(dataset, tags=["example"])

    Publishing a scikit-learn estimator:

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(max_depth=5)
    >>> openml.publish(clf, name="MyDecisionTree", tags=["tutorial"])

    Publishing an OpenML flow directly:

    >>> flow = openml.flows.OpenMLFlow(...)
    >>> openml.publish(flow)

    Publishing an OpenML run (after execution with predictions):

    >>> run = openml.runs.OpenMLRun(
    ...     task_id=1, flow_id=100, dataset_id=61,
    ...     data_content=predictions  # predictions from estimator evaluation
    ... )
    >>> openml.publish(run, tags=["experiment"])

    Notes
    -----
    For external estimators (e.g., scikit-learn), the corresponding extension must be
    installed (e.g., ``openml-sklearn``). The extension will be automatically imported
    if available.
    """
    # Case 1: Object is already an OpenML entity
    if isinstance(obj, OpenMLBase):
        if tags is not None and hasattr(obj, "tags"):
            existing = list(getattr(obj, "tags", []) or [])
            merged = list(dict.fromkeys([*existing, *tags]))
            obj.tags = merged
        if name is not None and hasattr(obj, "name"):
            obj.name = name
        return obj.publish()

    # Case 2: Object is an external estimator - use extension registry
    extension = extensions.functions.get_extension_by_model(obj, raise_if_no_extension=True)
    if extension is None:  # Defensive check (should not occur with raise_if_no_extension=True)
        raise ValueError("No extension registered to handle the provided object.")
    flow = extension.model_to_flow(obj)

    if name is not None:
        flow.name = name

    if tags is not None:
        existing_tags = list(getattr(flow, "tags", []) or [])
        flow.tags = list(dict.fromkeys([*existing_tags, *tags]))

    return flow.publish()
