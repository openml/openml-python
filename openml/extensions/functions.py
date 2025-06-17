# License: BSD 3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Need to implement the following by its full path because otherwise it won't be possible to
# access openml.extensions.extensions
import openml.extensions

# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from openml.flows import OpenMLFlow

    from . import Extension


def register_extension(extension: type[Extension]) -> None:
    """Register an extension.

    Registered extensions are considered by ``get_extension_by_flow`` and
    ``get_extension_by_model``, which are used by ``openml.flow`` and ``openml.runs``.

    Parameters
    ----------
    extension : Type[Extension]

    Returns
    -------
    None
    """
    openml.extensions.extensions.append(extension)


def get_extension_by_flow(
    flow: OpenMLFlow,
    raise_if_no_extension: bool = False,  # noqa: FBT001, FBT002
) -> Extension | None:
    """Get an extension which can handle the given flow.

    Iterates all registered extensions and checks whether they can handle the presented flow.
    Raises an exception if two extensions can handle a flow.

    Parameters
    ----------
    flow : OpenMLFlow

    raise_if_no_extension : bool (optional, default=False)
        Raise an exception if no registered extension can handle the presented flow.

    Returns
    -------
    Extension or None
    """
    candidates = []
    for extension_class in openml.extensions.extensions:
        if extension_class.can_handle_flow(flow):
            candidates.append(extension_class())
    if len(candidates) == 0:
        if raise_if_no_extension:
            raise ValueError(f"No extension registered which can handle flow: {flow}")

        return None

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Multiple extensions registered which can handle flow: {flow}, but only one "
        f"is allowed ({candidates}).",
    )


def get_extension_by_model(
    model: Any,
    raise_if_no_extension: bool = False,  # noqa: FBT001, FBT002
) -> Extension | None:
    """Get an extension which can handle the given flow.

    Iterates all registered extensions and checks whether they can handle the presented model.
    Raises an exception if two extensions can handle a model.

    Parameters
    ----------
    model : Any

    raise_if_no_extension : bool (optional, default=False)
        Raise an exception if no registered extension can handle the presented model.

    Returns
    -------
    Extension or None
    """
    candidates = []
    for extension_class in openml.extensions.extensions:
        if extension_class.can_handle_model(model):
            candidates.append(extension_class())
    if len(candidates) == 0:
        if raise_if_no_extension:
            raise ValueError(f"No extension registered which can handle model: {model}")

        return None

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Multiple extensions registered which can handle model: {model}, but only one "
        f"is allowed ({candidates}).",
    )
