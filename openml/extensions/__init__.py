from typing import Any, Optional, Type, TYPE_CHECKING

from .extension_interface import Extension

if TYPE_CHECKING:
    from openml.flows import OpenMLFlow


extensions = []


def register_extension(extension: Type[Extension]) -> None:
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
    extensions.append(extension)


def get_extension_by_flow(
    flow: 'OpenMLFlow',
    raise_if_no_extension: bool = False,
) -> Optional[Extension]:
    """Get an extension which can handle the given flow.

    Iterates all registered extensions and checks whether they can handle the presented flow.
    In case an extension can, it is immediately returned.

    Parameters
    ----------
    flow : OpenMLFlow

    raise_if_no_extension : bool (optional, default=False)
        Raise an exception if no registered extension can handle the presented flow.

    Returns
    -------
    Extension or None
    """
    for extension_class in extensions:
        if extension_class.can_handle_flow(flow):
            return extension_class()
    if raise_if_no_extension:
        raise ValueError('No extension registered which can handle flow: %s' % flow)
    else:
        return None


def get_extension_by_model(
    model: Any,
    raise_if_no_extension: bool = False,
) -> Optional[Extension]:
    """Get an extension which can handle the given flow.

    Iterates all registered extensions and checks whether they can handle the presented model.
    In case an extension can, it is immediately returned.

    Parameters
    ----------
    model : Any

    raise_if_no_extension : bool (optional, default=False)
        Raise an exception if no registered extension can handle the presented model.

    Returns
    -------
    Extension or None
    """
    for extension_class in extensions:
        if extension_class.can_handle_model(model):
            return extension_class()
    if raise_if_no_extension:
        raise ValueError('No extension registered which can handle model: %s' % model)
    else:
        return None


__all__ = [
    'Extension',
    'register_extension',
    'get_extension_by_model',
    'get_extension_by_flow',
]
