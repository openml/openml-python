from typing import Any, Optional, Type, TYPE_CHECKING

from .extension_interface import Extension

if TYPE_CHECKING:
    from openml.flows import OpenMLFlow


extensions = []


def register_extension(extension: Type[Extension]) -> None:
    extensions.append(extension)


def get_extension_by_flow(
    flow: 'OpenMLFlow',
    raise_if_no_extension: bool = False,
) -> Optional[Extension]:
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
    'get_extension_by_flow',
    'get_extension_by_model',
]
