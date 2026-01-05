from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openml._api.resources.base import ResourceAPI


class FallbackProxy:
    def __init__(self, primary: ResourceAPI, fallback: ResourceAPI):
        self._primary = primary
        self._fallback = fallback
