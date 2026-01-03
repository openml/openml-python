from openml._api.runtime.core import APIContext


def set_api_version(version: str, strict=False):
    api_context.set_version(version=version, strict=strict)


api_context = APIContext()
