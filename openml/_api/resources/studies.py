from __future__ import annotations

from typing import Any

from openml._api.resources.base import StudiesAPI


class StudiesV1(StudiesAPI):
    def list(self, **kwargs: Any) -> Any:
        limit = kwargs.get("limit")
        offset = kwargs.get("offset")
        status = kwargs.get("status")
        main_entity_type = kwargs.get("main_entity_type")
        uploader = kwargs.get("uploader")
        benchmark_suite = kwargs.get("benchmark_suite")

        api_call = "study/list"

        if limit is not None:
            api_call += f"/limit/{limit}"
        if offset is not None:
            api_call += f"/offset/{offset}"
        if status is not None:
            api_call += f"/status/{status}"
        if main_entity_type is not None:
            api_call += f"/main_entity_type/{main_entity_type}"
        if uploader is not None:
            api_call += f"/uploader/{','.join(str(u) for u in uploader)}"
        if benchmark_suite is not None:
            api_call += f"/benchmark_suite/{benchmark_suite}"

        # Make the GET request and return the XML text
        response = self._http.get(api_call)
        return response.text


class StudiesV2(StudiesAPI):
    def list(self, **kwargs: Any) -> Any:
        raise NotImplementedError("V2 API implementation is not yet available")
