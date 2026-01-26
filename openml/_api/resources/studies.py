from __future__ import annotations

import pandas as pd
import xmltodict

from openml._api.resources.base import StudiesAPI


class StudiesV1(StudiesAPI):
    def list(  # noqa: PLR0913
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: str | None = None,
        main_entity_type: str | None = None,
        uploader: list[int] | None = None,
        benchmark_suite: int | None = None,
    ) -> pd.DataFrame:
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

        response = self._http.get(api_call)
        xml_string = response.text

        # Parse XML and convert to DataFrame
        study_dict = xmltodict.parse(xml_string, force_list=("oml:study",))

        assert isinstance(study_dict["oml:study_list"]["oml:study"], list), type(
            study_dict["oml:study_list"],
        )
        assert study_dict["oml:study_list"]["@xmlns:oml"] == "http://openml.org/openml", study_dict[
            "oml:study_list"
        ]["@xmlns:oml"]

        studies = {}
        for study_ in study_dict["oml:study_list"]["oml:study"]:
            expected_fields = {
                "oml:id": ("id", int),
                "oml:alias": ("alias", str),
                "oml:main_entity_type": ("main_entity_type", str),
                "oml:benchmark_suite": ("benchmark_suite", int),
                "oml:name": ("name", str),
                "oml:status": ("status", str),
                "oml:creation_date": ("creation_date", str),
                "oml:creator": ("creator", int),
            }
            study_id = int(study_["oml:id"])
            current_study = {}
            for oml_field_name, (real_field_name, cast_fn) in expected_fields.items():
                if oml_field_name in study_:
                    current_study[real_field_name] = cast_fn(study_[oml_field_name])
            current_study["id"] = int(current_study["id"])
            studies[study_id] = current_study

        return pd.DataFrame.from_dict(studies, orient="index")


class StudiesV2(StudiesAPI):
    def list(  # noqa: PLR0913
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: str | None = None,
        main_entity_type: str | None = None,
        uploader: list[int] | None = None,
        benchmark_suite: int | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("V2 API implementation is not yet available")
