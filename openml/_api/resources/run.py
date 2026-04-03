from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

import pandas as pd
import xmltodict

import openml
from openml._api.resources.base import ResourceV1API, ResourceV2API, RunAPI
from openml.tasks.task import TaskType

if TYPE_CHECKING:
    from openml.runs.run import OpenMLRun


class RunV1API(ResourceV1API, RunAPI):
    def get(
        self,
        run_id: int,
        *,
        reset_cache: bool = False,
    ) -> OpenMLRun:  # type: ignore[override]
        """Fetch a single run from the OpenML server.

        Parameters
        ----------
        run_id : int
            The ID of the run to fetch.
        reset_cache : bool, default=False
            Whether to reset the cache.

        Returns
        -------
        OpenMLRun
            The run object with all details populated.

        Raises
        ------
        openml.exceptions.OpenMLServerException
            If the run does not exist or server error occurs.
        """
        path = f"run/{run_id}"
        response = self._http.get(
            path,
            enable_cache=True,
            refresh_cache=reset_cache,
        )
        xml_content = response.text
        return openml.runs.functions._create_run_from_xml(xml_content)

    def list(  # type: ignore[valid-type]  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        ids: builtins.list[int] | None = None,
        task: builtins.list[int] | None = None,
        setup: builtins.list[int] | None = None,
        flow: builtins.list[int] | None = None,
        uploader: builtins.list[int] | None = None,
        study: int | None = None,
        tag: str | None = None,
        display_errors: bool = False,
        task_type: TaskType | int | None = None,
    ) -> pd.DataFrame:
        """List runs from the OpenML server with optional filtering.

        Parameters
        ----------
        limit : int
            Maximum number of runs to return.
        offset : int
            Starting position for pagination.
        ids : list of int, optional
            List of run IDs to filter by.
        task : list of int, optional
            List of task IDs to filter by.
        setup : list of int, optional
            List of setup IDs to filter by.
        flow : list of int, optional
            List of flow IDs to filter by.
        uploader : list of int, optional
            List of uploader user IDs to filter by.
        study : int, optional
            Study ID to filter by.
        tag : str, optional
            Tag to filter by.
        display_errors : bool, default=False
            If True, include runs with error messages.
        task_type : TaskType or int, optional
            Task type ID to filter by.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: run_id, task_id, setup_id, flow_id,
            uploader, task_type, upload_time, error_message.

        Raises
        ------
        ValueError
            If the server response is invalid or malformed.
        """
        path = self._build_url(
            limit=limit,
            offset=offset,
            ids=ids,
            task=task,
            setup=setup,
            flow=flow,
            uploader=uploader,
            study=study,
            tag=tag,
            display_errors=display_errors,
            task_type=task_type,
        )
        xml_string = self._http.get(path).text
        return self._parse_list_xml(xml_string)

    def _build_url(  # noqa: PLR0913, C901
        self,
        limit: int,
        offset: int,
        *,
        ids: builtins.list[int] | None = None,
        task: builtins.list[int] | None = None,
        setup: builtins.list[int] | None = None,
        flow: builtins.list[int] | None = None,
        uploader: builtins.list[int] | None = None,
        study: int | None = None,
        tag: str | None = None,
        display_errors: bool = False,
        task_type: TaskType | int | None = None,
    ) -> str:
        path = "run/list"
        if limit is not None:
            path += f"/limit/{limit}"
        if offset is not None:
            path += f"/offset/{offset}"
        if ids is not None:
            path += f"/run/{','.join([str(int(i)) for i in ids])}"
        if task is not None:
            path += f"/task/{','.join([str(int(i)) for i in task])}"
        if setup is not None:
            path += f"/setup/{','.join([str(int(i)) for i in setup])}"
        if flow is not None:
            path += f"/flow/{','.join([str(int(i)) for i in flow])}"
        if uploader is not None:
            path += f"/uploader/{','.join([str(int(i)) for i in uploader])}"
        if study is not None:
            path += f"/study/{study}"
        if display_errors:
            path += "/show_errors/true"
        if tag is not None:
            path += f"/tag/{tag}"
        if task_type is not None:
            tvalue = task_type.value if isinstance(task_type, TaskType) else task_type
            path += f"/task_type/{tvalue}"
        return path

    def _parse_list_xml(self, xml_string: str) -> pd.DataFrame:
        runs_dict = xmltodict.parse(xml_string, force_list=("oml:run",))
        # Minimalistic check if the XML is useful
        if "oml:runs" not in runs_dict:
            raise ValueError(f'Error in return XML, does not contain "oml:runs": {runs_dict}')

        if "@xmlns:oml" not in runs_dict["oml:runs"]:
            raise ValueError(
                f'Error in return XML, does not contain "oml:runs"/@xmlns:oml: {runs_dict}'
            )

        if runs_dict["oml:runs"]["@xmlns:oml"] != "http://openml.org/openml":
            raise ValueError(
                "Error in return XML, value of  "
                '"oml:runs"/@xmlns:oml is not '
                f'"http://openml.org/openml": {runs_dict}',
            )

        assert isinstance(runs_dict["oml:runs"]["oml:run"], list), type(runs_dict["oml:runs"])

        runs = {
            int(r["oml:run_id"]): {
                "run_id": int(r["oml:run_id"]),
                "task_id": int(r["oml:task_id"]),
                "setup_id": int(r["oml:setup_id"]),
                "flow_id": int(r["oml:flow_id"]),
                "uploader": int(r["oml:uploader"]),
                "task_type": TaskType(int(r["oml:task_type_id"])),
                "upload_time": str(r["oml:upload_time"]),
                "error_message": str((r["oml:error_message"]) or ""),
            }
            for r in runs_dict["oml:runs"]["oml:run"]
        }
        return pd.DataFrame.from_dict(runs, orient="index")

    def download_text_file(
        self,
        source: str,
        *,
        md5_checksum: str | None = None,
    ) -> str:
        response = self._http.get(
            source,
            use_api_key=False,
            md5_checksum=md5_checksum,
        )
        return response.text

    def file_id_to_url(
        self,
        file_id: int,
        filename: str | None = None,
    ) -> str:
        server_base = self._http.server.split("/api/", 1)[0].rstrip("/")
        url = f"{server_base}/data/download/{file_id}"
        if filename is not None:
            url += f"/{filename}"
        return url


class RunV2API(ResourceV2API, RunAPI):
    """V2 API resource for runs. Currently read-only until V2 server supports POST."""

    def get(
        self,
        run_id: int,  # noqa: ARG002
        *,
        reset_cache: bool = False,  # noqa: ARG002
    ) -> OpenMLRun:  # type: ignore[override]
        """Fetch a single run from the V2 server.

        Parameters
        ----------
        run_id : int
            The ID of the run to fetch.
        reset_cache : bool, default=False
            Whether to reset the cache.

        Returns
        -------
        OpenMLRun
            The run object.

        Raises
        ------
        OpenMLNotSupportedError
            V2 server API not yet available for this operation.
        """
        self._not_supported(method="get")

    def list(  # type: ignore[valid-type]  # noqa: PLR0913
        self,
        limit: int,  # noqa: ARG002
        offset: int,  # noqa: ARG002
        *,
        ids: builtins.list[int] | None = None,  # noqa: ARG002
        task: builtins.list[int] | None = None,  # noqa: ARG002
        setup: builtins.list[int] | None = None,  # noqa: ARG002
        flow: builtins.list[int] | None = None,  # noqa: ARG002
        uploader: builtins.list[int] | None = None,  # noqa: ARG002
        study: int | None = None,  # noqa: ARG002
        tag: str | None = None,  # noqa: ARG002
        display_errors: bool = False,  # noqa: ARG002
        task_type: TaskType | int | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """List runs from the V2 server.

        Raises
        ------
        OpenMLNotSupportedError
            V2 server API not yet available for this operation.
        """
        self._not_supported(method="list")

    def download_text_file(
        self,
        source: str,  # noqa: ARG002
        *,
        md5_checksum: str | None = None,  # noqa: ARG002
    ) -> str:
        self._not_supported(method="download_text_file")

    def file_id_to_url(
        self,
        file_id: int,  # noqa: ARG002
        filename: str | None = None,  # noqa: ARG002
    ) -> str:
        self._not_supported(method="file_id_to_url")
