from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xmltodict

import openml
from openml._api.resources.base import RunsAPI
from openml.tasks.task import TaskType

if TYPE_CHECKING:
    from openml.runs.run import OpenMLRun


class RunsV1(RunsAPI):
    def get(self, run_id: int) -> OpenMLRun:
        """Fetch a single run from the OpenML server.

        Parameters
        ----------
        run_id : int
            The ID of the run to fetch.

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
        response = self._http.get(path)
        xml_content = response.text
        return openml.runs.functions._create_run_from_xml(xml_content)

    def list(  # noqa: PLR0913, C901, PLR0912
        self,
        limit: int,
        offset: int,
        *,
        ids: list | None = None,
        task: list | None = None,
        setup: list | None = None,
        flow: list | None = None,
        uploader: list | None = None,
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
        id : list of int, optional
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

        xml_string = self._http.get(path).text
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

    def delete(self, run_id: int) -> bool:
        """Delete a run from the OpenML server.

        Parameters
        ----------
        run_id : int
            The ID of the run to delete.

        Returns
        -------
        bool
            True if deletion was successful, False otherwise.

        Raises
        ------
        openml.exceptions.OpenMLServerException
            If the run does not exist or user lacks permissions.

        Notes
        -----
        Only the uploader or server administrators can delete a run.
        """
        path = f"run/{run_id}"
        response = self._http.delete(path)
        # Parse XML response to check if deletion was successful
        xml_response = xmltodict.parse(response.text)
        return "oml:run_delete" in xml_response

    def create(self, run: OpenMLRun) -> OpenMLRun:
        """Create (publish) a run on the OpenML server.

        Parameters
        ----------
        run : OpenMLRun
            The run object to publish.

        Returns
        -------
        OpenMLRun
            The published run with run_id assigned.
        """
        # TODO: Implement V1 multipart upload
        # 1. Ensure flow is published
        # 2. Get file elements (description.xml, predictions.arff, trace.arff)
        # 3. POST multipart to /run/
        # 4. Parse XML response and set run_id
        raise NotImplementedError("RunsV1.create() is not implemented yet.")


class RunsV2(RunsAPI):
    """V2 API resource for runs. Currently read-only until V2 server supports POST."""

    def get(self, run_id: int) -> OpenMLRun:
        """Fetch a single run from the V2 server.

        Parameters
        ----------
        run_id : int
            The ID of the run to fetch.

        Returns
        -------
        OpenMLRun
            The run object.

        Raises
        ------
        NotImplementedError
            V2 server API not yet available for this operation.
        """
        raise NotImplementedError("RunsV2.get is not implemented yet.")

    def list(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        ids: list | None = None,
        task: list | None = None,
        setup: list | None = None,
        flow: list | None = None,
        uploader: list | None = None,
        study: int | None = None,
        tag: str | None = None,
        display_errors: bool = False,
        task_type: TaskType | int | None = None,
    ) -> pd.DataFrame:
        """List runs from the V2 server.

        Raises
        ------
        NotImplementedError
            V2 server API not yet available for this operation.
        """
        raise NotImplementedError("RunsV2.list is not implemented yet.")

    def delete(self, run_id: int) -> bool:
        """Delete a run from the V2 server.

        Parameters
        ----------
        run_id : int
            The ID of the run to delete.

        Returns
        -------
        bool
            True if deletion was successful, False otherwise.

        Raises
        ------
        NotImplementedError
            V2 server API not yet available for this operation.
        """
        raise NotImplementedError("RunsV2.delete is not implemented yet.")

    def create(self, run: OpenMLRun) -> OpenMLRun:
        """Create (publish) a run on the V2 server.

        Parameters
        ----------
        run : OpenMLRun
            The run object to publish.

        Returns
        -------
        OpenMLRun
            The published run with run_id assigned.

        Raises
        ------
        NotImplementedError
            V2 server does not yet support POST /runs/ endpoint.
            Expected availability: Q2 2025
        """
        raise NotImplementedError("RunsV2.create is not implemented yet.")
