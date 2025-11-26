# License: BSD 3-Clause
# TODO(eddiebergman): Begging for dataclassses to shorten this all
from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from openml.base import OpenMLBase
from openml.config import get_server_base_url
from openml.datasets.functions import list_datasets
from openml.exceptions import OpenMLServerException
from openml.tasks.functions import _list_tasks


class BaseStudy(OpenMLBase):
    """
    An OpenMLStudy represents the OpenML concept of a study. It contains
    the following information: name, id, description, creation date,
    creator id and a set of tags.

    According to this list of tags, the study object receives a list of
    OpenML object ids (datasets, flows, tasks and setups).

    Can be used to obtain all relevant information from a study at once.

    Parameters
    ----------
    study_id : int
        the study id
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    main_entity_type : str
        the entity type (e.g., task, run) that is core in this study.
        only entities of this type can be added explicitly
    benchmark_suite : int (optional)
        the benchmark suite (another study) upon which this study is ran.
        can only be active if main entity type is runs.
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    status : str
        Whether the study is in preparation, active or deactivated
    creation_date : str
        date of creation (meta-info)
    creator : int
        openml user id of the owner / creator
    tags : list(dict)
        The list of tags shows which tags are associated with the study.
        Each tag is a dict of (tag) name, window_start and write_access.
    data : list
        a list of data ids associated with this study
    tasks : list
        a list of task ids associated with this study
    flows : list
        a list of flow ids associated with this study
    runs : list
        a list of run ids associated with this study
    setups : list
        a list of setup ids associated with this study
    """

    def __init__(  # noqa: PLR0913
        self,
        study_id: int | None,
        alias: str | None,
        main_entity_type: str,
        benchmark_suite: int | None,
        name: str,
        description: str,
        status: str | None,
        creation_date: str | None,
        creator: int | None,
        tags: list[dict] | None,
        data: list[int] | None,
        tasks: list[int] | None,
        flows: list[int] | None,
        runs: list[int] | None,
        setups: list[int] | None,
    ):
        self.study_id = study_id
        self.alias = alias
        self.main_entity_type = main_entity_type
        self.benchmark_suite = benchmark_suite
        self.name = name
        self.description = description
        self.status = status
        self.creation_date = creation_date
        self.creator = creator
        self.tags = tags  # LEGACY. Can be removed soon
        self.data = data
        self.tasks = tasks
        self.flows = flows
        self.setups = setups
        self.runs = runs

    @classmethod
    def _entity_letter(cls) -> str:
        return "s"

    @property
    def id(self) -> int | None:
        """Return the id of the study."""
        return self.study_id

    def _get_repr_body_fields(self) -> Sequence[tuple[str, str | int | list[str]]]:
        """Collect all information to display in the __repr__ body."""
        fields: dict[str, Any] = {
            "Name": self.name,
            "Status": self.status,
            "Main Entity Type": self.main_entity_type,
        }
        if self.study_id is not None:
            fields["ID"] = self.study_id
            fields["Study URL"] = self.openml_url
        if self.creator is not None:
            fields["Creator"] = f"{get_server_base_url()}/u/{self.creator}"
        if self.creation_date is not None:
            fields["Upload Time"] = self.creation_date.replace("T", " ")
        if self.data is not None:
            fields["# of Data"] = len(self.data)
        if self.tasks is not None:
            fields["# of Tasks"] = len(self.tasks)
        if self.flows is not None:
            fields["# of Flows"] = len(self.flows)
        if self.runs is not None:
            fields["# of Runs"] = len(self.runs)

        # determines the order in which the information will be printed
        order = [
            "ID",
            "Name",
            "Status",
            "Main Entity Type",
            "Study URL",
            "# of Data",
            "# of Tasks",
            "# of Flows",
            "# of Runs",
            "Creator",
            "Upload Time",
        ]
        return [(key, fields[key]) for key in order if key in fields]

    def _parse_publish_response(self, xml_response: dict) -> None:
        """Parse the id from the xml_response and assign it to self."""
        self.study_id = int(xml_response["oml:study_upload"]["oml:id"])

    def _to_dict(self) -> dict[str, dict]:
        """Creates a dictionary representation of self."""
        # some can not be uploaded, e.g., id, creator, creation_date
        simple_props = ["alias", "main_entity_type", "name", "description"]

        # TODO(eddiebergman): Begging for a walrus if we can drop 3.7
        simple_prop_values = {}
        for prop_name in simple_props:
            content = getattr(self, prop_name, None)
            if content is not None:
                simple_prop_values["oml:" + prop_name] = content

        # maps from attribute name (which is used as outer tag name) to immer
        # tag name e.g., self.tasks -> <oml:tasks><oml:task_id>1987</oml:task_id></oml:tasks>
        complex_props = {"tasks": "task_id", "runs": "run_id"}

        # TODO(eddiebergman): Begging for a walrus if we can drop 3.7
        complex_prop_values = {}
        for prop_name, inner_name in complex_props.items():
            content = getattr(self, prop_name, None)
            if content is not None:
                complex_prop_values["oml:" + prop_name] = {"oml:" + inner_name: content}

        return {
            "oml:study": {
                "@xmlns:oml": "http://openml.org/openml",
                **simple_prop_values,
                **complex_prop_values,
            }
        }

    def push_tag(self, tag: str) -> None:
        """Add a tag to the study."""
        raise NotImplementedError("Tags for studies is not (yet) supported.")

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the study."""
        raise NotImplementedError("Tags for studies is not (yet) supported.")


class OpenMLStudy(BaseStudy):
    """
    An OpenMLStudy represents the OpenML concept of a study (a collection of runs).

    It contains the following information: name, id, description, creation date,
    creator id and a list of run ids.

    According to this list of run ids, the study object receives a list of
    OpenML object ids (datasets, flows, tasks and setups).

    Parameters
    ----------
    study_id : int
        the study id
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    benchmark_suite : int (optional)
        the benchmark suite (another study) upon which this study is ran.
        can only be active if main entity type is runs.
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    status : str
        Whether the study is in preparation, active or deactivated
    creation_date : str
        date of creation (meta-info)
    creator : int
        openml user id of the owner / creator
    tags : list(dict)
        The list of tags shows which tags are associated with the study.
        Each tag is a dict of (tag) name, window_start and write_access.
    data : list
        a list of data ids associated with this study
    tasks : list
        a list of task ids associated with this study
    flows : list
        a list of flow ids associated with this study
    runs : list
        a list of run ids associated with this study
    setups : list
        a list of setup ids associated with this study
    """

    def __init__(  # noqa: PLR0913
        self,
        study_id: int | None,
        alias: str | None,
        benchmark_suite: int | None,
        name: str,
        description: str,
        status: str | None,
        creation_date: str | None,
        creator: int | None,
        tags: list[dict] | None,
        data: list[int] | None,
        tasks: list[int] | None,
        flows: list[int] | None,
        runs: list[int] | None,
        setups: list[int] | None,
    ):
        super().__init__(
            study_id=study_id,
            alias=alias,
            main_entity_type="run",
            benchmark_suite=benchmark_suite,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=data,
            tasks=tasks,
            flows=flows,
            runs=runs,
            setups=setups,
        )


class OpenMLBenchmarkSuite(BaseStudy):
    """
    An OpenMLBenchmarkSuite represents the OpenML concept of a suite (a collection of tasks).

    It contains the following information: name, id, description, creation date,
    creator id and the task ids.

    According to this list of task ids, the suite object receives a list of
    OpenML object ids (datasets).

    Parameters
    ----------
    suite_id : int
        the study id
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    main_entity_type : str
        the entity type (e.g., task, run) that is core in this study.
        only entities of this type can be added explicitly
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    status : str
        Whether the study is in preparation, active or deactivated
    creation_date : str
        date of creation (meta-info)
    creator : int
        openml user id of the owner / creator
    tags : list(dict)
        The list of tags shows which tags are associated with the study.
        Each tag is a dict of (tag) name, window_start and write_access.
    data : list
        a list of data ids associated with this study
    tasks : list
        a list of task ids associated with this study
    """

    def __init__(  # noqa: PLR0913
        self,
        suite_id: int | None,
        alias: str | None,
        name: str,
        description: str,
        status: str | None,
        creation_date: str | None,
        creator: int | None,
        tags: list[dict] | None,
        data: list[int] | None,
        tasks: list[int] | None,
    ):
        super().__init__(
            study_id=suite_id,
            alias=alias,
            main_entity_type="task",
            benchmark_suite=None,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=data,
            tasks=tasks,
            flows=None,
            runs=None,
            setups=None,
        )
        # Initialize metadata cache
        self._metadata: pd.DataFrame | None = None

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing metadata for all tasks in the suite.

        The DataFrame includes:
        - Task-level information: task ID (tid), task type, estimation procedure,
          target feature, evaluation measure
        - Dataset-level information: dataset ID (did), dataset name, version,
          uploader, number of instances, number of features, number of classes,
          and other dataset qualities

        The result is cached after the first access. Subsequent calls return the
        cached DataFrame without making additional API calls.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per task in the suite. The DataFrame is indexed
            by the default integer index. Columns include both task and dataset metadata.

        Raises
        ------
        RuntimeError
            If task metadata cannot be retrieved from the OpenML server.

        Examples
        --------
        >>> import openml
        >>> suite = openml.study.get_suite(99)  # OpenML-CC18
        >>> meta = suite.metadata
        >>> print(meta.columns.tolist()[:5])  # First 5 columns
        ['tid', 'did', 'name', 'task_type', 'status']

        >>> # Export to LaTeX
        >>> columns = ['name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']
        >>> latex_table = meta[columns].style.to_latex(
        ...     caption="Dataset Characteristics",
        ...     label="tab:suite_metadata"
        ... )
        """
        # Return cached result if available
        if self._metadata is not None:
            return self._metadata

        # Handle empty suites gracefully
        if not self.tasks:
            self._metadata = pd.DataFrame()
            return self._metadata

        # Step 1: Fetch Task Metadata
        # Use internal _list_tasks because public API doesn't support task_id filtering
        try:
            task_df = _list_tasks(
                limit=max(len(self.tasks), 1000),
                offset=0,
                task_id=self.tasks,
            )

            # _list_tasks returns DataFrame with 'tid' as index (from orient="index")
            # Reset index to make 'tid' a column for easier merging
            if task_df.index.name == "tid":
                task_df = task_df.reset_index()

            # Verify we got the expected tasks
            if len(task_df) == 0:
                # No tasks found - return empty DataFrame
                self._metadata = pd.DataFrame()
                return self._metadata

            # Ensure 'tid' column exists (should after reset_index if index was named 'tid')
            if "tid" not in task_df.columns:
                # This shouldn't happen, but handle gracefully
                raise RuntimeError(
                    f"Task metadata missing 'tid' column. Columns: {task_df.columns.tolist()}"
                )

        except OpenMLServerException as e:
            raise RuntimeError(f"Failed to retrieve task metadata for suite {self.id}: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error retrieving task metadata for suite {self.id}: {e}"
            ) from e

        # Step 2: Extract unique dataset IDs and fetch dataset metadata
        if "did" in task_df.columns and len(task_df) > 0:
            unique_dids = task_df["did"].unique().tolist()

            try:
                dataset_df = list_datasets(data_id=unique_dids)
            except OpenMLServerException as e:
                raise RuntimeError(f"Failed to retrieve dataset metadata: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error retrieving dataset metadata: {e}") from e

            # Step 3: Merge DataFrames
            # Use left join to preserve all tasks (one row per task)
            # Apply suffixes to handle column name collisions
            self._metadata = pd.merge(
                task_df,
                dataset_df,
                on="did",
                how="left",
                suffixes=("", "_dataset"),
            )
        else:
            # Fallback: return task DataFrame only if 'did' column is missing
            self._metadata = task_df

        return self._metadata
