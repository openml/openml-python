# License: BSD 3-Clause

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Any

import openml
from openml.base import OpenMLBase


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

    def __init__(
        self,
        study_id: Optional[int],
        alias: Optional[str],
        main_entity_type: str,
        benchmark_suite: Optional[int],
        name: str,
        description: str,
        status: Optional[str],
        creation_date: Optional[str],
        creator: Optional[int],
        tags: Optional[List[Dict]],
        data: Optional[List[int]],
        tasks: Optional[List[int]],
        flows: Optional[List[int]],
        runs: Optional[List[int]],
        setups: Optional[List[int]],
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
    def id(self) -> Optional[int]:
        return self.study_id

    def _get_repr_body_fields(self) -> List[Tuple[str, Union[str, int, List[str]]]]:
        """Collect all information to display in the __repr__ body."""
        fields = {
            "Name": self.name,
            "Status": self.status,
            "Main Entity Type": self.main_entity_type,
        }  # type: Dict[str, Any]
        if self.study_id is not None:
            fields["ID"] = self.study_id
            fields["Study URL"] = self.openml_url
        if self.creator is not None:
            fields["Creator"] = "{}/u/{}".format(openml.config.get_server_base_url(), self.creator)
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

    def _parse_publish_response(self, xml_response: Dict):
        """Parse the id from the xml_response and assign it to self."""
        self.study_id = int(xml_response["oml:study_upload"]["oml:id"])

    def _to_dict(self) -> "OrderedDict[str, OrderedDict]":
        """Creates a dictionary representation of self."""
        # some can not be uploaded, e.g., id, creator, creation_date
        simple_props = ["alias", "main_entity_type", "name", "description"]
        # maps from attribute name (which is used as outer tag name) to immer
        # tag name (e.g., self.tasks -> <oml:tasks><oml:task_id>1987
        # </oml:task_id></oml:tasks>)
        complex_props = {
            "tasks": "task_id",
            "runs": "run_id",
        }

        study_container = OrderedDict()  # type: 'OrderedDict'
        namespace_list = [("@xmlns:oml", "http://openml.org/openml")]
        study_dict = OrderedDict(namespace_list)  # type: 'OrderedDict'
        study_container["oml:study"] = study_dict

        for prop_name in simple_props:
            content = getattr(self, prop_name, None)
            if content is not None:
                study_dict["oml:" + prop_name] = content
        for prop_name, inner_name in complex_props.items():
            content = getattr(self, prop_name, None)
            if content is not None:
                sub_dict = {"oml:" + inner_name: content}
                study_dict["oml:" + prop_name] = sub_dict
        return study_container

    def push_tag(self, tag: str):
        raise NotImplementedError("Tags for studies is not (yet) supported.")

    def remove_tag(self, tag: str):
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

    def __init__(
        self,
        study_id: Optional[int],
        alias: Optional[str],
        benchmark_suite: Optional[int],
        name: str,
        description: str,
        status: Optional[str],
        creation_date: Optional[str],
        creator: Optional[int],
        tags: Optional[List[Dict]],
        data: Optional[List[int]],
        tasks: Optional[List[int]],
        flows: Optional[List[int]],
        runs: Optional[List[int]],
        setups: Optional[List[int]],
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

    def __init__(
        self,
        suite_id: Optional[int],
        alias: Optional[str],
        name: str,
        description: str,
        status: Optional[str],
        creation_date: Optional[str],
        creator: Optional[int],
        tags: Optional[List[Dict]],
        data: Optional[List[int]],
        tasks: List[int],
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
