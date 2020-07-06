# License: BSD 3-Clause

from typing import cast, Dict, List, Optional, Union
import warnings

import dateutil.parser
import xmltodict
import pandas as pd

from openml.study import OpenMLStudy, OpenMLBenchmarkSuite
from openml.study.study import BaseStudy
import openml._api_calls


def get_suite(suite_id: Union[int, str]) -> OpenMLBenchmarkSuite:
    """
    Retrieves all relevant information of an OpenML benchmarking suite from the server.

    Parameters
    ----------
    study id : int, str
        study id (numeric or alias)

    Returns
    -------
    OpenMLSuite
        The OpenML suite object
    """
    suite = cast(OpenMLBenchmarkSuite, _get_study(suite_id, entity_type="task"))
    return suite


def get_study(
    study_id: Union[int, str], arg_for_backwards_compat: Optional[str] = None,
) -> OpenMLStudy:  # noqa F401
    """
    Retrieves all relevant information of an OpenML study from the server.

    Parameters
    ----------
    study id : int, str
        study id (numeric or alias)

    arg_for_backwards_compat : str, optional
        The example given in https://arxiv.org/pdf/1708.03731.pdf uses an older version of the
        API which required specifying the type of study, i.e. tasks. We changed the
        implementation of studies since then and split them up into suites (collections of tasks)
        and studies (collections of runs) so this argument is no longer needed.

    Returns
    -------
    OpenMLStudy
        The OpenML study object
    """
    if study_id == "OpenML100":
        message = (
            "It looks like you are running code from the OpenML100 paper. It still works, but lots "
            "of things have changed since then. Please use `get_suite('OpenML100')` instead."
        )
        warnings.warn(message, DeprecationWarning)
        openml.config.logger.warn(message)
        study = _get_study(study_id, entity_type="task")
        return cast(OpenMLBenchmarkSuite, study)  # type: ignore
    else:
        study = cast(OpenMLStudy, _get_study(study_id, entity_type="run"))
        return study


def _get_study(id_: Union[int, str], entity_type) -> BaseStudy:
    call_suffix = "study/{}".format(str(id_))
    xml_string = openml._api_calls._perform_api_call(call_suffix, "get")
    force_list_tags = (
        "oml:data_id",
        "oml:flow_id",
        "oml:task_id",
        "oml:setup_id",
        "oml:run_id",
        "oml:tag",  # legacy.
    )
    result_dict = xmltodict.parse(xml_string, force_list=force_list_tags)["oml:study"]
    study_id = int(result_dict["oml:id"])
    alias = result_dict["oml:alias"] if "oml:alias" in result_dict else None
    main_entity_type = result_dict["oml:main_entity_type"]
    if entity_type != main_entity_type:
        raise ValueError(
            "Unexpected entity type '{}' reported by the server, expected '{}'".format(
                main_entity_type, entity_type,
            )
        )
    benchmark_suite = (
        result_dict["oml:benchmark_suite"] if "oml:benchmark_suite" in result_dict else None
    )
    name = result_dict["oml:name"]
    description = result_dict["oml:description"]
    status = result_dict["oml:status"]
    creation_date = result_dict["oml:creation_date"]
    creation_date_as_date = dateutil.parser.parse(creation_date)
    creator = result_dict["oml:creator"]

    # tags is legacy. remove once no longer needed.
    tags = []
    if "oml:tag" in result_dict:
        for tag in result_dict["oml:tag"]:
            current_tag = {"name": tag["oml:name"], "write_access": tag["oml:write_access"]}
            if "oml:window_start" in tag:
                current_tag["window_start"] = tag["oml:window_start"]
            tags.append(current_tag)

    if "oml:data" in result_dict:
        datasets = [int(x) for x in result_dict["oml:data"]["oml:data_id"]]
    else:
        raise ValueError("No datasets attached to study {}!".format(id_))
    if "oml:tasks" in result_dict:
        tasks = [int(x) for x in result_dict["oml:tasks"]["oml:task_id"]]
    else:
        raise ValueError("No tasks attached to study {}!".format(id_))

    if main_entity_type in ["runs", "run"]:

        if "oml:flows" in result_dict:
            flows = [int(x) for x in result_dict["oml:flows"]["oml:flow_id"]]
        else:
            raise ValueError("No flows attached to study {}!".format(id_))
        if "oml:setups" in result_dict:
            setups = [int(x) for x in result_dict["oml:setups"]["oml:setup_id"]]
        else:
            raise ValueError("No setups attached to study {}!".format(id_))
        if "oml:runs" in result_dict:
            runs = [
                int(x) for x in result_dict["oml:runs"]["oml:run_id"]
            ]  # type: Optional[List[int]]
        else:
            if creation_date_as_date < dateutil.parser.parse("2019-01-01"):
                # Legacy studies did not require runs
                runs = None
            else:
                raise ValueError("No runs attached to study {}!".format(id_))

        study = OpenMLStudy(
            study_id=study_id,
            alias=alias,
            benchmark_suite=benchmark_suite,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=datasets,
            tasks=tasks,
            flows=flows,
            setups=setups,
            runs=runs,
        )  # type: BaseStudy

    elif main_entity_type in ["tasks", "task"]:

        study = OpenMLBenchmarkSuite(
            suite_id=study_id,
            alias=alias,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=datasets,
            tasks=tasks,
        )

    else:
        raise ValueError("Unknown entity type {}".format(main_entity_type))

    return study


def create_study(
    name: str,
    description: str,
    run_ids: List[int],
    alias: Optional[str],
    benchmark_suite: Optional[int],
) -> OpenMLStudy:
    """
    Creates an OpenML study (collection of data, tasks, flows, setups and run),
    where the runs are the main entity (collection consists of runs and all
    entities (flows, tasks, etc) that are related to these runs)

    Parameters
    ----------
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    benchmark_suite : int (optional)
        the benchmark suite (another study) upon which this study is ran.
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    run_ids : list
        a list of run ids associated with this study

    Returns
    -------
    OpenMLStudy
        A local OpenML study object (call publish method to upload to server)
    """
    return OpenMLStudy(
        study_id=None,
        alias=alias,
        benchmark_suite=benchmark_suite,
        name=name,
        description=description,
        status=None,
        creation_date=None,
        creator=None,
        tags=None,
        data=None,
        tasks=None,
        flows=None,
        runs=run_ids,
        setups=None,
    )


def create_benchmark_suite(
    name: str, description: str, task_ids: List[int], alias: Optional[str],
) -> OpenMLBenchmarkSuite:
    """
    Creates an OpenML benchmark suite (collection of entity types, where
    the tasks are the linked entity)

    Parameters
    ----------
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    task_ids : list
        a list of task ids associated with this study

    Returns
    -------
    OpenMLStudy
        A local OpenML study object (call publish method to upload to server)
    """
    return OpenMLBenchmarkSuite(
        suite_id=None,
        alias=alias,
        name=name,
        description=description,
        status=None,
        creation_date=None,
        creator=None,
        tags=None,
        data=None,
        tasks=task_ids,
    )


def update_suite_status(suite_id: int, status: str) -> None:
    """
    Updates the status of a study to either 'active' or 'deactivated'.

    Parameters
    ----------
    suite_id : int
        The data id of the dataset
    status : str,
        'active' or 'deactivated'
    """
    return update_study_status(suite_id, status)


def update_study_status(study_id: int, status: str) -> None:
    """
    Updates the status of a study to either 'active' or 'deactivated'.

    Parameters
    ----------
    study_id : int
        The data id of the dataset
    status : str,
        'active' or 'deactivated'
    """
    legal_status = {"active", "deactivated"}
    if status not in legal_status:
        raise ValueError("Illegal status value. " "Legal values: %s" % legal_status)
    data = {"study_id": study_id, "status": status}
    result_xml = openml._api_calls._perform_api_call("study/status/update", "post", data=data)
    result = xmltodict.parse(result_xml)
    server_study_id = result["oml:study_status_update"]["oml:id"]
    server_status = result["oml:study_status_update"]["oml:status"]
    if status != server_status or int(study_id) != int(server_study_id):
        # This should never happen
        raise ValueError("Study id/status does not collide")


def delete_suite(suite_id: int) -> bool:
    """Deletes a study from the OpenML server.

    Parameters
    ----------
    suite_id : int
        OpenML id of the study

    Returns
    -------
    bool
        True iff the deletion was successful. False otherwise
    """
    return delete_study(suite_id)


def delete_study(study_id: int) -> bool:
    """Deletes a study from the OpenML server.

    Parameters
    ----------
    study_id : int
        OpenML id of the study

    Returns
    -------
    bool
        True iff the deletion was successful. False otherwise
    """
    return openml.utils._delete_entity("study", study_id)


def attach_to_suite(suite_id: int, task_ids: List[int]) -> int:
    """Attaches a set of tasks to a benchmarking suite.

    Parameters
    ----------
    suite_id : int
        OpenML id of the study

    task_ids : list (int)
        List of entities to link to the collection

    Returns
    -------
    int
        new size of the suite (in terms of explicitly linked entities)
    """
    return attach_to_study(suite_id, task_ids)


def attach_to_study(study_id: int, run_ids: List[int]) -> int:
    """Attaches a set of runs to a study.

    Parameters
    ----------
    study_id : int
        OpenML id of the study

    run_ids : list (int)
        List of entities to link to the collection

    Returns
    -------
    int
        new size of the study (in terms of explicitly linked entities)
    """

    # Interestingly, there's no need to tell the server about the entity type, it knows by itself
    uri = "study/%d/attach" % study_id
    post_variables = {"ids": ",".join(str(x) for x in run_ids)}
    result_xml = openml._api_calls._perform_api_call(uri, "post", post_variables)
    result = xmltodict.parse(result_xml)["oml:study_attach"]
    return int(result["oml:linked_entities"])


def detach_from_suite(suite_id: int, task_ids: List[int]) -> int:
    """Detaches a set of task ids from a suite.

    Parameters
    ----------
    suite_id : int
        OpenML id of the study

    task_ids : list (int)
        List of entities to unlink from the collection

    Returns
    -------
    int
        new size of the study (in terms of explicitly linked entities)"""
    return detach_from_study(suite_id, task_ids)


def detach_from_study(study_id: int, run_ids: List[int]) -> int:
    """Detaches a set of run ids from a study.

    Parameters
    ----------
    study_id : int
        OpenML id of the study

    run_ids : list (int)
        List of entities to unlink from the collection

    Returns
    -------
    int
        new size of the study (in terms of explicitly linked entities)
    """

    # Interestingly, there's no need to tell the server about the entity type, it knows by itself
    uri = "study/%d/detach" % study_id
    post_variables = {"ids": ",".join(str(x) for x in run_ids)}
    result_xml = openml._api_calls._perform_api_call(uri, "post", post_variables)
    result = xmltodict.parse(result_xml)["oml:study_detach"]
    return int(result["oml:linked_entities"])


def list_suites(
    offset: Optional[int] = None,
    size: Optional[int] = None,
    status: Optional[str] = None,
    uploader: Optional[List[int]] = None,
    output_format: str = "dict",
) -> Union[Dict, pd.DataFrame]:
    """
    Return a list of all suites which are on OpenML.

    Parameters
    ----------
    offset : int, optional
        The number of suites to skip, starting from the first.
    size : int, optional
        The maximum number of suites to show.
    status : str, optional
        Should be {active, in_preparation, deactivated, all}. By default active
        suites are returned.
    uploader : list (int), optional
        Result filter. Will only return suites created by these users.
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    datasets : dict of dicts, or dataframe
        - If output_format='dict'
            Every suite is represented by a dictionary containing the following information:
            - id
            - alias (optional)
            - name
            - main_entity_type
            - status
            - creator
            - creation_date

        - If output_format='dataframe'
            Every row is represented by a dictionary containing the following information:
            - id
            - alias (optional)
            - name
            - main_entity_type
            - status
            - creator
            - creation_date
    """
    if output_format not in ["dataframe", "dict"]:
        raise ValueError(
            "Invalid output format selected. " "Only 'dict' or 'dataframe' applicable."
        )

    return openml.utils._list_all(
        output_format=output_format,
        listing_call=_list_studies,
        offset=offset,
        size=size,
        main_entity_type="task",
        status=status,
        uploader=uploader,
    )


def list_studies(
    offset: Optional[int] = None,
    size: Optional[int] = None,
    status: Optional[str] = None,
    uploader: Optional[List[str]] = None,
    benchmark_suite: Optional[int] = None,
    output_format: str = "dict",
) -> Union[Dict, pd.DataFrame]:
    """
    Return a list of all studies which are on OpenML.

    Parameters
    ----------
    offset : int, optional
        The number of studies to skip, starting from the first.
    size : int, optional
        The maximum number of studies to show.
    status : str, optional
        Should be {active, in_preparation, deactivated, all}. By default active
        studies are returned.
    uploader : list (int), optional
        Result filter. Will only return studies created by these users.
    benchmark_suite : int, optional
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    datasets : dict of dicts, or dataframe
        - If output_format='dict'
            Every dataset is represented by a dictionary containing
            the following information:
            - id
            - alias (optional)
            - name
            - benchmark_suite (optional)
            - status
            - creator
            - creation_date
            If qualities are calculated for the dataset, some of
            these are also returned.

        - If output_format='dataframe'
            Every dataset is represented by a dictionary containing
            the following information:
            - id
            - alias (optional)
            - name
            - benchmark_suite (optional)
            - status
            - creator
            - creation_date
            If qualities are calculated for the dataset, some of
            these are also returned.
    """
    if output_format not in ["dataframe", "dict"]:
        raise ValueError(
            "Invalid output format selected. " "Only 'dict' or 'dataframe' applicable."
        )

    return openml.utils._list_all(
        output_format=output_format,
        listing_call=_list_studies,
        offset=offset,
        size=size,
        main_entity_type="run",
        status=status,
        uploader=uploader,
        benchmark_suite=benchmark_suite,
    )


def _list_studies(output_format="dict", **kwargs) -> Union[Dict, pd.DataFrame]:
    """
    Perform api call to return a list of studies.

    Parameters
    ----------
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    kwargs : dict, optional
        Legal filter operators (keys in the dict):
        status, limit, offset, main_entity_type, uploader

    Returns
    -------
    studies : dict of dicts
    """
    api_call = "study/list"
    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)
    return __list_studies(api_call=api_call, output_format=output_format)


def __list_studies(api_call, output_format="object") -> Union[Dict, pd.DataFrame]:
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    study_dict = xmltodict.parse(xml_string, force_list=("oml:study",))

    # Minimalistic check if the XML is useful
    assert type(study_dict["oml:study_list"]["oml:study"]) == list, type(
        study_dict["oml:study_list"]
    )
    assert study_dict["oml:study_list"]["@xmlns:oml"] == "http://openml.org/openml", study_dict[
        "oml:study_list"
    ]["@xmlns:oml"]

    studies = dict()
    for study_ in study_dict["oml:study_list"]["oml:study"]:
        # maps from xml name to a tuple of (dict name, casting fn)
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
        current_study = dict()
        for oml_field_name, (real_field_name, cast_fn) in expected_fields.items():
            if oml_field_name in study_:
                current_study[real_field_name] = cast_fn(study_[oml_field_name])
        current_study["id"] = int(current_study["id"])
        studies[study_id] = current_study

    if output_format == "dataframe":
        studies = pd.DataFrame.from_dict(studies, orient="index")
    return studies
