# License: BSD 3-Clause
# ruff: noqa: PLR0913
from __future__ import annotations

import json
from functools import partial
from itertools import chain
from typing import Any, Literal
from typing_extensions import overload

import numpy as np
import pandas as pd
import xmltodict

import openml
import openml._api_calls
import openml.utils
from openml.evaluations import OpenMLEvaluation


@overload
def list_evaluations(
    function: str,
    offset: int | None = None,
    size: int | None = None,
    tasks: list[str | int] | None = None,
    setups: list[str | int] | None = None,
    flows: list[str | int] | None = None,
    runs: list[str | int] | None = None,
    uploaders: list[str | int] | None = None,
    tag: str | None = None,
    study: int | None = None,
    per_fold: bool | None = None,
    sort_order: str | None = None,
    output_format: Literal["dataframe"] = ...,
) -> pd.DataFrame: ...


@overload
def list_evaluations(
    function: str,
    offset: int | None = None,
    size: int | None = None,
    tasks: list[str | int] | None = None,
    setups: list[str | int] | None = None,
    flows: list[str | int] | None = None,
    runs: list[str | int] | None = None,
    uploaders: list[str | int] | None = None,
    tag: str | None = None,
    study: int | None = None,
    per_fold: bool | None = None,
    sort_order: str | None = None,
    output_format: Literal["object"] = "object",
) -> dict[int, OpenMLEvaluation]: ...


def list_evaluations(
    function: str,
    offset: int | None = None,
    size: int | None = None,
    tasks: list[str | int] | None = None,
    setups: list[str | int] | None = None,
    flows: list[str | int] | None = None,
    runs: list[str | int] | None = None,
    uploaders: list[str | int] | None = None,
    tag: str | None = None,
    study: int | None = None,
    per_fold: bool | None = None,
    sort_order: str | None = None,
    output_format: Literal["object", "dataframe"] = "object",
) -> dict[int, OpenMLEvaluation] | pd.DataFrame:
    """List all run-evaluation pairs matching all of the given filters.

    (Supports large amount of results)

    Parameters
    ----------
    function : str
        the evaluation function. e.g., predictive_accuracy
    offset : int, optional
        the number of runs to skip, starting from the first
    size : int, default 10000
        The maximum number of runs to show.
        If set to ``None``, it returns all the results.

    tasks : list[int,str], optional
        the list of task IDs
    setups: list[int,str], optional
        the list of setup IDs
    flows : list[int,str], optional
        the list of flow IDs
    runs :list[int,str], optional
        the list of run IDs
    uploaders : list[int,str], optional
        the list of uploader IDs
    tag : str, optional
        filter evaluation based on given tag

    study : int, optional

    per_fold : bool, optional

    sort_order : str, optional
       order of sorting evaluations, ascending ("asc") or descending ("desc")

    output_format: str, optional (default='object')
        The parameter decides the format of the output.
        - If 'object' the output is a dict of OpenMLEvaluation objects
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict or dataframe
    """
    if output_format not in ("dataframe", "object"):
        raise ValueError("Invalid output format. Only 'object', 'dataframe'.")

    per_fold_str = None
    if per_fold is not None:
        per_fold_str = str(per_fold).lower()

    listing_call = partial(
        _list_evaluations,
        function=function,
        tasks=tasks,
        setups=setups,
        flows=flows,
        runs=runs,
        uploaders=uploaders,
        tag=tag,
        study=study,
        sort_order=sort_order,
        per_fold=per_fold_str,
    )
    eval_collection = openml.utils._list_all(listing_call, offset=offset, limit=size)

    flattened = list(chain.from_iterable(eval_collection))
    if output_format == "dataframe":
        records = [item._to_dict() for item in flattened]
        return pd.DataFrame.from_records(records)  # No index...

    return {e.run_id: e for e in flattened}


def _list_evaluations(  # noqa: C901
    limit: int,
    offset: int,
    *,
    function: str,
    tasks: list | None = None,
    setups: list | None = None,
    flows: list | None = None,
    runs: list | None = None,
    uploaders: list | None = None,
    study: int | None = None,
    sort_order: str | None = None,
    **kwargs: Any,
) -> list[OpenMLEvaluation]:
    """
    Perform API call ``/evaluation/function{function}/{filters}``

    Parameters
    ----------
    The arguments that are lists are separated from the single value
    ones which are put into the kwargs.

    limit : int
        the number of evaluations to return
    offset : int
        the number of evaluations to skip, starting from the first
    function : str
        the evaluation function. e.g., predictive_accuracy

    tasks : list[int,str], optional
        the list of task IDs
    setups: list[int,str], optional
        the list of setup IDs
    flows : list[int,str], optional
        the list of flow IDs
    runs :list[int,str], optional
        the list of run IDs
    uploaders : list[int,str], optional
        the list of uploader IDs

    study : int, optional

    kwargs: dict, optional
        Legal filter operators: tag, per_fold

    sort_order : str, optional
        order of sorting evaluations, ascending ("asc") or descending ("desc")

    Returns
    -------
    list of OpenMLEvaluation objects
    """
    api_call = f"evaluation/list/function/{function}"
    if limit is not None:
        api_call += f"/limit/{limit}"
    if offset is not None:
        api_call += f"/offset/{offset}"
    if kwargs is not None:
        for operator, value in kwargs.items():
            if value is not None:
                api_call += f"/{operator}/{value}"
    if tasks is not None:
        api_call += f"/task/{','.join([str(int(i)) for i in tasks])}"
    if setups is not None:
        api_call += f"/setup/{','.join([str(int(i)) for i in setups])}"
    if flows is not None:
        api_call += f"/flow/{','.join([str(int(i)) for i in flows])}"
    if runs is not None:
        api_call += f"/run/{','.join([str(int(i)) for i in runs])}"
    if uploaders is not None:
        api_call += f"/uploader/{','.join([str(int(i)) for i in uploaders])}"
    if study is not None:
        api_call += f"/study/{study}"
    if sort_order is not None:
        api_call += f"/sort_order/{sort_order}"

    return __list_evaluations(api_call)


def __list_evaluations(api_call: str) -> list[OpenMLEvaluation]:
    """Helper function to parse API calls which are lists of runs"""
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    evals_dict = xmltodict.parse(xml_string, force_list=("oml:evaluation",))
    # Minimalistic check if the XML is useful
    if "oml:evaluations" not in evals_dict:
        raise ValueError(
            f'Error in return XML, does not contain "oml:evaluations": {evals_dict!s}',
        )

    assert isinstance(evals_dict["oml:evaluations"]["oml:evaluation"], list), (
        "Expected 'oml:evaluation' to be a list, but got"
        f"{type(evals_dict['oml:evaluations']['oml:evaluation']).__name__}. "
    )

    uploader_ids = list(
        {eval_["oml:uploader"] for eval_ in evals_dict["oml:evaluations"]["oml:evaluation"]},
    )
    api_users = "user/list/user_id/" + ",".join(uploader_ids)
    xml_string_user = openml._api_calls._perform_api_call(api_users, "get")

    users = xmltodict.parse(xml_string_user, force_list=("oml:user",))
    user_dict = {user["oml:id"]: user["oml:username"] for user in users["oml:users"]["oml:user"]}

    evals = []
    for eval_ in evals_dict["oml:evaluations"]["oml:evaluation"]:
        run_id = int(eval_["oml:run_id"])
        value = float(eval_["oml:value"]) if "oml:value" in eval_ else None
        values = json.loads(eval_["oml:values"]) if eval_.get("oml:values", None) else None
        array_data = eval_.get("oml:array_data")

        evals.append(
            OpenMLEvaluation(
                run_id=run_id,
                task_id=int(eval_["oml:task_id"]),
                setup_id=int(eval_["oml:setup_id"]),
                flow_id=int(eval_["oml:flow_id"]),
                flow_name=eval_["oml:flow_name"],
                data_id=int(eval_["oml:data_id"]),
                data_name=eval_["oml:data_name"],
                function=eval_["oml:function"],
                upload_time=eval_["oml:upload_time"],
                uploader=int(eval_["oml:uploader"]),
                uploader_name=user_dict[eval_["oml:uploader"]],
                value=value,
                values=values,
                array_data=array_data,
            )
        )

    return evals


def list_evaluation_measures() -> list[str]:
    """Return list of evaluation measures available.

    The function performs an API call to retrieve the entire list of
    evaluation measures that are available.

    Returns
    -------
    list

    """
    api_call = "evaluationmeasure/list"
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    qualities = xmltodict.parse(xml_string, force_list=("oml:measures"))
    # Minimalistic check if the XML is useful
    if "oml:evaluation_measures" not in qualities:
        raise ValueError('Error in return XML, does not contain "oml:evaluation_measures"')

    if not isinstance(qualities["oml:evaluation_measures"]["oml:measures"][0]["oml:measure"], list):
        raise TypeError('Error in return XML, does not contain "oml:measure" as a list')

    return qualities["oml:evaluation_measures"]["oml:measures"][0]["oml:measure"]


def list_estimation_procedures() -> list[str]:
    """Return list of evaluation procedures available.

    The function performs an API call to retrieve the entire list of
    evaluation procedures' names that are available.

    Returns
    -------
    list
    """
    api_call = "estimationprocedure/list"
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    api_results = xmltodict.parse(xml_string)

    # Minimalistic check if the XML is useful
    if "oml:estimationprocedures" not in api_results:
        raise ValueError('Error in return XML, does not contain "oml:estimationprocedures"')

    if "oml:estimationprocedure" not in api_results["oml:estimationprocedures"]:
        raise ValueError('Error in return XML, does not contain "oml:estimationprocedure"')

    if not isinstance(api_results["oml:estimationprocedures"]["oml:estimationprocedure"], list):
        raise TypeError('Error in return XML, does not contain "oml:estimationprocedure" as a list')

    return [
        prod["oml:name"]
        for prod in api_results["oml:estimationprocedures"]["oml:estimationprocedure"]
    ]


def list_evaluations_setups(
    function: str,
    offset: int | None = None,
    size: int | None = None,
    tasks: list | None = None,
    setups: list | None = None,
    flows: list | None = None,
    runs: list | None = None,
    uploaders: list | None = None,
    tag: str | None = None,
    per_fold: bool | None = None,
    sort_order: str | None = None,
    parameters_in_separate_columns: bool = False,  # noqa: FBT002
) -> pd.DataFrame:
    """List all run-evaluation pairs matching all of the given filters
    and their hyperparameter settings.

    Parameters
    ----------
    function : str
        the evaluation function. e.g., predictive_accuracy
    offset : int, optional
        the number of runs to skip, starting from the first
    size : int, optional
        the maximum number of runs to show
    tasks : list[int], optional
        the list of task IDs
    setups: list[int], optional
        the list of setup IDs
    flows : list[int], optional
        the list of flow IDs
    runs : list[int], optional
        the list of run IDs
    uploaders : list[int], optional
        the list of uploader IDs
    tag : str, optional
        filter evaluation based on given tag
    per_fold : bool, optional
    sort_order : str, optional
       order of sorting evaluations, ascending ("asc") or descending ("desc")
    parameters_in_separate_columns: bool, optional (default= False)
        Returns hyperparameters in separate columns if set to True.
        Valid only for a single flow

    Returns
    -------
    dataframe with hyperparameter settings as a list of tuples.
    """
    if parameters_in_separate_columns and (flows is None or len(flows) != 1):
        raise ValueError("Can set parameters_in_separate_columns to true only for single flow_id")

    # List evaluations
    evals = list_evaluations(
        function=function,
        offset=offset,
        size=size,
        runs=runs,
        tasks=tasks,
        setups=setups,
        flows=flows,
        uploaders=uploaders,
        tag=tag,
        per_fold=per_fold,
        sort_order=sort_order,
        output_format="dataframe",
    )
    # List setups
    # list_setups by setup id does not support large sizes (exceeds URL length limit)
    # Hence we split the list of unique setup ids returned by list_evaluations into chunks of size N
    _df = pd.DataFrame()
    if len(evals) != 0:
        N = 100  # size of section
        uniq = np.asarray(evals["setup_id"].unique())
        length = len(uniq)

        # array_split - allows indices_or_sections to not equally divide the array
        # array_split -length % N sub-arrays of size length//N + 1 and the rest of size length//N.
        split_size = ((length - 1) // N) + 1
        setup_chunks = np.array_split(uniq, split_size)

        setup_data = pd.DataFrame()
        for _setups in setup_chunks:
            result = openml.setups.list_setups(setup=_setups, output_format="dataframe")
            assert isinstance(result, pd.DataFrame)
            result = result.drop("flow_id", axis=1)
            # concat resulting setup chunks into single datframe
            setup_data = pd.concat([setup_data, result])

        parameters = []
        # Convert parameters of setup into dict of (hyperparameter, value)
        for parameter_dict in setup_data["parameters"]:
            if parameter_dict is not None:
                parameters.append(
                    {param["full_name"]: param["value"] for param in parameter_dict.values()},
                )
            else:
                parameters.append({})
        setup_data["parameters"] = parameters
        # Merge setups with evaluations
        _df = evals.merge(setup_data, on="setup_id", how="left")

    if parameters_in_separate_columns:
        _df = pd.concat(
            [_df.drop("parameters", axis=1), _df["parameters"].apply(pd.Series)],
            axis=1,
        )

    return _df
