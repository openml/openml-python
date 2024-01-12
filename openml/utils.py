# License: BSD 3-Clause
from __future__ import annotations

import contextlib
import shutil
import warnings
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeVar, overload
from typing_extensions import Literal, ParamSpec

import numpy as np
import pandas as pd
import xmltodict

import openml
import openml._api_calls
import openml.exceptions

from . import config

# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from openml.base import OpenMLBase

    P = ParamSpec("P")
    R = TypeVar("R")


@overload
def extract_xml_tags(
    xml_tag_name: str,
    node: Mapping[str, Any],
    *,
    allow_none: Literal[True] = ...,
) -> Any | None:
    ...


@overload
def extract_xml_tags(
    xml_tag_name: str,
    node: Mapping[str, Any],
    *,
    allow_none: Literal[False],
) -> Any:
    ...


def extract_xml_tags(
    xml_tag_name: str,
    node: Mapping[str, Any],
    *,
    allow_none: bool = True,
) -> Any | None:
    """Helper to extract xml tags from xmltodict.

    Parameters
    ----------
    xml_tag_name : str
        Name of the xml tag to extract from the node.

    node : Mapping[str, Any]
        Node object returned by ``xmltodict`` from which ``xml_tag_name``
        should be extracted.

    allow_none : bool
        If ``False``, the tag needs to exist in the node. Will raise a
        ``ValueError`` if it does not.

    Returns
    -------
    object
    """
    if xml_tag_name in node and node[xml_tag_name] is not None:
        if isinstance(node[xml_tag_name], (dict, str)):
            return [node[xml_tag_name]]
        if isinstance(node[xml_tag_name], list):
            return node[xml_tag_name]

        raise ValueError("Received not string and non list as tag item")

    if allow_none:
        return None

    raise ValueError(f"Could not find tag '{xml_tag_name}' in node '{node!s}'")


def _get_rest_api_type_alias(oml_object: OpenMLBase) -> str:
    """Return the alias of the openml entity as it is defined for the REST API."""
    rest_api_mapping: list[tuple[type | tuple, str]] = [
        (openml.datasets.OpenMLDataset, "data"),
        (openml.flows.OpenMLFlow, "flow"),
        (openml.tasks.OpenMLTask, "task"),
        (openml.runs.OpenMLRun, "run"),
        ((openml.study.OpenMLStudy, openml.study.OpenMLBenchmarkSuite), "study"),
    ]
    _, api_type_alias = next(
        (python_type, api_alias)
        for (python_type, api_alias) in rest_api_mapping
        if isinstance(oml_object, python_type)
    )
    return api_type_alias


def _tag_openml_base(oml_object: OpenMLBase, tag: str, untag: bool = False) -> None:  # noqa: FBT001, FBT002
    api_type_alias = _get_rest_api_type_alias(oml_object)
    if oml_object.id is None:
        raise openml.exceptions.ObjectNotPublishedError(
            f"Cannot tag an {api_type_alias} that has not been published yet."
            "Please publish the object first before being able to tag it."
            f"\n{oml_object}",
        )
    _tag_entity(entity_type=api_type_alias, entity_id=oml_object.id, tag=tag, untag=untag)


def _tag_entity(entity_type: str, entity_id: int, tag: str, *, untag: bool = False) -> list[str]:
    """
    Function that tags or untags a given entity on OpenML. As the OpenML
    API tag functions all consist of the same format, this function covers
    all entity types (currently: dataset, task, flow, setup, run). Could
    be used in a partial to provide dataset_tag, dataset_untag, etc.

    Parameters
    ----------
    entity_type : str
        Name of the entity to tag (e.g., run, flow, data)

    entity_id : int
        OpenML id of the entity

    tag : str
        The tag

    untag : bool
        Set to true if needed to untag, rather than tag

    Returns
    -------
    tags : list
        List of tags that the entity is (still) tagged with
    """
    legal_entities = {"data", "task", "flow", "setup", "run"}
    if entity_type not in legal_entities:
        raise ValueError(f"Can't tag a {entity_type}")

    if untag:
        uri = f"{entity_type}/untag"
        main_tag = f"oml:{entity_type}_untag"
    else:
        uri = f"{entity_type}/tag"
        main_tag = f"oml:{entity_type}_tag"

    result_xml = openml._api_calls._perform_api_call(
        uri,
        "post",
        {f"{entity_type}_id": entity_id, "tag": tag},
    )

    result = xmltodict.parse(result_xml, force_list={"oml:tag"})[main_tag]

    if "oml:tag" in result:
        return result["oml:tag"]  # type: ignore

    # no tags, return empty list
    return []


# TODO(eddiebergman): Maybe this can be made more specific with a Literal
def _delete_entity(entity_type: str, entity_id: int) -> bool:
    """
    Function that deletes a given entity on OpenML. As the OpenML
    API tag functions all consist of the same format, this function covers
    all entity types that can be deleted (currently: dataset, task, flow,
    run, study and user).

    Parameters
    ----------
    entity_type : str
        Name of the entity to tag (e.g., run, flow, data)

    entity_id : int
        OpenML id of the entity

    Returns
    -------
    bool
        True iff the deletion was successful. False otherwse
    """
    legal_entities = {
        "data",
        "flow",
        "task",
        "run",
        "study",
        "user",
    }
    if entity_type not in legal_entities:
        raise ValueError("Can't delete a %s" % entity_type)

    url_suffix = "%s/%d" % (entity_type, entity_id)
    try:
        result_xml = openml._api_calls._perform_api_call(url_suffix, "delete")
        result = xmltodict.parse(result_xml)
        return f"oml:{entity_type}_delete" in result
    except openml.exceptions.OpenMLServerException as e:
        # https://github.com/openml/OpenML/blob/21f6188d08ac24fcd2df06ab94cf421c946971b0/openml_OS/views/pages/api_new/v1/xml/pre.php
        # Most exceptions are descriptive enough to be raised as their standard
        # OpenMLServerException, however there are two cases where we add information:
        #  - a generic "failed" message, we direct them to the right issue board
        #  - when the user successfully authenticates with the server,
        #    but user is not allowed to take the requested action,
        #    in which case we specify a OpenMLNotAuthorizedError.
        by_other_user = [323, 353, 393, 453, 594]
        has_dependent_entities = [324, 326, 327, 328, 354, 454, 464, 595]
        unknown_reason = [325, 355, 394, 455, 593]
        if e.code in by_other_user:
            raise openml.exceptions.OpenMLNotAuthorizedError(
                message=(
                    f"The {entity_type} can not be deleted because it was not uploaded by you."
                ),
            ) from e
        if e.code in has_dependent_entities:
            raise openml.exceptions.OpenMLNotAuthorizedError(
                message=(
                    f"The {entity_type} can not be deleted because "
                    f"it still has associated entities: {e.message}"
                ),
            ) from e
        if e.code in unknown_reason:
            raise openml.exceptions.OpenMLServerError(
                message=(
                    f"The {entity_type} can not be deleted for unknown reason,"
                    " please open an issue at: https://github.com/openml/openml/issues/new"
                ),
            ) from e
        raise


@overload
def _list_all(
    listing_call: Callable[P, Any],
    list_output_format: Literal["dict"] = ...,
    *args: P.args,
    **filters: P.kwargs,
) -> dict:
    ...


@overload
def _list_all(
    listing_call: Callable[P, Any],
    list_output_format: Literal["object"],
    *args: P.args,
    **filters: P.kwargs,
) -> dict:
    ...


@overload
def _list_all(
    listing_call: Callable[P, Any],
    list_output_format: Literal["dataframe"],
    *args: P.args,
    **filters: P.kwargs,
) -> pd.DataFrame:
    ...


def _list_all(  # noqa: C901, PLR0912
    listing_call: Callable[P, Any],
    list_output_format: Literal["dict", "dataframe", "object"] = "dict",
    *args: P.args,
    **filters: P.kwargs,
) -> dict | pd.DataFrame:
    """Helper to handle paged listing requests.

    Example usage:

    ``evaluations = list_all(list_evaluations, "predictive_accuracy", task=mytask)``

    Parameters
    ----------
    listing_call : callable
        Call listing, e.g. list_evaluations.
    list_output_format : str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
        - If 'object' the output is a dict of objects (only for some `listing_call`)
    *args : Variable length argument list
        Any required arguments for the listing call.
    **filters : Arbitrary keyword arguments
        Any filters that can be applied to the listing function.
        additionally, the batch_size can be specified. This is
        useful for testing purposes.

    Returns
    -------
    dict or dataframe
    """
    # eliminate filters that have a None value
    active_filters = {key: value for key, value in filters.items() if value is not None}
    page = 0
    result = pd.DataFrame() if list_output_format == "dataframe" else {}

    # Default batch size per paging.
    # This one can be set in filters (batch_size), but should not be
    # changed afterwards. The derived batch_size can be changed.
    BATCH_SIZE_ORIG = active_filters.pop("batch_size", 10000)
    if not isinstance(BATCH_SIZE_ORIG, int):
        raise ValueError(f"'batch_size' should be an integer but got {BATCH_SIZE_ORIG}")

    # max number of results to be shown
    LIMIT: int | float | None = active_filters.pop("size", None)  # type: ignore
    if (LIMIT is not None) and (not isinstance(LIMIT, int)) and (not np.isinf(LIMIT)):
        raise ValueError(f"'limit' should be an integer or inf but got {LIMIT}")

    if LIMIT is not None and BATCH_SIZE_ORIG > LIMIT:
        BATCH_SIZE_ORIG = LIMIT

    offset = active_filters.pop("offset", 0)
    if not isinstance(offset, int):
        raise ValueError(f"'offset' should be an integer but got {offset}")

    batch_size = BATCH_SIZE_ORIG
    while True:
        try:
            current_offset = offset + BATCH_SIZE_ORIG * page
            new_batch = listing_call(
                *args,
                output_format=list_output_format,  # type: ignore
                **{**active_filters, "limit": batch_size, "offset": current_offset},  # type: ignore
            )
        except openml.exceptions.OpenMLServerNoResult:
            # we want to return an empty dict in this case
            # NOTE: This above statement may not actually happen, but we could just return here
            # to enforce it...
            break

        if list_output_format == "dataframe":
            if len(result) == 0:
                result = new_batch
            else:
                result = pd.concat([result, new_batch], ignore_index=True)
        else:
            # For output_format = 'dict' (or catch all)
            result.update(new_batch)

        if len(new_batch) < batch_size:
            break

        page += 1
        if LIMIT is not None:
            # check if the number of required results has been achieved
            # always do a 'bigger than' check,
            # in case of bugs to prevent infinite loops
            if len(result) >= LIMIT:
                break

            # check if there are enough results to fulfill a batch
            if LIMIT - len(result) < BATCH_SIZE_ORIG:
                batch_size = LIMIT - len(result)

    return result


def _get_cache_dir_for_key(key: str) -> Path:
    return Path(config.get_cache_directory()) / key


def _create_cache_directory(key: str) -> Path:
    cache_dir = _get_cache_dir_for_key(key)

    try:
        cache_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:  # noqa: BLE001
        raise openml.exceptions.OpenMLCacheException(
            f"Cannot create cache directory {cache_dir}."
        ) from e

    return cache_dir


def _get_cache_dir_for_id(key: str, id_: int, create: bool = False) -> Path:  # noqa: FBT001, FBT002
    cache_dir = _create_cache_directory(key) if create else _get_cache_dir_for_key(key)
    return Path(cache_dir) / str(id_)


def _create_cache_directory_for_id(key: str, id_: int) -> Path:
    """Create the cache directory for a specific ID

    In order to have a clearer cache structure and because every task
    is cached in several files (description, split), there
    is a directory for each task witch the task ID being the directory
    name. This function creates this cache directory.

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    key : str

    id_ : int

    Returns
    -------
    cache_dir : Path
        Path of the created dataset cache directory.
    """
    cache_dir = _get_cache_dir_for_id(key, id_, create=True)
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ValueError("%s cache dir exists but is not a directory!" % key)

    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def _remove_cache_dir_for_id(key: str, cache_dir: Path) -> None:
    """Remove the task cache directory

    This function is NOT thread/multiprocessing safe.

    Parameters
    ----------
    key : str

    cache_dir : str
    """
    try:
        shutil.rmtree(cache_dir)
    except OSError as e:
        raise ValueError(
            f"Cannot remove faulty {key} cache directory {cache_dir}. Please do this manually!",
        ) from e


def thread_safe_if_oslo_installed(func: Callable[P, R]) -> Callable[P, R]:
    try:
        # Currently, importing oslo raises a lot of warning that it will stop working
        # under python3.8; remove this once they disappear
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from oslo_concurrency import lockutils

        @wraps(func)
        def safe_func(*args: P.args, **kwargs: P.kwargs) -> R:
            # Lock directories use the id that is passed as either positional or keyword argument.
            id_parameters = [parameter_name for parameter_name in kwargs if "_id" in parameter_name]
            if len(id_parameters) == 1:
                id_ = kwargs[id_parameters[0]]
            elif len(args) > 0:
                id_ = args[0]
            else:
                raise RuntimeError(
                    f"An id must be specified for {func.__name__}, was passed: ({args}, {kwargs}).",
                )
            # The [7:] gets rid of the 'openml.' prefix
            lock_name = f"{func.__module__[7:]}.{func.__name__}:{id_}"
            with lockutils.external_lock(name=lock_name, lock_path=_create_lockfiles_dir()):
                return func(*args, **kwargs)

        return safe_func
    except ImportError:
        return func


def _create_lockfiles_dir() -> Path:
    path = Path(config.get_cache_directory()) / "locks"
    # TODO(eddiebergman): Not sure why this is allowed to error and ignore???
    with contextlib.suppress(OSError):
        path.mkdir(exist_ok=True, parents=True)
    return path
