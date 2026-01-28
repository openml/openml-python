# License: BSD 3-Clause
from __future__ import annotations

import contextlib
import re
import shutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    overload,
)
from typing_extensions import ParamSpec

import numpy as np
import xmltodict
from minio.helpers import ProgressType
from tqdm import tqdm

import openml
import openml._api_calls
import openml.exceptions

from . import config

# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from openml.base import OpenMLBase

    P = ParamSpec("P")
    R = TypeVar("R")
    _SizedT = TypeVar("_SizedT", bound=Sized)


@overload
def extract_xml_tags(
    xml_tag_name: str,
    node: Mapping[str, Any],
    *,
    allow_none: Literal[True] = ...,
) -> Any | None: ...


@overload
def extract_xml_tags(
    xml_tag_name: str,
    node: Mapping[str, Any],
    *,
    allow_none: Literal[False],
) -> Any: ...


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


def _tag_openml_base(oml_object: OpenMLBase, tag: str, untag: bool = False) -> None:  # noqa: FBT002
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
        raise ValueError(f"Can't delete a {entity_type}")

    url_suffix = f"{entity_type}/{entity_id}"
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
        raise e


def _list_all(  # noqa: C901
    listing_call: Callable[[int, int], _SizedT],
    *,
    limit: int | None = None,
    offset: int | None = None,
    batch_size: int | None = 10_000,
) -> list[_SizedT]:
    """Helper to handle paged listing requests.

    Example usage:

    ``evaluations = list_all(list_evaluations, "predictive_accuracy", task=mytask)``

    Parameters
    ----------
    listing_call : callable
        Call listing, e.g. list_evaluations. Takes two positional
        arguments: batch_size and offset.
    batch_size : int, optional
        The batch size to use for the listing call.
    offset : int, optional
        The initial offset to use for the listing call.
    limit : int, optional
        The total size of the listing. If not provided, the function will
        request the first batch and then continue until no more results are
        returned

    Returns
    -------
    List of types returned from type of the listing call
    """
    page = 0
    results: list[_SizedT] = []

    offset = offset if offset is not None else 0
    batch_size = batch_size if batch_size is not None else 10_000

    LIMIT = limit
    BATCH_SIZE_ORIG = batch_size

    # Default batch size per paging.
    # This one can be set in filters (batch_size), but should not be
    # changed afterwards. The derived batch_size can be changed.
    if not isinstance(BATCH_SIZE_ORIG, int):
        raise ValueError(f"'batch_size' should be an integer but got {BATCH_SIZE_ORIG}")

    if (LIMIT is not None) and (not isinstance(LIMIT, int)) and (not np.isinf(LIMIT)):
        raise ValueError(f"'limit' should be an integer or inf but got {LIMIT}")

    # If our batch size is larger than the limit, we should only
    # request one batch of size of LIMIT
    if LIMIT is not None and BATCH_SIZE_ORIG > LIMIT:
        BATCH_SIZE_ORIG = LIMIT

    if not isinstance(offset, int):
        raise ValueError(f"'offset' should be an integer but got {offset}")

    batch_size = BATCH_SIZE_ORIG
    while True:
        try:
            current_offset = offset + BATCH_SIZE_ORIG * page
            new_batch = listing_call(batch_size, current_offset)
        except openml.exceptions.OpenMLServerNoResult:
            # NOTE: This above statement may not actually happen, but we could just return here
            # to enforce it...
            break

        results.append(new_batch)

        # If the batch is less than our requested batch_size, that's the last batch
        # and we can bail out.
        if len(new_batch) < batch_size:
            break

        page += 1
        if LIMIT is not None:
            # check if the number of required results has been achieved
            # always do a 'bigger than' check,
            # in case of bugs to prevent infinite loops
            n_received = sum(len(result) for result in results)
            if n_received >= LIMIT:
                break

            # check if there are enough results to fulfill a batch
            if LIMIT - n_received < BATCH_SIZE_ORIG:
                batch_size = LIMIT - n_received

    return results


def _get_cache_dir_for_key(key: str) -> Path:
    return Path(config.get_cache_directory()) / key


def _create_cache_directory(key: str) -> Path:
    cache_dir = _get_cache_dir_for_key(key)

    try:
        cache_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        raise openml.exceptions.OpenMLCacheException(
            f"Cannot create cache directory {cache_dir}."
        ) from e

    return cache_dir


def _get_cache_dir_for_id(key: str, id_: int, create: bool = False) -> Path:  # noqa: FBT002
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
        raise ValueError(f"{key} cache dir exists but is not a directory!")

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


class ProgressBar(ProgressType):
    """Progressbar for MinIO function's `progress` parameter."""

    def __init__(self) -> None:
        self._object_name = ""
        self._progress_bar: tqdm | None = None

    def set_meta(self, object_name: str, total_length: int) -> None:
        """Initializes the progress bar.

        Parameters
        ----------
        object_name: str
          Not used.

        total_length: int
          File size of the object in bytes.
        """
        self._object_name = object_name
        self._progress_bar = tqdm(total=total_length, unit_scale=True, unit="B")

    def update(self, length: int) -> None:
        """Updates the progress bar.

        Parameters
        ----------
        length: int
          Number of bytes downloaded since last `update` call.
        """
        if not self._progress_bar:
            raise RuntimeError("Call `set_meta` before calling `update`.")
        self._progress_bar.update(length)
        if self._progress_bar.total <= self._progress_bar.n:
            self._progress_bar.close()


class ReprMixin(ABC):
    """A mixin class that provides a customizable string representation for OpenML objects.

    This mixin standardizes the __repr__ output format across OpenML classes.
    Classes inheriting from this mixin should implement the
    _get_repr_body_fields method to specify which fields to display.
    """

    def __repr__(self) -> str:
        body_fields = self._get_repr_body_fields()
        return self._apply_repr_template(body_fields)

    @abstractmethod
    def _get_repr_body_fields(self) -> Sequence[tuple[str, str | int | list[str] | None]]:
        """Collect all information to display in the __repr__ body.

        Returns
        -------
        body_fields : List[Tuple[str, Union[str, int, List[str]]]]
            A list of (name, value) pairs to display in the body of the __repr__.
            E.g.: [('metric', 'accuracy'), ('dataset', 'iris')]
            If value is a List of str, then each item of the list will appear in a separate row.
        """
        # Should be implemented in the base class.

    def _apply_repr_template(
        self,
        body_fields: Iterable[tuple[str, str | int | list[str] | None]],
    ) -> str:
        """Generates the header and formats the body for string representation of the object.

        Parameters
        ----------
        body_fields: List[Tuple[str, str]]
           A list of (name, value) pairs to display in the body of the __repr__.
        """
        # We add spaces between capitals, e.g. ClassificationTask -> Classification Task
        name_with_spaces = re.sub(
            r"(\w)([A-Z])",
            r"\1 \2",
            self.__class__.__name__[len("OpenML") :],
        )
        header_text = f"OpenML {name_with_spaces}"
        header = f"{header_text}\n{'=' * len(header_text)}\n"

        _body_fields: list[tuple[str, str | int | list[str]]] = [
            (k, "None" if v is None else v) for k, v in body_fields
        ]
        longest_field_name_length = max(len(name) for name, _ in _body_fields)
        field_line_format = f"{{:.<{longest_field_name_length}}}: {{}}"
        body = "\n".join(field_line_format.format(name, value) for name, value in _body_fields)
        return header + body
