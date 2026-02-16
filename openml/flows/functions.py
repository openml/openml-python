# License: BSD 3-Clause
from __future__ import annotations

import os
import re
from collections import OrderedDict
from functools import partial
from typing import Any

import dateutil.parser
import pandas as pd
import xmltodict

import openml._api_calls
import openml.utils
from openml.exceptions import OpenMLCacheException

from . import OpenMLFlow

FLOWS_CACHE_DIR_NAME = "flows"


def _get_cached_flows() -> OrderedDict:
    """Return all the cached flows.

    Returns
    -------
    flows : OrderedDict
        Dictionary with flows. Each flow is an instance of OpenMLFlow.
    """
    flows = OrderedDict()  # type: 'OrderedDict[int, OpenMLFlow]'

    flow_cache_dir = openml.utils._create_cache_directory(FLOWS_CACHE_DIR_NAME)
    directory_content = os.listdir(flow_cache_dir)  # noqa: PTH208
    directory_content.sort()
    # Find all flow ids for which we have downloaded
    # the flow description

    for filename in directory_content:
        if not re.match(r"[0-9]*", filename):
            continue

        fid = int(filename)
        flows[fid] = _get_cached_flow(fid)

    return flows


def _get_cached_flow(fid: int) -> OpenMLFlow:
    """Get the cached flow with the given id.

    Parameters
    ----------
    fid : int
        Flow id.

    Returns
    -------
    OpenMLFlow.
    """
    fid_cache_dir = openml.utils._create_cache_directory_for_id(FLOWS_CACHE_DIR_NAME, fid)
    flow_file = fid_cache_dir / "flow.xml"

    try:
        with flow_file.open(encoding="utf8") as fh:
            return _create_flow_from_xml(fh.read())
    except OSError as e:
        openml.utils._remove_cache_dir_for_id(FLOWS_CACHE_DIR_NAME, fid_cache_dir)
        raise OpenMLCacheException(f"Flow file for fid {fid} not cached") from e


@openml.utils.thread_safe_if_oslo_installed
def get_flow(flow_id: int, reinstantiate: bool = False, strict_version: bool = True) -> OpenMLFlow:  # noqa: FBT002
    """Fetch an OpenMLFlow by its server-assigned ID.

    Queries the OpenML REST API for the flow metadata and returns an
    :class:`OpenMLFlow` instance. If the flow is already cached locally,
    the cached copy is returned. Optionally the flow can be re-instantiated
    into a concrete model instance using the registered extension.

    Parameters
    ----------
    flow_id : int
        The OpenML flow id.
    reinstantiate : bool, optional (default=False)
        If True, convert the flow description into a concrete model instance
        using the flow's extension (e.g., sklearn). If conversion fails and
        ``strict_version`` is True, an exception will be raised.
    strict_version : bool, optional (default=True)
        When ``reinstantiate`` is True, whether to enforce exact version
        requirements for the extension/model. If False, a new flow may
        be returned when versions differ.

    Returns
    -------
    OpenMLFlow
        The flow object with metadata; ``model`` may be populated when
        ``reinstantiate=True``.

    Raises
    ------
    OpenMLCacheException
        When cached flow files are corrupted or cannot be read.
    OpenMLServerException
        When the REST API call fails.

    Side Effects
    ------------
    - Writes to ``openml.config.cache_directory/flows/{flow_id}/flow.xml``
      when the flow is downloaded from the server.

    Preconditions
    -------------
    - Network access to the OpenML server is required unless the flow is cached.
    - For private flows, ``openml.config.apikey`` must be set.

    Notes
    -----
    Results are cached to speed up subsequent calls. When ``reinstantiate`` is
    True and version mismatches occur, a new flow may be returned to reflect
    the converted model (only when ``strict_version`` is False).

    Examples
    --------
    >>> import openml
    >>> flow = openml.flows.get_flow(5)  # doctest: +SKIP
    """
    flow_id = int(flow_id)
    flow = _get_flow_description(flow_id)

    if reinstantiate:
        flow.model = flow.extension.flow_to_model(flow, strict_version=strict_version)
        if not strict_version:
            # check if we need to return a new flow b/c of version mismatch
            new_flow = flow.extension.model_to_flow(flow.model)
            if new_flow.dependencies != flow.dependencies:
                return new_flow
    return flow


def _get_flow_description(flow_id: int) -> OpenMLFlow:
    """Get the Flow for a given  ID.

    Does the real work for get_flow. It returns a cached flow
    instance if the flow exists locally, otherwise it downloads the
    flow and returns an instance created from the xml representation.

    Parameters
    ----------
    flow_id : int
        The OpenML flow id.

    Returns
    -------
    OpenMLFlow
    """
    try:
        return _get_cached_flow(flow_id)
    except OpenMLCacheException:
        xml_file = (
            openml.utils._create_cache_directory_for_id(FLOWS_CACHE_DIR_NAME, flow_id) / "flow.xml"
        )
        flow_xml = openml._api_calls._perform_api_call(f"flow/{flow_id}", request_method="get")

        with xml_file.open("w", encoding="utf8") as fh:
            fh.write(flow_xml)

        return _create_flow_from_xml(flow_xml)


def list_flows(
    offset: int | None = None,
    size: int | None = None,
    tag: str | None = None,
    uploader: str | None = None,
) -> pd.DataFrame:
    """List flows available on the OpenML server.

    This function supports paging and filtering and returns a pandas
    DataFrame with one row per flow and columns for id, name, version,
    external_version, full_name and uploader.

    Parameters
    ----------
    offset : int, optional
        Number of flows to skip, starting from the first (for paging).
    size : int, optional
        Maximum number of flows to return.
    tag : str, optional
        Only return flows having this tag.
    uploader : str, optional
        Only return flows uploaded by this user.

    Returns
    -------
    pandas.DataFrame
        Rows correspond to flows. Columns include ``id``, ``full_name``,
        ``name``, ``version``, ``external_version``, and ``uploader``.

    Raises
    ------
    OpenMLServerException
        When the API call fails.

    Side Effects
    ------------
    - None: results are fetched and returned; Read-only operation.

    Preconditions
    -------------
    - Network access is required to list flows unless cached mechanisms are
      used by the underlying API helper.

    Examples
    --------
    >>> import openml
    >>> flows = openml.flows.list_flows(size=100)  # doctest: +SKIP
    """
    listing_call = partial(_list_flows, tag=tag, uploader=uploader)
    batches = openml.utils._list_all(listing_call, offset=offset, limit=size)
    if len(batches) == 0:
        return pd.DataFrame()

    return pd.concat(batches)


def _list_flows(limit: int, offset: int, **kwargs: Any) -> pd.DataFrame:
    """
    Perform the api call that return a list of all flows.

    Parameters
    ----------
    limit : int
        the maximum number of flows to return
    offset : int
        the number of flows to skip, starting from the first
    kwargs: dict, optional
        Legal filter operators: uploader, tag

    Returns
    -------
    flows : dataframe
    """
    api_call = "flow/list"

    if limit is not None:
        api_call += f"/limit/{limit}"
    if offset is not None:
        api_call += f"/offset/{offset}"

    if kwargs is not None:
        for operator, value in kwargs.items():
            if value is not None:
                api_call += f"/{operator}/{value}"

    return __list_flows(api_call=api_call)


def flow_exists(name: str, external_version: str) -> int | bool:
    """Check whether a flow (name + external_version) exists on the server.

    The OpenML server defines uniqueness of flows by the pair
    ``(name, external_version)``. This helper queries the server and
    returns the corresponding flow id when present.

    Parameters
    ----------
    name : str
        Flow name (e.g., ``sklearn.tree._classes.DecisionTreeClassifier(1)``).
    external_version : str
        Version information associated with flow.

    Returns
    -------
    int or bool
        The flow id if the flow exists on the server, otherwise ``False``.

    Raises
    ------
    ValueError
        If ``name`` or ``external_version`` are empty or not strings.
    OpenMLServerException
        When the API request fails.

    Examples
    --------
    >>> import openml
    >>> openml.flows.flow_exists("weka.JRip", "Weka_3.9.0_10153")  # doctest: +SKIP
    """
    if not (isinstance(name, str) and len(name) > 0):
        raise ValueError("Argument 'name' should be a non-empty string")
    if not (isinstance(name, str) and len(external_version) > 0):
        raise ValueError("Argument 'version' should be a non-empty string")

    xml_response = openml._api_calls._perform_api_call(
        "flow/exists",
        "post",
        data={"name": name, "external_version": external_version},
    )

    result_dict = xmltodict.parse(xml_response)
    flow_id = int(result_dict["oml:flow_exists"]["oml:id"])
    return flow_id if flow_id > 0 else False


def get_flow_id(
    model: Any | None = None,
    name: str | None = None,
    exact_version: bool = True,  # noqa: FBT002
) -> int | bool | list[int]:
    """Retrieve flow id(s) for a model instance or a flow name.

    Provide either a concrete ``model`` (which will be converted to a flow by
    the appropriate extension) or a flow ``name``. Behavior depends on
    ``exact_version``:

    - ``model`` + ``exact_version=True``: convert ``model`` to a flow and call
        :func:`flow_exists` to get a single flow id (or False).
    - ``model`` + ``exact_version=False``: convert ``model`` to a flow and
        return all server flow ids with the same flow name.
    - ``name``: ignore ``exact_version`` and return all server flow ids that
        match ``name``.

    Parameters
    ----------
    model : object, optional
            A model instance that can be handled by a registered extension. Either
            ``model`` or ``name`` must be provided.
    name : str, optional
            Flow name to query for. Either ``model`` or ``name`` must be provided.
    exact_version : bool, optional (default=True)
            When True and ``model`` is provided, only return the id for the exact
            external version. When False, return a list of matching ids.

    Returns
    -------
    int or bool or list[int]
            If ``exact_version`` is True: the flow id if found, otherwise ``False``.
            If ``exact_version`` is False: a list of matching flow ids (may be empty).

    Raises
    ------
    ValueError
            If neither ``model`` nor ``name`` is provided, or if both are provided.
    OpenMLServerException
            If underlying API calls fail.

    Side Effects
    ------------
    - May call server APIs (``flow/exists``, ``flow/list``) and therefore
        depends on network access and API keys for private flows.

    Examples
    --------
    >>> import openml
    >>> # Lookup by flow name
    >>> openml.flows.get_flow_id(name="weka.JRip")  # doctest: +SKIP
    >>> # Lookup by model instance (requires a registered extension)
    >>> import sklearn
    >>> import openml_sklearn
    >>> clf = sklearn.tree.DecisionTreeClassifier()
    >>> openml.flows.get_flow_id(model=clf)  # doctest: +SKIP
    """
    if model is not None and name is not None:
        raise ValueError("Must provide either argument `model` or argument `name`, but not both.")

    if model is not None:
        extension = openml.extensions.get_extension_by_model(model, raise_if_no_extension=True)
        if extension is None:
            # This should never happen and is only here to please mypy will be gone soon once the
            # whole function is removed
            raise TypeError(extension)
        flow = extension.model_to_flow(model)
        flow_name = flow.name
        external_version = flow.external_version
    elif name is not None:
        flow_name = name
        exact_version = False
        external_version = None
    else:
        raise ValueError(
            "Need to provide either argument `model` or argument `name`, but both are `None`."
        )

    if exact_version:
        if external_version is None:
            raise ValueError("exact_version should be False if model is None!")
        return flow_exists(name=flow_name, external_version=external_version)

    flows = list_flows()
    flows = flows.query(f'name == "{flow_name}"')
    return flows["id"].to_list()  # type: ignore[no-any-return]


def __list_flows(api_call: str) -> pd.DataFrame:
    """Retrieve information about flows from OpenML API
    and parse it to a dictionary or a Pandas DataFrame.

    Parameters
    ----------
    api_call: str
        Retrieves the information about flows.

    Returns
    -------
        The flows information in the specified output format.
    """
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    flows_dict = xmltodict.parse(xml_string, force_list=("oml:flow",))

    # Minimalistic check if the XML is useful
    assert isinstance(flows_dict["oml:flows"]["oml:flow"], list), type(flows_dict["oml:flows"])
    assert flows_dict["oml:flows"]["@xmlns:oml"] == "http://openml.org/openml", flows_dict[
        "oml:flows"
    ]["@xmlns:oml"]

    flows = {}
    for flow_ in flows_dict["oml:flows"]["oml:flow"]:
        fid = int(flow_["oml:id"])
        flow = {
            "id": fid,
            "full_name": flow_["oml:full_name"],
            "name": flow_["oml:name"],
            "version": flow_["oml:version"],
            "external_version": flow_["oml:external_version"],
            "uploader": flow_["oml:uploader"],
        }
        flows[fid] = flow

    return pd.DataFrame.from_dict(flows, orient="index")


def _check_flow_for_server_id(flow: OpenMLFlow) -> None:
    """Raises a ValueError if the flow or any of its subflows has no flow id."""
    # Depth-first search to check if all components were uploaded to the
    # server before parsing the parameters
    stack = [flow]
    while len(stack) > 0:
        current = stack.pop()
        if current.flow_id is None:
            raise ValueError(f"Flow {current.name} has no flow_id!")

        for component in current.components.values():
            stack.append(component)


def assert_flows_equal(  # noqa: C901, PLR0912, PLR0913, PLR0915
    flow1: OpenMLFlow,
    flow2: OpenMLFlow,
    ignore_parameter_values_on_older_children: str | None = None,
    ignore_parameter_values: bool = False,  # noqa: FBT002
    ignore_custom_name_if_none: bool = False,  # noqa: FBT002
    check_description: bool = True,  # noqa: FBT002
) -> None:
    """Check equality of two flows.

    Two flows are equal if their all keys which are not set by the server
    are equal, as well as all their parameters and components.

    Parameters
    ----------
    flow1 : OpenMLFlow

    flow2 : OpenMLFlow

    ignore_parameter_values_on_older_children : str (optional)
        If set to ``OpenMLFlow.upload_date``, ignores parameters in a child
        flow if it's upload date predates the upload date of the parent flow.

    ignore_parameter_values : bool
        Whether to ignore parameter values when comparing flows.

    ignore_custom_name_if_none : bool
        Whether to ignore the custom name field if either flow has `custom_name` equal to `None`.

    check_description : bool
        Whether to ignore matching of flow descriptions.

    Raises
    ------
    TypeError
        When either argument is not an :class:`OpenMLFlow`.
    ValueError
        When a relevant mismatch is found between the two flows.

    Examples
    --------
    >>> import openml
    >>> f1 = openml.flows.get_flow(5)  # doctest: +SKIP
    >>> f2 = openml.flows.get_flow(5)  # doctest: +SKIP
    >>> openml.flows.assert_flows_equal(f1, f2)  # doctest: +SKIP
    >>> # If flows differ, a ValueError is raised
    """
    if not isinstance(flow1, OpenMLFlow):
        raise TypeError(f"Argument 1 must be of type OpenMLFlow, but is {type(flow1)}")

    if not isinstance(flow2, OpenMLFlow):
        raise TypeError(f"Argument 2 must be of type OpenMLFlow, but is {type(flow2)}")

    # TODO as they are actually now saved during publish, it might be good to
    # check for the equality of these as well.
    generated_by_the_server = [
        "flow_id",
        "uploader",
        "version",
        "upload_date",
        # Tags aren't directly created by the server,
        # but the uploader has no control over them!
        "tags",
    ]
    ignored_by_python_api = ["binary_url", "binary_format", "binary_md5", "model", "_entity_id"]

    for key in set(flow1.__dict__.keys()).union(flow2.__dict__.keys()):
        if key in generated_by_the_server + ignored_by_python_api:
            continue
        attr1 = getattr(flow1, key, None)
        attr2 = getattr(flow2, key, None)
        if key == "components":
            if not (isinstance(attr1, dict) and isinstance(attr2, dict)):
                raise TypeError("Cannot compare components because they are not dictionary.")

            for name in set(attr1.keys()).union(attr2.keys()):
                if name not in attr1:
                    raise ValueError(
                        f"Component {name} only available in argument2, but not in argument1.",
                    )
                if name not in attr2:
                    raise ValueError(
                        f"Component {name} only available in argument2, but not in argument1.",
                    )
                assert_flows_equal(
                    attr1[name],
                    attr2[name],
                    ignore_parameter_values_on_older_children,
                    ignore_parameter_values,
                    ignore_custom_name_if_none,
                )
        elif key == "_extension":
            continue
        elif check_description and key == "description":
            # to ignore matching of descriptions since sklearn based flows may have
            # altering docstrings and is not guaranteed to be consistent
            continue
        else:
            if key == "parameters":
                if ignore_parameter_values or ignore_parameter_values_on_older_children:
                    params_flow_1 = set(flow1.parameters.keys())
                    params_flow_2 = set(flow2.parameters.keys())
                    symmetric_difference = params_flow_1 ^ params_flow_2
                    if len(symmetric_difference) > 0:
                        raise ValueError(
                            f"Flow {flow1.name}: parameter set of flow "
                            "differs from the parameters stored "
                            "on the server.",
                        )

                if ignore_parameter_values_on_older_children:
                    assert flow1.upload_date is not None, (
                        "Flow1 has no upload date that allows us to compare age of children."
                    )
                    upload_date_current_flow = dateutil.parser.parse(flow1.upload_date)
                    upload_date_parent_flow = dateutil.parser.parse(
                        ignore_parameter_values_on_older_children,
                    )
                    if upload_date_current_flow < upload_date_parent_flow:
                        continue

                if ignore_parameter_values:
                    # Continue needs to be done here as the first if
                    # statement triggers in both special cases
                    continue
            elif (
                key == "custom_name"
                and ignore_custom_name_if_none
                and (attr1 is None or attr2 is None)
            ):
                # If specified, we allow `custom_name` inequality if one flow's name is None.
                # Helps with backwards compatibility as `custom_name` is now auto-generated, but
                # before it used to be `None`.
                continue
            elif key == "parameters_meta_info":
                # this value is a dictionary where each key is a parameter name, containing another
                # dictionary with keys specifying the parameter's 'description' and 'data_type'
                # checking parameter descriptions can be ignored since that might change
                # data type check can also be ignored if one of them is not defined, i.e., None
                params1 = set(flow1.parameters_meta_info)
                params2 = set(flow2.parameters_meta_info)
                if params1 != params2:
                    raise ValueError(
                        "Parameter list in meta info for parameters differ in the two flows.",
                    )
                # iterating over the parameter's meta info list
                for param in params1:
                    if (
                        isinstance(flow1.parameters_meta_info[param], dict)
                        and isinstance(flow2.parameters_meta_info[param], dict)
                        and "data_type" in flow1.parameters_meta_info[param]
                        and "data_type" in flow2.parameters_meta_info[param]
                    ):
                        value1 = flow1.parameters_meta_info[param]["data_type"]
                        value2 = flow2.parameters_meta_info[param]["data_type"]
                    else:
                        value1 = flow1.parameters_meta_info[param]
                        value2 = flow2.parameters_meta_info[param]
                    if value1 is None or value2 is None:
                        continue

                    if value1 != value2:
                        raise ValueError(
                            f"Flow {flow1.name}: data type for parameter {param} in {key} differ "
                            f"as {value1}\nvs\n{value2}",
                        )
                # the continue is to avoid the 'attr != attr2' check at end of function
                continue

            if attr1 != attr2:
                raise ValueError(
                    f"Flow {flow1.name!s}: values for attribute '{key!s}' differ: "
                    f"'{attr1!s}'\nvs\n'{attr2!s}'.",
                )


def _create_flow_from_xml(flow_xml: str) -> OpenMLFlow:
    """Create flow object from xml

    Parameters
    ----------
    flow_xml: xml representation of a flow

    Returns
    -------
    OpenMLFlow
    """
    return OpenMLFlow._from_dict(xmltodict.parse(flow_xml))


def delete_flow(flow_id: int) -> bool:
    """Delete flow with id `flow_id` from the OpenML server.

    You can only delete flows which you uploaded and which
    which are not linked to runs.

    Parameters
    ----------
    flow_id : int
        OpenML id of the flow

    Returns
    -------
    bool
        True if the deletion was successful. False otherwise.

    Raises
    ------
    OpenMLServerException
        If the server-side deletion fails due to permissions or other errors.

    Side Effects
    ------------
    - Removes the flow from the OpenML server (if permitted).

    Examples
    --------
    >>> import openml
    >>> # Deletes flow 23 if you are the uploader and it's not linked to runs
    >>> openml.flows.delete_flow(23)  # doctest: +SKIP
    """
    return openml.utils._delete_entity("flow", flow_id)
