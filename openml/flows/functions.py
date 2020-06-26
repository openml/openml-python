# License: BSD 3-Clause

import dateutil.parser
from collections import OrderedDict
import os
import io
import re
import xmltodict
import pandas as pd
from typing import Any, Union, Dict, Optional, List

from ..exceptions import OpenMLCacheException
import openml._api_calls
from . import OpenMLFlow
import openml.utils


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
    directory_content = os.listdir(flow_cache_dir)
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
    flow_file = os.path.join(fid_cache_dir, "flow.xml")

    try:
        with io.open(flow_file, encoding="utf8") as fh:
            return _create_flow_from_xml(fh.read())
    except (OSError, IOError):
        openml.utils._remove_cache_dir_for_id(FLOWS_CACHE_DIR_NAME, fid_cache_dir)
        raise OpenMLCacheException("Flow file for fid %d not " "cached" % fid)


@openml.utils.thread_safe_if_oslo_installed
def get_flow(flow_id: int, reinstantiate: bool = False, strict_version: bool = True) -> OpenMLFlow:
    """Download the OpenML flow for a given flow ID.

    Parameters
    ----------
    flow_id : int
        The OpenML flow id.

    reinstantiate: bool
        Whether to reinstantiate the flow to a model instance.

    strict_version : bool, default=True
        Whether to fail if version requirements are not fulfilled.

    Returns
    -------
    flow : OpenMLFlow
        the flow
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

        xml_file = os.path.join(
            openml.utils._create_cache_directory_for_id(FLOWS_CACHE_DIR_NAME, flow_id), "flow.xml",
        )

        flow_xml = openml._api_calls._perform_api_call("flow/%d" % flow_id, request_method="get")
        with io.open(xml_file, "w", encoding="utf8") as fh:
            fh.write(flow_xml)

        return _create_flow_from_xml(flow_xml)


def list_flows(
    offset: Optional[int] = None,
    size: Optional[int] = None,
    tag: Optional[str] = None,
    output_format: str = "dict",
    **kwargs
) -> Union[Dict, pd.DataFrame]:

    """
    Return a list of all flows which are on OpenML.
    (Supports large amount of results)

    Parameters
    ----------
    offset : int, optional
        the number of flows to skip, starting from the first
    size : int, optional
        the maximum number of flows to return
    tag : str, optional
        the tag to include
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    kwargs: dict, optional
        Legal filter operators: uploader.

    Returns
    -------
    flows : dict of dicts, or dataframe
        - If output_format='dict'
            A mapping from flow_id to a dict giving a brief overview of the
            respective flow.
            Every flow is represented by a dictionary containing
            the following information:
            - flow id
            - full name
            - name
            - version
            - external version
            - uploader

        - If output_format='dataframe'
            Each row maps to a dataset
            Each column contains the following information:
            - flow id
            - full name
            - name
            - version
            - external version
            - uploader
    """
    if output_format not in ["dataframe", "dict"]:
        raise ValueError(
            "Invalid output format selected. " "Only 'dict' or 'dataframe' applicable."
        )

    return openml.utils._list_all(
        output_format=output_format,
        listing_call=_list_flows,
        offset=offset,
        size=size,
        tag=tag,
        **kwargs
    )


def _list_flows(output_format="dict", **kwargs) -> Union[Dict, pd.DataFrame]:
    """
    Perform the api call that return a list of all flows.

    Parameters
    ----------
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    kwargs: dict, optional
        Legal filter operators: uploader, tag, limit, offset.

    Returns
    -------
    flows : dict, or dataframe
    """
    api_call = "flow/list"

    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)

    return __list_flows(api_call=api_call, output_format=output_format)


def flow_exists(name: str, external_version: str) -> Union[int, bool]:
    """Retrieves the flow id.

    A flow is uniquely identified by name + external_version.

    Parameters
    ----------
    name : string
        Name of the flow
    external_version : string
        Version information associated with flow.

    Returns
    -------
    flow_exist : int or bool
        flow id iff exists, False otherwise

    Notes
    -----
    see http://www.openml.org/api_docs/#!/flow/get_flow_exists_name_version
    """
    if not (isinstance(name, str) and len(name) > 0):
        raise ValueError("Argument 'name' should be a non-empty string")
    if not (isinstance(name, str) and len(external_version) > 0):
        raise ValueError("Argument 'version' should be a non-empty string")

    xml_response = openml._api_calls._perform_api_call(
        "flow/exists", "post", data={"name": name, "external_version": external_version},
    )

    result_dict = xmltodict.parse(xml_response)
    flow_id = int(result_dict["oml:flow_exists"]["oml:id"])
    if flow_id > 0:
        return flow_id
    else:
        return False


def get_flow_id(
    model: Optional[Any] = None, name: Optional[str] = None, exact_version=True,
) -> Union[int, bool, List[int]]:
    """Retrieves the flow id for a model or a flow name.

    Provide either a model or a name to this function. Depending on the input, it does

    * ``model`` and ``exact_version == True``: This helper function first queries for the necessary
      extension. Second, it uses that extension to convert the model into a flow. Third, it
      executes ``flow_exists`` to potentially obtain the flow id the flow is published to the
      server.
    * ``model`` and ``exact_version == False``: This helper function first queries for the
      necessary extension. Second, it uses that extension to convert the model into a flow. Third
      it calls ``list_flows`` and filters the returned values based on the flow name.
    * ``name``: Ignores ``exact_version`` and calls ``list_flows``, then filters the returned
      values based on the flow name.

    Parameters
    ----------
    model : object
        Any model. Must provide either ``model`` or ``name``.
    name : str
        Name of the flow. Must provide either ``model`` or ``name``.
    exact_version : bool
        Whether to return the ``flow_id`` of the exact version or all ``flow_id``s where the name
        of the flow matches. This is only taken into account for a model where a version number
        is available.

    Returns
    -------
    int or bool, List
        flow id iff exists, ``False`` otherwise, List if exact_version is ``False``
    """
    if model is None and name is None:
        raise ValueError(
            "Need to provide either argument `model` or argument `name`, but both are `None`."
        )
    elif model is not None and name is not None:
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
    else:
        flow_name = name
        exact_version = False

    if exact_version:
        return flow_exists(name=flow_name, external_version=external_version)
    else:
        flows = list_flows(output_format="dataframe")
        assert isinstance(flows, pd.DataFrame)  # Make mypy happy
        flows = flows.query('name == "{}"'.format(flow_name))
        return flows["id"].to_list()


def __list_flows(api_call: str, output_format: str = "dict") -> Union[Dict, pd.DataFrame]:

    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    flows_dict = xmltodict.parse(xml_string, force_list=("oml:flow",))

    # Minimalistic check if the XML is useful
    assert type(flows_dict["oml:flows"]["oml:flow"]) == list, type(flows_dict["oml:flows"])
    assert flows_dict["oml:flows"]["@xmlns:oml"] == "http://openml.org/openml", flows_dict[
        "oml:flows"
    ]["@xmlns:oml"]

    flows = dict()
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

    if output_format == "dataframe":
        flows = pd.DataFrame.from_dict(flows, orient="index")

    return flows


def _check_flow_for_server_id(flow: OpenMLFlow) -> None:
    """ Raises a ValueError if the flow or any of its subflows has no flow id. """

    # Depth-first search to check if all components were uploaded to the
    # server before parsing the parameters
    stack = list()
    stack.append(flow)
    while len(stack) > 0:
        current = stack.pop()
        if current.flow_id is None:
            raise ValueError("Flow %s has no flow_id!" % current.name)
        else:
            for component in current.components.values():
                stack.append(component)


def assert_flows_equal(
    flow1: OpenMLFlow,
    flow2: OpenMLFlow,
    ignore_parameter_values_on_older_children: str = None,
    ignore_parameter_values: bool = False,
    ignore_custom_name_if_none: bool = False,
    check_description: bool = True,
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
    """
    if not isinstance(flow1, OpenMLFlow):
        raise TypeError("Argument 1 must be of type OpenMLFlow, but is %s" % type(flow1))

    if not isinstance(flow2, OpenMLFlow):
        raise TypeError("Argument 2 must be of type OpenMLFlow, but is %s" % type(flow2))

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
            for name in set(attr1.keys()).union(attr2.keys()):
                if name not in attr1:
                    raise ValueError(
                        "Component %s only available in " "argument2, but not in argument1." % name
                    )
                if name not in attr2:
                    raise ValueError(
                        "Component %s only available in " "argument2, but not in argument1." % name
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
                            "Flow %s: parameter set of flow "
                            "differs from the parameters stored "
                            "on the server." % flow1.name
                        )

                if ignore_parameter_values_on_older_children:
                    upload_date_current_flow = dateutil.parser.parse(flow1.upload_date)
                    upload_date_parent_flow = dateutil.parser.parse(
                        ignore_parameter_values_on_older_children
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
                params1 = set(flow1.parameters_meta_info.keys())
                params2 = set(flow2.parameters_meta_info.keys())
                if params1 != params2:
                    raise ValueError(
                        "Parameter list in meta info for parameters differ " "in the two flows."
                    )
                # iterating over the parameter's meta info list
                for param in params1:
                    if (
                        isinstance(flow1.parameters_meta_info[param], Dict)
                        and isinstance(flow2.parameters_meta_info[param], Dict)
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
                    elif value1 != value2:
                        raise ValueError(
                            "Flow {}: data type for parameter {} in {} differ "
                            "as {}\nvs\n{}".format(flow1.name, param, key, value1, value2)
                        )
                # the continue is to avoid the 'attr != attr2' check at end of function
                continue

            if attr1 != attr2:
                raise ValueError(
                    "Flow %s: values for attribute '%s' differ: "
                    "'%s'\nvs\n'%s'." % (str(flow1.name), str(key), str(attr1), str(attr2))
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
