# License: BSD 3-Clause
from __future__ import annotations

from collections import OrderedDict
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Iterable
from typing_extensions import Literal

import pandas as pd
import xmltodict

import openml
import openml.exceptions
import openml.utils
from openml import config
from openml.flows import OpenMLFlow, flow_exists

from .setup import OpenMLParameter, OpenMLSetup


def setup_exists(flow: OpenMLFlow) -> int:
    """
    Checks whether a hyperparameter configuration already exists on the server.

    Parameters
    ----------
    flow : OpenMLFlow
        The openml flow object. Should have flow id present for the main flow
        and all subflows (i.e., it should be downloaded from the server by
        means of flow.get, and not instantiated locally)

    Returns
    -------
    setup_id : int
        setup id iff exists, False otherwise
    """
    # sadly, this api call relies on a run object
    openml.flows.functions._check_flow_for_server_id(flow)
    if flow.model is None:
        raise ValueError("Flow should have model field set with the actual model.")
    if flow.extension is None:
        raise ValueError("Flow should have model field set with the correct extension.")

    # checks whether the flow exists on the server and flow ids align
    exists = flow_exists(flow.name, flow.external_version)
    if exists != flow.flow_id:
        raise ValueError(
            f"Local flow id ({flow.id}) differs from server id ({exists}). "
            "If this issue persists, please contact the developers.",
        )

    openml_param_settings = flow.extension.obtain_parameter_values(flow)
    description = xmltodict.unparse(_to_dict(flow.flow_id, openml_param_settings), pretty=True)
    file_elements = {
        "description": ("description.arff", description),
    }  # type: openml._api_calls.FILE_ELEMENTS_TYPE
    result = openml._api_calls._perform_api_call(
        "/setup/exists/",
        "post",
        file_elements=file_elements,
    )
    result_dict = xmltodict.parse(result)
    setup_id = int(result_dict["oml:setup_exists"]["oml:id"])
    return setup_id if setup_id > 0 else False


def _get_cached_setup(setup_id: int) -> OpenMLSetup:
    """Load a run from the cache.

    Parameters
    ----------
    setup_id : int
        ID of the setup to be loaded.

    Returns
    -------
    OpenMLSetup
        The loaded setup object.

    Raises
    ------
    OpenMLCacheException
        If the setup file for the given setup ID is not cached.
    """
    cache_dir = Path(config.get_cache_directory())
    setup_cache_dir = cache_dir / "setups" / str(setup_id)
    try:
        setup_file = setup_cache_dir / "description.xml"
        with setup_file.open(encoding="utf8") as fh:
            setup_xml = xmltodict.parse(fh.read())
            return _create_setup_from_xml(setup_xml)

    except OSError as e:
        raise openml.exceptions.OpenMLCacheException(
            "Setup file for setup id %d not cached" % setup_id,
        ) from e


def get_setup(setup_id: int) -> OpenMLSetup:
    """
     Downloads the setup (configuration) description from OpenML
     and returns a structured object

    Parameters
    ----------
    setup_id : int
        The Openml setup_id

    Returns
    -------
    OpenMLSetup (an initialized openml setup object)
    """
    setup_dir = Path(config.get_cache_directory()) / "setups" / str(setup_id)
    setup_dir.mkdir(exist_ok=True, parents=True)

    setup_file = setup_dir / "description.xml"

    try:
        return _get_cached_setup(setup_id)
    except openml.exceptions.OpenMLCacheException:
        url_suffix = f"/setup/{setup_id}"
        setup_xml = openml._api_calls._perform_api_call(url_suffix, "get")
        with setup_file.open("w", encoding="utf8") as fh:
            fh.write(setup_xml)

    result_dict = xmltodict.parse(setup_xml)
    return _create_setup_from_xml(result_dict)


def list_setups(  # noqa: PLR0913
    offset: int | None = None,
    size: int | None = None,
    flow: int | None = None,
    tag: str | None = None,
    setup: Iterable[int] | None = None,
    output_format: Literal["object", "dataframe"] = "object",
) -> dict[int, OpenMLSetup] | pd.DataFrame:
    """
    List all setups matching all of the given filters.

    Parameters
    ----------
    offset : int, optional
    size : int, optional
    flow : int, optional
    tag : str, optional
    setup : Iterable[int], optional
    output_format: str, optional (default='object')
        The parameter decides the format of the output.
        - If 'dataframe' the output is a pandas DataFrame
        - If 'object' the output is a dictionary of OpenMLSetup objects

    Returns
    -------
    dict or dataframe
    """
    if output_format not in ["dataframe", "object"]:
        raise ValueError(
            "Invalid output format selected. Only 'object', or 'dataframe' applicable.",
        )

    listing_call = partial(_list_setups, flow=flow, tag=tag, setup=setup)
    batches = openml.utils._list_all(
        listing_call,
        batch_size=1_000,  # batch size for setups is lower
        offset=offset,
        limit=size,
    )
    flattened = list(chain.from_iterable(batches))
    if output_format == "object":
        return {setup.setup_id: setup for setup in flattened}

    records = [setup._to_dict() for setup in flattened]
    return pd.DataFrame.from_records(records, index="setup_id")


def _list_setups(
    limit: int,
    offset: int,
    *,
    setup: Iterable[int] | None = None,
    flow: int | None = None,
    tag: str | None = None,
) -> list[OpenMLSetup]:
    """Perform API call `/setup/list/{filters}`

    Parameters
    ----------
    The setup argument that is a list is separated from the single value
    filters which are put into the kwargs.

    limit : int
    offset : int
    setup : list(int), optional
    flow : int, optional
    tag : str, optional

    Returns
    -------
    The setups that match the filters, going from id to the OpenMLSetup object.
    """
    api_call = "setup/list"
    if limit is not None:
        api_call += f"/limit/{limit}"
    if offset is not None:
        api_call += f"/offset/{offset}"
    if setup is not None:
        api_call += "/setup/{}".format(",".join([str(int(i)) for i in setup]))
    if flow is not None:
        api_call += f"/flow/{flow}"
    if tag is not None:
        api_call += f"/tag/{tag}"

    return __list_setups(api_call=api_call)


def __list_setups(api_call: str) -> list[OpenMLSetup]:
    """Helper function to parse API calls which are lists of setups"""
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    setups_dict = xmltodict.parse(xml_string, force_list=("oml:setup",))
    openml_uri = "http://openml.org/openml"
    # Minimalistic check if the XML is useful
    if "oml:setups" not in setups_dict:
        raise ValueError(
            f'Error in return XML, does not contain "oml:setups": {setups_dict!s}',
        )

    if "@xmlns:oml" not in setups_dict["oml:setups"]:
        raise ValueError(
            f'Error in return XML, does not contain "oml:setups"/@xmlns:oml: {setups_dict!s}',
        )

    if setups_dict["oml:setups"]["@xmlns:oml"] != openml_uri:
        raise ValueError(
            "Error in return XML, value of  "
            '"oml:seyups"/@xmlns:oml is not '
            f'"{openml_uri}": {setups_dict!s}',
        )

    assert isinstance(setups_dict["oml:setups"]["oml:setup"], list), type(setups_dict["oml:setups"])

    return [
        _create_setup_from_xml({"oml:setup_parameters": setup_})
        for setup_ in setups_dict["oml:setups"]["oml:setup"]
    ]


def initialize_model(setup_id: int, *, strict_version: bool = True) -> Any:
    """
    Initialized a model based on a setup_id (i.e., using the exact
    same parameter settings)

    Parameters
    ----------
    setup_id : int
        The Openml setup_id
    strict_version: bool (default=True)
        See `flow_to_model` strict_version.

    Returns
    -------
    model
    """
    setup = get_setup(setup_id)
    flow = openml.flows.get_flow(setup.flow_id)

    # instead of using scikit-learns or any other library's "set_params" function, we override the
    # OpenMLFlow objects default parameter value so we can utilize the
    # Extension.flow_to_model() function to reinitialize the flow with the set defaults.
    if setup.parameters is not None:
        for hyperparameter in setup.parameters.values():
            structure = flow.get_structure("flow_id")
            if len(structure[hyperparameter.flow_id]) > 0:
                subflow = flow.get_subflow(structure[hyperparameter.flow_id])
            else:
                subflow = flow
            subflow.parameters[hyperparameter.parameter_name] = hyperparameter.value

    return flow.extension.flow_to_model(flow, strict_version=strict_version)


def _to_dict(flow_id: int, openml_parameter_settings: list[dict[str, Any]]) -> OrderedDict:
    """Convert a flow ID and a list of OpenML parameter settings to
    a dictionary representation that can be serialized to XML.

    Parameters
    ----------
    flow_id : int
        ID of the flow.
    openml_parameter_settings : list[dict[str, Any]]
        A list of OpenML parameter settings.

    Returns
    -------
    OrderedDict
        A dictionary representation of the flow ID and parameter settings.
    """
    # for convenience, this function (ab)uses the run object.
    xml: OrderedDict = OrderedDict()
    xml["oml:run"] = OrderedDict()
    xml["oml:run"]["@xmlns:oml"] = "http://openml.org/openml"
    xml["oml:run"]["oml:flow_id"] = flow_id
    xml["oml:run"]["oml:parameter_setting"] = openml_parameter_settings

    return xml


def _create_setup_from_xml(result_dict: dict) -> OpenMLSetup:
    """Turns an API xml result into a OpenMLSetup object (or dict)"""
    setup_id = int(result_dict["oml:setup_parameters"]["oml:setup_id"])
    flow_id = int(result_dict["oml:setup_parameters"]["oml:flow_id"])

    if "oml:parameter" not in result_dict["oml:setup_parameters"]:
        return OpenMLSetup(setup_id, flow_id, parameters=None)

    xml_parameters = result_dict["oml:setup_parameters"]["oml:parameter"]
    if isinstance(xml_parameters, dict):
        parameters = {
            int(xml_parameters["oml:id"]): _create_setup_parameter_from_xml(xml_parameters),
        }
    elif isinstance(xml_parameters, list):
        parameters = {
            int(xml_parameter["oml:id"]): _create_setup_parameter_from_xml(xml_parameter)
            for xml_parameter in xml_parameters
        }
    else:
        raise ValueError(
            f"Expected None, list or dict, received something else: {type(xml_parameters)!s}",
        )

    return OpenMLSetup(setup_id, flow_id, parameters)


def _create_setup_parameter_from_xml(result_dict: dict[str, str]) -> OpenMLParameter:
    """Create an OpenMLParameter object or a dictionary from an API xml result."""
    return OpenMLParameter(
        input_id=int(result_dict["oml:id"]),
        flow_id=int(result_dict["oml:flow_id"]),
        flow_name=result_dict["oml:flow_name"],
        full_name=result_dict["oml:full_name"],
        parameter_name=result_dict["oml:parameter_name"],
        data_type=result_dict["oml:data_type"],
        default_value=result_dict["oml:default_value"],
        value=result_dict["oml:value"],
    )
