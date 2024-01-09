# License: BSD 3-Clause
from __future__ import annotations

import warnings
from collections import OrderedDict
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
            return _create_setup_from_xml(setup_xml, output_format="object")  # type: ignore

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
        url_suffix = "/setup/%d" % setup_id
        setup_xml = openml._api_calls._perform_api_call(url_suffix, "get")
        with setup_file.open("w", encoding="utf8") as fh:
            fh.write(setup_xml)

    result_dict = xmltodict.parse(setup_xml)
    return _create_setup_from_xml(result_dict, output_format="object")  # type: ignore


def list_setups(  # noqa: PLR0913
    offset: int | None = None,
    size: int | None = None,
    flow: int | None = None,
    tag: str | None = None,
    setup: Iterable[int] | None = None,
    output_format: Literal["object", "dict", "dataframe"] = "object",
) -> dict | pd.DataFrame:
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
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict or dataframe
    """
    if output_format not in ["dataframe", "dict", "object"]:
        raise ValueError(
            "Invalid output format selected. " "Only 'dict', 'object', or 'dataframe' applicable.",
        )

    # TODO: [0.15]
    if output_format == "dict":
        msg = (
            "Support for `output_format` of 'dict' will be removed in 0.15. "
            "To ensure your code will continue to work, "
            "use `output_format`='dataframe' or `output_format`='object'."
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=2)

    batch_size = 1000  # batch size for setups is lower
    return openml.utils._list_all(  # type: ignore
        list_output_format=output_format,  # type: ignore
        listing_call=_list_setups,
        offset=offset,
        size=size,
        flow=flow,
        tag=tag,
        setup=setup,
        batch_size=batch_size,
    )


def _list_setups(
    setup: Iterable[int] | None = None,
    output_format: Literal["dict", "dataframe", "object"] = "object",
    **kwargs: Any,
) -> dict[int, dict] | pd.DataFrame | dict[int, OpenMLSetup]:
    """
    Perform API call `/setup/list/{filters}`

    Parameters
    ----------
    The setup argument that is a list is separated from the single value
    filters which are put into the kwargs.

    setup : list(int), optional

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
        - If 'object' the output is a dict of OpenMLSetup objects

    kwargs: dict, optional
        Legal filter operators: flow, setup, limit, offset, tag.

    Returns
    -------
    dict or dataframe or list[OpenMLSetup]
    """
    api_call = "setup/list"
    if setup is not None:
        api_call += "/setup/%s" % ",".join([str(int(i)) for i in setup])
    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += f"/{operator}/{value}"

    return __list_setups(api_call=api_call, output_format=output_format)


def __list_setups(
    api_call: str, output_format: Literal["dict", "dataframe", "object"] = "object"
) -> dict[int, dict] | pd.DataFrame | dict[int, OpenMLSetup]:
    """Helper function to parse API calls which are lists of setups"""
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    setups_dict = xmltodict.parse(xml_string, force_list=("oml:setup",))
    openml_uri = "http://openml.org/openml"
    # Minimalistic check if the XML is useful
    if "oml:setups" not in setups_dict:
        raise ValueError(
            'Error in return XML, does not contain "oml:setups":' " %s" % str(setups_dict),
        )

    if "@xmlns:oml" not in setups_dict["oml:setups"]:
        raise ValueError(
            "Error in return XML, does not contain "
            '"oml:setups"/@xmlns:oml: %s' % str(setups_dict),
        )

    if setups_dict["oml:setups"]["@xmlns:oml"] != openml_uri:
        raise ValueError(
            "Error in return XML, value of  "
            '"oml:seyups"/@xmlns:oml is not '
            f'"{openml_uri}": {setups_dict!s}',
        )

    assert isinstance(setups_dict["oml:setups"]["oml:setup"], list), type(setups_dict["oml:setups"])

    setups = {}
    for setup_ in setups_dict["oml:setups"]["oml:setup"]:
        # making it a dict to give it the right format
        current = _create_setup_from_xml(
            {"oml:setup_parameters": setup_},
            output_format=output_format,
        )
        if output_format == "object":
            setups[current.setup_id] = current  # type: ignore
        else:
            setups[current["setup_id"]] = current  # type: ignore

    if output_format == "dataframe":
        setups = pd.DataFrame.from_dict(setups, orient="index")

    return setups


def initialize_model(setup_id: int) -> Any:
    """
    Initialized a model based on a setup_id (i.e., using the exact
    same parameter settings)

    Parameters
    ----------
    setup_id : int
        The Openml setup_id

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

    return flow.extension.flow_to_model(flow)


def _to_dict(
    flow_id: int, openml_parameter_settings: list[OpenMLParameter] | list[dict[str, Any]]
) -> OrderedDict:
    """Convert a flow ID and a list of OpenML parameter settings to
    a dictionary representation that can be serialized to XML.

    Parameters
    ----------
    flow_id : int
        ID of the flow.
    openml_parameter_settings : List[OpenMLParameter]
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


def _create_setup_from_xml(
    result_dict: dict, output_format: Literal["dict", "dataframe", "object"] = "object"
) -> OpenMLSetup | dict[str, int | dict[int, Any] | None]:
    """Turns an API xml result into a OpenMLSetup object (or dict)"""
    if output_format in ["dataframe", "dict"]:
        _output_format: Literal["dict", "object"] = "dict"
    elif output_format == "object":
        _output_format = "object"
    else:
        raise ValueError(
            f"Invalid output format selected: {output_format}"
            "Only 'dict', 'object', or 'dataframe' applicable.",
        )

    setup_id = int(result_dict["oml:setup_parameters"]["oml:setup_id"])
    flow_id = int(result_dict["oml:setup_parameters"]["oml:flow_id"])
    if "oml:parameter" not in result_dict["oml:setup_parameters"]:
        parameters = None
    else:
        parameters = {}
        # basically all others
        xml_parameters = result_dict["oml:setup_parameters"]["oml:parameter"]
        if isinstance(xml_parameters, dict):
            oml_id = int(xml_parameters["oml:id"])
            parameters[oml_id] = _create_setup_parameter_from_xml(
                result_dict=xml_parameters,
                output_format=_output_format,
            )
        elif isinstance(xml_parameters, list):
            for xml_parameter in xml_parameters:
                oml_id = int(xml_parameter["oml:id"])
                parameters[oml_id] = _create_setup_parameter_from_xml(
                    result_dict=xml_parameter,
                    output_format=_output_format,
                )
        else:
            raise ValueError(
                "Expected None, list or dict, received "
                "something else: %s" % str(type(xml_parameters)),
            )

    if _output_format in ["dataframe", "dict"]:
        return {"setup_id": setup_id, "flow_id": flow_id, "parameters": parameters}
    return OpenMLSetup(setup_id, flow_id, parameters)


def _create_setup_parameter_from_xml(
    result_dict: dict[str, str], output_format: Literal["object", "dict"] = "object"
) -> dict[str, int | str] | OpenMLParameter:
    """Create an OpenMLParameter object or a dictionary from an API xml result."""
    if output_format == "object":
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

    # FIXME: likely we want to crash here if unknown output_format but not backwards compatible
    # output_format == "dict" case,
    return {
        "input_id": int(result_dict["oml:id"]),
        "flow_id": int(result_dict["oml:flow_id"]),
        "flow_name": result_dict["oml:flow_name"],
        "full_name": result_dict["oml:full_name"],
        "parameter_name": result_dict["oml:parameter_name"],
        "data_type": result_dict["oml:data_type"],
        "default_value": result_dict["oml:default_value"],
        "value": result_dict["oml:value"],
    }
