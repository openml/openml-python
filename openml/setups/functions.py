# License: BSD 3-Clause
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import xmltodict

import openml
import openml.exceptions
import openml.utils
from openml import config
from openml._api import api_context
from openml.flows import OpenMLFlow, flow_exists

if TYPE_CHECKING:
    from .setup import OpenMLSetup


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
            f"Setup file for setup id {setup_id} not cached",
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

    listing_call = partial(api_context.backend.setups.list, flow=flow, tag=tag, setup=setup)
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
    return api_context.backend.setups._create_setup(result_dict)
