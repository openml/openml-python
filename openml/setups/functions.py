# License: BSD 3-Clause
from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

import openml
import openml.exceptions
import openml.utils
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

    return openml._backend.setup.exists(flow, openml_param_settings)


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
    setup: OpenMLSetup = openml._backend.setup.get(setup_id=setup_id)
    return setup


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

    listing_call = partial(openml._backend.setup.list, flow=flow, tag=tag, setup=setup)
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


def _create_setup(result_dict: dict) -> OpenMLSetup:
    """Turns an API xml result into a OpenMLSetup object (or dict)"""
    return openml._backend.setup._create_setup(result_dict)
