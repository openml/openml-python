# License: BSD 3-Clause
from __future__ import annotations

from typing import Any

import openml.config
import openml.flows


class OpenMLSetup:
    """Setup object (a.k.a. Configuration).

    Parameters
    ----------
    setup_id : int
        The OpenML setup id
    flow_id : int
        The flow that it is build upon
    parameters : dict
        The setting of the parameters
    """

    def __init__(self, setup_id: int, flow_id: int, parameters: dict[int, Any] | None):
        if not isinstance(setup_id, int):
            raise ValueError("setup id should be int")

        if not isinstance(flow_id, int):
            raise ValueError("flow id should be int")

        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("parameters should be dict")

        self.setup_id = setup_id
        self.flow_id = flow_id
        self.parameters = parameters

    def __repr__(self) -> str:
        header = "OpenML Setup"
        header = "{}\n{}\n".format(header, "=" * len(header))

        fields = {
            "Setup ID": self.setup_id,
            "Flow ID": self.flow_id,
            "Flow URL": openml.flows.OpenMLFlow.url_for_id(self.flow_id),
            "# of Parameters": (
                len(self.parameters) if self.parameters is not None else float("nan")
            ),
        }

        # determines the order in which the information will be printed
        order = ["Setup ID", "Flow ID", "Flow URL", "# of Parameters"]
        _fields = [(key, fields[key]) for key in order if key in fields]

        longest_field_name_length = max(len(name) for name, _ in _fields)
        field_line_format = f"{{:.<{longest_field_name_length}}}: {{}}"
        body = "\n".join(field_line_format.format(name, value) for name, value in _fields)
        return header + body


class OpenMLParameter:
    """Parameter object (used in setup).

    Parameters
    ----------
    input_id : int
        The input id from the openml database
    flow id : int
        The flow to which this parameter is associated
    flow name : str
        The name of the flow (no version number) to which this parameter
        is associated
    full_name : str
        The name of the flow and parameter combined
    parameter_name : str
        The name of the parameter
    data_type : str
        The datatype of the parameter. generally unused for sklearn flows
    default_value : str
        The default value. For sklearn parameters, this is unknown and a
        default value is selected arbitrarily
    value : str
        If the parameter was set, the value that it was set to.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_id: int,
        flow_id: int,
        flow_name: str,
        full_name: str,
        parameter_name: str,
        data_type: str,
        default_value: str,
        value: str,
    ):
        self.id = input_id
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.full_name = full_name
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value
        self.value = value

    def __repr__(self) -> str:
        header = "OpenML Parameter"
        header = "{}\n{}\n".format(header, "=" * len(header))

        fields = {
            "ID": self.id,
            "Flow ID": self.flow_id,
            # "Flow Name": self.flow_name,
            "Flow Name": self.full_name,
            "Flow URL": openml.flows.OpenMLFlow.url_for_id(self.flow_id),
            "Parameter Name": self.parameter_name,
        }
        # indented prints for parameter attributes
        # indention = 2 spaces + 1 | + 2 underscores
        indent = "{}|{}".format(" " * 2, "_" * 2)
        parameter_data_type = f"{indent}Data Type"
        fields[parameter_data_type] = self.data_type
        parameter_default = f"{indent}Default"
        fields[parameter_default] = self.default_value
        parameter_value = f"{indent}Value"
        fields[parameter_value] = self.value

        # determines the order in which the information will be printed
        order = [
            "ID",
            "Flow ID",
            "Flow Name",
            "Flow URL",
            "Parameter Name",
            parameter_data_type,
            parameter_default,
            parameter_value,
        ]
        _fields = [(key, fields[key]) for key in order if key in fields]

        longest_field_name_length = max(len(name) for name, _ in _fields)
        field_line_format = f"{{:.<{longest_field_name_length}}}: {{}}"
        body = "\n".join(field_line_format.format(name, value) for name, value in _fields)
        return header + body
