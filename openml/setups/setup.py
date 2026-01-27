# License: BSD 3-Clause
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import openml.config
import openml.flows


@dataclass
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

    setup_id: int
    flow_id: int
    parameters: dict[int, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.setup_id, int):
            raise ValueError("setup id should be int")

        if not isinstance(self.flow_id, int):
            raise ValueError("flow id should be int")

        if self.parameters is not None and not isinstance(self.parameters, dict):
            raise ValueError("parameters should be dict")

    def _to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "flow_id": self.flow_id,
            "parameters": {p.id: p._to_dict() for p in self.parameters.values()}
            if self.parameters is not None
            else None,
        }

    def __repr__(self) -> str:
        header = "OpenML Setup"
        header = f"{header}\n{'=' * len(header)}\n"

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


@dataclass
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

    input_id: int
    flow_id: int
    flow_name: str
    full_name: str
    parameter_name: str
    data_type: str
    default_value: str
    value: str

    def __post_init__(self) -> None:
        # Map input_id to id for backward compatibility
        self.id = self.input_id

    def _to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Replaces input_id with id for backward compatibility
        result["id"] = result.pop("input_id")
        return result

    def __repr__(self) -> str:
        header = "OpenML Parameter"
        header = f"{header}\n{'=' * len(header)}\n"

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
        indent = f"{' ' * 2}|{'_' * 2}"
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
