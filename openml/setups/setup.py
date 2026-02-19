# License: BSD 3-Clause
from __future__ import annotations

from typing import Any

import openml.config
import openml.flows
from openml.base import OpenMLBase


class OpenMLSetup(OpenMLBase):
    """Setup object (a.k.a. Configuration).

    An OpenML Setup corresponds to a flow with a specific parameter configuration.
    It inherits :class:`~openml.base.OpenMLBase` to gain shared functionality
    like tagging, URL access and ``__repr__``.

    Parameters
    ----------
    setup_id : int
        The OpenML setup id.
    flow_id : int
        The flow that it is built upon.
    parameters : dict or None
        The setting of the parameters, mapping parameter id to
        :class:`OpenMLParameter`.
    """

    def __init__(
        self,
        setup_id: int,
        flow_id: int,
        parameters: dict[int, Any] | None,
    ) -> None:
        if not isinstance(setup_id, int):
            raise ValueError("setup id should be int")
        if not isinstance(flow_id, int):
            raise ValueError("flow id should be int")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("parameters should be dict")

        self.setup_id = setup_id
        self.flow_id = flow_id
        self.parameters = parameters

    @property
    def id(self) -> int | None:
        """The setup id, unique among OpenML setups."""
        return self.setup_id

    @classmethod
    def _entity_letter(cls) -> str:
        """Return the letter used in OpenML URLs for setups (``'s'``)."""
        return "s"

    def _get_repr_body_fields(
        self,
    ) -> list[tuple[str, str | int | list[str] | None]]:
        """Return fields shown in :meth:`__repr__`."""
        n_params = len(self.parameters) if self.parameters is not None else float("nan")
        return [
            ("Setup ID", self.setup_id),
            ("Flow ID", self.flow_id),
            ("Flow URL", openml.flows.OpenMLFlow.url_for_id(self.flow_id)),
            ("# of Parameters", n_params),
        ]

    def _to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of this setup."""
        return {
            "setup_id": self.setup_id,
            "flow_id": self.flow_id,
            "parameters": (
                {p.id: p._to_dict() for p in self.parameters.values()}
                if self.parameters is not None
                else None
            ),
        }

    def _parse_publish_response(self, xml_response: dict[str, str]) -> None:
        """Setups cannot be published; raises :class:`NotImplementedError`."""
        raise NotImplementedError("OpenML Setups cannot be published.")

    def publish(self) -> OpenMLBase:
        """Setups cannot be published to the server.

        Raises
        ------
        TypeError
            Always raised — OpenML Setups are created server-side and cannot be
            uploaded by the client.
        """
        raise TypeError(
            "OpenML Setups are created automatically when a run is published "
            "and cannot be uploaded directly."
        )


class OpenMLParameter:
    """Parameter object (used in setup).

    Parameters
    ----------
    input_id : int
        The input id from the openml database
    flow_id : int
        The flow to which this parameter is associated
    flow_name : str
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

    def __init__(
        self,
        input_id: int,
        flow_id: int,
        flow_name: str,
        full_name: str,
        parameter_name: str,
        data_type: str,
        default_value: str,
        value: str,
    ) -> None:
        self.input_id = input_id
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.full_name = full_name
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value
        self.value = value
        # Map input_id to id for backward compatibility
        self.id = self.input_id

    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": self.input_id,
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "full_name": self.full_name,
            "parameter_name": self.parameter_name,
            "data_type": self.data_type,
            "default_value": self.default_value,
            "value": self.value,
        }

    def __repr__(self) -> str:
        header = "OpenML Parameter"
        header = f"{header}\n{'=' * len(header)}\n"

        fields: dict[str, Any] = {
            "ID": self.id,
            "Flow ID": self.flow_id,
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
