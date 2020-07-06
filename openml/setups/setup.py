# License: BSD 3-Clause

import openml.config


class OpenMLSetup(object):
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

    def __init__(self, setup_id, flow_id, parameters):
        if not isinstance(setup_id, int):
            raise ValueError("setup id should be int")
        if not isinstance(flow_id, int):
            raise ValueError("flow id should be int")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError("parameters should be dict")

        self.setup_id = setup_id
        self.flow_id = flow_id
        self.parameters = parameters

    def __repr__(self):
        header = "OpenML Setup"
        header = "{}\n{}\n".format(header, "=" * len(header))

        fields = {
            "Setup ID": self.setup_id,
            "Flow ID": self.flow_id,
            "Flow URL": openml.flows.OpenMLFlow.url_for_id(self.flow_id),
            "# of Parameters": len(self.parameters),
        }

        # determines the order in which the information will be printed
        order = ["Setup ID", "Flow ID", "Flow URL", "# of Parameters"]
        fields = [(key, fields[key]) for key in order if key in fields]

        longest_field_name_length = max(len(name) for name, value in fields)
        field_line_format = "{{:.<{}}}: {{}}".format(longest_field_name_length)
        body = "\n".join(field_line_format.format(name, value) for name, value in fields)
        return header + body


class OpenMLParameter(object):
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

    def __init__(
        self,
        input_id,
        flow_id,
        flow_name,
        full_name,
        parameter_name,
        data_type,
        default_value,
        value,
    ):
        self.id = input_id
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.full_name = full_name
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value
        self.value = value

    def __repr__(self):
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
        parameter_data_type = "{}Data Type".format(indent)
        fields[parameter_data_type] = self.data_type
        parameter_default = "{}Default".format(indent)
        fields[parameter_default] = self.default_value
        parameter_value = "{}Value".format(indent)
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
        fields = [(key, fields[key]) for key in order if key in fields]

        longest_field_name_length = max(len(name) for name, value in fields)
        field_line_format = "{{:.<{}}}: {{}}".format(longest_field_name_length)
        body = "\n".join(field_line_format.format(name, value) for name, value in fields)
        return header + body
