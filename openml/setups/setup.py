
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
            raise ValueError('setup id should be int')
        if not isinstance(flow_id, int):
            raise ValueError('flow id should be int')
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError('parameters should be dict')

        self.setup_id = setup_id
        self.flow_id = flow_id
        self.parameters = parameters


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
    def __init__(self, input_id, flow_id, flow_name, full_name, parameter_name,
                 data_type, default_value, value):
        self.id = input_id
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.full_name = full_name
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value
        self.value = value
