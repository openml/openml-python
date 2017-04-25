
class OpenMLSetup(object):
    """Setup object (a.k.a. Configuration).

       Parameters
       ----------
       flow_id : int
            The flow that it is build upon
        parameters : dict
            The setting of the parameters
           """

    def __init__(self, flow_id, parameters):
        self.flow_id = flow_id
        self.parameters = parameters


class OpenMLParameter(object):
    """Parameter object (used in setup).

       Parameters
       ----------
       flow_id : int
            The flow that it is build upon
        parameters : dict
            The setting of the parameters
    """
    def __init__(self, id, flow_id, full_name, parameter_name, data_type, default_value, value):
        self.id = id
        self.flow_id = flow_id
        self.full_name = full_name
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value
        self.value = value
