
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

    def __str__(self):
        object_dict = self.__dict__
        output_str = ''
        header = 'OpenML Setup'
        header = '{}\n{}\n'.format(header, '=' * len(header))
        setup = '{:.<15}: {}\n'.format("Setup ID", object_dict['setup_id'])
        flow = '{:.<15}: {}\n'.format("Flow ID", object_dict['flow_id'])
        url = 'https://www.openml.org/f/' + str(object_dict['flow_id'])
        flow = flow + '{:.<15}: {}\n'.format("Flow URL", url)
        params = '{:.<15}: {}\n'.format("# of Parameters", len(object_dict['parameters']))
        output_str = '\n' + header + setup + flow + params + '\n'
        return(output_str)


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

    def __str__(self):
        object_dict = self.__dict__
        output_str = ''
        header = 'OpenML Parameter'
        header = '{}\n{}\n'.format(header, '=' * len(header))
        id = '{:.<18}: {}\n'.format("ID", object_dict['id'])
        flow = '{:.<18}: {}\n'.format("Flow ID", object_dict['flow_id'])
        flow = flow + '{:.<18}: {}\n'.format("Flow Name", object_dict['flow_name'])
        flow = flow + '{:.<18}: {}\n'.format("Flow Full Name", object_dict['full_name'])
        url = 'https://www.openml.org/f/' + str(object_dict['flow_id'])
        flow = flow + '{:.<18}: {}\n'.format("Flow URL", url)
        filler = " |" + "_" * 2
        params = '{:.<18}: {}\n'.format("Parameter Name", object_dict['parameter_name'])
        params = params + filler + '{:.<14}: {}\n'.format("Data_Type", object_dict['data_type'])
        params = params + filler + '{:.<14}: {}\n'.format("Default", object_dict['default_value'])
        params = params + filler + '{:.<14}: {}\n'.format("Value", object_dict['value'])
        output_str = '\n' + header + id + flow + params + '\n'
        return(output_str)
