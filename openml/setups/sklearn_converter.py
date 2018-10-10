import openml
from openml.flows import OpenMLFlow
from openml.setups import OpenMLParameter


def openml_param_name_to_sklearn(openml_parameter, flow):
    """
    Converts the name of an OpenMLParameter into the sklean name, given a flow.

    Parameters
    ----------
    openml_parameter: OpenMLParameter
        The parameter under consideration

    flow: OpenMLFlow
        The flow that provides context.

    Returns
    -------
    sklearn_parameter_name: str
        The name the parameter will have once used in scikit-learn
    """
    if not isinstance(openml_parameter, OpenMLParameter):
        raise ValueError('openml_parameter should be an instance of '
                         'OpenMLParameter')
    if not isinstance(flow, OpenMLFlow):
        raise ValueError('flow should be an instance of OpenMLFlow')

    flow_structure = openml.flows.flow_structure(flow, 'name')
    if openml_parameter.flow_name not in flow_structure:
        raise ValueError('Obtained OpenMLParameter and OpenMLFlow do not '
                         'correspond. ')

    return '__'.join(flow_structure[openml_parameter.flow_name] +
                     [openml_parameter.parameter_name])


def sklearn_param_name_to_openml(sklearn_parameter_name, flow):
    """
    Converts the name of a sklearn parameter into the name that it would have
    in the OpenMLParameter, given a flow.
    The flow needs to be downloaded from the server, such that the flow.version
    field is filled.

    Parameters
    ----------
    sklearn_parameter_name: str
        The parameter under consideration

    flow: OpenMLFlow
        The flow that provides context.

    Returns
    -------
    openml_parameter_name: str
        The full name that this parameter will take when retrieved from an
        OpenMLParameter object from the server
    """
    if not isinstance(flow, OpenMLFlow):
        raise ValueError('flow should be an instance of OpenMLFlow')
    splitted = sklearn_parameter_name.split('__')
    subflow = flow.get_subflow(splitted[0:-1])
    if subflow.flow_id is None or subflow.version is None:
        raise ValueError('For this fn, OpenMLFlow should be downloaded from '
                         'the server, rather than being initiated locally. ')
    return '%s(%s)_%s' % (subflow.name, subflow.version, splitted[-1])
