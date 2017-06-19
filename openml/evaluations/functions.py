import xmltodict

from .._api_calls import _perform_api_call
from ..evaluations import OpenMLEvaluation

def list_evaluations(function, task_id):
    """Helper function to parse API calls which are lists of runs"""

    xml_string = _perform_api_call("evaluation/list/funtion/%s/task_id/%d" %(function, task_id))

    evals_dict = xmltodict.parse(xml_string)
    # Minimalistic check if the XML is useful
    if 'oml:evaluations' not in evals_dict:
        raise ValueError('Error in return XML, does not contain "oml:evaluations": %s'
                         % str(evals_dict))

    if isinstance(evals_dict['oml:evaluations']['oml:evaluation'], list):
        evals_list = evals_dict['oml:evaluations']['oml:evaluation']
    elif isinstance(evals_dict['oml:evaluations']['oml:evaluation'], dict):
        evals_list = [evals_dict['oml:runs']['oml:run']]
    else:
        raise TypeError()

    evals = dict()
    for eval_ in evals_list:
        run_id = int(eval_['oml:run_id'])
        evaluation = OpenMLEvaluation(int(eval_['oml:run_id']), int(eval_['task_id']),
                                      int(eval_['oml:setup_id']), int(eval_['oml:flow_id']),
                                      eval_['oml:flow_name'], eval_['oml:data_name'],
                                      eval_['oml:function'], eval_['oml:upload_time'],
                                      float(eval_['oml:value']), eval_['oml:array_data'])
        evals[run_id] = evaluation
    return evaluation

