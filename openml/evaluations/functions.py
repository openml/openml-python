# License: BSD 3-Clause

import json
import xmltodict
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict
import collections

import openml.utils
import openml._api_calls
from ..evaluations import OpenMLEvaluation
import openml


def list_evaluations(
    function: str,
    offset: Optional[int] = None,
    size: Optional[int] = None,
    task: Optional[List] = None,
    setup: Optional[List] = None,
    flow: Optional[List] = None,
    run: Optional[List] = None,
    uploader: Optional[List] = None,
    tag: Optional[str] = None,
    study: Optional[int] = None,
    per_fold: Optional[bool] = None,
    sort_order: Optional[str] = None,
    output_format: str = 'object'
) -> Union[Dict, pd.DataFrame]:
    """
    List all run-evaluation pairs matching all of the given filters.
    (Supports large amount of results)

    Parameters
    ----------
    function : str
        the evaluation function. e.g., predictive_accuracy
    offset : int, optional
        the number of runs to skip, starting from the first
    size : int, optional
        the maximum number of runs to show

    task : list, optional

    setup: list, optional

    flow : list, optional

    run : list, optional

    uploader : list, optional

    tag : str, optional

    study : int, optional

    per_fold : bool, optional

    sort_order : str, optional
       order of sorting evaluations, ascending ("asc") or descending ("desc")

    output_format: str, optional (default='object')
        The parameter decides the format of the output.
        - If 'object' the output is a dict of OpenMLEvaluation objects
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict or dataframe
    """
    if output_format not in ['dataframe', 'dict', 'object']:
        raise ValueError("Invalid output format selected. "
                         "Only 'object', 'dataframe', or 'dict' applicable.")

    per_fold_str = None
    if per_fold is not None:
        per_fold_str = str(per_fold).lower()

    return openml.utils._list_all(output_format=output_format,
                                  listing_call=_list_evaluations,
                                  function=function,
                                  offset=offset,
                                  size=size,
                                  task=task,
                                  setup=setup,
                                  flow=flow,
                                  run=run,
                                  uploader=uploader,
                                  tag=tag,
                                  study=study,
                                  sort_order=sort_order,
                                  per_fold=per_fold_str)


def _list_evaluations(
    function: str,
    task: Optional[List] = None,
    setup: Optional[List] = None,
    flow: Optional[List] = None,
    run: Optional[List] = None,
    uploader: Optional[List] = None,
    study: Optional[int] = None,
    sort_order: Optional[str] = None,
    output_format: str = 'object',
    **kwargs
) -> Union[Dict, pd.DataFrame]:
    """
    Perform API call ``/evaluation/function{function}/{filters}``

    Parameters
    ----------
    The arguments that are lists are separated from the single value
    ones which are put into the kwargs.

    function : str
        the evaluation function. e.g., predictive_accuracy

    task : list, optional

    setup: list, optional

    flow : list, optional

    run : list, optional

    uploader : list, optional

    study : int, optional

    kwargs: dict, optional
        Legal filter operators: tag, limit, offset.

    sort_order : str, optional
        order of sorting evaluations, ascending ("asc") or descending ("desc")

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict of objects, or dataframe
    """

    api_call = "evaluation/list/function/%s" % function
    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)
    if task is not None:
        api_call += "/task/%s" % ','.join([str(int(i)) for i in task])
    if setup is not None:
        api_call += "/setup/%s" % ','.join([str(int(i)) for i in setup])
    if flow is not None:
        api_call += "/flow/%s" % ','.join([str(int(i)) for i in flow])
    if run is not None:
        api_call += "/run/%s" % ','.join([str(int(i)) for i in run])
    if uploader is not None:
        api_call += "/uploader/%s" % ','.join([str(int(i)) for i in uploader])
    if study is not None:
        api_call += "/study/%d" % study
    if sort_order is not None:
        api_call += "/sort_order/%s" % sort_order

    return __list_evaluations(api_call, output_format=output_format)


def __list_evaluations(api_call, output_format='object'):
    """Helper function to parse API calls which are lists of runs"""
    xml_string = openml._api_calls._perform_api_call(api_call, 'get')
    evals_dict = xmltodict.parse(xml_string, force_list=('oml:evaluation',))
    # Minimalistic check if the XML is useful
    if 'oml:evaluations' not in evals_dict:
        raise ValueError('Error in return XML, does not contain '
                         '"oml:evaluations": %s' % str(evals_dict))

    assert type(evals_dict['oml:evaluations']['oml:evaluation']) == list, \
        type(evals_dict['oml:evaluations'])

    evals = collections.OrderedDict()
    uploader_ids = list(set([eval_['oml:uploader'] for eval_ in
                             evals_dict['oml:evaluations']['oml:evaluation']]))
    api_users = "user/list/user_id/" + ','.join(uploader_ids)
    xml_string_user = openml._api_calls._perform_api_call(api_users, 'get')
    users = xmltodict.parse(xml_string_user, force_list=('oml:user',))
    user_dict = {user['oml:id']: user['oml:username'] for user in users['oml:users']['oml:user']}
    for eval_ in evals_dict['oml:evaluations']['oml:evaluation']:
        run_id = int(eval_['oml:run_id'])
        value = None
        values = None
        array_data = None
        if 'oml:value' in eval_:
            value = float(eval_['oml:value'])
        if 'oml:values' in eval_:
            values = json.loads(eval_['oml:values'])
        if 'oml:array_data' in eval_:
            array_data = eval_['oml:array_data']

        if output_format == 'object':
            evals[run_id] = OpenMLEvaluation(int(eval_['oml:run_id']),
                                             int(eval_['oml:task_id']),
                                             int(eval_['oml:setup_id']),
                                             int(eval_['oml:flow_id']),
                                             eval_['oml:flow_name'],
                                             int(eval_['oml:data_id']),
                                             eval_['oml:data_name'],
                                             eval_['oml:function'],
                                             eval_['oml:upload_time'],
                                             int(eval_['oml:uploader']),
                                             user_dict[eval_['oml:uploader']],
                                             value, values, array_data)
        else:
            # for output_format in ['dict', 'dataframe']
            evals[run_id] = {'run_id': int(eval_['oml:run_id']),
                             'task_id': int(eval_['oml:task_id']),
                             'setup_id': int(eval_['oml:setup_id']),
                             'flow_id': int(eval_['oml:flow_id']),
                             'flow_name': eval_['oml:flow_name'],
                             'data_id': int(eval_['oml:data_id']),
                             'data_name': eval_['oml:data_name'],
                             'function': eval_['oml:function'],
                             'upload_time': eval_['oml:upload_time'],
                             'uploader': int(eval_['oml:uploader']),
                             'uploader_name': user_dict[eval_['oml:uploader']],
                             'value': value,
                             'values': values,
                             'array_data': array_data}

    if output_format == 'dataframe':
        rows = [value for key, value in evals.items()]
        evals = pd.DataFrame.from_records(rows, columns=rows[0].keys())
    return evals


def list_evaluation_measures() -> List[str]:
    """ Return list of evaluation measures available.

    The function performs an API call to retrieve the entire list of
    evaluation measures that are available.

    Returns
    -------
    list

    """
    api_call = "evaluationmeasure/list"
    xml_string = openml._api_calls._perform_api_call(api_call, 'get')
    qualities = xmltodict.parse(xml_string, force_list=('oml:measures'))
    # Minimalistic check if the XML is useful
    if 'oml:evaluation_measures' not in qualities:
        raise ValueError('Error in return XML, does not contain '
                         '"oml:evaluation_measures"')
    if not isinstance(qualities['oml:evaluation_measures']['oml:measures'][0]['oml:measure'],
                      list):
        raise TypeError('Error in return XML, does not contain '
                        '"oml:measure" as a list')
    qualities = qualities['oml:evaluation_measures']['oml:measures'][0]['oml:measure']
    return qualities


def list_evaluations_setups(
        function: str,
        offset: Optional[int] = None,
        size: Optional[int] = None,
        task: Optional[List] = None,
        setup: Optional[List] = None,
        flow: Optional[List] = None,
        run: Optional[List] = None,
        uploader: Optional[List] = None,
        tag: Optional[str] = None,
        per_fold: Optional[bool] = None,
        sort_order: Optional[str] = None,
        output_format: str = 'dataframe',
        parameters_in_separate_columns: bool = False
) -> Union[Dict, pd.DataFrame]:
    """
    List all run-evaluation pairs matching all of the given filters
    and their hyperparameter settings.

    Parameters
    ----------
    function : str
        the evaluation function. e.g., predictive_accuracy
    offset : int, optional
        the number of runs to skip, starting from the first
    size : int, optional
        the maximum number of runs to show
    task : list[int], optional
        the list of task IDs
    setup: list[int], optional
        the list of setup IDs
    flow : list[int], optional
        the list of flow IDs
    run : list[int], optional
        the list of run IDs
    uploader : list[int], optional
        the list of uploader IDs
    tag : str, optional
        filter evaluation based on given tag
    per_fold : bool, optional
    sort_order : str, optional
       order of sorting evaluations, ascending ("asc") or descending ("desc")
    output_format: str, optional (default='dataframe')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    parameters_in_separate_columns: bool, optional (default= False)
        Returns hyperparameters in separate columns if set to True.
        Valid only for a single flow


    Returns
    -------
    dict or dataframe with hyperparameter settings as a list of tuples.
    """
    if parameters_in_separate_columns and (flow is None or len(flow) != 1):
        raise ValueError("Can set parameters_in_separate_columns to true "
                         "only for single flow_id")

    # List evaluations
    evals = list_evaluations(function=function, offset=offset, size=size, run=run, task=task,
                             setup=setup, flow=flow, uploader=uploader, tag=tag,
                             per_fold=per_fold, sort_order=sort_order, output_format='dataframe')
    # List setups
    # list_setups by setup id does not support large sizes (exceeds URL length limit)
    # Hence we split the list of unique setup ids returned by list_evaluations into chunks of size N
    df = pd.DataFrame()
    if len(evals) != 0:
        N = 100  # size of section
        length = len(evals['setup_id'].unique())  # length of the array we want to split
        # array_split - allows indices_or_sections to not equally divide the array
        # array_split -length % N sub-arrays of size length//N + 1 and the rest of size length//N.
        setup_chunks = np.array_split(ary=evals['setup_id'].unique(),
                                      indices_or_sections=((length - 1) // N) + 1)
        setups = pd.DataFrame()
        for setup in setup_chunks:
            result = pd.DataFrame(openml.setups.list_setups(setup=setup, output_format='dataframe'))
            result.drop('flow_id', axis=1, inplace=True)
            # concat resulting setup chunks into single datframe
            setups = pd.concat([setups, result], ignore_index=True)
        parameters = []
        # Convert parameters of setup into list of tuples of (hyperparameter, value)
        for parameter_dict in setups['parameters']:
            if parameter_dict is not None:
                parameters.append({param['full_name']: param['value']
                                   for param in parameter_dict.values()})
            else:
                parameters.append({})
        setups['parameters'] = parameters
        # Merge setups with evaluations
        df = pd.merge(evals, setups, on='setup_id', how='left')

    if parameters_in_separate_columns:
        df = pd.concat([df.drop('parameters', axis=1),
                        df['parameters'].apply(pd.Series)], axis=1)

    if output_format == 'dataframe':
        return df
    else:
        return df.to_dict(orient='index')
