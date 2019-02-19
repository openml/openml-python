import xmltodict

from openml.study import OpenMLStudy
import openml._api_calls


def get_study(study_id, knowledge_type=None):
    """
    Retrieves all relevant information of an OpenML study from the server
    Note that some of the (data, tasks, flows, setups) fields can be empty
    (depending on information on the server)
    """
    call_suffix = "study/%s" % str(study_id)
    if knowledge_type is not None:
        call_suffix += "/" + knowledge_type
    xml_string = openml._api_calls._perform_api_call(call_suffix, 'get')
    force_list_tags = (
            'oml:data_id', 'oml:flow_id', 'oml:task_id',
            'oml:setup_id', 'oml:run_id',
            'oml:tag'  # legacy.
    )
    result_dict = xmltodict.parse(xml_string, 
                                  force_list=force_list_tags)['oml:study']
    study_id = int(result_dict['oml:id'])
    alias = result_dict['oml:alias'] if 'oml:alias' in result_dict else None
    main_knowledge_type = result_dict['oml:main_knowledge_type']
    benchmark_suite = result_dict['oml:benchmark_suite'] \
        if 'oml:benchmark_suite' in result_dict else None
    name = result_dict['oml:name']
    description = result_dict['oml:description']
    creation_date = result_dict['oml:creation_date']
    creator = result_dict['oml:creator']
    
    # tags is legacy. remove once no longer needed.
    tags = []
    if 'oml:tag' in result_dict:
        for tag in result_dict['oml:tag']:
            current_tag = {'name': tag['oml:name'],
                           'write_access': tag['oml:write_access']}
            if 'oml:window_start' in tag:
                current_tag['window_start'] = tag['oml:window_start']
            tags.append(current_tag)

    datasets = None
    tasks = None
    flows = None
    setups = None
    runs = None
    
    if 'oml:data' in result_dict:
        datasets = [int(x) for x in result_dict['oml:data']['oml:data_id']]

    if 'oml:tasks' in result_dict:
        tasks = [int(x) for x in result_dict['oml:tasks']['oml:task_id']]

    if 'oml:flows' in result_dict:
        flows = [int(x) for x in result_dict['oml:flows']['oml:flow_id']]

    if 'oml:setups' in result_dict:
        setups = [int(x) for x in result_dict['oml:setups']['oml:setup_id']]
    
    if 'oml:runs' in result_dict:
        runs = [int(x) for x in result_dict['oml:runs']['oml:run_id']]

    study = OpenMLStudy(
            study_id=study_id,
            alias=alias,
            main_knowledge_type=main_knowledge_type,
            benchmark_suite=benchmark_suite,
            name=name, 
            description=description, 
            creation_date=creation_date, 
            creator=creator, 
            tags=tags,
            data=datasets, 
            tasks=tasks, 
            flows=flows, 
            setups=setups,
            runs=runs
    )
    return study


def create_study(alias, benchmark_suite, name, description, run_ids):
    """
    Creates an OpenML study (collection of knowledge types, where the runs
    are the linked entity)
    
    Parameters:
    -----------
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    benchmark_suite : int (optional)
        the benchmark suite (another study) upon which this study is ran.
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    run_ids : list
        a list of run ids associated with this study
    
    Returns:
    --------
    OpenMLStudy
        A local OpenML study object (call publish method to upload to server)
    """
    return OpenMLStudy(
            study_id=None,
            alias=alias,
            main_knowledge_type='run',
            benchmark_suite=benchmark_suite,
            name=name, 
            description=description,
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=None,
            flows=None,
            setups=None,
            runs=run_ids
    )


def create_benchmark_suite(alias, name, description, task_ids):
    """
    Creates an OpenML benchmark suite (collection of knowledge types, where 
    the tasks are the linked entity)
    
    Parameters:
    -----------
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    task_ids : list
        a list of task ids associated with this study
    
    Returns:
    --------
    OpenMLStudy
        A local OpenML study object (call publish method to upload to server)
    """
    return OpenMLStudy(
            study_id=None,
            alias=alias,
            main_knowledge_type='task',
            benchmark_suite=None,
            name=name, 
            description=description, 
            creation_date=None,
            creator=None,
            tags=None,
            data=None,
            tasks=task_ids,
            flows=None,
            setups=None,
            runs=None
    )


def delete_study(study_id):
    """
    Deletes an study from the OpenML server.
    
    Parameters
    ----------
    study_id : int
        OpenML id of the study

    Returns
    -------
    bool
        True iff the deletion was successful. False otherwse
    """
    return openml.utils._delete_entity('study', study_id)


def attach_to_study(study_id, entity_ids):
    """
    Attaches a set of entities to a collection
        - provide run ids of existsing runs if the main knowledge type is
          runs (study)
        - provide task ids of existing tasks if the main knowledge type is
          tasks (benchmark suite)

    Parameters
    ----------
    study_id : int
        OpenML id of the study

    entity_ids : list (int)
        List of entities to link to the collection

    Returns
    -------
    int
        new size of the study (in terms of explicitly linked entities)
    """
    uri = 'study/%d/attach' % study_id
    post_variables = {'ids': ','.join(str(x) for x in entity_ids)}
    result_xml = openml._api_calls._perform_api_call(uri,
                                                     'post',
                                                     post_variables)
    result = xmltodict.parse(result_xml)['oml:study_attach']
    return int(result['oml:linked_entities'])


def detach_from_study(study_id, entity_ids):
    """
    Detaches a set of entities to a collection
        - provide run ids of existsing runs if the main knowledge type is
          runs (study)
        - provide task ids of existing tasks if the main knowledge type is
          tasks (benchmark suite)

    Parameters
    ----------
    study_id : int
        OpenML id of the study

    entity_ids : list (int)
        List of entities to link to the collection

    Returns
    -------
    int
        new size of the study (in terms of explicitly linked entities)
    """
    uri = 'study/%d/detach' % study_id
    post_variables = {'ids': ','.join(str(x) for x in entity_ids)}
    result_xml = openml._api_calls._perform_api_call(uri,
                                                     'post',
                                                     post_variables)
    result = xmltodict.parse(result_xml)['oml:study_detach']
    return int(result['oml:linked_entities'])
