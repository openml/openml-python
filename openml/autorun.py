# Made for a course at Eindhoven University of Technology
# Author: Pieter Gijsbers
# Supervisor: Joaquin Vanschoren

from collections import OrderedDict
# pickleshare 0.5
# liac-arff 2.1.1.dev0
# xmltodict 0.9.2
import xmltodict
import os
import sys
import time
import pickle
import arff


# This can possibly be done by a package such as pyxb, but I could not get
# it to work properly.
def construct_description_dictionary(taskid, flow_id, setup_string, parameter_settings, tags):
    """ Creates a dictionary corresponding to the desired xml desired by openML

    Parameters
    ----------
    taskid : int
        the identifier of the task
    setup_string : string
        a CLI string which can invoke the learning with the correct parameter settings
    parameter_settings : array of dicts
        each dict containing keys name, value and component, one per parameter setting
    tags : array of strings
        information that give a description of the run, must conform to regex "([a-zA-Z0-9_\-\.])+"

    Returns
    -------
    result : an array with version information of the above packages
    """
    description = OrderedDict()
    description['oml:run'] = OrderedDict()
    description['oml:run']['@xmlns:oml'] = 'http://openml.org/openml'
    description['oml:run']['oml:task_id'] = taskid

    description['oml:run']['oml:flow_id'] = flow_id

    params = []
    for k, v in parameter_settings.items():
        param_dict = OrderedDict()
        param_dict['oml:name'] = k
        param_dict['oml:value'] = ('None' if v is None else v)
        params.append(param_dict)

    description['oml:run']['oml:parameter_setting'] = params
    description['oml:run']['oml:tag'] = tags  # Tags describing the run
    #description['oml:run']['oml:output_data'] = 0;
    # all data that was output of this run, which can be evaluation scores
    # (though those are also calculated serverside)
    # must be of special data type
    return description


def get_version_information():
    """Gets versions of python, sklearn, numpy and scipy, returns them in an array,

    Returns
    -------
    result : an array with version information of the above packages
    """
    import sklearn
    import scipy
    import numpy

    major, minor, micro, _, _ = sys.version_info
    python_version = 'Python_{}.'.format(
        ".".join([str(major), str(minor), str(micro)]))
    sklearn_version = 'Sklearn_{}.'.format(sklearn.__version__)
    numpy_version = 'NumPy_{}.'.format(numpy.__version__)
    scipy_version = 'SciPy_{}.'.format(scipy.__version__)

    return [python_version, sklearn_version, numpy_version, scipy_version]


def generate_arff(arff_datacontent, task):
    """Generates an arff

    Parameters
    ----------
    arff_datacontent : list
        a list of lists containing, in order:
                        - repeat (int)
                        - fold (int)
                        - test index (int)
                        - predictions per task label (float)
                        - predicted class label (string)
                        - actual class label (string)
    task : Task
        the OpenML task for which the run is done
    """
    run_environment = get_version_information(
    ) + [time.strftime("%c")] + ['Created by openml_run()']
    class_labels = task.class_labels

    arff_dict = {}
    arff_dict['attributes'] = [('repeat', 'NUMERIC'),  # lowercase 'numeric' gives an error
                               ('fold', 'NUMERIC'),
                               ('row_id', 'NUMERIC')] + \
        [('confidence.' + class_labels[i], 'NUMERIC') for i in range(len(class_labels))] +\
        [('prediction', class_labels),
         ('correct', class_labels)]
    arff_dict['data'] = arff_datacontent
    arff_dict['description'] = "\n".join(run_environment)
    arff_dict['relation'] = 'openml_task_' + str(task.task_id) + '_predictions'
    return arff_dict


def create_description_xml(taskid, flow_id, classifier):
    run_environment = get_version_information()
    setup_string = ''  # " ".join(sys.argv);

    parameter_settings = classifier.get_params()
    # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
    # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
    well_formatted_time = time.strftime("%c").replace(
        ' ', '_').replace('/', '-').replace(':', '.')
    tags = run_environment + [well_formatted_time] + ['openml_run'] + \
        [classifier.__module__ + "." + classifier.__class__.__name__]
    description = construct_description_dictionary(
        taskid, flow_id, setup_string, parameter_settings, tags)
    description_xml = xmltodict.unparse(description, pretty=True)
    return description_xml


def generate_flow_xml(classifier):
    import sklearn
    flow_dict = OrderedDict()
    flow_dict['oml:flow'] = OrderedDict()
    flow_dict['oml:flow']['@xmlns:oml'] = 'http://openml.org/openml'
    flow_dict['oml:flow']['oml:name'] = classifier.__module__ + \
        "." + classifier.__class__.__name__
    flow_dict['oml:flow'][
        'oml:external_version'] = 'Tsklearn_' + sklearn.__version__
    flow_dict['oml:flow']['oml:description'] = 'Flow generated by openml_run'

    clf_params = classifier.get_params()
    flow_parameters = []
    for k, v in clf_params.items():
        # data_type, default_value, description, recommendedRange
        # type = v.__class__.__name__    Not using this because it doesn't conform standards
        # eg. int instead of integer
        param_dict = {'oml:name': k}
        flow_parameters.append(param_dict)

    flow_dict['oml:flow']['oml:parameter'] = flow_parameters

    flow_xml = xmltodict.unparse(flow_dict, pretty=True)

    # A flow may not be uploaded with the encoding specification..
    flow_xml = flow_xml.split('\n', 1)[-1]
    return flow_xml


def ensure_flow_exists(connector, classifier):
    """
    First checks if a flow exists for the given classifier.
    If it does, then it will return the corresponding flow id.
    If it does not, then it will create a flow, and return the flow id
    of the newly created flow.
    """
    import sklearn
    flow_name = classifier.__module__ + "." + classifier.__class__.__name__
    flow_version = 'Tsklearn_' + sklearn.__version__
    _, _, flow_id = connector.check_flow_exists(flow_name, flow_version)

    if int(flow_id) == -1:
        # flow does not exist yet, create it
        flow_xml = generate_flow_xml(classifier)
        file_name = classifier.__class__.__name__ + '_flow.xml'
        abs_file_path = os.path.abspath(file_name)
        with open(abs_file_path, 'w') as fh:
            fh.write(flow_xml)

        flow_binary = open(abs_file_path, 'rb').read()
        return_code, response_xml = connector.upload_flow(flow_binary)

        response_dict = xmltodict.parse(response_xml)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        return int(flow_id)

    elif int(flow_id) == -2:
        # Something went wrong retrieving the flow
        raise NotImplementedError('Error handling - check_flow_exists fail')

    return int(flow_id)


def openml_run(task, classifier):
    """Performs a CV run on the dataset of the given task, using the split.

    Parameters
    ----------
    connector : APIConnector
        Openml APIConnector which is used to download the OpenML Task and Dataset
    taskid : int
        The integer identifier of the task to run the classifier on
    classifier : sklearn classifier
        a classifier which has a function fit(X,Y) and predict(X),
        all supervised estimators of scikit learn follow this definition of a classifier [1]
        [1](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)


    Returns
    -------
    classifier : sklearn classifier
        the classifier, trained on the whole dataset
    arff-dict : dict
        a dictionary with an 'attributes' and 'data' entry for an arff file
    """
    flow_id = ensure_flow_exists(task.api_connector, classifier)
    if(flow_id < 0):
        print("No flow")
        return 0, 2
    print(flow_id)

    runname = "t" + str(task.task_id) + "_" + classifier.__class__.__name__
    arff_datacontent = []

    dataset = task.get_dataset()
    X, Y = dataset.get_dataset(target=task.target_feature)

    class_labels = task.class_labels
    if(class_labels is None):
        raise ValueError('The task has no class labels. This method currently '
                         'only works for tasks with class labels.')

    train_times = []

    rep_no = 0
    for rep in task.iterate_repeats():
        fold_no = 0
        for fold in rep:
            train_indices, test_indices = fold
            trainX = X[train_indices]
            trainY = Y[train_indices]
            testX = X[test_indices]
            testY = Y[test_indices]

            start_time = time.time()
            classifier.fit(trainX, trainY)
            ProbaY = classifier.predict_proba(testX)
            PredY = classifier.predict(testX)
            end_time = time.time()

            train_times.append(end_time - start_time)

            for i in range(0, len(test_indices)):
                arff_line = [rep_no, fold_no, test_indices[i],
                             class_labels[PredY[i]], class_labels[testY[i]]]
                arff_line[3:3] = ProbaY[i]
                arff_datacontent.append(arff_line)

            fold_no = fold_no + 1
        rep_no = rep_no + 1

    # Generate a dictionary which represents the arff file (with predictions)
    arff_dict = generate_arff(arff_datacontent, task)
    predictions_path = runname + '.arff'
    with open(predictions_path, 'w') as fh:
        arff.dump(arff_dict, fh)

    description_xml = create_description_xml(task.task_id, flow_id, classifier)
    description_path = runname + '.xml'
    with open(description_path, 'w') as fh:
        fh.write(description_xml)

    # Retrain on all data to save the final model
    classifier.fit(X, Y)

    # While serializing the model with joblib is often more efficient than pickle[1],
    # for now we use pickle[2].
    # [1] http://scikit-learn.org/stable/modules/model_persistence.html
    # [2] https://github.com/openml/python/issues/21 and correspondence with my supervisor
    pickle.dump(classifier, open(runname + '.pkl', "wb"))

    # TODO (?) Return an OpenML run instead.
    return predictions_path, description_path


def run_all(tasks, classifiers):
    """ Calls run(task, classifier) with all combinations of tasks and classifiers

    Parameters
    ----------
    tasks : list of tasks
        a list of OpenML Task objects
    classifiers : list of classifiers
        list of (scikit learn) classifiers which fit the definition specified
        for function run(task, classifier)
    """
    for task in tasks:
        # Getting the split through the task object is not yet possible in the
        # OpenML API (17-12)
        for clf in classifiers:
            runname = "task" + str(task.task_id) + "_" + clf.__class__.__name__

            clf, arff_dict = openml_run(task, clf)

            description_xml = create_description_xml(task.task_id)
            with open(runname + '.xml', 'w') as fh:
                fh.write(description_xml)
