from collections import OrderedDict
import xmltodict
import sys
import time
import arff

from .flow import OpenMLFlow


class OpenMLRun(object):
    def __init__(self, task_id, flow_id, setup_string, dataset_id, files=None,
                 setup_id=None, tags=None, run_id=None, uploader=None,
                 uploader_name=None, evaluations=None, data_content=None,
                 classifier=None, task_type=None, task_evaluation_measure=None,
                 flow_name=None, parameter_settings=None, predictions_url=None):
        self.run_id = run_id
        self.uploader = uploader
        self.uploader_name = uploader_name
        self.task_id = task_id
        self.task_type = task_type
        self.task_evaluation_measure = task_evaluation_measure
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.setup_id = setup_id
        self.setup_string = setup_string
        self.parameter_settings = parameter_settings
        self.dataset_id = dataset_id
        self.predictions_url = predictions_url
        self.evaluations = evaluations
        self.data_content = data_content
        self.classifier = classifier

    @staticmethod
    def construct_description_dictionary(taskid, flow_id, setup_string,
                                         parameter_settings, tags):
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
            information that give a description of the run, must conform to
            regex "([a-zA-Z0-9_\-\.])+"

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

    def generate_arff(self, api_connector):
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
        run_environment = (OpenMLRun.get_version_information() +
                           [time.strftime("%c")] + ['Created by openml_run()'])
        task = api_connector.download_task(self.task_id)
        class_labels = task.class_labels

        arff_dict = {}
        arff_dict['attributes'] = [('repeat', 'NUMERIC'),  # lowercase 'numeric' gives an error
                                   ('fold', 'NUMERIC'),
                                   ('row_id', 'NUMERIC')] + \
            [('confidence.' + class_labels[i], 'NUMERIC') for i in range(len(class_labels))] +\
            [('prediction', class_labels),
             ('correct', class_labels)]
        arff_dict['data'] = self.data_content
        arff_dict['description'] = "\n".join(run_environment)
        arff_dict['relation'] = 'openml_task_' + str(task.task_id) + '_predictions'
        return arff_dict

    @staticmethod
    def create_setup_string(classifier):
        run_environment = " ".join(OpenMLRun.get_version_information())
        # fixme str(classifier) might contain (...)
        return run_environment + " " + str(classifier)

    def create_description_xml(self):
        run_environment = OpenMLRun.get_version_information()
        setup_string = ''  # " ".join(sys.argv);

        parameter_settings = self.classifier.get_params()
        # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
        # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
        well_formatted_time = time.strftime("%c").replace(
            ' ', '_').replace('/', '-').replace(':', '.')
        tags = run_environment + [well_formatted_time] + ['openml_run'] + \
            [self.classifier.__module__ + "." + self.classifier.__class__.__name__]
        description = OpenMLRun.construct_description_dictionary(
            self.task_id, self.flow_id, setup_string, parameter_settings, tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml

    # This can possibly be done by a package such as pyxb, but I could not get
    # it to work properly.
    @staticmethod
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

    @staticmethod
    def openml_run(connector, task, classifier):
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
        flow_id = OpenMLFlow.ensure_flow_exists(task.api_connector, classifier)
        if(flow_id < 0):
            print("No flow")
            return 0, 2
        print(flow_id)

        #runname = "t" + str(task.task_id) + "_" + str(classifier)
        arff_datacontent = []

        dataset = task.get_dataset()
        X, Y = dataset.get_dataset(target=task.target_feature)

        class_labels = task.class_labels
        if class_labels is None:
            raise ValueError('The task has no class labels. This method currently '
                             'only works for tasks with class labels.')
        setup_string = OpenMLRun.create_setup_string(classifier)

        run = OpenMLRun(task.task_id, flow_id, setup_string, dataset.id)

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

        run.data_content = arff_datacontent
        run.classifier = classifier.fit(X, Y)

        # Generate a dictionary which represents the arff file (with predictions)
        #arff_dict = OpenMLRun.generate_arff(arff_datacontent, task)
        #predictions_path = runname + '.arff'
        #with open(predictions_path, 'w') as fh:
        #    #arff.dump(arff_dict, fh)

        #description_xml = OpenMLRun.create_description_xml(task.task_id, flow_id, classifier)
        #description_path = runname + '.xml'
        #with open(description_path, 'w') as fh:
        #    #fh.write(description_xml)

        # Retrain on all data to save the final model

        # While serializing the model with joblib is often more efficient than pickle[1],
        # for now we use pickle[2].
        # [1] http://scikit-learn.org/stable/modules/model_persistence.html
        # [2] https://github.com/openml/python/issues/21 and correspondence with my supervisor
        #pickle.dump(classifier, open(runname + '.pkl', "wb"))

        # TODO (?) Return an OpenML run instead.
        return run

    def publish(self, api_connector):
        predictions = arff.dumps(self.generate_arff(api_connector))
        description_xml = self.create_description_xml()
        data = {'predictions': predictions, 'description':
                description_xml}
        return_code, dataset_xml = api_connector._perform_api_call(
            "/run/", file_elements=data)
        return return_code, dataset_xml
