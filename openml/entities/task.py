import sys

if sys.version_info[0] > 3:
    import pickle
else:
    try:
        import cPickle as pickle
    except:
        import pickle


class OpenMLTask(object):
    def __init__(self, task_id, task_type, data_set_id, target_feature,
                 estimation_procedure_type, data_splits_url,
                 estimation_parameters, evaluation_measure,cost_matrix, api_connector, class_labels = None):
        self.task_id = int(task_id)
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.target_feature = target_feature
        # TODO: this can become its own class if necessary
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procedure_type
        # TODO: ideally this has the indices for the different splits...but
        # the evaluation procedure 3foldtest/10foldvalid is not available
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure["parameters"] = estimation_parameters
        #
        self.estimation_parameters = estimation_parameters
        self.evaluation_measure = evaluation_measure
        self.cost_matrix = cost_matrix
        self.api_connector = api_connector
        self.class_labels = class_labels

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")

    def __str__(self):
        return "OpenMLTask instance.\nTask ID: %s\n" \
               "Task type: %s\nDataset id: %s" \
               % (self.task_id, self.task_type, self.dataset_id)

    def get_dataset(self):
        return self.api_connector.download_dataset(self.dataset_id)

    def get_X_and_Y(self):
        dataset = self.get_dataset()
        # Replace with retrieve from cache
        if 'Supervised Classification'.lower() in self.task_type.lower():
            target_dtype = int
        elif 'Supervised Regression'.lower() in self.task_type.lower():
            target_dtype = float
        else:
            raise NotImplementedError(self.task_type)
        X_and_Y = dataset.get_dataset(target=self.target_feature,
                                      target_dtype=target_dtype)
        return X_and_Y

    def evaluate(self, algo):
        """Evaluate an algorithm on the test data.
        """
        raise NotImplementedError()

    def validate(self, algo):
        """Evaluate an algorithm on the validation data.
        """
        raise NotImplementedError()

    def get_train_test_split_indices(self, fold=0, repeat=0):
        # Replace with retrieve from cache
        split = self.api_connector.download_split(self)
        train_indices, test_indices = split.get(repeat=repeat, fold=fold)
        return train_indices, test_indices

    def get_train_and_test_set(self, fold=0, repeat=0):
        X, Y = self.get_X_and_Y()
        train_indices, test_indices = self.get_train_test_split_indices(
            fold=fold, repeat=repeat)
        return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

    """
    def get_validation_split(self, fold):
        ""This is not part of the OpenML specification!
        ""
        split = OpenMLSplit.from_arff_file(
            self.estimation_procedure["local_validation_split_file"])

        if len(split.split.keys()) != 1:
            raise NotImplementedError("Repeats are not implemented yet...")

        # TODO: write a test that always a subset of the train/test split is
        # returned
        vtrain_indices, validation_indices = split.split[0][fold]
        train_indices, test_indices = self.get_train_test_split()

        return train_indices[vtrain_indices], train_indices[validation_indices]

    def get_CV_fold(self, X, Y, fold, folds, shuffle=True):
        ""This is not part of the OpenML specification
        ""
        fold = int(fold)
        folds = int(folds)
        if fold >= folds:
            raise ValueError((fold, folds))
        if X.shape[0] != Y.shape[0]:
            raise ValueError("The first dimension of the X and Y array must "
                             "be equal.")

        if shuffle == True:
            rs = np.random.RandomState(42)
            indices = np.arange(X.shape[0])
            rs.shuffle(indices)
            Y = Y[indices]

        kf = StratifiedKFold(Y, n_folds=folds, indices=True)
        for idx, split in enumerate(kf):
            if idx == fold:
                break

        if shuffle == True:
            return indices[split[0]], indices[split[1]]
        return split
    """

    """
    def perform_cv_fold(self, algo, fold, folds):
        ""Allows the user to perform cross validation for hyperparameter
        optimization on the training data.""
        # TODO: this is only done for hyperparameter optimization and is not
        # part of the OpenML specification. The OpenML specification would
        # like to have the hyperparameter evaluation inside the evaluate
        # performed by the target algorithm itself. Hyperparameter
        # optimization on the other hand needs these both things to be decoupled
        # For being closer to OpenML one could also call evaluate and pass
        # everything else through kwargs.
        if self.task_type != "Supervised Classification":
            raise NotImplementedError(self.task_type)

        print("Procedure", self.estimation_procedure)
        print("Type", self.estimation_procedure["type"])
        # TODO fix Task generation!
        # if self.estimation_procedure["type"] not in ["holdout",
        # "customholdout"]:
        #    raise NotImplementedError(self.estimation_procedure["type"])

        #if self.estimation_procedure["parameters"]["stratified_sampling"] != \
        #        'true':
        #    raise NotImplementedError(
        #        self.estimation_procedure["parameters"]["stratified_sampling"])

        #if self.evaluation_measure not in ["predictive accuracy",
        #                                   "predictive_accuracy"]:
        #    raise NotImplementedError(self.evaluation_measure)

        # #######################################################################
        # Test folds
        train_indices, test_indices = self.get_train_test_split()

        ########################################################################
        # Crossvalidation folds
        train_indices, validation_indices = self.get_validation_split(fold)

        X, Y = self.get_dataset()

        algo.fit(X[train_indices], Y[train_indices])

        predictions = algo.predict(X[validation_indices])
        accuracy = sklearn.metrics.accuracy_score(Y[validation_indices],
                                                  predictions)
        return accuracy
    """
