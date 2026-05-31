# Extensions

OpenML-Python provides an extension interface to connect other machine
learning libraries than scikit-learn to OpenML. Please check the
[`api_extensions`](../reference/extensions/extension_interface/) and use the scikit-learn
extension as a starting point.

## List of extensions

Here is a list of currently maintained OpenML extensions:

-   [openml-sklearn](https://github.com/openml/openml-sklearn)
-   [openml-keras](https://github.com/openml/openml-keras)
-   [openml-pytorch](https://github.com/openml/openml-pytorch)
-   [openml-tensorflow (for tensorflow
    2+)](https://github.com/openml/openml-tensorflow)

## Connecting new machine learning libraries

### Content of the Library

To leverage support from the community and to tap in the potential of
OpenML, interfacing with popular machine learning libraries is
essential. The OpenML-Python package is capable of downloading meta-data
and results (data, flows, runs), regardless of the library that was used
to upload it. However, in order to simplify the process of uploading
flows and runs from a specific library, an additional interface can be
built. The OpenML-Python team does not have the capacity to develop and
maintain such interfaces on its own. For this reason, we have built an
extension interface to allow others to contribute back. Building a
suitable extension therefore requires an understanding of the
current OpenML-Python support.

[This tutorial](../examples/Basics/simple_flows_and_runs_tutorial) shows how the scikit-learn 
extension works with OpenML-Python.

#### API

-   The extension scripts must import the openml-python package
    and be able to interface with any function from the API.
-   The extension has to be defined as a Python class and must inherit
    from [`openml.extensions.Extension`](../reference/extensions/extension_interface/#openml.extensions.extension_interface.Extension).
-   This class needs to have all the functions from `openml.extensions.Extension` overloaded as required.
-   The redefined functions should have adequate and appropriate
    docstrings. The sklearn Extension API is a good example to follow.

#### Interfacing with OpenML-Python

Once the new extension class has been defined, the openml-python module
to [`openml.extensions.register_extension`](../reference/extensions/functions/#openml.extensions.functions.register_extension)
must be called to allow OpenML-Python to interface the new extension.

The following methods should get implemented. Although the documentation
in the extension interface should always be leading, here
we list some additional information and best practices. 
Note that most methods are relatively simple
and can be implemented in several lines of code.

-   General setup (required)
    -   `can_handle_flow`: Takes as
        argument an OpenML flow, and checks whether this can be handled
        by the current extension. The OpenML database consists of many
        flows, from various workbenches (e.g., scikit-learn, Weka, mlr).
        This method is called before a model is being deserialized.
        Typically, the flow-dependency field is used to check whether
        the specific library is present, and no unknown libraries are
        present there.
    -   `can_handle_model`: Similar to
        `can_handle_flow`, except that in
        this case a Python object is given. As such, in many cases, this
        method can be implemented by checking whether this adheres to a
        certain base class.
-   Serialization and De-serialization (required)
    -   `flow_to_model`: deserializes the
        OpenML Flow into a model (if the library can indeed handle the
        flow). This method has an important interplay with
        `model_to_flow`. Running these
        two methods in succession should result in exactly the same
        model (or flow). This property can be used for unit testing
        (e.g., build a model with hyperparameters, make predictions on a
        task, serialize it to a flow, deserialize it back, make it
        predict on the same task, and check whether the predictions are
        exactly the same.) The example in the scikit-learn interface
        might seem daunting, but note that here some complicated design
        choices were made, that allow for all sorts of interesting
        research questions. It is probably good practice to start easy.
    -   `model_to_flow`: The inverse of `flow_to_model`. Serializes a
        model into an OpenML Flow. The flow should preserve the class,
        the library version, and the tunable hyperparameters.
    -   `get_version_information`: Return
        a tuple with the version information of the important libraries.
    -   `create_setup_string`: No longer
        used, and will be deprecated soon.
-   Performing runs (required)
    -   `is_estimator`: Gets as input a
        class, and checks whether it has the status of estimator in the
        library (typically, whether it has a train method and a predict
        method).
    -   `seed_model`: Sets a random seed to the model.
    -   `_run_model_on_fold`: One of the
        main requirements for a library to generate run objects for the
        OpenML server. Obtains a train split (with labels) and a test
        split (without labels) and the goal is to train a model on the
        train split and return the predictions on the test split. On top
        of the actual predictions, also the class probabilities should
        be determined. For classifiers that do not return class
        probabilities, this can just be the hot-encoded predicted label.
        The predictions will be evaluated on the OpenML server. Also,
        additional information can be returned, for example,
        user-defined measures (such as runtime information, as this can
        not be inferred on the server). Additionally, information about
        a hyperparameter optimization trace can be provided.
    -   `obtain_parameter_values`:
        Obtains the hyperparameters of a given model and the current
        values. Please note that in the case of a hyperparameter
        optimization procedure (e.g., random search), you only should
        return the hyperparameters of this procedure (e.g., the
        hyperparameter grid, budget, etc) and that the chosen model will
        be inferred from the optimization trace.
    -   `check_if_model_fitted`: Check
        whether the train method of the model has been called (and as
        such, whether the predict method can be used).
-   Hyperparameter optimization (optional)
    -   `instantiate_model_from_hpo_class`: If a given run has recorded the hyperparameter
        optimization trace, then this method can be used to
        reinstantiate the model with hyperparameters of a given
        hyperparameter optimization iteration. Has some similarities
        with `flow_to_model` (as this
        method also sets the hyperparameters of a model). Note that
        although this method is required, it is not necessary to
        implement any logic if hyperparameter optimization is not
        implemented. Simply raise a `NotImplementedError`
        then.

### Hosting the library

Each extension created should be a stand-alone repository, compatible
with the [OpenML-Python repository](https://github.com/openml/openml-python). 
The extension repository should work off-the-shelf with *OpenML-Python* installed.

Create a public GitHub repo with the following directory structure:

    | [repo name]
    |    |-- [extension name]
    |    |    |-- __init__.py
    |    |    |-- extension.py
    |    |    |-- config.py (optionally)

### Recommended

-   Test cases to keep the extension up to date with the
    OpenML-Python upstream changes.
-   Documentation of the extension API, especially if any new
    functionality added to OpenML-Python\'s extension design.
-   Examples to show how the new extension interfaces and works with
    OpenML-Python.
-   Create a PR to add the new extension to the OpenML-Python API
    documentation.

Happy contributing!
