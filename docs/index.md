# OpenML

**The Python API for a World of Data and More**

Welcome to the documentation of the OpenML Python API, a connector to
the collaborative machine learning platform
[OpenML.org](https://www.openml.org). 
OpenML-Python can download or upload data from OpenML, such as datasets
and machine learning experiment results.

If you are new to OpenML, we recommend checking out the [OpenML documentation](https://docs.openml.org/)
to get familiar with the concepts and features of OpenML. In particular, we recommend 
reading more about the [OpenML concepts](https://docs.openml.org/concepts/). 

## :joystick: Minimal Examples

Use the following code to get the [credit-g](https://www.openml.org/search?type=data&sort=runs&status=active&id=31) [dataset](https://docs.openml.org/concepts/data/):

```python
import openml

dataset = openml.datasets.get_dataset("credit-g") # or by ID get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(target="class")
```

Get a missing-value summary for a dataset:

```python
import openml

dataset = openml.datasets.get_dataset(31)
summary = dataset.get_missing_summary()
```

Get a [task](https://docs.openml.org/concepts/tasks/) for [supervised classification on credit-g](https://www.openml.org/search?type=task&id=31&source_data.data_id=31):

```python
import openml

task = openml.tasks.get_task(31)
dataset = task.get_dataset()
X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name)
# get splits for the first fold of 10-fold cross-validation
train_indices, test_indices = task.get_train_test_split_indices(fold=0)
```

Use an [OpenML benchmarking suite](https://docs.openml.org/concepts/benchmarking/) to get a curated list of machine-learning tasks:
```python
import openml

suite = openml.study.get_suite("amlb-classification-all")  # Get a curated list of tasks for classification
for task_id in suite.tasks:
    task = openml.tasks.get_task(task_id)
```
Find more examples in the navbar at the top.

## :magic_wand: Installation

OpenML-Python is available on Linux, MacOS, and Windows.

You can install OpenML-Python with:

```bash
pip install openml
```

For more advanced installation information, please see the
["Introduction"](../examples/Basics/introduction_tutorial) example.


## Further information

-   [OpenML documentation](https://docs.openml.org/)
-   [OpenML client APIs](https://docs.openml.org/APIs/)
-   [OpenML developer guide](https://docs.openml.org/contributing/)
-   [Contact information](https://www.openml.org/contact)
-   [Citation request](https://www.openml.org/cite)
-   [OpenML blog](https://medium.com/open-machine-learning)
-   [OpenML twitter account](https://twitter.com/open_ml)


## Contributing

Contributing to the OpenML package is highly appreciated. Please see the
["Contributing"](contributing.md) page for more information.

## Citing OpenML-Python

If you use OpenML-Python in a scientific publication, we would
appreciate a reference to our JMLR-MLOSS paper 
["OpenML-Python: an extensible Python API for OpenML"](https://www.jmlr.org/papers/v22/19-920.html):

=== "Bibtex"

    ```bibtex
    @article{JMLR:v22:19-920,
        author  = {Matthias Feurer and Jan N. van Rijn and Arlind Kadra and Pieter Gijsbers and Neeratyoy Mallik and Sahithya Ravi and Andreas MÃ¼ller and Joaquin Vanschoren and Frank Hutter},
        title   = {OpenML-Python: an extensible Python API for OpenML},
        journal = {Journal of Machine Learning Research},
        year    = {2021},
        volume  = {22},
        number  = {100},
        pages   = {1--5},
        url     = {http://jmlr.org/papers/v22/19-920.html}
    }
    ```

=== "MLA"

    Feurer, Matthias, et al. 
    "OpenML-Python: an extensible Python API for OpenML."
    _Journal of Machine Learning Research_ 22.100 (2021):1−5.
