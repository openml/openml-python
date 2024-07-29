# OpenML

**Collaborative Machine Learning in Python**

Welcome to the documentation of the OpenML Python API, a connector to
the collaborative machine learning platform
[OpenML.org](https://www.openml.org). The OpenML Python package allows
to use datasets and tasks from OpenML together with scikit-learn and
share the results online.

## Example

```python
import openml
from sklearn import impute, tree, pipeline

# Define a scikit-learn classifier or pipeline
clf = pipeline.Pipeline(
    steps=[
        ('imputer', impute.SimpleImputer()),
        ('estimator', tree.DecisionTreeClassifier())
    ]
)
# Download the OpenML task for the pendigits dataset with 10-fold
# cross-validation.
task = openml.tasks.get_task(32)
# Run the scikit-learn model on the task.
run = openml.runs.run_model_on_task(clf, task)
# Publish the experiment on OpenML (optional, requires an API key.
# You can get your own API key by signing up to OpenML.org)
run.publish()
print(f'View the run online: {run.openml_url}')
```

Find more examples in our ["Example Gallery"][example-gallery].

## How to get OpenML for python

You can install the OpenML package via `pip` (we recommend using a virtual environment):

```bash
python -m pip install openml
```

For more advanced installation information, please see the
["Installation"][installation-guide] section.

## Further information

-   [OpenML documentation](https://docs.openml.org/)
-   [OpenML client APIs](https://docs.openml.org/APIs/)
-   [OpenML developer guide](https://docs.openml.org/Contributing/)
-   [Contact information](https://www.openml.org/contact)
-   [Citation request](https://www.openml.org/cite)
-   [OpenML blog](https://medium.com/open-machine-learning)
-   [OpenML twitter account](https://twitter.com/open_ml)

## Contributing

Contribution to the OpenML package is highly appreciated. Please see the
["Contributing"][contributing] page for more information.

## Citing OpenML-Python

If you use OpenML-Python in a scientific publication, we would
appreciate a reference to our JMLR-MLOSS paper 
["OpenML-Python: an extensible Python API for OpenML"](https://www.jmlr.org/papers/v22/19-920.html):

=== "MLA"

    Feurer, Matthias, et al. 
    "OpenML-Python: an extensible Python API for OpenML."
    _Journal of Machine Learning Research_ 22.100 (2021):1−5.

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