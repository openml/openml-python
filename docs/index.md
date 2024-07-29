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

You can find more examples in our `examples-index`{.interpreted-text
role="ref"}.

## How to get OpenML for python

You can install the OpenML package via `pip` (we recommend using a virtual environment):

```bash
python -m pip install openml
```

For more advanced installation information, please see the
["Installation"][installation] section.

## Content

-   `usage`{.interpreted-text role="ref"}
-   `api`{.interpreted-text role="ref"}
-   `examples-index`{.interpreted-text role="ref"}
-   `extensions`{.interpreted-text role="ref"}
-   `contributing`{.interpreted-text role="ref"}
-   `progress`{.interpreted-text role="ref"}

## Further information

-   [OpenML documentation](https://docs.openml.org/)
-   [OpenML client APIs](https://docs.openml.org/APIs/)
-   [OpenML developer guide](https://docs.openml.org/Contributing/)
-   [Contact information](https://www.openml.org/contact)
-   [Citation request](https://www.openml.org/cite)
-   [OpenML blog](https://medium.com/open-machine-learning)
-   [OpenML twitter account](https://twitter.com/open_ml)

## Contributing

Contribution to the OpenML package is highly appreciated. The OpenML
package currently has a 1/4 position for the development and all help
possible is needed to extend and maintain the package, create new
examples and improve the usability. Please see the
`contributing`{.interpreted-text role="ref"} page for more information.

## Citing OpenML-Python

If you use OpenML-Python in a scientific publication, we would
appreciate a reference to the following paper:

| Matthias Feurer, Jan N. van Rijn, Arlind Kadra, Pieter Gijsbers,
  Neeratyoy Mallik, Sahithya Ravi, Andreas Müller, Joaquin Vanschoren,
  Frank Hutter
| **OpenML-Python: an extensible Python API for OpenML**
| Journal of Machine Learning Research, 22(100):1−5, 2021
| <https://www.jmlr.org/papers/v22/19-920.html>

> Bibtex entry:
>
>     @article{JMLR:v22:19-920,
>         author  = {Matthias Feurer and Jan N. van Rijn and Arlind Kadra and Pieter Gijsbers and Neeratyoy Mallik and Sahithya Ravi and Andreas MÃ¼ller and Joaquin Vanschoren and Frank Hutter},
>         title   = {OpenML-Python: an extensible Python API for OpenML},
>         journal = {Journal of Machine Learning Research},
>         year    = {2021},
>         volume  = {22},
>         number  = {100},
>         pages   = {1--5},
>         url     = {http://jmlr.org/papers/v22/19-920.html}
>     }
