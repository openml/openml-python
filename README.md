

<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://github.com/openml/openml.org/blob/master/app/public/static/svg/logo.svg" width="50" alt="OpenML Logo"/>
      <h1>OpenML-Python</h1>
      <img src="https://github.com/openml/docs/blob/master/docs/img/python.png" width="50" alt="Python Logo"/>
    </summary>
  </ul>
</div>

## The Python API for a World of Data and More :dizzy:

[![Latest Release](https://img.shields.io/github/v/release/openml/openml-python)](https://github.com/openml/openml-python/releases)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/openml/)
[![Downloads](https://static.pepy.tech/badge/openml)](https://pepy.tech/project/openml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
<!-- Add green badges for CI and precommit -->

[Installation](https://openml.github.io/openml-python/main/#how-to-get-openml-for-python) | [Documentation](https://openml.github.io/openml-python) | [Contribution guidelines](https://github.com/openml/openml-python/blob/main/CONTRIBUTING.md)
</div>

OpenML-Python provides an easy-to-use and straightforward Python interface for [OpenML](http://openml.org), an online platform for open science collaboration in machine learning.
It can download or upload data from OpenML, such as datasets and machine learning experiment results.

## :joystick: Minimal Example

Use the following code to get the [credit-g](https://www.openml.org/search?type=data&sort=runs&status=active&id=31) [dataset](https://docs.openml.org/concepts/data/):

```python
import openml

dataset = openml.datasets.get_dataset("credit-g") # or by ID get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(target="class")
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

## :magic_wand: Installation

OpenML-Python is supported on Python 3.10 - 3.14 and is available on Linux, MacOS, and Windows.

You can install OpenML-Python with:

```bash
pip install openml
```

## :page_facing_up: Citing OpenML-Python

If you use OpenML-Python in a scientific publication, we would appreciate a reference to the following paper:

[Matthias Feurer, Jan N. van Rijn, Arlind Kadra, Pieter Gijsbers, Neeratyoy Mallik, Sahithya Ravi, Andreas Müller, Joaquin Vanschoren, Frank Hutter<br/>
**OpenML-Python: an extensible Python API for OpenML**<br/>
Journal of Machine Learning Research, 22(100):1−5, 2021](https://www.jmlr.org/papers/v22/19-920.html)

Bibtex entry:
```bibtex
@article{JMLR:v22:19-920,
  author  = {Matthias Feurer and Jan N. van Rijn and Arlind Kadra and Pieter Gijsbers and Neeratyoy Mallik and Sahithya Ravi and Andreas Müller and Joaquin Vanschoren and Frank Hutter},
  title   = {OpenML-Python: an extensible Python API for OpenML},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {100},
  pages   = {1--5},
  url     = {http://jmlr.org/papers/v22/19-920.html}
}
```
## :handshake: Contributing

We welcome contributions from both new and experienced developers!

If you would like to contribute to OpenML-Python, please read our  
[Contribution Guidelines](https://github.com/openml/openml-python/blob/main/CONTRIBUTING.md).

If you are new to open-source development, a great way to get started is by
looking at issues labeled **"good first issue"** in our GitHub issue tracker.
These tasks are beginner-friendly and help you understand the project structure,
development workflow, and how to submit a pull request.
