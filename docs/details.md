# Advanced User Guide

This document highlights some of the more advanced features of
`openml-python`. 

## Configuration

The configuration file resides in a directory `~/.config/openml` in the
home directory of the user and is called config (More specifically, it
resides in the [configuration directory specified by the XDG Base
Directory
Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)).
It consists of `key = value` pairs which are separated by newlines. The
following keys are defined:

- apikey: required to access the server.
- server: the server to connect to (default: `https://www.openml.org`).
          For connection to the test server, set this to `https://test.openml.org`.
- cachedir: the root folder where the cache file directories should be created.
    If not given, will default to `~/.openml/cache`
- avoid_duplicate_runs: if set to `True` (default), when certain functions
            are called a lookup is performed to see if there already
            exists such a run on the server. If so, download those
            results instead.
- retry_policy: Defines how to react when the server is unavailable or
            experiencing high load. It determines both how often to
            attempt to reconnect and how quickly to do so. Please don't
            use `human` in an automated script that you run more than
            one instance of, it might increase the time to complete your
            jobs and that of others. One of:
            -   human (default): For people running openml in interactive
                fashion. Try only a few times, but in quick succession.
            -   robot: For people using openml in an automated fashion. Keep
                trying to reconnect for a longer time, quickly increasing
                the time between retries.

- connection_n_retries: number of times to retry a request if they fail. 
Default depends on retry_policy (5 for `human`, 50 for `robot`)
- verbosity: the level of output:
      -   0: normal output
      -   1: info output
      -   2: debug output

This file is easily configurable by the `openml` command line interface.
To see where the file is stored, and what its values are, use openml
configure none. 

## Docker

It is also possible to try out the latest development version of
`openml-python` with docker:

``` bash
docker run -it openml/openml-python
```

See the [openml-python docker
documentation](https://github.com/openml/openml-python/blob/main/docker/readme.md)
for more information.

## Key concepts

OpenML contains several key concepts which it needs to make machine
learning research shareable. A machine learning experiment consists of
one or several **runs**, which describe the performance of an algorithm
(called a **flow** in OpenML), its hyperparameter settings (called a
**setup**) on a **task**. A **Task** is the combination of a
**dataset**, a split and an evaluation metric. In this user guide we
will go through listing and exploring existing **tasks** to actually
running machine learning algorithms on them. In a further user guide we
will examine how to search through **datasets** in order to curate a
list of **tasks**.

A further explanation is given in the [OpenML user
guide](https://docs.openml.org/concepts/).

## Inspecting class distributions

Before training a classifier, inspect the target's class counts: a rare class
can affect the choice of evaluation metrics or the number of cross-validation
folds. `OpenMLDataset.get_class_distribution()` returns `count` and `proportion`
for each observed class. It does not resample the data or apply an imbalance
threshold.

For an existing dataset, call `dataset.get_class_distribution(target="class")`,
or omit `target` to use its single default target. The target must have nominal
feature metadata and cannot be a row-ID or ignored column. Numeric regression,
string and multi-target defaults are not supported; select a single nominal
column explicitly. An explicit name is matched exactly, even if it contains a
comma. Feature metadata and the full data may be downloaded if not cached.
Server-computed qualities may describe a different default target, so they are
not used to calculate this report.

The following complete example uses local files and requires no network access:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from openml.datasets import OpenMLDataset

with TemporaryDirectory() as directory:
    directory = Path(directory)
    data_file = directory / "classes.arff"
    data_file.write_text(
        "@relation classes\n"
        "@attribute x numeric\n"
        "@attribute class {A,B,C}\n"
        "@data\n1,A\n2,A\n3,A\n4,B\n5,?\n"
    )
    features_file = directory / "features.xml"
    features_file.write_text(
        '<oml:data_features xmlns:oml="http://openml.org/openml">'
        '<oml:feature><oml:index>0</oml:index><oml:name>x</oml:name>'
        '<oml:data_type>numeric</oml:data_type>'
        '<oml:number_of_missing_values>0</oml:number_of_missing_values></oml:feature>'
        '<oml:feature><oml:index>1</oml:index><oml:name>class</oml:name>'
        '<oml:data_type>nominal</oml:data_type>'
        '<oml:nominal_value>A</oml:nominal_value>'
        '<oml:nominal_value>B</oml:nominal_value>'
        '<oml:nominal_value>C</oml:nominal_value>'
        '<oml:number_of_missing_values>1</oml:number_of_missing_values></oml:feature>'
        '</oml:data_features>'
    )
    dataset = OpenMLDataset(
        name="classes",
        description="Local class distribution example",
        data_file=str(data_file),
        features_file=str(features_file),
        default_target_attribute="class",
    )
    print(dataset.get_class_distribution())
    print(dataset.get_class_distribution(dropna=False))
```

The first report contains A: count 3, proportion 0.75 and B: count 1,
proportion 0.25. The second contains A: 3, 0.6; B: 1, 0.2; and a missing-label
row: 1, 0.2. Declared but unobserved category C is omitted.

By default, missing labels are excluded and proportions divide by the number
of nonmissing labels. With `dropna=False`, missing values are combined into
one final row with a NaN index label; select it using `report.index.isna()`.
Nonmissing rows follow first appearance, not category order or frequency.
The index is an object index named `class`; counts are int64 and proportions
are float64. A single observed class has proportion 1.0. Empty data (or
all-missing data with the default `dropna=True`) returns an empty frame with
the same columns and dtypes. An unpublished local dataset needs feature
metadata, as supplied in the example.
