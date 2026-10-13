# License: BSD 3-Clause
"""Offline tests for target-specific class distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_mock import MockerFixture

from openml.datasets import OpenMLDataFeature, OpenMLDataset


@pytest.fixture(autouse=True)
def no_network(mocker: MockerFixture) -> None:
    mocker.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Unexpected network")
    )


def make_dataset(mocker: MockerFixture, labels: pd.Series, target: str = "class") -> OpenMLDataset:
    dataset = OpenMLDataset(
        name="distribution", description="Offline fixture", default_target_attribute=target
    )
    dataset._features = {
        0: OpenMLDataFeature(0, "x", "numeric", None, 0),
        1: OpenMLDataFeature(1, target, "nominal", ["A", "B", "C"], 0),
    }
    data = pd.DataFrame({"x": np.arange(len(labels)), target: labels})
    mocker.patch.object(
        dataset,
        "_load_data",
        side_effect=lambda: (data.copy(deep=True), [False, True], list(data.columns)),
    )
    return dataset


def expected(labels: list[Any], counts: list[int], proportions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "count": np.array(counts, dtype="int64"),
            "proportion": np.array(proportions, dtype="float64"),
        },
        index=pd.Index(labels, dtype=object, name="class"),
    )


@pytest.mark.parametrize(
    ("labels", "classes", "counts", "proportions"),
    [
        (["A", "B", "A", "B"], ["A", "B"], [2, 2], [0.5, 0.5]),
        (["A", "A", "A", "B"], ["A", "B"], [3, 1], [0.75, 0.25]),
        (["C", "B", "C", "A"], ["C", "B", "A"], [2, 1, 1], [0.5, 0.25, 0.25]),
        (["A", "A"], ["A"], [2], [1.0]),
        (["B"], ["B"], [1], [1.0]),
        ([True, False, True], [True, False], [2, 1], [2 / 3, 1 / 3]),
        ([2, 1, 2], [2, 1], [2, 1], [2 / 3, 1 / 3]),
        (["2", 1, "2"], ["2", 1], [2, 1], [2 / 3, 1 / 3]),
    ],
)
def test_observed_classes(
    mocker: MockerFixture,
    labels: list[Any],
    classes: list[Any],
    counts: list[int],
    proportions: list[float],
) -> None:
    dataset = make_dataset(mocker, pd.Series(labels))
    assert_frame_equal(dataset.get_class_distribution(), expected(classes, counts, proportions))
    assert_frame_equal(
        dataset.get_class_distribution("class"), expected(classes, counts, proportions)
    )


@pytest.mark.parametrize("dropna", [True, False])
def test_missing_values(mocker: MockerFixture, dropna: bool) -> None:
    dataset = make_dataset(mocker, pd.Series([None, "A", np.nan, "B", pd.NA, "A"], dtype=object))
    result = dataset.get_class_distribution(dropna=dropna)
    wanted = (
        expected(["A", "B"], [2, 1], [2 / 3, 1 / 3])
        if dropna
        else expected(["A", "B", np.nan], [2, 1, 3], [2 / 6, 1 / 6, 3 / 6])
    )
    assert_frame_equal(result, wanted)


@pytest.mark.parametrize("dtype", ["category", "string", "object"])
def test_missing_representations(mocker: MockerFixture, dtype: str) -> None:
    dataset = make_dataset(mocker, pd.Series(["None", None, "nan", pd.NA], dtype=dtype))
    assert_frame_equal(
        dataset.get_class_distribution(dropna=False),
        expected(["None", "nan", np.nan], [1, 1, 2], [0.25, 0.25, 0.5]),
    )


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("labels", [[], [None], [None, np.nan, pd.NA]])
def test_empty_and_all_missing(mocker: MockerFixture, labels: list[Any], dropna: bool) -> None:
    dataset = make_dataset(mocker, pd.Series(labels, dtype=object))
    wanted = (
        expected([np.nan], [len(labels)], [1.0]) if labels and not dropna else expected([], [], [])
    )
    assert_frame_equal(dataset.get_class_distribution(dropna=dropna), wanted)


@pytest.mark.parametrize("labels", [[], ["B", "A", "B"], ["B", None]])
@pytest.mark.parametrize("dropna", [True, False])
def test_unused_categories_and_order(
    mocker: MockerFixture, labels: list[Any], dropna: bool
) -> None:
    series = pd.Series(pd.Categorical(labels, categories=["C", "A", "B"], ordered=True))
    dataset = make_dataset(mocker, series)
    before, *_ = dataset.get_data()
    result = dataset.get_class_distribution(dropna=dropna)
    if not labels:
        wanted = expected([], [], [])
    elif None in labels:
        wanted = (
            expected(["B"], [1], [1.0]) if dropna else expected(["B", np.nan], [1, 1], [0.5, 0.5])
        )
    else:
        wanted = expected(["B", "A"], [2, 1], [2 / 3, 1 / 3])
    assert_frame_equal(result, wanted)
    assert_frame_equal(dataset.get_class_distribution(dropna=dropna), wanted)
    after, *_ = dataset.get_data()
    assert_frame_equal(before, after)
    assert list(series.cat.categories) == ["C", "A", "B"]


@pytest.mark.parametrize("target", [1, True, ["class"], ("class",)])
def test_invalid_target_type(mocker: MockerFixture, target: Any) -> None:
    dataset = OpenMLDataset(name="invalid", description="No metadata should load")
    load = mocker.patch.object(dataset, "_load_features")
    with pytest.raises(TypeError, match="target must"):
        dataset.get_class_distribution(target)
    load.assert_not_called()


@pytest.mark.parametrize("dropna", [None, 0, 1, "false", np.True_])
def test_invalid_dropna(mocker: MockerFixture, dropna: Any) -> None:
    dataset = OpenMLDataset(name="invalid", description="No metadata should load")
    load = mocker.patch.object(dataset, "_load_features")
    with pytest.raises(TypeError, match="dropna must"):
        dataset.get_class_distribution(dropna=dropna)
    load.assert_not_called()


@pytest.mark.parametrize("target", ["", " ", "absent", "class,other"])
def test_invalid_target_name(mocker: MockerFixture, target: str) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    with pytest.raises(ValueError, match=r"target|Target"):
        dataset.get_class_distribution(target)
    dataset._load_data.assert_not_called()


@pytest.mark.parametrize("default", [None, "", " ", "absent", "class,other"])
def test_default_resolution(mocker: MockerFixture, default: str | None) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    dataset.default_target_attribute = default
    error = NotImplementedError if default == "class,other" else ValueError
    with pytest.raises(error):
        dataset.get_class_distribution()
    assert_frame_equal(dataset.get_class_distribution("class"), expected(["A"], [1], [1.0]))


@pytest.mark.parametrize("target", ["label,part", " label "])
def test_literal_target_names(mocker: MockerFixture, target: str) -> None:
    dataset = make_dataset(mocker, pd.Series(["B"]), target=target)
    assert_frame_equal(dataset.get_class_distribution(), expected(["B"], [1], [1.0]))
    assert_frame_equal(dataset.get_class_distribution(target), expected(["B"], [1], [1.0]))


def test_nondefault_target_and_qualities(mocker: MockerFixture) -> None:
    dataset = make_dataset(mocker, pd.Series(["A", "B"]))
    dataset._features[0] = OpenMLDataFeature(0, "x", "nominal", ["0", "1"], 0)
    dataset._qualities = {"MajorityClassSize": 1000.0}
    assert_frame_equal(dataset.get_class_distribution("x"), expected([0, 1], [1, 1], [0.5, 0.5]))
    assert dataset._qualities == {"MajorityClassSize": 1000.0}


@pytest.mark.parametrize("data_type", ["numeric", "string", "date"])
def test_non_nominal_target(mocker: MockerFixture, data_type: str) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    dataset._features[1].data_type = data_type
    with pytest.raises(ValueError, match="must be nominal"):
        dataset.get_class_distribution()
    dataset._load_data.assert_not_called()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("row_id_attribute", "class"),
        ("row_id_attribute", ["class"]),
        ("ignore_attribute", "class"),
        ("ignore_attribute", ["class"]),
    ],
)
def test_excluded_target(mocker: MockerFixture, attribute: str, value: Any) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    setattr(dataset, attribute, value)
    with pytest.raises(ValueError, match="row-ID or ignored"):
        dataset.get_class_distribution()


def test_unrelated_exclusion(mocker: MockerFixture) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    dataset.ignore_attribute = ["x"]
    assert_frame_equal(dataset.get_class_distribution(), expected(["A"], [1], [1.0]))


@pytest.mark.parametrize("columns", [["x", "other"], ["class", "class"]])
def test_mismatched_or_duplicate_data_columns(mocker: MockerFixture, columns: list[str]) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    mocker.patch.object(
        dataset,
        "_load_data",
        return_value=(pd.DataFrame([["A", "B"]], columns=columns), [True, True], columns),
    )
    with pytest.raises(ValueError, match="exactly one data column"):
        dataset.get_class_distribution()


def test_duplicate_metadata(mocker: MockerFixture) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    dataset._features[0].name = "class"
    with pytest.raises(ValueError, match="exactly one feature"):
        dataset.get_class_distribution()


def test_sparse_labels(mocker: MockerFixture) -> None:
    dataset = make_dataset(
        mocker, pd.Series(pd.arrays.SparseArray([1.0, 0.0, 1.0, np.nan], fill_value=0.0))
    )
    assert_frame_equal(
        dataset.get_class_distribution(dropna=False),
        expected([1.0, 0.0, np.nan], [2, 1, 1], [0.5, 0.25, 0.25]),
    )


def test_loading_error_propagates(mocker: MockerFixture) -> None:
    dataset = make_dataset(mocker, pd.Series(["A"]))
    mocker.patch.object(dataset, "_load_data", side_effect=OSError("broken cache"))
    with pytest.raises(OSError, match="broken cache"):
        dataset.get_class_distribution()


def test_local_dataset_without_metadata() -> None:
    dataset = OpenMLDataset(
        name="local", description="No metadata", default_target_attribute="class"
    )
    with pytest.raises(ValueError, match="No dataset id"):
        dataset.get_class_distribution()


@pytest.mark.parametrize("cache_format", ["pickle", "feather"])
def test_real_arff_loading(tmp_path: Path, cache_format: str) -> None:
    data = tmp_path / "classes.arff"
    data.write_text(
        "@relation classes\n@attribute x numeric\n@attribute class {A,B,C}\n"
        "@data\n1,B\n2,A\n3,B\n4,?\n"
    )
    metadata = tmp_path / "features.xml"
    metadata.write_text("""<oml:data_features xmlns:oml="http://openml.org/openml">
<oml:feature>
<oml:index>0</oml:index>
<oml:name>x</oml:name>
<oml:data_type>numeric</oml:data_type>
<oml:number_of_missing_values>0</oml:number_of_missing_values>
</oml:feature>
<oml:feature>
<oml:index>1</oml:index>
<oml:name>class</oml:name>
<oml:data_type>nominal</oml:data_type>
<oml:nominal_value>A</oml:nominal_value>
<oml:nominal_value>B</oml:nominal_value>
<oml:nominal_value>C</oml:nominal_value>
<oml:number_of_missing_values>1</oml:number_of_missing_values>
</oml:feature>
</oml:data_features>""")
    dataset = OpenMLDataset(
        name="classes",
        description="Local ARFF",
        data_file=str(data),
        features_file=str(metadata),
        default_target_attribute="class",
        cache_format=cache_format,
    )
    wanted = expected(["B", "A", np.nan], [2, 1, 1], [0.5, 0.25, 0.25])
    assert_frame_equal(dataset.get_class_distribution(dropna=False), wanted)
    # The second call exercises the written cache rather than ARFF parsing.
    assert_frame_equal(dataset.get_class_distribution(dropna=False), wanted)
