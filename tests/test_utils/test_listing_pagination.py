# License: BSD 3-Clause
from __future__ import annotations

import unittest.mock

import pandas as pd
import pytest

import openml


@pytest.mark.parametrize(
    ("batch_size", "limit", "offset", "expected_calls"),
    [
        (None, 2_500, 125, [(1_000, 125), (1_000, 1_125), (500, 2_125)]),
        (10_000, 2_500, 125, [(1_000, 125), (1_000, 1_125), (500, 2_125)]),
        (None, 250, 125, [(250, 125)]),
        (10_000, 2_000, 125, [(1_000, 125), (1_000, 1_125)]),
        (400, 1_050, 125, [(400, 125), (400, 525), (250, 925)]),
    ],
)
def test_list_all_preserves_limit_and_offset(batch_size, limit, offset, expected_calls):
    listing_call = unittest.mock.Mock(
        side_effect=lambda size, start: list(range(start, start + size)),
    )

    batches = openml.utils._list_all(
        listing_call,
        batch_size=batch_size,
        limit=limit,
        offset=offset,
    )

    assert listing_call.call_args_list == [unittest.mock.call(*call) for call in expected_calls]
    assert [item for batch in batches for item in batch] == list(range(offset, offset + limit))


@pytest.mark.parametrize("limit", [None, 5_000])
@pytest.mark.parametrize("available", [0, 1_000, 1_250])
def test_list_all_stops_when_results_end(limit, available):
    offset = 125
    listing_call = unittest.mock.Mock(
        side_effect=lambda size, start: list(range(start, min(start + size, offset + available))),
    )

    batches = openml.utils._list_all(
        listing_call,
        batch_size=10_000,
        limit=limit,
        offset=offset,
    )

    expected_calls = [unittest.mock.call(1_000, offset)]
    if available >= 1_000:
        expected_calls.append(unittest.mock.call(1_000, offset + 1_000))
    assert listing_call.call_args_list == expected_calls
    assert [item for batch in batches for item in batch] == list(range(offset, offset + available))


def test_list_all_stops_on_no_result():
    listing_call = unittest.mock.Mock(
        side_effect=[list(range(125, 1_125)), openml.exceptions.OpenMLServerNoResult("No results")],
    )

    batches = openml.utils._list_all(listing_call, batch_size=10_000, offset=125)

    assert listing_call.call_args_list == [
        unittest.mock.call(1_000, 125),
        unittest.mock.call(1_000, 1_125),
    ]
    assert batches == [list(range(125, 1_125))]


def test_list_datasets_pages_preserve_filters():
    def listing_call(size, offset, **filters):
        assert filters["tag"] == "pagination-test"
        assert filters["status"] == "active"
        assert filters["data_name"] == "example"
        return pd.DataFrame({"did": range(offset, offset + size)})

    with unittest.mock.patch(
        "openml.datasets.functions._list_datasets",
        side_effect=listing_call,
    ) as page_fetch:
        datasets = openml.datasets.list_datasets(
            size=2_500,
            offset=125,
            tag="pagination-test",
            status="active",
            data_name="example",
        )

    assert [call.args for call in page_fetch.call_args_list] == [
        (1_000, 125),
        (1_000, 1_125),
        (500, 2_125),
    ]
    assert isinstance(datasets, pd.DataFrame)
    assert datasets["did"].tolist() == list(range(125, 2_625))


def test_list_runs_pages_preserve_filters():
    def listing_call(size, offset, **filters):
        assert filters["tag"] == "pagination-test"
        assert filters["task"] == [42]
        return pd.DataFrame({"run_id": range(offset, offset + size)})

    with unittest.mock.patch.object(
        openml._backend.run,
        "list",
        side_effect=listing_call,
    ) as page_fetch:
        runs = openml.runs.list_runs(size=2_500, offset=125, tag="pagination-test", task=[42])

    assert [call.args for call in page_fetch.call_args_list] == [
        (1_000, 125),
        (1_000, 1_125),
        (500, 2_125),
    ]
    assert isinstance(runs, pd.DataFrame)
    assert runs["run_id"].tolist() == list(range(125, 2_625))


@pytest.mark.parametrize("output_format", ["object", "dataframe"])
def test_list_evaluations_pages_preserve_output(output_format):
    def listing_call(size, offset, **filters):
        assert filters["function"] == "predictive_accuracy"
        assert filters["tag"] == "pagination-test"
        return [
            openml.evaluations.OpenMLEvaluation(
                run_id=run_id,
                task_id=42,
                setup_id=1,
                flow_id=1,
                flow_name="example",
                data_id=1,
                data_name="example",
                function="predictive_accuracy",
                upload_time="2026-01-01",
                uploader=1,
                uploader_name="example",
                value=0.8,
                values=None,
            )
            for run_id in range(offset, offset + size)
        ]

    with unittest.mock.patch.object(
        openml._backend.evaluation,
        "list",
        side_effect=listing_call,
    ) as page_fetch:
        evaluations = openml.evaluations.list_evaluations(
            "predictive_accuracy",
            size=2_500,
            offset=125,
            tag="pagination-test",
            output_format=output_format,
        )

    assert [call.args for call in page_fetch.call_args_list] == [
        (1_000, 125),
        (1_000, 1_125),
        (500, 2_125),
    ]
    if output_format == "dataframe":
        assert isinstance(evaluations, pd.DataFrame)
        assert evaluations["run_id"].tolist() == list(range(125, 2_625))
    else:
        assert isinstance(evaluations, dict)
        assert list(evaluations) == list(range(125, 2_625))
        assert all(
            isinstance(item, openml.evaluations.OpenMLEvaluation) for item in evaluations.values()
        )
