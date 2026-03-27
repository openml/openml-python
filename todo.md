# TODOs in `tests/test_runs/test_run_functions.py`

## 1. Line 170 — Assert holdout task

**Context:** `_rerun_model_and_compare_predictions` fetches a run by ID and re-runs the model, but never verifies the task type.

```python
# TODO: assert holdout task
```

**Resolution:** The method is called from `_run_and_upload` which handles both holdout (classification) and cross-validation (regression) tasks — so a blanket "assert holdout" isn't correct here. Instead, assert that the run's task type matches what the caller expects. The task type is available on the task object:

```python
task = openml.tasks.get_task(run.task_id)
assert task.task_type_id in (
    TaskType.SUPERVISED_CLASSIFICATION,
    TaskType.SUPERVISED_REGRESSION,
), f"Unexpected task type: {task.task_type_id}"
```

However, since `_rerun_model_and_compare_predictions` doesn't receive the expected task type, and the callers already validate behavior through prediction comparison, this TODO is low value. Consider removing the comment.

---

## 2. Lines 287 & 291 — Assert on `_to_xml()` and `trace_to_arff()` output

**Context:** In `_perform_run`, the XML and ARFF trace outputs are generated but never checked.

```python
# This is only a smoke check right now
# TODO add a few asserts here
run._to_xml()
```

```python
# This is only a smoke check right now
# TODO add a few asserts here
run.trace.trace_to_arff()
```

**Resolution:** Add basic structural assertions:

For `_to_xml()` (line 288):
```python
xml_output = run._to_xml()
assert isinstance(xml_output, str)
assert len(xml_output) > 0
assert "<oml:run" in xml_output
assert f"<oml:task_id>{task_id}</oml:task_id>" in xml_output
```

For `trace_to_arff()` (line 292):
```python
trace_arff = run.trace.trace_to_arff()
assert isinstance(trace_arff, dict)
assert "data" in trace_arff
assert len(trace_arff["data"]) > 0
```

---

## 3. Lines 341–346 — Compare trace objects between local run and downloaded run

**Context:** After uploading and re-downloading a run, the test verifies tags but skips comparing trace data.

```python
# TODO make sure that these attributes are instantiated when
# downloading a run? Or make sure that the trace object is created when
# running a flow on a task (and not only the arff object is created,
# so that the two objects can actually be compared):
# downloaded_run_trace = downloaded._generate_trace_arff_dict()
# self.assertEqual(run_trace, downloaded_run_trace)
```

**Resolution:** The downloaded run from `get_run()` does not always have the trace populated (it depends on server-side processing). This comparison would be flaky because:
- The server may not have processed the run yet when it's downloaded.
- The trace is only present for search-based estimators (GridSearchCV, RandomizedSearchCV).

A safe resolution is to check that the trace is present when expected (i.e., when the original run had one):
```python
if run.trace is not None:
    downloaded = openml.runs.get_run(run_.run_id)
    # Trace may not be immediately available after upload
    if downloaded.trace is not None:
        assert len(downloaded.trace.trace_iterations) == len(run.trace.trace_iterations)
```

Alternatively, remove the TODO and keep the comment explaining why the comparison is not done, since the flakiness concern is valid.

---

## 4. Line 514 — Mock `_wait_for_processed_run` for trace initialization

**Context:** In `_run_and_upload`, when the classifier is a `BaseSearchCV`, the test waits up to 600 seconds for the server to process the run before downloading the best model from the trace.

```python
# TODO: mock this? We have the arff already on the server
self._wait_for_processed_run(run.run_id, 600)
```

**Resolution:** This is about test performance, not correctness. Mocking this would require:
1. Mocking `openml.runs.get_run_trace()` to return the local trace object.
2. Mocking `openml.runs.initialize_model_from_trace()` to use local data.

This would remove the integration aspect of the test. Since these tests are already marked `@pytest.mark.uses_test_server()` (i.e., they're integration tests), the wait is intentional. The better fix is to keep the wait but reduce `max_waiting_time_seconds` if server performance improves, or split the trace-dependent assertions into a separate test marked with a longer timeout.

Recommended: Remove the TODO comment and add a note explaining why mocking is not appropriate here (integration test by design).

---

## 5. Line 555 — Check if runtime is present in fold evaluations

**Context:** In `_run_and_upload`, after the run completes, `_check_fold_timing_evaluations` is called, which already validates timing measures exist and are within bounds.

```python
# todo: check if runtime is present
self._check_fold_timing_evaluations(
    fold_evaluations=run.fold_evaluations,
    num_repeats=1,
    num_folds=num_folds,
    task_type=task_type,
)
```

**Resolution:** This TODO is effectively already done — `_check_fold_timing_evaluations` (defined in `openml/testing.py:223`) checks that keys like `usercpu_time_millis`, `wall_clock_time_millis_training`, `wall_clock_time_millis_testing`, etc. are present and within valid bounds. The TODO comment is stale and should simply be removed.

---

## 6. Lines 1447, 1461, 1480, 1502, 1521, 1534 — "Comes from live, no such lists on test"

**Context:** Six `list_runs` tests have the same TODO:

```python
# TODO: comes from live, no such lists on test
```

**Resolution:** These are informational comments, not actionable TODOs. They explain why the tests use `@pytest.mark.production()` and call `self.use_production_server()` — the test server doesn't have enough data for these list operations. No code change is needed. Either:
- Remove the `TODO` prefix since it's not a task (change to a plain comment), or
- Leave as-is; they serve as documentation for why production server is required.

Recommended: Change `# TODO:` to `# NOTE:` to avoid confusion.
