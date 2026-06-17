# Mock Remote Server Calls in Tests (#1586)

Several tests rely on direct connections to the live production or test server, causing race conditions, server overload, and sporadic failures. These should be converted to use mocks/patches.

## Overview

| Category | Count | Description |
|----------|-------|-------------|
| `@pytest.mark.production()` | ~41 | Hit the live OpenML production server |
| `@pytest.mark.uses_test_server()` | ~158 | Hit the OpenML test server |
| Already mocked | ~27 | Use `@mock.patch` — no server calls |
| Fully local | ~85 | Pure unit tests, no markers |

## Mocking Strategy

Mock at the `requests.Session` level or patch `openml._api_calls._perform_api_call` to return pre-recorded XML/JSON responses. Store fixtures under `tests/files/mock_responses/<entity>/`.

Existing examples to follow:
- [tests/test_runs/test_run_functions.py:1814](tests/test_runs/test_run_functions.py#L1814) — `@mock.patch.object(requests.Session, "delete")` for delete_run
- [tests/test_flows/test_flow_functions.py:454](tests/test_flows/test_flow_functions.py#L454) — mocking `_perform_api_call`
- [tests/test_utils/test_utils.py:118](tests/test_utils/test_utils.py#L118) — `@unittest.mock.patch("openml._api_calls._perform_api_call")`

---

## Done (Mocked, No Server Calls)

| File | Change |
|------|--------|
| `test_openml/test_api_calls.py` | All 3 tests mocked with `@unittest.mock.patch("requests.Session")`. No server markers. |
| `test_tasks/test_task_methods.py` | `test_get_train_and_test_split_indices` — cache-only, marker removed. `test_tagging` — kept as server test. |
| `test_tasks/test_task_functions.py` | Cache tests unmarked; `_perform_api_call` mock for list/get tasks; `get_dataset` mock + static cache for get_task. Helpers: `_make_task_xml`, `_make_task_list_xml`, `_make_api_side_effect`, `_make_mock_dataset`, `ESTIMATION_PROCEDURES_XML`. |
| `test_tasks/test_classification_task.py` | All 3 tests mocked with `get_dataset` + static cache. Helper: `_make_mock_dataset()`. |
| `test_tasks/test_learning_curve_task.py` | All 3 tests mocked with `get_dataset` + static cache. Same helper. |
| Cache files | `tests/files/org/openml/test/tasks/119/task.xml`, `tests/files/org/openml/test/tasks/801/task.xml` |

**Review (small files):** Mocking follows the strategy correctly. `test_api_calls.py` patches `requests.Session` so all HTTP traffic is intercepted. Task tests use `openml._api_calls._perform_api_call` for list/get and `openml.tasks.functions.get_dataset` for dataset access. Static cache covers tasks 1, 3, 119, 801, 1882. `test_get_task_different_types` mocks both for uncached task IDs (5001, 64, 126033).

---

## Todo (Remaining to Mock)

- [ ] `test_tasks/test_clustering_task.py` — 2 tests
- [ ] `test_tasks/test_regression_task.py` — 2 tests
- [ ] `test_tasks/test_task.py` — 10 tests
- [ ] `test_tasks/test_supervised_task.py` — 16 test server + 3 production (base class for classification/learning curve)
- [ ] `test_setups/test_setup_functions.py` — 10 tests
- [ ] `test_study/test_study_functions.py` — 11 tests
- [ ] `test_datasets/`, `test_flows/`, `test_evaluations/`, `test_runs/` — see breakdown below

---

## File-by-File Breakdown

### 1. `test_datasets/test_dataset_functions.py`

**Production tests** (7) — need mock responses for dataset listing/metadata:
- [Line 143](tests/test_datasets/test_dataset_functions.py#L143): `test_check_datasets_active`
- [Line 182](tests/test_datasets/test_dataset_functions.py#L182): `test__name_to_id_with_deactivated`
- [Line 190](tests/test_datasets/test_dataset_functions.py#L190): `test__name_to_id_with_multiple`
- [Line 196](tests/test_datasets/test_dataset_functions.py#L196): `test__name_to_id`
- [Line 202](tests/test_datasets/test_dataset_functions.py#L202): `test__name_to_id_with_version`
- [Line 285](tests/test_datasets/test_dataset_functions.py#L285): `test_list_datasets_by_tag`
- [Line 1552](tests/test_datasets/test_dataset_functions.py#L1552): (additional production test)

**Test server tests** (58) — representative examples:
- [Line 110](tests/test_datasets/test_dataset_functions.py#L110): `test_tag_untag_dataset`
- [Line 533](tests/test_datasets/test_dataset_functions.py#L533): `test_publish_dataset`
- [Line 699](tests/test_datasets/test_dataset_functions.py#L699): dataset get/download tests
- [Line 891](tests/test_datasets/test_dataset_functions.py#L891): dataset feature tests

**Already mocked** (6):
- [Line 410](tests/test_datasets/test_dataset_functions.py#L410), [Line 525](tests/test_datasets/test_dataset_functions.py#L525), [Line 1735](tests/test_datasets/test_dataset_functions.py#L1735), [Line 1757](tests/test_datasets/test_dataset_functions.py#L1757), [Line 1779](tests/test_datasets/test_dataset_functions.py#L1779), [Line 1798](tests/test_datasets/test_dataset_functions.py#L1798)

---

### 2. `test_datasets/test_dataset.py`

**Production tests** (2) — class-level markers, all methods in class hit production:
- [Line 21](tests/test_datasets/test_dataset.py#L21): `OpenMLDatasetTest` class (`use_production_server()` at [Line 27](tests/test_datasets/test_dataset.py#L27))
- [Line 350](tests/test_datasets/test_dataset.py#L350): `OpenMLDatasetTestSparse` class (`use_production_server()` at [Line 356](tests/test_datasets/test_dataset.py#L356))

**Test server tests** (8):
- [Line 284](tests/test_datasets/test_dataset.py#L284): `test_tagging` — calls `push_tag()`, `remove_tag()`
- [Line 301](tests/test_datasets/test_dataset.py#L301), [Line 310](tests/test_datasets/test_dataset.py#L310), [Line 318](tests/test_datasets/test_dataset.py#L318), [Line 327](tests/test_datasets/test_dataset.py#L327), [Line 339](tests/test_datasets/test_dataset.py#L339), [Line 411](tests/test_datasets/test_dataset.py#L411), [Line 443](tests/test_datasets/test_dataset.py#L443)

---

### 3. `test_evaluations/test_evaluation_functions.py`

**Production tests** (10) — nearly every test hits production:
- [Line 53](tests/test_evaluations/test_evaluation_functions.py#L53): `test_evaluation_list_filter_task`
- [Line 73](tests/test_evaluations/test_evaluation_functions.py#L73): `test_evaluation_list_filter_uploader`
- [Line 88](tests/test_evaluations/test_evaluation_functions.py#L88): `test_evaluation_list_filter_flow`
- [Line 107](tests/test_evaluations/test_evaluation_functions.py#L107): `test_evaluation_list_filter_run`
- [Line 127](tests/test_evaluations/test_evaluation_functions.py#L127): `test_evaluation_list_filter_study`
- [Line 147](tests/test_evaluations/test_evaluation_functions.py#L147): `test_evaluation_list_limit`
- [Line 166](tests/test_evaluations/test_evaluation_functions.py#L166): `test_list_evaluations_setups_filter`
- [Line 204](tests/test_evaluations/test_evaluation_functions.py#L204): `test_evaluation_list_per_fold`
- [Line 242](tests/test_evaluations/test_evaluation_functions.py#L242): `test_evaluation_list_sort`
- [Line 260](tests/test_evaluations/test_evaluation_functions.py#L260): (additional production test)

**Test server tests** (2):
- [Line 158](tests/test_evaluations/test_evaluation_functions.py#L158): `test_list_evaluations_empty`
- [Line 236](tests/test_evaluations/test_evaluation_functions.py#L236): `test_list_evaluation_measures`

---

### 4. `test_flows/test_flow_functions.py`

**Production tests** (10):
- [Line 50](tests/test_flows/test_flow_functions.py#L50): `test_list_flows`
- [Line 61](tests/test_flows/test_flow_functions.py#L61): `test_list_flows_by_tag`
- [Line 70](tests/test_flows/test_flow_functions.py#L70): `test_list_flows_paginate`
- [Line 76](tests/test_flows/test_flow_functions.py#L76): `test_list_flows_empty`
- [Line 84](tests/test_flows/test_flow_functions.py#L84): `test_list_flows_output_format`
- [Line 304](tests/test_flows/test_flow_functions.py#L304): `test_get_flow1`
- [Line 341](tests/test_flows/test_flow_functions.py#L341), [Line 362](tests/test_flows/test_flow_functions.py#L362), [Line 376](tests/test_flows/test_flow_functions.py#L376), [Line 388](tests/test_flows/test_flow_functions.py#L388)

**Test server tests** (5):
- [Line 283](tests/test_flows/test_flow_functions.py#L283), [Line 313](tests/test_flows/test_flow_functions.py#L313), [Line 325](tests/test_flows/test_flow_functions.py#L325), [Line 396](tests/test_flows/test_flow_functions.py#L396), [Line 431](tests/test_flows/test_flow_functions.py#L431)

**Already mocked** (5):
- [Line 454](tests/test_flows/test_flow_functions.py#L454), [Line 474](tests/test_flows/test_flow_functions.py#L474), [Line 494](tests/test_flows/test_flow_functions.py#L494), [Line 514](tests/test_flows/test_flow_functions.py#L514), [Line 531](tests/test_flows/test_flow_functions.py#L531)

---

### 5. `test_flows/test_flow.py`

**Production tests** (3):
- [Line 47](tests/test_flows/test_flow.py#L47): `test_get_flow`
- [Line 80](tests/test_flows/test_flow.py#L80): (flow test)
- [Line 568](tests/test_flows/test_flow.py#L568): (flow test)

**Test server tests** (8):
- [Line 106](tests/test_flows/test_flow.py#L106): `test_tagging`
- [Line 124](tests/test_flows/test_flow.py#L124): `test_from_xml_to_xml` — calls `_perform_api_call` directly at [Line 136](tests/test_flows/test_flow.py#L136)
- [Line 184](tests/test_flows/test_flow.py#L184): `test_publish_flow`
- [Line 226](tests/test_flows/test_flow.py#L226), [Line 277](tests/test_flows/test_flow.py#L277), [Line 369](tests/test_flows/test_flow.py#L369), [Line 387](tests/test_flows/test_flow.py#L387), [Line 428](tests/test_flows/test_flow.py#L428)

**Already mocked** (4):
- [Line 211](tests/test_flows/test_flow.py#L211), [Line 302](tests/test_flows/test_flow.py#L302), [Line 303](tests/test_flows/test_flow.py#L303), [Line 304](tests/test_flows/test_flow.py#L304)

---

### 6. `test_openml/test_api_calls.py`

**Test server tests** (3):
- [Line 18](tests/test_openml/test_api_calls.py#L18): `test_too_long_uri`
- [Line 25](tests/test_openml/test_api_calls.py#L25): `test_retry_on_database_error` — partially mocked at [Line 23](tests/test_openml/test_api_calls.py#L23)
- [Line 120](tests/test_openml/test_api_calls.py#L120): (api calls test)

**Already mocked** (4):
- [Line 23](tests/test_openml/test_api_calls.py#L23), [Line 24](tests/test_openml/test_api_calls.py#L24), [Line 63](tests/test_openml/test_api_calls.py#L63), [Line 83](tests/test_openml/test_api_calls.py#L83)

---

### 7. `test_runs/test_run_functions.py`

**Production tests** (10):
- [Line 1099](tests/test_runs/test_run_functions.py#L1099): `test_get_run`
- [Line 1410](tests/test_runs/test_run_functions.py#L1410): `test_get_run` (format prediction)
- [Line 1445](tests/test_runs/test_run_functions.py#L1445): `test_get_runs_list`
- [Line 1459](tests/test_runs/test_run_functions.py#L1459): `test_get_runs_list_by_task`
- [Line 1478](tests/test_runs/test_run_functions.py#L1478): `test_get_runs_list_by_uploader`
- [Line 1500](tests/test_runs/test_run_functions.py#L1500): `test_get_runs_list_by_flow`
- [Line 1519](tests/test_runs/test_run_functions.py#L1519): `test_get_runs_pagination`
- [Line 1532](tests/test_runs/test_run_functions.py#L1532): `test_get_runs_list_by_filters`
- [Line 1569](tests/test_runs/test_run_functions.py#L1569): `test_get_runs_list_by_tag`
- [Line 1690](tests/test_runs/test_run_functions.py#L1690): `test_format_prediction_non_supervised`

**Test server tests** (35) — heaviest file, includes run-and-upload integration tests:
- [Line 631](tests/test_runs/test_run_functions.py#L631): `test_run_and_upload_logistic_regression`
- [Line 802](tests/test_runs/test_run_functions.py#L802): `test_run_and_upload_gridsearch`
- [Line 825](tests/test_runs/test_run_functions.py#L825): `test_run_and_upload_randomsearch`
- [Line 1176](tests/test_runs/test_run_functions.py#L1176): `test__run_exists`
- [Line 1776](tests/test_runs/test_run_functions.py#L1776): `test_delete_run`
- (30 more at lines listed in overview)

**Already mocked** (6):
- [Line 755](tests/test_runs/test_run_functions.py#L755), [Line 1814](tests/test_runs/test_run_functions.py#L1814), [Line 1834](tests/test_runs/test_run_functions.py#L1834), [Line 1851](tests/test_runs/test_run_functions.py#L1851), [Line 1876](tests/test_runs/test_run_functions.py#L1876), [Line 1954](tests/test_runs/test_run_functions.py#L1954)

---

### 8. `test_runs/test_run.py`

**Test server tests** (6):
- [Line 28](tests/test_runs/test_run.py#L28): `test_tagging`
- [Line 122](tests/test_runs/test_run.py#L122): `test_to_from_filesystem_vanilla`
- [Line 158](tests/test_runs/test_run.py#L158), [Line 193](tests/test_runs/test_run.py#L193), [Line 299](tests/test_runs/test_run.py#L299), [Line 343](tests/test_runs/test_run.py#L343)

---

### 9. `test_runs/test_trace.py`

**No server calls.** All 3 tests are pure unit tests — no changes needed.

---

### 10. `test_setups/test_setup_functions.py`

**Production tests** (3):
- [Line 121](tests/test_setups/test_setup_functions.py#L121): `test_get_setup`
- [Line 138](tests/test_setups/test_setup_functions.py#L138): `test_setup_list_filter_flow`
- [Line 158](tests/test_setups/test_setup_functions.py#L158): `test_list_setups_output_format`

**Test server tests** (7):
- [Line 38](tests/test_setups/test_setup_functions.py#L38): `test_nonexisting_setup_exists`
- [Line 86](tests/test_setups/test_setup_functions.py#L86): `test_existing_setup_exists_1`
- [Line 102](tests/test_setups/test_setup_functions.py#L102): `test_exisiting_setup_exists_2`
- [Line 108](tests/test_setups/test_setup_functions.py#L108): `test_existing_setup_exists_3`
- [Line 150](tests/test_setups/test_setup_functions.py#L150): `test_list_setups_empty`
- [Line 171](tests/test_setups/test_setup_functions.py#L171): `test_setuplist_offset`
- [Line 183](tests/test_setups/test_setup_functions.py#L183): `test_get_cached_setup`

---

### 11. `test_study/test_study_functions.py`

**Production tests** (6):
- [Line 15](tests/test_study/test_study_functions.py#L15): `test_get_study_old`
- [Line 27](tests/test_study/test_study_functions.py#L27), [Line 38](tests/test_study/test_study_functions.py#L38), [Line 48](tests/test_study/test_study_functions.py#L48), [Line 57](tests/test_study/test_study_functions.py#L57), [Line 68](tests/test_study/test_study_functions.py#L68)

**Test server tests** (5):
- [Line 77](tests/test_study/test_study_functions.py#L77): `test_publish_benchmark_suite`
- [Line 146](tests/test_study/test_study_functions.py#L146), [Line 150](tests/test_study/test_study_functions.py#L150), [Line 155](tests/test_study/test_study_functions.py#L155), [Line 225](tests/test_study/test_study_functions.py#L225)

---

### 12. `test_tasks/test_classification_task.py`

**Test server tests** (3):
- [Line 21](tests/test_tasks/test_classification_task.py#L21): `test_download_task`
- [Line 29](tests/test_tasks/test_classification_task.py#L29): `test_class_labels`
- [Line 35](tests/test_tasks/test_classification_task.py#L35): (classification task test)

---

### 13. `test_tasks/test_clustering_task.py`

**Production + test server** (2 tests, dual-marked):
- [Line 23](tests/test_tasks/test_clustering_task.py#L23): `test_get_dataset` (production at [Line 26](tests/test_tasks/test_clustering_task.py#L26))
- [Line 30](tests/test_tasks/test_clustering_task.py#L30): `test_download_task` (production at [Line 34](tests/test_tasks/test_clustering_task.py#L34))

---

### 14. `test_tasks/test_learning_curve_task.py`

**Test server tests** (3):
- [Line 21](tests/test_tasks/test_learning_curve_task.py#L21): `test_get_X_and_Y`
- [Line 30](tests/test_tasks/test_learning_curve_task.py#L30): `test_download_task`
- [Line 37](tests/test_tasks/test_learning_curve_task.py#L37): (learning curve test)

---

### 15. `test_tasks/test_regression_task.py`

**Test server tests** (2):
- [Line 52](tests/test_tasks/test_regression_task.py#L52): `test_get_X_and_Y`
- [Line 61](tests/test_tasks/test_regression_task.py#L61): `test_download_task`

---

### 16. `test_tasks/test_supervised_task.py`

**Production tests** (3):
- [Line 59](tests/test_tasks/test_supervised_task.py#L59), [Line 147](tests/test_tasks/test_supervised_task.py#L147), [Line 215](tests/test_tasks/test_supervised_task.py#L215)

**Test server tests** (16):
- [Line 29](tests/test_tasks/test_supervised_task.py#L29), [Line 37](tests/test_tasks/test_supervised_task.py#L37), [Line 52](tests/test_tasks/test_supervised_task.py#L52), [Line 76](tests/test_tasks/test_supervised_task.py#L76), [Line 86](tests/test_tasks/test_supervised_task.py#L86), [Line 92](tests/test_tasks/test_supervised_task.py#L92), [Line 97](tests/test_tasks/test_supervised_task.py#L97), [Line 105](tests/test_tasks/test_supervised_task.py#L105), [Line 112](tests/test_tasks/test_supervised_task.py#L112), [Line 122](tests/test_tasks/test_supervised_task.py#L122), [Line 139](tests/test_tasks/test_supervised_task.py#L139), [Line 154](tests/test_tasks/test_supervised_task.py#L154), [Line 168](tests/test_tasks/test_supervised_task.py#L168), [Line 191](tests/test_tasks/test_supervised_task.py#L191), [Line 209](tests/test_tasks/test_supervised_task.py#L209), [Line 225](tests/test_tasks/test_supervised_task.py#L225)

**Already mocked** (5):
- [Line 190](tests/test_tasks/test_supervised_task.py#L190), [Line 245](tests/test_tasks/test_supervised_task.py#L245), [Line 265](tests/test_tasks/test_supervised_task.py#L265), [Line 285](tests/test_tasks/test_supervised_task.py#L285), [Line 302](tests/test_tasks/test_supervised_task.py#L302)

---

### 17. `test_tasks/test_task_functions.py`

**Test server tests** (2):
- [Line 19](tests/test_tasks/test_task_functions.py#L19), [Line 35](tests/test_tasks/test_task_functions.py#L35)

---

### 18. `test_tasks/test_task_methods.py`

**Test server tests** (2):
- [Line 35](tests/test_tasks/test_task_methods.py#L35), [Line 39](tests/test_tasks/test_task_methods.py#L39)

---

### 19. `test_tasks/test_task.py`

**Test server tests** (10):
- [Line 51](tests/test_tasks/test_task.py#L51), [Line 56](tests/test_tasks/test_task.py#L56), [Line 62](tests/test_tasks/test_task.py#L62), [Line 75](tests/test_tasks/test_task.py#L75), [Line 86](tests/test_tasks/test_task.py#L86), [Line 93](tests/test_tasks/test_task.py#L93), [Line 101](tests/test_tasks/test_task.py#L101), [Line 108](tests/test_tasks/test_task.py#L108), [Line 119](tests/test_tasks/test_task.py#L119), [Line 144](tests/test_tasks/test_task.py#L144)

---

### 20. `test_utils/test_utils.py`

**Test server tests** (0 explicitly marked, but some call server implicitly):
- Several tests call `openml.utils._list_all()` etc.

**Already mocked** (3):
- [Line 118](tests/test_utils/test_utils.py#L118), [Line 127](tests/test_utils/test_utils.py#L127), [Line 156](tests/test_utils/test_utils.py#L156)

---

## Suggested PR Ordering (by complexity)

1. **Easy wins** — files with few server tests and existing mock patterns:
   - `test_runs/test_trace.py` — already fully local, no work needed
   - `test_openml/test_api_calls.py` — 3 tests, already has mocks
   - `test_tasks/test_task_functions.py` — 2 tests
   - `test_tasks/test_task_methods.py` — 2 tests

2. **Medium** — moderate number of tests, straightforward mocking:
   - `test_tasks/test_task.py` — 10 tests, all test_server
   - `test_tasks/test_regression_task.py` — 2 tests
   - `test_tasks/test_learning_curve_task.py` — 3 tests
   - `test_tasks/test_classification_task.py` — 3 tests
   - `test_tasks/test_clustering_task.py` — 2 tests
   - `test_setups/test_setup_functions.py` — 10 tests
   - `test_study/test_study_functions.py` — 11 tests

3. **Large** — many tests, complex server interactions:
   - `test_evaluations/test_evaluation_functions.py` — 12 tests (almost all production)
   - `test_flows/test_flow.py` — 11 server tests
   - `test_flows/test_flow_functions.py` — 15 server tests
   - `test_datasets/test_dataset.py` — 10 server tests
   - `test_runs/test_run.py` — 6 tests
   - `test_tasks/test_supervised_task.py` — 19 server tests
   - `test_utils/test_utils.py` — needs audit

4. **Heaviest** — require mock response fixtures for run uploads, model training:
   - `test_datasets/test_dataset_functions.py` — 65 server tests
   - `test_runs/test_run_functions.py` — 45 server tests
