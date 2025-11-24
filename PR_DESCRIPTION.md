## feat: Add CLI Commands for Browsing and Searching OpenML Runs

#### Metadata
* Reference Issue: Enhancement #1486 - Add CLI commands for browsing and searching OpenML flows (models)
* New Tests Added: Yes
* Documentation Updated: No (CLI help text serves as documentation)
* Change Log Entry: "Add CLI commands for browsing and searching OpenML runs: `openml runs list`, `openml runs info`, and `openml runs download`"

#### Details

### What does this PR implement/fix?

This PR adds three new CLI subcommands under `openml runs` to improve the user experience of the run catalogue:

1. **`openml runs list`** - List runs with optional filtering (task_id, flow_id, uploader, tag, pagination, output format)
2. **`openml runs info <run_id>`** - Display detailed information about a specific run including task, flow, evaluations, and parameter settings
3. **`openml runs download <run_id>`** - Download a run and save predictions to local cache

### Why is this change necessary? What is the problem it solves?

Currently, users must write Python code to browse or search OpenML runs, even for simple tasks like listing runs for a specific task or downloading run results. This creates a barrier to entry and makes the run catalogue less accessible. Adding CLI commands allows users to interact with the run catalogue directly from the command line without writing code.

This directly addresses the **ESoC 2025 goal** of "Improving user experience of the run catalogue in AIoD and OpenML".

### How can I reproduce the issue this PR is solving and its solution?

**Before (requires Python code):**
```python
import openml
runs = openml.runs.list_runs(task=[1], size=10)
for rid, run_dict in runs.items():
    print(f"{rid}: Task {run_dict['task_id']}")
```

**After (CLI commands):**
```bash
# List first 10 runs for a specific task
openml runs list --task 1 --size 10

# List runs by a specific uploader
openml runs list --uploader "John Doe"

# Get detailed info about a run
openml runs info 12345

# Download a run and cache predictions
openml runs download 12345

# List runs for a specific flow, formatted as table
openml runs list --flow 42 --format table --verbose

# Filter by both task and flow with JSON output
openml runs list --task 1 --flow 42 --format json
```

### Implementation Details

**Files Modified:**
- `openml/cli.py` - Added three new functions and integration into main CLI parser
  - `runs_list()` - List runs with filtering and formatting
  - `runs_info()` - Display detailed run information
  - `runs_download()` - Download and cache runs
  - Helper functions: `_format_runs_output()`, `_format_runs_table()`, `_format_runs_list()`, `_print_run_evaluations()`
  - `runs()` - Dispatcher for runs subcommands
  - Updated `main()` to register runs subparser

**Files Created:**
- `tests/test_openml/test_cli.py` - Comprehensive test suite with 18 tests
  - Tests for all three commands (list, info, download)
  - Tests for different output formats (list, table, json)
  - Tests for verbose mode
  - Tests for error handling
  - Tests for argument parsing
  - All tests use mocked API calls (no server dependency)

**Key Features:**
- ✅ Multiple output formats: list (default), table, json
- ✅ Filtering options: task, flow, uploader, tag
- ✅ Pagination support: size, offset
- ✅ Verbose mode for detailed information
- ✅ Proper error handling with user-friendly messages
- ✅ Uses existing `openml.runs.list_runs()` and `openml.runs.get_run()` functions
- ✅ No changes to core API
- ✅ Follows existing CLI patterns (similar to configure command)

### Testing

All tests pass successfully:
```
======================= 18 passed in 0.16s ========================
```

**Test Coverage:**
- `test_runs_list_simple` - Basic list functionality
- `test_runs_list_with_filters` - Filtering with task, flow, uploader, tag
- `test_runs_list_verbose` - Verbose output mode
- `test_runs_list_table_format` - Table format output
- `test_runs_list_json_format` - JSON format output
- `test_runs_list_empty_results` - Empty result handling
- `test_runs_list_error_handling` - Error scenarios
- `test_runs_info` - Detailed run information display
- `test_runs_info_with_fold_evaluations` - Fold evaluation summary
- `test_runs_info_error_handling` - Info error scenarios
- `test_runs_download` - Download and cache functionality
- `test_runs_download_error_handling` - Download error scenarios
- `test_runs_dispatcher` - Command routing
- `test_runs_dispatcher_invalid_subcommand` - Invalid command handling
- Integration tests for argument parsing

### Code Quality

- ✅ All pre-commit hooks pass (ruff, mypy, formatting)
- ✅ No breaking changes
- ✅ Follows project code style and patterns
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Helper functions extracted to reduce complexity

### CLI Help Output

```bash
$ openml runs --help
usage: openml runs [-h] {list,info,download} ...

Browse and search OpenML runs from the command line.

positional arguments:
  {list,info,download}
    list                List runs with optional filtering.
    info                Display detailed information about a specific run.
    download            Download a run and cache it locally.

$ openml runs list --help
usage: openml runs list [-h] [--task TASK] [--flow FLOW] [--uploader UPLOADER]
                        [--tag TAG] [--size SIZE] [--offset OFFSET]
                        [--format {list,table,json}] [--verbose]

List runs with optional filtering.

options:
  --task TASK           Filter by task ID
  --flow FLOW           Filter by flow ID
  --uploader UPLOADER   Filter by uploader name or ID
  --tag TAG             Filter by tag
  --size SIZE           Number of runs to retrieve (default: 10)
  --offset OFFSET       Offset for pagination (default: 0)
  --format {list,table,json}
                        Output format (default: list)
  --verbose             Show detailed information
```

### Benefits

1. **Lower Barrier to Entry** - Users can explore runs without writing Python code
2. **Quick Prototyping** - Faster iteration when searching for specific runs
3. **Scripting Support** - JSON output enables shell script integration
4. **Consistent Interface** - Matches existing OpenML CLI patterns
5. **Offline Access** - Download command enables working without constant connectivity

### Future Enhancements (Optional)

Potential improvements for future PRs:
- Add `--sort` option for custom ordering
- Support multiple task/flow IDs in filters (comma-separated)
- Add `--export` option to save results to file
- Implement `openml runs compare` to compare multiple runs
- Add progress bars for download operations

### Screenshots/Examples

**List Command (Simple):**
```bash
$ openml runs list --task 1 --size 3
1: Task 1
2: Task 1
3: Task 1
```

**List Command (Table Format):**
```bash
$ openml runs list --task 1 --size 3 --format table
run_id  task_id  flow_id  uploader  upload_time
1       1        100      1         2024-01-01 10:00:00
2       1        101      2         2024-01-02 11:00:00
3       1        100      1         2024-01-03 12:00:00
```

**Info Command:**
```bash
$ openml runs info 12345
Run ID: 12345
Task ID: 1
Task Type: Supervised Classification
Flow ID: 100
Flow Name: sklearn.ensemble.RandomForestClassifier
Setup ID: 12445
Dataset ID: 1
Uploader: John Doe (ID: 1)

Parameter Settings:
  n_estimators: 100
  max_depth: 10

Evaluations:
  predictive_accuracy: 0.95
  area_under_roc_curve: 0.98

Tags: test, openml-python

Predictions URL: https://test.openml.org/predictions/12345
```

### Any other comments?

- All code follows BSD 3-Clause license
- No external dependencies added (uses existing pandas, which is already a dependency)
- Backward compatible - purely additive functionality
- Ready for review and merge into `develop` branch
- Documentation can be added to mkdocs if desired (currently CLI help text is comprehensive)

This implementation makes the OpenML run catalogue significantly more accessible and user-friendly, aligning with the project's goals of improving the user experience for both novice and experienced users.
