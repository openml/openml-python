# Final Implementation Plan: Metadata Property for OpenMLBenchmarkSuite
## Issue #1126: Support for Exporting Benchmarking Suites to LaTeX

---

## Executive Summary

This plan implements a `metadata` property on the `OpenMLBenchmarkSuite` class that returns a pandas DataFrame combining task-level and dataset-level metadata. This enables researchers to easily generate LaTeX tables for academic publications using pandas' `to_latex()` method.

**Key Corrections from Initial Plan:**
- `list_tasks()` returns DataFrame with `tid` as INDEX, not a column
- `list_tasks()` doesn't accept `task_id` parameter directly - must use internal `_list_tasks()`
- No `output_format` parameter needed (always returns DataFrame)
- Proper error handling using OpenML exceptions

---

## 1. Architecture Overview

### 1.1 Data Flow
```
OpenMLBenchmarkSuite.metadata
    ↓
1. Check cache (_metadata)
    ↓ (if not cached)
2. Call openml.tasks.functions._list_tasks(task_id=suite.tasks)
    → Returns DataFrame with 'tid' as index
    → Contains: tid, did, name, task_type, status, NumberOfInstances, etc.
    ↓
3. Extract unique dataset IDs (did)
    ↓
4. Call openml.datasets.list_datasets(data_id=unique_dids)
    → Returns DataFrame with dataset metadata
    → Contains: did, version, uploader, name, etc.
    ↓
5. Merge DataFrames on 'did' (left join)
    ↓
6. Cache result in self._metadata
    ↓
7. Return DataFrame
```

### 1.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use `_list_tasks()` directly | Public API doesn't support `task_id` filtering |
| Left merge on tasks DataFrame | Preserves one row per task (suite structure) |
| Lazy loading with caching | Avoids API calls during initialization |
| Handle index properly | `list_tasks` returns `tid` as index, need to reset |

---

## 2. Implementation Details

### 2.1 File to Modify
- **File**: `openml/study/study.py`
- **Class**: `OpenMLBenchmarkSuite` (lines 265-332)

### 2.2 Required Imports

Add to existing imports at top of `openml/study/study.py` (after line 8):

```python
import pandas as pd

from openml.tasks.functions import _list_tasks
from openml.datasets.functions import list_datasets
from openml.exceptions import OpenMLServerException
```

**Note**: 
- We use `_list_tasks` (internal function) because the public `list_tasks()` doesn't accept `task_id` parameter
- This is acceptable for internal library use within the same package
- `pandas` should already be a dependency, but verify it's imported

### 2.3 Modify `__init__` Method

Add cache initialization in `OpenMLBenchmarkSuite.__init__`:

```python
def __init__(  # noqa: PLR0913
    self,
    suite_id: int | None,
    alias: str | None,
    name: str,
    description: str,
    status: str | None,
    creation_date: str | None,
    creator: int | None,
    tags: list[dict] | None,
    data: list[int] | None,
    tasks: list[int] | None,
):
    super().__init__(
        study_id=suite_id,
        alias=alias,
        main_entity_type="task",
        benchmark_suite=None,
        name=name,
        description=description,
        status=status,
        creation_date=creation_date,
        creator=creator,
        tags=tags,
        data=data,
        tasks=tasks,
        flows=None,
        runs=None,
        setups=None,
    )
    # Initialize metadata cache
    self._metadata: pd.DataFrame | None = None
```

### 2.4 Implement `metadata` Property

Add this property method to `OpenMLBenchmarkSuite` class (after `__init__`):

```python
@property
def metadata(self) -> pd.DataFrame:
    """
    Returns a pandas DataFrame containing metadata for all tasks in the suite.

    The DataFrame includes:
    - Task-level information: task ID (tid), task type, estimation procedure,
      target feature, evaluation measure
    - Dataset-level information: dataset ID (did), dataset name, version,
      uploader, number of instances, number of features, number of classes,
      and other dataset qualities

    The result is cached after the first access. Subsequent calls return the
    cached DataFrame without making additional API calls.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per task in the suite. The DataFrame is indexed
        by the default integer index. Columns include both task and dataset metadata.

    Raises
    ------
    RuntimeError
        If task metadata cannot be retrieved from the OpenML server.
    OpenMLServerException
        If there is an error communicating with the OpenML server.

    Examples
    --------
    >>> import openml
    >>> suite = openml.study.get_suite(99)  # OpenML-CC18
    >>> meta = suite.metadata
    >>> print(meta.columns.tolist()[:5])  # First 5 columns
    ['tid', 'did', 'name', 'task_type', 'status']
    
    >>> # Export to LaTeX
    >>> columns = ['name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']
    >>> latex_table = meta[columns].style.to_latex(
    ...     caption="Dataset Characteristics",
    ...     label="tab:suite_metadata"
    ... )
    """
    # Return cached result if available
    if self._metadata is not None:
        return self._metadata

    # Handle empty suites gracefully
    if not self.tasks:
        self._metadata = pd.DataFrame()
        return self._metadata

    # Step 1: Fetch Task Metadata
    # Use internal _list_tasks because public API doesn't support task_id filtering
    try:
        # _list_tasks requires limit and offset as positional arguments
        # Since we're filtering by specific task_ids, we need at least len(self.tasks) limit
        # Use a limit larger than the number of tasks to ensure we get all of them
        task_df = _list_tasks(
            limit=max(len(self.tasks), 1000),  # Request enough for all tasks
            offset=0,
            task_id=self.tasks,  # Pass as kwarg - will be converted to comma-separated string
        )
        
        # _list_tasks returns DataFrame with 'tid' as index (from pd.DataFrame.from_dict(..., orient="index"))
        # Reset index to make 'tid' a column for easier merging
        if task_df.index.name == 'tid':
            task_df = task_df.reset_index()
        # If index is RangeIndex but we have 'tid' column, that's fine
        # If index is RangeIndex and no 'tid' column, something went wrong
        
        # Verify we got the expected tasks
        if len(task_df) == 0:
            # No tasks found - return empty DataFrame
            self._metadata = pd.DataFrame()
            return self._metadata
        
        # Ensure 'tid' column exists (should after reset_index if index was named 'tid')
        if 'tid' not in task_df.columns:
            # This shouldn't happen, but handle gracefully
            raise RuntimeError(
                f"Task metadata missing 'tid' column. Columns: {task_df.columns.tolist()}"
            )
            
    except OpenMLServerException as e:
        raise RuntimeError(
            f"Failed to retrieve task metadata for suite {self.id}: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error retrieving task metadata for suite {self.id}: {e}"
        ) from e

    # Step 2: Extract unique dataset IDs and fetch dataset metadata
    if "did" in task_df.columns and len(task_df) > 0:
        unique_dids = task_df["did"].unique().tolist()
        
        try:
            dataset_df = list_datasets(data_id=unique_dids)
        except OpenMLServerException as e:
            raise RuntimeError(
                f"Failed to retrieve dataset metadata: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error retrieving dataset metadata: {e}"
            ) from e

        # Step 3: Merge DataFrames
        # Use left join to preserve all tasks (one row per task)
        # Apply suffixes to handle column name collisions
        self._metadata = pd.merge(
            task_df,
            dataset_df,
            on="did",
            how="left",
            suffixes=("", "_dataset")
        )
    else:
        # Fallback: return task DataFrame only if 'did' column is missing
        self._metadata = task_df

    return self._metadata
```

### 2.5 Critical Implementation Notes

1. **Index Handling**: `_list_tasks` returns DataFrame with `tid` as index. We need to reset it:
   ```python
   if task_df.index.name == 'tid':
       task_df = task_df.reset_index()
   ```

2. **Empty Suite Handling**: Return empty DataFrame, don't raise error.

3. **Error Handling**: Use `OpenMLServerException` for API errors, `RuntimeError` for unexpected errors.

4. **Column Name Collisions**: Use suffixes `("", "_dataset")` so task columns take precedence.

---

## 3. Testing Strategy

### 3.1 Test File Location
Create: `tests/test_study/test_benchmark_suite_metadata.py`

### 3.2 Unit Tests (with Mocks)

```python
# License: BSD 3-Clause
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from openml.study import OpenMLBenchmarkSuite
from openml.testing import TestBase


class TestBenchmarkSuiteMetadata(TestBase):
    """Test suite for OpenMLBenchmarkSuite.metadata property."""

    def setUp(self):
        """Create a test suite instance."""
        super().setUp()
        self.suite = OpenMLBenchmarkSuite(
            suite_id=99,
            alias="test-suite",
            name="Test Suite",
            description="A test suite",
            status="active",
            creation_date="2022-01-01",
            creator=1,
            tags=None,
            data=None,
            tasks=[1, 2, 3]
        )

    @patch("openml.study.study.list_datasets")
    @patch("openml.study.study._list_tasks")
    def test_metadata_basic_structure(self, mock_list_tasks, mock_list_datasets):
        """Test that metadata returns a DataFrame with expected structure."""
        # Mock task response (with tid as index)
        task_data = {
            1: {"tid": 1, "did": 10, "name": "Task1", "NumberOfInstances": 100},
            2: {"tid": 2, "did": 11, "name": "Task2", "NumberOfInstances": 200},
            3: {"tid": 3, "did": 10, "name": "Task3", "NumberOfInstances": 150},
        }
        task_df = pd.DataFrame.from_dict(task_data, orient="index")
        task_df.index.name = "tid"
        mock_list_tasks.return_value = task_df

        # Mock dataset response
        dataset_df = pd.DataFrame({
            "did": [10, 11],
            "version": [1, 1],
            "uploader": [5, 5],
            "name": ["Dataset1", "Dataset2"]
        })
        mock_list_datasets.return_value = dataset_df

        # Access property
        metadata = self.suite.metadata

        # Assertions
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) == 3  # One row per task
        assert "tid" in metadata.columns
        assert "did" in metadata.columns
        assert "version" in metadata.columns
        assert "NumberOfInstances" in metadata.columns

        # Verify API calls
        mock_list_tasks.assert_called_once()
        mock_list_datasets.assert_called_once()

    @patch("openml.study.study._list_tasks")
    def test_metadata_caching(self, mock_list_tasks):
        """Test that metadata is cached after first access."""
        task_df = pd.DataFrame({
            "tid": [1],
            "did": [10],
            "name": ["Task1"]
        })
        mock_list_tasks.return_value = task_df

        # First access
        meta1 = self.suite.metadata
        # Second access
        meta2 = self.suite.metadata

        # Should be same object (cached)
        assert meta1 is meta2
        # Should only call API once
        assert mock_list_tasks.call_count == 1

    def test_metadata_empty_suite(self):
        """Test metadata for suite with no tasks."""
        empty_suite = OpenMLBenchmarkSuite(
            suite_id=1,
            alias=None,
            name="Empty Suite",
            description="",
            status="active",
            creation_date="2022-01-01",
            creator=1,
            tags=None,
            data=None,
            tasks=[]  # Empty tasks
        )

        metadata = empty_suite.metadata
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) == 0

    @patch("openml.study.study.list_datasets")
    @patch("openml.study.study._list_tasks")
    def test_metadata_merge_behavior(self, mock_list_tasks, mock_list_datasets):
        """Test that merge preserves task structure (left join)."""
        # Task with dataset that doesn't exist in dataset_df
        task_df = pd.DataFrame({
            "tid": [1, 2],
            "did": [10, 99],  # did=99 doesn't exist in dataset_df
            "name": ["Task1", "Task2"]
        })
        task_df.index.name = "tid"
        mock_list_tasks.return_value = task_df.reset_index()

        dataset_df = pd.DataFrame({
            "did": [10],
            "version": [1]
        })
        mock_list_datasets.return_value = dataset_df

        metadata = self.suite.metadata

        # Should have 2 rows (one per task)
        assert len(metadata) == 2
        # Task 1 should have version
        assert metadata.loc[metadata["tid"] == 1, "version"].iloc[0] == 1
        # Task 2 should have NaN for version (missing dataset)
        assert pd.isna(metadata.loc[metadata["tid"] == 2, "version"].iloc[0])

    @patch("openml.study.study._list_tasks")
    def test_metadata_error_handling(self, mock_list_tasks):
        """Test error handling when API calls fail."""
        from openml.exceptions import OpenMLServerException

        mock_list_tasks.side_effect = OpenMLServerException("Server error", code=500)

        with pytest.raises(RuntimeError, match="Failed to retrieve task metadata"):
            _ = self.suite.metadata
```

### 3.3 Integration Test (Optional, for verification)

```python
@pytest.mark.production()
def test_metadata_with_real_suite():
    """Integration test with real OpenML server (optional)."""
    import openml
    
    suite = openml.study.get_suite(99)  # OpenML-CC18
    metadata = suite.metadata
    
    # Basic checks
    assert isinstance(metadata, pd.DataFrame)
    assert len(metadata) > 0
    assert "tid" in metadata.columns
    assert "did" in metadata.columns
    
    # Test LaTeX export
    latex = metadata.head().style.to_latex()
    assert "\\begin{tabular}" in latex
```

---

## 4. Documentation

### 4.1 Docstring
The property docstring (shown in section 2.4) includes:
- Description of returned data
- Column information
- Caching behavior
- Examples including LaTeX export

### 4.2 Example Script
Create: `examples/Advanced/suite_metadata_latex_export.py`

```python
"""
Example: Exporting Benchmark Suite Metadata to LaTeX

This example demonstrates how to use the metadata property
to generate LaTeX tables for academic publications.
"""
import openml

# Get a benchmark suite
suite = openml.study.get_suite(99)  # OpenML-CC18

# Access metadata
metadata = suite.metadata

# Select columns for publication
columns = [
    'name',
    'NumberOfInstances',
    'NumberOfFeatures',
    'NumberOfClasses',
    'NumberOfMissingValues'
]

# Generate LaTeX table with formatting
latex_table = metadata[columns].style \
    .format({
        "NumberOfInstances": "{:,}",
        "NumberOfFeatures": "{:d}",
        "NumberOfClasses": "{:d}"
    }) \
    .hide(axis="index") \
    .to_latex(
        caption="Dataset Characteristics for OpenML-CC18",
        label="tab:cc18_metadata",
        hrules=True,
        position="H"
    )

print(latex_table)

# Save to file
with open("suite_metadata.tex", "w") as f:
    f.write(latex_table)
```

---

## 5. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add imports to `openml/study/study.py`
- [ ] Initialize `_metadata` cache in `__init__`
- [ ] Implement `metadata` property with:
  - [ ] Cache check
  - [ ] Empty suite handling
  - [ ] Task metadata retrieval using `_list_tasks`
  - [ ] Index reset handling
  - [ ] Dataset metadata retrieval
  - [ ] DataFrame merge
  - [ ] Error handling

### Phase 2: Testing
- [ ] Create test file `test_benchmark_suite_metadata.py`
- [ ] Write unit tests:
  - [ ] Basic structure test
  - [ ] Caching test
  - [ ] Empty suite test
  - [ ] Merge behavior test
  - [ ] Error handling test
- [ ] Run tests: `pytest tests/test_study/test_benchmark_suite_metadata.py`

### Phase 3: Documentation
- [ ] Verify docstring is complete
- [ ] Create example script
- [ ] Test example script manually
- [ ] Update CHANGELOG (if project maintains one)

### Phase 4: Code Quality
- [ ] Run linter: `pre-commit run --all-files`
- [ ] Fix any style issues
- [ ] Verify type hints are correct
- [ ] Check for unused imports

### Phase 5: Final Verification
- [ ] Test with real suite (e.g., suite 99)
- [ ] Verify LaTeX export works
- [ ] Check DataFrame column names match expectations
- [ ] Test edge cases (empty suite, missing datasets)

---

## 6. Potential Issues & Solutions

### Issue 1: `_list_tasks` is internal function
**Solution**: This is acceptable for internal library use. Document that we're using it for batch filtering by task_id.

### Issue 2: Index handling complexity
**Solution**: Check index name and type, reset if needed. Handle both cases (indexed by tid vs. RangeIndex).

### Issue 3: Column name collisions
**Solution**: Use merge suffixes. Task columns take precedence (empty suffix), dataset columns get "_dataset" suffix.

### Issue 4: Missing datasets
**Solution**: Use left join - tasks without matching datasets will have NaN values for dataset columns.

---

## 7. Expected Output

### 7.1 DataFrame Structure
```
Index: RangeIndex (0, 1, 2, ...)
Columns:
  - tid (int): Task ID
  - did (int): Dataset ID
  - name (str): Task/Dataset name
  - task_type (str): Type of task
  - status (str): Task status
  - estimation_procedure (str): CV strategy
  - target_feature (str): Target variable name
  - NumberOfInstances (int): Number of rows
  - NumberOfFeatures (int): Number of columns
  - NumberOfClasses (int): Number of classes (classification)
  - NumberOfMissingValues (int): Missing value count
  - ... (other quality columns)
  - version (int): Dataset version
  - uploader (int): Uploader ID
  - ... (other dataset columns with "_dataset" suffix if collision)
```

### 7.2 Usage Example Output
```python
>>> suite = openml.study.get_suite(99)
>>> meta = suite.metadata
>>> print(meta.shape)
(72, 35)  # 72 tasks, 35 columns

>>> print(meta.columns.tolist()[:10])
['tid', 'did', 'name', 'task_type', 'status', 'estimation_procedure', 
 'target_feature', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']
```

---

## 8. Success Criteria

✅ Property returns pandas DataFrame  
✅ DataFrame has one row per task in suite  
✅ Contains both task and dataset metadata  
✅ Caching works (no duplicate API calls)  
✅ Handles empty suites gracefully  
✅ Error handling is informative  
✅ LaTeX export works with pandas Styler  
✅ Unit tests pass  
✅ Code follows project style guidelines  

---

## 9. Next Steps After Implementation

1. Create Pull Request with:
   - Implementation code
   - Tests
   - Example script
   - Updated documentation

2. PR Description should include:
   - Reference to issue #1126
   - Brief explanation of approach
   - Usage example
   - Testing results

3. Address reviewer feedback:
   - May need to adjust error messages
   - Could add more edge case handling
   - Documentation improvements

---

## 10. References

- Issue: https://github.com/openml/openml-python/issues/1126
- OpenML Documentation: https://docs.openml.org/
- Pandas Styler: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html

---

---

## 11. Quick Reference: Key Code Snippets

### 11.1 Complete Property Implementation (Copy-Paste Ready)

```python
@property
def metadata(self) -> pd.DataFrame:
    """Returns a pandas DataFrame containing metadata for all tasks in the suite."""
    if self._metadata is not None:
        return self._metadata

    if not self.tasks:
        self._metadata = pd.DataFrame()
        return self._metadata

    try:
        task_df = _list_tasks(
            limit=max(len(self.tasks), 1000),
            offset=0,
            task_id=self.tasks,
        )
        
        if task_df.index.name == 'tid':
            task_df = task_df.reset_index()
        
        if len(task_df) == 0:
            self._metadata = pd.DataFrame()
            return self._metadata
        
        if 'tid' not in task_df.columns:
            raise RuntimeError(
                f"Task metadata missing 'tid' column. Columns: {task_df.columns.tolist()}"
            )
            
    except OpenMLServerException as e:
        raise RuntimeError(
            f"Failed to retrieve task metadata for suite {self.id}: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error retrieving task metadata for suite {self.id}: {e}"
        ) from e

    if "did" in task_df.columns and len(task_df) > 0:
        unique_dids = task_df["did"].unique().tolist()
        
        try:
            dataset_df = list_datasets(data_id=unique_dids)
        except OpenMLServerException as e:
            raise RuntimeError(f"Failed to retrieve dataset metadata: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error retrieving dataset metadata: {e}") from e

        self._metadata = pd.merge(
            task_df,
            dataset_df,
            on="did",
            how="left",
            suffixes=("", "_dataset")
        )
    else:
        self._metadata = task_df

    return self._metadata
```

### 11.2 Test Mock Setup

```python
# In test file
from unittest.mock import patch
import pandas as pd

@patch("openml.study.study.list_datasets")
@patch("openml.study.study._list_tasks")
def test_example(self, mock_list_tasks, mock_list_datasets):
    # Setup mocks
    task_df = pd.DataFrame.from_dict({
        1: {"tid": 1, "did": 10, "name": "Task1"},
    }, orient="index")
    task_df.index.name = "tid"
    mock_list_tasks.return_value = task_df
    
    dataset_df = pd.DataFrame({"did": [10], "version": [1]})
    mock_list_datasets.return_value = dataset_df
    
    # Test
    metadata = self.suite.metadata
    assert len(metadata) == 1
```

---

## 12. Common Pitfalls to Avoid

1. ❌ **Don't forget to reset index** - `_list_tasks` returns DataFrame with `tid` as index
2. ❌ **Don't use public `list_tasks()`** - It doesn't support `task_id` parameter
3. ❌ **Don't forget empty suite check** - Return empty DataFrame, don't raise error
4. ❌ **Don't use inner join** - Use left join to preserve all tasks
5. ❌ **Don't forget error handling** - Wrap API calls in try/except
6. ❌ **Don't skip caching** - Initialize `_metadata = None` in `__init__`

---

**End of Implementation Plan**

**Last Updated**: Based on codebase analysis of openml-python repository
**Status**: Ready for implementation

