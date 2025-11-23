# License: BSD 3-Clause
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openml.study.benchmarking import run_suite_with_progress
from openml.testing import TestBase


class TestRunSuiteWithProgress(TestBase):
    """Tests for run_suite_with_progress function."""

    def test_import_error_without_tqdm(self):
        """Test that ImportError is raised when tqdm is unavailable."""
        with patch("openml.study.benchmarking.TQDM_AVAILABLE", new=False):
            with pytest.raises(ImportError, match="tqdm is required"):
                run_suite_with_progress(
                    suite_id=99,
                    model=MagicMock(),
                    show_progress=True,
                )

    @patch("openml.study.get_suite")
    @patch("openml.runs.run_model_on_task")
    def test_basic_functionality(self, mock_run_task, mock_get_suite):
        """Test basic benchmarking with mocked suite."""
        # Setup mock suite
        mock_suite = MagicMock()
        mock_suite.study_id = 99
        mock_suite.name = "Test Suite"
        mock_suite.tasks = [1, 2, 3]
        mock_get_suite.return_value = mock_suite

        # Setup mock runs
        mock_runs = [MagicMock(run_id=100), MagicMock(run_id=200), MagicMock(run_id=300)]
        mock_run_task.side_effect = mock_runs

        # Run benchmark without progress bar
        model = MagicMock()
        runs = run_suite_with_progress(suite_id=99, model=model, show_progress=False)

        # Assertions
        assert len(runs) == 3
        assert mock_run_task.call_count == 3

    @patch("openml.study.get_suite")
    def test_empty_suite(self, mock_get_suite):
        """Test handling of empty suite."""
        mock_suite = MagicMock()
        mock_suite.tasks = []
        mock_get_suite.return_value = mock_suite

        runs = run_suite_with_progress(suite_id=99, model=MagicMock(), show_progress=False)

        assert len(runs) == 0
