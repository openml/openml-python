# License: BSD 3-Clause

"""Utilities for running benchmarks on OpenML suites with progress tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import openml

if TYPE_CHECKING:
    from openml.runs import OpenMLRun

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def run_suite_with_progress(
    suite_id: int | str,
    model: Any,
    *,
    show_progress: bool = True,
    **run_kwargs: Any,
) -> list[OpenMLRun]:
    """Run a model on all tasks in an OpenML suite with progress tracking.

    Parameters
    ----------
    suite_id : int or str
        OpenML suite ID or alias
    model : estimator
        A scikit-learn compatible estimator to benchmark
    show_progress : bool, default=True
        Whether to display a progress bar (requires tqdm)
    **run_kwargs : dict
        Additional keyword arguments passed to `openml.runs.run_model_on_task`

    Returns
    -------
    list of OpenMLRun
        List of successful run objects

    Raises
    ------
    ImportError
        If show_progress=True but tqdm is not installed

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> runs = run_suite_with_progress(suite_id=99, model=clf)
    >>> print(f"Completed {len(runs)} tasks")
    """
    if show_progress and not TQDM_AVAILABLE:
        raise ImportError(
            "tqdm is required for progress tracking. "
            "Install it with: pip install tqdm\n"
            "Or set show_progress=False"
        )

    # Get the suite
    suite = openml.study.get_suite(suite_id)

    if suite.tasks is None or len(suite.tasks) == 0:
        return []

    # Setup progress bar or plain iterator
    if show_progress:
        task_iterator = tqdm(
            suite.tasks,
            desc=f"Benchmarking {suite.name}",
            unit="task",
        )
    else:
        task_iterator = suite.tasks

    # Run benchmark on each task
    runs = []
    for task_id in task_iterator:
        result = openml.runs.run_model_on_task(model, task=task_id, **run_kwargs)
        run = result[0] if isinstance(result, tuple) else result
        runs.append(run)

    return runs
