# License: BSD 3-Clause
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import sklearn.metrics
from tqdm import tqdm

import openml

if TYPE_CHECKING:
    from openml.runs import OpenMLRun
    from openml.study import OpenMLBenchmarkSuite
    from openml.tasks import OpenMLTask


class OpenMLBenchmarkRunner:
    """Run an estimator against every task in an OpenML benchmark suite.

    Parameters
    ----------
    benchmark_suite : OpenMLBenchmarkSuite | str | int
        Suite object, alias (e.g. "OpenML-CC18"), or integer suite ID.
    estimator : sklearn-compatible estimator
    n_jobs : int, optional (default=1)
        Number of parallel workers. -1 uses one thread per task.
    upload_runs : bool, optional (default=True)
        Whether to publish runs to the OpenML server.
    """

    def __init__(
        self,
        benchmark_suite: OpenMLBenchmarkSuite | str | int,
        estimator: Any,
        *,
        n_jobs: int = 1,
        upload_runs: bool = True,
    ) -> None:
        if isinstance(benchmark_suite, (str, int)):
            benchmark_suite = openml.study.get_suite(benchmark_suite)

        self.benchmark_suite: OpenMLBenchmarkSuite = benchmark_suite
        self.estimator: Any = estimator
        self.n_jobs: int = n_jobs
        self.upload_runs: bool = upload_runs
        self.results: dict[int, dict[str, Any]] = {}

    def run(self) -> dict[int, dict[str, Any]]:
        """Run the benchmark and return results dict keyed by task_id."""
        tasks = self.benchmark_suite.tasks
        if tasks is None:
            raise ValueError("Benchmark suite has no tasks.")

        pending = [tid for tid in tasks if tid not in self.results]

        if not pending:
            return self.results

        workers = len(pending) if self.n_jobs == -1 else max(1, self.n_jobs)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Future[dict[str, Any]], int] = {
                executor.submit(self._run_single_task, tid): tid for tid in pending
            }

            with tqdm(as_completed(futures), total=len(pending), desc="Benchmark") as pbar:
                for future in pbar:
                    result = future.result()
                    self.results[result["task_id"]] = result
                    self._log_task_result(result, pbar)

        self._log_summary()
        return self.results

    def _run_single_task(self, task_id: int) -> dict[str, Any]:
        try:
            task: OpenMLTask = openml.tasks.get_task(task_id)

            run_obj = openml.runs.run_model_on_task(self.estimator, task)

            if isinstance(run_obj, tuple):
                run: OpenMLRun = run_obj[0]
            else:
                run = run_obj

            score = run.get_metric_fn(sklearn.metrics.accuracy_score)

            run_id: int | None = None
            if self.upload_runs:
                run.publish()
                run_id = run.run_id

            return {
                "task_id": task_id,
                "dataset_name": task.get_dataset().name,
                "accuracy": float(score.mean()),
                "run_id": run_id,
                "status": "success",
            }

        except Exception as e:  # noqa: BLE001
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
            }

    def _log_task_result(
        self,
        result: dict[str, Any],
        pbar: tqdm,
    ) -> None:
        if result["status"] == "success":
            pbar.write(
                f"✅ {result['dataset_name']} "
                f"(task {result['task_id']}): "
                f"accuracy={result['accuracy']:.4f}"
            )
        else:
            pbar.write(f"❌ Task {result['task_id']} failed: {result['error']}")

    def _log_summary(self) -> None:
        successful = [r for r in self.results.values() if r["status"] == "success"]

        [r for r in self.results.values() if r["status"] == "failed"]

        (sum(r["accuracy"] for r in successful) / len(successful) if successful else 0.0)
