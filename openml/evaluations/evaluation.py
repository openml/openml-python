# License: BSD 3-Clause
from __future__ import annotations

from dataclasses import asdict, dataclass

import openml.datasets
import openml.flows
import openml.runs
import openml.tasks


@dataclass
class OpenMLEvaluation:
    """
    Contains all meta-information about a run / evaluation combination,
    according to the evaluation/list function
    """

    run_id: int
    task_id: int
    setup_id: int
    flow_id: int
    flow_name: str
    data_id: int
    data_name: str
    function: str
    upload_time: str
    uploader: int
    uploader_name: str
    value: float | None = None
    values: list[float] | None = None
    array_data: str | None = None

    def _to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        header = "OpenML Evaluation"
        header = f"{header}\n{'=' * len(header)}\n"

        fields = {
            "Upload Date": self.upload_time,
            "Run ID": self.run_id,
            "OpenML Run URL": openml.runs.OpenMLRun.url_for_id(self.run_id),
            "Task ID": self.task_id,
            "OpenML Task URL": openml.tasks.OpenMLTask.url_for_id(self.task_id),
            "Flow ID": self.flow_id,
            "OpenML Flow URL": openml.flows.OpenMLFlow.url_for_id(self.flow_id),
            "Setup ID": self.setup_id,
            "Data ID": self.data_id,
            "Data Name": self.data_name,
            "OpenML Data URL": openml.datasets.OpenMLDataset.url_for_id(self.data_id),
            "Metric Used": self.function,
            "Result": self.value,
        }

        order = [
            "Upload Date",
            "Run ID",
            "OpenML Run URL",
            "Task ID",
            "OpenML Task URL",
            "Flow ID",
            "OpenML Flow URL",
            "Setup ID",
            "Data ID",
            "Data Name",
            "OpenML Data URL",
            "Metric Used",
            "Result",
        ]

        _fields = [(key, fields[key]) for key in order if key in fields]

        longest_field_name_length = max(len(name) for name, _ in _fields)
        field_line_format = f"{{:.<{longest_field_name_length}}}: {{}}"

        body = "\n".join(field_line_format.format(name, value) for name, value in _fields)

        return header + body
