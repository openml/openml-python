# License: BSD 3-Clause
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openml.tasks import TaskType


@dataclass
class OpenMLEstimationProcedure:
    """
    Contains all meta-information about a run / evaluation combination,
    according to the evaluation/list function

    Parameters
    ----------
    id : int
        ID of estimation procedure
    task_type_id : TaskType
        Assosiated task type
    name : str
        Name of estimation procedure
    type : str
        Type of estimation procedure
    """

    id: int
    task_type_id: TaskType
    name: str
    type: str

    def _to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        header = "OpenML Estimation Procedure"
        header = f"{header}\n{'=' * len(header)}\n"

        fields = {
            "ID": self.id,
            "Task Type": self.task_type_id,
            "Name": self.name,
            "Type": self.type,
        }

        order = [
            "ID",
            "Name",
            "Type",
            "Task Type",
        ]

        _fields = [(key, fields[key]) for key in order if key in fields]

        longest_field_name_length = max(len(name) for name, _ in _fields)
        field_line_format = f"{{:.<{longest_field_name_length}}}: {{}}"
        body = "\n".join(field_line_format.format(name, value) for name, value in _fields)
        return header + body
