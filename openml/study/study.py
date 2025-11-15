# License: BSD 3-Clause
# TODO(eddiebergman): Begging for dataclassses to shorten this all
from __future__ import annotations

from typing import Any, Sequence

from openml.base import OpenMLBase
from openml.config import get_server_base_url


class BaseStudy(OpenMLBase):
    """
    An OpenMLStudy represents the OpenML concept of a study. It contains
    the following information: name, id, description, creation date,
    creator id and a set of tags.

    According to this list of tags, the study object receives a list of
    OpenML object ids (datasets, flows, tasks and setups).

    Can be used to obtain all relevant information from a study at once.

    Parameters
    ----------
    study_id : int
        the study id
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    main_entity_type : str
        the entity type (e.g., task, run) that is core in this study.
        only entities of this type can be added explicitly
    benchmark_suite : int (optional)
        the benchmark suite (another study) upon which this study is ran.
        can only be active if main entity type is runs.
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    status : str
        Whether the study is in preparation, active or deactivated
    creation_date : str
        date of creation (meta-info)
    creator : int
        openml user id of the owner / creator
    tags : list(dict)
        The list of tags shows which tags are associated with the study.
        Each tag is a dict of (tag) name, window_start and write_access.
    data : list
        a list of data ids associated with this study
    tasks : list
        a list of task ids associated with this study
    flows : list
        a list of flow ids associated with this study
    runs : list
        a list of run ids associated with this study
    setups : list
        a list of setup ids associated with this study
    """

    def __init__(  # noqa: PLR0913
        self,
        study_id: int | None,
        alias: str | None,
        main_entity_type: str,
        benchmark_suite: int | None,
        name: str,
        description: str,
        status: str | None,
        creation_date: str | None,
        creator: int | None,
        tags: list[dict] | None,
        data: list[int] | None,
        tasks: list[int] | None,
        flows: list[int] | None,
        runs: list[int] | None,
        setups: list[int] | None,
    ):
        self.study_id = study_id
        self.alias = alias
        self.main_entity_type = main_entity_type
        self.benchmark_suite = benchmark_suite
        self.name = name
        self.description = description
        self.status = status
        self.creation_date = creation_date
        self.creator = creator
        self.tags = tags  # LEGACY. Can be removed soon
        self.data = data
        self.tasks = tasks
        self.flows = flows
        self.setups = setups
        self.runs = runs

    @classmethod
    def _entity_letter(cls) -> str:
        return "s"

    @property
    def id(self) -> int | None:
        """Return the id of the study."""
        return self.study_id

    def _get_repr_body_fields(self) -> Sequence[tuple[str, str | int | list[str]]]:
        """Collect all information to display in the __repr__ body."""
        fields: dict[str, Any] = {
            "Name": self.name,
            "Status": self.status,
            "Main Entity Type": self.main_entity_type,
        }
        if self.study_id is not None:
            fields["ID"] = self.study_id
            fields["Study URL"] = self.openml_url
        if self.creator is not None:
            fields["Creator"] = f"{get_server_base_url()}/u/{self.creator}"
        if self.creation_date is not None:
            fields["Upload Time"] = self.creation_date.replace("T", " ")
        if self.data is not None:
            fields["# of Data"] = len(self.data)
        if self.tasks is not None:
            fields["# of Tasks"] = len(self.tasks)
        if self.flows is not None:
            fields["# of Flows"] = len(self.flows)
        if self.runs is not None:
            fields["# of Runs"] = len(self.runs)

        # determines the order in which the information will be printed
        order = [
            "ID",
            "Name",
            "Status",
            "Main Entity Type",
            "Study URL",
            "# of Data",
            "# of Tasks",
            "# of Flows",
            "# of Runs",
            "Creator",
            "Upload Time",
        ]
        return [(key, fields[key]) for key in order if key in fields]

    def _parse_publish_response(self, xml_response: dict) -> None:
        """Parse the id from the xml_response and assign it to self."""
        self.study_id = int(xml_response["oml:study_upload"]["oml:id"])

    def _to_dict(self) -> dict[str, dict]:
        """Creates a dictionary representation of self."""
        # some can not be uploaded, e.g., id, creator, creation_date
        simple_props = ["alias", "main_entity_type", "name", "description"]

        # TODO(eddiebergman): Begging for a walrus if we can drop 3.7
        simple_prop_values = {}
        for prop_name in simple_props:
            content = getattr(self, prop_name, None)
            if content is not None:
                simple_prop_values["oml:" + prop_name] = content

        # maps from attribute name (which is used as outer tag name) to immer
        # tag name e.g., self.tasks -> <oml:tasks><oml:task_id>1987</oml:task_id></oml:tasks>
        complex_props = {"tasks": "task_id", "runs": "run_id"}

        # TODO(eddiebergman): Begging for a walrus if we can drop 3.7
        complex_prop_values = {}
        for prop_name, inner_name in complex_props.items():
            content = getattr(self, prop_name, None)
            if content is not None:
                complex_prop_values["oml:" + prop_name] = {"oml:" + inner_name: content}

        return {
            "oml:study": {
                "@xmlns:oml": "http://openml.org/openml",
                **simple_prop_values,
                **complex_prop_values,
            }
        }

    def push_tag(self, tag: str) -> None:
        """Add a tag to the study."""
        raise NotImplementedError("Tags for studies is not (yet) supported.")

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the study."""
        raise NotImplementedError("Tags for studies is not (yet) supported.")


class OpenMLStudy(BaseStudy):
    """
    An OpenMLStudy represents the OpenML concept of a study (a collection of runs).

    It contains the following information: name, id, description, creation date,
    creator id and a list of run ids.

    According to this list of run ids, the study object receives a list of
    OpenML object ids (datasets, flows, tasks and setups).

    Parameters
    ----------
    study_id : int
        the study id
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    benchmark_suite : int (optional)
        the benchmark suite (another study) upon which this study is ran.
        can only be active if main entity type is runs.
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    status : str
        Whether the study is in preparation, active or deactivated
    creation_date : str
        date of creation (meta-info)
    creator : int
        openml user id of the owner / creator
    tags : list(dict)
        The list of tags shows which tags are associated with the study.
        Each tag is a dict of (tag) name, window_start and write_access.
    data : list
        a list of data ids associated with this study
    tasks : list
        a list of task ids associated with this study
    flows : list
        a list of flow ids associated with this study
    runs : list
        a list of run ids associated with this study
    setups : list
        a list of setup ids associated with this study
    """

    def __init__(  # noqa: PLR0913
        self,
        study_id: int | None,
        alias: str | None,
        benchmark_suite: int | None,
        name: str,
        description: str,
        status: str | None,
        creation_date: str | None,
        creator: int | None,
        tags: list[dict] | None,
        data: list[int] | None,
        tasks: list[int] | None,
        flows: list[int] | None,
        runs: list[int] | None,
        setups: list[int] | None,
    ):
        super().__init__(
            study_id=study_id,
            alias=alias,
            main_entity_type="run",
            benchmark_suite=benchmark_suite,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=data,
            tasks=tasks,
            flows=flows,
            runs=runs,
            setups=setups,
        )


class OpenMLBenchmarkSuite(BaseStudy):
    """
    An OpenMLBenchmarkSuite represents the OpenML concept of a suite (a collection of tasks).

    It contains the following information: name, id, description, creation date,
    creator id and the task ids.

    According to this list of task ids, the suite object receives a list of
    OpenML object ids (datasets).

    Parameters
    ----------
    suite_id : int
        the study id
    alias : str (optional)
        a string ID, unique on server (url-friendly)
    main_entity_type : str
        the entity type (e.g., task, run) that is core in this study.
        only entities of this type can be added explicitly
    name : str
        the name of the study (meta-info)
    description : str
        brief description (meta-info)
    status : str
        Whether the study is in preparation, active or deactivated
    creation_date : str
        date of creation (meta-info)
    creator : int
        openml user id of the owner / creator
    tags : list(dict)
        The list of tags shows which tags are associated with the study.
        Each tag is a dict of (tag) name, window_start and write_access.
    data : list
        a list of data ids associated with this study
    tasks : list
        a list of task ids associated with this study
    """

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

    @property
    def metadata(self):
        """
        Return a pandas DataFrame containing dataset metadata joined with the
        suite's task ids.

        This property returns the canonical DataFrame that users can
        format/export themselves (e.g., to LaTeX).
        """
        import pandas as pd
        import openml
        from openml.tasks import get_task

        tasks = [get_task(tid, download_data=False, download_qualities=False)
                 for tid in (self.tasks or [])]
        dataset_ids = [t.dataset_id for t in tasks]

        if not dataset_ids:
            return pd.DataFrame(
                columns=[
                    "did",
                    "name",
                    "NumberOfInstances",
                    "NumberOfFeatures",
                    "NumberOfClasses",
                    "tid",
                ]
            )

        metadata = openml.datasets.list_datasets(data_id=dataset_ids, output_format="dataframe")

        if "did" not in metadata.columns:
            for alt in ("data_id", "id"):
                if alt in metadata.columns:
                    metadata = metadata.rename(columns={alt: "did"})
                    break

        if metadata.index.name == "did":
            metadata = metadata.reset_index()
        if "did" not in metadata.columns:
            raise RuntimeError("expected dataset metadata to contain 'did' column")

        task_rows = [(t.dataset_id, t.id) for t in tasks]
        task_df = (
            pd.DataFrame(task_rows, columns=["did", "tid"])
            .drop_duplicates(subset="did")
            .set_index("did")
        )

        metadata = metadata.set_index("did").join(task_df, how="left").reset_index()

        rename_map = {
            "NumberOfInstances": "n",
            "NumberOfFeatures": "p",
            "NumberOfClasses": "C",
        }
        rename_actual = {k: v for k, v in rename_map.items() if k in metadata.columns}
        metadata = metadata.rename(columns=rename_actual)

        for col in ("n", "p", "C"):
            if col in metadata.columns:
                try:
                    metadata[col] = metadata[col].astype("Int64")
                except Exception:
                    pass

        if "name" in metadata.columns:
            metadata = metadata.sort_values("name", key=lambda s: s.str.lower())

        return metadata.reset_index(drop=True)

    def to_latex(self, filename: str, caption: str | None = None, label: str | None = None) -> str:
        """
        Write a LaTeX table for this benchmark suite's metadata to `filename`.

        This method uses the DataFrame returned by `self.metadata`. It first tries
        to use pandas' Styler (which requires jinja2). If jinja2 is not installed,
        it falls back to a simple, dependency-free LaTeX table generator.
        """
        import html
        import importlib
        import pandas as pd

        df = self.metadata

        preferred = ["did", "tid", "name", "n", "p", "C"]
        cols = [c for c in preferred if c in df.columns]
        df_present = df[cols].copy()
        rename = {"did": "Dataset ID", "tid": "Task ID", "name": "Name", "n": "n", "p": "p", "C": "C"}
        df_present = df_present.rename(columns={k: v for k, v in rename.items() if k in df_present.columns})

        # Try the nicer pandas Styler (requires jinja2). If unavailable, fallback:
        try:
            # Only attempt Styler if jinja2 is importable
            if importlib.util.find_spec("jinja2") is not None:
                styler = df_present.style.hide(axis="index")
                latex_body = styler.to_latex() if hasattr(styler, "to_latex") else df_present.to_latex(index=False, escape=False)
            else:
                raise ImportError("jinja2 not present")
        except Exception:
            # Simple fallback generator (no jinja2 required)
            def _escape_cell(val: object) -> str:
                if pd.isna(val):
                    return ""
                s = str(val)
                # basic LaTeX-escaping for underscore, percent, ampersand, #, $, {, }, ~, ^, \
                replace_map = {
                    "\\": r"\textbackslash{}",
                    "&": r"\&",
                    "%": r"\%",
                    "$": r"\$",
                    "#": r"\#",
                    "_": r"\_",
                    "{": r"\{",
                    "}": r"\}",
                    "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}",
                }
                for k, v in replace_map.items():
                    s = s.replace(k, v)
                # escape backslashes introduced by html escaping
                return s

            headers = list(df_present.columns)
            # build tabular column spec: left-aligned for text, r for numbers (simple heuristic)
            def _col_spec(col_name):
                if pd.api.types.is_numeric_dtype(df_present[col_name]):
                    return "r"
                return "l"

            colspec = "".join(_col_spec(c) for c in headers)
            lines = []
            lines.append(r"\begin{longtable}{" + colspec + "}")
            # header row
            header_row = " & ".join(_escape_cell(h) for h in headers) + r" \\"
            lines.append(r"\toprule")
            lines.append(header_row)
            lines.append(r"\midrule")
            lines.append(r"\endfirsthead")
            # repeated head (simple)
            lines.append(r"\toprule")
            lines.append(header_row)
            lines.append(r"\midrule")
            lines.append(r"\endhead")
            # body rows
            for _, row in df_present.iterrows():
                row_cells = [_escape_cell(row[c]) for c in headers]
                lines.append(" & ".join(row_cells) + r" \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{longtable}")
            latex_body = "\n".join(lines)

        # Wrap in table environment if caption/label provided
        if caption or label:
            parts = ["\\begin{table}"]
            if caption:
                parts.append(f"\\caption{{{caption}}}")
            if label:
                parts.append(f"\\label{{{label}}}")
            parts.append(latex_body)
            parts.append("\\end{table}")
            latex = "\n".join(parts)
        else:
            latex = latex_body

        if not filename.endswith(".tex"):
            filename = filename + ".tex"

        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(latex)

        return latex