# License: BSD 3-Clause
from __future__ import annotations

import pytest

from openml.runs import OpenMLRunTrace, OpenMLTraceIteration
from openml.testing import TestBase


class TestTrace(TestBase):
    def test_get_selected_iteration(self):
        trace_iterations = {}
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    t = OpenMLTraceIteration(
                        repeat=i,
                        fold=j,
                        iteration=5,
                        setup_string=f"parameter_{i}{j}{k}",
                        evaluation=1.0 * i + 0.1 * j + 0.01 * k,
                        selected=(i == j and i == k and i == 2),
                        parameters=None,
                    )
                    trace_iterations[(i, j, k)] = t

        trace = OpenMLRunTrace(-1, trace_iterations=trace_iterations)
        # This next one should simply not fail
        assert trace.get_selected_iteration(2, 2) == 2
        with pytest.raises(
            ValueError, match="Could not find the selected iteration for rep/fold 3/3"
        ):
            trace.get_selected_iteration(3, 3)

    def test_initialization(self):
        """Check all different ways to fail the initialization"""
        with pytest.raises(ValueError, match=r"Trace content not available."):
            OpenMLRunTrace.generate(attributes="foo", content=None)
        with pytest.raises(ValueError, match=r"Trace attributes not available."):
            OpenMLRunTrace.generate(attributes=None, content="foo")
        with pytest.raises(ValueError, match=r"Trace content is empty."):
            OpenMLRunTrace.generate(attributes="foo", content=[])
        with pytest.raises(ValueError, match=r"Trace_attributes and trace_content not compatible:"):
            OpenMLRunTrace.generate(attributes=["abc"], content=[[1, 2]])

    def test_duplicate_name(self):
        # Test that the user does not pass a parameter which has the same name
        # as one of the required trace attributes
        trace_attributes = [
            ("repeat", "NUMERICAL"),
            ("fold", "NUMERICAL"),
            ("iteration", "NUMERICAL"),
            ("evaluation", "NUMERICAL"),
            ("selected", ["true", "false"]),
            ("repeat", "NUMERICAL"),
        ]
        trace_content = [[0, 0, 0, 0.5, "true", 1], [0, 0, 0, 0.9, "false", 2]]
        with pytest.raises(
            ValueError,
            match=r"Either `setup_string` or `parameters` needs to be passed as argument.",
        ):
            OpenMLRunTrace.generate(trace_attributes, trace_content)

        trace_attributes = [
            ("repeat", "NUMERICAL"),
            ("fold", "NUMERICAL"),
            ("iteration", "NUMERICAL"),
            ("evaluation", "NUMERICAL"),
            ("selected", ["true", "false"]),
            ("sunshine", "NUMERICAL"),
        ]
        trace_content = [[0, 0, 0, 0.5, "true", 1], [0, 0, 0, 0.9, "false", 2]]
        with pytest.raises(
            ValueError,
            match="Encountered unknown attribute sunshine that does not start with "
            "prefix parameter_",
        ):
            OpenMLRunTrace.generate(trace_attributes, trace_content)

    def test_trace_iteration_get_parameters_with_json_string(self):
        setup_str = (
            '{"parameter_alpha": "0.1", "parameter_max_depth": "5", '
            '"parameter_kernel": "\\"linear\\""}'
        )
        t = OpenMLTraceIteration(
            repeat=0,
            fold=0,
            iteration=0,
            setup_string=setup_str,
            evaluation=0.95,
            selected=True,
        )
        assert t.get_parameters() == {"alpha": 0.1, "max_depth": 5, "kernel": "linear"}

    def test_trace_iteration_get_parameters_with_dict(self):
        t = OpenMLTraceIteration(
            repeat=0,
            fold=0,
            iteration=0,
            setup_string={"parameter_alpha": "0.1", "parameter_max_depth": 5},
            evaluation=0.95,
            selected=True,
        )
        assert t.get_parameters() == {"alpha": 0.1, "max_depth": 5}

    def test_trace_iteration_get_parameters_with_parameters_dict(self):
        t = OpenMLTraceIteration(
            repeat=0,
            fold=0,
            iteration=0,
            parameters={"parameter_alpha": 0.1, "max_depth": 5},
            evaluation=0.95,
            selected=True,
        )
        assert t.get_parameters() == {"alpha": 0.1, "max_depth": 5}

    def test_trace_iteration_invalid_setup_string(self):
        t1 = OpenMLTraceIteration(
            repeat=0,
            fold=0,
            iteration=0,
            setup_string="invalid json",
            evaluation=0.95,
            selected=True,
        )
        with pytest.raises(ValueError, match="Could not parse `setup_string` as JSON"):
            t1.get_parameters()

        t2 = OpenMLTraceIteration(
            repeat=0,
            fold=0,
            iteration=0,
            setup_string="[1, 2, 3]",
            evaluation=0.95,
            selected=True,
        )
        with pytest.raises(
            ValueError, match=r"`setup_string` must represent a dictionary of parameters"
        ):
            t2.get_parameters()

    def test_trace_iteration_type_errors(self):
        with pytest.raises(
            TypeError, match=r"argument parameters is not an instance of dict, but <class 'int'>"
        ):
            OpenMLTraceIteration(
                repeat=0,
                fold=0,
                iteration=0,
                parameters=123,  # type: ignore
                evaluation=0.95,
                selected=True,
            )

        with pytest.raises(
            TypeError,
            match=r"argument setup_string is not an instance of str or dict, but <class 'int'>",
        ):
            OpenMLTraceIteration(
                repeat=0,
                fold=0,
                iteration=0,
                setup_string=123,  # type: ignore
                evaluation=0.95,
                selected=True,
            )

    def test_trace_to_arff_with_string_setup_string(self):
        trace_iterations = {
            (0, 0, 0): OpenMLTraceIteration(
                repeat=0,
                fold=0,
                iteration=0,
                setup_string='{"parameter_alpha": "0.1"}',
                evaluation=0.8,
                selected=True,
            )
        }
        trace = OpenMLRunTrace(run_id=1, trace_iterations=trace_iterations)
        arff_dict = trace.trace_to_arff()
        assert arff_dict["attributes"] == [
            ("repeat", "NUMERIC"),
            ("fold", "NUMERIC"),
            ("iteration", "NUMERIC"),
            ("evaluation", "NUMERIC"),
            ("selected", ["true", "false"]),
            ("parameter_alpha", "STRING"),
        ]
        assert arff_dict["data"] == [[0, 0, 0, 0.8, "true", 0.1]]
