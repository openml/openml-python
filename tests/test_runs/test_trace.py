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
                        setup_string="parameter_%d%d%d" % (i, j, k),
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
        with pytest.raises(ValueError, match="Trace content not available."):
            OpenMLRunTrace.generate(attributes="foo", content=None)
        with pytest.raises(ValueError, match="Trace attributes not available."):
            OpenMLRunTrace.generate(attributes=None, content="foo")
        with pytest.raises(ValueError, match="Trace content is empty."):
            OpenMLRunTrace.generate(attributes="foo", content=[])
        with pytest.raises(ValueError, match="Trace_attributes and trace_content not compatible:"):
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
            match="Either `setup_string` or `parameters` needs to be passed as argument.",
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
