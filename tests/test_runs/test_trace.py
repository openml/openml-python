import unittest

from openml.runs import OpenMLRunTrace, OpenMLTraceIteration


class TestTrace(unittest.TestCase):
    def test_get_selected_iteration(self):
        trace_iterations = {}
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    t = OpenMLTraceIteration(
                        repeat=i,
                        fold=j,
                        iteration=5,
                        setup_string='parameter_%d%d%d' % (i, j, k),
                        evaluation=1.0 * i + 0.1 * j + 0.01 * k,
                        selected=(i == j and i == k and i == 2),
                        paramaters=None,
                    )
                    trace_iterations[(i, j, k)] = t

        trace = OpenMLRunTrace(-1, trace_iterations=trace_iterations)
        # This next one should simply not fail
        self.assertEqual(trace.get_selected_iteration(2, 2), 2)
        with self.assertRaisesRegex(
            ValueError,
                'Could not find the selected iteration for rep/fold 3/3',
        ):

            trace.get_selected_iteration(3, 3)

    def test_initialization(self):
        """Check all different ways to fail the initialization """
        rval = OpenMLRunTrace.generate(None, None)
        self.assertIsNone(rval)
        with self.assertRaisesRegex(
            ValueError,
            'Trace content not available.',
        ):
            OpenMLRunTrace.generate(attributes='foo', content=None)
        with self.assertRaisesRegex(
            ValueError,
            'Trace content is empty.'
        ):
            OpenMLRunTrace.generate(attributes='foo', content=[])
        with self.assertRaisesRegex(
            ValueError,
            'Trace_attributes and trace_content not compatible:'
        ):
            OpenMLRunTrace.generate(attributes=['abc'], content=[[1, 2]])