# License: BSD 3-Clause
from __future__ import annotations

import unittest

import openml

class TestEvaluationsExample(unittest.TestCase):
    def test_example_python_paper(self):
        # Example script which will appear in the upcoming OpenML-Python paper
        # This test ensures that the example will keep running!
        with openml.config.overwrite_config_context(  # noqa: F823
            {
                "server": "https://www.openml.org/api/v1/xml",
                "apikey": None,
            }
        ):
            import matplotlib.pyplot as plt
            import numpy as np

            df = openml.evaluations.list_evaluations_setups(
                "predictive_accuracy",
                flows=[8353],
                tasks=[6],
                parameters_in_separate_columns=True,
            )  # Choose an SVM flow, for example 8353, and a task.

            assert len(df) > 0, (
                "No evaluation found for flow 8353 on task 6, could "
                "be that this task is not available on the test server."
            )

            hp_names = ["sklearn.svm.classes.SVC(16)_C", "sklearn.svm.classes.SVC(16)_gamma"]
            df[hp_names] = df[hp_names].astype(float).apply(np.log)
            C, gamma, score = df[hp_names[0]], df[hp_names[1]], df["value"]

            cntr = plt.tricontourf(C, gamma, score, levels=12, cmap="RdBu_r")
            plt.colorbar(cntr, label="accuracy")
            plt.xlim((min(C), max(C)))
            plt.ylim((min(gamma), max(gamma)))
            plt.xlabel("C (log10)", size=16)
            plt.ylabel("gamma (log10)", size=16)
            plt.title("SVM performance landscape", size=20)

            plt.tight_layout()
