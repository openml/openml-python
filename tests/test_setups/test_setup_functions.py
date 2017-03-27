import sys
import hashlib
import time

import openml
import openml.exceptions
from openml.testing import TestBase

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock


def get_sentinel():
    # Create a unique prefix for the flow. Necessary because the flow is
    # identified by its name and external version online. Having a unique
    #  name allows us to publish the same flow in each test run
    md5 = hashlib.md5()
    md5.update(str(time.time()).encode('utf-8'))
    sentinel = md5.hexdigest()[:10]
    sentinel = 'TEST%s' % sentinel
    return sentinel



class TestRun(TestBase):

    def test_nonexisting_setup_exists(self):
        from sklearn.tree import DecisionTreeClassifier
        # first publish a nonexiting flow
        sentinel = get_sentinel()
        dectree = DecisionTreeClassifier()
        flow = openml.flows.sklearn_to_flow(dectree)
        flow.name = 'TEST%s%s' % (sentinel, flow.name)
        flow.publish()

        # although the flow exists, we can be sure there are no
        # setups (yet) as it hasn't been ran
        setup_id = openml.setups.setup_exists(flow, dectree)
        self.assertEquals(setup_id, False)


    def test_existing_setup_exists(self):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        # first publish a nonexiting flow
        bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=5,
                                                           min_samples_split=1),
                                    n_estimators=3,
                                    max_samples=0.5)
        flow = openml.flows.sklearn_to_flow(bagging)
        flow.name = 'TEST%s%s' % (get_sentinel(), flow.name)
        flow = flow.publish()
        flow = openml.flows.get_flow(flow.flow_id)

        # although the flow exists, we can be sure there are no
        # setups (yet) as it hasn't been ran
        setup_id = openml.setups.setup_exists(flow, bagging)
        self.assertEquals(setup_id, False)

        # now run the flow on an easy task:
        task = openml.tasks.get_task(115) #diabetes
        run = openml.runs.run_task(task, bagging)
        # spoof flow id, otherwise the sentinel is ignored
        run.flow_id = flow.flow_id
        run = run.publish()
        # download the run, as it contains the right setup id
        run = openml.runs.get_run(run.run_id)

        # execute the function we are interested in
        setup_id = openml.setups.setup_exists(flow, bagging)
        self.assertEquals(setup_id, run.setup_id)

