"""
OpenML Run Example
==================

An example of an automated machine learning experiment.
"""
import openml
from sklearn import tree, preprocessing, pipeline

############################################################################
# .. warning:: This example uploads data. For that reason, this example
#   connects to the test server at test.openml.org. This prevents the main
#   server from crowding with example datasets, tasks, runs, and so on.

openml.config.start_using_configuration_for_example()
############################################################################

# Uncomment and set your OpenML key. Don't share your key with others.
# openml.config.apikey = 'YOURKEY'

# Define a scikit-learn pipeline
clf = pipeline.Pipeline(
    steps=[
        ('imputer', preprocessing.Imputer()),
        ('estimator', tree.DecisionTreeClassifier())
    ]
)
############################################################################
# Download the OpenML task for the german credit card dataset.
task = openml.tasks.get_task(97)
############################################################################
# Run the scikit-learn model on the task (requires an API key).
run = openml.runs.run_model_on_task(clf, task)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
openml.config.stop_using_configuration_for_example()
