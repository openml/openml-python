"""
OpenML Run Example
==================

An example of an automated machine learning experiment.
"""
import openml
from sklearn import tree, preprocessing, pipeline

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
