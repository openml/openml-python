"""
OpenML Run Example
==================

An example of an automated machine learning experiment.
"""
import openml
import sklearn


# Uncomment and set your OpenML key. Don't share your key with others.
# openml.config.apikey = 'YOURKEY'
# Define a scikit-learn pipeline
clf = sklearn.pipeline.Pipeline(
    steps=[
        ('imputer', sklearn.preprocessing.Imputer()),
        ('estimator', sklearn.tree.DecisionTreeClassifier())
    ]
)
# Download the OpenML task for the german credit card dataset with 10-fold
# cross-validation.
task = openml.tasks.get_task(31)

# Run the scikit-learn model on the task (requires an API key).
run = openml.runs.run_model_on_task(task, clf)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
