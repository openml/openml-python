"""
OpenML introduction
===================

An introduction to OpenML, followed up by a simple example.
"""
import openml
from sklearn import neighbors

############################################################################
# OpenML in Python
# ################
# OpenML is an online collaboration platform for machine learning:
#
# * Find or share interesting, well-documented datasets
# * Define research / modelling goals (tasks)
# * Explore large amounts of machine learning algorithms, with APIs in Java, R, Python
# * Log and share reproducible experiments, models, results
# * Works seamlessly with scikit-learn and other libraries
# * Large scale benchmarking, compare to state of the art
#
# Installation
# ############
#
# * Up to now: pip install git+https://github.com/openml/openml-python.git@develop
# * In the future: pip install openml
# * Check out the installation guide: https://openml.github.io/openml-python/stable/#installation
#
# Authentication
# ##############
#
# * Create an OpenML account (free) on http://www.openml.org.
# * After logging in, open your account page (avatar on the top right)
# * Open 'Account Settings', then 'API authentication' to find your API key.
#
# There are two ways to authenticate:
#
# * Create a plain text file ~/.openml/config with the line 'apikey=MYKEY', replacing MYKEY with your API key.
# * Run the code below, replacing 'YOURKEY' with your API key.

############################################################################
# Uncomment and set your OpenML key. Don't share your key with others.
# oml.config.apikey = 'YOURKEY'
############################################################################
# Download the OpenML task for the eeg-eye-state.
task = openml.tasks.get_task(403)
data = openml.datasets.get_dataset(task.dataset_id)
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
flow = openml.flows.sklearn_to_flow(clf)
try:
    run = openml.runs.run_flow_on_task(flow, task)
    # Publish the experiment on OpenML (optional, requires an API key).
    # For this tutorial, our configuration publishes to the test server
    # as to not pollute the main server.
    myrun = run.publish()
    print("kNN on %s: http://test.openml.org/r/%d" % (data.name, myrun.run_id))
except openml.exceptions.PyOpenMLError as err:
    print("OpenML: {0}".format(err))
