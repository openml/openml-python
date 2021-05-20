"""
==================
Obtaining Flow IDs
==================

This tutorial discusses different ways to obtain the ID of a flow in order to perform further
analysis.
"""

####################################################################################################

# License: BSD 3-Clause

import sklearn.tree

import openml


############################################################################
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()


############################################################################
# Defining a classifier
clf = sklearn.tree.DecisionTreeClassifier()

####################################################################################################
# 1. Obtaining a flow given a classifier
# ======================================
#

flow = openml.extensions.get_extension_by_model(clf).model_to_flow(clf).publish()
flow_id = flow.flow_id
print(flow_id)

####################################################################################################
# This piece of code is rather involved. First, it retrieves a
# :class:`~openml.extensions.Extension` which is registered and can handle the given model,
# in our case it is :class:`openml.extensions.sklearn.SklearnExtension`. Second, the extension
# converts the classifier into an instance of :class:`openml.OpenMLFlow`. Third and finally,
# the publish method checks whether the current flow is already present on OpenML. If not,
# it uploads the flow, otherwise, it updates the current instance with all information computed
# by the server (which is obviously also done when uploading/publishing a flow).
#
# To simplify the usage we have created a helper function which automates all these steps:

flow_id = openml.flows.get_flow_id(model=clf)
print(flow_id)

####################################################################################################
# 2. Obtaining a flow given its name
# ==================================
# The schema of a flow is given in XSD (`here
# <https://github.com/openml/OpenML/blob/master/openml_OS/views/pages/api_new/v1/xsd/openml.implementation.upload.xsd>`_).  # noqa E501
# Only two fields are required, a unique name, and an external version. While it should be pretty
# obvious why we need a name, the need for the additional external version information might not
# be immediately clear. However, this information is very important as it allows to have multiple
# flows with the same name for different versions of a software. This might be necessary if an
# algorithm or implementation introduces, renames or drop hyperparameters over time.

print(flow.name, flow.external_version)

####################################################################################################
# The name and external version are automatically added to a flow when constructing it from a
# model. We can then use them to retrieve the flow id as follows:

flow_id = openml.flows.flow_exists(name=flow.name, external_version=flow.external_version)
print(flow_id)

####################################################################################################
# We can also retrieve all flows for a given name:
flow_ids = openml.flows.get_flow_id(name=flow.name)
print(flow_ids)

####################################################################################################
# This also works with the actual model (generalizing the first part of this example):
flow_ids = openml.flows.get_flow_id(model=clf, exact_version=False)
print(flow_ids)

# Deactivating test server
openml.config.stop_using_configuration_for_example()
