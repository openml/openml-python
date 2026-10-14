"""
=========================================
Find and reinstantiate a Flow
=========================================

This example shows how to find a flow in OpenML, filtering by specific criteria
(like the library used), and then reinstantiate it locally to use the exact
same model configuration.
"""

# %%
# **Warning:** calculating whether a flow can be reinstantiated can be time consuming.
#For that reason, this example connects to the production server.

import openml
import pandas as pd

# %%
# Switch to the test server
# openml.config.start_using_configuration_for_example()


# %%
# 1. List Flows
# =============
# We list flows available on the server.
# For this example, we use an offset to find newer flows and a size limit to keep it fast.
# To list all flows, remove the `offset` and `size` arguments.
print("Downloading flow list (this may take a while)...")
flows = openml.flows.list_flows(offset=15000, size=1000)
print(f"Total flows found: {len(flows)}")

# %%
# 2. Filter for scikit-learn flows
# ================================
# We filter the DataFrame to find flows that mention "sklearn" in their name.
# These are the ones we will try to reinstantiate.
sklearn_flows = flows[flows["name"].str.contains("sklearn")]
sklearn_flow_ids = list(sklearn_flows.id)
print(f"Scikit-learn flows found: {len(sklearn_flow_ids)}")

# %%
# 3. Check for reinstantiability
# ==============================
# We try to download and reinstantiate each flow.
# ``reinstantiate=True`` will attempt to return the actual python object.
# ``strict_version=False`` allows loading the flow even if local library versions differ
# from the ones used to upload the flow (which is common).
reinstantiable_flows = []

# Ideally you would check all flows, but for this example we limit to the first 10
# to keep execution time short.
print("Checking last 10 flows for reinstantiability...")
for i, flow_id in enumerate(sklearn_flow_ids[-10:]):
    try:
        flow = openml.flows.get_flow(flow_id, reinstantiate=True, strict_version=False)
        # If no exception is raised, we successfully reinstantiated the flow object
        if flow.model is not None:
            reinstantiable_flows.append(flow_id)
            print(f"Flow {flow_id}: Success ({flow.name})")
    except Exception as e:
        print(f"Flow {flow_id}: Failed ({e})")
        pass

print(f"\nFound {len(reinstantiable_flows)} reinstantiable flows in the sample.")

# %%
# 4. Use a reinstantiated flow
# ============================
# Now that we have a flow ID we know works, we can get it and use the model.
if reinstantiable_flows:
    flow_id = reinstantiable_flows[0]
    flow = openml.flows.get_flow(flow_id, reinstantiate=True, strict_version=False)
    
    # flow.model is the actual scikit-learn estimator
    clf = flow.model
    print(f"\nReinstantiated model: {clf}")
    
    # You can now use clf.fit(), clf.predict(), etc.
    # For example, let's just print the parameters
    print(f"Model parameters: {clf.get_params()}")
else:
    print("No reinstantiable flows found in the sample subset.")

# %%
# cleanup
# openml.config.stop_using_configuration_for_example()

