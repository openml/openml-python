"""
HuggingFace Hub Integration Tutorial
=====================================

This example demonstrates how to share OpenML flows with HuggingFace Hub,
enabling bidirectional model sharing between the two platforms.

Prerequisites:
- huggingface_hub installed: pip install huggingface_hub
- HuggingFace account with API token
"""

import openml
from openml.extensions.huggingface import (
    download_flow_from_huggingface,
    upload_flow_to_huggingface,
)

# %%
# Setup
# -----
# Configure OpenML (you need an API key from openml.org)
openml.config.apikey = "YOUR_OPENML_API_KEY"

# Your HuggingFace token (get from huggingface.co/settings/tokens)
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"

# %%
# Example 1: Upload an OpenML Flow to HuggingFace
# ------------------------------------------------

# Get a flow from OpenML (this example uses a RandomForest classifier)
flow_id = 8365  # sklearn RandomForestClassifier
flow = openml.flows.get_flow(flow_id, reinstantiate=True)

print(f"Flow Name: {flow.name}")
print(f"Flow ID: {flow.flow_id}")

# Upload to HuggingFace Hub
hf_url = upload_flow_to_huggingface(
    flow=flow,
    repo_id="your-username/openml-randomforest",  # Change to your username
    token=HF_TOKEN,
    private=False,  # Set to True for private repositories
)

print(f"Model uploaded to: {hf_url}")

# %%
# Example 2: Download a Model from HuggingFace
# ---------------------------------------------

result = download_flow_from_huggingface(
    repo_id="your-username/openml-randomforest",
    token=HF_TOKEN,  # Only needed for private repos
)

# Access the model
model = result["model"]
metadata = result["metadata"]

print(f"Downloaded model: {type(model)}")
print(f"Original OpenML Flow ID: {metadata['openml_flow_id']}")
print(f"OpenML URL: {metadata['openml_url']}")

# %%
# Example 3: Share Your Own Model
# --------------------------------
# Train a model, create a flow, publish to OpenML, then share on HuggingFace

from sklearn.ensemble import RandomForestClassifier

# Get a dataset
dataset = openml.datasets.get_dataset(31)  # credit-g dataset
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Train a model
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y)

# Create and publish flow
flow = openml.flows.sklearn_to_flow(clf)
flow.publish()

print(f"Published flow with ID: {flow.flow_id}")

# Share on HuggingFace
hf_url = upload_flow_to_huggingface(
    flow=flow,
    repo_id="your-username/my-credit-model",
    token=HF_TOKEN,
    commit_message="Initial upload of credit scoring model",
)

print(f"Shared on HuggingFace: {hf_url}")

# %%
# Example 4: Using Configuration
# -------------------------------
from openml.extensions.huggingface.config import get_config, set_cache_directory

# Set custom cache directory
set_cache_directory("/path/to/custom/cache")

# Check configuration
config = get_config()
print(f"Cache directory: {config.cache_dir}")
print(f"Model filename: {config.model_filename}")