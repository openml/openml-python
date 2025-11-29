"""
Hugging Face Integration Tutorial
=================================

This example demonstrates how to use the experimental Hugging Face integration
to push models to the Hugging Face Hub and link them to OpenML runs.

Requirements:
    pip install openml[huggingface]
    or
    pip install huggingface_hub transformers
"""
import logging
import sys

import openml
from openml.extensions.huggingface_integration import (
    push_model_to_hub_for_run,
    load_model_from_run,
    run_task_with_hf_sync,
    is_hf_transformer
)

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Check if HF dependencies are available
    try:
        import transformers
        from transformers import AutoModel, AutoConfig
    except ImportError:
        print("This example requires 'transformers' and 'huggingface_hub'.")
        print("Please install them with: pip install openml[huggingface]")
        sys.exit(0)

    print("Hugging Face integration is available.")

    # 1. Create a dummy model (or load one)
    # For demonstration, we'll create a tiny random model
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.num_hidden_layers = 1
    config.hidden_size = 32
    config.num_attention_heads = 2
    config.vocab_size = 100
    
    model = AutoModel.from_config(config)
    
    if is_hf_transformer(model):
        print("Model is recognized as a Hugging Face transformer.")

    # 2. Setup a dummy run (in a real scenario, you would run a task)
    # Here we just simulate a run object
    run = openml.runs.OpenMLRun(task_id=1, flow_id=1, dataset_id=1)
    run.run_id = 12345  # Fake run ID
    
    # 3. Push model to Hub
    # NOTE: You need to be logged in to Hugging Face Hub or provide a token.
    # You can login with `huggingface-cli login`
    
    repo_id = "your-username/openml-test-model" # CHANGE THIS
    
    print(f"\nAttempting to push to {repo_id}...")
    print("Note: This will fail if you don't have write access to the repo or aren't logged in.")
    
    try:
        # We pass a token=None to use the locally stored token
        run = push_model_to_hub_for_run(model, run, repo_id=repo_id)
        
        print("\nRun tags after push:")
        print(run.tags)
        
        # 4. Load model back
        print("\nLoading model back from run...")
        loaded_model = load_model_from_run(run.run_id)
        print(f"Loaded model: {type(loaded_model)}")
        
    except Exception as e:
        print(f"\nSkipping actual push/load in this tutorial due to error (likely auth): {e}")
        print("To run the full example, ensure you are logged in to HF Hub and set a valid repo_id.")

    # 5. Convenience wrapper usage
    # run = run_task_with_hf_sync(model, task_id=31, repo_id=repo_id)

if __name__ == "__main__":
    main()
