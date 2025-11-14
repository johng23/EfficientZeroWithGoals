# in quick_test.py

import torch
import hydra
from omegaconf import DictConfig

# We need to import the class we want to test
from ez.agents.ez_dmc_state import EZDMCStateAgent


# This is the key. Make this script a Hydra application.
@hydra.main(config_path='./config', config_name='john_test2', version_base='1.1')
def run_test(config):
    """
    This is a simple "scratchpad" for testing a new component.
    It loads the config, builds the model, and runs a few checks.
    """
    print("--- Starting Quick Test ---")

    # 1. Instantiate the Agent with the loaded config
    # This verifies that your agent's __init__ works with the config.
    print("Building agent...")
    agent = EZDMCStateAgent(config)
    print("Agent built successfully.")

    # 2. Build the Model
    # This verifies that your build_model() method works.
    print("Building model...")
    model = agent.build_model()
    # If you have a GPU, move the model to it
    if torch.cuda.is_available():
        model.cuda()
    print("Model built successfully.")

    # 3. Test the Dynamics Network (`do_dynamics`)
    print("\n--- Testing Dynamics Network ---")

    batch_size = 4  # Use a small batch size

    # Create correctly shaped dummy tensors. Use the config to get the shapes.
    # This is much better than hardcoding numbers.
    dummy_state_shape = (batch_size, config.model.hidden_shape)
    dummy_abstract_action_shape = (batch_size, config.model.abstract_action_dim)

    dummy_state = torch.randn(dummy_state_shape)
    dummy_abstract_action = torch.randn(dummy_abstract_action_shape)

    # Move tensors to GPU if available
    if torch.cuda.is_available():
        dummy_state = dummy_state.cuda()
        dummy_abstract_action = dummy_abstract_action.cuda()

    print(f"Input state shape: {dummy_state.shape}")
    print(f"Input abstract action shape: {dummy_abstract_action.shape}")

    # Run the function you want to test
    next_state, reward, gamma, realization_params = model.do_dynamics(dummy_state, dummy_abstract_action)

    # 4. Print and Verify the outputs
    print("\n--- Outputs ---")
    print(f"Output next_state shape: {next_state.shape}")
    print(f"Output reward shape: {reward.shape}")
    print(f"Output gamma shape: {gamma.shape}")
    print(f"Output realization_params shape: {realization_params.shape}")
    print(f"Sample gamma values: {gamma.squeeze().tolist()}")

    # --- Simple, manual assertions ---
    assert dummy_state.shape == next_state.shape, "FAIL: Next state shape is wrong!"
    assert gamma.shape == (batch_size, 1), "FAIL: Gamma shape is wrong!"
    assert all(0.0 <= g <= 1.0 for g in gamma.squeeze().tolist()), "FAIL: Gamma is not between 0 and 1!"

    print("\n--- All checks passed! ---")


# Standard Python entry point
if __name__ == "__main__":
    run_test()