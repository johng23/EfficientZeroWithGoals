# in quick_test.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# We need to import the class we want to test
from ez.agents.ez_dmc_state import EZDMCStateAgent
import numpy as np
np.int = int

# This is the key. Make this script a Hydra application.
@hydra.main(config_path='../ez/config', config_name='config', version_base='1.1')
def run_test(config):
    if config.exp_config is not None:
        print(f"Loading experiment config from: {config.exp_config}")
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)
    else:
        print("Warning: No experiment config provided. Using base config only.")
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Create correctly shaped dummy tensors. Use the config to get the shapes.
    # This is much better than hardcoding numbers.

    # Create dummy input tensors with the correct shapes from the config
    dummy_state = torch.randn(batch_size, config.model.hidden_shape).to(device)
    dummy_abstract_action = torch.randn(batch_size, config.model.abstract_action_dim).to(device)

    print(f"Input state shape: {dummy_state.shape}")
    print(f"Input abstract action shape: {dummy_abstract_action.shape}")

    # Run the function to test
    next_state, reward, gamma, realization_params = model.do_dynamics(dummy_state, dummy_abstract_action)

    # Verify the outputs
    print(f"-> Output next_state shape: {next_state.shape}")
    print(f"-> Output reward shape: {reward.shape}")
    print(f"-> Output gamma shape: {gamma.shape}")
    print(f"-> Output realization_params shape: {realization_params.shape}")

    assert dummy_state.shape == next_state.shape, "FAIL: Next state shape is wrong!"
    assert gamma.shape == (batch_size, 1), "FAIL: Gamma shape is wrong!"
    assert all(0.0 <= g <= 1.0 for g in gamma.squeeze().tolist()), "FAIL: Gamma is not between 0 and 1!"
    print("--- DynamicsNetwork Test PASSED! ---")


    # --- 3. TEST: ValuePolicyNetwork (via `model.do_value_policy_prediction`) ---
    print("\n--- [TEST 2/4] Testing ValuePolicyNetwork ---")

    # This network only needs a state as input
    print(f"Input state shape: {dummy_state.shape}")

    # Run the function to test
    value, policy = model.do_value_policy_prediction(dummy_state)

    # Verify the outputs
    print(f"-> Output value shape: {value.shape}")
    print(f"-> Output policy shape: {policy.shape}")

    expected_value_shape = (config.train.v_num, batch_size, config.model.value_support.size)
    expected_policy_shape = (batch_size, config.model.abstract_action_dim * 2)

    assert value.shape == expected_value_shape, f"FAIL: Value shape is wrong! Expected {expected_value_shape}"
    assert policy.shape == expected_policy_shape, f"FAIL: Policy shape is wrong! Expected {expected_policy_shape}"
    print("# Policy head shape confirms it outputs a distribution over ABSTRACT actions.")
    print("--- ValuePolicyNetwork Test PASSED! ---")


    # --- 4. TEST: InjectionNetwork (via `model.injection_model`) ---
    print("\n--- [TEST 3/4] Testing InjectionNetwork ---")

    # This network takes a CONCRETE action as input
    dummy_concrete_action = torch.randn(batch_size, config.env.action_space_size).to(device)
    print(f"Input concrete action shape: {dummy_concrete_action.shape}")

    # Run the function to test
    action_embedding = model.injection_model(dummy_concrete_action)

    # Verify the output
    print(f"-> Output action embedding shape: {action_embedding.shape}")

    expected_embedding_shape = (batch_size, config.model.abstract_action_dim)
    assert action_embedding.shape == expected_embedding_shape, f"FAIL: Injection output shape is wrong! Expected {expected_embedding_shape}"
    print("# Injection network correctly maps concrete action space to abstract action space.")
    print("--- InjectionNetwork Test PASSED! ---")


    # --- 5. TEST: Recurrent Inference (Integration Test) ---
    print("\n--- [TEST 4/4] Testing recurrent_inference (training mode) ---")

    print(f"Input state shape: {dummy_state.shape}")
    print(f"Input abstract action shape: {dummy_abstract_action.shape}")

    # Run the function in training mode to get all the outputs
    outputs = model.recurrent_inference(dummy_state, dummy_abstract_action, reward_hidden=None, training=True)
    next_state, value_prefix, values, policy, gamma_pred, realization_params, _ = outputs

    # Verify the outputs (we just care about shapes here, as values were tested above)
    print(f"-> Output next_state shape: {next_state.shape}")
    print(f"-> Output value_prefix (reward) shape: {value_prefix.shape}")
    print(f"-> Output values shape: {values.shape}")
    print(f"-> Output policy shape: {policy.shape}")
    print(f"-> Output gamma_pred shape: {gamma_pred.shape}")
    print(f"-> Output realization_params shape: {realization_params.shape}")

    assert next_state.shape == dummy_state.shape
    assert values.shape == expected_value_shape
    assert policy.shape == expected_policy_shape
    assert gamma_pred.shape == (batch_size, 1)
    print("# recurrent_inference correctly passes through all dynamics and policy outputs.")
    print("--- recurrent_inference Test PASSED! ---")

# Standard Python entry point
if __name__ == "__main__":
    run_test()
