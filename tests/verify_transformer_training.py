import requests
import json
import time
import numpy as np

BASE_URL = "http://localhost:8081"

def get_weights(model_id="transformer"):
    try:
        # Run inference to get weights
        res = requests.post(f"{BASE_URL}/run/{model_id}", json={"inputs": {"sequence": "0.1,0.2,0.3,0.4"}})
        if res.status_code == 200:
            data = res.json()
            return data.get("weights", []), data.get("activations", [])
        else:
            print(f"Error getting weights: {res.status_code} {res.text}")
            return [], []
    except Exception as e:
        print(f"Exception getting weights: {e}")
        return [], []

def train_step(model_id="transformer"):
    try:
        res = requests.post(f"{BASE_URL}/train/{model_id}", json={"inputs": {"sequence": "0.1,0.2,0.3,0.4"}})
        if res.status_code == 200:
            data = res.json()
            return data.get("loss"), data.get("weights", [])
        else:
            print(f"Error training: {res.status_code} {res.text}")
            return None, []
    except Exception as e:
        print(f"Exception training: {e}")
        return None, []

def main():
    print("Verifying Transformer Training...")
    
    # 1. Get initial weights
    print("Fetching initial weights...")
    initial_weights, _ = get_weights()
    if not initial_weights:
        print("Failed to get initial weights. Is server running?")
        return

    print(f"Initial weights count: {len(initial_weights)}")
    print(f"Sample initial weights: {initial_weights[:5]}")
    
    # 2. Train loop
    print("\nStarting training loop (20 steps)...")
    losses = []
    final_weights = []
    
    for i in range(20):
        loss, weights = train_step()
        if loss is not None:
            losses.append(loss)
            final_weights = weights
            print(f"Step {i+1}: Loss = {loss:.6f}")
        else:
            print(f"Step {i+1}: Failed")
            
    # 3. Analyze results
    if not losses:
        print("No training occurred.")
        return

    print("\nAnalysis:")
    print(f"Initial Loss: {losses[0]}")
    print(f"Final Loss: {losses[-1]}")
    
    loss_improved = losses[-1] < losses[0]
    print(f"Loss Improved: {loss_improved}")
    
    if final_weights:
        weights_changed = final_weights != initial_weights
        
        # Flatten for diff calculation
        def flatten(w):
            flat = []
            if isinstance(w, list):
                for item in w:
                    flat.extend(flatten(item))
            else:
                flat.append(w)
            return flat

        flat_initial = np.array(flatten(initial_weights))
        flat_final = np.array(flatten(final_weights))
        
        diff = np.sum(np.abs(flat_final - flat_initial))
        print(f"Weights Changed: {weights_changed}")
        print(f"Weight Diff L1: {diff}")

        # Analyze structure of weights
        print("\nWeight Structure Analysis:")
        print(f"Top-level list length: {len(final_weights)}")
        for i, layer_w in enumerate(final_weights):
            if isinstance(layer_w, list):
                print(f"  Layer {i}: List of length {len(layer_w)}")
            else:
                print(f"  Layer {i}: Not a list ({type(layer_w)})")

        if weights_changed and loss_improved:
            print("\nSUCCESS: Transformer is training (weights changing and loss improving).")
        elif weights_changed:
            print("\nPARTIAL SUCCESS: Weights are changing, but loss didn't strictly improve (could be stochastic).")
        else:
            print("\nFAILURE: Weights did not change.")
    else:
        print("FAILURE: No weights returned during training.")

if __name__ == "__main__":
    main()
