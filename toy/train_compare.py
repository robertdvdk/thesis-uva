# train_comparison.py

import torch
import numpy as np
import random
from typing import List, Dict, Tuple

from hmodel import HToy
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
from hypll.tensors import ManifoldTensor
import hypll.optim

# --- Constants for the Experiment ---
SEEDS = range(10)
N_VALUES = range(46)
LEARNING_RATE = 1.0
MAX_ITERATIONS = 10000
LOSS_THRESHOLD = 0.001
BATCH_SIZE = 64
DIM = 2
OUTPUT_FILENAME = "./results/comparison_results_all_final.txt"

# --- Define the parameter combinations to test ---
# (key, {param_dict})
# We explicitly define the 3 valid combinations, skipping (tangent_space=True, correction=True)
COMBINATIONS: List[Tuple[str, Dict[str, bool]]] = [
    ("ts_false_corr_false", {"impl": "naive", "correction": False}),
    ("ts_false_corr_true", {"impl": "naive", "correction": True}),
    ("ts_true_corr_false", {"impl": "tangent", "correction": False}),
]


# --- Helper Functions ---

def tangent_space_mse_loss(y_pred_man: ManifoldTensor, y_target_man: ManifoldTensor) -> torch.Tensor:
    """Calculates MSE in the tangent space at the origin."""
    tangent_pred = y_pred_man.manifold.logmap(y_pred_man)
    tangent_target = y_target_man.manifold.logmap(y_target_man)
    return torch.nn.functional.mse_loss(tangent_pred.tensor, tangent_target.tensor)


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Core Experiment Logic ---

def run_experiment(
        manifold: Hyperboloid,
        N_val: int,
        seed: int,
        use_correction: bool,
        impl: str,
) -> int:
    """
    Runs a single training instance and returns the number of iterations to converge.

    Returns:
        Number of iterations, or MAX_ITERATIONS if it did not converge.
    """
    set_seed(seed)

    # 1. Initialize model and data, passing the tangent_space parameter
    model = HToy(dim=DIM, manifold=manifold, impl=impl)
    N_tensor = torch.tensor(float(N_val))

    # Define points x (origin) and y on the hyperboloid
    x = torch.tensor([np.sqrt(2)] + [1.0] + [0.0] * (DIM - 1)).float()
    y = torch.tensor([torch.cosh(N_tensor), torch.sinh(N_tensor)] + [0.0] * (DIM - 1)).float()

    x_man = ManifoldTensor(data=x.repeat(BATCH_SIZE, 1), manifold=manifold)
    y_man = ManifoldTensor(data=y.repeat(BATCH_SIZE, 1), manifold=manifold)

    # 2. Setup optimizer
    optimizer = hypll.optim.RiemannianAdam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-15,
        correction=use_correction,
    )
    param_to_name = {param: name for name, param in model.named_parameters()}

    # 3. Training loop
    for i in range(MAX_ITERATIONS):
        optimizer.zero_grad()
        y_pred = model(x_man)
        loss = tangent_space_mse_loss(y_pred, y_man)

        if loss.item() < LOSS_THRESHOLD:
            return i + 1  # Return iteration count on convergence

        loss.backward()
        optimizer.step(param_to_name)

    return MAX_ITERATIONS  # Return max iterations if it failed to converge


# --- Results Processing ---

def save_results_to_txt(results: Dict[int, Dict[str, List[int]]], filename: str):
    """Saves the aggregated results to a plain text file."""
    # Define headers for the new 3-column layout
    header1 = f"{'N Value':<10} | {'TS=False, Corr=False':<25} | {'TS=False, Corr=True':<25} | {'TS=True, Corr=False':<25}"
    header2 = f"{'':-<10} | {'Mean ± Std (Iters)':<25} | {'Mean ± Std (Iters)':<25} | {'Mean ± Std (Iters)':<25}"
    separator = "-" * len(header1)

    with open(filename, 'w') as f:
        # Write the header
        f.write("=" * len(header1) + "\n")
        f.write("📊 Convergence Results Summary\n")
        f.write("=" * len(header1) + "\n")
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        f.write(separator + "\n")

        # Write the data rows
        for n, data in sorted(results.items()):
            # Calculate stats for each combination
            res1 = np.array(data["ts_false_corr_false"])
            str1 = f"{np.mean(res1):7.1f} ± {np.std(res1):5.1f}"

            res2 = np.array(data["ts_false_corr_true"])
            str2 = f"{np.mean(res2):7.1f} ± {np.std(res2):5.1f}"

            res3 = np.array(data["ts_true_corr_false"])
            str3 = f"{np.mean(res3):7.1f} ± {np.std(res3):5.1f}"

            f.write(f"{n:<10} | {str1:<25} | {str2:<25} | {str3:<25}\n")

        f.write(separator + "\n")
    print(f"\n✅ Results successfully saved to '{filename}'")


# --- Main Execution ---

def main():
    """
    Main function to run the full comparison experiment, print a summary, and save to file.
    """
    manifold = Hyperboloid(Curvature(value=np.log(np.exp(1) - 1)))

    # Initialize results dict to hold data for all combinations
    results: Dict[int, Dict[str, List[int]]] = {
        n: {key: [] for key, _ in COMBINATIONS} for n in N_VALUES
    }

    # Track convergence status for each combination.
    # True means it's still converging, False means it has failed and should be skipped.
    convergence_status = {key: True for key, _ in COMBINATIONS}

    print("🚀 Starting Hyperbolic Optimizer Comparison...")
    print(f"Seeds: {len(SEEDS)}, N values: {len(N_VALUES)}, Combinations: {len(COMBINATIONS)}")
    print("-" * 80)

    for n in N_VALUES:
        for key, params in COMBINATIONS:
            # Check if this combination has already failed to converge on a previous N
            if not convergence_status[key]:
                print(f"Skipping... N={n}, Combo={key} (previously failed to converge)")
                # Populate results with MAX_ITERATIONS to indicate failure and skip the run
                results[n][key] = [MAX_ITERATIONS] * len(SEEDS)
                continue

            print(f"Running... N={n}, Tangent Space={params['impl']}, Correction={params['correction']}")

            current_iterations = []
            for seed in SEEDS:
                iterations = run_experiment(
                    manifold=manifold,
                    N_val=n,
                    seed=seed,
                    use_correction=params['correction'],
                    impl=params['impl']
                )
                current_iterations.append(iterations)

            results[n][key] = current_iterations

            # After running all seeds, check for convergence failure for this N
            # If the mean is MAX_ITERATIONS, it means none of the seeds converged.
            if np.mean(current_iterations) >= MAX_ITERATIONS:
                print(
                    f"    -> INFO: Combination '{key}' failed to converge at N={n}. Skipping for subsequent N values.")
                convergence_status[key] = False

    # --- Print Summary Table to Console ---
    header1 = f"{'N Value':<10} | {'TS=F, Corr=F':<25} | {'TS=F, Corr=T':<25} | {'TS=T, Corr=F':<25}"
    header2 = f"{'':-<10} | {'Mean ± Std (Iters)':<25} | {'Mean ± Std (Iters)':<25} | {'Mean ± Std (Iters)':<25}"
    separator = "-" * len(header1)

    print("\n" + "=" * len(header1))
    print("📊 Convergence Results Summary")
    print("=" * len(header1))
    print(header1)
    print(header2)
    print(separator)

    for n, data in sorted(results.items()):
        res1 = np.array(data["ts_false_corr_false"])
        str1 = f"{np.mean(res1):7.1f} ± {np.std(res1):5.1f}"
        res2 = np.array(data["ts_false_corr_true"])
        str2 = f"{np.mean(res2):7.1f} ± {np.std(res2):5.1f}"
        res3 = np.array(data["ts_true_corr_false"])
        str3 = f"{np.mean(res3):7.1f} ± {np.std(res3):5.1f}"
        print(f"{n:<10} | {str1:<25} | {str2:<25} | {str3:<25}")
    print(separator)

    # --- Save Results to Text File ---
    save_results_to_txt(results, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
