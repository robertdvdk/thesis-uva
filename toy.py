import torch
import random
from typing import List, Dict, Tuple
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
import hypll.optim
import torch.nn as nn
from hypll.manifolds.base import Manifold
from hypll.tensors import ManifoldTensor
from hypll.nn import HLinear
import matplotlib.pyplot as plt
import numpy as np
import re
import os

SEEDS = range(10)
N_VALUES = range(46)
LEARNING_RATE = 1.0
MAX_ITERATIONS = 10000
LOSS_THRESHOLD = 0.001
BATCH_SIZE = 64
DIM = 2
OUTPUT_FILENAME = "./results/toy_results.txt"
OUTPUT_PLOT_FILENAME = "./results/toy_plot.pdf"

COMBINATIONS: List[Tuple[str, Dict[str, bool]]] = [
    ("ts_false_corr_false", {"impl": "naive", "correction": False}),
    ("ts_false_corr_true", {"impl": "naive", "correction": True}),
    ("ts_true_corr_false", {"impl": "tangent", "correction": False}),
]


class HToy(nn.Module):
    def __init__(self, dim: int, manifold: Manifold, impl: str):
        super().__init__()

        self.manifold = manifold
        if impl == "chen":
            self.model = nn.ModuleList([HLinear(in_features=dim, out_features=dim+1, manifold=manifold, bias=True, impl=impl) for _ in range(1)])
        else:
            self.lin = HLinear(in_features=dim, out_features=dim, manifold=manifold, bias=True, impl=impl)

    def forward(self, x: ManifoldTensor):
        x = self.lin(x)
        return x


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


def parse_results_file(filename: str) -> dict:
    """
    Parses the summary text file with 3 data columns and extracts the data.

    Args:
        filename: The name of the input text file.

    Returns:
        A dictionary containing lists of data for plotting.
    """
    # Initialize data structure for the 3 combinations
    data = {
        "N": [],
        "combo1_mean": [], "combo1_std": [],  # TS=False, Corr=False
        "combo2_mean": [], "combo2_std": [],  # TS=False, Corr=True
        "combo3_mean": [], "combo3_std": [],  # TS=True,  Corr=False
    }

    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() and line.strip()[0].isdigit():
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if len(numbers) == 7:
                        data["N"].append(int(numbers[0]))
                        data["combo1_mean"].append(float(numbers[1]))
                        data["combo1_std"].append(float(numbers[2]))
                        data["combo2_mean"].append(float(numbers[3]))
                        data["combo2_std"].append(float(numbers[4]))
                        data["combo3_mean"].append(float(numbers[5]))
                        data["combo3_std"].append(float(numbers[6]))
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure you have run the training script first to generate the results file.")
        return None

    return data


def create_plot(data: dict):
    """
    Generates and saves a publication-quality line plot suitable for LaTeX.
    """
    if not data or not data["N"]:
        print("No data to plot.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Computer Modern Roman"],  # STIX is a good fallback
        "mathtext.fontset": "stix",  # Use STIX fonts for math
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    n_vals = np.array(data["N"])

    plot_configs = {
        'combo1': {
            'mean': np.array(data["combo1_mean"]),
            'std': np.array(data["combo1_std"]),
            'label': r'Hyperbolic Model (baseline)',  # TS=False, Corr=False
            'color': '#0072B2',
            'marker': 'o',
            'linestyle': '-',
        },
        'combo2': {
            'mean': np.array(data["combo2_mean"]),
            'std': np.array(data["combo2_std"]),
            'label': r'Hyperbolic Model + LR Correction',  # TS=False, Corr=True
            'color': '#D55E00',
            'marker': 's',
            'linestyle': '--',
        },
        'combo3': {
            'mean': np.array(data["combo3_mean"]),
            'std': np.array(data["combo3_std"]),
            'label': r'Tangent Space Model',  # TS=True, Corr=False
            'color': '#009E73',
            'marker': '^',
            'linestyle': ':',
        }
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for key, config in plot_configs.items():
        ax.plot(
            n_vals,
            config['mean'],
            marker=config['marker'],
            linestyle=config['linestyle'],
            label=config['label'],
            color=config['color'],
            linewidth=1.5,
            markersize=5
        )
        ax.fill_between(
            n_vals,
            config['mean'] - config['std'],
            config['mean'] + config['std'],
            alpha=0.15,
            color=config['color'],
            linewidth=0
        )

    ax.set_xlabel(r'Target Hyperbolic Distance from Origin ($d_L(o, y_{\mathrm{true}})$)')
    ax.set_ylabel(r'Iterations to Converge')
    ax.set_yscale('log')

    ax.set_xticks(n_vals)
    # Create labels, but only show a number for every 5th tick
    x_tick_labels = [str(val) if i % 5 == 0 else '' for i, val in enumerate(n_vals)]
    ax.set_xticklabels(x_tick_labels)

    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.25, color='lightgray')
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgray')

    # Set y-axis limits
    ax.set_ylim(bottom=10, top=MAX_ITERATIONS * 2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left')
    fig.tight_layout(pad=0.5)

    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved to '{OUTPUT_PLOT_FILENAME}'")
    plt.show()

def main():
    """
    Main function to run the full comparison experiment, print a summary, and save to file.
    """
    os.makedirs('results', exist_ok=True)
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

    save_results_to_txt(results, OUTPUT_FILENAME)

    parsed_data = parse_results_file(OUTPUT_FILENAME)
    if parsed_data:
        create_plot(parsed_data)
if __name__ == "__main__":
    main()

