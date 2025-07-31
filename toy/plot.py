import matplotlib.pyplot as plt
import numpy as np
import re

# --- Configuration ---
INPUT_FILENAME = "./results/comparison_results_all_final.txt"
# Use a vector format like PDF for LaTeX for perfect scaling.
OUTPUT_PLOT_FILENAME = "./results/convergence_plot_all_final.pdf"
MAX_ITERATIONS = 5000  # Used for capping the y-axis


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

    # --- Font and Style Configuration for Publication ---
    # This setup uses Matplotlib's built-in mathtext for a LaTeX-like feel
    # without requiring a local LaTeX installation.
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

    # --- Data Preparation ---
    n_vals = np.array(data["N"])

    plot_configs = {
        'combo1': {
            'mean': np.array(data["combo1_mean"]),
            'std': np.array(data["combo1_std"]),
            'label': r'Hyperbolic Model (baseline)',  # TS=False, Corr=False
            'color': '#0072B2',  # Professional Blue
            'marker': 'o',
            'linestyle': '-',
        },
        'combo2': {
            'mean': np.array(data["combo2_mean"]),
            'std': np.array(data["combo2_std"]),
            'label': r'Hyperbolic Model + LR Correction',  # TS=False, Corr=True
            'color': '#D55E00',  # Professional Orange
            'marker': 's',
            'linestyle': '--',
        },
        'combo3': {
            'mean': np.array(data["combo3_mean"]),
            'std': np.array(data["combo3_std"]),
            'label': r'Tangent Space Model',  # TS=True, Corr=False
            'color': '#009E73',  # Professional Green
            'marker': '^',
            'linestyle': ':',
        }
    }

    # --- Create the Plot ---
    # Figure size is adjusted for a standard single-column LaTeX paper width.
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

    # --- Customize Axes and Labels for LaTeX ---
    # The title is removed, as it's handled by \caption{} in the TeX file.
    ax.set_xlabel(r'Target Hyperbolic Distance from Origin ($d_L(o, y_{\mathrm{true}})$)')
    ax.set_ylabel(r'Iterations to Converge')
    ax.set_yscale('log')

    # --- Ticks and Grid ---
    ax.set_xticks(n_vals)
    # Create labels, but only show a number for every 5th tick
    x_tick_labels = [str(val) if i % 5 == 0 else '' for i, val in enumerate(n_vals)]
    ax.set_xticklabels(x_tick_labels)

    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.25, color='lightgray')
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgray')

    # Set y-axis limits
    ax.set_ylim(bottom=10, top=MAX_ITERATIONS * 2)

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left')

    # --- Final Touches ---
    fig.tight_layout(pad=0.5)

    # --- Save and Show ---
    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved to '{OUTPUT_PLOT_FILENAME}'")
    plt.show()


def main():
    """Main function to run the script."""
    parsed_data = parse_results_file(INPUT_FILENAME)
    if parsed_data:
        create_plot(parsed_data)


if __name__ == "__main__":
    main()
