"""
Comprehensive experiment runner for the general-K paper.

Runs all 4 simulation models × K values × signal strengths,
with timing and convergence tracking.

Usage:
    python run_all_experiments.py              # K=3,4,5, all models
    python run_all_experiments.py --k 3 4 5 6  # custom K range
    python run_all_experiments.py --models trunc mixture  # subset of models
    python run_all_experiments.py --estimate    # print time estimates only
"""

import argparse
import json
import os
import time

import numpy as np

from general_k.alternatives import (
    truncated_normal_g,
    mixture_normal_g,
    student_t_g,
    beta_g,
)
from general_k.simulations import compare_methods


# ---- Model configurations ----

MODEL_CONFIGS = {
    "trunc": {
        "name": "Truncated Normal",
        "factory": lambda theta: truncated_normal_g(theta),
        "thetas": [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0],
        "theta_label": r"$\theta$",
    },
    "mixture": {
        "name": "Mixture Normal",
        "factory": lambda theta: mixture_normal_g(theta),
        "thetas": [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0],
        "theta_label": r"$\theta$",
    },
    "t": {
        "name": "Student-t",
        "factory": lambda df: student_t_g(df),
        "thetas": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "theta_label": "df",
    },
    "beta": {
        "name": "Beta",
        "factory": lambda theta: beta_g(theta),
        "thetas": [0.8, 0.6, 0.4, 0.2],
        "theta_label": r"$\theta$",
    },
}


def run_single(K, model_key, theta, alpha=0.05, n_algo_samples=100000, n_mc=50000):
    """Run one (K, model, theta) experiment. Returns results dict + timing info."""
    config = MODEL_CONFIGS[model_key]
    g_func = config["factory"](theta)

    # Scale max_iter with K to ensure convergence
    max_iter = max(100, 30 * K)

    t0 = time.time()
    results = compare_methods(
        K=K,
        g_func=g_func,
        alpha=alpha,
        n_algo_samples=n_algo_samples,
        n_mc=n_mc,
        algo_tol=1e-2,
        max_iter=max_iter,
        seed=42,
    )
    elapsed = time.time() - t0

    # Extract convergence info
    max_fwer = 0.0
    algo_res = results.get("Algorithm", {})
    for key in algo_res:
        if key.startswith("fwer_"):
            max_fwer = max(max_fwer, algo_res[key])

    return {
        "results": {m: {k: v for k, v in r.items()} for m, r in results.items()},
        "elapsed_seconds": elapsed,
        "max_fwer_algo": max_fwer,
    }


def run_model_suite(K, model_key, alpha=0.05):
    """Run all thetas for one (K, model) pair."""
    config = MODEL_CONFIGS[model_key]
    thetas = config["thetas"]
    suite_results = {}

    for theta in thetas:
        print(f"\n  K={K}, {config['name']}, {config['theta_label']}={theta}")
        result = run_single(K, model_key, theta, alpha=alpha)
        suite_results[theta] = result

        # Print summary line
        algo = result["results"]["Algorithm"]
        hommel = result["results"].get("Hommel", {})
        print(
            f"    pi_K: Algo={algo['pi_K']:.3f}, Hommel={hommel.get('pi_K', 0):.3f}"
            f"  |  max_fwer={result['max_fwer_algo']:.3f}"
            f"  |  time={result['elapsed_seconds']:.1f}s"
        )

    return suite_results


def run_all(k_values, model_keys, alpha=0.05):
    """Run everything. Returns nested dict and saves incrementally."""
    all_results = {}
    total_t0 = time.time()

    for K in k_values:
        all_results[K] = {}
        for model_key in model_keys:
            config = MODEL_CONFIGS[model_key]
            print(f"\n{'='*60}")
            print(f"K={K}, Model: {config['name']}")
            print(f"{'='*60}")

            suite = run_model_suite(K, model_key, alpha=alpha)
            all_results[K][model_key] = suite

            # Save incrementally
            save_results(all_results, "../results/all_experiments.json")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    return all_results


def save_results(all_results, path):
    """Save results to JSON (convert numpy types)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"  [saved to {path}]")


def estimate_time(k_values, model_keys):
    """Print estimated runtimes without running anything."""
    # Empirical: K=3 ~68s, K=5 ~356s per (model, theta)
    # Approximate: time ~ K^2.5 * 3 seconds
    print("\nEstimated runtimes:")
    print("-" * 50)
    total = 0.0
    for K in k_values:
        per_point = 3.0 * K**2.5  # rough fit
        for model_key in model_keys:
            n_thetas = len(MODEL_CONFIGS[model_key]["thetas"])
            est = per_point * n_thetas
            total += est
            print(f"  K={K}, {MODEL_CONFIGS[model_key]['name']:20s}: "
                  f"{n_thetas} points x {per_point:.0f}s = {est/60:.1f} min")
    print("-" * 50)
    print(f"  TOTAL: {total/60:.1f} min ({total/3600:.1f} hours)")
    return total


def generate_plots(all_results, k_values, model_keys):
    """Generate power-curve plots for each (K, model) and summary scaling plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    style_map = {
        "Algorithm":  {"color": "black",  "lw": 2.0, "ls": "-",  "marker": "o"},
        "Bonferroni": {"color": "blue",   "lw": 1.5, "ls": "--", "marker": "s"},
        "Holm":       {"color": "green",  "lw": 1.5, "ls": "-.", "marker": "^"},
        "Hochberg":   {"color": "orange", "lw": 1.5, "ls": ":",  "marker": "D"},
        "Hommel":     {"color": "red",    "lw": 1.5, "ls": "--", "marker": "v"},
    }

    # ---- Per-(K, model) power curves ----
    for K in k_values:
        if K not in all_results:
            continue
        for model_key in model_keys:
            if model_key not in all_results[K]:
                continue

            config = MODEL_CONFIGS[model_key]
            suite = all_results[K][model_key]
            thetas = sorted(suite.keys(), key=lambda x: float(x))

            if not thetas:
                continue

            methods = list(suite[thetas[0]]["results"].keys())

            pi_k_series = {m: [] for m in methods}
            pi_any_series = {m: [] for m in methods}
            for theta in thetas:
                for m in methods:
                    pi_k_series[m].append(suite[theta]["results"][m].get("pi_K", float("nan")))
                    pi_any_series[m].append(suite[theta]["results"][m].get("pi_any", float("nan")))

            theta_vals = [float(t) for t in thetas]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"K={K}, {config['name']}", fontsize=13)

            for m in methods:
                style = style_map.get(m, {"color": "gray", "lw": 1.2, "ls": "-", "marker": "."})
                axes[0].plot(theta_vals, pi_k_series[m], label=m, **style, markersize=5)
                axes[1].plot(theta_vals, pi_any_series[m], label=m, **style, markersize=5)

            for ax, ylabel, title in zip(
                axes,
                [rf"$\Pi_{{{K}}}$", r"$\Pi_{\rm any}$"],
                [rf"Average power $\Pi_{{{K}}}$", r"Any-discovery power $\Pi_{\rm any}$"],
            ):
                ax.set_xlabel(config["theta_label"], fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(title, fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout()
            path = f"../plots/general_k_{model_key}_K{K}.pdf"
            plt.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved: {path}")

    # ---- Scaling plot: power advantage vs K ----
    # Use truncated normal at theta=-2.0 as the reference
    if "trunc" in model_keys:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(r"Power advantage scaling with K (Truncated Normal, $\theta = -2.0$)", fontsize=13)

        ks_available = sorted([k for k in k_values if k in all_results and "trunc" in all_results[k]])
        methods_for_scaling = ["Bonferroni", "Holm", "Hommel"]

        for m in methods_for_scaling:
            advantages = []
            ks_plot = []
            for K in ks_available:
                suite = all_results[K].get("trunc", {})
                # Find theta=-2.0 (or closest)
                theta_key = None
                for t in suite:
                    if abs(float(t) - (-2.0)) < 0.01:
                        theta_key = t
                        break
                if theta_key is None:
                    continue
                algo_pi = suite[theta_key]["results"]["Algorithm"]["pi_K"]
                base_pi = suite[theta_key]["results"].get(m, {}).get("pi_K", algo_pi)
                advantages.append(algo_pi - base_pi)
                ks_plot.append(K)

            if ks_plot:
                style = style_map.get(m, {"color": "gray", "lw": 1.2, "ls": "-", "marker": "."})
                axes[0].plot(ks_plot, advantages, label=f"vs {m}", **style, markersize=6)

        axes[0].set_xlabel("K", fontsize=11)
        axes[0].set_ylabel("Power advantage (Algorithm - baseline)", fontsize=11)
        axes[0].set_title("Absolute power gain", fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # ---- Timing scaling ----
        times_by_k = []
        ks_time = []
        for K in ks_available:
            suite = all_results[K].get("trunc", {})
            for t in suite:
                if abs(float(t) - (-2.0)) < 0.01:
                    times_by_k.append(suite[t]["elapsed_seconds"])
                    ks_time.append(K)
                    break

        if ks_time:
            axes[1].plot(ks_time, times_by_k, "ko-", markersize=6, linewidth=2)
            axes[1].set_xlabel("K", fontsize=11)
            axes[1].set_ylabel("Wall-clock time (seconds)", fontsize=11)
            axes[1].set_title("Computation time per experiment", fontsize=11)
            axes[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        path = "../plots/scaling_with_K.pdf"
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {path}")

    # ---- Summary table (LaTeX) ----
    generate_latex_tables(all_results, k_values, model_keys)


def generate_latex_tables(all_results, k_values, model_keys):
    """Generate LaTeX tables for the paper."""
    os.makedirs("results", exist_ok=True)

    # Table 1: Power scaling across K (truncated normal, theta=-2.0)
    if "trunc" in model_keys:
        lines = []
        lines.append(r"\begin{tabular}{llrrrr}")
        lines.append(r"\hline")
        lines.append(r"$K$ & Metric & Bonferroni & Holm & Hommel & Algorithm \\")
        lines.append(r"\hline")

        for K in k_values:
            suite = all_results.get(K, {}).get("trunc", {})
            theta_key = None
            for t in suite:
                if abs(float(t) - (-2.0)) < 0.01:
                    theta_key = t
                    break
            if theta_key is None:
                continue

            res = suite[theta_key]["results"]
            pi_k_vals = {m: res[m]["pi_K"] for m in ["Bonferroni", "Holm", "Hommel", "Algorithm"]}
            fwer_vals = {}
            for m in ["Bonferroni", "Holm", "Hommel", "Algorithm"]:
                max_fwer = max(res[m].get(f"fwer_{g}", 0) for g in range(K))
                fwer_vals[m] = max_fwer

            lines.append(f"{K} & $\\Pi_{{{K}}}$ & {pi_k_vals['Bonferroni']:.3f} & "
                        f"{pi_k_vals['Holm']:.3f} & {pi_k_vals['Hommel']:.3f} & "
                        f"\\textbf{{{pi_k_vals['Algorithm']:.3f}}} \\\\")
            lines.append(f"  & max FWER & {fwer_vals['Bonferroni']:.3f} & "
                        f"{fwer_vals['Holm']:.3f} & {fwer_vals['Hommel']:.3f} & "
                        f"{fwer_vals['Algorithm']:.3f} \\\\")
            lines.append(r"\hline")

        lines.append(r"\end{tabular}")

        with open("../results/table_scaling.tex", "w") as f:
            f.write("\n".join(lines))
        print("  LaTeX table saved: results/table_scaling.tex")


def main():
    parser = argparse.ArgumentParser(description="Run general-K experiments")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 4, 5],
                        help="K values to test (default: 3 4 5)")
    parser.add_argument("--models", nargs="+", default=["trunc", "mixture", "t", "beta"],
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Models to run (default: all)")
    parser.add_argument("--estimate", action="store_true",
                        help="Print time estimates and exit")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Path to existing results JSON; skip experiments, just plot")
    args = parser.parse_args()

    if args.estimate:
        estimate_time(args.k, args.models)
        return

    if args.plot_only:
        with open(args.plot_only) as f:
            all_results = json.load(f)
        # Convert string keys back to int/float
        converted = {}
        for k_str, models in all_results.items():
            K = int(k_str)
            converted[K] = {}
            for model_key, suite in models.items():
                converted[K][model_key] = {float(t): v for t, v in suite.items()}
        generate_plots(converted, args.k, args.models)
        return

    estimate_time(args.k, args.models)
    print("\nStarting experiments...\n")

    all_results = run_all(args.k, args.models)
    generate_plots(all_results, args.k, args.models)

    print("\nDone!")


if __name__ == "__main__":
    main()
