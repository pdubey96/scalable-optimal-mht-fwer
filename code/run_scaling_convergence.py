"""
Scaling, timing, convergence-trajectory, and sensitivity runs for the paper.

Sequential and single-process by design: the wall-clock times reported in the
paper's convergence table come from this script running alone on one core.

Produces
--------
results/scaling_results.json        K in {3,...,12}: power/FWER for all
                                    methods (N_eval = 50,000 for every K),
                                    optimization + total wall-clock, sweep
                                    counts, mu*, and per-sweep histories.
results/sensitivity_results.json    alpha-sensitivity (K=6) and
                                    N_opt-sensitivity (K=6) runs.
plots/convergence_trajectory.pdf    ||mu^t - mu^{t-1}|| vs sweep (log scale).
plots/scaling_with_K.pdf            power advantage and SPOT optimisation
                                    timing vs K.

Usage
-----
``python run_scaling_convergence.py --quick`` exercises the entire workflow
with reduced Monte Carlo budgets.  ``--paper`` selects the submitted settings.
Both modes write under ``code/reproduced/`` unless output paths are supplied,
so a new run cannot silently replace the bundled reference artifacts.
"""

import argparse
import os
import re
import sys
import tempfile
import time

import numpy as np

from general_k import figstyle
from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g
from general_k.simulations import (evaluate_optimal_policy, evaluate_baseline,
                                   evaluate_closed_test)
from general_k.baselines import bonferroni, holm, hochberg, hommel, romano_wolf
from general_k.closed_testing import CLOSED_TESTS
from general_k import paths
from general_k.io import atomic_json_dump

ALPHA = 0.05
N_OPT = 100_000
N_EVAL = 50_000
TOL = 1e-3
MAX_ITER = 100
K_VALUES = [3, 4, 5, 6, 8, 10, 12]
ALPHA_SENSITIVITY = [0.01, 0.10]
N_OPT_SENSITIVITY = [50_000, 200_000]

QUICK_N_OPT = 2_000
QUICK_N_EVAL = 5_000
QUICK_MAX_ITER = 25
QUICK_N_OPT_SENSITIVITY = [1_000, 4_000]

GIB = 1024 ** 3
AUTO_FULL_CACHE_LIMIT = 2 * GIB
BASELINES = {"Bonferroni": bonferroni, "Holm": holm,
             "Hochberg": hochberg, "Hommel": hommel,
             "Romano-Wolf": romano_wolf}


def _positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(text):
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _probability(text):
    value = float(text)
    if not np.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("must lie strictly between 0 and 1")
    return value


def parse_cache_bytes(text):
    """Parse ``auto``, a byte count, or a decimal/binary memory size."""
    value = str(text).strip().lower()
    if value == "auto":
        return "auto"
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)?", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "use 'auto', a byte count, or a size such as 2GiB or 3500MB"
        )
    number, suffix = match.groups()
    scale = {
        None: 1, "b": 1,
        "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3, "tb": 1000 ** 4,
        "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3,
        "tib": 1024 ** 4,
    }[suffix]
    return int(float(number) * scale)


def optimizer_memory_plan(K, n_opt, cache_spec="auto"):
    """Estimate persistent optimizer arrays and resolve the cache mode."""
    fixed_batches = 16 * n_opt * K ** 2
    one_precompute = 8 * n_opt * K + 8 * n_opt * K ** 2
    coefficient_cache = 8 * n_opt * K ** 3
    streaming = fixed_batches + one_precompute
    full_cache = fixed_batches + 8 * n_opt * K ** 2 + coefficient_cache

    if cache_spec == "auto":
        cache_enabled = full_cache <= AUTO_FULL_CACHE_LIMIT
        resolved = coefficient_cache + 1 if cache_enabled else 0
    else:
        resolved = int(cache_spec)
        cache_enabled = coefficient_cache < resolved
    return {
        "fixed_batches": fixed_batches,
        "streaming_lower_bound": streaming,
        "full_cache_lower_bound": full_cache,
        "coefficient_cache": coefficient_cache,
        "cache_enabled": cache_enabled,
        "resolved_cache_bytes": resolved,
    }


def _format_bytes(n_bytes):
    gib = n_bytes / GIB
    return f"{gib:.2f} GiB" if gib >= 0.01 else f"{n_bytes / 1024 ** 2:.1f} MiB"


def _print_memory_plan(label, K, n_opt, cache_spec):
    plan = optimizer_memory_plan(K, n_opt, cache_spec)
    selected = (plan["full_cache_lower_bound"] if plan["cache_enabled"]
                else plan["streaming_lower_bound"])
    cache_mode = "full coefficient cache" if plan["cache_enabled"] else "streaming"
    print(f"Resource plan ({label}, K={K}, N_opt={n_opt:,}):")
    print(f"  selected {cache_mode}; optimizer persistent-array lower bound "
          f"{_format_bytes(selected)}, before NumPy temporaries/overhead")
    print(f"  full-cache lower bound "
          f"{_format_bytes(plan['full_cache_lower_bound'])}; coefficient-cache "
          f"threshold {_format_bytes(plan['coefficient_cache'])}")
    if not plan["cache_enabled"]:
        print("  streaming uses less memory but can be substantially slower; "
              "enable caching only on a high-memory host with, for example, "
              f"--cache-bytes {plan['coefficient_cache'] + 1}")
    elif plan["full_cache_lower_bound"] > AUTO_FULL_CACHE_LIMIT:
        print("  WARNING: the explicit cache setting exceeds the safe auto cap; "
              "actual peak memory is higher than the reported lower bound. "
              "Use --cache-bytes auto or 0 if this memory is not available.")
    return plan


def _atomic_savefig(figure, output_path, **kwargs):
    """Save a figure beside its destination, then atomically replace it."""
    output_path = os.path.abspath(os.fspath(output_path))
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)
    extension = os.path.splitext(output_path)[1].lstrip(".").lower()
    if not extension:
        raise ValueError("figure output path must have a file extension")
    fd, temporary = tempfile.mkstemp(dir=parent, prefix=".figure.",
                                     suffix=f".{extension}")
    os.close(fd)
    try:
        figure.savefig(temporary, format=extension, **kwargs)
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_setting(K, alpha=ALPHA, n_opt=N_OPT, n_eval=N_EVAL, theta=-2.0,
                tol=TOL, max_iter=MAX_ITER, seed_opt=42, seed_eval=43,
                cache_bytes=None):
    """One full compare run (same seeds as simulations.compare_methods)."""
    g = truncated_normal_g(theta)
    t0 = time.time()
    mu, history, diag = coordinate_descent_fast(
        g, K=K, alpha=alpha, n_samples=n_opt, tol=tol, max_iter=max_iter,
        seed=seed_opt, cache_bytes=cache_bytes, return_diagnostics=True)
    t_opt = time.time() - t0

    results = {"Algorithm": evaluate_optimal_policy(
        mu, g, K, n_mc=n_eval, seed=seed_eval)}
    for name, fn in BASELINES.items():
        results[name] = evaluate_baseline(
            fn, g, K, alpha, n_mc=n_eval, seed=seed_eval)
    t_total = time.time() - t0

    # Closed testing is evaluated after the clock stops on purpose: total_seconds
    # records SPOT plus the evaluation phase and rank-based baselines for the
    # supplementary timing table. Same seed as the baselines above, so all
    # comparators see common random numbers.
    for name, mask_fn in CLOSED_TESTS.items():
        results[name] = evaluate_closed_test(mask_fn, g, K, alpha,
                                             n_mc=n_eval, seed=seed_eval)

    return {
        "mu": mu.tolist(),
        "sweeps": diag["sweeps"],
        "opt_seconds": t_opt,
        "total_seconds": t_total,
        "deltas": diag["deltas"],
        "residuals": diag["residuals"],
        "converged": diag["converged"],
        "reason": diag["reason"],
        "res_tol": diag["res_tol"],
        "results": results,
    }


def _tolerance_mathtext(tol):
    """Format a positive tolerance compactly for a mathtext plot label."""
    exponent = int(np.floor(np.log10(tol)))
    mantissa = tol / (10.0 ** exponent)
    if np.isclose(mantissa, 1.0):
        return f"10^{{{exponent}}}"
    return rf"{mantissa:g}\times10^{{{exponent}}}"


def plot_convergence_figure(scaling, output_path=None, tol=TOL):
    """Plot ||mu^t - mu^{t-1}|| against sweep, from saved or fresh results."""
    if output_path is None:
        output_path = paths.reproduced_plot("convergence_trajectory.pdf")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scaling = {int(K): value for K, value in scaling.items()}

    # Drawn at the width it is printed at, so the type is not shrunk on inclusion.
    figstyle.apply()
    fig, ax = plt.subplots(figsize=figstyle.size(aspect=3.6 / 5.7))
    # Colour-blind-safe palette (Okabe-Ito) with a distinct marker and line style
    # per K, so the curves read without colour (dual encoding).
    conv_style = {
        3:  dict(color=(0.00, 0.45, 0.70), marker="o", ls="-"),
        4:  dict(color=(0.90, 0.60, 0.00), marker="s", ls="--"),
        6:  dict(color=(0.00, 0.60, 0.50), marker="^", ls="-."),
        8:  dict(color=(0.80, 0.40, 0.00), marker="D", ls=":"),
        12: dict(color=(0.80, 0.60, 0.70), marker="v", ls=(0, (5, 1))),
    }
    paper_convergence_K = [3, 4, 6, 8, 12]
    shown = (paper_convergence_K if sorted(scaling) == K_VALUES
             else sorted(scaling))
    if not shown:
        raise ValueError("cannot plot convergence without at least one K setting")
    fallback = [
        dict(color=(0.34, 0.71, 0.91), marker="P", ls="--"),
        dict(color=(0.00, 0.00, 0.00), marker="X", ls="-."),
    ]
    longest = 0
    for index, K in enumerate(shown):
        d = scaling[K]["deltas"]
        longest = max(longest, len(d))
        style_K = conv_style.get(K, fallback[index % len(fallback)])
        ax.semilogy(range(1, len(d) + 1), d, markersize=4,
                    label=f"$K={K}$", **style_K)
    ax.axhline(tol, color="gray", lw=0.8, ls=":")
    # Sweeps are whole numbers, and the tolerance label sits inside the axes so
    # it cannot ride over the y-axis tick labels.
    ax.set_xticks(range(1, longest + 1))
    tolerance_label = _tolerance_mathtext(tol)
    ax.text(1.1, tol * 1.3,
            rf"iterate safeguard $\varepsilon={tolerance_label}$",
            fontsize=figstyle.BASE_PT,
            color="gray")
    # Plain decimals rather than 10^{-k}: a mathtext exponent renders at about
    # 60% of the base size and would be the smallest text in the figure.
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: ("%g" % v) if v >= 1e-3 else ("%.0e" % v)))
    ax.set_xlabel("Outer iteration (sweep) $t$", fontsize=figstyle.BASE_PT)
    ax.set_ylabel(r"$\|\mu^{(t)}-\mu^{(t-1)}\|_2$", fontsize=figstyle.BASE_PT)
    ax.tick_params(labelsize=figstyle.BASE_PT)
    ax.legend(fontsize=figstyle.BASE_PT, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    _atomic_savefig(fig, output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def plot_scaling_figure(scaling, output_path=None, k_values=None):
    """Plot power advantage and timing from saved or freshly computed results."""
    if output_path is None:
        output_path = paths.reproduced_plot("scaling_with_K.pdf")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # JSON object keys are strings; fresh results assembled below use integers.
    scaling = {int(K): value for K, value in scaling.items()}
    if k_values is None:
        k_values = sorted(scaling)
    else:
        k_values = [int(K) for K in k_values if int(K) in scaling]
    if not k_values:
        raise ValueError("cannot plot scaling without at least one K setting")

    # The compact typography and spacing follow the faceted reference figures.
    # The nominal size below prints just over the journal's 10pt minimum after
    # accounting for the saved PDF's bounding box and manuscript inclusion.
    figstyle.apply()
    # Two panels need not span the full text block. Drawing and printing at 80%
    # width gives each panel proportions close to the three-panel reference.
    fig, axes = plt.subplots(
        1, 2, figsize=figstyle.size(aspect=0.55, fraction=0.80))
    PLOT_PT = 10.0
    TITLE = LABEL = TICK = LEGEND = PLOT_PT
    # Dash and marker encoding distinguishes both methods and the seven
    # measured K values in print.
    style = {"Bonferroni": dict(color=(0.00, 0.45, 0.70), ls="--", marker="o"),
             "Holm": dict(color=(0.00, 0.60, 0.50), ls=(0, (5, 1.5, 1, 1.5)),
                          marker="s"),
             "Hommel": dict(color=(0.80, 0.40, 0.00), ls=(0, (6, 2)), marker="^"),
             "Fisher-closed": dict(color=(0.60, 0.00, 0.45),
                                   ls=(0, (3, 1, 1, 1)), marker="D")}
    def facet_strip(ax, label):
        ax.add_patch(Rectangle(
            (0, 1.01), 1, 0.14, transform=ax.transAxes, clip_on=False,
            facecolor="0.94", edgecolor="0.35", linewidth=0.65))
        ax.text(0.5, 1.08, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=TITLE)

    adv_max = 0.0
    for m in ["Bonferroni", "Holm", "Hommel", "Fisher-closed"]:
        adv = [scaling[K]["results"]["Algorithm"]["pi_K"]
               - scaling[K]["results"][m]["pi_K"] for K in k_values]
        adv_max = max(adv_max, max(adv))
        display_name = "Closed Fisher" if m == "Fisher-closed" else m
        axes[0].plot(k_values, adv, label=display_name, lw=1.15,
                     markersize=3.1, markerfacecolor="white",
                     markeredgewidth=0.65, solid_capstyle="butt", **style[m])
    # All gains are positive here, so a zero baseline gives their magnitudes an
    # honest common reference.  The shared legend sits above the panels rather
    # than obscuring the curves.
    axes[0].set_ylim(0, max(adv_max * 1.08, 1e-12))
    axes[0].set_ylabel("SPOT power gain", fontsize=LABEL, labelpad=3)
    axes[0].set_xticks(k_values)
    axes[0].tick_params(labelsize=TICK, length=2.5, width=0.65, pad=1.5)
    axes[0].grid(True, linestyle=(0, (2, 3)), color="0.89", linewidth=0.55)
    facet_strip(axes[0], "(a) Power advantage")

    k_values = np.asarray(k_values, dtype=float)
    times = np.asarray([scaling[int(K)]["opt_seconds"] for K in k_values])
    axes[1].loglog(k_values, times, color="black", linewidth=1.35,
                   marker="o", markersize=3.4, solid_capstyle="butt")
    if len(k_values) >= 2 and np.all(times > 0):
        slope, intercept = np.polyfit(np.log(k_values), np.log(times), 1)
        fitted = np.exp(intercept) * k_values ** slope
        axes[1].loglog(k_values, fitted, color="0.45", linewidth=1.0,
                       ls=(0, (4, 2)), solid_capstyle="butt")
    if list(map(int, k_values)) == K_VALUES:
        axes[1].set_ylim(2.7, 52)
    axes[1].set_ylabel("SPOT time (s)", fontsize=LABEL, labelpad=3)
    axes[1].set_xticks(k_values)
    axes[1].set_xticklabels([str(int(K)) for K in k_values])
    if list(map(int, k_values)) == K_VALUES:
        axes[1].set_yticks([3, 10, 30])
        axes[1].set_yticklabels(["3", "10", "30"])
    axes[1].tick_params(labelsize=TICK, length=2.5, width=0.65, pad=1.5)
    axes[1].grid(True, linestyle=(0, (2, 3)), color="0.89", linewidth=0.55)
    facet_strip(axes[1], "(b) Optimisation time")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("0.25")
            spine.set_linewidth(0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=LEGEND, frameon=False, columnspacing=1.1,
               handlelength=2.0, handletextpad=0.4, borderaxespad=0,
               bbox_to_anchor=(0.5, 0.985))
    fig.supxlabel("$K$", fontsize=LABEL, y=0.025)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.79,
                        wspace=0.25)
    _atomic_savefig(fig, output_path, bbox_inches="tight", pad_inches=0.03,
                    facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def _solver_status(result):
    """Return a compact, explicit certificate status for console reports."""
    if result.get("converged", False):
        return "status=certified"
    residuals = result.get("residuals", [])
    residual = residuals[-1] if residuals else float("nan")
    res_tol = result.get("res_tol", float("nan"))
    return (f"status=UNCERTIFIED, reason={result.get('reason', 'unknown')}, "
            f"residual={residual:.2e}, certificate_tol={res_tol:.2e}")


def _uses_reference_schema(config):
    """Whether this is the exact submitted run checked by the strict verifier."""
    return (
        config.mode == "paper"
        and config.k_values == K_VALUES
        and config.alpha == ALPHA
        and config.theta == -2.0
        and config.n_opt == N_OPT
        and config.n_eval == N_EVAL
        and config.tol == TOL
        and config.max_iter == MAX_ITER
        and config.seed_opt == 42
        and config.seed_eval == 43
        and config.cache_bytes == "auto"
        and not config.skip_sensitivity
        and config.sensitivity_k == 6
        and config.alpha_sensitivity == ALPHA_SENSITIVITY
        and config.n_opt_sensitivity == N_OPT_SENSITIVITY
    )


def _run_config_payload(config, *, arm, K, alpha, n_opt,
                        resolved_cache_bytes):
    """Serializable provenance embedded in non-reference result records."""
    return {
        "mode": config.mode,
        "arm": arm,
        "K": K,
        "alpha": alpha,
        "theta": config.theta,
        "n_opt": n_opt,
        "n_eval": config.n_eval,
        "tol": config.tol,
        "max_iter": config.max_iter,
        "seed_opt": config.seed_opt,
        "seed_eval": config.seed_eval,
        "cache_bytes_requested": config.cache_bytes,
        "cache_bytes_resolved": resolved_cache_bytes,
        "k_values": config.k_values,
        "skip_sensitivity": config.skip_sensitivity,
        "sensitivity_k": config.sensitivity_k,
        "alpha_sensitivity": config.alpha_sensitivity,
        "n_opt_sensitivity": config.n_opt_sensitivity,
        "no_plots": config.no_plots,
        "allow_uncertified": config.allow_uncertified,
    }


def main(config):
    """Run scaling, optional sensitivity arms, and optional figures."""
    scaling = {}
    solver_runs = []
    preserve_reference_schema = _uses_reference_schema(config)

    def setting(K, *, alpha, n_opt, arm):
        cache_bytes = optimizer_memory_plan(
            K, n_opt, config.cache_bytes
        )["resolved_cache_bytes"]
        result = run_setting(
            K, alpha=alpha, n_opt=n_opt, n_eval=config.n_eval,
            theta=config.theta, tol=config.tol, max_iter=config.max_iter,
            seed_opt=config.seed_opt, seed_eval=config.seed_eval,
            cache_bytes=cache_bytes,
        )
        if not preserve_reference_schema:
            result["_run_config"] = _run_config_payload(
                config, arm=arm, K=K, alpha=alpha, n_opt=n_opt,
                resolved_cache_bytes=cache_bytes,
            )
        solver_runs.append(result)
        return result

    for K in config.k_values:
        print(f"\n=== K={K} (trunc, theta={config.theta}, "
              f"alpha={config.alpha}) ===", flush=True)
        out = setting(K, alpha=config.alpha, n_opt=config.n_opt,
                      arm="scaling")
        scaling[K] = out
        algo = out["results"]["Algorithm"]
        maxf = max(v for k, v in algo.items() if k.startswith("fwer_"))
        print(f"  sweeps={out['sweeps']} ({out['reason']}, residual="
              f"{out['residuals'][-1]:.2e})  opt={out['opt_seconds']:.1f}s  "
              f"total={out['total_seconds']:.1f}s  pi_K={algo['pi_K']:.4f}  "
              f"max_fwer={maxf:.4f}  {_solver_status(out)}", flush=True)
        atomic_json_dump(scaling, config.scaling_out)

    if not config.skip_sensitivity:
        # ---- sensitivity: alpha ----
        sens = {"alpha": {}, "n_opt": {}}
        for alpha in config.alpha_sensitivity:
            print(f"\n=== sensitivity: K={config.sensitivity_k}, "
                  f"alpha={alpha} ===", flush=True)
            sens["alpha"][alpha] = setting(
                config.sensitivity_k, alpha=alpha, n_opt=config.n_opt,
                arm="alpha_sensitivity")
            print(f"  {_solver_status(sens['alpha'][alpha])}", flush=True)
            atomic_json_dump(sens, config.sensitivity_out)

        # ---- sensitivity: N_opt ----
        for n_opt in config.n_opt_sensitivity:
            print(f"\n=== sensitivity: K={config.sensitivity_k}, "
                  f"N_opt={n_opt} ===", flush=True)
            sens["n_opt"][n_opt] = setting(
                config.sensitivity_k, alpha=config.alpha, n_opt=n_opt,
                arm="n_opt_sensitivity")
            print(f"  {_solver_status(sens['n_opt'][n_opt])}", flush=True)
            atomic_json_dump(sens, config.sensitivity_out)

    if not config.no_plots:
        plot_convergence_figure(
            scaling, config.convergence_plot, tol=config.tol)
        plot_scaling_figure(scaling, config.scaling_plot, config.k_values)

    print("\nDone.")
    return solver_runs


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Scaling and convergence runs, plus the alpha and N_opt sensitivity arms.",
        epilog="Examples: --quick; --paper; --quick --k-values 3 6 --no-plots. "
               "Paper mode is sequential by design and may take hours; use "
               "--show-plan to inspect settings without computing.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick", dest="mode", action="store_const", const="quick",
        help=f"run the full workflow with N_opt={QUICK_N_OPT:,} and "
             f"N_eval={QUICK_N_EVAL:,}",
    )
    mode.add_argument(
        "--paper", "--full", dest="mode", action="store_const", const="paper",
        help=f"use submitted budgets N_opt={N_OPT:,}, N_eval={N_EVAL:,}; "
             "--full is an alias",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=_positive_int, metavar="K",
        help="scaling K values (paper default: 3 4 5 6 8 10 12)",
    )
    parser.add_argument(
        "--alpha", type=_probability, default=ALPHA,
        help=f"main-run FWER level (paper default: {ALPHA:g})",
    )
    parser.add_argument(
        "--theta", type=float, default=-2.0,
        help="truncated-normal alternative parameter (paper default: -2.0)",
    )
    parser.add_argument(
        "--n-opt", type=_positive_int,
        help="optimizer Monte Carlo samples per configuration (overrides mode default)",
    )
    parser.add_argument(
        "--n-eval", type=_positive_int,
        help="evaluation Monte Carlo replications (overrides mode default)",
    )
    parser.add_argument(
        "--tol", type=_positive_float,
        help=f"outer-iterate stopping safeguard (paper default: {TOL:g})",
    )
    parser.add_argument(
        "--max-iter", type=_positive_int,
        help=f"maximum coordinate-descent sweeps (paper default: {MAX_ITER})",
    )
    parser.add_argument(
        "--seed-opt", type=int, default=42,
        help="optimizer random seed (paper default: 42)",
    )
    parser.add_argument(
        "--seed-eval", type=int, default=43,
        help="evaluation random seed (paper default: 43)",
    )
    parser.add_argument(
        "--cache-bytes", type=parse_cache_bytes, default="auto", metavar="SIZE",
        help="coefficient-cache budget: auto, bytes, or a size such as 2GiB; "
             "0 always streams (default: auto, with a 2 GiB persistent-array cap)",
    )
    parser.add_argument(
        "--allow-uncertified", action="store_true",
        help="exit successfully even if one or more optimizer runs lack a "
             "convergence certificate",
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="skip both alpha and N_opt sensitivity arms",
    )
    parser.add_argument(
        "--sensitivity-k", type=_positive_int, default=6,
        help="K used by sensitivity arms (paper default: 6)",
    )
    parser.add_argument(
        "--alpha-sensitivity", nargs="+", type=_probability, metavar="ALPHA",
        help="alpha sensitivity values (paper/quick default: 0.01 0.10)",
    )
    parser.add_argument(
        "--n-opt-sensitivity", nargs="+", type=_positive_int, metavar="N",
        help="N_opt sensitivity values (paper: 50000 200000; quick: 1000 4000)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="write JSON only; do not draw the two figures",
    )
    parser.add_argument(
        "--scaling-out",
        help="scaling JSON path (default: mode-specific reproduced/results path)",
    )
    parser.add_argument(
        "--sensitivity-out",
        help="sensitivity JSON path (default: mode-specific reproduced/results path)",
    )
    parser.add_argument(
        "--convergence-plot",
        help="convergence PDF path (default: mode-specific reproduced/plots path)",
    )
    parser.add_argument(
        "--scaling-plot",
        help="scaling PDF path (default: mode-specific reproduced/plots path)",
    )
    parser.add_argument(
        "--show-plan", action="store_true",
        help="print resolved settings and memory estimates, then exit",
    )
    return parser.parse_args(argv)


def _absolute_user_path(value):
    return os.path.abspath(os.path.expanduser(value))


def _resolve_args(config):
    if config.k_values is None:
        config.k_values = list(K_VALUES)
    else:
        # Duplicate settings add cost without adding information.
        config.k_values = list(dict.fromkeys(config.k_values))
    if not np.isfinite(config.theta):
        raise ValueError("--theta must be finite")
    if config.n_opt is None:
        config.n_opt = N_OPT if config.mode == "paper" else QUICK_N_OPT
    if config.n_eval is None:
        config.n_eval = N_EVAL if config.mode == "paper" else QUICK_N_EVAL
    if config.tol is None:
        config.tol = TOL
    if config.max_iter is None:
        config.max_iter = MAX_ITER if config.mode == "paper" else QUICK_MAX_ITER
    if config.alpha_sensitivity is None:
        config.alpha_sensitivity = list(ALPHA_SENSITIVITY)
    if config.n_opt_sensitivity is None:
        values = (N_OPT_SENSITIVITY if config.mode == "paper"
                  else QUICK_N_OPT_SENSITIVITY)
        config.n_opt_sensitivity = list(values)

    suffix = "" if config.mode == "paper" else "_quick"

    # Resolve defaults only when their output is active and no explicit path
    # was supplied.  This lets a fully explicit run work even if the source
    # checkout itself is read-only.
    if config.scaling_out is None:
        config.scaling_out = paths.reproduced_result(
            f"scaling_results{suffix}.json")
    else:
        config.scaling_out = _absolute_user_path(config.scaling_out)

    if config.sensitivity_out is not None:
        config.sensitivity_out = _absolute_user_path(config.sensitivity_out)
    elif not config.skip_sensitivity:
        config.sensitivity_out = paths.reproduced_result(
            f"sensitivity_results{suffix}.json")

    if config.convergence_plot is not None:
        config.convergence_plot = _absolute_user_path(config.convergence_plot)
    elif not config.no_plots:
        config.convergence_plot = paths.reproduced_plot(
            f"convergence_trajectory{suffix}.pdf")

    if config.scaling_plot is not None:
        config.scaling_plot = _absolute_user_path(config.scaling_plot)
    elif not config.no_plots:
        config.scaling_plot = paths.reproduced_plot(
            f"scaling_with_K{suffix}.pdf")

    _validate_output_paths(config)
    return config


def _active_output_paths(config):
    outputs = [("--scaling-out", config.scaling_out, ".json")]
    if not config.skip_sensitivity:
        outputs.append(("--sensitivity-out", config.sensitivity_out, ".json"))
    if not config.no_plots:
        outputs.extend([
            ("--convergence-plot", config.convergence_plot, ".pdf"),
            ("--scaling-plot", config.scaling_plot, ".pdf"),
        ])
    return outputs


def _validate_output_paths(config):
    """Reject ambiguous destinations before any expensive computation."""
    seen = {}
    for flag, output, suffix in _active_output_paths(config):
        if os.path.splitext(output)[1].lower() != suffix:
            raise ValueError(f"{flag} must have a {suffix} suffix")
        identity = os.path.normcase(os.path.realpath(output))
        if identity in seen:
            raise ValueError(
                f"{flag} and {seen[identity]} resolve to the same output path"
            )
        seen[identity] = flag


def _largest_resource_setting(config):
    candidates = [(K, config.n_opt, "scaling") for K in config.k_values]
    if not config.skip_sensitivity:
        candidates.extend(
            (config.sensitivity_k, config.n_opt, "alpha sensitivity")
            for _ in config.alpha_sensitivity
        )
        candidates.extend(
            (config.sensitivity_k, n_opt, "N_opt sensitivity")
            for n_opt in config.n_opt_sensitivity
        )
    def selected_lower_bound(item):
        plan = optimizer_memory_plan(item[0], item[1], config.cache_bytes)
        return (plan["full_cache_lower_bound"] if plan["cache_enabled"]
                else plan["streaming_lower_bound"])

    return max(candidates, key=selected_lower_bound)


def cli(argv=None):
    try:
        config = _resolve_args(_parse_args(argv))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Mode: {config.mode}; K={','.join(map(str, config.k_values))}; "
          f"N_opt={config.n_opt:,}; N_eval={config.n_eval:,}; "
          f"tol={config.tol:g}; max_iter={config.max_iter}; "
          f"seeds={config.seed_opt}/{config.seed_eval}")
    K, n_opt, label = _largest_resource_setting(config)
    _print_memory_plan(label, K, n_opt, config.cache_bytes)
    for flag, output, _suffix in _active_output_paths(config):
        print(f"{flag}: {output}")
        if os.path.exists(output):
            print("  WARNING: output already exists and will be replaced atomically.")
    if config.show_plan:
        return 0
    solver_runs = main(config)
    uncertified = [run for run in (solver_runs or [])
                   if not run.get("converged", False)]
    if uncertified:
        message = (f"uncertified optimizer runs: {len(uncertified)} of "
                   f"{len(solver_runs)}")
        if config.allow_uncertified:
            print(f"WARNING: {message}; continuing because "
                  "--allow-uncertified was supplied.", file=sys.stderr)
        else:
            print(f"ERROR: {message}. Inspect the reported residuals or rerun "
                  "with a larger budget. Use --allow-uncertified only to "
                  "accept these results explicitly.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
