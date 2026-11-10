"""
Comprehensive experiment runner for the general-K paper.

Runs all 4 simulation models × K values × signal strengths,
with timing and convergence tracking. Writes reproduced/results/all_experiments.json only:
the figures and the scaling table are made by the dedicated scripts, which is
where their current styling and comparator sets live.

    plot_power_curves.py       reproduced/plots/power_curves_K{3,4,5,6}.pdf
    run_scaling_convergence.py reproduced/plots/scaling_with_K.pdf
    make_table_scaling.py      the scaling table in the manuscript

Usage:
    python run_all_experiments.py              # published K=3,4,5,6 grid
    python run_all_experiments.py --quick      # one small installation check
    python run_all_experiments.py --k 3 4 5 6  # custom K range
    python run_all_experiments.py --models trunc mixture  # subset of models
    python run_all_experiments.py --estimate    # print time estimates only
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from general_k.algorithm import default_res_tol
from general_k.alternatives import (
    normal_g,
    truncated_normal_g,
    mixture_normal_g,
    student_t_g,
    beta_g,
)
from general_k.simulations import compare_methods
from general_k import paths
from general_k.io import atomic_json_dump


# ---- Model configurations ----

MODEL_CONFIGS = {
    "trunc": {
        "name": "Truncated Normal",
        "factory": lambda theta: truncated_normal_g(theta),
        "thetas": [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0],
        "theta_label": r"$\theta$",
    },
    "normal": {
        # The classical statistic, to separate the gain due to
        # truncation from the gain due to the policy.  Same thetas as "trunc"
        # so the two arms are read off against each other row by row.  Not in
        # the default --models list: it is an out-of-model arm, since g is
        # unbounded as u -> 0, and adding it silently would change what "the
        # 112-setting grid" means.
        "name": "Normal (untruncated)",
        "factory": lambda theta: normal_g(theta),
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


# ---- Solve ladder ----
#
# Stage 1 is the grid default and settles 109 of the 112 settings.  A setting
# that exits without the residual certificate -- on delta, cycle or the sweep
# cap -- is re-solved at the next stage, from scratch and with the same seed.
# The ladder is therefore deterministic: a full grid re-run reproduces the
# recorded solve exactly, with no by-hand re-runs in the reproduction path.
#
# The stages separate the two things that stop a run short.  Stage 2 raises only
# the sweep cap, for a recursion that is still descending when it runs out of
# budget.  Stage 3 additionally asks bisection to resolve each crossing to
# within the residual tolerance the run is judged against, which is what a
# coordinate with a steep target needs: a fixed bracket width leaves
# |F - alpha| at roughly slope x width, and the recursion then sits at a fixed
# point of the discretised map with its residual still above res_tol.  Keeping
# the two apart is what tells a run that was merely cut off early from one that
# was stuck.
#
# "res_tol" is resolved against the sample size actually used, so the bisection
# target and the certification threshold stay in step if that size changes.
SOLVE_STAGES = (
    {"max_iter": 400, "f_tol": None},
    {"max_iter": 1000, "f_tol": None},
    {"max_iter": 1000, "f_tol": "res_tol"},
)


def _floor_hint(diagnostics):
    """One line on whether the bisection bracket is a plausible culprit.

    Recommendation 4 of the convergence diagnosis: a delta/cycle/max_iter exit
    should not be silent about why.  The width quoted is sufficient to remove
    the bracket as the obstacle, not necessary.  Returns "" when the run
    certified.
    """
    floor = diagnostics.get("floor")
    if not floor:
        return ""
    gamma = floor["binding_gamma"]
    if floor["sufficient_bisect_tol"] is None:
        return (f"; coordinate {gamma} binds through its {floor['binding_term']} "
                f"with zero slope, so no bracket width certifies")
    return (f"; coordinate {gamma} ({floor['binding_term']}) has slope "
            f"{floor['binding_slope']:.3g}, so the bracket admits a gap up to "
            f"{floor['gap_bound']:.2e}; bisect_tol <= "
            f"{floor['sufficient_bisect_tol']:.2e} would rule that out, "
            f"against {diagnostics['bisect_tol']:.1e} used")


def run_single(K, model_key, theta, alpha=0.05, n_algo_samples=100000, n_mc=50000,
               algo_tol=1e-6, bisect_tol=1e-4, stages=None):
    """Run one (K, model, theta) experiment. Returns results dict + timing info.

    Walks SOLVE_STAGES until a stage certifies (see the comment above); the
    returned diagnostics carry the certifying stage's numbers, its index under
    "stage", and a summary of each abandoned stage under "escalation".
    """
    config = MODEL_CONFIGS[model_key]
    g_func = config["factory"](theta)
    stages = SOLVE_STAGES if stages is None else stages
    res_tol = default_res_tol(alpha, n_algo_samples)

    t0 = time.time()
    escalation = []
    for stage, spec in enumerate(stages, 1):
        f_tol = res_tol if spec["f_tol"] == "res_tol" else spec["f_tol"]
        t1 = time.time()
        results, diagnostics = compare_methods(
            K=K,
            g_func=g_func,
            alpha=alpha,
            n_algo_samples=n_algo_samples,
            n_mc=n_mc,
            algo_tol=algo_tol,
            max_iter=spec["max_iter"],
            seed=42,
            bisect_tol=bisect_tol,
            f_tol=f_tol,
        )
        diagnostics["stage"] = stage
        diagnostics["escalation"] = list(escalation)
        if diagnostics["converged"] or stage == len(stages):
            if not diagnostics["converged"]:
                # The last stage failed too. Say so here rather than leaving it
                # to be noticed later in the JSON.
                print(f"  NOT CERTIFIED after {len(stages)} stages "
                      f"({diagnostics['reason']}, "
                      f"{diagnostics['final_residual']:.2e} = "
                      f"{diagnostics['final_residual']/res_tol:.2f}x res_tol)"
                      f"{_floor_hint(diagnostics)}", flush=True)
            break
        escalation.append({
            "stage": stage,
            "max_iter": spec["max_iter"],
            "f_tol": f_tol,
            "converged": diagnostics["converged"],
            "reason": diagnostics["reason"],
            "sweeps": diagnostics["sweeps"],
            "final_residual": diagnostics["final_residual"],
            "elapsed_seconds": time.time() - t1,
        })
        print(f"  stage {stage} did not certify "
              f"({diagnostics['reason']}, {diagnostics['final_residual']:.2e} "
              f"= {diagnostics['final_residual']/res_tol:.2f}x res_tol)"
              f"{_floor_hint(diagnostics)}; "
              f"escalating to stage {stage + 1}", flush=True)
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
        "diagnostics": diagnostics,
    }


def _print_summary(result):
    algo = result["results"]["Algorithm"]
    hommel = result["results"].get("Hommel", {})
    print(
        f"    pi_K: Algo={algo['pi_K']:.3f}, Hommel={hommel.get('pi_K', 0):.3f}"
        f"  |  max_fwer={result['max_fwer_algo']:.3f}"
        f"  |  time={result['elapsed_seconds']:.1f}s"
    )


def run_all(k_values, model_keys, alpha=0.05, jobs=1, thetas_filter=None,
            n_algo_samples=100_000, n_mc=50_000, out_path=None):
    """Run everything. Returns nested dict and saves incrementally.

    jobs > 1 fans the (K, model, theta) settings out over a process pool
    (each setting is fully independent; per-setting seeding is unchanged, so
    parallel and sequential runs produce identical results).
    """
    if out_path is None:
        out_path = paths.reproduced_result("all_experiments.json")

    all_results = {}
    settings = []
    for K in k_values:
        all_results[K] = {}
        for model_key in model_keys:
            all_results[K][model_key] = {}
            for theta in MODEL_CONFIGS[model_key]["thetas"]:
                if thetas_filter is not None and not any(
                    abs(float(theta) - t) < 1e-9 for t in thetas_filter
                ):
                    continue
                settings.append((K, model_key, theta))

    if not settings:
        raise ValueError("the selected K/model/theta filters contain no settings")

    total_t0 = time.time()

    if jobs <= 1:
        for i, (K, model_key, theta) in enumerate(settings, 1):
            config = MODEL_CONFIGS[model_key]
            print(f"\n[{i}/{len(settings)}] K={K}, {config['name']}, "
                  f"{config['theta_label']}={theta}")
            result = run_single(
                K, model_key, theta, alpha=alpha,
                n_algo_samples=n_algo_samples, n_mc=n_mc,
            )
            all_results[K][model_key][theta] = result
            _print_summary(result)
            save_results(all_results, out_path)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    run_single, K, mk, th, alpha, n_algo_samples, n_mc
                ): (K, mk, th)
                for (K, mk, th) in settings
            }
            for i, fut in enumerate(as_completed(futures), 1):
                K, mk, th = futures[fut]
                result = fut.result()
                all_results[K][mk][th] = result
                print(f"\n[{i}/{len(settings)}] done: K={K}, {mk}, theta={th}")
                _print_summary(result)
                save_results(all_results, out_path)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    return all_results


def save_results(all_results, path):
    """Atomically save results to strict JSON (including NumPy values)."""
    atomic_json_dump(all_results, path)
    print(f"  [saved to {path}]")


def estimate_time(k_values, model_keys, thetas_filter=None,
                  n_algo_samples=100_000, n_mc=50_000):
    """Print estimated runtimes without running anything."""
    try:
        with open(paths.result("all_experiments.json"), encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        recorded = {}

    print("\nEstimated runtimes:")
    print("-" * 50)
    total = 0.0
    for K in k_values:
        for model_key in model_keys:
            theta_values = MODEL_CONFIGS[model_key]["thetas"]
            if thetas_filter is not None:
                theta_values = [
                    theta for theta in theta_values
                    if any(abs(float(theta) - value) < 1e-9
                           for value in thetas_filter)
                ]
            n_thetas = len(theta_values)
            recorded_settings = recorded.get(str(K), {}).get(model_key, {})
            recorded_times = [
                entry["elapsed_seconds"]
                for theta, entry in recorded_settings.items()
                if thetas_filter is None or any(
                    abs(float(theta) - value) < 1e-9
                    for value in thetas_filter
                )
            ]
            # Prefer the actual single-core timings bundled with the paper.
            # For a new K/model, retain the old coarse K^2.5 fallback.
            per_point = (sum(recorded_times) / len(recorded_times)
                         if recorded_times else 3.0 * K**2.5)
            # A rough split between optimisation and evaluation work scales a
            # recorded paper run to a nonstandard sample budget.
            sample_scale = (
                0.7 * n_algo_samples / 100_000
                + 0.3 * n_mc / 50_000
            )
            per_point_scaled = per_point * sample_scale
            est = per_point_scaled * n_thetas
            total += est
            print(f"  K={K}, {MODEL_CONFIGS[model_key]['name']:20s}: "
                  f"{n_thetas} points x {per_point_scaled:.0f}s = {est/60:.1f} min")
    print("-" * 50)
    print(f"  TOTAL single-core work: {total/60:.1f} min ({total/3600:.1f} hours)")
    return total


def _selected_setting_count(k_values, model_keys, thetas_filter=None):
    """Count settings selected by the same theta matching rule as ``run_all``."""
    count_per_k = 0
    for model_key in model_keys:
        count_per_k += sum(
            thetas_filter is None or any(
                abs(float(theta) - value) < 1e-9 for value in thetas_filter
            )
            for theta in MODEL_CONFIGS[model_key]["thetas"]
        )
    return len(k_values) * count_per_k


def _certification_status(all_results, allow_uncertified=False):
    """Return nonzero if any final solve failed its residual certificate."""
    failed = []
    for K, models in all_results.items():
        for model, theta_rows in models.items():
            for theta, result in theta_rows.items():
                diagnostics = result["diagnostics"]
                if not diagnostics["converged"]:
                    failed.append((K, model, theta, diagnostics))
    if not failed:
        return 0

    print(
        f"\nWARNING: {len(failed)} final solve(s) did not meet the residual "
        "certificate:",
        file=sys.stderr,
    )
    for K, model, theta, diagnostics in failed:
        print(
            f"  K={K}, model={model}, theta={theta}: "
            f"reason={diagnostics['reason']}, "
            f"residual={diagnostics['final_residual']:.3e}",
            file=sys.stderr,
        )
    if allow_uncertified:
        print(
            "Continuing with exit status 0 because --allow-uncertified was set.",
            file=sys.stderr,
        )
        return 0
    print(
        "Results were saved for diagnosis; pass --allow-uncertified only if "
        "non-certifying output is intentionally acceptable.",
        file=sys.stderr,
    )
    return 2


def _build_parser():
    """Build the command-line parser so its contract is directly testable."""
    parser = argparse.ArgumentParser(
        description="Run the published general-K simulation grid.",
        epilog=("Outputs go to code/reproduced/results by default, leaving the "
                "bundled reference results unchanged."),
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run a small K=3 truncated-normal installation check "
                             "unless its preset values are overridden")
    parser.add_argument("--k", nargs="+", type=int, default=None,
                        help="K values (paper default: 3 4 5 6; quick: 3)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Models (paper default: the four in-model "
                             "families that make up the 112-setting grid; "
                             "quick: trunc; 'normal' is the untruncated arm)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Family-wise error level (paper default: 0.05)")
    parser.add_argument("--estimate", action="store_true",
                        help="Print time estimates and exit")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel worker processes (default: 1, sequential)")
    parser.add_argument("--thetas", nargs="+", type=float, default=None,
                        help="Restrict each selected model to matching theta/df values "
                             "(quick default: -2)")
    parser.add_argument("--n-algo-samples", type=int, default=None,
                        help="Monte Carlo batch per solver configuration "
                             "(paper: 100000; quick: 5000)")
    parser.add_argument("--n-mc", type=int, default=None,
                        help="Monte Carlo draws per power/FWER evaluation "
                             "(paper: 50000; quick: 5000)")
    parser.add_argument("--allow-uncertified", action="store_true",
                        help="Exit zero despite final solves that miss the residual certificate")
    parser.add_argument("--out", default=None,
                        help="Result JSON (default: reproduced/results/"
                             "all_experiments.json, or smoke_grid.json in quick mode)")
    return parser



def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    k_values = args.k if args.k is not None else ([3] if args.quick else [3, 4, 5, 6])
    model_keys = args.models if args.models is not None else (
        ["trunc"] if args.quick else ["trunc", "mixture", "t", "beta"]
    )
    thetas = args.thetas if args.thetas is not None else ([-2.0] if args.quick else None)
    n_algo_samples = (args.n_algo_samples if args.n_algo_samples is not None
                      else (5_000 if args.quick else 100_000))
    n_mc = (args.n_mc if args.n_mc is not None
            else (5_000 if args.quick else 50_000))
    out_path = args.out or paths.reproduced_result(
        "smoke_grid.json" if args.quick else "all_experiments.json"
    )

    if any(K < 2 for K in k_values):
        parser.error("every K must be at least 2")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if n_algo_samples < 1 or n_mc < 1:
        parser.error("sample counts must be positive")
    if _selected_setting_count(k_values, model_keys, thetas) == 0:
        parser.error("the selected K/model/theta filters contain no settings")

    if args.estimate:
        total = estimate_time(k_values, model_keys, thetas, n_algo_samples, n_mc)
        if args.jobs > 1:
            print(f"  idealised --jobs {args.jobs} lower bound: "
                  f"{total / args.jobs / 3600:.1f} hours (actual scheduling "
                  "and memory contention add overhead)")
        return 0

    total = estimate_time(k_values, model_keys, thetas, n_algo_samples, n_mc)
    if args.jobs > 1:
        print(f"  idealised --jobs {args.jobs} lower bound: "
              f"{total / args.jobs / 3600:.1f} hours; allow for overhead")
    print("\nStarting experiments...\n")

    try:
        results = run_all(k_values, model_keys, alpha=args.alpha, jobs=args.jobs,
                          thetas_filter=thetas,
                          n_algo_samples=n_algo_samples, n_mc=n_mc,
                          out_path=out_path)
    except ValueError as exc:
        parser.error(str(exc))

    status = _certification_status(results, args.allow_uncertified)
    if status == 0:
        print("\nDone!")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
