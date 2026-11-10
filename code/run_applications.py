"""
Real-data applications for the paper: the BCG subgroup analysis at K = 3, and
Camerer/SSRP, SPRINT and RPP/OSC.

The BCG trial counts are not transcribed here: they are carried in
general_k/data_bcg.py with their source and a checksum, so that application
needs no download.  Provenance for the other three follows.

Data provenance (all values transcribed from primary sources):

1. Camerer et al. (2018), Nature Human Behaviour 2:637-644 (SSRP).
   Stage-1 replication two-sided p-values for all 21 studies, from
   Supplementary Table 3 ("Replication results for Stage 1") of the paper's
   Supplementary Information.  Values reported there as "< 0.001" are set to
   0.001, an upper-bound substitution; because optimal-multiple-testing
   rejection regions need not be monotone in a single p-value, this is not
   guaranteed to weaken the evidence, so main() also runs a sensitivity check
   that lowers these six values to 1e-10 and confirms the rejection sets are
   unchanged (see the "camerer_sensitivity_1e-10" entry in applications.json).
   The 21 studies form the prespecified analysis family: every study
   replicated in Stage 1 of the SSRP.

2. SPRINT (NEJM 2015; DOI 10.1056/NEJMoa1511939), Table 2: two-sided
   p-values for the five individual cardiovascular outcomes (excluding the
   primary composite): MI 0.19, non-MI ACS 0.99, stroke 0.50, heart failure
   0.002, CVD death 0.005.

3. Reproducibility Project: Psychology (OSC 2015), rpp_data.csv from
   https://osf.io/fgjvw/: six JPSP social-psychology studies with similar
   original effect sizes (r in [0.21, 0.27]); replication p-values
   (T_pval_USE (R)) and replication sample sizes (N (R)) as recorded there.

Two-sided p-values are modeled with the two-sided mixture-normal alternative
g(u) = exp(-theta^2/2) cosh{theta * Phi^{-1}(1 - u/2)} (the paper's mixture
model), the correct likelihood-ratio form for a two-sided z-test against a
common standardized effect |theta|; beta(theta, 1) is used as a
model-robustness check.

Usage
-----
``python run_applications.py --quick`` runs every application with a reduced
Monte Carlo optimizer budget.  ``--paper`` selects the submitted numerical
settings.  Both modes write under ``code/reproduced/`` unless ``--out`` is
given, so a new run cannot silently replace the bundled reference result.
"""

import argparse
import os
import re
import sys
import time

import numpy as np

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import mixture_normal_g, beta_g
from general_k.decision import optimal_l_star
from general_k.baselines import bonferroni, holm, hochberg, hommel, romano_wolf
from general_k.closed_testing import CLOSED_TESTS
from general_k import paths
from general_k import data_bcg
from general_k.io import atomic_json_dump

ALPHA = 0.05
N_OPT = 100_000
TOL = 1e-3
MAX_ITER = 200
QUICK_N_OPT = 2_000
QUICK_MAX_ITER = 25

# ``coordinate_descent_fast`` either retains every coefficient tensor or
# streams one at a time.  In auto mode, cap the *persistent-array lower bound*
# for the retained form at 2 GiB.  This makes the small/quick cases fast while
# keeping the K=21 paper run away from the old unconditional 9 GB cache request.
GIB = 1024 ** 3
AUTO_FULL_CACHE_LIMIT = 2 * GIB
BASELINES = {"Bonferroni": bonferroni, "Holm": holm,
             "Hochberg": hochberg, "Hommel": hommel,
             "Romano-Wolf": romano_wolf}

# --- Camerer et al. (2018) SSRP: Stage-1 p-values, Supplementary Table 3 ---
# "< 0.001" entries set to 0.001 (an upper-bound substitution; NOT guaranteed
# conservative, since optimal rejection regions need not be monotone in a single
# p-value -- see the module header and the 1e-10 sensitivity check in main()).
CAMERER = [
    ("Ackerman et al. (2010)",        0.024),
    ("Aviezer et al. (2012)",         0.001),   # < 0.001
    ("Balafoutas and Sutter (2012)",  0.022),
    ("Derex et al. (2013)",           0.003),
    ("Duncan et al. (2012)",          0.279),
    ("Gervais and Norenzayan (2012)", 0.416),
    ("Gneezy et al. (2014)",          0.001),   # < 0.001
    ("Hauser et al. (2014)",          0.001),   # < 0.001
    ("Janssen et al. (2010)",         0.025),
    ("Karpicke and Blunt (2011)",     0.006),
    ("Kidd and Castano (2013)",       0.273),
    ("Kovacs et al. (2010)",          0.001),   # < 0.001
    ("Lee and Schwarz (2010)",        0.455),
    ("Morewedge et al. (2010)",       0.001),   # < 0.001
    ("Nishi et al. (2015)",           0.011),
    ("Pyc and Rawson (2010)",         0.089),
    ("Ramirez and Beilock (2011)",    0.716),
    ("Rand et al. (2012)",            0.366),
    ("Shah et al. (2012)",            0.150),
    ("Sparrow et al. (2011)",         0.265),
    ("Wilson et al. (2014)",          0.001),   # < 0.001
]

# --- SPRINT (NEJM 2015): five individual CV endpoints, two-sided p ---
SPRINT = [
    ("Myocardial infarction",   0.19),
    ("Non-MI acute coronary",   0.99),
    ("Stroke",                  0.50),
    ("Heart failure",           0.002),
    ("Cardiovascular death",    0.005),
]

# --- RPP / OSC (2015): six JPSP studies, replication p and N from rpp_data ---
OSC = [
    ("Albarracin et al. (2008)",      0.0791, 105),
    ("Fischer et al. (2008)",         0.1406, 140),
    ("Shnabel and Nadler (2008)",     0.2338, 141),
    ("Centerbar et al. (2008)",       0.3219, 113),
    ("Correll (2008)",                0.3744, 148),
    ("Cox et al. (2008)",             0.4690, 200),
]


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


def parse_cache_bytes(text):
    """Parse ``auto``, a byte count, or a decimal/binary memory size."""
    value = str(text).strip().lower()
    if value == "auto":
        return "auto"
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)?", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "use 'auto', a byte count, or a size such as 2GiB or 9000MB"
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
    """Return conservative-choice and lower-bound array sizes for one solve.

    The estimates count persistent arrays visible from the implementation:
    fixed sorted samples and indices, plus either one or all ``a``/``b``
    coefficient tensors.  NumPy temporaries and Python overhead are excluded,
    so the reported figures are deliberately labelled lower bounds.
    """
    fixed_batches = 16 * n_opt * K ** 2
    one_precompute = 8 * n_opt * K + 8 * n_opt * K ** 2
    coefficient_cache = 8 * n_opt * K ** 3
    streaming = fixed_batches + one_precompute
    full_cache = fixed_batches + 8 * n_opt * K ** 2 + coefficient_cache

    if cache_spec == "auto":
        cache_enabled = full_cache <= AUTO_FULL_CACHE_LIMIT
        # The solver's cache comparison is strict, so add one byte to enable it.
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
    print(f"  selected {cache_mode}; persistent-array lower bound "
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


def algorithm_decision(g, K, p_values, alpha=ALPHA, n_opt=N_OPT, seed=42,
                       cache_bytes=None, tol=TOL, max_iter=MAX_ITER):
    """Compute mu* for (g, K, alpha) and apply the policy to the observed
    p-values.  Returns (l_star, rejected_indices_sorted_by_p, sweeps, secs)."""
    t0 = time.time()
    mu, history, diag = coordinate_descent_fast(
        g, K=K, alpha=alpha, n_samples=n_opt, tol=tol, max_iter=max_iter,
        seed=seed, cache_bytes=cache_bytes, return_diagnostics=True)
    secs = time.time() - t0
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    u_sorted = p[order]
    l_star = optimal_l_star(mu, u_sorted, g, K)
    rejected = sorted(order[:l_star].tolist())
    return {"l_star": int(l_star), "rejected": rejected,
            "sweeps": diag["sweeps"], "seconds": secs,
            "converged": diag["converged"], "reason": diag["reason"],
            "final_residual": diag["final_residual"],
            "res_tol": diag["res_tol"],
            # Full precision retained so the 1e-10 sensitivity check reuses the
            # exact multipliers, not a 4-decimal rounding.
            "mu": mu.tolist()}


A_CLAMP = (0.10, 0.90)


def fit_beta_shape(p_values, clamp=A_CLAMP):
    """MLE of the shape a in Beta(a, 1), clamped as the submitted analysis clamps it.

    The MLE is -n / sum(log p).  The clamp keeps the fitted density inside the
    range the K = 3 analysis admitted; on the BCG splits it binds at the lower
    end, which is worth reporting rather than hiding, since it means the
    alternative is taken at the boundary rather than estimated freely.
    """
    p = np.asarray(p_values, dtype=float)
    p = p[(p > 0) & (p < 1)]
    a = -len(p) / np.sum(np.log(p))
    return float(min(max(a, clamp[0]), clamp[1]))


def baseline_decisions(p_values, alpha=ALPHA):
    """Decisions for all eight comparators on one observed vector of p-values.

    Closed testing is a comparator in the applications as well as in the
    simulation study, so the three closed tests are scored here
    alongside the five rank-based procedures, by the same code that scores them
    in the grid.  The closed-testing masks are defined on a batch of ascending
    rows, so the observed vector is sorted into a batch of one and the mask is
    mapped back through the sort order.
    """
    p = np.asarray(p_values, dtype=float)
    out = {}
    for name, fn in BASELINES.items():
        d = fn(p, alpha)
        out[name] = {"n_rejected": int(np.sum(d)),
                     "rejected": sorted(np.where(d == 1)[0].tolist())}

    order = np.argsort(p)
    sorted_p = p[order][None, :]
    for name, fn in CLOSED_TESTS.items():
        mask_sorted = fn(sorted_p, alpha)[0]
        rejected = sorted(order[mask_sorted].tolist())
        out[name] = {"n_rejected": len(rejected), "rejected": rejected}
    return out


def report(app_name, names, p_values, algo_runs, baselines):
    print(f"\n===== {app_name} (K={len(p_values)}) =====")
    print("  baselines:")
    for m, r in baselines.items():
        studs = ", ".join(names[i] for i in r["rejected"]) or "(none)"
        print(f"    {m:<11}: {r['n_rejected']} rejected  [{studs}]")
    print("  algorithm:")
    for label, r in algo_runs.items():
        studs = ", ".join(names[i] for i in r["rejected"]) or "(none)"
        status = _solver_status(r)
        print(f"    {label:<24}: l*={r['l_star']}  "
              f"(sweeps={r['sweeps']}, {r['seconds']:.0f}s; {status})  "
              f"[{studs}]")


def _solver_status(result):
    """Return a compact, explicit certificate status for console reports."""
    if result.get("converged", False):
        return "status=certified"
    residual = result.get("final_residual", float("nan"))
    res_tol = result.get("res_tol", float("nan"))
    return (f"status=UNCERTIFIED, reason={result.get('reason', 'unknown')}, "
            f"residual={residual:.2e}, certificate_tol={res_tol:.2e}")


def _uses_reference_schema(config):
    """Whether this is the exact submitted run checked by the strict verifier.

    The released JSON schema predates embedded run metadata.  Preserve that
    schema only for the exact paper configuration; quick and customized runs
    carry their configuration in the artifact itself.
    """
    return (
        config.mode == "paper"
        and config.n_opt == N_OPT
        and config.tol == TOL
        and config.max_iter == MAX_ITER
        and config.seed == 42
        and config.cache_bytes == "auto"
    )


def _run_config_payload(config, cache_by_K):
    return {
        "mode": config.mode,
        "n_opt": config.n_opt,
        "tol": config.tol,
        "max_iter": config.max_iter,
        "seed": config.seed,
        "cache_bytes_requested": config.cache_bytes,
        "cache_bytes_resolved_by_k": {
            str(K): value for K, value in sorted(cache_by_K.items())
        },
        "allow_uncertified": config.allow_uncertified,
    }


def main(config):
    """Run all four applications using a resolved command-line config."""
    results = {}

    cache_by_K = {}
    solver_runs = []

    def decide(g, K, p_values, alpha=ALPHA):
        if K not in cache_by_K:
            cache_by_K[K] = optimizer_memory_plan(
                K, config.n_opt, config.cache_bytes
            )["resolved_cache_bytes"]
        result = algorithm_decision(
            g, K, p_values, alpha=alpha, n_opt=config.n_opt,
            seed=config.seed, cache_bytes=cache_by_K[K], tol=config.tol,
            max_iter=config.max_iter,
        )
        solver_runs.append(result)
        return result

    # ---------------- Camerer / SSRP: all 21 Stage-1 p-values --------------
    names = [n for n, _ in CAMERER]
    p = [v for _, v in CAMERER]
    K = len(p)
    base = baseline_decisions(p)
    runs = {}
    for theta in [1.5, 2.0, 2.5, 3.0]:
        label = f"mixture theta={theta}"
        print(f"[camerer] computing mu* for {label} (K={K}) ...", flush=True)
        runs[label] = decide(mixture_normal_g(theta), K, p)
    report("Camerer/SSRP Stage 1", names, p, runs, base)
    results["camerer"] = {"names": names, "p": p, "baselines": base,
                          "algorithm": runs}

    # Sensitivity: lower the six "< 0.001" values to 1e-10 and confirm the
    # rejection sets are unchanged (guards the upper-bound substitution, since
    # OMT rejection regions need not be monotone in a single p-value).  The
    # optimal policy mu* depends on (g, K, alpha) only, not on the observed
    # p-values, so we reuse each already-computed mu* and merely re-apply it.
    p_low = np.array([1e-10 if v == 0.001 else v for v in p], dtype=float)
    order_low = np.argsort(p_low)
    sens = {}
    for theta in [1.5, 2.0, 2.5, 3.0]:
        label = f"mixture theta={theta}"
        mu = np.array(runs[label]["mu"], dtype=float)
        l_star = optimal_l_star(mu, p_low[order_low], mixture_normal_g(theta), K)
        rejected = sorted(order_low[:l_star].tolist())
        sens[label] = {
            "l_star": int(l_star),
            "matches_upper_bound_run": (rejected == runs[label]["rejected"]),
        }
        print(f"[camerer] 1e-10 sensitivity, {label}: l*={l_star} "
              f"(was {runs[label]['l_star']}), "
              f"match={rejected == runs[label]['rejected']}", flush=True)
    results["camerer_sensitivity_1e-10"] = sens

    # ---------------- SPRINT: five CV endpoints ----------------------------
    names = [n for n, _ in SPRINT]
    p = [v for _, v in SPRINT]
    base = baseline_decisions(p)
    runs = {}
    for theta in [2.0, 2.5, 3.0]:
        runs[f"mixture theta={theta}"] = decide(
            mixture_normal_g(theta), len(p), p)
    # hierarchical remedy: two prespecified groups at alpha/2 each
    groupA = [("Heart failure", 0.002), ("Cardiovascular death", 0.005)]
    groupB = [("Myocardial infarction", 0.19), ("Non-MI acute coronary", 0.99),
              ("Stroke", 0.50)]
    hier = {}
    for gname, grp in [("A (strong-effect)", groupA), ("B (uncertain)", groupB)]:
        pg = [v for _, v in grp]
        hier[gname] = decide(mixture_normal_g(2.5), len(pg), pg,
                             alpha=ALPHA / 2)
        hier[gname]["names"] = [n for n, _ in grp]
    report("SPRINT endpoints", names, p, runs, base)
    print("  hierarchical (alpha/2 per group, mixture theta=2.5):")
    for gname, r in hier.items():
        studs = ", ".join(r["names"][i] for i in r["rejected"]) or "(none)"
        print(f"    group {gname}: l*={r['l_star']}  "
              f"({_solver_status(r)})  [{studs}]")
    results["sprint"] = {"names": names, "p": p, "baselines": base,
                         "algorithm": runs, "hierarchical": hier}

    # ---------------- OSC / RPP: six JPSP studies --------------------------
    names = [n for n, _, _ in OSC]
    p = [v for _, v, _ in OSC]
    base = baseline_decisions(p)
    runs = {"beta theta=0.6": decide(beta_g(0.6), len(p), p),
            "mixture theta=2.0": decide(mixture_normal_g(2.0), len(p), p)}
    report("OSC/RPP six JPSP studies", names, p, runs, base)
    results["osc"] = {"names": names, "p": p,
                      "N_rep": [n for _, _, n in OSC],
                      "baselines": base, "algorithm": runs}

    # ---------------- BCG subgroup analysis (K = 3), four splits -----------
    # The K = 3 application of Dubey and Huo (2025), recomputed here. The
    # alternative is fitted to the observed p-values as Beta(a, 1) by maximum
    # likelihood, clamped to [0.10, 0.90] as in the submitted analysis; a hits
    # the lower clamp in every split, so g -- and therefore mu* , which depends
    # on (g, K, alpha) and not on the data -- is common to all four.
    results["bcg"] = {"source": data_bcg.SOURCE,
                      "checksum": data_bcg.checksum(),
                      "splits": {}}
    for split in data_bcg.SPLITS:
        pooled = data_bcg.pooled(split)
        names = list(pooled)
        p = np.array([pooled[n]["p"] for n in names])
        a = fit_beta_shape(p)
        g = beta_g(a)
        base = baseline_decisions(p)
        runs = {f"beta a={a:.2f}": decide(g, len(p), p)}
        report(f"BCG subgroup analysis, {split} split", names, p, runs, base)
        results["bcg"]["splits"][split] = {
            "names": names, "p": p.tolist(), "a_hat": a,
            "pooled": pooled, "baselines": base, "algorithm": runs}

    if not _uses_reference_schema(config):
        results["_run_config"] = _run_config_payload(config, cache_by_K)
    atomic_json_dump(results, config.out)
    print(f"\nSaved {config.out}")
    return solver_runs


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply SPOT and the eight comparators to the four published datasets "
                    "(BCG subgroups, Camerer/SSRP, SPRINT, RPP/OSC).",
        epilog="Examples: --quick; --paper; --paper --cache-bytes 9GB "
               "--out /scratch/applications.json. Paper mode runs four expensive "
               "K=21 optimizations; inspect the memory plan before opting into caching.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick", dest="mode", action="store_const", const="quick",
        help=f"smoke-test every application (N_opt={QUICK_N_OPT:,})",
    )
    mode.add_argument(
        "--paper", "--full", dest="mode", action="store_const", const="paper",
        help=f"use the submitted settings (N_opt={N_OPT:,}); --full is an alias",
    )
    parser.add_argument(
        "--n-opt", type=_positive_int,
        help="optimizer Monte Carlo samples per configuration (overrides mode default)",
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
        "--seed", type=int, default=42,
        help="optimizer random seed (paper default: 42)",
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
        "--out",
        help="output JSON; relative paths use the current directory (default: a "
             "mode-specific file below code/reproduced/results)",
    )
    parser.add_argument(
        "--show-plan", action="store_true",
        help="print resolved settings and K=21 memory estimates, then exit",
    )
    return parser.parse_args(argv)


def _resolve_args(config):
    if config.n_opt is None:
        config.n_opt = N_OPT if config.mode == "paper" else QUICK_N_OPT
    if config.tol is None:
        config.tol = TOL
    if config.max_iter is None:
        config.max_iter = MAX_ITER if config.mode == "paper" else QUICK_MAX_ITER
    if config.out is None:
        name = "applications.json" if config.mode == "paper" else "applications_quick.json"
        config.out = paths.reproduced_result(name)
    else:
        config.out = os.path.abspath(os.path.expanduser(config.out))
    if os.path.splitext(config.out)[1].lower() != ".json":
        raise ValueError("--out must have a .json suffix")
    return config


def cli(argv=None):
    try:
        config = _resolve_args(_parse_args(argv))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Mode: {config.mode}; N_opt={config.n_opt:,}; tol={config.tol:g}; "
          f"max_iter={config.max_iter}; seed={config.seed}")
    _print_memory_plan("largest application", 21, config.n_opt,
                       config.cache_bytes)
    print(f"Output: {config.out}")
    if os.path.exists(config.out):
        print("WARNING: output already exists and will be replaced atomically.")
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
