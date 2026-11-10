"""Diagnose which coordinate carries each recorded solver residual.

The default input is the released Student-t prior-comparison result.  No policy
is re-solved: the script reads its recorded multipliers and regenerates the
fixed Monte Carlo batches under the solver's recorded setting and seed.  The
input may be supplied from any working directory.

Examples
--------
    python diag_residual_terms.py
    python diag_residual_terms.py --results /path/to/prior_comparison_t.json
    python diag_residual_terms.py --seed 42
"""

import argparse
import json
import os

import numpy as np

from general_k import paths
from general_k.algorithm import default_res_tol
from general_k.coefficients import precompute_coefficients
from general_k.prior_objective import (
    beta_binomial_prior, point_prior, prior_power_coefficients,
)
from general_k.target import prepare_samples, target_function_fast
from run_all_experiments import MODEL_CONFIGS


DEFAULT_RESULTS = os.path.join(paths.RESULTS_DIR, "prior_comparison_t.json")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Preserve examples while displaying argument defaults."""


def _existing_file(value):
    """An argparse type for a readable regular file."""
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"file is not readable: {value}")
    return path


def _nonnegative_int(value):
    """An argparse type for NumPy-compatible random seeds."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=_existing_file,
        default=DEFAULT_RESULTS,
        help="Prior-comparison JSON to diagnose; all rows must describe one setting",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=42,
        help="Seed used to regenerate the fixed optimization batches",
    )
    return parser


def _priors_for(K):
    """Prior vectors keyed by the names stored by run_prior_comparison.py."""
    return {
        "point gamma=K (manuscript)": point_prior(K, K),
        "Beta(1,1) = uniform": beta_binomial_prior(K, 1.0, 1.0),
        "Beta(0.5,0.5) Jeffreys": beta_binomial_prior(K, 0.5, 0.5),
        "Beta(0.25,0.25)": beta_binomial_prior(K, 0.25, 0.25),
        "Beta(1,2) sparse-tilt": beta_binomial_prior(K, 1.0, 2.0),
        "Beta(1,3) sparse-tilt": beta_binomial_prior(K, 1.0, 3.0),
        "Beta(2,1) dense-tilt": beta_binomial_prior(K, 2.0, 1.0),
    }


def _setting(records):
    """Return the shared setting, rejecting mixed or incomplete result files."""
    if not isinstance(records, list) or not records:
        raise ValueError("results must be a nonempty JSON list")

    required = {"K", "model", "theta", "alpha", "n_algo_samples", "prior_name",
                "mu", "converged", "reason", "sweeps", "final_residual"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"row {index} is not a JSON object")
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"row {index} is missing fields: {', '.join(missing)}")

    settings = {
        (int(record["K"]), record["model"], float(record["theta"]),
         float(record["alpha"]), int(record["n_algo_samples"]))
        for record in records
    }
    if len(settings) != 1:
        raise ValueError(
            "results contain multiple settings; pass a file for one "
            "(K, model, theta, alpha, n_algo_samples) setting"
        )
    return settings.pop()


def diagnose(records, seed=42):
    """Print the per-coordinate residual decomposition for recorded policies."""
    K, model, theta, alpha, n_samples = _setting(records)
    if model not in MODEL_CONFIGS:
        raise ValueError(
            f"unknown model {model!r}; expected one of {tuple(MODEL_CONFIGS)}"
        )

    priors = _priors_for(K)
    unknown = sorted({record["prior_name"] for record in records}.difference(priors))
    if unknown:
        raise ValueError(f"unrecognized prior name(s): {', '.join(unknown)}")

    g_func = MODEL_CONFIGS[model]["factory"](theta)
    res_tol = default_res_tol(alpha, n_samples)
    rng = np.random.default_rng(seed)
    batches = [
        prepare_samples(gamma, g_func, K, n_samples, rng) for gamma in range(K)
    ]

    print(f"setting: K={K}, model={model}, theta={theta:g}, alpha={alpha:g}, "
          f"n_algo_samples={n_samples:,}, seed={seed}")
    print(f"res_tol = {res_tol:.4e}\n")

    for record in records:
        name = record["prior_name"]
        mu = np.asarray(record["mu"], dtype=float)
        if mu.shape != (K,):
            raise ValueError(
                f"prior {name!r} has mu shape {mu.shape}; expected ({K},)"
            )
        power_coeff_fn = prior_power_coefficients(priors[name])

        terms, fwers = [], []
        for gamma in range(K):
            sorted_p, sort_indices, null_label = batches[gamma]
            _, b_coeff = precompute_coefficients(sorted_p, g_func, K)
            a_batch = power_coeff_fn(sorted_p, g_func, K)
            fwer = alpha + target_function_fast(
                gamma, mu[gamma], mu, K, alpha,
                sort_indices, null_label, a_batch, b_coeff,
            )
            fwers.append(fwer)
            terms.append(abs(min(mu[gamma], alpha - fwer)))

        binding_gamma = int(np.argmax(terms))
        flag = "CONVERGED" if record["converged"] else "  FAILED "
        print(f"{flag}  {name}")
        print(f"          reason={record['reason']:<9s} "
              f"sweeps={record['sweeps']:4d} "
              f"recorded_residual={record['final_residual']:.3e}")
        print(f"          {'gamma':>5s} {'mu_g':>10s} {'Fhat_g':>10s} "
              f"{'alpha-Fhat':>11s} {'term':>10s}")
        for gamma in range(K):
            star = " <== max" if gamma == binding_gamma else ""
            violation = (
                " VIOLATED" if mu[gamma] == 0.0 and fwers[gamma] > alpha else ""
            )
            print(f"          {gamma:5d} {mu[gamma]:10.6f} "
                  f"{fwers[gamma]:10.6f} {alpha-fwers[gamma]:+11.6f} "
                  f"{terms[gamma]:10.3e}{star}{violation}")
        print(f"          -> max-residual coordinate gamma={binding_gamma}, "
              f"mu={mu[binding_gamma]:.6f}, term={terms[binding_gamma]:.3e} "
              f"({terms[binding_gamma]/res_tol:.1f} x res_tol)\n")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        with open(args.results, encoding="utf-8") as stream:
            records = json.load(stream)
        diagnose(records, seed=args.seed)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
