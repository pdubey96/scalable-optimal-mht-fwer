# optimal-fwer

This is the runnable Python source for the simulations and applications in
“Most Powerful Tests with Exact Family-Wise Error Rate Control: Necessary
Conditions and Fast Computation for General K.” It implements SPOT for
exchangeable hypotheses using elementary symmetric polynomials and a
fixed-batch, residual-certified coordinate-descent algorithm.

## Source layout and installation

Run the commands below from this `code/` directory, or invoke a script by its
absolute path. The experiment drivers are source-material scripts rather than
installed console commands. In the supplied layout, `../results/` contains the
read-only reference results reported in the paper. A new run goes to
`code/reproduced/` by default. The bundled reference results are treated as
immutable; safe defaults and the orchestrator keep new runs separate from them.
An explicit path remains the caller's responsibility.

For ordinary package use, Python 3.10 or newer is supported and
`pyproject.toml` declares dependency floors:

```bash
python3 -m pip install -e '.[plots,dev]'
python3 -m pytest tests/
```

For numerical reproduction, use Python 3.11 or newer because the pinned NumPy
and SciPy releases require it. The released results used Python 3.14.3:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pip check
python3 -m general_k.environment --check-packages
```

`--check-packages` verifies the NumPy, SciPy, and Matplotlib versions while
allowing a different operating system or Python patch version. The stricter
`--check` also compares Python, operating system, and architecture, so it is
expected to return nonzero away from the recorded Windows/Python 3.14.3
machine. `requirements.txt` pins direct numerical, plotting, and test
dependencies; transitive plotting dependencies and the BLAS implementation are
not bit-for-bit locked.

## Fast independent check

After installing, this is the shortest end-to-end check:

```bash
python3 -m pytest tests/
python3 reproduce_paper.py --quick --output-dir reproduced/quick
```

In the standalone code bundle, the test run intentionally reports 15 skipped
manuscript-figure build checks: two typography checks for each of seven figure
PDFs and one all-figures presence check. These skips are not computational test
failures.

Quick mode exercises every main result driver at reduced Monte Carlo budgets,
runs the postprocessors against the bundled reference files, records commands
and logs, and writes only beneath the chosen directory. It is an installation
and workflow check, not a replacement for the paper-scale estimates. In
particular, bare `run_prior_objective.py --quick` now runs one known-certifying
four-prior slice (truncated normal, K=3) and returns nonzero if a requested
custom slice does not meet the residual certificate.

For a manual clean environment on Unix-like systems:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m general_k.environment --check-packages
python -m pytest tests/
python reproduce_paper.py --quick --output-dir reproduced/clean-check
```

On Windows, run the equivalent clean check below from `code/`. These commands
avoid environment activation and work in either PowerShell or Command Prompt.
Replace `3.14` with any installed Python version 3.11 or newer if needed:

```text
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m general_k.environment --check-packages
.\.venv\Scripts\python.exe -m pytest tests/
.\.venv\Scripts\python.exe reproduce_paper.py --quick --output-dir reproduced\clean-check
```

No driver downloads data or relies on a personal file path.

## Minimal library example

The built-in alternative factories are callable densities with a
`sample(n_samples, rng)` method. The same interface can be used for a custom
alternative.

```python
import numpy as np

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g
from general_k.decision import optimal_l_star

p_values = np.array([0.004, 0.018, 0.31])
g = truncated_normal_g(-2.0)
mu, history, diagnostics = coordinate_descent_fast(
    g, K=len(p_values), alpha=0.05, n_samples=100_000,
    seed=42, return_diagnostics=True,
)
order = np.argsort(p_values)
l_star = optimal_l_star(mu, p_values[order], g, len(p_values))
rejected_indices = order[:l_star]
```

Check `diagnostics["converged"]` before treating a computed policy as
certified. A bare density callable is sufficient for deterministic coefficient
and decision helpers, but Monte Carlo routines require both call and `sample`
methods because a density alone does not define a generally valid rejection
sampler.

## Reproduce and verify every paper result

Inspect the exact command plan first; paper mode is deliberately explicit
because it takes many hours:

```bash
python3 reproduce_paper.py --full --jobs 6 --dry-run
python3 reproduce_paper.py --full --jobs 6 --output-dir reproduced/paper
```

The orchestrator runs every numerical driver and postprocessor in a fixed
order, writes a manifest plus per-step logs, and finishes by running
`verify_reproduction.py`. An interrupted run can be continued with `--resume`.
Only a contiguous leading prefix with matching completed records, commands, and
logs is skipped; artifact-producing steps additionally require existing files
with matching SHA-256 hashes. Resume requires the existing manifest and refuses
any drift in source files, dependency versions, platform, or bundled reference
inputs. The first invalid step and everything downstream rerun, with declared
old outputs cleared first so stale artifacts cannot masquerade as new ones.

The verifier compares scientific values, decisions, solver diagnostics, and
data provenance with the bundled JSON. Configuration and embedded source-data
values must match exactly; computed floating-point values use tolerances well
below the paper's reporting precision. Only named elapsed-time fields are
ignored, and `table_scaling.tex` must match exactly:

```bash
python3 verify_reproduction.py reproduced/paper/results
```

Different-platform floating-point summation can prevent bitwise identity. The
default verifier tolerances are `rtol=1e-5` and `atol=5e-5`; both are
configurable and mismatches produce a nonzero exit status. Figure PDFs are not
compared byte-for-byte because renderer metadata varies. The orchestrator
records their hashes and requires them to exist; separate tests inspect the
committed manuscript figures' effective typography and the plotting styles'
greyscale-distinguishing encodings.

### Exact manual command sequence

The following lists the equivalent computational commands and every variant
explicitly. It does not provide the orchestrator's manifest, logs, collision
checks, or resume safeguards. Choose a fresh directory outside the bundled
results, plots, and manuscript paths; the orchestrator enforces this
automatically.

```bash
RUN_DIR="$(pwd)/reproduced/paper"

python3 -m general_k.environment --check-packages
python3 -m general_k.environment --out "$RUN_DIR/results/environment.json"

python3 run_all_experiments.py --k 3 4 5 6 \
  --models trunc mixture t beta --jobs 6 \
  --out "$RUN_DIR/results/all_experiments.json"
python3 run_all_experiments.py --k 3 4 5 6 --models normal --jobs 6 \
  --out "$RUN_DIR/results/untruncated_normal.json"

python3 run_scaling_convergence.py --paper \
  --scaling-out "$RUN_DIR/results/scaling_results.json" \
  --sensitivity-out "$RUN_DIR/results/sensitivity_results.json" \
  --convergence-plot "$RUN_DIR/plots/convergence_trajectory.pdf" \
  --scaling-plot "$RUN_DIR/plots/scaling_with_K.pdf"
python3 run_gamma_sweep.py --out "$RUN_DIR/results/gamma_sweep.json"

# One disclosed Student-t row in the released file is not residual-certified.
python3 run_prior_objective.py --allow-uncertified \
  --out "$RUN_DIR/results/prior_objective.json"

python3 run_prior_comparison.py --k 6 12 --model trunc --theta -2 \
  --out "$RUN_DIR/results/prior_comparison.json"
# Three disclosed Student-t rows are not residual-certified.
python3 run_prior_comparison.py --k 6 --model t --theta 6 \
  --allow-uncertified --out "$RUN_DIR/results/prior_comparison_t.json"
python3 run_prior_comparison.py --k 6 12 --model beta --theta 0.4 \
  --out "$RUN_DIR/results/prior_comparison_beta.json"

python3 run_applications.py --paper \
  --out "$RUN_DIR/results/applications.json"

python3 make_table_scaling.py \
  --input "$RUN_DIR/results/scaling_results.json" \
  --out "$RUN_DIR/results/table_scaling.tex"
python3 plot_power_curves.py \
  --input "$RUN_DIR/results/all_experiments.json" --out-dir "$RUN_DIR/plots"
python3 plot_prior_objective.py \
  --input "$RUN_DIR/results/prior_objective.json" --out-dir "$RUN_DIR/plots"
python3 plot_gamma_sweep.py \
  --input "$RUN_DIR/results/gamma_sweep.json" --out-dir "$RUN_DIR/plots"
python3 rank_correlation_diagnostic.py \
  --input "$RUN_DIR/results/prior_objective.json"
python3 check_weak_monotonicity.py \
  --allow-violations \
  --results "$RUN_DIR/results/prior_objective.json"

python3 verify_reproduction.py "$RUN_DIR/results"
```

All result JSON is written atomically, so interruption cannot leave a malformed
destination file. Relative paths supplied explicitly on the command line use
the caller's working directory. Set `OPTIMAL_FWER_OUTPUT_DIR` to change the
default reproduced-output root without changing individual commands; use an
absolute value because a relative environment-variable value is resolved from
the caller's working directory.

Quick and customized application/scaling results include a `_run_config` block
with the resolved budgets, seeds, tolerances, and cache choice. The exact paper
preset preserves the released legacy JSON schema; its complete command and
environment are recorded in the reproduction manifest. `make_table_scaling.py`
uses the recorded `n_eval` for custom confidence intervals (or accepts an
explicit `--n-eval` override); legacy paper results use 50,000.

## Runtime and memory

Runtimes below are single-core wall times from the reference environment;
hardware and cache mode matter.

| Driver | Paper result | Reference runtime |
|---|---|---:|
| `run_all_experiments.py` | 112-setting grid | 14.5 h sequential; about 2.5 h with 6 workers |
| `run_all_experiments.py --models normal` | untruncated arm | 47 min |
| `run_scaling_convergence.py --paper` | scaling, convergence, α/N sensitivity | about 1 h |
| `run_gamma_sweep.py` | configuration crossover sweep | 43 min |
| `run_prior_objective.py` | prior-objective table and figure | 141 min |
| `run_prior_comparison.py` (three commands) | choice-of-prior study | 76 min for the truncated-normal command alone; no three-command total is stored |
| `run_applications.py --paper` | all four applications | about 1 h with the high-memory cache |

The three prior workflows use 400,000 evaluation replications because their
sparse-configuration margins are only a few tenths of a percent. The main grid
and scaling study use 50,000 where substantive contrasts are larger.

The K=21 Camerer application is the memory-critical step. Inspect it without
running:

```bash
python3 run_applications.py --paper --show-plan
```

The safe automatic mode streams coefficient tensors: its persistent-array
lower bound is about 1.00 GiB, before NumPy temporaries and interpreter
overhead. It uses less memory but can be substantially slower. The reference
run cached every tensor; that mode has a 7.89 GiB persistent-array lower bound
before temporaries. On a host with comfortably more memory, the former caching
behavior is selected explicitly with `--cache-bytes 9GB`. Cache mode changes
runtime and memory, not the fixed-batch numerical calculation. The largest
paper scaling solve (K=12) retains caching automatically and has a 1.61 GiB
persistent-array lower bound before temporaries.

## Result map and data provenance

| Driver/postprocessor | Backs | Output |
|---|---|---|
| `run_all_experiments.py` | §4 robustness grid | `all_experiments.json`, `untruncated_normal.json` |
| `run_scaling_convergence.py` | Main Table 2/Figure 2; Supplementary Figure S1 and Tables S1, S2, S4 | `scaling_results.json`, `sensitivity_results.json`, two PDFs |
| `run_gamma_sweep.py` | §4.1 crossover statements | `gamma_sweep.json` |
| `run_prior_objective.py` | Table 1 and Figure 1 | `prior_objective.json` |
| `run_prior_comparison.py` | choice-of-prior discussion | three `prior_comparison*.json` files |
| `run_applications.py` | §5 and Supplementary §S7, including Tables S3, S5, S6 | `applications.json` |
| `make_table_scaling.py` | Table 2 body | `table_scaling.tex` |
| `plot_power_curves.py` | Figures 3–4 and Supplementary Figures S2–S5 | four PDFs |
| `plot_prior_objective.py` | Figure 1 | PDF and PNG |
| `plot_gamma_sweep.py` | configuration-sweep diagnostic | PDF and PNG |
| `rank_correlation_diagnostic.py` | reported 0.70 rank correlation | stdout/log |
| `check_weak_monotonicity.py` | §3.2 reasonableness audit | stdout/log |

The field-level mapping, grouping, rounding, and timing qualifications for every
numbered main and supplementary table and figure are in `../readme.txt`.

Application values and primary sources are documented at the top of
`run_applications.py`. The BCG trial counts, source, and checksum are in
`general_k/data_bcg.py`. All other application values are embedded with their
source-table provenance. No network access is used at runtime.

## Certification and diagnostic tools

A policy is certified only when its empirical complementarity residual is at
most `sqrt(alpha * (1 - alpha) / N)`, one Monte Carlo standard error at the
constraint level. Iterate-change and sweep-limit exits are safeguards, not
certificates. The grid's deterministic escalation ladder restarts an
uncertified setting with the same seed, first allowing more sweeps and then
refining the bisection target; every published grid setting certifies.

All six experiment drivers return nonzero if a requested policy misses its
residual certificate. The prior scripts preserve the few non-certifying stress-test rows
used and disclosed in the paper; `--allow-uncertified` explicitly acknowledges
those diagnostic rows. The weak-monotonicity audit likewise requires
`--allow-violations` for the paper's disclosed Student-t counterexamples.
`rerun_uncertified.py`
reads a chosen grid and writes a patched copy under `reproduced/results` by
default; it no longer modifies the bundled grid.

Additional validation executables are retained separately from the main
workflow:

- `check_prior_monotonicity.py`: fixed-batch own-coordinate monotonicity check.
- `diag_residual_terms.py`: package-anchored/configurable residual diagnosis.
- `verify_general_gamma_objective.py`: ESP objective identity check.
- `verify_tie_set_polynomial.py`: tie-set polynomial calculation.
- `test_sparse_tilt_ftol.py`: exploratory steep-target tolerance check.

Each supported command documents its parameters with `--help`. The code is
distributed under the MIT License; see `LICENSE`.
