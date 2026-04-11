# scalable-optimal-mht-fwer

A Python implementation of a scalable algorithm for the **most powerful multiple hypothesis test with exact family-wise error rate (FWER) control** under an arbitrary number $K$ of exchangeable hypotheses.

This repository contains the algorithm, simulation code, unit tests, raw simulation outputs, and figures that accompany the paper *"Optimal multiple testing under family-wise error control: elementary symmetric polynomials and a scalable algorithm"* by Prasanjit Dubey and Xiaoming Huo (Georgia Institute of Technology).

---

## Background

Classical FWER-controlling procedures such as **Bonferroni**, **Holm**, and **Hommel** are widely used because they are distribution-free and simple, but they are not optimal: for a fixed nominal level $\alpha$ they leave statistical power on the table, particularly as $K$ grows.

Rosset, Heller, Painsky & Aharoni (2022, *JRSS-B*) cast the problem of the most powerful $\alpha$-level FWER-controlling test as a linear program in the space of decision rules and solved it for $K=2$. The case $K=3$ was subsequently worked out in closed form in a companion paper ([arXiv:2512.14131](https://arxiv.org/abs/2512.14131)). For general $K$, however, a direct attack on the linear program is infeasible: a naive enumeration involves $2^K - 1$ configurations and produces error-rate coefficients that no longer admit a tractable closed form.

The approach implemented here gives a **scalable** alternative that works for arbitrary $K$ (in practice, up to moderate $K$ of a few dozen). The key ingredients are:

1. **A closed-form characterization of the FWER constraint coefficients via elementary symmetric polynomials.** Instead of enumerating configurations, the coefficients $b_{l,k}(\vec u)$ are written as products involving the elementary symmetric polynomial $e_{l-k+1}\{g(u_{k+1}),\ldots,g(u_K)\}$, where $g$ is the likelihood ratio (alternative-to-null $p$-value density ratio). This reduces the per-sample computation from combinatorial to polynomial in $K$.
2. **A global monotonicity theorem** for the dual objective $F_\gamma(\mu)$, which guarantees that a coordinate-descent scheme with one-dimensional bisection converges to the optimal dual multiplier $\mu^*$.
3. **A coordinate-descent algorithm with linear convergence** that computes $\mu^*$ and the corresponding decision rule. Empirically, the algorithm converges in a handful of outer passes for the models considered.
4. **Distribution-specific Monte Carlo estimators** of the constraint and objective integrals over the ordered simplex $Q = \{\vec u \in [0,1]^K : u_1 \le \cdots \le u_K\}$, implemented for four alternative models: truncated normal, mixture of normals, Student's $t$, and beta.

The net result is a procedure that matches Bonferroni/Holm/Hommel in validity (empirical FWER stays at or below the nominal level) while delivering materially higher power as $K$ grows.

---

## Headline result

The power advantage of the optimal algorithm over standard procedures grows with $K$. Under the truncated normal model at $\theta = -2.0$ and $\alpha = 0.05$:

| $K$ | Optimal algorithm | Hommel | Absolute gain | Relative gain |
|----|-------------------|--------|---------------|---------------|
| 3  | 0.635             | 0.555  | 0.080         | 15%           |
| 6  | 0.626             | 0.454  | 0.172         | 38%           |
| 12 | 0.615             | 0.334  | 0.281         | 84%           |

Hommel's power collapses with $K$ while the optimal algorithm's remains nearly flat, reflecting its ability to exploit joint information across hypotheses. The maximum empirical FWER across all settings in `results/all_experiments.json` is at most 0.053 (within Monte Carlo error of the nominal 0.05).

---

## Repository layout

```
scalable-optimal-mht-fwer/
├── code/
│   ├── general_k/              Core Python package
│   │   ├── __init__.py
│   │   ├── algorithm.py         Coordinate-descent solver for optimal dual multipliers
│   │   ├── alternatives.py      Alternative-density models: trunc normal, mixture, t, beta
│   │   ├── baselines.py         Bonferroni, Holm, Hommel implementations
│   │   ├── coefficients.py      Elementary symmetric polynomial coefficient routines
│   │   ├── decision.py          Decision-rule evaluation and block rejection
│   │   ├── simulations.py       Monte Carlo power / FWER estimation
│   │   └── target.py            Dual objective F_gamma and its partial derivatives
│   ├── tests/                   Pytest suite (unit + integration)
│   │   ├── test_algorithm.py
│   │   ├── test_baselines.py
│   │   ├── test_coefficients.py
│   │   ├── test_decision.py
│   │   └── test_target.py
│   ├── run_all_experiments.py   Reproduce every simulation setting end to end
│   └── pyproject.toml           Package metadata and dependencies
├── plots/                       Figures generated from results/ (all PDF)
│   ├── general_k_{model}_K{3..6}.pdf   Power curves per model and K
│   ├── scaling_with_K.pdf              Power advantage scaling with K
│   └── convergence_trajectory.pdf      Coordinate-descent convergence
├── results/                     Frozen simulation outputs
│   ├── all_experiments.json            Main grid (4 models × K=3..6 × signal strengths)
│   ├── larger_k_results.json           Extended grid at K=8, 10, 12
│   └── table_scaling.tex               LaTeX table summarizing the scaling result
└── README.md
```

---

## Installation

Requires Python 3.10 or newer. Only hard dependencies are `numpy` and `scipy`; `matplotlib` is needed for plotting.

```bash
git clone <repository-url>
cd scalable-optimal-mht-fwer/code
pip install -e .
```

To also install plotting and development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Running the tests

```bash
cd code
python -m pytest tests/ -v
```

The test suite covers:

- **`test_coefficients.py`** — elementary symmetric polynomial evaluation, $b_{l,k}$ formulas, edge cases at $l=0$ and $l=K-1$.
- **`test_target.py`** — dual objective $F_\gamma(\mu)$, its monotonicity properties, and agreement with a brute-force enumeration for small $K$.
- **`test_algorithm.py`** — coordinate-descent convergence and matching of the optimal decision rule against the known $K=3$ closed-form solution.
- **`test_decision.py`** — block-rejection semantics and ordered-simplex bookkeeping.
- **`test_baselines.py`** — Bonferroni, Holm, Hommel implementations against textbook expectations.

---

## Reproducing the simulations

The full experiment grid takes several hours on a single machine. The entry point supports subsetting by $K$ and by model:

```bash
cd code

# Full grid (takes hours)
python run_all_experiments.py

# A single K and model (fast smoke test)
python run_all_experiments.py --k 3 --models trunc

# Custom K range over two models
python run_all_experiments.py --k 3 4 5 6 --models trunc mixture

# Print time estimates only (no actual runs)
python run_all_experiments.py --estimate
```

Outputs are written to `../results/all_experiments.json` (and `../results/larger_k_results.json` for the extended scaling experiment) and figures are written to `../plots/`.

### Simulation settings

Unless overridden on the command line, each run uses:

- Nominal level $\alpha = 0.05$
- $N = 100{,}000$ Monte Carlo samples for estimating the coefficients used inside the coordinate-descent solver
- $N_{\text{MC}} = 50{,}000$ independent replicates for estimating realized power and FWER
- Signal-strength grids per model as defined in `MODEL_CONFIGS` inside `run_all_experiments.py`

The four alternative-density models are:

| Key       | Model          | Signal parameter | Parameter grid                     |
|-----------|----------------|------------------|-------------------------------------|
| `trunc`   | Truncated normal | $\theta$       | $-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0$ |
| `mixture` | Mixture of normals | $\theta$     | $-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0$ |
| `t`       | Student's $t$  | degrees of freedom | $2, 4, 6, 8, 10, 12, 14, 16, 18, 20$ |
| `beta`    | Beta           | $\theta < 1$   | $0.8, 0.6, 0.4, 0.2$                |

---

## Interpreting the frozen outputs

### `results/all_experiments.json`

The top-level JSON is a nested dictionary keyed first by $K$ and then by model. Each leaf contains:

- `thetas` — signal-strength grid actually run
- `power_algo`, `power_bonferroni`, `power_holm`, `power_hommel` — estimated power per method at each $\theta$
- `fwer_algo`, `fwer_bonferroni`, `fwer_holm`, `fwer_hommel` — estimated realized FWER
- `mu_star` — optimal dual multiplier vector found by the algorithm
- `iterations`, `elapsed_seconds` — convergence diagnostics

### `results/larger_k_results.json`

Same schema as above but restricted to the truncated normal model at $K \in \{8, 10, 12\}$, which drives the headline scaling plot.

### `results/table_scaling.tex`

LaTeX source for the scaling table reproduced above, generated directly from `all_experiments.json` and `larger_k_results.json`.

### `plots/`

- **`general_k_{model}_K{3..6}.pdf`** — power curves comparing the optimal algorithm against Bonferroni, Holm, and Hommel across signal strengths, one figure per model / $K$ combination.
- **`scaling_with_K.pdf`** — two-panel figure showing (left) absolute power advantage over each baseline as a function of $K$, and (right) each method's raw power.
- **`convergence_trajectory.pdf`** — per-iteration trajectory of the coordinate-descent solver demonstrating linear convergence.

All plot PDFs are produced by `run_all_experiments.py`; regenerating them overwrites existing files.

---

## Dependencies

Pinned minimums (see `code/pyproject.toml`):

- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7` (optional, required for plotting)
- `pytest >= 7.0` (optional, required for tests)

---

## Authors

- **Prasanjit Dubey** — Georgia Institute of Technology (`pdubey31@gatech.edu`)
- **Xiaoming Huo** — Georgia Institute of Technology (`huo@gatech.edu`)

---

## License

MIT. See `code/pyproject.toml` for the declared license string.

---

## Related work

- Companion paper handling the $K=3$ case in closed form: [arXiv:2512.14131](https://arxiv.org/abs/2512.14131)
- Theoretical foundation for the linear-programming dual formulation: Rosset, Heller, Painsky & Aharoni (2022), *J. R. Stat. Soc. B*, [doi:10.1111/rssb.12507](https://doi.org/10.1111/rssb.12507)
