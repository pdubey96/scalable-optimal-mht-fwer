Code and data for:
"Most Powerful Tests with Exact Family-Wise Error Rate Control:
Necessary Conditions and Fast Computation for General K"

OVERVIEW
========

This folder contains the Python source, embedded application data, released
reference results, and plotting code needed to reproduce the computational
results in the manuscript and its supplement. Start in the code/ directory.

Folder contents
---------------

  code/     Runnable source, tests, pinned requirements, and technical README
  results/  Read-only reference JSON and generated Table 2 tabular source
  plots/    Reference diagnostic plot supplied with the computational results

New runs are written below code/reproduced/ by default and do not overwrite
the bundled reference results. The application inputs are embedded in the
source with provenance, so the workflow makes no runtime downloads and does
not depend on personal or absolute file paths.

For implementation details, command usage, runtime and memory guidance, data
provenance, certification diagnostics, and the broader result map, see
code/README.md.


INSTALLATION AND A FAST CHECK
=============================

Python 3.11 or newer is required for the pinned numerical-reproduction
environment. On macOS/Linux, create a clean virtual environment and run the
checks below from this "Codes and Datasets" folder. Equivalent Windows commands
are in code/README.md.

  cd code
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install -r requirements.txt
  python -m pip install -e .
  python -m pip check
  python -m general_k.environment --check-packages
  python -m pytest tests/
  python reproduce_paper.py --quick --output-dir reproduced/quick

Quick mode is a bounded installation and workflow check. It runs every main
result driver at reduced Monte Carlo budgets; it does not reproduce the
paper-scale estimates.

When this standalone code bundle is tested, pytest intentionally reports 15
skips: two typography checks for each of seven manuscript figure PDFs and one
all-figures presence check. These are build-artifact checks, not computational
test failures.


ONE COMMAND FOR ALL PAPER-SCALE RESULTS
=======================================

Inspect the full command plan, then run it:

  python3 reproduce_paper.py --full --jobs 6 --dry-run \
    --output-dir reproduced/full-plan
  python3 reproduce_paper.py --full --jobs 6 --output-dir reproduced/paper

This seeded workflow runs every numerical driver and postprocessor, records
the commands and environment in a manifest, and finishes by comparing the new
scientific results with the bundled reference results. It can take many hours;
code/README.md gives per-step runtime and memory estimates and explains how to
resume an interrupted run.

The full run creates the numerical content for all three main-manuscript
tables, fresh versions of the four main-manuscript figure PDFs, and the
supplementary results. Figure 2's timing panel necessarily records the current
computer; the exact submitted timing points and fit can instead be re-rendered
from the bundled reference JSON with the fast command below. Table 2 is emitted
as LaTeX. Tables 1 and 3 use compact, editorially grouped LaTeX in the
manuscript; their exact displayed numbers and decisions come from the JSON
fields identified below rather than from separate table-formatting scripts.


EXACT MAIN-MANUSCRIPT FIGURE AND TABLE MAP
==========================================

The command blocks use POSIX shell syntax (macOS/Linux); code/README.md gives
the corresponding Windows environment setup. Run every command in this
section from code/. Set a fresh output directory once per shell session:

  RUN_DIR="$(pwd)/reproduced/by-item"

The list below covers every numbered figure and table in the main manuscript:
Tables 1-3 and Figures 1-4. There are no other numbered main-manuscript figures
or tables.


Table 1: configuration robustness of the flat-prior objective
----------------------------------------------------------------

Manuscript label: tab:gamma_sweep

Run:

  python3 run_prior_objective.py --allow-uncertified \
    --out "$RUN_DIR/results/prior_objective.json"

Use: $RUN_DIR/results/prior_objective.json

The six displayed settings are truncated normal theta=-2 at K=3, 6, and 12;
truncated normal theta=-3 at K=6; Student-t with 6 degrees of freedom at K=6;
and Beta theta=0.4 at K=6. For each setting:

  * The "gamma=K objective" columns use the entry whose prior_name is
    "point gamma=K": 100*min_gain and 100*gains[-1].
  * The "flat prior" columns use the entry whose prior_name is "uniform":
    100*min_gain, 100*max(gains), 100*gains[-1], and 100*price_at_K.
  * The manuscript displays these percentage values rounded to one decimal.

The flag --allow-uncertified acknowledges the one disclosed Student-t,
binomial-p=0.5 prior row that misses the residual certificate. That diagnostic
row is retained in the result file but is not one of the point/flat-prior rows
displayed in Table 1.


Figure 1: prior-weighted power, gain, trade-off, and FWER panels
----------------------------------------------------------------

Manuscript label and file: fig:prior_objective, prior_objective.pdf

First run the Table 1 command above, then run:

  python3 plot_prior_objective.py \
    --input "$RUN_DIR/results/prior_objective.json" \
    --out-dir "$RUN_DIR/plots"

Outputs:

  $RUN_DIR/plots/prior_objective.pdf
  $RUN_DIR/plots/prior_objective.png


Table 2: power and maximum FWER as K grows
-------------------------------------------

Manuscript label: tab:scaling

Run the shared Table 2/Figure 2 computation:

  python3 run_scaling_convergence.py --paper \
    --scaling-out "$RUN_DIR/results/scaling_results.json" \
    --sensitivity-out "$RUN_DIR/results/sensitivity_results.json" \
    --convergence-plot "$RUN_DIR/plots/convergence_trajectory.pdf" \
    --scaling-plot "$RUN_DIR/plots/scaling_with_K.pdf"

Then generate the table's LaTeX tabular source:

  python3 make_table_scaling.py \
    --input "$RUN_DIR/results/scaling_results.json" \
    --out "$RUN_DIR/results/table_scaling.tex"

Outputs used for Table 2:

  $RUN_DIR/results/scaling_results.json
  $RUN_DIR/results/table_scaling.tex

The generated file supplies every displayed numerical cell. The manuscript
uses the reader-facing heading "Closed Fisher" for the JSON key
"Fisher-closed" and applies its caption, label, spacing, and table wrapper.


Figure 2: power advantage and optimization-time scaling
---------------------------------------------------------

Manuscript label and file: fig:scaling, scaling_with_K.pdf

The run_scaling_convergence.py command under Table 2 creates:

  $RUN_DIR/plots/scaling_with_K.pdf

The same scaling_results.json therefore supplies both Table 2 and Figure 2.
The power panel is directly comparable across machines, while the timing panel
reports the current machine's measured optimization times. To reproduce the
submitted timing points, slope, and R-squared exactly, render Figure 2 from the
bundled scaling_results.json using the one-line command in "Fast re-creation
from the bundled reference results" below.


Figures 3 and 4: power curves for K=3 and K=6
------------------------------------------------

Manuscript labels and files:

  fig:power_curves_K3, power_curves_K3.pdf
  fig:power_curves_K6, power_curves_K6.pdf

Run:

  python3 run_all_experiments.py --k 3 4 5 6 \
    --models trunc mixture t beta --jobs 6 \
    --out "$RUN_DIR/results/all_experiments.json"

  python3 plot_power_curves.py \
    --input "$RUN_DIR/results/all_experiments.json" \
    --out-dir "$RUN_DIR/plots"

Outputs used in the main manuscript:

  $RUN_DIR/plots/power_curves_K3.pdf
  $RUN_DIR/plots/power_curves_K6.pdf

The same plotting command also creates the K=4 and K=5 figures used in the
supplement.


Table 3: Camerer et al. Stage-1 replication decisions
------------------------------------------------------

Manuscript label: tab:camerer

Run:

  python3 run_applications.py --paper \
    --out "$RUN_DIR/results/applications.json"

Use: the `camerer` entry in $RUN_DIR/results/applications.json.

The entry records all 21 p-values and the exact rejection sets for
Bonferroni, Holm, Hochberg, Hommel, Romano-Wolf, the three closed tests, and
SPOT at theta=1.5, 2, 2.5, and 3. The manuscript groups identical rows and
columns for compact presentation: six p-values reported as <.001, the rows at
.003, .006, and .011, three p-values from .022-.025, and nine p-values at
least .089. "Stepwise" groups the four identical stepwise decisions, and
"St./md." groups the identical closed-Stouffer and closed-median decisions.
The rejection totals are 6, 7, 6, 0, 12, 9, 8, and 8 in manuscript column
order.


EXACT SUPPLEMENTARY FIGURE AND TABLE MAP
========================================

The commands above also cover every numbered supplementary figure and table:
Figures S1-S5 and Tables S1-S6. The bundled references below are relative to
code/; a fresh per-item or full run writes the same-named JSON below
$RUN_DIR/results.

Figure S1: convergence trajectory
----------------------------------

Supplement label and file: fig:supp_convergence,
convergence_trajectory.pdf. The run_scaling_convergence.py command under main
Table 2 creates $RUN_DIR/plots/convergence_trajectory.pdf from the `deltas`
arrays in scaling_results.json. The bundled numerical reference is
../results/scaling_results.json. The fast reference command below re-renders
the submitted trajectory without repeating the Monte Carlo calculation.

Figures S2-S5: power curves for K=3, 4, 5, and 6
-------------------------------------------------

Supplement labels: fig:supp_K3 through fig:supp_K6. The
run_all_experiments.py and plot_power_curves.py commands under main Figures 3
and 4 create all four power_curves_K*.pdf files. Figures S2 and S5 repeat main
Figures 3 and 4; Figures S3 and S4 are the K=4 and K=5 outputs. The bundled
numerical reference is ../results/all_experiments.json.

Tables S1-S6
------------

There are no separate LaTeX-formatting scripts for the supplementary tables.
Their formatted LaTeX is manually transcribed in the supplement from the JSON
fields below, using the stated grouping and rounding rules.

  * Table S1 (tab:supp_scaling_fwer) uses ../results/scaling_results.json.
    For each displayed K and procedure, take the maximum of that procedure's
    fwer_0 through fwer_{K-1} fields and round to three decimals. "Closed
    Fisher" is the JSON procedure "Fisher-closed"; "SPOT" is "Algorithm".

  * Table S2 (tab:supp_sensitivity_alpha) uses
    `alpha["0.01"].results` and `alpha["0.1"].results` in
    ../results/sensitivity_results.json. The alpha=0.05 column uses
    `["6"].results` in ../results/scaling_results.json. Procedure rows use
    `pi_K`; the last row is the maximum of SPOT's fwer_0 through fwer_5. All
    displayed values are rounded to three decimals.

  * Table S3 (tab:supp_bcg) uses the `bcg.splits` entries in
    ../results/applications.json. The rows are `alloc`, `lat`, `risk`, and
    `era`; use `p`, each baseline's `n_rejected`, and
    `algorithm["beta a=0.10"].l_star`. The "Stepwise and closed" column groups
    procedures whose counts coincide. P-values are shown to two significant
    figures in scientific notation and rejection counts as integers.

  * Table S4 (tab:supp_timing) uses each K entry in
    ../results/scaling_results.json: `opt_seconds`, `total_seconds`, `sweeps`,
    and `results.Algorithm.pi_K`. Optimisation time is rounded to one decimal,
    total time to the nearest second, and power to three decimals. The two time
    columns are machine-dependent: a fresh run records the current machine and
    will normally differ. The bundled JSON preserves the submitted timings,
    and verify_reproduction.py deliberately excludes timing fields while still
    comparing the scientific results and convergence diagnostics.

  * Table S5 (tab:supp_camerer) uses the `camerer` entry in
    ../results/applications.json. Pair `names` with `p`, sort by p-value, and
    mark rejection when the corresponding index occurs in a method's
    `rejected` list. The stepwise column groups Holm, Hochberg, Hommel, and
    Romano-Wolf; "St./md." groups closed Stouffer and closed median; the SPOT
    column uses `algorithm["mixture theta=2.0"]`. P-values are displayed to
    three decimals, with the six values reported as <.001 stored as .001 and
    marked by a dagger.

  * Table S6 (tab:supp_camerer_sensitivity) uses `l_star` from each entry under
    `camerer.algorithm` in ../results/applications.json, ordered by theta=1.5,
    2.0, 2.5, and 3.0. The `camerer_sensitivity_1e-10` entry records the check
    that replacing the six reported <.001 values by 1e-10 leaves those counts
    unchanged.


FAST RE-CREATION FROM THE BUNDLED REFERENCE RESULTS
===================================================

Run the following commands from code/. To recreate all distinct numbered figure
files and Table 2 formatting without rerunning the long Monte Carlo
calculations, use the released reference JSON in ../results:

  mkdir -p reproduced/from-reference/plots reproduced/from-reference/results

  python3 plot_prior_objective.py \
    --input ../results/prior_objective.json \
    --out-dir reproduced/from-reference/plots

  python3 plot_power_curves.py \
    --input ../results/all_experiments.json \
    --out-dir reproduced/from-reference/plots

  python3 -c 'import json; from run_scaling_convergence import plot_convergence_figure; d=json.load(open("../results/scaling_results.json", encoding="utf-8")); plot_convergence_figure(d, "reproduced/from-reference/plots/convergence_trajectory.pdf")'

  python3 -c 'import json; from run_scaling_convergence import plot_scaling_figure; d=json.load(open("../results/scaling_results.json", encoding="utf-8")); plot_scaling_figure(d, "reproduced/from-reference/plots/scaling_with_K.pdf")'

  python3 make_table_scaling.py \
    --input ../results/scaling_results.json \
    --out reproduced/from-reference/results/table_scaling.tex

Tables 1 and 3 can be checked immediately against
../results/prior_objective.json and ../results/applications.json using the
field and grouping rules above.


REPRODUCIBILITY NOTES
=====================

All paper-scale drivers are seeded. The verifier compares scientific values,
decisions, solver diagnostics, and embedded-data provenance at tolerances well
below the manuscript's reporting precision:

  python3 verify_reproduction.py reproduced/paper/results

Floating-point summation can vary slightly across platforms. Figure PDFs are
not expected to be byte-identical because renderer metadata can vary, but the
plotted data and styling are generated by the listed scripts. The bundled
results/environment.json records the released environment.

Application data and sources are documented at the top of
code/run_applications.py. BCG trial counts, provenance, and a checksum are in
code/general_k/data_bcg.py. No external data download is needed to run any
application.

The code is distributed under the MIT License; see code/LICENSE.
