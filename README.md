# Most Powerful Tests with Exact Family-Wise Error Rate Control: Necessary Conditions and Fast Computation for General K

Code, data and reference results for the article by Prasanjit Dubey and Xiaoming Huo
(H. Milton Stewart School of Industrial and Systems Engineering, Georgia Institute of
Technology), accepted for publication in the *Journal of Computational and Graphical
Statistics*. The same files are supplied with the article as online supplementary material.

## Contents

| Path | What it holds |
|---|---|
| [`code/`](code/) | The `general_k` Python package (SPOT and the comparator procedures), experiment and plotting drivers, tests, pinned requirements, and the technical [`code/README.md`](code/README.md) |
| [`results/`](results/) | Read-only reference JSON behind the article's tables and figures, and the generated Table 2 source |
| [`plots/`](plots/) | Reference diagnostic plot supplied with the computational results |
| [`readme.txt`](readme.txt) | Reproduction guide: the command and JSON fields behind every numbered table and figure in the article and its supplement |

New runs are written below `code/reproduced/` by default and do not overwrite the bundled
reference results. Application inputs are embedded in the source with provenance, so no
data are downloaded at run time.

## Installation and a fast check

Python 3.11 or newer is required for the pinned numerical-reproduction environment. On
macOS or Linux, run the following from the repository root; Windows commands are in
[`code/README.md`](code/README.md).

```bash
cd code
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python -m general_k.environment --check-packages
python -m pytest tests/
python reproduce_paper.py --quick --output-dir reproduced/quick
```

Quick mode runs every main result driver at reduced Monte Carlo budgets. It checks the
installation and the workflow; it does not reproduce the paper-scale estimates. In this
repository pytest intentionally reports 15 skips: typography and presence checks on the
seven manuscript figure PDFs, which are not part of the repository.

## Reproducing the paper-scale results

From `code/`, inspect the command plan and then run it:

```bash
python3 reproduce_paper.py --full --jobs 6 --dry-run --output-dir reproduced/full-plan
python3 reproduce_paper.py --full --jobs 6 --output-dir reproduced/paper
```

The workflow is seeded, records its commands and environment, and finishes by comparing the
new results with the bundled reference results; `python3 verify_reproduction.py
reproduced/paper/results` repeats that comparison. A full run can take many hours.
[`code/README.md`](code/README.md) gives per-step runtime and memory estimates and explains
how to resume an interrupted run. [`readme.txt`](readme.txt) maps each numbered table and
figure to its command and shows how to re-create the figures from the bundled reference
results without rerunning the Monte Carlo calculations.

## Earlier code

The code linked from the preprint [arXiv:2604.10986v1](https://arxiv.org/abs/2604.10986v1)
is preserved at tag
[`arxiv-2604.10986v1`](https://github.com/pdubey96/scalable-optimal-mht-fwer/tree/arxiv-2604.10986v1).

## License

MIT; see [`LICENSE`](LICENSE).
