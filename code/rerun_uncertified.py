"""
Re-run only the grid settings that failed to certify, and patch them in.

Safe because a setting that certifies is untouched by everything this changes.
The residual criterion is tested before the iterate-change criterion, so
tightening algo_tol cannot move a certified run (pinned by
tests/test_bisection_tolerance.py::TestGridStoppingToleranceIsInert and verified
directly against the stored results); and the escalation ladder in
run_all_experiments.SOLVE_STAGES only ever runs past stage 1 for a setting that
did not certify there, so raising the sweep cap and refining the bisection
cannot reach one either.

A full `python run_all_experiments.py` therefore reproduces the patched file
exactly; this script just avoids spending ~6 hours re-deriving the settings that
cannot change.

Usage:
    python rerun_uncertified.py [--dry-run] [--jobs N]

The bundled grid is read-only by default.  Any patched grid is written under
``reproduced/results`` unless ``--out`` explicitly selects another file.
"""

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from general_k import paths
from general_k.io import atomic_json_dump
from run_all_experiments import run_single

REFERENCE_RESULTS = paths.result("all_experiments.json")


def uncertified(data):
    out = []
    for K, by_model in data.items():
        for model, by_theta in by_model.items():
            for theta, rec in by_theta.items():
                if not rec["diagnostics"]["converged"]:
                    out.append((K, model, theta, rec))
    return out


def _report(i, n, K, model, theta, old, new, rt, secs):
    nd = new["diagnostics"]
    trail = " -> ".join(
        f"s{e['stage']}:{e['reason']}/{e['final_residual']/rt:.2f}x"
        for e in nd.get("escalation", [])
    )
    print(f"[{i:2d}/{n}] K={K} {model} theta={theta}: "
          f"{old['reason']}/{old['final_residual']:.2e} "
          f"({old['final_residual']/rt:.2f}x) -> "
          f"{nd['reason']}/{nd['final_residual']:.2e} "
          f"({nd['final_residual']/rt:.2f}x) "
          f"sweeps {old['sweeps']}->{nd['sweeps']} "
          f"stage {nd.get('stage')} f_tol={nd.get('f_tol')} "
          f"certified={nd['converged']}  {secs:.0f}s"
          + (f"  [{trail}]" if trail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="List uncertified settings without running or writing")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Settings to re-solve concurrently (default: 1)")
    ap.add_argument("--input", default=REFERENCE_RESULTS,
                    help="Grid JSON to inspect (default: bundled reference grid)")
    ap.add_argument("--out", default=paths.reproduced_result("all_experiments.json"),
                    help="Patched JSON (default: reproduced/results/all_experiments.json)")
    args = ap.parse_args()

    if args.jobs < 1:
        ap.error("--jobs must be at least 1")

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    todo = uncertified(data)
    if not todo:
        print("every setting certified; nothing to re-run", flush=True)
        return
    rt = todo[0][3]["diagnostics"]["res_tol"]
    print(f"{len(todo)} settings to re-run; res_tol = {rt:.4e}", flush=True)
    for K, model, theta, rec in todo:
        d = rec["diagnostics"]
        print(f"   K={K} {model} theta={theta}: {d['reason']} "
              f"{d['final_residual']:.3e} ({d['final_residual']/rt:.2f}x)", flush=True)
    if args.dry_run:
        return

    patched = copy.deepcopy(data)
    t0 = time.time()
    fixed = 0

    if args.jobs <= 1:
        for i, (K, model, theta, rec) in enumerate(todo, 1):
            t1 = time.time()
            new = run_single(int(K), model, float(theta))
            fixed += bool(new["diagnostics"]["converged"])
            _report(i, len(todo), K, model, theta, rec["diagnostics"], new, rt,
                    time.time() - t1)
            patched[K][model][theta] = new
            atomic_json_dump(patched, args.out, indent=1)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_single, int(K), model, float(theta)):
                    (K, model, theta, rec)
                for (K, model, theta, rec) in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                K, model, theta, rec = futures[fut]
                new = fut.result()
                fixed += bool(new["diagnostics"]["converged"])
                _report(i, len(todo), K, model, theta, rec["diagnostics"], new, rt,
                        new["elapsed_seconds"])
                patched[K][model][theta] = new
                atomic_json_dump(patched, args.out, indent=1)

    total = sum(1 for K, bm in patched.items() for m, bt in bm.items() for t in bt)
    still = len(uncertified(patched))
    print(f"\ncertified {fixed}/{len(todo)} of the re-runs", flush=True)
    print(f"grid now {total - still}/{total} certified "
          f"(was {total - len(todo)}/{total})", flush=True)
    print(f"total {(time.time()-t0)/3600:.2f} h", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
