"""Statistics for the rewrite-validation study (modelforge_rewritebench).

Author: Vaishnavi
Usage: python analyze_rewrites.py <rewritebench out dir> [--out summary.json]
(needs scipy)

Definitions
  observable  some run of some method found an input on which the rewritten
              graph's logits differ from the original model's by more than the
              tolerance (and ONNX Runtime confirms the witness, ort_check.json)
  loud        observable, and the very first probe of both random and data
              testing detects it in every seed
  subtle      observable but not loud: here the choice of method matters
  control     a correct rewrite (must never be flagged)

Reported: detection rate vs budget (model evaluations) per method, per fault
family; time to first detection; paired comparisons of Guardian-APC+ with
every other method at the case-seed level (exact McNemar) and at the model
level (Wilcoxon signed-rank over per-model detection rates, the unit of
replication being the model); 95% bootstrap intervals over models.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from scipy import stats

METHODS = ["random", "data", "data_noise", "tensorfuzz", "deepxplore", "diffchaser",
           "guardian_apc", "guardian_rw", "guardian_nm", "guardian_apc_plus"]
BUDGETS = [8, 16, 32, 64, 128, 256, 512]
OURS = "guardian_apc_plus"


def family(fault):
    return fault.split("@")[0]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(args.dir)
    rows = list(csv.DictReader(open(root / "runs.csv")))
    ort = {}
    if (root / "ort_check.json").exists():
        for c in json.load(open(root / "ort_check.json"))["cases"]:
            ort[(c["model"], c["pass"], c["fault"])] = c["confirmed"]
    cases = defaultdict(list)
    for r in rows:
        cases[(r["model"], r["pass"], r["fault"], r["source"], r["control"] == "1")].append(r)

    summary = {"controls": 0, "control_runs": 0, "false_positives": 0, "loud": 0, "subtle": 0, "unobserved": 0,
               "ort_confirmed": sum(ort.values()), "ort_checked": len(ort)}
    subtle = {}
    families = defaultdict(lambda: defaultdict(int))
    for key, runs in cases.items():
        model, pas, fault, source, control = key
        if control:
            summary["controls"] += 1
            summary["control_runs"] += len(runs)
            summary["false_positives"] += sum(int(r["first"]) > 0 for r in runs)
            continue
        if max(int(r["first"]) for r in runs) == 0:
            summary["unobserved"] += 1
            families[family(fault)]["unobserved"] += 1
            continue
        first = [int(r["first"]) for r in runs if r["method"] in ("random", "data")]
        if all(f == 2 for f in first):
            summary["loud"] += 1
            families[family(fault)]["loud"] += 1
            continue
        summary["subtle"] += 1
        families[family(fault)]["subtle"] += 1
        subtle[key] = runs
    summary["families"] = {f: dict(v) for f, v in families.items()}

    def detected(run, budget):
        return 0 < int(run["first"]) <= budget

    # Detection rate vs budget on the subtle faults.
    curves = {m: [] for m in METHODS}
    for b in BUDGETS:
        for m in METHODS:
            hits = [detected(r, b) for runs in subtle.values() for r in runs if r["method"] == m]
            curves[m].append(sum(hits) / max(1, len(hits)))
    summary["budgets"] = BUDGETS
    summary["curves"] = curves

    # Per family at budgets 64 and 512.
    per_family = defaultdict(dict)
    for key, runs in subtle.items():
        pass
    fam_rates = defaultdict(lambda: defaultdict(list))
    for key, runs in subtle.items():
        for r in runs:
            fam_rates[family(key[2])][r["method"]].append(r)
    for f, by in fam_rates.items():
        per_family[f] = {m: {str(b): sum(detected(r, b) for r in rs) / len(rs) for b in (64, 512)}
                         for m, rs in by.items()}
        per_family[f]["cases"] = len({(r["model"], r["fault"]) for rs in by.values() for r in rs})
    summary["per_family"] = per_family

    # Mean evaluations to first detection (censored at the budget) on subtle faults.
    budget = int(rows[0]["budget"])
    summary["mean_first"] = {m: sum((int(r["first"]) or budget) for runs in subtle.values() for r in runs
                                    if r["method"] == m) / max(1, sum(r["method"] == m for runs in subtle.values()
                                                                      for r in runs))
                             for m in METHODS}
    summary["mean_seconds"] = {m: sum(float(r["seconds"]) for r in rows if r["method"] == m) /
                               max(1, sum(r["method"] == m for r in rows)) for m in METHODS}

    # Paired tests: Guardian-APC+ vs every method at budgets 64 and 512.
    tests = {}
    models = sorted({k[0] for k in subtle})
    for b in (64, 512):
        for m in METHODS:
            if m == OURS:
                continue
            only_ours = only_other = 0
            per_model = defaultdict(lambda: [0, 0, 0])
            for key, runs in subtle.items():
                by_seed = defaultdict(dict)
                for r in runs:
                    by_seed[r["seed"]][r["method"]] = detected(r, b)
                for seed, d in by_seed.items():
                    if OURS in d and m in d:
                        only_ours += d[OURS] and not d[m]
                        only_other += d[m] and not d[OURS]
                        pm = per_model[key[0]]
                        pm[0] += d[OURS]
                        pm[1] += d[m]
                        pm[2] += 1
            diffs = [(v[0] - v[1]) / v[2] for v in per_model.values() if v[2]]
            nonzero = [d for d in diffs if d != 0]
            wilcoxon_p = stats.wilcoxon(nonzero).pvalue if len(nonzero) >= 1 else 1.0
            rng = random.Random(7)
            boots = []
            for _ in range(2000):
                sample = [rng.choice(diffs) for _ in diffs]
                boots.append(sum(sample) / len(sample))
            boots.sort()
            tests[f"{m}@{b}"] = {
                "only_ours": only_ours, "only_other": only_other, "mcnemar_p": mcnemar(only_ours, only_other),
                "models": len(diffs), "mean_model_diff": sum(diffs) / max(1, len(diffs)),
                "ci95": [boots[50], boots[1949]], "wilcoxon_p": float(wilcoxon_p),
                "models_better": sum(d > 0 for d in diffs), "models_worse": sum(d < 0 for d in diffs)}
    summary["tests"] = tests
    summary["subtle_models"] = models

    text = [f"controls {summary['controls']} ({summary['control_runs']} runs): {summary['false_positives']} false positives",
            f"faults: {summary['loud']} loud, {summary['subtle']} subtle, {summary['unobserved']} not observed",
            f"ONNX Runtime confirms {summary['ort_confirmed']} of {summary['ort_checked']} witnesses",
            "", "detection rate on subtle faults vs budget (model evaluations)",
            "method              " + "".join(f"{b:>7}" for b in BUDGETS) + "  mean first   ms/run"]
    for m in METHODS:
        text.append(f"{m:20}" + "".join(f"{v:7.3f}" for v in curves[m]) +
                    f"  {summary['mean_first'][m]:10.1f} {1000 * summary['mean_seconds'][m]:8.1f}")
    text.append("")
    text.append("per fault family (subtle cases): detection at 64 / 512")
    for f, by in per_family.items():
        text.append(f"  {f} ({by['cases']} cases): " + ", ".join(
            f"{m}={by[m]['64']:.2f}/{by[m]['512']:.2f}" for m in METHODS if m in by))
    text.append("")
    text.append(f"{OURS} vs other methods (McNemar over case-seeds; Wilcoxon over {len(models)} models)")
    for k, t in tests.items():
        text.append(f"  {k:24} ours-only {t['only_ours']:4} other-only {t['only_other']:4} p={t['mcnemar_p']:.2g} | "
                    f"models +{t['models_better']}/-{t['models_worse']} mean diff {t['mean_model_diff']:+.3f} "
                    f"CI [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] p={t['wilcoxon_p']:.2g}")
    print("\n".join(text))
    out = Path(args.out) if args.out else root / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    (out.with_suffix(".txt")).write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
