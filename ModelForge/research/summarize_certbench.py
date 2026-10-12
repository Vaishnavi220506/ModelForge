"""Summarise modelforge_certbench results into the paper's tables.

Author: Vaishnavi
Usage: python summarize_certbench.py certbench.json   (standard library only)
"""

import json
import sys
from collections import defaultdict

IMAGE = ("mnist", "fashion", "cifar")
ZOO = ("lenet", "dscnn", "resnet")


def suite_of(case):
    if any(arch in case["model"] for arch in ZOO):
        return "zoo"
    return "image" if case["model"].startswith(IMAGE) else "tabular"


def table(cases, title):
    if not cases:
        return
    names = []
    for case in cases:
        for o in case["outcomes"]:
            if o["name"] not in names:
                names.append(o["name"])
    agg = defaultdict(lambda: defaultdict(float))
    for case in cases:
        outcomes = {o["name"]: o for o in case["outcomes"]}
        int8 = outcomes["uniform_int8"]
        for o in case["outcomes"]:
            a = agg[o["name"]]
            a["cases"] += 1
            a["robust"] += case["robust"]
            a["proved"] += o["proved"]
            a["cex_regions"] += o["counterexamples"]
            a["cases_with_cex"] += o["counterexamples"] > 0
            a["silent"] += o["counterexamples"] > 0 and o["test_flips"] == 0
            a["certified"] += o["unproved"] == 0
            a["flash_vs_fp32"] += o["flash"] / case["float32_bytes"]
            a["flash_vs_int8"] += o["flash"] / int8["flash"]
            a["cycles_vs_int8"] += o.get("cycles", 0) / max(1.0, int8.get("cycles", 1))
            a["seconds"] += o["seconds"]
            a["analyses"] += o["analyses"]
    print(f"\n== {title}: {len(cases)} cases, {int(agg['uniform_int8']['robust'])} regions on which f is constant")
    print(f"{'method':34}{'certified':>10}{'proved':>13}{'cex':>6}{'cases':>6}{'silent':>7}"
          f"{'flash/fp32':>11}{'flash/int8':>11}{'cyc/int8':>9}{'time s':>9}")
    for name in names:
        a = agg[name]
        n = a["cases"]
        print(f"{name:34}{int(a['certified']):>6}/{int(n):<3}{int(a['proved']):>7}/{int(a['robust']):<5}"
              f"{int(a['cex_regions']):>6}{int(a['cases_with_cex']):>6}{int(a['silent']):>7}"
              f"{a['flash_vs_fp32'] / n:>11.3f}{a['flash_vs_int8'] / n:>11.3f}{a['cycles_vs_int8'] / n:>9.3f}"
              f"{a['seconds']:>9.1f}")


def paired(cases, a, b, key, only_certified=True):
    smaller = larger = tie = 0
    ratio = 0.0
    for case in cases:
        o = {x["name"]: x for x in case["outcomes"]}
        if a not in o or b not in o:
            continue
        if only_certified and (o[a]["unproved"] or o[b]["unproved"]):
            continue
        x, y = o[a][key], o[b][key]
        smaller += x < y * (1 - 1e-9)
        larger += x > y * (1 + 1e-9)
        tie += abs(x - y) <= y * 1e-9
        ratio += x / y if y else 1.0
    n = smaller + larger + tie
    if n:
        print(f"  {a} vs {b} on {key}: smaller {smaller}, larger {larger}, tie {tie}, mean ratio {ratio / n:.3f} (n={n})")


def main(path):
    data = json.load(open(path))
    cases = data["cases"]
    print(f"{len(cases)} cases, {data['seconds']:.0f} s total")
    for suite in ("tabular", "image", "zoo"):
        table([c for c in cases if suite_of(c) == suite], suite)
    print("\nPaired comparisons (cases where both certify every region):")
    for a, b, key in [("forgecert", "uniform_int16", "cost"), ("forgecert", "accuracy_guided", "cost"),
                      ("forgecert_latency", "forgecert", "cycles"), ("forgecert_latency", "uniform_int8", "cycles"),
                      ("forgecert_latency", "uniform_int8_calibrated", "cycles"),
                      ("forgecert_calibrated", "forgecert", "cost"),
                      ("ablation_per_tensor_scales", "forgecert", "cost"),
                      ("forgecert", "ablation_no_modular_accumulation", "cycles"),
                      ("bits_cost_model", "bits_uniform_refine", "ideal_bytes"),
                      ("bits_cost_model", "bits_random_refine_s1", "ideal_bytes"),
                      ("bits_cost_model", "bits_random_refine_s2", "ideal_bytes"),
                      ("bits_cost_model", "bits_random_refine_s3", "ideal_bytes")]:
        paired(cases, a, b, key)
    exact = gap = n = 0
    for case in cases:
        e = case["exhaustive"]
        if e["optimum_cost"] is None or e["optimum_cost"] < 0:
            continue
        ours = next(o for o in case["outcomes"] if o["name"] == "forgecert")
        n += 1
        exact += abs(ours["cost"] - e["optimum_cost"]) < 0.5
        gap += ours["cost"] / e["optimum_cost"] - 1
    if n:
        print(f"\nForgeCert reaches the exhaustive optimum in {exact}/{n} cases (mean gap {100 * gap / n:.1f}%)")
    audits = [c["audit"] for c in cases]
    print(f"soundness audit: {sum(a['samples'] for a in audits)} samples, "
          f"{sum(a['violations'] for a in audits)} violations, max observed/proved error "
          f"{max(a['max_observed_over_proved'] for a in audits):.3f}")
    checks = [c["checker"] for c in cases if c["checker"]["seconds"] >= 0]
    if checks:
        print(f"independent checker: {sum(c['valid'] for c in checks)}/{len(checks)} valid, "
              f"max {max(c['seconds'] for c in checks):.3f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "certbench.json")
