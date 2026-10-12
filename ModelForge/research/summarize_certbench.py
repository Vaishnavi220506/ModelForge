"""Summarise modelforge_certbench results into the paper's tables.

Author: Vaishnavi
Usage: python summarize_certbench.py certbench.json   (standard library only)
"""

import json
import sys
from collections import defaultdict


def main(path):
    data = json.load(open(path))
    cases = data["cases"]
    methods = [o["name"] for o in cases[0]["outcomes"]]
    agg = defaultdict(lambda: defaultdict(float))
    for case in cases:
        robust = case["robust"]
        int8 = next(o for o in case["outcomes"] if o["name"] == "uniform_int8")
        for o in case["outcomes"]:
            a = agg[o["name"]]
            a["cases"] += 1
            a["robust"] += robust
            a["proved"] += o["proved"]
            a["unproved"] += o["unproved"]
            a["cex_regions"] += o["counterexamples"]
            a["cases_with_cex"] += o["counterexamples"] > 0
            a["fully_certified"] += o["unproved"] == 0
            a["flash"] += o["flash"]
            a["flash_vs_fp32"] += o["flash"] / case["float32_bytes"]
            a["flash_vs_int8"] += o["flash"] / int8["flash"]
            a["test_flips"] += o["test_flips"]
            a["seconds"] += o["seconds"]
            a["analyses"] += o["analyses"]
            a["cases_cex_no_test_flip"] += o["counterexamples"] > 0 and o["test_flips"] == 0

    print(f"{len(cases)} cases (8 models x radii), {data['seconds']:.0f} s total\n")
    print(f"{'method':32}{'certified':>10}{'regions proved':>18}{'cex regions':>12}{'cases w/ cex':>13}"
          f"{'cex, 0 test flips':>18}{'flash/fp32':>11}{'flash/int8':>11}{'time s':>8}")
    for m in methods:
        a = agg[m]
        n = a["cases"]
        print(f"{m:32}{int(a['fully_certified']):>6}/{int(n):<3}{int(a['proved']):>11}/{int(a['robust']):<6}"
              f"{int(a['cex_regions']):>12}{int(a['cases_with_cex']):>13}{int(a['cases_cex_no_test_flip']):>18}"
              f"{a['flash_vs_fp32'] / n:>11.3f}{a['flash_vs_int8'] / n:>11.3f}{a['seconds']:>8.1f}")

    # Optimality against the exhaustive search (eps = 0.1).
    exact = gap = n = 0
    for case in cases:
        e = case["exhaustive"]
        if e["optimum_cost"] is None or e["optimum_cost"] < 0:
            continue
        ours = next(o for o in case["outcomes"] if o["name"] == "forgecert")
        n += 1
        exact += abs(ours["cost"] - e["optimum_cost"]) < 0.5
        gap += ours["cost"] / e["optimum_cost"] - 1
        print(f"  optimum {case['model']:24} lattice {e['lattice']:4}  optimum {e['optimum_cost']:7.0f}  "
              f"ForgeCert {ours['cost']:7.0f}  {e['config']}")
    if n:
        print(f"\nForgeCert reaches the exhaustive optimum in {exact}/{n} cases (mean gap {100 * gap / n:.1f}%)")

    # Accumulator widths: proved vs range-agnostic worst case.
    narrower = layers = int16 = int16_worst = 0
    for case in cases:
        ours = next(o for o in case["outcomes"] if o["name"] == "forgecert")
        for proved, worst in zip(ours["acc_bits"], ours["worst_case_acc_bits"]):
            layers += 1
            narrower += proved < worst
            int16 += proved == 16
            int16_worst += worst == 16
    print(f"accumulators: {layers} layers, narrower than range-agnostic in {narrower}, "
          f"int16 proved {int16} (range-agnostic {int16_worst})")

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
