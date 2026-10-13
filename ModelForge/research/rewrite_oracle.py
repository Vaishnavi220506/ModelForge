"""Independent re-check of the rewrite study's witnesses with ONNX Runtime.

Author: Vaishnavi

modelforge_rewritebench writes, for every rewrite it found broken, the
rewritten graph (<model>_<k>_candidate.mforge), the original model
(<model>_original.mforge) and the witness input (<model>_<k>_input.csv).
This script converts both graphs to ONNX with its own converter, runs them in
ONNX Runtime (an implementation that shares no code with ModelForge), and
confirms that the logits differ by more than the study's tolerance on the
witness. ONNX Runtime runs with graph optimisations disabled, so it provides
plain reference semantics for both graphs.

Usage: python rewrite_oracle.py <rewritebench out dir> [--tau 1e-3]
(needs numpy, onnx, onnxruntime)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def parse(path):
    model = {"tensors": {}, "nodes": []}
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "model":
            model["name"] = t[1]
        elif t[0] in ("input", "output"):
            model[t[0]] = (t[1], [int(d) for d in t[3].split(",")])
        elif t[0] == "tensor":
            dims = [int(d) for d in t[3].split(",")]
            values = np.array([float(v) for v in t[4].split("=", 1)[1].split(",")], dtype=np.float32)
            model["tensors"][t[1]] = values.reshape(dims)
        elif t[0] == "node":
            arrow = t.index("->")
            attrs = dict(a.split("=", 1) for a in t[arrow + 2:])
            model["nodes"].append({"name": t[1], "op": t[2], "inputs": t[3:arrow], "output": t[arrow + 1],
                                   "attrs": attrs})
    return model


def logits_value(model):
    out = model["output"][0]
    for node in model["nodes"]:
        if node["output"] == out and node["op"] in ("Softmax", "Sigmoid"):
            return node["inputs"][0]
    return out


def to_onnx(model):
    nodes, inits, extra = [], [], 0

    def const(name, array):
        inits.append(numpy_helper.from_array(np.asarray(array, dtype=np.float32), name))

    for name, value in model["tensors"].items():
        const(name, value)
    for n in model["nodes"]:
        op, a, ins, out = n["op"], n["attrs"], n["inputs"], n["output"]
        if op == "Conv":
            p = int(a.get("pads", 0))
            src = ins[0]
            pads = [p, p, p, p]
            if a.get("padmode") == "edge" and p > 0:
                extra += 1
                padded, pv = f"__pad{extra}", f"__padv{extra}"
                inits.append(numpy_helper.from_array(np.array([0, 0, p, p, 0, 0, p, p], dtype=np.int64), pv))
                nodes.append(helper.make_node("Pad", [src, pv], [padded], mode="edge"))
                src, pads = padded, [0, 0, 0, 0]
            nodes.append(helper.make_node("Conv", [src] + ins[1:], [out], strides=[int(a.get("strides", 1))] * 2,
                                          pads=pads, group=int(a.get("group", 1))))
        elif op == "Gemm":
            nodes.append(helper.make_node("Gemm", ins, [out], transB=int(a.get("transB", 0))))
        elif op in ("AveragePool", "MaxPool"):
            k, s = int(a["kernel"]), int(a.get("strides", 1))
            nodes.append(helper.make_node(op, ins, [out], kernel_shape=[k, k], strides=[s, s]))
        elif op == "BatchNormalization":
            nodes.append(helper.make_node(op, ins, [out], epsilon=float(a.get("epsilon", 1e-5))))
        elif op == "Softmax":
            nodes.append(helper.make_node(op, ins, [out], axis=int(a.get("axis", 1))))
        elif op == "Flatten":
            nodes.append(helper.make_node(op, ins, [out], axis=1))
        elif op == "ReluClamp":
            extra += 1
            lo, hi = f"__lo{extra}", f"__hi{extra}"
            const(lo, np.float32(0.0))
            const(hi, np.float32(float(a["parameter"])))
            nodes.append(helper.make_node("Clip", [ins[0], lo, hi], [out]))
        elif op == "ReluDeadZone":
            # x if x >= p, 0 otherwise (for x > 0; ReLU elsewhere)
            extra += 1
            r, th, ge, mask = f"__r{extra}", f"__th{extra}", f"__ge{extra}", f"__m{extra}"
            const(th, np.float32(float(a["parameter"])))
            nodes += [helper.make_node("Relu", [ins[0]], [r]),
                      helper.make_node("GreaterOrEqual", [ins[0], th], [ge]),
                      helper.make_node("Cast", [ge], [mask], to=TensorProto.FLOAT),
                      helper.make_node("Mul", [r, mask], [out])]
        else:  # MatMul, Add, Relu, Sigmoid, GlobalAveragePool
            nodes.append(helper.make_node(op, ins, [out]))
    name, shape = model["input"]
    target = logits_value(model)
    graph = helper.make_graph(nodes, model.get("name", "m"),
                              [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info(target, TensorProto.FLOAT, None)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def run(model, x):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # plain reference semantics
    session = ort.InferenceSession(to_onnx(model).SerializeToString(), options, providers=["CPUExecutionProvider"])
    name, shape = model["input"]
    return session.run(None, {name: x.reshape(shape).astype(np.float32)})[0].ravel().astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--tau", type=float, default=1e-3)
    args = parser.parse_args()
    root = Path(args.dir)
    rows = list(csv.DictReader(open(root / "cases.csv")))
    confirmed = refuted = 0
    report = []
    for row in rows:
        base = row["witness"]
        if not base:
            continue
        w = root / "witnesses"
        original = parse(w / f"{row['model']}_original.mforge")
        candidate = parse(w / f"{base}_candidate.mforge")
        x = np.array([float(v) for v in (w / f"{base}_input.csv").read_text().split(",")], dtype=np.float64)
        a, b = run(original, x), run(candidate, x)
        if a.size == 1:  # single sigmoid logit: compare against the implicit 0 logit too
            a, b = np.array([0.0, a[0]]), np.array([0.0, b[0]])
        divergence = float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(a)))))
        ok = divergence > args.tau
        confirmed += ok
        refuted += not ok
        report.append({"model": row["model"], "pass": row["pass"], "fault": row["fault"], "control": row["control"],
                       "ort_divergence": divergence, "confirmed": ok,
                       "decision_changed": int(np.argmax(a) != np.argmax(b))})
    (root / "ort_check.json").write_text(json.dumps({"confirmed": confirmed, "refuted": refuted,
                                                     "cases": report}, indent=1))
    print(f"ONNX Runtime confirms {confirmed} of {confirmed + refuted} witnesses (tau {args.tau})")
    sys.exit(0 if refuted == 0 else 1)


if __name__ == "__main__":
    main()
