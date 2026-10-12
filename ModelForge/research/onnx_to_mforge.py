"""Convert an ONNX model to a ModelForge manifest (.mforge).

Author: Vaishnavi

Any framework that exports ONNX (PyTorch torch.onnx.export, TensorFlow via
tf2onnx, Keras, scikit-learn via skl2onnx, ...) can then be compiled and
certified by ModelForge / ForgeCert. Supported ONNX operators (opset >= 9):

  Conv (strides, symmetric pads, group; no dilation), BatchNormalization,
  Relu, MaxPool / AveragePool (no padding, floor mode), GlobalAveragePool,
  Flatten, Reshape to [N, -1], Gemm, MatMul, Add, Softmax, Sigmoid,
  Identity, Dropout (inference), Constant (for Reshape shapes)

Usage: python onnx_to_mforge.py model.onnx model.mforge [--name NAME]
(needs the `onnx` Python package)
"""

import argparse
import sys

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:  # pragma: no cover
    sys.exit("onnx_to_mforge.py needs the onnx package (pip install onnx)")


def _values(array):
    return ",".join(f"{v:.9g}" for v in np.asarray(array, dtype=np.float32).ravel())


def _attributes(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _square(values, what):
    values = list(values)
    if len(set(values)) > 1:
        raise ValueError(f"{what} must be the same on every axis, got {values}")
    return values[0] if values else None


def _shape(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_value if d.dim_value > 0 else 1 for d in dims]  # symbolic batch -> 1


def convert(model, name=None):
    graph = model.graph
    constants = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:  # Constant nodes (e.g. Reshape targets) and Identity over initializers
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(_attributes(node)["value"])
        elif node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
    inputs = [i for i in graph.input if i.name not in constants]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("the model must have exactly one input and one output")
    rename = {}

    def ident(text):
        clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
        return clean if clean and not clean[0].isdigit() else "t_" + clean

    def value(text):
        return rename.get(text, ident(text))

    tensors, nodes, used = [], [], set()

    def tensor(text, array):
        key = ident(text)
        if key not in used:
            used.add(key)
            array = np.asarray(array, dtype=np.float32)
            dims = ",".join(str(d) for d in (array.shape or (1,)))
            tensors.append(f"tensor {key} float32 {dims} values={_values(array)}")
        return key

    for index, node in enumerate(graph.node):
        op, a = node.op_type, _attributes(node)
        out = ident(node.output[0])
        args = list(node.input)
        label = ident(node.name or f"{op.lower()}_{index}")
        if op == "Constant":
            continue
        if op in ("Identity", "Dropout"):
            if node.output[0] not in constants:
                rename[node.output[0]] = value(args[0])
            continue
        if op == "Conv":
            if any(d != 1 for d in a.get("dilations", [1, 1])):
                raise ValueError("dilated convolutions are not supported")
            pads = a.get("pads", [0, 0, 0, 0])
            operands = [value(args[0]), tensor(args[1], constants[args[1]])]
            if len(args) > 2 and args[2]:
                operands.append(tensor(args[2], constants[args[2]]))
            nodes.append(f"node {label} Conv {' '.join(operands)} -> {out} "
                         f"strides={_square(a.get('strides', [1, 1]), 'strides')} pads={_square(pads, 'pads')} "
                         f"group={a.get('group', 1)}")
        elif op == "BatchNormalization":
            operands = [value(args[0])] + [tensor(t, constants[t]) for t in args[1:5]]
            nodes.append(f"node {label} BatchNormalization {' '.join(operands)} -> {out} "
                         f"epsilon={a.get('epsilon', 1e-5):.9g}")
        elif op in ("MaxPool", "AveragePool"):
            if any(p != 0 for p in a.get("pads", [0])) or a.get("ceil_mode", 0):
                raise ValueError(f"{op} must have no padding and floor mode")
            k = _square(a["kernel_shape"], "kernel_shape")
            s = _square(a.get("strides", [k, k]), "strides")
            nodes.append(f"node {label} {op} {value(args[0])} -> {out} kernel={k} strides={s}")
        elif op in ("Relu", "GlobalAveragePool", "Softmax", "Sigmoid"):
            extra = " axis=1" if op == "Softmax" else ""
            nodes.append(f"node {label} {op} {value(args[0])} -> {out}{extra}")
        elif op == "Flatten":
            if a.get("axis", 1) != 1:
                raise ValueError("Flatten must use axis=1")
            nodes.append(f"node {label} Flatten {value(args[0])} -> {out}")
        elif op == "Reshape":
            target = constants.get(args[1])
            if target is None or len(target) != 2 or not (target[0] in (0, 1, -1) or target[1] == -1):
                raise ValueError("Reshape is supported only as a flatten to [N, -1]")
            nodes.append(f"node {label} Flatten {value(args[0])} -> {out}")
        elif op == "Gemm":
            if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                raise ValueError("Gemm must have transA=0, alpha=beta=1")
            w = constants[args[1]]
            if a.get("transB", 0):
                w = w.T  # ModelForge Gemm weights are [in, out]
            operands = [value(args[0]), tensor(args[1] + "_kn", w)]
            if len(args) > 2 and args[2]:
                operands.append(tensor(args[2], constants[args[2]].reshape(1, -1)))
            nodes.append(f"node {label} Gemm {' '.join(operands)} -> {out}")
        elif op == "MatMul":
            nodes.append(f"node {label} MatMul {value(args[0])} {tensor(args[1], constants[args[1]])} -> {out}")
        elif op == "Add":
            operands = [tensor(t, constants[t].reshape(1, -1) if constants[t].ndim == 1 else constants[t])
                        if t in constants else value(t) for t in args]
            nodes.append(f"node {label} Add {' '.join(operands)} -> {out}")
        else:
            raise ValueError(f"unsupported ONNX operator {op}")

    source = inputs[0]
    in_shape = _shape(source)
    out_shape = _shape(graph.output[0])
    title = name or graph.name or "onnx_model"
    lines = [f"# {title}: converted from ONNX by research/onnx_to_mforge.py (author: Vaishnavi)",
             f"model {ident(title)}",
             f"input {ident(source.name)} float32 {','.join(str(d) for d in in_shape)}",
             f"output {value(graph.output[0].name)} float32 {','.join(str(d) for d in out_shape)}"]
    return "\n".join(lines + tensors + nodes) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model")
    parser.add_argument("output")
    parser.add_argument("--name")
    args = parser.parse_args()
    model = onnx.load(args.onnx_model)
    with open(args.output, "w") as handle:
        handle.write(convert(model, args.name))


if __name__ == "__main__":
    main()
