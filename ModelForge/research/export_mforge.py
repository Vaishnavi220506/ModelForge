"""Export a PyTorch model to a ModelForge manifest (.mforge).

Author: Vaishnavi

The model is traced with torch.fx, so any network built from the supported
operators can be exported, whatever its Python structure (Sequential, custom
forward with residual connections, ...):

  nn.Conv2d (stride, symmetric padding, groups; depthwise when groups = C)
  nn.BatchNorm2d / nn.BatchNorm1d   (eval statistics; ModelForge folds them)
  nn.ReLU, torch.relu, F.relu
  nn.MaxPool2d, nn.AvgPool2d        (square window, no padding)
  nn.AdaptiveAvgPool2d(1)           (-> GlobalAveragePool)
  nn.Flatten, torch.flatten
  nn.Linear
  +, torch.add                      (residual connections)
  nn.Dropout, nn.Identity           (no-ops in eval mode)

Usage as a library:
  from export_mforge import export
  text = export(model.eval(), (1, 28, 28), "mnist_resnet", comment="...")
"""

import operator

import numpy as np
import torch
from torch import fx, nn
from torch.nn import functional as F


def _values(array):
    return ",".join(f"{v:.9g}" for v in np.asarray(array, dtype=np.float32).ravel())


def _pair(value):
    return value if isinstance(value, int) else value[0]


def _square(value, what):
    if isinstance(value, (tuple, list)):
        if len(set(value)) != 1:
            raise ValueError(f"{what} must be square / symmetric, got {value}")
        return value[0]
    return value


class _Writer:
    def __init__(self):
        self.tensors = []
        self.nodes = []
        self.count = 0

    def name(self, prefix):
        self.count += 1
        return f"{prefix}{self.count}"

    def tensor(self, prefix, array, shape=None):
        array = np.asarray(array, dtype=np.float32)
        name = self.name(prefix)
        dims = shape if shape is not None else array.shape
        self.tensors.append(f"tensor {name} float32 {','.join(str(d) for d in dims)} values={_values(array)}")
        return name

    def node(self, op, inputs, attributes=""):
        out = self.name("v")
        self.nodes.append(f"node n{self.count} {op} {' '.join(inputs)} -> {out}{(' ' + attributes) if attributes else ''}")
        return out


def export(model, input_shape, name, comment="", classes=None, output_activation="Softmax"):
    """Returns the .mforge text of `model` (in eval mode) for inputs of shape input_shape (no batch)."""
    model = model.eval()
    traced = fx.symbolic_trace(model)
    modules = dict(traced.named_modules())
    writer = _Writer()
    value = {}
    with torch.no_grad():
        example = torch.zeros(1, *input_shape)
        shapes = {}
        # Record every intermediate shape with a forward pass through the graph.
        env = {}
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = example
            elif node.op == "output":
                break
            else:
                args = fx.node.map_arg(node.args, lambda n: env[n.name])
                kwargs = fx.node.map_arg(node.kwargs, lambda n: env[n.name])
                if node.op == "call_module":
                    env[node.name] = modules[node.target](*args, **kwargs)
                elif node.op == "call_function":
                    env[node.name] = node.target(*args, **kwargs)
                elif node.op == "call_method":
                    env[node.name] = getattr(args[0], node.target)(*args[1:], **kwargs)
                elif node.op == "get_attr":
                    raise ValueError("parameters used directly in forward() are not supported")
            if isinstance(env.get(node.name), torch.Tensor):
                shapes[node.name] = tuple(env[node.name].shape)
    output = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            value[node.name] = "x"
            continue
        if node.op == "output":
            output = value[node.args[0].name]
            classes = classes or shapes[node.args[0].name][-1]
            break
        source = [value[a.name] for a in node.args if isinstance(a, fx.Node)]
        if node.op == "call_module":
            module = modules[node.target]
            if isinstance(module, (nn.Dropout, nn.Identity)):
                value[node.name] = source[0]
            elif isinstance(module, nn.Conv2d):
                if module.padding_mode != "zeros" or _square(module.dilation, "dilation") != 1:
                    raise ValueError("only zero padding and no dilation are supported")
                inputs = [source[0], writer.tensor("w", module.weight.detach().numpy())]
                if module.bias is not None:
                    inputs.append(writer.tensor("b", module.bias.detach().numpy()))
                value[node.name] = writer.node(
                    "Conv", inputs, f"strides={_square(module.stride, 'stride')} "
                    f"pads={_square(module.padding, 'padding')} group={module.groups}")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                c = module.num_features
                gamma = module.weight.detach().numpy() if module.affine else np.ones(c)
                beta = module.bias.detach().numpy() if module.affine else np.zeros(c)
                params = [writer.tensor("bn", a) for a in (gamma, beta, module.running_mean.numpy(),
                                                            module.running_var.numpy())]
                value[node.name] = writer.node("BatchNormalization", [source[0]] + params,
                                               f"epsilon={module.eps:.9g}")
            elif isinstance(module, nn.ReLU):
                value[node.name] = writer.node("Relu", source)
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                k = _square(module.kernel_size, "kernel")
                s = _square(module.stride if module.stride is not None else k, "stride")
                if _square(module.padding, "padding") != 0 or getattr(module, "ceil_mode", False):
                    raise ValueError("pooling must have no padding and floor mode")
                op = "MaxPool" if isinstance(module, nn.MaxPool2d) else "AveragePool"
                value[node.name] = writer.node(op, source, f"kernel={k} strides={s}")
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                if _square(module.output_size, "output size") != 1:
                    raise ValueError("only AdaptiveAvgPool2d(1) is supported")
                value[node.name] = writer.node("GlobalAveragePool", source)
            elif isinstance(module, nn.Flatten):
                value[node.name] = writer.node("Flatten", source)
            elif isinstance(module, nn.Linear):
                inputs = [source[0], writer.tensor("w", module.weight.detach().numpy().T)]
                if module.bias is not None:
                    b = module.bias.detach().numpy()
                    inputs.append(writer.tensor("b", b, (1, b.shape[0])))
                value[node.name] = writer.node("Gemm", inputs)
            else:
                raise ValueError(f"unsupported module {type(module).__name__}")
        elif node.op == "call_function":
            if node.target in (operator.add, torch.add):
                if len(source) != 2:
                    raise ValueError("only tensor + tensor additions are supported")
                value[node.name] = writer.node("Add", source)
            elif node.target in (torch.relu, F.relu):
                value[node.name] = writer.node("Relu", source)
            elif node.target is torch.flatten:
                value[node.name] = writer.node("Flatten", source)
            else:
                raise ValueError(f"unsupported function {node.target}")
        elif node.op == "call_method" and node.target in ("relu", "flatten"):
            value[node.name] = writer.node("Relu" if node.target == "relu" else "Flatten", source)
        else:
            raise ValueError(f"unsupported node {node.op} {node.target}")
    lines = [f"# {name}: {comment}".rstrip(), f"model {name}"]
    lines.append(f"input x float32 1,{','.join(str(d) for d in input_shape)}")
    lines.append(f"output y float32 1,{classes}")
    lines += writer.tensors
    lines += writer.nodes
    lines.append(f"node probabilities {output_activation} {output} -> y axis=1")
    return "\n".join(lines) + "\n"
