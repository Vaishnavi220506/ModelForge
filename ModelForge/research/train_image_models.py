"""Train MNIST / Fashion-MNIST classifiers and export them as ModelForge manifests.

Author: Vaishnavi

The models are sized for an 8-bit microcontroller (Arduino Mega, ATmega2560:
256 KB flash, 8 KB RAM), so the certified integer code can run on it:

  *_mlp   Flatten -> Dense 784x32 -> ReLU -> Dense 32x10
  *_cnn   Conv 1->4 (3x3, stride 2) -> ReLU -> Conv 4->8 (3x3, stride 2) -> ReLU
          -> Flatten -> Dense 288x10

Pixels are scaled to [0, 1]; the input domain of every certificate is [0, 1]^784.

Outputs (committed, so the C++ tools never need Python):
  review_2_core_implementation/models/image/<dataset>_<arch>.mforge
  review_2_core_implementation/models/image/<dataset>_test.csv   (first 200 test images, label last)
  review_2_core_implementation/models/image/<dataset>_calib.csv  (500 training images: calibration/domain)
  research/results/image_models.json                              (test accuracy on all 10,000)

  cifar_cnn  Conv 3->16 (3x3, stride 2, pad 1) -> ReLU -> Conv 16->32 (3x3, stride 2, pad 1)
             -> ReLU -> Flatten -> Dense 2048x10            (CIFAR-10, 32x32 RGB)

Usage: python train_image_models.py <dir with the idx .gz files> [<dir with CIFAR-10 parquet files>]
(needs torch, numpy; CIFAR also pandas + pillow)
Download: MNIST from https://storage.googleapis.com/cvdf-datasets/mnist/ and
Fashion-MNIST from https://github.com/zalandoresearch/fashion-mnist (data/fashion).
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review_2_core_implementation" / "models" / "image"
RESULTS = ROOT / "research" / "results" / "image_models.json"
TEST_ROWS = 200
CALIBRATION_ROWS = 500


def load_idx(path):
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    if data[2] == 8 and data[3] == 3:  # images
        count, rows, cols = (int.from_bytes(data[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(data, np.uint8, offset=16).reshape(count, rows * cols).astype(np.float32) / 255.0
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)


def load(directory, prefix):
    d = Path(directory)
    return (load_idx(d / f"{prefix}train-images-idx3-ubyte.gz"), load_idx(d / f"{prefix}train-labels-idx1-ubyte.gz"),
            load_idx(d / f"{prefix}t10k-images-idx3-ubyte.gz"), load_idx(d / f"{prefix}t10k-labels-idx1-ubyte.gz"))


def load_cifar(directory):
    import io

    import pandas as pd
    from PIL import Image

    def read(name):
        frame = pd.read_parquet(Path(directory) / name)
        images = np.stack([np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"), dtype=np.float32)
                           for item in frame["img"]]) / 255.0
        # NHWC -> NCHW, flattened in C, H, W order (ModelForge's layout)
        return images.transpose(0, 3, 1, 2).reshape(len(frame), -1), frame["label"].to_numpy().astype(np.int64)
    x_train, y_train = read("train-00000-of-00001.parquet")
    x_test, y_test = read("test-00000-of-00001.parquet")
    return x_train, y_train, x_test, y_test


def cifar_cnn():
    return nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(2048, 10))


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))


def cnn():
    return nn.Sequential(nn.Conv2d(1, 4, 3, stride=2), nn.ReLU(), nn.Conv2d(4, 8, 3, stride=2), nn.ReLU(),
                         nn.Flatten(), nn.Linear(288, 10))


def train(model, x, y, epochs, seed, shape=(1, 28, 28)):
    torch.manual_seed(seed)
    xs = torch.tensor(x).reshape(-1, *shape)
    ys = torch.tensor(y)
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        order = torch.randperm(len(xs))
        for start in range(0, len(xs), 128):
            batch = order[start:start + 128]
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(model(xs[batch]), ys[batch])
            loss.backward()
            optimiser.step()
    return model


def values(array):
    return ",".join(f"{v:.9g}" for v in np.asarray(array, dtype=np.float32).ravel())


def manifest(name, model, arch, shape=(1, 28, 28)):
    lines = [f"# {name}: trained with PyTorch on {shape[1]}x{shape[2]} images, pixels in [0, 1] (author: Vaishnavi).",
             f"model {name}"]
    layers = [m for m in model if isinstance(m, (nn.Linear, nn.Conv2d))]
    if arch == "cnn":
        lines.append(f"input x float32 1,{shape[0]},{shape[1]},{shape[2]}")
    else:
        lines.append("input x float32 1,784")
    lines.append("output y float32 1,10")
    for i, layer in enumerate(layers):
        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        if isinstance(layer, nn.Linear):
            w = w.T  # ModelForge Gemm weights are [in, out]
            lines.append(f"tensor w{i} float32 {w.shape[0]},{w.shape[1]} values={values(w)}")
            lines.append(f"tensor b{i} float32 1,{b.shape[0]} values={values(b)}")
        else:
            lines.append(f"tensor w{i} float32 {','.join(str(d) for d in w.shape)} values={values(w)}")
            lines.append(f"tensor b{i} float32 {b.shape[0]} values={values(b)}")
    previous = "x"
    for i, layer in enumerate(layers):
        last = i == len(layers) - 1
        if isinstance(layer, nn.Conv2d):
            lines.append(f"node conv_{i} Conv {previous} w{i} b{i} -> z{i} strides={layer.stride[0]} "
                         f"pads={layer.padding[0]}")
            lines.append(f"node relu_{i} Relu z{i} -> h{i}")
            previous = f"h{i}"
            if not isinstance(layers[i + 1], nn.Conv2d):
                lines.append(f"node flatten Flatten {previous} -> flat")
                previous = "flat"
        else:
            out = "logits" if last else f"z{i}"
            lines.append(f"node dense_{i} Gemm {previous} w{i} b{i} -> {out}")
            if not last:
                lines.append(f"node relu_{i} Relu z{i} -> h{i}")
                previous = f"h{i}"
    lines.append("node probabilities Softmax logits -> y axis=1")
    return "\n".join(lines) + "\n"


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"author": "Vaishnavi", "models": {}}
    for dataset, prefix in (("mnist", ""), ("fashion", "fashion-")):
        x_train, y_train, x_test, y_test = load(directory, prefix)
        with open(OUT / f"{dataset}_test.csv", "w") as handle:
            for row, label in zip(x_test[:TEST_ROWS], y_test[:TEST_ROWS]):
                handle.write(values(row) + f",{int(label)}\n")
        with open(OUT / f"{dataset}_calib.csv", "w") as handle:
            for row, label in zip(x_train[:CALIBRATION_ROWS], y_train[:CALIBRATION_ROWS]):
                handle.write(values(row) + f",{int(label)}\n")
        for arch, build, epochs in (("mlp", mlp, 4), ("cnn", cnn, 6)):
            model = train(build(), x_train, y_train, epochs, seed=0).eval()
            with torch.no_grad():
                logits = model(torch.tensor(x_test).reshape(-1, 1, 28, 28))
            accuracy = float((logits.argmax(1).numpy() == y_test).mean())
            name = f"{dataset}_{arch}"
            (OUT / f"{name}.mforge").write_text(manifest(name, model, arch))
            parameters = sum(p.numel() for p in model.parameters())
            results["models"][name] = {"test_accuracy": accuracy, "parameters": parameters}
            print(f"{name}: test accuracy {accuracy:.4f}, {parameters} parameters")
    if len(sys.argv) > 2:
        x_train, y_train, x_test, y_test = load_cifar(sys.argv[2])
        shape = (3, 32, 32)
        with open(OUT / "cifar_test.csv", "w") as handle:  # 100 rows each: 3,072 values per image
            for row, label in zip(x_test[:100], y_test[:100]):
                handle.write(values(row) + f",{int(label)}\n")
        with open(OUT / "cifar_calib.csv", "w") as handle:
            for row, label in zip(x_train[:100], y_train[:100]):
                handle.write(values(row) + f",{int(label)}\n")
        model = train(cifar_cnn(), x_train, y_train, 15, seed=0, shape=shape).eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test).reshape(-1, *shape))
        accuracy = float((logits.argmax(1).numpy() == y_test).mean())
        (OUT / "cifar_cnn.mforge").write_text(manifest("cifar_cnn", model, "cnn", shape))
        parameters = sum(p.numel() for p in model.parameters())
        results["models"]["cifar_cnn"] = {"test_accuracy": accuracy, "parameters": parameters}
        print(f"cifar_cnn: test accuracy {accuracy:.4f}, {parameters} parameters")
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
