"""Train real classifiers and export them as ModelForge manifests.

Author: Vaishnavi

Uses the four datasets bundled with scikit-learn (no download): Iris, Wine,
Breast Cancer and Digits. Features are standardised with the training split's
mean and standard deviation, so every exported model takes z-scores and the
realistic input box is roughly [-4, 4] per feature.

Outputs (committed, so the C++ tools never need Python):
  review_2_core_implementation/models/real/<dataset>_<arch>.mforge
  review_2_core_implementation/models/real/<dataset>_test.csv   (features..., label)
  review_2_core_implementation/models/real/<dataset>_train.csv
  website/models.js   (weights, test data, feature scaling for the website)

Usage: python train_real_models.py   (needs scikit-learn 1.5)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

OUT = Path(__file__).resolve().parents[1] / "review_2_core_implementation" / "models" / "real"
DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer,
            "digits": load_digits}
WEBSITE = Path(__file__).resolve().parents[2] / "website" / "models.js"
ARCHITECTURES = {"mlp16": (16,), "mlp32x16": (32, 16)}


def values(array):
    return ",".join(f"{v:.9g}" for v in np.asarray(array, dtype=np.float32).ravel())


def manifest(name, model, n_inputs, n_classes):
    weights = [np.asarray(w, dtype=np.float32) for w in model.coefs_]
    biases = [np.asarray(b, dtype=np.float32) for b in model.intercepts_]
    if n_classes == 2:
        # scikit-learn's binary head is one logistic unit z. softmax([0, z])
        # equals [1 - sigmoid(z), sigmoid(z)], so prepend a zero column.
        weights[-1] = np.hstack([np.zeros_like(weights[-1]), weights[-1]])
        biases[-1] = np.concatenate([[0.0], biases[-1]]).astype(np.float32)
    lines = [f"# Real {name} classifier trained with scikit-learn (author: Vaishnavi).",
             "# Input: standardised features (z-scores of the training split).",
             f"model {name}", f"input x float32 1,{n_inputs}", f"output y float32 1,{n_classes}"]
    for i, (w, b) in enumerate(zip(weights, biases)):
        lines.append(f"tensor w{i} float32 {w.shape[0]},{w.shape[1]} values={values(w)}")
        lines.append(f"tensor b{i} float32 1,{b.shape[0]} values={values(b)}")
    previous = "x"
    for i in range(len(weights) - 1):
        lines.append(f"node dense_{i} Gemm {previous} w{i} b{i} -> z{i}")
        lines.append(f"node relu_{i} Relu z{i} -> h{i}")
        previous = f"h{i}"
    last = len(weights) - 1
    lines.append(f"node classifier Gemm {previous} w{last} b{last} -> logits")
    lines.append("node probabilities Softmax logits -> y axis=1")
    return "\n".join(lines) + "\n"


def write_csv(path, features, labels):
    with open(path, "w") as handle:
        for row, label in zip(features, labels):
            handle.write(values(row) + f",{int(label)}\n")


def rounded(array, digits=5):
    return [float(f"{v:.{digits}g}") for v in np.asarray(array, dtype=np.float32).ravel()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    site = {"author": "Vaishnavi", "datasets": {}}
    for dataset, loader in DATASETS.items():
        data = loader()
        x_train, x_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        std[std == 0] = 1.0
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        write_csv(OUT / f"{dataset}_train.csv", x_train, y_train)
        write_csv(OUT / f"{dataset}_test.csv", x_test, y_test)
        n_classes = len(np.unique(data.target))
        names = [str(n) for n in getattr(data, "feature_names", range(x_train.shape[1]))]
        site["datasets"][dataset] = {
            "features": names, "classes": [str(c) for c in data.target_names],
            "mean": rounded(mean, 7), "std": rounded(std, 7),
            "test": [rounded(row) for row in x_test], "labels": [int(v) for v in y_test],
            "models": {}}
        for arch, hidden in ARCHITECTURES.items():
            model = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", max_iter=2000,
                                  random_state=0).fit(x_train, y_train)
            name = f"{dataset}_{arch}"
            (OUT / f"{name}.mforge").write_text(
                manifest(name, model, x_train.shape[1], n_classes))
            print(f"{name}: test accuracy {model.score(x_test, y_test):.3f}")
            weights = [np.asarray(w, dtype=np.float32) for w in model.coefs_]
            biases = [np.asarray(b, dtype=np.float32) for b in model.intercepts_]
            if n_classes == 2:
                weights[-1] = np.hstack([np.zeros_like(weights[-1]), weights[-1]])
                biases[-1] = np.concatenate([[0.0], biases[-1]]).astype(np.float32)
            site["datasets"][dataset]["models"][arch] = {
                "layers": [{"in": int(w.shape[0]), "out": int(w.shape[1]),
                            "w": [float(v) for v in w.ravel()], "b": [float(v) for v in b]}
                           for w, b in zip(weights, biases)]}
    WEBSITE.parent.mkdir(parents=True, exist_ok=True)
    WEBSITE.write_text("// Generated by ModelForge/research/train_real_models.py (author: Vaishnavi).\n"
                       "window.MODELFORGE_MODELS = " + json.dumps(site, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
