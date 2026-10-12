"""Train a zoo of CNN architectures and export them with export_mforge.py.

Author: Vaishnavi

ForgeCert is not tied to one network shape: these models exercise every
operator of the integer target (all sized for an 8-bit microcontroller):

  lenet   LeNet-5 style: Conv 5x5 -> ReLU -> MaxPool 2 -> Conv 5x5 -> ReLU -> MaxPool 2
          -> Dense -> ReLU -> Dense
  lenet_avg  the same with average pooling (lowered to depthwise convolutions)
  dscnn   MobileNet-style depthwise-separable CNN: Conv/BN/ReLU, then depthwise 3x3 and
          pointwise 1x1 blocks with batch normalisation, GlobalAveragePool, Dense
  resnet  ResNet style: stem Conv/BN/ReLU, two residual blocks (x + BN(Conv(ReLU(BN(Conv x))))
          then ReLU; the second downsamples with a 1x1 projection), GlobalAveragePool, Dense

Usage: python train_cnn_zoo.py <dir with the MNIST/Fashion idx .gz files> [names...]
Writes review_2_core_implementation/models/image/<dataset>_<arch>.mforge and
research/results/cnn_zoo.json (test accuracy on all 10,000 test images).
"""

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_mforge import export  # noqa: E402
from train_image_models import OUT, load, train  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "cnn_zoo.json"


def lenet(pool=nn.MaxPool2d):
    return nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), pool(2), nn.Conv2d(6, 12, 5), nn.ReLU(), pool(2),
                         nn.Flatten(), nn.Linear(192, 32), nn.ReLU(), nn.Linear(32, 10))


def dscnn():
    def block(cin, cout, stride):
        return [nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(),
                nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
    return nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                         *block(16, 32, 1), *block(32, 48, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 10))


class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.project = None
        if stride != 1 or cin != cout:
            self.project = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        skip = x if self.project is None else self.project(x)
        return F.relu(y + skip)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.block1 = Residual(8, 8, 1)
        self.block2 = Residual(8, 16, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.block2(self.block1(self.stem(x)))), 1))


ARCHS = {
    "lenet": (lambda: lenet(nn.MaxPool2d), 4),
    "lenet_avg": (lambda: lenet(nn.AvgPool2d), 4),
    "dscnn": (dscnn, 8),
    "resnet": (ResNet, 4),
}


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    wanted = sys.argv[2:] or list(ARCHS)
    torch.set_num_threads(8)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"author": "Vaishnavi", "models": {}}
    for dataset, prefix in (("mnist", ""), ("fashion", "fashion-")):
        x_train, y_train, x_test, y_test = load(directory, prefix)
        for arch in wanted:
            build, epochs = ARCHS[arch]
            model = train(build(), x_train, y_train, epochs, seed=0).eval()
            with torch.no_grad():
                logits = model(torch.tensor(x_test).reshape(-1, 1, 28, 28))
            accuracy = float((logits.argmax(1).numpy() == y_test).mean())
            name = f"{dataset}_{arch}"
            text = export(model, (1, 28, 28), name,
                          comment="trained with PyTorch on 28x28 images, pixels in [0, 1] (author: Vaishnavi); "
                                  "exported by research/export_mforge.py")
            (OUT / f"{name}.mforge").write_text(text)
            parameters = sum(p.numel() for p in model.parameters())
            results["models"][name] = {"test_accuracy": accuracy, "parameters": parameters}
            print(f"{name}: test accuracy {accuracy:.4f}, {parameters} parameters", flush=True)
            RESULTS.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
