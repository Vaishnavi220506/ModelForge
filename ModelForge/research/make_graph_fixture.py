"""Writes tests/fixtures/graph_zoo.mforge (+ graph_zoo_data.csv): a small seeded network that uses every
operator of the integer target (author: Vaishnavi).

  conv 3x3 + BN (folded) + ReLU -> a
  depthwise 3x3 + BN (folded) + ReLU on a -> b
  ReLU(a + b)                           residual Add
  MaxPool 2, BN after the pool (not foldable: a 1x1 depthwise conv)
  AvgPool 2 (lowered to a depthwise conv), Flatten, Dense 16 -> 3
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_mforge import export  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "review_2_core_implementation" / "tests" / "fixtures" / "graph_zoo.mforge"


class Zoo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.dw = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(4)
        self.avg = nn.AvgPool2d(2)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        a = F.relu(self.bn1(self.conv(x)))
        b = F.relu(self.bn2(self.dw(a)))
        y = F.relu(a + b)
        return self.head(torch.flatten(self.avg(self.bn3(self.pool(y))), 1))


def main():
    torch.manual_seed(7)
    model = Zoo()
    with torch.no_grad():
        for bn in (model.bn1, model.bn2, model.bn3):
            bn.running_mean.uniform_(-0.2, 0.2)
            bn.running_var.uniform_(0.5, 1.5)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.1, 0.1)
    # 24 seeded inputs in [0, 1]^64 (label = the float model's class)
    x = torch.rand(24, 1, 8, 8, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        labels = model.eval()(x).argmax(1)
    rows = [",".join(f"{v:.6g}" for v in row.ravel().tolist()) + f",{int(c)}" for row, c in zip(x, labels)]
    OUT.with_name("graph_zoo_data.csv").write_text("\n".join(rows) + "\n")
    OUT.write_text(export(model.eval(), (1, 8, 8), "graph_zoo",
                          comment="seeded fixture covering every integer-target operator (author: Vaishnavi)"))


if __name__ == "__main__":
    main()
