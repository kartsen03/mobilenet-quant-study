"""Download CIFAR-10 (python version, ~163 MB) into data/ and verify its MD5.

torchvision fetches https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and
checks the archive checksum before extracting.
"""
import sys
from pathlib import Path

from torchvision.datasets import CIFAR10

root = Path(__file__).resolve().parents[1] / "data"
root.mkdir(exist_ok=True)
for train in (True, False):
    ds = CIFAR10(str(root), train=train, download=True)
    print(("train" if train else "test"), len(ds), "images")
sys.exit(0)
