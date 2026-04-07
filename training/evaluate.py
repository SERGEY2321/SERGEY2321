"""
Evaluate a trained symbol classifier on the validation set.

Usage:
    python -m training.evaluate \
        --data dataset/symbols \
        --model backend/models/symbol_clf.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms

from backend.ocr.handwritten import SYMBOL_CLASSES
from training.train_ocr import SymbolDataset, VAL_TRANSFORM, build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(SYMBOL_CLASSES)
    class_total   = [0] * len(SYMBOL_CLASSES)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        for p, l in zip(preds.cpu(), labels.cpu()):
            class_total[l] += 1
            class_correct[l] += int(p == l)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    overall_acc = correct / total if total else 0.0
    return overall_acc, class_correct, class_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="dataset/symbols")
    parser.add_argument("--model", default="backend/models/symbol_clf.pth")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = Path(args.data) / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    val_ds = SymbolDataset(metadata["val"], root=args.data, transform=VAL_TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(SYMBOL_CLASSES)).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model}")

    overall_acc, class_correct, class_total = evaluate(model, val_loader, device)

    print(f"\nOverall val accuracy: {overall_acc*100:.2f}%\n")
    print(f"{'Symbol':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 42)
    for i, sym in enumerate(SYMBOL_CLASSES):
        if class_total[i] == 0:
            continue
        acc = class_correct[i] / class_total[i]
        print(f"{sym:<12} {class_correct[i]:>8} {class_total[i]:>8} {acc*100:>7.1f}%")


if __name__ == "__main__":
    main()
