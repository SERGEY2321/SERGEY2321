"""
Train a ResNet-18 classifier to recognise individual math symbols
from the handwritten symbol dataset.

Usage:
    python -m training.train_ocr \
        --data dataset/symbols \
        --epochs 20 \
        --batch 64 \
        --lr 1e-3 \
        --out backend/models/symbol_clf.pth

The script automatically uses GPU if available, otherwise CPU.
For Google Colab training see notebooks/train_colab.ipynb.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from backend.ocr.handwritten import SYMBOL_CLASSES


# ── Dataset ───────────────────────────────────────────────────────────────────

class SymbolDataset(Dataset):
    def __init__(self, records: list[dict], root: str, transform=None):
        self.records = records
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = Path(self.root) / rec["file"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = rec["class_idx"]
        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset/symbols")
    parser.add_argument("--epochs", type=int,   default=20)
    parser.add_argument("--batch",  type=int,   default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--out",    default="backend/models/symbol_clf.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata
    meta_path = Path(args.data) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. "
            "Run dataset/generator/generate_handwritten.py first."
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    train_ds = SymbolDataset(metadata["train"], root=args.data, transform=TRAIN_TRANSFORM)
    val_ds   = SymbolDataset(metadata["val"],   root=args.data, transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = build_model(num_classes=len(SYMBOL_CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  Model saved  (best val_acc={best_acc*100:.2f}%)")

    print(f"\nTraining complete.  Best val accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {out_path}")


if __name__ == "__main__":
    main()
