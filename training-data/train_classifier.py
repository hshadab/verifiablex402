#!/usr/bin/env python3
"""Train the transaction integrity classifier.

Architecture: MLP [24] -> [36] -> [36] -> [5]
- 24 input features
- 2 hidden layers with 36 neurons each (ReLU)
- 5 output classes (cross-entropy loss)

Reads: training_features.csv
Outputs: tx_integrity_model.pt (PyTorch state dict)
         training_metrics.json (training metrics for reproducibility)
"""

import json
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_FILE = os.path.join(os.path.dirname(__file__), "training_features.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "tx_integrity_model.pt")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "training_metrics.json")

INPUT_DIM = 24
HIDDEN_DIM = 36
OUTPUT_DIM = 5
SCALE = 128  # fixed-point scale (2^7)

EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
VALIDATION_SPLIT = 0.15

CLASS_NAMES = [
    "GENUINE_COMMERCE",
    "LOW_ACTIVITY",
    "SCRIPTED_BENIGN",
    "CIRCULAR_PAYMENTS",
    "WASH_TRADING",
]


class TxIntegrityMLP(nn.Module):
    """3-layer MLP for transaction integrity classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data():
    """Load training data from CSV."""
    features = []
    labels = []

    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            feat = [float(x) for x in row[:-1]]
            label = int(row[-1])
            features.append(feat)
            labels.append(label)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Normalize to [0, 1] for training (the model operates on [0, 128] in Rust)
    X = X / SCALE

    return X, y


def train():
    """Train the model."""
    print(f"Loading data from {DATA_FILE}")
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {OUTPUT_DIM} classes")

    # Split into train/val
    n = X.shape[0]
    n_val = int(n * VALIDATION_SPLIT)
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TxIntegrityMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.shape[0]
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, val_predicted = val_outputs.max(1)
            val_acc = val_predicted.eq(y_val).sum().item() / y_val.size(0)

        if (epoch + 1) % 10 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1:3d}: loss={total_loss/total:.4f} "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_FILE)

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {MODEL_FILE}")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    for name, p in model.named_parameters():
        print(f"  {name}: {list(p.shape)} = {p.numel()}")

    # Compute per-class accuracy and confusion matrix
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, val_predicted = val_outputs.max(1)

    confusion = [[0] * OUTPUT_DIM for _ in range(OUTPUT_DIM)]
    per_class_correct = [0] * OUTPUT_DIM
    per_class_total = [0] * OUTPUT_DIM

    for true, pred in zip(y_val.tolist(), val_predicted.tolist()):
        confusion[true][pred] += 1
        per_class_total[true] += 1
        if true == pred:
            per_class_correct[true] += 1

    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        acc = per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
        per_class_acc[name] = round(acc, 4)

    # Export training metrics
    metrics = {
        "total_samples": n,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "best_val_accuracy": round(best_val_acc, 4),
        "total_parameters": total_params,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": confusion,
        },
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining metrics saved to {METRICS_FILE}")
    print(f"Per-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f"  {name}: {acc:.1%}")

    return model


if __name__ == "__main__":
    train()
