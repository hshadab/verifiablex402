#!/usr/bin/env python3
"""Train the transaction integrity classifier.

Architecture: MLP [24] -> [36] -> [36] -> [5]
- 24 input features
- 2 hidden layers with 36 neurons each (ReLU)
- 5 output classes (cross-entropy loss)

Reads: training_features.csv
Outputs: tx_integrity_model.pt (PyTorch state dict)
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_FILE = os.path.join(os.path.dirname(__file__), "training_features.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "tx_integrity_model.pt")

INPUT_DIM = 24
HIDDEN_DIM = 36
OUTPUT_DIM = 5
SCALE = 128  # fixed-point scale (2^7)

EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
VALIDATION_SPLIT = 0.15


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

    return model


if __name__ == "__main__":
    train()
