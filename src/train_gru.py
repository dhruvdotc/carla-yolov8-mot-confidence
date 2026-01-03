# Train a GRU model to predict FUTURE absolute (x,y) positions from PAST (x,y)
# Uses dataset.npz created by make_windows.py
# Reports ADE and FDE on train/val/test splits

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# USER PATHS (EDIT IF NEEDED)
# =========================

DATASET_NPZ = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/dataset.npz"
SAVE_WEIGHTS = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/gru_abs.pt"

# =========================
# TRAIN SETTINGS (SIMPLE DEFAULTS)
# =========================

SEED = 0
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 1

# dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# FUNCTION DEFINITIONS
# =========================

def set_seed(seed):
    # Make runs repeatable
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_time(X, Y):
    n = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[n_train:n_train + n_val]
    Y_val = Y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    Y_test = Y[n_train + n_val:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def ade_fde(pred, true):
    # pred, true: (B, T, 2)
    diff = pred - true
    dists = torch.sqrt(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1])
    ade = torch.mean(dists)
    fde = torch.mean(dists[:, -1])
    return ade, fde


# =========================
# DATASET WRAPPER
# =========================

class TrajDataset(Dataset):
    # Stores X (past) and Y (future) as torch tensors

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================
# MODEL
# =========================

class GRUPredictor(nn.Module):
    # GRU encoder -> MLP head -> outputs future absolute positions

    def __init__(self, hidden_size, num_layers, future_len):
        super().__init__()
        self.future_len = future_len
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, future_len * 2),
        )

    def forward(self, past_xy):
        out, h = self.gru(past_xy)        # h: (num_layers, B, H)
        last_h = h[-1]                    # (B, H)
        pred_flat = self.head(last_h)     # (B, future_len*2)
        pred = pred_flat.view(-1, self.future_len, 2)
        return pred


# =========================
# TRAIN / EVAL
# =========================

def run_epoch(model, loader, optimizer=None):
    # If optimizer is provided -> train mode, else eval mode
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    count = 0

    for past_xy, future_xy in loader:
        past_xy = past_xy.to(DEVICE)
        future_xy = future_xy.to(DEVICE)

        pred = model(past_xy)

        loss = nn.functional.mse_loss(pred, future_xy)
        ade, fde = ade_fde(pred, future_xy)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = past_xy.size(0)
        total_loss += float(loss.item()) * bs
        total_ade += float(ade.item()) * bs
        total_fde += float(fde.item()) * bs
        count += bs

    return total_loss / count, total_ade / count, total_fde / count


# =========================
# MAIN
# =========================

set_seed(SEED)

data = np.load(DATASET_NPZ, allow_pickle=True)
X = data["X"]
Y = data["Y"]
past_len = int(data["past"])
future_len = int(data["future"])

perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]

X_train, Y_train, X_val, Y_val, X_test, Y_test = split_by_time(X, Y)

train_ds = TrajDataset(X_train, Y_train)
val_ds = TrajDataset(X_val, Y_val)
test_ds = TrajDataset(X_test, Y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

model = GRUPredictor(HIDDEN_SIZE, NUM_LAYERS, future_len).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("DEVICE:", DEVICE)
print("Windows total:", len(X))
print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))
print("Past:", past_len, "Future:", future_len)

best_val_ade = 1e9

for epoch in range(1, EPOCHS + 1):
    train_loss, train_ade, train_fde = run_epoch(model, train_loader, optimizer)
    val_loss, val_ade, val_fde = run_epoch(model, val_loader, optimizer=None)

    print("Epoch", epoch,
          "| train loss", round(train_loss, 6),
          "ADE", round(train_ade, 6),
          "FDE", round(train_fde, 6),
          "| val loss", round(val_loss, 6),
          "ADE", round(val_ade, 6),
          "FDE", round(val_fde, 6))

    # Save best by val ADE
    if val_ade < best_val_ade:
        best_val_ade = val_ade
        torch.save(model.state_dict(), SAVE_WEIGHTS)

# Load best and test
model.load_state_dict(torch.load(SAVE_WEIGHTS, map_location=DEVICE))
test_loss, test_ade, test_fde = run_epoch(model, test_loader, optimizer=None)

print("DONE")
print("Best val ADE:", best_val_ade)
print("Test loss:", test_loss)
print("Test ADE:", test_ade)
print("Test FDE:", test_fde)
print("Saved weights:", SAVE_WEIGHTS)
