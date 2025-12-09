"""
PyTorch LSTM implementation for RUL prediction on NASA C-MAPSS

- Uses sequences of sensor readings to predict Remaining Useful Life (RUL)
- Works with standard C-MAPSS files:
    train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt (RUL file optional)

Author: Shivansh (BTP)
Inspired by: Zheng et al. (2017) "Long Short-Term Memory Network for RUL estimation"
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ============================================================
# PLOT FUNCTIONS
# ============================================================

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_true_vs_pred(y_true, y_pred, max_points=200):
    _ensure_plots_dir()
    n = min(len(y_true), max_points)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:n], label="True RUL", linewidth=2)
    plt.plot(y_pred[:n], label="Predicted RUL", linewidth=2)
    plt.xlabel("Sample index")
    plt.ylabel("RUL (cycles)")
    plt.title("True vs Predicted RUL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "true_vs_pred.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def plot_error_histogram(y_true, y_pred):
    _ensure_plots_dir()
    errors = y_pred - y_true
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=40, alpha=0.7)
    plt.xlabel("Prediction Error (Pred - True)")
    plt.ylabel("Frequency")
    plt.title("RUL Prediction Error Histogram")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "error_histogram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def plot_scatter(y_true, y_pred):
    _ensure_plots_dir()
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([0, max_val], [0, max_val], "r--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL (Scatter)")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "scatter_true_vs_pred.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def plot_unit_rul(df, unit_id=1):
    _ensure_plots_dir()
    unit_data = df[df["unit"] == unit_id].sort_values("cycle")
    if unit_data.empty:
        print(f"[WARN] Unit {unit_id} not found, skipping RUL curve.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(unit_data["cycle"], unit_data["RUL"], linewidth=2)
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.title(f"RUL Degradation Curve for Engine {unit_id}")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"rul_curve_unit_{unit_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


# ============================================================
# 1. Data Loading & Preprocessing
# ============================================================

def get_cmapss_column_names():
    """Standard C-MAPSS column names: unit, cycle, 3 settings, 21 sensors."""
    cols = ["unit", "cycle", "setting1", "setting2", "setting3"]
    cols += [f"s{i}" for i in range(1, 22)]  # s1 ... s21
    return cols  # total 26


def load_cmapss_fd(data_dir, fd_str="FD001"):
    """
    Load C-MAPSS data for a specific subset, e.g. FD001.

    Expected files in data_dir:
        train_FD001.txt
        test_FD001.txt
        (optionally) RUL_FD001.txt
    """
    col_names_full = get_cmapss_column_names()

    data_dir = os.path.expanduser(data_dir)
    train_path = os.path.join(data_dir, f"train_{fd_str}.txt")
    test_path = os.path.join(data_dir, f"test_{fd_str}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{fd_str}.txt")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    # ---- TRAIN ----
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    if train_df.shape[1] >= 26:
        train_df = train_df.iloc[:, :26]
        train_df.columns = col_names_full
    else:
        col_subset = col_names_full[:train_df.shape[1]]
        train_df.columns = col_subset

    # ---- TEST ----
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    if test_df.shape[1] >= 26:
        test_df = test_df.iloc[:, :26]
        test_df.columns = col_names_full
    else:
        col_subset = col_names_full[:test_df.shape[1]]
        test_df.columns = col_subset

    # ---- RUL (optional) ----
    if os.path.exists(rul_path):
        rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
        rul_df.columns = ["RUL"]
    else:
        print(f"[WARN] RUL file not found: {rul_path} -> using approximate test RUL.")
        rul_df = None

    return train_df, test_df, rul_df


def add_rul_target_train(df, rul_clip=125):
    """Add RUL column for training data: max_cycle - cycle, optionally clipped."""
    df = df.copy()
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    if rul_clip is not None:
        df["RUL"] = df["RUL"].clip(upper=rul_clip)

    return df


def add_rul_target_test(test_df, rul_df=None, rul_clip=125):
    """
    Add RUL column for test data.

    If rul_df is provided (NASA RUL_FD00x.txt), use:
        RUL(cycle) = (max_cycle - cycle) + RUL_last

    If rul_df is None, approximate:
        RUL = max_cycle - cycle
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    test_df = test_df.merge(max_cycles, on="unit", how="left")

    if rul_df is not None:
        last_rul_array = rul_df["RUL"].values  # length = number of units
        units = sorted(test_df["unit"].unique())
        if len(last_rul_array) == len(units):
            # assuming units are 1..N in order
            unit_to_rul_last = {u: last_rul_array[i] for i, u in enumerate(units)}
            test_df["RUL_last"] = test_df["unit"].map(unit_to_rul_last)
            test_df["RUL"] = (test_df["max_cycle"] - test_df["cycle"]) + test_df["RUL_last"]
            test_df.drop("RUL_last", axis=1, inplace=True)
        else:
            print("[WARN] RUL file length mismatch – using approximate RUL for test.")
            test_df["RUL"] = test_df["max_cycle"] - test_df["cycle"]
    else:
        test_df["RUL"] = test_df["max_cycle"] - test_df["cycle"]

    test_df.drop("max_cycle", axis=1, inplace=True)

    if rul_clip is not None:
        test_df["RUL"] = test_df["RUL"].clip(upper=rul_clip)

    return test_df


def scale_features(train_df, test_df, feature_cols):
    """
    Fit MinMaxScaler on train, apply to both train and test.
    Returns scaled copies and the scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, test_scaled, scaler


# ============================================================
# 2. Sequence Generation
# ============================================================

def generate_sequences(df, feature_cols, seq_len=30):
    """
    Convert per-engine time series into sliding-window sequences.

    Returns:
        X: (num_samples, seq_len, num_features)
        y: (num_samples,)
    """
    sequences = []
    labels = []

    for unit_id in df["unit"].unique():
        unit_data = df[df["unit"] == unit_id].sort_values("cycle")
        feature_mat = unit_data[feature_cols].values
        rul_array = unit_data["RUL"].values

        if len(feature_mat) < seq_len:
            continue

        for start in range(0, len(feature_mat) - seq_len + 1):
            end = start + seq_len
            seq_x = feature_mat[start:end, :]
            seq_y = rul_array[end - 1]
            sequences.append(seq_x)
            labels.append(seq_y)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y


class CmapssDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]).float().unsqueeze(-1)
        )


# ============================================================
# 3. LSTM Model
# ============================================================

class LSTMRUL(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# ============================================================
# 4. Training & Evaluation
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    y_pred = np.vstack(all_preds).flatten()
    y_true = np.vstack(all_targets).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse, y_true, y_pred


# ============================================================
# 5. Main
# ============================================================

def main(args):
    # 1) Load data
    print(f"Loading C-MAPSS {args.fd} from {args.data_dir} ...")
    train_df, test_df, rul_df = load_cmapss_fd(args.data_dir, args.fd)

    # 2) Add RUL targets
    train_df = add_rul_target_train(train_df, rul_clip=args.rul_clip)
    test_df = add_rul_target_test(test_df, rul_df, rul_clip=args.rul_clip)

    # 3) Select feature columns
    feature_cols = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

    # 4) Scale features
    train_scaled, test_scaled, _ = scale_features(train_df, test_df, feature_cols)

    # 5) Build sequences
    print("Generating sequences...")
    X_train, y_train = generate_sequences(train_scaled, feature_cols, seq_len=args.seq_len)
    X_test, y_test = generate_sequences(test_scaled, feature_cols, seq_len=args.seq_len)

    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    # 6) Datasets & DataLoaders
    train_loader = DataLoader(CmapssDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(CmapssDataset(X_test, y_test),
                             batch_size=args.batch_size, shuffle=False)

    # 7) Device, model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = LSTMRUL(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 8) Training loop
    best_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        rmse, _, _ = evaluate(model, test_loader, device)

        if rmse < best_rmse:
            best_rmse = rmse

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}  "
            f"Test RMSE: {rmse:.4f}  (Best: {best_rmse:.4f})"
        )

    print("Training finished.")
    print(f"Best RMSE on test set: {best_rmse:.4f}")

    # 9) Final evaluation and plots
    rmse, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"Final RMSE (for plots): {rmse:.4f}")
    print("Generating plots into ./plots ...")
    plot_true_vs_pred(y_true, y_pred)
    plot_error_histogram(y_true, y_pred)
    plot_scatter(y_true, y_pred)
    plot_unit_rul(train_df, unit_id=1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Users/shivanshtiwari/Downloads/CMAPSSData")
    parser.add_argument("--fd", type=str, default="FD001")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--rul_clip", type=int, default=125)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    main(args)
