#######################################################################################################################
#######################################################################################################################
# Title:        Spike NILM
# Topic:        Non-intrusive load monitoring
# File:         helper
# Date:         18.07.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Helper functions for the Spike NILM main function.
"""

#######################################################################################################################
# Import external libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import snntorch as snn
import copy

try:
    from snntorch import spikegen
except ImportError:
    spikegen = None


#######################################################################################################################
# Data Loading and Handling
#######################################################################################################################
# ==============================================================================
# FNC: data loading
# ==============================================================================
def load_data(mat_file, device_ids, maxLen=-1):
    """Load X and Y for multiple device IDs in a single file read.

    Returns:
        X: ndarray [n_samples, 275, 2]
        Y: ndarray [n_samples, n_devices]
    """
    data = loadmat(mat_file)
    X = data["input"]
    Y_all = data["output"]

    if maxLen is not None and maxLen > 0:
        X = X[:maxLen]
        Y_all = Y_all[:maxLen]

    X = X[:, 2:277, :]
    # Stack selected device columns into a 2D array [n_samples, n_devices]
    Y = np.stack([Y_all[:, 1 + dev_id] for dev_id in device_ids], axis=1)
    return X, Y


# ==============================================================================
# FNC: Downsampling
# ==============================================================================
def downsample(X, y, rate=1):
    """Downsample X and y by an integer rate.

    Args:
        X: ndarray [n_samples, seq_len, channels]
        y: ndarray [n_samples, n_devices]
        rate: integer downsampling factor (1 = no downsampling)

    Returns:
        X_ds, y_ds: downsampled arrays
    """
    if rate is None or rate <= 1:
        return X.copy(), y.copy()
    return X[::rate].copy(), y[::rate].copy()


# ==============================================================================
# FNC: Filtering
# ==============================================================================
def filter_output(y_raw, method="moving_average", window=5):
    """Filter the per-device power time-series.

    Args:
        y_raw: ndarray [n_samples, n_devices]
        method: 'moving_average' or 'median' (median not yet optimized)
        window: integer filter window (samples)

    Returns:
        y_filtered: ndarray same shape as y_raw (float32)
    """
    y = np.asarray(y_raw)
    n_samples, n_devices = y.shape
    y_filtered = np.empty_like(y, dtype=np.float32)

    if method == "moving_average":
        kernel = np.ones(window, dtype=np.float32) / float(window)
        for c in range(n_devices):
            y_filtered[:, c] = np.convolve(y[:, c], kernel, mode="same")
    elif method == "median":
        # Fallback: use a simple running median via padding and sliding window
        pad = window // 2
        y_padded = np.pad(y, ((pad, pad), (0, 0)), mode="edge")
        for c in range(n_devices):
            col = y_padded[:, c]
            out = np.empty(n_samples, dtype=np.float32)
            for i in range(n_samples):
                out[i] = np.median(col[i:i + window])
            y_filtered[:, c] = out
    else:
        raise ValueError("Unknown filter method: %s" % method)

    return y_filtered


# ==============================================================================
# FNC: Binary conversion
# ==============================================================================
def binarize_output(y_filtered, threshold):
    """Convert continuous power to binary ON/OFF states.

    Args:
        y_filtered: ndarray [n_samples, n_devices]
        threshold: scalar or iterable of per-device thresholds

    Returns:
        y_binary: ndarray of ints {0,1}
    """
    y = np.asarray(y_filtered)
    th = threshold
    # allow per-device thresholds
    if hasattr(th, "__iter__") and len(th) == y.shape[1]:
        th_arr = np.asarray(th).reshape(1, -1)
    else:
        th_arr = float(th)
    return (y >= th_arr).astype(np.int32)


# ==============================================================================
# FNC: Data Loader
# ==============================================================================
class NNDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.Y[i]


# ==============================================================================
# FNC: Data Balancing
# ==============================================================================
def balance_sequences(X_seq, Y_seq, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    classes, counts = np.unique(Y_seq, return_counts=True)
    min_count = counts.min()
    selected = []

    for class_id in classes:
        class_indices = np.where(Y_seq == class_id)[0]
        selected.append(rng.choice(class_indices, size=min_count, replace=False))

    indices = np.sort(np.concatenate(selected))
    return X_seq[indices], Y_seq[indices]


#######################################################################################################################
# Feature Calculation
#######################################################################################################################
# ==============================================================================
# FNC: Frequency Transform
# ==============================================================================
def extract_features(X, n_harmonics=15, selector=None, return_names=False):
    """Extract configurable spectral/statistical features from voltage/current waveforms."""
    if selector is None:
        selector = {
            "voltage_harmonics": True,
            "current_harmonics": True,
            "voltage_stats": True,
            "current_stats": True,
            "power_stats": True,
        }

    V = X[:, :, 0]
    I = X[:, :, 1]

    V_fft = np.abs(np.fft.rfft(V, axis=1))[:, 1:n_harmonics + 1] / V.shape[1] * 2
    I_fft = np.abs(np.fft.rfft(I, axis=1))[:, 1:n_harmonics + 1] / I.shape[1] * 2

    rms_v = np.sqrt(np.mean(V ** 2, axis=1, keepdims=True))
    rms_i = np.sqrt(np.mean(I ** 2, axis=1, keepdims=True))
    peak_v = np.max(np.abs(V), axis=1, keepdims=True)
    peak_i = np.max(np.abs(I), axis=1, keepdims=True)
    real_power = np.mean(V * I, axis=1, keepdims=True)
    apparent_power = rms_v * rms_i

    feature_blocks = []
    feature_names = []

    if selector.get("voltage_harmonics", True):
        feature_blocks.append(V_fft)
        feature_names.extend([f"V_h{idx}" for idx in range(1, n_harmonics + 1)])
    if selector.get("current_harmonics", True):
        feature_blocks.append(I_fft)
        feature_names.extend([f"I_h{idx}" for idx in range(1, n_harmonics + 1)])
    if selector.get("voltage_stats", True):
        feature_blocks.extend([rms_v, peak_v])
        feature_names.extend(["rms_v", "peak_v"])
    if selector.get("current_stats", True):
        feature_blocks.extend([rms_i, peak_i])
        feature_names.extend(["rms_i", "peak_i"])
    if selector.get("power_stats", True):
        feature_blocks.extend([real_power, apparent_power])
        feature_names.extend(["real_power", "apparent_power"])

    if not feature_blocks:
        raise ValueError("Feature selector disabled all feature groups. Enable at least one input feature group.")

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    if return_names:
        return features, feature_names

    return features


# ==============================================================================
# FNC: Delta Features
# ==============================================================================
def feature_deltas(X, mode="absolute"):
    """Compute consecutive-frame feature differences.

    Args:
        X: ndarray [n_samples, n_features]
        mode: 'absolute' -> abs(x_t - x_{t-1}), 'signed' -> x_t - x_{t-1}
    """
    deltas = np.diff(X, axis=0, prepend=X[:1])
    if mode == "absolute":
        deltas = np.abs(deltas)
    elif mode != "signed":
        raise ValueError(f"Unknown delta mode: {mode}. Use 'absolute' or 'signed'.")
    deltas[0] = 0.0

    return deltas.astype(np.float32, copy=False)


#######################################################################################################################
# SNN Functions
#######################################################################################################################
# ==============================================================================
# SNN Model
# ==============================================================================
class SNNModel(nn.Module):
    def __init__(self, Nh, hidden, C):
        super().__init__()
        self.fc1 = nn.Linear(Nh, hidden)
        self.lif1 = snn.Leaky(beta=0.95)
        self.fc2 = nn.Linear(hidden, C)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spks = []
        for t in range(x.shape[1]):
            cur1 = self.fc1(x[:, t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spks.append(spk2)
        return torch.stack(spks, 1)


# ==============================================================================
# Spike Encoding
# ==============================================================================
def encode(X, coding):
    if spikegen is None:
        raise ImportError("snntorch is required for encode_spikes but is not installed in this environment.")

    if coding == 'current':
        return X

    if coding == 'rate':
        return spikegen.rate(X)

    if coding == 'latency':
        # returns (steps,batch,...); here use Nt encoding steps
        return spikegen.latency(X, num_steps=X.shape[1])

    return None


# ==============================================================================
# FNC: SNN Training Loop
# ==============================================================================
def train_snn(mdl, X_train, y_train, X_val, y_val, opt, loss_fnc, cfg, device):
    # ------------------------------------------
    # Description
    # ------------------------------------------

    # ------------------------------------------
    # Init
    # ------------------------------------------
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # ------------------------------------------
    # Data Prep
    # ------------------------------------------
    snn_train_dataset = TensorDataset(X_train, y_train)
    snn_val_dataset = TensorDataset(X_val, y_val)
    snn_train_loader = DataLoader(snn_train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    snn_val_loader = DataLoader(snn_val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ------------------------------------------
    # Train
    # ------------------------------------------
    for epoch in range(cfg["EPOCHS"]):
        # Init
        mdl.train()
        epoch_train_loss = 0.0

        # Loop over Batches
        for xb, yb in snn_train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            _, mem_rec = mdl(xb)
            logits = mem_rec.mean(dim=1)
            loss = loss_fnc(logits, yb)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item() * xb.size(0)

        # Eval Validation data
        mdl.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in snn_val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                _, mem_rec = mdl(xb)
                logits = mem_rec.mean(dim=1)
                epoch_val_loss += loss_fnc(logits, yb).item() * xb.size(0)

        # Report loss
        epoch_train_loss /= max(len(snn_train_dataset), 1)
        epoch_val_loss /= max(len(snn_val_dataset), 1)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"SNN {epoch + 1}/{cfg['snn_epochs']} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(mdl.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["snn_patience"]:
                print(f"SNN early stopping at epoch {epoch + 1}")
                break

    # ------------------------------------------
    # Finalize
    # ------------------------------------------
    # Best State
    if best_state is not None:
        mdl.load_state_dict(best_state)

    # Saving
    torch.save(
        {
            "model_state_dict": mdl.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        cfg["snn_checkpoint_path"],
    )
    print(f"Saved SNN checkpoint to {cfg['snn_checkpoint_path']}")

    return mdl, train_losses, val_losses


# ==============================================================================
# FNC: SNN Testing Loop
# ==============================================================================
def test_snn(mdl, X_test, cfg, device):
    # ------------------------------------------
    # Description
    # ------------------------------------------

    # ------------------------------------------
    # Load Checkpoint
    # ------------------------------------------
    if not os.path.exists(cfg["snn_checkpoint_path"]):
        raise FileNotFoundError(f"Missing SNN checkpoint: {cfg['snn_checkpoint_path']}")

    checkpoint = torch.load(cfg["snn_checkpoint_path"], map_location=device, weights_only=False)
    mdl.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded SNN checkpoint from {cfg['snn_checkpoint_path']}")

    # ------------------------------------------
    # Eval
    # ------------------------------------------
    # Init
    mdl.eval()

    # Calc
    with torch.no_grad():
        _, mem_all = mdl(X_test.to(device))
        logits_all = mem_all.mean(dim=1).cpu()
        probs_all = torch.sigmoid(logits_all)
        y_pred = (probs_all >= 0.5).float()

        _, mem_sample = mdl(X_test.to(device))
        mem_sample = mem_sample[0].cpu().numpy()

    y_pred = y_pred.numpy().astype(int).ravel()

    return y_pred


#######################################################################################################################
# Regression Functions
#######################################################################################################################
# ==============================================================================
# FNC: LSTM Training Loop
# ==============================================================================