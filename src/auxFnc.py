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
    from snntorch import surrogate
except ImportError:
    spikegen = None

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    hamming_loss,
)

#######################################################################################################################
# General
#######################################################################################################################
# ==============================================================================
# FNC: Machine hardware
# ==============================================================================
def check_hardware(config):
    requested_device = str(config.get("DEVICE", "auto")).lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE must be 'auto', 'cpu', or 'cuda'.")
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        gpu_index = int(config.get("GPU_INDEX", 0))
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(f"GPU_INDEX={gpu_index} is not available.")
        device = torch.device(f"cuda:{gpu_index}")
    elif requested_device == "cuda":
        raise RuntimeError("DEVICE='cuda' was requested but CUDA is unavailable.")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return device


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
    if device_ids[0] == 999:
        Y = Y_all[:, 1:1 + Y_all.shape[1]]
    else:
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
        return X, y
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
    """Undersample sequence rows by their joint binary target state."""
    rng = np.random.default_rng(rng_seed)
    targets = np.asarray(Y_seq)
    if targets.ndim == 2:
        class_ids = targets.astype(np.int64) @ (1 << np.arange(targets.shape[1]))
    else:
        class_ids = targets

    classes, counts = np.unique(class_ids, return_counts=True)
    if len(classes) < 2:
        return X_seq, Y_seq
    min_count = counts.min()
    selected = []

    for class_id in classes:
        class_indices = np.where(class_ids == class_id)[0]
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


def create_sequences(features, targets, sequence_length, stride=1, mode="s2p"):
    """
    Create sliding windows for sequence-to-point or sequence-to-sequence learning.

    Parameters
    ----------
    features : ndarray, shape (N, F)
        Input features.
    targets : ndarray, shape (N, C)
        Target values.
    sequence_length : int
        Length of each input sequence.
    stride : int
        Window stride.
    mode : {"s2p", "s2s"}
        s2p -> target is the final sample of each window.
        s2s -> target is the complete target sequence.

    Returns
    -------
    X : ndarray
        Shape (num_windows, sequence_length, F)
    y : ndarray
        s2p: (num_windows, C)
        s2s: (num_windows, sequence_length, C)
    """

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError("features must have shape (N, F).")

    if targets.ndim != 2:
        raise ValueError("targets must have shape (N, C).")

    if len(features) != len(targets):
        raise ValueError("features and targets must contain the same number of samples.")

    if sequence_length < 1 or sequence_length > len(features):
        raise ValueError("Invalid sequence_length.")

    if stride < 1:
        raise ValueError("stride must be at least 1.")

    # Input windows
    X = np.lib.stride_tricks.sliding_window_view(features, window_shape=sequence_length, axis=0)
    X = np.moveaxis(X, -1, 1)[::stride].copy()

    if mode.lower() == "s2p":
        y = targets[sequence_length - 1::stride].copy()

    elif mode.lower() == "s2s":
        y = np.lib.stride_tricks.sliding_window_view(targets, window_shape=sequence_length, axis=0)
        y = np.moveaxis(y, -1, 1)[::stride].copy()

    else:
        raise ValueError("mode must be 's2p' or 's2s'.")

    return X, y


#######################################################################################################################
# SNN Functions
#######################################################################################################################
# ==============================================================================
# SNN s2s and s2p
# ==============================================================================
def get_logits(mem_rec, mode):
    mode = mode.lower()

    if mode == "s2p":
        return mem_rec.mean(dim=1)

    if mode == "s2s":
        return mem_rec

    raise ValueError("mode must be 's2p' or 's2s'")


# ==============================================================================
# SNN Model
# ==============================================================================
class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, beta=0.95, dropout=0.2):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_layers = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_size

        for _ in range(num_layers):
            self.hidden_layers.append(nn.Sequential(nn.Linear(in_features, hidden_size), nn.LayerNorm(hidden_size)))
            self.hidden_lifs.append(snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid()))
            self.dropouts.append(nn.Dropout(dropout))
            in_features = hidden_size

        # Residual classifier
        self.output_layer = nn.Sequential(nn.Linear(hidden_size + input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Input must be [batch, sequence, features]")

        hidden_memories = [lif.init_leaky() for lif in self.hidden_lifs]
        outputs = []
        T = x.shape[1]

        for t in range(T):
            # Current FFT
            current_input = x[:, t]
            activations = current_input

            for i, (layer, lif, dropout) in enumerate(zip(self.hidden_layers, self.hidden_lifs, self.dropouts)):
                current = layer(activations)
                spikes, hidden_memories[i] = lif(current, hidden_memories[i])
                activations = dropout(hidden_memories[i])

            # Residual connection
            classifier_input = torch.cat([activations, current_input], dim=1,)
            logits = self.output_layer(classifier_input)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)


# ==============================================================================
# Spike Encoding
# ==============================================================================
def encode(X, coding):
    """Convert normalized sequence features to the configured SNN input coding."""
    coding = coding.lower()
    if coding in {"raw", "current"}:
        return X

    if coding == "rate":
        # return spikegen.rate(X.clamp(0.0, 1.0), num_steps=100)
        return spikegen.rate(X.clamp(0.0, 1.0), time_var_input=True)

    if coding == "latency":
        return spikegen.latency(X.clamp(0.0, 1.0), num_steps=100, normalize=True, linear=True)

    if coding == "delta":
        return torch.diff(X, dim=1, prepend=X[:, :1]).abs()

    raise ValueError(f"Unknown SNN_CODING: {coding}. Use 'raw', 'rate', or 'delta'.")


# ==============================================================================
# FNC: SNN Training Loop
# ==============================================================================
def train_snn(mdl, X_train, y_train, X_val, y_val, opt, loss_fnc, cfg, device):
    # ------------------------------------------
    # Init
    # ------------------------------------------
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # ------------------------------------------
    # Data
    # ------------------------------------------
    snn_train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    snn_val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32))
    loader_kwargs = {"batch_size": cfg["SNN_BATCH_SIZE"], "num_workers": cfg.get("NUM_WORKERS", 0), "pin_memory": device.type == "cuda"}
    snn_train_loader = DataLoader(snn_train_dataset, shuffle=True, **loader_kwargs)
    snn_val_loader = DataLoader(snn_val_dataset, shuffle=False, **loader_kwargs)

    # ------------------------------------------
    # Learning Rate Scheduler
    # ------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.get("SNN_LR_FACTOR", 0.5),
                                                           patience=cfg.get("SNN_LR_PATIENCE", 5),
                                                           min_lr=cfg.get("SNN_MIN_LR", 1e-6))

    # ------------------------------------------
    # Training Loop
    # ------------------------------------------
    for epoch in range(cfg["SNN_EPOCHS"]):
        # --------------------------
        # Training
        # --------------------------
        mdl.train()
        epoch_train_loss = 0.0

        for xb, yb in snn_train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            opt.zero_grad()
            logits = mdl(encode(xb, cfg["SNN_CODING"]))
            loss = loss_fnc(get_logits(logits, cfg["SNN_MODE"]), yb)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=cfg.get("SNN_GRAD_CLIP", 1.0))
            opt.step()
            epoch_train_loss += loss.item() * xb.size(0)

        # --------------------------
        # Validation
        # --------------------------
        mdl.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for xb, yb in snn_val_loader:
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                logits = mdl(encode(xb, cfg["SNN_CODING"]))
                loss = loss_fnc(get_logits(logits, cfg["SNN_MODE"]), yb)
                epoch_val_loss += loss.item() * xb.size(0)

        # --------------------------
        # Average losses
        # --------------------------
        epoch_train_loss /= max(len(snn_train_dataset), 1)
        epoch_val_loss /= max(len(snn_val_dataset), 1)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # --------------------------
        # Update learning rate
        # --------------------------
        scheduler.step(epoch_val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(
            f"SNN {epoch+1:3d}/{cfg['SNN_EPOCHS']} | "
            f"Train: {epoch_train_loss:.5f} | "
            f"Val: {epoch_val_loss:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        # --------------------------
        # Save best model
        # --------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(mdl.state_dict())
            no_improve = 0

        else:
            no_improve += 1
            if no_improve >= cfg.get("SNN_PATIENCE", cfg["SNN_EPOCHS"]):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # ------------------------------------------
    # Restore Best Model
    # ------------------------------------------
    if best_state is not None:
        mdl.load_state_dict(best_state)

    # ------------------------------------------
    # Save Checkpoint
    # ------------------------------------------
    torch.save(
        {
            "model_state_dict": mdl.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rate": opt.param_groups[0]["lr"],
        },
        cfg["SNN_SAVE_PATH"],
    )

    print(f"\nSaved SNN checkpoint to {cfg['SNN_SAVE_PATH']}")

    return mdl


# ==============================================================================
# FNC: SNN Testing Loop
# ==============================================================================
def test_snn(mdl, X_test, cfg, device, load_checkpoint=True):

    # ------------------------------------------
    # Load Checkpoint
    # ------------------------------------------
    if load_checkpoint:
        if not os.path.exists(cfg["SNN_SAVE_PATH"]):
            raise FileNotFoundError(f"Missing SNN checkpoint: {cfg['SNN_SAVE_PATH']}")

        checkpoint = torch.load(cfg["SNN_SAVE_PATH"], map_location=device, weights_only=False)
        mdl.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded SNN checkpoint from {cfg['SNN_SAVE_PATH']}")

    # ------------------------------------------
    # Evaluation
    # ------------------------------------------
    mdl.eval()

    with torch.no_grad():
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        logits = mdl(encode(X_test, cfg["SNN_CODING"]))
        logits = get_logits(logits, cfg["SNN_MODE"])
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(torch.int64)

    return {
        "logits": logits.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "predictions": predictions.cpu().numpy(),
    }


# ==============================================================================
# FNC: SNN Testing Loop
# ==============================================================================
def evaluate_classification(y_true, y_pred, verbose=True):
    """
    Evaluate a multi-label binary classification model.

    Parameters
    ----------
    y_true : ndarray (N, C)
        Ground truth labels.
    y_pred : ndarray (N, C)
        Predicted binary labels.
    verbose : bool
        Print metrics if True.

    Returns
    -------
    metrics : dict
        Dictionary containing overall and per-device metrics.
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    metrics = {}

    # Overall metrics
    metrics["accuracy"] = accuracy_score(y_true.flatten(), y_pred.flatten())
    metrics["precision"] = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    metrics["recall"] = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    metrics["f1"] = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["exact_match"] = (y_true == y_pred).all(axis=1).mean()

    if verbose:
        print("\nOverall classification metrics")
        print("-" * 40)
        print(f"Accuracy        : {metrics['accuracy']:.4f}")
        print(f"Precision       : {metrics['precision']:.4f}")
        print(f"Recall          : {metrics['recall']:.4f}")
        print(f"F1-score        : {metrics['f1']:.4f}")
        print(f"Hamming Loss    : {metrics['hamming_loss']:.4f}")
        print(f"Exact Match     : {metrics['exact_match']:.4f}")

    # Per-device metrics
    device_metrics = []

    if verbose:
        print("\nPer-device metrics")
        print("-" * 75)

    for i in range(y_true.shape[1]):
        m = {
            "accuracy": accuracy_score(y_true[:, i], y_pred[:, i]),
            "balanced_accuracy": balanced_accuracy_score(y_true[:, i], y_pred[:, i]),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }

        device_metrics.append(m)

        if verbose:
            print(
                f"Device {i+1:2d}: "
                f"ACC={m['accuracy']:.4f}  "
                f"BAL_ACC={m['balanced_accuracy']:.4f}  "
                f"P={m['precision']:.4f}  "
                f"R={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}"
            )

    metrics["devices"] = device_metrics

    return metrics


#######################################################################################################################
# Regression Functions
#######################################################################################################################
# ==============================================================================
# FNC: LSTM Model
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers=1, dropout=0.2, output_mode="s2p"):
        super().__init__()

        self.output_mode = output_mode.lower()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, sequence, features).")

        # outputs: (batch, seq, hidden)
        outputs, _ = self.lstm(x)

        # logits for every timestep
        logits = self.output_layer(outputs)

        if self.output_mode == "s2p":
            return logits[:, -1, :]

        elif self.output_mode == "s2s":
            return logits

        else:
            raise ValueError("output_mode must be 's2p' or 's2s'.")


# ==============================================================================
# FNC: LSTM Training Loop
# ==============================================================================
def train_lstm(mdl, X_train, y_train, X_val, y_val, opt, loss_fnc, cfg, device):
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
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32))

    loader_kwargs = {
        "batch_size": cfg["REG_BATCH_SIZE"],
        "num_workers": cfg.get("NUM_WORKERS", 0),
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # ------------------------------------------
    # Train
    # ------------------------------------------
    for epoch in range(cfg["REG_EPOCHS"]):
        # Init
        mdl.train()
        epoch_train_loss = 0.0

        # Loop over Batches
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            opt.zero_grad()
            y_hat = mdl(xb)
            loss = loss_fnc(y_hat, yb)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item() * xb.size(0)

        # Eval Validation data
        mdl.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                y_hat = mdl(xb)
                epoch_val_loss += loss_fnc(y_hat, yb).item() * xb.size(0)

        # Report loss
        epoch_train_loss /= max(len(train_dataset), 1)
        epoch_val_loss /= max(len(val_dataset), 1)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"REG {epoch + 1}/{cfg['REG_EPOCHS']} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(mdl.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.get("REG_PATIENCE", cfg["REG_EPOCHS"]):
                print(f"REG early stopping at epoch {epoch + 1}")
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
        cfg["REG_SAVE_PATH"],
    )
    print(f"Saved REG checkpoint to {cfg['REG_SAVE_PATH']}")

    return mdl


# ==============================================================================
# FNC: SNN Testing Loop
# ==============================================================================
def test_lstm(mdl, X_test, cfg, device, load_checkpoint=True):
    # ------------------------------------------
    # Load Checkpoint
    # ------------------------------------------
    if load_checkpoint:
        if not os.path.exists(cfg["REG_SAVE_PATH"]):
            raise FileNotFoundError(f"Missing REG checkpoint: {cfg['REG_SAVE_PATH']}")

        checkpoint = torch.load(cfg["REG_SAVE_PATH"], map_location=device, weights_only=False)
        mdl.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded REG checkpoint from {cfg['REG_SAVE_PATH']}")

    # ------------------------------------------
    # Evaluation
    # ------------------------------------------
    mdl.eval()

    with torch.no_grad():

        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_hat = mdl(X_test)

        threshold = float(np.asarray(cfg["THRESHOLD"]).squeeze())
        probabilities = torch.sigmoid(y_hat)
        predictions = (y_hat >= threshold).to(torch.int64)

    return {
        "logits": y_hat.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "predictions": predictions.cpu().numpy(),
    }
