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


def create_sequences(features, targets, sequence_length, stride=1):
    """Create sliding feature windows and align each one to its final target."""
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape [n_samples, n_features].")
    if len(features) != len(targets):
        raise ValueError("features and targets must contain the same number of samples.")
    if sequence_length < 1 or sequence_length > len(features):
        raise ValueError("sequence_length must be between 1 and the number of samples.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    windows = np.lib.stride_tricks.sliding_window_view(
        features, window_shape=sequence_length, axis=0
    )
    windows = np.moveaxis(windows, -1, 1)[::stride].copy()
    aligned_targets = targets[sequence_length - 1::stride].copy()
    return windows, aligned_targets


#######################################################################################################################
# SNN Functions
#######################################################################################################################
# ==============================================================================
# SNN Model
# ==============================================================================
class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, beta=0.95):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_layers = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        layer_input_size = input_size
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(layer_input_size, hidden_size))
            self.hidden_lifs.append(snn.Leaky(beta=beta))
            layer_input_size = hidden_size
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_lif = snn.Leaky(beta=beta)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("SNN input must have shape [batch, sequence, features].")

        hidden_memories = [lif.init_leaky() for lif in self.hidden_lifs]
        output_memory = self.output_lif.init_leaky()
        spks = []
        memories = []
        for t in range(x.shape[1]):
            activations = x[:, t]
            for index, (layer, lif) in enumerate(zip(self.hidden_layers, self.hidden_lifs)):
                activations, hidden_memories[index] = lif(layer(activations), hidden_memories[index])
            output_spikes, output_memory = self.output_lif(
                self.output_layer(activations), output_memory
            )
            spks.append(output_spikes)
            memories.append(output_memory)
        return torch.stack(spks, dim=1), torch.stack(memories, dim=1)


# ==============================================================================
# Spike Encoding
# ==============================================================================
def encode(X, coding):
    """Convert normalized sequence features to the configured SNN input coding."""
    coding = coding.lower()
    if coding in {"raw", "current"}:
        return X

    if coding == "rate":
        if spikegen is None:
            return torch.bernoulli(X.clamp(0.0, 1.0))
        return spikegen.rate(X.clamp(0.0, 1.0), num_steps=100)

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
    snn_train_dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    snn_val_dataset = TensorDataset(
        torch.as_tensor(X_val, dtype=torch.float32),
        torch.as_tensor(y_val, dtype=torch.float32),
    )
    loader_kwargs = {
        "batch_size": cfg["SNN_BATCH_SIZE"],
        "num_workers": cfg.get("NUM_WORKERS", 0),
        "pin_memory": device.type == "cuda",
    }
    snn_train_loader = DataLoader(snn_train_dataset, shuffle=True, **loader_kwargs)
    snn_val_loader = DataLoader(snn_val_dataset, shuffle=False, **loader_kwargs)

    # ------------------------------------------
    # Train
    # ------------------------------------------
    for epoch in range(cfg["SNN_EPOCHS"]):
        # Init
        mdl.train()
        epoch_train_loss = 0.0

        # Loop over Batches
        for xb, yb in snn_train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            opt.zero_grad()
            _, mem_rec = mdl(encode(xb, cfg["SNN_CODING"]))
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
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                _, mem_rec = mdl(encode(xb, cfg["SNN_CODING"]))
                logits = mem_rec.mean(dim=1)
                epoch_val_loss += loss_fnc(logits, yb).item() * xb.size(0)

        # Report loss
        epoch_train_loss /= max(len(snn_train_dataset), 1)
        epoch_val_loss /= max(len(snn_val_dataset), 1)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"SNN {epoch + 1}/{cfg['SNN_EPOCHS']} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(mdl.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.get("SNN_PATIENCE", cfg["SNN_EPOCHS"]):
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
        cfg["SNN_SAVE_PATH"],
    )
    print(f"Saved SNN checkpoint to {cfg['SNN_SAVE_PATH']}")

    return mdl


# ==============================================================================
# FNC: SNN Testing Loop
# ==============================================================================
def test_snn(mdl, X_test, cfg, device, load_checkpoint=True):
    # ------------------------------------------
    # Description
    # ------------------------------------------

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
    # Eval
    # ------------------------------------------
    # Init
    mdl.eval()

    # Calc
    with torch.no_grad():
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        spk_all, mem_all = mdl(encode(X_test, cfg["SNN_CODING"]))
        logits_all = mem_all.mean(dim=1)
        probabilities = torch.sigmoid(logits_all)
        if cfg.get("SNN_EVAL_MODE", "membrane") == "spike_count":
            predictions = (spk_all.mean(dim=1) >= 0.5).to(torch.int64)
        elif cfg.get("SNN_EVAL_MODE") == "spike_any":
            predictions = (spk_all.any(dim=1)).to(torch.int64)
        else:
            predictions = (probabilities >= 0.5).to(torch.int64)

    return {
        "predictions": predictions.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
    }


#######################################################################################################################
# Regression Functions
#######################################################################################################################
# ==============================================================================
# FNC: LSTM Training Loop
# ==============================================================================
