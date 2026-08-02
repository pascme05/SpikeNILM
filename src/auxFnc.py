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
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

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
