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
# ─── Standard library ────────────────────────────────────────────────────────
import numpy as np
from scipy.io import loadmat

# ─── Third-party ─────────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    torch = None

try:
    from snntorch import spikegen
except ImportError:
    spikegen = None

#######################################################################################################################
# Functions
#######################################################################################################################
def load_data(mat_file, ID=0, maxLen=10000):
    data = loadmat(mat_file)
    X = data["input"]
    Y = data["output"]

    if maxLen is not None and maxLen > 0:
        X = X[:maxLen]
        Y = Y[:maxLen]

    X = X[:, 2:277, :]
    Y = Y[:, 1 + ID]
    return X, Y


def load_data_multi(mat_file, device_ids, maxLen=-1):
    """Load X and Y for multiple device IDs in a single file read.

    Returns:
        X: ndarray [n_samples, 275, 2]
        Y_devices: dict {device_id: ndarray [n_samples]}
    """
    data = loadmat(mat_file)
    X = data["input"]
    Y_all = data["output"]

    if maxLen is not None and maxLen > 0:
        X = X[:maxLen]
        Y_all = Y_all[:maxLen]

    X = X[:, 2:277, :]
    Y_devices = {dev_id: Y_all[:, 1 + dev_id] for dev_id in device_ids}
    return X, Y_devices


def select_input_channels(X, channels=None):
    """Select raw waveform channels by name.

    Args:
        X: ndarray [n_samples, seq_len, 2] with channel order [voltage, current]
        channels: iterable of 'voltage' and/or 'current'
    """
    channel_map = {"voltage": 0, "current": 1}
    if channels is None:
        channels = ("voltage", "current")

    indices = []
    for channel in channels:
        if channel not in channel_map:
            raise ValueError(f"Unknown raw input channel: {channel}. Use 'voltage' and/or 'current'.")
        indices.append(channel_map[channel])

    if not indices:
        raise ValueError("At least one raw input channel must be selected.")

    return X[:, :, indices].copy()


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


def compute_feature_deltas(X, mode="absolute"):
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


def prepare_input(X_flat, n_samples):
    X_flat = X_flat.reshape(n_samples, -1).astype(np.float32)
    x_min = X_flat.min(axis=0, keepdims=True)
    x_max = X_flat.max(axis=0, keepdims=True)
    return (X_flat - x_min) / (x_max - x_min + 1e-8)


def create_sequences(X, Y, seq_len, stride=1):
    X_seq = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    X_seq = np.moveaxis(X_seq, -1, 1)
    Y_seq = Y[seq_len - 1:]
    return X_seq[::stride].copy(), Y_seq[::stride].copy()


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


def encode_spikes(X_batch, coding, device):
    if spikegen is None:
        raise ImportError("snntorch is required for encode_spikes but is not installed in this environment.")

    batch_size, seq_len, feature_count = X_batch.shape

    if coding == "rate":
        spike_input = spikegen.rate(X_batch.reshape(batch_size * seq_len, feature_count), num_steps=1)
        spike_input = spike_input.squeeze(0).reshape(batch_size, seq_len, feature_count).permute(1, 0, 2)
    elif coding == "latency":
        spike_input = spikegen.latency(
            X_batch.permute(1, 0, 2),
            num_steps=seq_len,
            tau=5.0,
            threshold=0.01,
            normalize=True,
            linear=True,
        )
    elif coding == "delta":
        spike_input = spikegen.delta(X_batch.permute(1, 0, 2), threshold=0.1, off_spike=True)
    elif coding == "raw":
        spike_input = X_batch.permute(1, 0, 2)
    else:
        raise ValueError(f"Unknown coding: {coding}. Use 'rate', 'latency', 'delta', or 'raw'.")

    return spike_input.to(device)


def build_target_spikes(labels, num_steps, num_classes, on_rate, off_rate):
    if torch is None or spikegen is None:
        raise ImportError("torch and snntorch are required for build_target_spikes but are not installed.")

    batch_size = labels.size(0)
    targets = torch.zeros(num_steps, batch_size, num_classes, device=labels.device)
    on_pattern, _ = spikegen.target_rate_code(num_steps=num_steps, rate=on_rate)
    off_pattern, _ = spikegen.target_rate_code(num_steps=num_steps, rate=off_rate)
    on_pattern = on_pattern.to(labels.device)
    off_pattern = off_pattern.to(labels.device)

    for class_id in range(num_classes):
        mask = labels == class_id
        targets[:, mask, class_id] = on_pattern.unsqueeze(1)
        targets[:, ~mask, class_id] = off_pattern.unsqueeze(1)

    return targets


def snn_predict(spk_rec, mem_rec, eval_mode):
    if eval_mode == "membrane":
        return mem_rec.mean(dim=0).argmax(dim=1)
    if eval_mode == "spike_count":
        return spk_rec.sum(dim=0).argmax(dim=1)
    if eval_mode == "spike_any":
        return (spk_rec.sum(dim=0)[:, 1] > 0).long()
    raise ValueError(f"Unknown SNN_EVAL_MODE: {eval_mode}")
