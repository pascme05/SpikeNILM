import numpy as np
import torch
from scipy.io import loadmat
from snntorch import spikegen


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


def extract_features(X, n_harmonics=15):
    V = X[:, :, 0]
    I = X[:, :, 1]

    V_fft = np.abs(np.fft.rfft(V, axis=1))[:, 1:n_harmonics + 1]
    I_fft = np.abs(np.fft.rfft(I, axis=1))[:, 1:n_harmonics + 1]

    rms_v = np.sqrt(np.mean(V ** 2, axis=1, keepdims=True))
    rms_i = np.sqrt(np.mean(I ** 2, axis=1, keepdims=True))
    peak_v = np.max(np.abs(V), axis=1, keepdims=True)
    peak_i = np.max(np.abs(I), axis=1, keepdims=True)
    real_power = np.mean(V * I, axis=1, keepdims=True)
    apparent_power = rms_v * rms_i

    return np.concatenate(
        [V_fft, I_fft, rms_v, rms_i, peak_v, peak_i, real_power, apparent_power],
        axis=1,
    ).astype(np.float32)


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
