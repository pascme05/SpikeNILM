import torch
import math
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# -----------------------------
# Load .mat data
# -----------------------------
def load_data(mat_file, maxLen=10000):
    data = loadmat(mat_file)

    X = data["input"]  # [Nt, 2+W, F]
    Y = data["output"]  # [Nt, D]

    return X[:maxLen], Y[:maxLen]


# -----------------------------
# Convert to tensors
# -----------------------------
def prepare_dataset(X, Y, thres=50.0, device_id=0):
    Nt, total_len, F = X.shape
    W = total_len - 2

    # Extract waveform
    signal = X[:, 2:, :]  # [Nt, W, 2]

    i = signal[:, :, 1]
    v = signal[:, :, 0]

    # Preprocess → [Nt, W, 3]
    x = preprocess(torch.tensor(i), torch.tensor(v))

    # Select device power
    power = Y[:, 1 + device_id]  # [Nt]

    # Spike encoding input
    x = harmonic_spike_encoding(x)

    # Convert to spikes output
    y = (power > thres).astype(np.float32)
    y = torch.tensor(y).unsqueeze(1)  # [Nt, 1]

    return x.float(), y.float()


# -----------------------------
# Create sequences of windows
# -----------------------------
def create_sequences(x, y, seq_len=20):
    xs, ys = [], []

    for i in range(len(x) - seq_len):
        xs.append(x[i:i+seq_len])
        ys.append(y[i:i+seq_len])

    return torch.stack(xs), torch.stack(ys)


# -----------------------------
# Split dataset
# -----------------------------
def train_test_split(x, y, split=0.8):
    N = len(x)
    idx = int(N * split)

    return x[:idx], y[:idx], x[idx:], y[idx:]

# -----------------------------
# Pre-processing
# -----------------------------
def preprocess(i, v):
    """
    i, v: [batch, time]
    returns: [batch, time, features]
    """
    # normalize per cycle
    i = i / (i.abs().max(dim=1, keepdim=True)[0] + 1e-6)
    v = v / (v.abs().max(dim=1, keepdim=True)[0] + 1e-6)

    # instantaneous power
    p = i * v

    x = torch.stack([i, v, p], dim=2)
    return x

# -----------------------------
# Visualization
# -----------------------------
def plot_sequences(inputs, targets, predictions):
    fig, axes = plt.subplots(targets.shape[1], 1, figsize=(12, 8))

    for i in range(0, targets.shape[1]):
        if targets.shape[1] == 1:
            ax = axes
        else:
            ax = axes[i]
        time_steps = targets.shape[0]

        ax.plot(range(time_steps), targets[:, i].numpy(), label='Target', marker='s', linestyle='-')
        ax.plot(range(time_steps), predictions[:, i].numpy(), label='Prediction', marker='^', linestyle='-.')

        ax.set_title(f'Example {i+1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Spike')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

# -----------------------------
# Harmonic Spike Encoding
# -----------------------------
def harmonic_spike_encoding(x, num_harmonics=8, base_freq=60, fs=16500, threshold=0.2):
    """
    x: [batch, time, features]  (i, v, p)
    returns: [batch, time, spike_features]
    """
    batch, T, F = x.shape
    device = x.device

    t = torch.arange(T, device=device) / fs
    spikes = []

    for k in range(1, num_harmonics + 1):
        freq = k * base_freq

        cos_wave = torch.cos(2 * math.pi * freq * t).view(1, T, 1)
        sin_wave = torch.sin(2 * math.pi * freq * t).view(1, T, 1)

        real = x * cos_wave
        imag = x * sin_wave

        energy = torch.sqrt(real**2 + imag**2)

        # normalize per sample
        energy = energy / (energy.amax(dim=1, keepdim=True) + 1e-6)

        spike = (energy > threshold).float()
        spikes.append(spike)

    return torch.cat(spikes, dim=2)