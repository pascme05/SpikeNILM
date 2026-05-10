#######################################################################################################################
#######################################################################################################################
# Title:        Scripting for testing a small SNN for grid simulation
# Topic:        ML & DL Smart Grid
# File:         smallSNN
# Date:         08.10.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
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
import warnings
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import snntorch as snn
from snntorch import spikegen
from snntorch import spikeplot as splt
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

#######################################################################################################################
# Helper Functions
#######################################################################################################################
# ==============================================================================
# Loading REDD Dataset
# ==============================================================================
def test_FFT_plot(If_ac, Vf_ac):
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_axis = np.arange(If_ac.shape[0])
    coeff_axis = np.arange(If_ac.shape[1])

    current_plot = axes[0].pcolormesh(
        time_axis,
        coeff_axis,
        If_ac.T,
        shading="auto",
        cmap="viridis",
    )
    axes[0].set_title("Current Fourier Coefficients")
    axes[0].set_ylabel("Fourier coefficient")
    fig.colorbar(current_plot, ax=axes[0], label="Amplitude")

    voltage_plot = axes[1].pcolormesh(
        time_axis,
        coeff_axis,
        Vf_ac.T,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_title("Voltage Fourier Coefficients")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Fourier coefficient")
    fig.colorbar(voltage_plot, ax=axes[1], label="Amplitude")

    plt.tight_layout()
    plt.show()

# ==============================================================================
# Loading REDD Dataset
# ==============================================================================
def load_data(mat_file, ID=0, maxLen=10000):
    data = loadmat(mat_file)
    X = data["input"]
    Y = data["output"]

    if maxLen is not None and maxLen > 0:
        X = X[:maxLen]
        Y = Y[:maxLen]

    X = X[:, 2:277, :]

    if ID == -1:
        Y = Y[:, 1:]
    else:   
        Y = Y[:, 1 + ID]

    return X, Y


class SmallSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        batch_size, num_steps, _ = x.shape
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        spk_rec = []
        mem_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=1), torch.stack(mem_rec, dim=1)

#######################################################################################################################
# Main Code
#######################################################################################################################
def main():
    # ==============================================================================
    # Parameters
    # ==============================================================================
    # ------------------------------------------
    # Dataset parameters
    # ------------------------------------------
    mat_file = "data/redd3HF.mat"
    ID = -1
    Ntmax = 10000
    y_max = 500
    th = 50

    # ------------------------------------------
    # Preprocessing parameters
    # ------------------------------------------
    Nf = 15 # Number of Fourier coefficients to keep
    encoding = "rate" # 'rate' or 'temporal'
    num_steps = 100 # Number of time steps for spike encoding

    # ------------------------------------------
    # SNN parameters
    # ------------------------------------------
    num_epochs = 5
    split = 0.8
    batch_size=256
    lr=1e-3

    # ------------------------------------------
    # Other parameters
    # ------------------------------------------
    plotting = 1


    # ==============================================================================
    # Loading REDD Dataset
    # ==============================================================================
    # ------------------------------------------
    # Loading the dataset
    # ------------------------------------------
    X, Y = load_data(mat_file, ID=ID, maxLen=Ntmax)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # ------------------------------------------
    # Selecting
    # ------------------------------------------
    I_ac = np.squeeze(X[:,:,1])
    V_ac = np.squeeze(X[:,:,0])

    # ------------------------------------------
    # Normalization Values
    # ------------------------------------------
    I_ac_max = np.max(np.abs(I_ac), axis=1, keepdims=True)
    V_ac_max = np.max(np.abs(V_ac), axis=1, keepdims=True)
    power_scale = float(np.max(Y)) if np.max(Y) > 0 else 1.0

    # ==============================================================================
    # Preprocessing
    # ==============================================================================
    # ------------------------------------------
    # Fourier Transform
    # ------------------------------------------
    If_ac = np.abs(np.fft.rfft(I_ac, axis=1))[:, 0:Nf + 1] / I_ac.shape[1] * 2
    Vf_ac = np.abs(np.fft.rfft(V_ac, axis=1))[:, 0:Nf + 1] / V_ac.shape[1] * 2

    # ------------------------------------------
    # Label Binarization
    # ------------------------------------------
    c_t = Y
    c_t(c_t < th) = 0
    c_t(c_t >= th) = 1

    # ------------------------------------------
    # RMS Values
    # ------------------------------------------
    Vrms = np.sqrt(np.mean(V_ac**2, axis=1))
    Irms = np.sqrt(np.mean(I_ac**2, axis=1))
    P_agg = If_ac[:, 1] * Vf_ac[:, 1] / 2
    S_agg = Vrms * Irms
    Q_agg = np.sqrt(S_agg**2 - P_agg**2)

    # ------------------------------------------
    # Transformation
    # ------------------------------------------
    If_ac = np.log(If_ac + 1)
    Vf_ac = np.log(Vf_ac + 1)
    I_ac_max = np.max(np.abs(If_ac))
    V_ac_max = np.max(np.abs(Vf_ac))

    # ------------------------------------------
    # Normalization
    # ------------------------------------------
    # Input normalization to [0, 1]
    If_ac_norm = If_ac / (I_ac_max + 1e-8)
    Vf_ac_norm = Vf_ac / (V_ac_max + 1e-8)

    # Target normalization to [0, 1]
    p_t = Y / power_scale  # Normalize target to [0, 1]
    
    # Test plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_axis = np.arange(If_ac_norm.shape[0])
    coeff_axis = np.arange(If_ac_norm.shape[1])

    current_plot = axes[0].pcolormesh(
        time_axis,
        coeff_axis,
        If_ac_norm.T,
        shading="auto",
        cmap="viridis",
    )
    axes[0].set_title("Normalized Current Fourier Coefficients")
    axes[0].set_ylabel("Fourier coefficient")
    fig.colorbar(current_plot, ax=axes[0], label="Normalized amplitude")

    voltage_plot = axes[1].pcolormesh(
        time_axis,
        coeff_axis,
        Vf_ac_norm.T,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_title("Normalized Voltage Fourier Coefficients")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Fourier coefficient")
    fig.colorbar(voltage_plot, ax=axes[1], label="Normalized amplitude")

    plt.tight_layout()
    plt.show()
    """

    # ------------------------------------------
    # Spike Encoding Input
    # ------------------------------------------
    if encoding == "rate":
        # Rate-based encoding
        Is_ac = spikegen.rate(torch.from_numpy(If_ac_norm), num_steps=num_steps)
        Vs_ac = spikegen.rate(torch.from_numpy(Vf_ac_norm), num_steps=num_steps)
    elif encoding == "temporal":
        # Temporal-based encoding
        pass
    
    # ------------------------------------------
    # Spike Encoding Target
    # ------------------------------------------
    if encoding == "rate":
        # Rate-based encoding
        ps_t = spikegen.rate(torch.from_numpy(p_t), num_steps=num_steps)
        cs_t = spikegen.rate(torch.from_numpy(cs_t), num_steps=num_steps)
    elif encoding == "temporal":
        # Temporal-based encoding
        pass
    
    """
    sample_idx = 0
    input_spikes = torch.cat((Vs_ac[:, sample_idx, :], Is_ac[:, sample_idx, :]), dim=1)
    target_spikes = ps_t[:, sample_idx].reshape(num_steps, 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    splt.raster(input_spikes, axes[0], s=1.5, c="black")
    axes[0].set_title("Input Spikes for One Time Domain Sample")
    axes[0].set_ylabel("Input channel")

    splt.raster(target_spikes, axes[1], s=1.5, c="black")
    axes[1].set_title("Target Spikes for One Time Domain Sample")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Target channel")

    plt.tight_layout()
    plt.show()
    """

    # ==============================================================================
    # Training SNN Network
    # ==============================================================================
    # ------------------------------------------
    # Settings
    # ------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------
    # Data Loaders
    # ------------------------------------------
    input_spikes = torch.cat((Vs_ac, Is_ac), dim=2).permute(1, 0, 2).float()
    target_spikes = cs_t.transpose(0, 1).unsqueeze(-1).float()
    split_idx = int(split * input_spikes.shape[0])
    train_dataset = TensorDataset(input_spikes[:split_idx], target_spikes[:split_idx])
    val_dataset = TensorDataset(input_spikes[split_idx:], target_spikes[split_idx:])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)


    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = model_class(input_dim=x.shape[-1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            spk_rec, mem_rec = model(xb)  # mem_rec: [B, 20]
            loss = loss_fn(mem_rec, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                _, mem_rec = model(xb)
                val_loss += loss_fn(mem_rec, yb).item() * xb.size(0)

        train_loss /= len(train_dataset)
        val_loss   /= max(len(val_dataset), 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # -------------------------------------------------
    # Testing / Evaluation
    # -------------------------------------------------
    model.eval()

    all_spk, all_logits, all_targets = [], [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)

            spk_rec, mem_rec = model(xb)

            all_spk.append(spk_rec.cpu())
            all_logits.append(mem_rec.cpu())
            all_targets.append(yb.cpu())

    if len(all_targets) == 0:
        print("Empty validation set.")
        return None

    spk = torch.cat(all_spk)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    y_true = targets.numpy().astype(int).ravel()
    y_pred = preds.numpy().astype(int).ravel()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("\nClassification Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # -------------------------------------------------
    # Power Decoding
    # -------------------------------------------------
    true_power = Y[split_idx:]

    pred_power = probs.mean(dim=1).numpy() * power_scale

    mae = np.mean(np.abs(pred_power - true_power))
    rmse = np.sqrt(np.mean((pred_power - true_power) ** 2))
    corr = np.corrcoef(pred_power, true_power)[0, 1] if len(true_power) > 1 else np.nan

    print("\nPower Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Corr: {corr:.4f}")

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    if plotting:
        plt.figure(figsize=(12,5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()

#######################################################################################################################
# Run Code
#######################################################################################################################
if __name__ == "__main__":
    main()
