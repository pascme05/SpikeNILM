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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from snntorch import spikegen
from snntorch import spikeplot as splt

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
    Y = Y[:, 1 + ID]
    return X, Y

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
    ID = 5
    Ntmax = 10000
    y_max = 500

    # ------------------------------------------
    # Preprocessing parameters
    # ------------------------------------------
    Nf = 15 # Number of Fourier coefficients to keep
    encoding = "rate" # 'rate' or 'temporal'
    num_steps = 100 # Number of time steps for spike encoding

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
    # Normalization
    # ------------------------------------------
    I_ac_max = np.max(np.abs(I_ac), axis=1, keepdims=True)
    V_ac_max = np.max(np.abs(V_ac), axis=1, keepdims=True)
    p_t = Y  / np.max(Y)  # Normalize target to [0, 1]

    # ==============================================================================
    # Preprocessing
    # ==============================================================================
    # ------------------------------------------
    # Fourier Transform
    # ------------------------------------------
    If_ac = np.abs(np.fft.rfft(I_ac, axis=1))[:, 0:Nf + 1] / I_ac.shape[1] * 2
    Vf_ac = np.abs(np.fft.rfft(V_ac, axis=1))[:, 0:Nf + 1] / V_ac.shape[1] * 2
    F_ac = np.concatenate([Vf_ac, If_ac], axis=1)

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
    If_ac_norm = If_ac / (I_ac_max + 1e-8)
    Vf_ac_norm = Vf_ac / (V_ac_max + 1e-8)
    
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
    # Data Loaders
    # ------------------------------------------
    test = 1


#######################################################################################################################
# Run Code
#######################################################################################################################
if __name__ == "__main__":
    main()
