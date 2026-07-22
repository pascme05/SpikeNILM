#######################################################################################################################
#######################################################################################################################
# Title:        Spike NILM
# Topic:        Non-intrusive load monitoring
# File:         main
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
Non-Intrusive Load Monitoring (NILM) pipeline for the REDD high-frequency dataset.
"""

#######################################################################################################################
# Import external libs
#######################################################################################################################
# ===============================================================================
# Internal
# ===============================================================================
from auxFnc import (
    SNNModel,
    balance_sequences,
    binarize_output,
    create_sequences,
    downsample,
    extract_features,
    feature_deltas,
    filter_output,
    load_data,
    test_snn,
    train_snn,
)
from config import build_config_default

# ===============================================================================
# External
# ===============================================================================
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#######################################################################################################################
# Helper Functions
#######################################################################################################################


#######################################################################################################################
# Config
#######################################################################################################################
config = build_config_default()


#######################################################################################################################
# Main
#######################################################################################################################
def main(config):
    # ===============================================================================
    # Description
    # ===============================================================================
    """
    This function performs the Spike NILM workflow.
    """

    # ===============================================================================
    # Load Config
    # ===============================================================================
    # ------------------------------------------
    # Setup
    # ------------------------------------------
    config = dict(config)
    project_root = Path(__file__).resolve().parents[1]
    if config["SNN_MODEL_TYPE"].lower() != "snn":
        raise NotImplementedError("mainSNN currently implements SNN_MODEL_TYPE='snn' only.")

    # ------------------------------------------
    # Machine Hardware
    # ------------------------------------------
    # Device
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

    # ===============================================================================
    # STAGE 1: Data Loading and Pre-processing
    # ===============================================================================
    # ------------------------------------------
    # Load Raw data
    # ------------------------------------------
    data_path = project_root / "data" / f"{config['NAME']}.mat"
    X_raw, y_raw = load_data(data_path, config["DEVICE_IDS"], maxLen=config["MAX_LEN"])
    N = X_raw.shape[0]  # Samples
    W = X_raw.shape[1]  # Window size
    F = X_raw.shape[2]  # Features
    C = y_raw.shape[1]  # Channels, aka devices

    # ------------------------------------------
    # Pre-Processing
    # ------------------------------------------
    # Output filtering
    y_t = filter_output(y_raw, method=config.get("FILTER_METHOD", "moving_average"), window=config.get("FILTER_WINDOW", 5))

    # Convert binary states
    s_t = binarize_output(y_t, threshold=config.get("THRESHOLD", 50))

    # Downsample data
    X_t, y_t = downsample(X_raw, y_t, rate=config.get("DOWNSAMPLE", 1))
    _, s_t = downsample(X_raw, s_t, rate=config.get("DOWNSAMPLE", 1))

    # ===============================================================================
    # STAGE 2: Feature Extraction
    # ===============================================================================
    # ------------------------------------------
    # Frequency Transform
    # ------------------------------------------
    Xf_t, f_names = extract_features(
        X_t,
        n_harmonics=config["N_HARMONICS"],
        selector=config.get("SNN_FEATURE_SELECTOR", config["FEATURE_SELECTOR"]),
        return_names=True,
    )
    del X_raw, X_t, y_raw
    gc.collect()

    # ------------------------------------------
    # Delta Features
    # ------------------------------------------
    if config.get("SNN_INPUT_TRANSFORM", "delta") == "delta":
        dXf_t = feature_deltas(Xf_t, mode=config.get("SNN_DELTA_MODE", "absolute"))
    elif config.get("SNN_INPUT_TRANSFORM") == "absolute":
        dXf_t = Xf_t
    else:
        raise ValueError("SNN_INPUT_TRANSFORM must be 'delta' or 'absolute'.")
    dy_t = feature_deltas(y_t, mode="absolute")
    ds_t = feature_deltas(s_t, mode="absolute")

    # ------------------------------------------
    # Split data
    # ------------------------------------------
    # Init
    split_idx = int(dXf_t.shape[0] * config["SPLIT"])

    # Train
    dXf_t_train = dXf_t[:split_idx]
    y_t_train = y_t[:split_idx]
    dy_t_train = dy_t[:split_idx]
    s_t_train = s_t[:split_idx]
    ds_t_train = ds_t[:split_idx]

    # Test
    dXf_t_test = dXf_t[split_idx:]
    y_t_test = y_t[split_idx:]
    dy_t_test = dy_t[split_idx:]
    s_t_test = s_t[split_idx:]
    ds_t_test = ds_t[split_idx:]

    # ------------------------------------------
    # Normalization
    # ------------------------------------------
    # Input
    xmin = dXf_t_train.min(axis=0, keepdims=True)
    xmax = dXf_t_train.max(axis=0, keepdims=True)
    dXf_t_train_norm = (dXf_t_train - xmin) / (xmax - xmin + 1e-8)
    dXf_t_test_norm = (dXf_t_test - xmin) / (xmax - xmin + 1e-8)

    # Output
    xmin = y_t_train.min(axis=0, keepdims=True)
    xmax = y_t_train.max(axis=0, keepdims=True)
    y_t_train_norm = (y_t_train - xmin) / (xmax - xmin + 1e-8)
    y_t_test_norm = (y_t_test - xmin) / (xmax - xmin + 1e-8)

    # ------------------------------------------
    # Balance
    # ------------------------------------------
    train_targets = ds_t_train if config["USE_DERIVATIVE"] else s_t_train
    test_targets = ds_t_test if config["USE_DERIVATIVE"] else s_t_test
    dXf_t_train_norm, train_targets = create_sequences(
        dXf_t_train_norm,
        train_targets,
        sequence_length=config["SNN_SEQ_LEN"],
        stride=config["STRIDE"],
    )
    dXf_t_test_norm, test_targets = create_sequences(
        dXf_t_test_norm,
        test_targets,
        sequence_length=config["SNN_SEQ_LEN"],
        stride=config["STRIDE"],
    )
    if config["BALANCE_DATA"]:
        dXf_t_train_norm, train_targets = balance_sequences(dXf_t_train_norm, train_targets)
    print(f"Training sequences: {len(dXf_t_train_norm):,}; validation sequences: {len(dXf_t_test_norm):,}")

    # ===============================================================================
    # STAGE 3: SNN Classification
    # ===============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    # Data

    # Model
    config["SNN_SAVE_PATH"] = str(
        project_root / config["SNN_SAVE_PATH"].format(device_id=config["DEVICE_IDS"][0])
    )
    mdlSNN = SNNModel(
        input_size=dXf_t_train_norm.shape[-1],
        hidden_size=config["SNN_HIDDEN_SIZE"],
        output_size=C,
        num_layers=config["SNN_NUM_LAYERS"],
        beta=config["SNN_BETA"],
    ).to(device)
    opt = torch.optim.Adam(mdlSNN.parameters(), lr=config["SNN_LR"])
    loss_fn = nn.BCEWithLogitsLoss()

    # ------------------------------------------
    # Train
    # ------------------------------------------
    if config["SNN_DO_TRAIN"]:
        mdlSNN = train_snn(
            mdlSNN,
            dXf_t_train_norm,
            train_targets,
            dXf_t_test_norm,
            test_targets,
            opt,
            loss_fn,
            config,
            device,
        )

    # ------------------------------------------
    # Inference
    # ------------------------------------------
    evaluation = test_snn(mdlSNN, dXf_t_test_norm, config, device, load_checkpoint=True)
    y_pred = evaluation["predictions"]

    # ===============================================================================
    # STAGE 4: LSTM Regression
    # ===============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------

    # ------------------------------------------
    # Setup Data
    # ------------------------------------------

    # ------------------------------------------
    # Train
    # ------------------------------------------

    # ===============================================================================
    # STAGE 5: Prediction and Accuracy
    # ===============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------

    # ------------------------------------------
    # Load Model
    # ------------------------------------------

    # ------------------------------------------
    # Inference
    # ------------------------------------------

    # ------------------------------------------
    # De-Normalization
    # ------------------------------------------

    # ------------------------------------------
    # Calc Accuracy
    # ------------------------------------------
    accuracy = float((y_pred == test_targets.astype(int)).mean())
    print(f"Validation accuracy: {accuracy:.4f}")

    # ===============================================================================
    # STAGE 6: Plotting
    # ===============================================================================
    # if config["PLOT_SNN"]:
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    sample_count = min(1_000, len(test_targets))
    fig, axis = plt.subplots(figsize=(12, 3))
    axis.plot(test_targets[:sample_count, 0], label="target", linewidth=1.2)
    axis.plot(y_pred[:sample_count, 0], label="prediction", linewidth=1.0, alpha=0.8)
    axis.set(xlabel="Test sequence", ylabel="ON state", ylim=(-0.1, 1.1))
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"snn_predictions_dev{config['DEVICE_IDS'][0]}.png", dpi=150)
    plt.close(fig)

    return {
        "accuracy": accuracy,
        "predictions": y_pred,
        "targets": test_targets,
        "probabilities": evaluation["probabilities"],
    }


#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == "__main__":
    main(config)
