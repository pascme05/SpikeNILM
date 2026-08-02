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
    LSTMModel,
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
    train_lstm,
    test_lstm,
    check_hardware,
    evaluate_classification,
)
from config import build_config_default

# ===============================================================================
# External
# ===============================================================================
from pathlib import Path
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
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

    # ------------------------------------------
    # Path
    # ------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / f"{config['NAME']}.mat"
    config["SNN_SAVE_PATH"] = str(project_root / config["SNN_SAVE_PATH"].format(device_id=config["DEVICE_IDS"][0]))
    config["REG_SAVE_PATH"] = str(project_root / config["REG_SAVE_PATH"].format(device_id=config["DEVICE_IDS"][0]))

    # ------------------------------------------
    # Machine Hardware
    # ------------------------------------------
    device = check_hardware(config)

    # ===============================================================================
    # STAGE 1: Data Loading and Pre-processing
    # ===============================================================================
    # ------------------------------------------
    # Load Raw data
    # ------------------------------------------
    # Loading
    X_raw, y_raw = load_data(data_path, config["DEVICE_IDS"], maxLen=config["MAX_LEN"])

    # Variables
    N = X_raw.shape[0]  # Samples
    W = X_raw.shape[1]  # Window size
    F = X_raw.shape[2]  # Features
    C = y_raw.shape[1]  # Channels, aka devices
    sampling_period = float(config["T_SAMPLING"])
    time_sequence = np.arange(N, dtype=np.float64) * sampling_period
    time_raw_cycles = time_sequence[:, None] + np.arange(W, dtype=np.float64) * sampling_period / W

    # ------------------------------------------
    # Pre-Processing
    # ------------------------------------------
    # Output filtering
    y_t = filter_output(y_raw, method=config.get("FILTER_METHOD", "moving_average"),
                        window=config.get("FILTER_WINDOW", 5))

    # Convert binary states
    s_t = binarize_output(y_t, threshold=config.get("THRESHOLD", 50))

    # Downsample data
    downsample_rate = config.get("DOWNSAMPLE", 1)
    X_t, y_t = downsample(X_raw, y_t, rate=downsample_rate)
    _, s_t = downsample(X_raw, s_t, rate=downsample_rate)
    time_sequence = time_sequence[::downsample_rate]
    time_raw_cycles = time_raw_cycles[::downsample_rate]
    time_raw = time_raw_cycles.reshape(-1)

    # ------------------------------------------
    # Normalization
    # ------------------------------------------
    # Init
    split_idx_train = int(X_t.shape[0] * config["SPLIT_TRAIN"])
    split_idx_val = int(X_t.shape[0] * config["SPLIT_VAL"])

    # Input
    xmin = X_t[split_idx_val:split_idx_train].min(axis=0, keepdims=True)
    xmax = X_t[split_idx_val:split_idx_train].max(axis=0, keepdims=True)
    X_t = (X_t - xmin) / (xmax - xmin + 1e-8)

    # Output
    ymin = y_t[split_idx_val:split_idx_train].min(axis=0, keepdims=True)
    ymax = y_t[split_idx_val:split_idx_train].max(axis=0, keepdims=True)
    y_t = (y_t - ymin) / (ymax - ymin + 1e-8)

    # Threshold
    config["THRESHOLD"] = config["THRESHOLD"] / (ymax - ymin + 1e-8)
    config["Scale"] = ymax - ymin + 1e-8

    # ===============================================================================
    # STAGE 2: Feature Extraction
    # ===============================================================================
    # ------------------------------------------
    # Frequency Transform
    # ------------------------------------------
    Xf_t, f_names = extract_features(X_t, n_harmonics=config["N_HARMONICS"],
                                     selector=config.get("SNN_FEATURE_SELECTOR", config["REG_FEATURE_SELECTOR"]),
                                     return_names=True)
    gc.collect()

    # ------------------------------------------
    # Delta Features
    # ------------------------------------------
    dXf_t = feature_deltas(Xf_t, mode=config.get("SNN_DELTA_MODE", "absolute"))
    dy_t = feature_deltas(y_t, mode="absolute")
    ds_t = feature_deltas(s_t, mode="absolute")

    # ------------------------------------------
    # Split data
    # ------------------------------------------
    # Train
    time_sequence_train = time_sequence[split_idx_val:split_idx_train]
    time_raw_train = time_raw_cycles[split_idx_val:split_idx_train].reshape(-1)
    X_raw_train = X_t[split_idx_val:split_idx_train]
    Xf_t_train = Xf_t[split_idx_val:split_idx_train]
    dXf_t_train = dXf_t[split_idx_val:split_idx_train]
    y_t_train = y_t[split_idx_val:split_idx_train]
    dy_t_train = dy_t[split_idx_val:split_idx_train]
    s_t_train = s_t[split_idx_val:split_idx_train]
    ds_t_train = ds_t[split_idx_val:split_idx_train]

    # Validation
    time_sequence_val = time_sequence[:split_idx_val]
    time_raw_val = time_raw_cycles[:split_idx_val].reshape(-1)
    X_raw_val = X_t[:split_idx_val]
    Xf_t_val = Xf_t[:split_idx_val]
    dXf_t_val = dXf_t[:split_idx_val]
    y_t_val = y_t[:split_idx_val]
    dy_t_val = dy_t[:split_idx_val]
    s_t_val = s_t[:split_idx_val]
    ds_t_val = ds_t[:split_idx_val]

    # Test
    time_sequence_test = time_sequence[split_idx_train:]
    time_raw_test = time_raw_cycles[split_idx_train:].reshape(-1)
    X_raw_test = X_t[split_idx_train:]
    Xf_t_test = Xf_t[split_idx_train:]
    dXf_t_test = dXf_t[split_idx_train:]
    y_t_test = y_t[split_idx_train:]
    dy_t_test = dy_t[split_idx_train:]
    s_t_test = s_t[split_idx_train:]
    ds_t_test = ds_t[split_idx_train:]

    # ------------------------------------------
    # Input Selector
    # ------------------------------------------
    # SNN
    if config.get("SNN_INPUT_TRANSFORM", "delta") == "delta":
        X_snn_train = dXf_t_train
        X_snn_test = dXf_t_test
        X_snn_val = dXf_t_val
    elif config.get("SNN_INPUT_TRANSFORM") == "absolute":
        X_snn_train = Xf_t_train
        X_snn_test = Xf_t_test
        X_snn_val = Xf_t_val
    else:
        raise ValueError("SNN_INPUT_TRANSFORM must be 'delta' or 'absolute'.")

    # REG
    X_reg_train = Xf_t_train
    X_reg_test = Xf_t_test
    X_reg_val = Xf_t_val

    # ------------------------------------------
    # Creat Targets
    # ------------------------------------------
    # SNN
    y_train_snn = ds_t_train if config["USE_DERIVATIVE"] else s_t_train
    y_val_snn = ds_t_val if config["USE_DERIVATIVE"] else s_t_val
    y_test_snn = ds_t_test if config["USE_DERIVATIVE"] else s_t_test

    # REG
    y_reg_train = y_t_train
    y_reg_val = y_t_val
    y_reg_test = y_t_test

    # ------------------------------------------
    # Windowing
    # ------------------------------------------
    # SNN
    X_snn_train, y_train_snn = create_sequences(X_snn_train, y_train_snn,
                                                sequence_length=config["WINDOW"],
                                                stride=config["STRIDE"],
                                                mode=config["SNN_MODE"])
    X_snn_val, y_val_snn = create_sequences(X_snn_val, y_val_snn,
                                            sequence_length=config["WINDOW"],
                                            stride=config["STRIDE"],
                                            mode=config["SNN_MODE"])

    if config["SNN_MODE"] == "s2s":
        X_snn_test, y_test_snn = create_sequences(X_snn_test, y_test_snn,
                                                  sequence_length=config["WINDOW"],
                                                  stride=config["WINDOW"],
                                                  mode=config["SNN_MODE"])
    else:
        X_snn_test, y_test_snn = create_sequences(X_snn_test, y_test_snn,
                                                  sequence_length=config["WINDOW"],
                                                  stride=1,
                                                  mode=config["SNN_MODE"])

    # REG
    X_reg_train, y_reg_train = create_sequences(X_reg_train, y_reg_train,
                                                sequence_length=config["WINDOW"],
                                                stride=config["STRIDE"],
                                                mode=config["SNN_MODE"])
    X_reg_val, y_reg_val = create_sequences(X_reg_val, y_reg_val,
                                            sequence_length=config["WINDOW"],
                                            stride=config["STRIDE"],
                                            mode=config["SNN_MODE"])
    X_reg_test, y_reg_test = create_sequences(X_reg_test, y_reg_test,
                                              sequence_length=config["WINDOW"],
                                              stride=1,
                                              mode=config["SNN_MODE"])

    # ------------------------------------------
    # Balance
    # ------------------------------------------
    if config["BALANCE_DATA"]:
        num_pos = y_train_snn.sum()
        num_neg = y_train_snn.size - num_pos
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
    else:
        pos_weight = torch.tensor([1 / 1], device=device)
    print(f"Training sequences: {len(X_snn_train):,}; test sequences: {len(X_snn_test):,} ; "
          f"validation sequences: {len(X_snn_val):,}")

    # ===============================================================================
    # STAGE 3: SNN Classification
    # ===============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    # Model
    if config["CLA_TYPE"] == "snn":
        mdlCLA = SNNModel(input_size=X_snn_train.shape[-1], hidden_size=config["SNN_HIDDEN_SIZE"], output_size=C,
                          num_layers=config["SNN_NUM_LAYERS"], beta=config["SNN_BETA"]).to(device)
    else:
        test = 1

    # Loss Fnc and Optimizer
    opt = torch.optim.Adam(mdlCLA.parameters(), lr=config["SNN_LR"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ------------------------------------------
    # Train
    # ------------------------------------------
    if config["SNN_DO_TRAIN"]:
        if config["CLA_TYPE"] == "snn":
            mdlCLA = train_snn(mdlCLA, X_snn_train, y_train_snn, X_snn_val, y_val_snn, opt, loss_fn, config, device)
        else:
            test = 1

    # -----------------------------------------
    # Inference
    # ------------------------------------------
    if config["SNN_MODE"] == "s2s":
        if config["CLA_TYPE"] == "snn":
            evaluation = test_snn(mdlCLA, X_snn_test, config, device, load_checkpoint=True)
        else:
            test = 1
        y_pred_snn = evaluation["predictions"].reshape(-1, evaluation["predictions"].shape[-1])
        y_test_snn = y_test_snn.reshape(-1, y_test_snn.shape[-1])
        y_prob_snn = evaluation["probabilities"].reshape(-1, evaluation["probabilities"].shape[-1])
    else:
        if config["CLA_TYPE"] == "snn":
            evaluation = test_snn(mdlCLA, X_snn_test, config, device, load_checkpoint=True)
        else:
            test = 1
        y_pred_snn = evaluation["predictions"]
        y_prob_snn = evaluation["probabilities"]
    time_sequence_pred = time_sequence_test[:len(y_pred_snn)]

    """
    # ===============================================================================
    # STAGE 4: LSTM Regression
    # ===============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    # Model
    mdlREG = LSTMModel(input_size=X_reg_train.shape[-1], hidden_size=config["REG_HIDDEN_SIZE"], output_size=C,
                       num_layers=config["REG_NUM_LAYERS"]).to(device)

    # Loss Fnc and Optimizer
    opt = torch.optim.Adam(mdlREG.parameters(), lr=config["REG_LR"])
    loss_fn = nn.MSELoss(reduction="mean")

    # ------------------------------------------
    # Train
    # ------------------------------------------
    if config["REG_DO_TRAIN"]:
        mdlREG = train_lstm(mdlREG, X_reg_train, y_reg_train, X_reg_val, y_reg_val, opt, loss_fn, config, device)

    # ------------------------------------------
    # Inference
    # ------------------------------------------
    evaluation = test_lstm(mdlREG, X_reg_test, config, device, load_checkpoint=True)
    y_pred_snn = evaluation["predictions"]
    y_prob_snn = evaluation["probabilities"]
    time_sequence_pred = time_sequence_test[:len(y_pred_snn)]
    """

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
    # SNN Stage
    metrics = evaluate_classification(y_test_snn, y_pred_snn)

    # RNN Stage

    # ===============================================================================
    # STAGE 6: Plotting
    # ===============================================================================
    # ------------------------------------------
    # SNN
    # ------------------------------------------
    if config["PLOT_SNN"]:
        for i in range(0, C):
            output_dir = project_root / "results"
            output_dir.mkdir(exist_ok=True)

            feature_magnitudes = np.abs(Xf_t_test)
            encoding_magnitudes = np.abs(dXf_t_test)
            positive_magnitudes = np.concatenate((feature_magnitudes.ravel(), encoding_magnitudes.ravel()))
            positive_magnitudes = positive_magnitudes[positive_magnitudes > 0]
            color_min = positive_magnitudes.min()
            color_max = max(positive_magnitudes.max(), color_min * 10)
            feature_norm = LogNorm(vmin=color_min, vmax=color_max)
            feature_cmap = "viridis"

            fig, axes = plt.subplots(3, 2, figsize=(18, 12), constrained_layout=True)
            raw_axis = axes[0, 0]
            feature_axis = axes[1, 0]
            delta_axis = axes[2, 0]
            power_axis = axes[0, 1]
            state_axis = axes[1, 1]
            prediction_axis = axes[2, 1]

            raw_axis.plot(time_raw_test, X_raw_test[:, :, 0].reshape(-1), color="tab:blue", linewidth=0.3,
                          rasterized=True, label="voltage")
            current_axis = raw_axis.twinx()
            current_axis.plot(time_raw_test, X_raw_test[:, :, 1].reshape(-1), color="tab:orange", linewidth=0.3,
                              rasterized=True, label="current")
            raw_axis.set(title="Raw Voltage and Current (Complete Test Set)", xlabel="Time (s)", ylabel="Voltage")
            current_axis.set_ylabel("Current")
            raw_axis.legend(loc="upper left")
            current_axis.legend(loc="upper right")

            feature_image = feature_axis.imshow(
                feature_magnitudes.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=feature_cmap,
                norm=feature_norm,
                extent=(time_sequence_test[0], time_sequence_test[-1], -0.5, len(f_names) - 0.5),
            )
            feature_axis.set(title="Extracted Feature Magnitudes (Complete Test Set)", xlabel="Time (s)", ylabel="Feature")
            feature_axis.set_yticks(range(len(f_names)), f_names)
            fig.colorbar(feature_image, ax=feature_axis, pad=0.01, label="Magnitude (log scale)")

            delta_image = delta_axis.imshow(
                encoding_magnitudes.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=feature_cmap,
                norm=feature_norm,
                extent=(time_sequence_test[0], time_sequence_test[-1], -0.5, len(f_names) - 0.5),
            )
            delta_axis.set(title="SNN Input Feature Magnitudes (Complete Test Set)", xlabel="Time (s)", ylabel="Feature")
            delta_axis.set_yticks(range(len(f_names)), f_names)
            fig.colorbar(delta_image, ax=delta_axis, pad=0.01, label="Magnitude (log scale)")

            power_axis.plot(time_sequence_test, y_t_test[:, i], color="tab:green", linewidth=0.5, rasterized=True, label="power")
            power_axis.set(title="Filtered Appliance Power (Resampled to Raw Time)", xlabel="Time (s)", ylabel="Power (W)")
            power_axis.legend(loc="upper right")

            state_axis.step(time_sequence_test, s_t_test[:, i], where="post", color="tab:blue", linewidth=0.5,
                            rasterized=True, label="ON/OFF state")
            state_axis.step(time_sequence_test, ds_t_test[:, i], where="post", color="tab:orange", linewidth=0.5,
                            rasterized=True, label="state change")
            state_axis.set(title="Binary State and State Change (Resampled to Raw Time)", xlabel="Time (s)", ylabel="State",
                           ylim=(-0.1, 1.1))
            state_axis.legend(loc="upper right")

            prediction_axis.step(time_sequence_pred, y_test_snn[:, i], where="post", color="tab:blue", linewidth=0.5,
                                 rasterized=True, label="target")
            prediction_axis.step(time_sequence_pred, y_pred_snn[:, i], where="post", color="tab:red", linewidth=0.5,
                                 rasterized=True, label="prediction")
            prediction_axis.plot(time_sequence_pred, y_prob_snn[:, i], color="tab:purple", linewidth=0.5, alpha=0.7,
                                 rasterized=True, label="membrane probability")
            prediction_axis.set(title="SNN Prediction (Resampled to Raw Time)", xlabel="Time (s)", ylabel="ON probability",
                                ylim=(-0.1, 1.1))
            prediction_axis.legend(loc="upper right")

            fig.savefig(output_dir / f"snn_predictions_dev{config['DEVICE_IDS'][i]}.png", dpi=150)
            plt.close(fig)


#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == "__main__":
    main(config)
