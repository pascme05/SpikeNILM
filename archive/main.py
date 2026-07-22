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

Pipeline overview
-----------------
Stage 1 – Shared feature extraction
    The raw voltage/current waveform matrix (X) is the same for every device.
    It is loaded once and converted to a compact feature vector per AC cycle
    (FFT harmonics + RMS / peak / power statistics).

Stage 2 – Per-device SNN classifier
    One Spiking Neural Network (SNN) is trained independently for each
    appliance device ID.  The SNN classifies each sliding window of feature
    vectors as appliance ON or OFF (or as a state change when USE_DERIVATIVE=True).

Stage 3 – Multi-device power regressor  (optional, DO_REGRESSION=True)
    A single CNN or LSTM regressor takes the shared feature sequences,
    optionally concatenated with the spike trains from all Stage-2 SNNs,
    and predicts the continuous power consumption of every device
    simultaneously (one output neuron per device, MSELoss).

Key control variables  (edit in build_config)
---------------------------------------------
DEVICE_IDS            : list[int]  – appliance IDs to model
MODEL_TYPE            : str        – 'snn' | 'cnn' | 'lstm' (classifier)
DO_TRAIN              : bool       – True = train; False = load checkpoint
DO_REGRESSION         : bool       – enable the regression stage
REG_USE_SNN_INPUT     : bool       – concatenate SNN spike trains with features (True)
                                     or use features only (False)
PLOT_SNN              : bool       – produce per-device SNN classification plots
PLOT_REGRESSION       : bool       – produce per-device regression plots
"""

#######################################################################################################################
# Import external libs
#######################################################################################################################
# ─── Standard library ────────────────────────────────────────────────────────
from copy import deepcopy
from pathlib import Path

# ─── Third-party ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import snntorch.spikeplot as splt
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    max_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

# Optuna is optional – only required when USE_OPTUNA=True
try:
    import optuna
except ImportError:
    optuna = None

# ─── Project modules ─────────────────────────────────────────────────────────
from helper import (
    balance_sequences,
    build_target_spikes,
    compute_feature_deltas,
    create_sequences,
    encode_spikes,
    extract_features,
    load_data_multi,
    select_input_channels,
    snn_predict,
)
from snnNet import build_model


#######################################################################################################################
# CONFIGURATION
#######################################################################################################################
def build_config():
    """Return the central configuration dictionary.

    All hyper-parameters, file paths, and pipeline switches are defined here
    so that no magic numbers are scattered throughout the code.
    Changing a single value in this dict is sufficient to alter behaviour
    anywhere in the pipeline.
    """
    return {
        # ── Dataset ──────────────────────────────────────────────────────────
        "NAME": "redd3HF",                                                      # Dataset name
        "DEVICE_IDS": [5],                                                      # Appliance device IDs to model (one SNN per ID)
        "THRESHOLD": 50,                                                        # Power threshold (W) separating ON from OFF state
        "SPLIT": 0.95,                                                          # Fraction of samples used for training
        "MAX_LEN": -1,                                                          # Max AC cycles to load  (-1 = full dataset)
        "N_HARMONICS": 9,                                                       # FFT harmonics extracted per voltage/current channel
        "USE_FEATURES": True,                                                   # True = FFT features;  False = flattened raw waveform
        "FEATURE_SELECTOR": {
            "voltage_harmonics": True,
            "current_harmonics": True,
            "voltage_stats": True,
            "current_stats": True,
            "power_stats": True,
        },
        "SNN_FEATURE_SELECTOR": {                                               # Classifier-specific selector (defaults to current-driven change features)
            "voltage_harmonics": False,
            "current_harmonics": True,
            "voltage_stats": False,
            "current_stats": True,
            "power_stats": False,
        },
        "RAW_INPUT_CHANNELS": ["voltage", "current"],                           # Raw mode selector: any subset of ['voltage', 'current']
        "SNN_RAW_INPUT_CHANNELS": ["current"],                                  # Raw SNN selector when USE_FEATURES=False
        "INPUT_NORM": "0-1",                                                    # Shared input normalisation: 'none' | '0-1' | 'mean/std'
        "OUTPUT_NORM": "none",                                                  # Regression target normalisation: 'none' | '0-1' | 'mean/std'
        "USE_DERIVATIVE": False,                                                 # True = predict state changes;  False = predict ON/OFF
        "BALANCE_DATA": True,                                                   # Undersample majority class for balanced training
        "STRIDE": 5,                                                            # Sliding-window stride used during training
        "DEVICE": "auto",                                                       # 'auto' = prefer CUDA, otherwise CPU; can also force 'cuda' or 'cpu'
        "GPU_INDEX": 0,                                                         # CUDA device index when DEVICE resolves to GPU
        "NUM_WORKERS": 0,                                                       # DataLoader workers (keep 0 on Windows unless profiling suggests otherwise)

        # ── SNN / Classifier ─────────────────────────────────────────────────
        "MODEL_TYPE": "snn",                                                    # Classifier type: 'snn' | 'cnn' | 'lstm'
        "SEQ_LEN": 100,                                                         # Number of AC cycles per input window
        "HIDDEN_SIZE": 64,                                                      # Hidden layer width (neurons / channels)
        "NUM_LAYERS": 3,                                                        # Number of stacked layers
        "BETA": 0.95,                                                           # Initial LIF membrane decay factor  (SNN only)
        "KERNEL_SIZE": 5,                                                       # Convolutional kernel size           (CNN only)
        "DROPOUT": 0.2,                                                         # Dropout probability                 (LSTM only)
        "CODING": "rate",                                                        # Spike encoding: 'raw'|'rate'|'latency'|'delta'
        "SNN_INPUT_TRANSFORM": "delta",                                         # 'delta' = consecutive-frame change signal, 'absolute' = original feature levels
        "SNN_DELTA_MODE": "absolute",                                           # 'absolute' = magnitude of change, 'signed' = signed change
        "SNN_LOSS_MODE": "membrane",                                            # Loss target: 'membrane' | 'spike'
        "SNN_EVAL_MODE": "spike_count",                                         # Prediction strategy: 'spike_count'|'membrane'|'spike_any'
        "ON_RATE": 0.8,                                                         # Target spike rate for ON class   (spike loss mode)
        "OFF_RATE": 0.0,                                                        # Target spike rate for OFF class  (spike loss mode)
        "BATCH_SIZE": 1024,                                                     # Mini-batch size for training                                        
        "LR": 1e-3,                                                             # Learning rate for Adam optimiser
        "EPOCHS": 50,                                                           # Number of training epochs
        "DO_TRAIN": True,                                                       # True = train;  False = load from checkpoint
        "SNN_SAVE_PATH_TEMPLATE": "best_snn_dev{device_id}.pt",                 # One checkpoint per device

        # ── Optuna hyper-parameter search ─────────────────────────────────────
        "USE_OPTUNA": False,                                                    # Run Optuna search before the final training run
        "OPTUNA_TRIALS": 15,                                                    # Number of Optuna trials
        "OPTUNA_EPOCHS": 10,                                                    # Epochs per trial  (kept short for speed)

        # ── Regression stage ──────────────────────────────────────────────────
        "DO_REGRESSION": True,                                                  # Enable Stage-3 power regression
        "REGRESSOR_TYPE": "lstm",                                               # Regressor architecture: 'cnn' | 'lstm'
        "REG_USE_SNN_INPUT": True,                                              # True = features + SNN spikes;  False = features only
        "DO_TRAIN_REGRESSOR": True,                                             # True = train;  False = load from checkpoint
        "REGRESSOR_EPOCHS": 100,                                                # Number of training epochs for the regressor
        "REGRESSOR_LR": 1e-3,                                                   # Learning rate for the regressor
        "REG_HIDDEN_SIZE": 64,                                                  # Hidden layer width for the regressor
        "REG_NUM_LAYERS": 2,                                                    # Number of stacked layers for the regressor
        "REG_DROPOUT": 0.2,                                                     # Dropout probability for the regressor (LSTM only)
        "REG_KERNEL_SIZE": 5,                                                   # Convolutional kernel size for the regressor (CNN only)
        "REGRESSOR_SAVE_PATH": "best_regressor.pt",                             # Checkpoint path for the regressor

        # ── Plotting ──────────────────────────────────────────────────────────
        # Disable plots during automated runs (e.g. Optuna sweeps) to save time.
        "PLOT_SNN": True,                                                       # Generate per-device SNN classification plots
        "PLOT_REGRESSION": True,                                                # Generate per-device regression plots
        "PLOT_DEBUG_BATCH": True,                                              # Plot the first test batch input/output for model debugging
        "DEBUG_SAMPLE_INDEX": 0,                                                # Sample index inside the debug batch used for detailed views
        "DEBUG_BATCH_PLOT_SAMPLES": 24,                                         # Max number of batch samples shown in debug heatmaps
    }


def resolve_device(config):
    """Resolve the requested compute device and fail loudly on invalid CUDA setups."""
    requested = str(config.get("DEVICE", "auto")).lower()
    gpu_index = int(config.get("GPU_INDEX", 0))

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown DEVICE setting: {requested}. Use 'auto', 'cpu', or 'cuda'.")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"GPU_INDEX={gpu_index} is out of range for {torch.cuda.device_count()} visible CUDA device(s)."
            )
        return torch.device(f"cuda:{gpu_index}")

    if requested == "cuda":
        cuda_build = torch.version.cuda or "CPU-only build"
        raise RuntimeError(
            "DEVICE='cuda' was requested, but CUDA is not available in this Python environment. "
            f"Installed torch: {torch.__version__} ({cuda_build})."
        )

    return torch.device("cpu")


def configure_runtime(device):
    """Enable CUDA performance settings when a GPU is active."""
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def build_dataloader(dataset, batch_size, shuffle, config, device):
    """Create a DataLoader with GPU-friendly host-to-device settings."""
    num_workers = int(config["NUM_WORKERS"])
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def move_tensor(tensor, device):
    """Move a tensor to the active device, using async copies on CUDA."""
    return tensor.to(device, non_blocking=device.type == "cuda")


def resolve_sample_index(config, batch_size):
    """Clamp the configured sample index to the current batch size."""
    return min(max(int(config.get("DEBUG_SAMPLE_INDEX", 0)), 0), batch_size - 1)


def fit_normalizer(data, mode):
    """Fit a per-feature normaliser on the first axis of `data`."""
    mode = mode.lower()
    if mode == "none":
        offset = np.zeros((1, data.shape[1]), dtype=np.float32)
        scale = np.ones((1, data.shape[1]), dtype=np.float32)
    elif mode == "0-1":
        offset = data.min(axis=0, keepdims=True)
        scale = data.max(axis=0, keepdims=True) - offset
    elif mode == "mean/std":
        offset = data.mean(axis=0, keepdims=True)
        scale = data.std(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown normalisation mode: {mode}. Use 'none', '0-1' or 'mean/std'.")

    return {
        "mode": mode,
        "offset": offset.astype(np.float32, copy=False),
        "scale": (scale.astype(np.float32, copy=False) + 1e-8),
    }


def apply_normalizer(data, normalizer):
    """Apply a fitted normaliser to a numpy array."""
    return ((data - normalizer["offset"]) / normalizer["scale"]).astype(np.float32, copy=False)


def denormalize_array(data, normalizer):
    """Invert a fitted normaliser on a numpy array."""
    return (data * normalizer["scale"] + normalizer["offset"]).astype(np.float32, copy=False)


def describe_selector(selector, config, raw_channels=None):
    """Create a short human-readable summary of an input selector."""
    if raw_channels is not None:
        return ", ".join(raw_channels)

    labels = []
    if selector.get("voltage_harmonics", True):
        labels.append(f"{config['N_HARMONICS']} V-harmonics")
    if selector.get("current_harmonics", True):
        labels.append(f"{config['N_HARMONICS']} I-harmonics")
    if selector.get("voltage_stats", True):
        labels.append("2 voltage stats")
    if selector.get("current_stats", True):
        labels.append("2 current stats")
    if selector.get("power_stats", True):
        labels.append("2 power stats")
    return ", ".join(labels)


def describe_feature_selection(config):
    """Describe the shared feature selector used by the non-SNN shared path."""
    if config["USE_FEATURES"]:
        return describe_selector(config["FEATURE_SELECTOR"], config)
    return describe_selector(None, config, raw_channels=config["RAW_INPUT_CHANNELS"])


def describe_snn_input(config):
    """Describe the classifier-specific SNN input path."""
    if config["USE_FEATURES"]:
        base = describe_selector(config["SNN_FEATURE_SELECTOR"], config)
    else:
        base = describe_selector(None, config, raw_channels=config["SNN_RAW_INPUT_CHANNELS"])
    if config["SNN_INPUT_TRANSFORM"] == "delta":
        return f"{config['SNN_DELTA_MODE']} delta of {base}"
    return base


#######################################################################################################################
# MAIN FUNCTIONS
#######################################################################################################################
# =============================================================================
# STAGE 1 – SHARED FEATURE EXTRACTION
# =============================================================================
def prepare_shared_features(config):
    """Load the mat file once and extract the shared feature matrix.

    The raw waveform matrix X is identical for every device ID; only Y (power)
    differs.  We therefore extract features once and reuse them for all devices.

    Feature vector per AC cycle (when USE_FEATURES=True):
        - configurable subsets of voltage harmonics, current harmonics,
          voltage/current statistics, and power statistics

    Returns a dict with:
        X_norm          : normalised shared feature array     [n_samples, input_size]
        X_snn_norm      : normalised classifier input array   [n_samples, snn_input_size]
        input_size      : number of shared features per cycle
        snn_input_size  : number of SNN input features per cycle
        feature_names   : shared feature names
        snn_feature_names : classifier-specific feature names
        n_samples       : total number of AC cycles loaded
        split_idx_raw   : sample index separating training from test data
        X_train_t_reg   : sliding-window sequences tensor (train) for the regressor
        X_test_t_reg    : sliding-window sequences tensor (test)  for the regressor
    """
    X_raw, _ = load_data_multi(f"data/{config['NAME']}.mat", config["DEVICE_IDS"], maxLen=config["MAX_LEN"])
    n_samples = X_raw.shape[0]

    split_idx_raw = int(n_samples * config["SPLIT"])

    if config["USE_FEATURES"]:
        # Compute compact spectral + statistical feature vectors
        X_processed, feature_names = extract_features(
            X_raw,
            n_harmonics=config["N_HARMONICS"],
            selector=config["FEATURE_SELECTOR"],
            return_names=True,
        )
        X_snn_source, snn_feature_names = extract_features(
            X_raw,
            n_harmonics=config["N_HARMONICS"],
            selector=config["SNN_FEATURE_SELECTOR"],
            return_names=True,
        )
        input_size = X_processed.shape[1]
        print(
            f"Samples: {n_samples}, Shared features per cycle: {input_size} "
            f"({describe_feature_selection(config)})"
        )
    else:
        # Select raw waveform channels before flattening
        X_selected = select_input_channels(X_raw, config["RAW_INPUT_CHANNELS"])
        X_processed = X_selected.reshape(n_samples, -1).astype(np.float32)
        input_size = X_processed.shape[1]
        feature_names = [
            f"{channel}_t{step}"
            for step in range(X_selected.shape[1])
            for channel in config["RAW_INPUT_CHANNELS"]
        ]
        X_snn_selected = select_input_channels(X_raw, config["SNN_RAW_INPUT_CHANNELS"])
        X_snn_source = X_snn_selected.reshape(n_samples, -1).astype(np.float32)
        snn_feature_names = [
            f"{channel}_t{step}"
            for step in range(X_snn_selected.shape[1])
            for channel in config["SNN_RAW_INPUT_CHANNELS"]
        ]
        print(
            f"Samples: {n_samples}, Shared raw input size: {input_size} "
            f"({describe_feature_selection(config)})"
        )

    if config["SNN_INPUT_TRANSFORM"] == "delta":
        X_snn_processed = compute_feature_deltas(X_snn_source, mode=config["SNN_DELTA_MODE"])
        snn_feature_names = [f"d_{name}" for name in snn_feature_names]
    elif config["SNN_INPUT_TRANSFORM"] == "absolute":
        X_snn_processed = X_snn_source.astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"Unknown SNN_INPUT_TRANSFORM: {config['SNN_INPUT_TRANSFORM']}. Use 'delta' or 'absolute'."
        )

    snn_input_size = X_snn_processed.shape[1]
    print(
        f"SNN input per cycle: {snn_input_size} "
        f"({describe_snn_input(config)})"
    )

    input_normalizer = fit_normalizer(X_processed[:split_idx_raw], config["INPUT_NORM"])
    snn_input_normalizer = fit_normalizer(X_snn_processed[:split_idx_raw], config["INPUT_NORM"])
    X_norm = apply_normalizer(X_processed, input_normalizer)
    X_snn_norm = apply_normalizer(X_snn_processed, snn_input_normalizer)
    seq_len = config["SEQ_LEN"]

    # Build sliding-window sequences from the feature matrix.
    # A dummy all-zero Y is used here because we only need the X sequences;
    # the actual per-device labels are created in prepare_device_classification.
    dummy_train = np.zeros(split_idx_raw, dtype=np.int64)
    dummy_test = np.zeros(n_samples - split_idx_raw, dtype=np.int64)
    X_train_seq, _ = create_sequences(X_norm[:split_idx_raw], dummy_train, seq_len, stride=config["STRIDE"])
    X_test_seq, _ = create_sequences(X_norm[split_idx_raw:], dummy_test, seq_len, stride=1)

    return {
        "X_norm": X_norm,
        "X_snn_norm": X_snn_norm,
        "input_size": input_size,
        "snn_input_size": snn_input_size,
        "feature_names": feature_names,
        "snn_feature_names": snn_feature_names,
        "input_normalizer": input_normalizer,
        "n_samples": n_samples,
        "split_idx_raw": split_idx_raw,
        "X_train_t_reg": torch.tensor(X_train_seq, dtype=torch.float32),
        "X_test_t_reg": torch.tensor(X_test_seq, dtype=torch.float32),
    }

# =============================================================================
# STAGE 2 – PER-DEVICE DATA PREPARATION & SNN TRAINING
# =============================================================================
def prepare_device_classification(config, shared, device_id, Y_device, device):
    """Build classification DataLoaders and power regression targets for one device.

    Steps:
        1. Binarise the raw power signal using THRESHOLD  → ON / OFF labels.
        2. Optionally compute the first-order difference to get state-change labels.
        3. Create overlapping sliding windows of the shared feature matrix.
        4. Store power targets (unbalanced) for later use in the regression stage.
        5. Optionally undersample the majority class (BALANCE_DATA=True).

    Returns a data_info dict with keys:
        train_loader    : DataLoader for balanced classification training
        test_loader     : DataLoader for test evaluation
        input_size      : feature dimension
        output_size     : number of classes (always 2)
        class_names     : human-readable class labels
        raw_y           : raw continuous power array (for plotting)
        split_idx_raw   : train/test split index in the raw timeline
        Y_train_power_t : torch.float32 tensor of training power values  [n_train_seq]
        Y_test_power_t  : torch.float32 tensor of test power values      [n_test_seq]
    """
    X_norm = shared["X_snn_norm"] if config["MODEL_TYPE"] == "snn" else shared["X_norm"]
    split_idx_raw = shared["split_idx_raw"]
    seq_len = config["SEQ_LEN"]

    # ── Label construction ────────────────────────────────────────────────────
    # Binarise: 1 if power exceeds threshold, 0 otherwise
    Y_bin = (Y_device > config["THRESHOLD"]).astype(np.int64)
    print(f"[Device {device_id}] Class distribution - OFF: {(Y_bin == 0).sum()}, ON: {(Y_bin == 1).sum()}")

    if config["USE_DERIVATIVE"]:
        # Detect transitions: |diff| = 1 at every ON→OFF and OFF→ON boundary
        Y_target = np.abs(np.diff(Y_bin, prepend=Y_bin[0])).astype(np.int64)
        class_names = ["No Change", "Change"]
        print(
            f"[Device {device_id}] Derivative - No Change: {(Y_target == 0).sum()}, "
            f"Change: {(Y_target == 1).sum()}"
        )
    else:
        Y_target = Y_bin
        class_names = ["OFF", "ON"]

    # ── Sliding-window sequencing ─────────────────────────────────────────────
    # Each window is SEQ_LEN AC cycles wide; the label is the state of the last cycle.
    X_train_seq, Y_train_seq = create_sequences(
        X_norm[:split_idx_raw], Y_target[:split_idx_raw], seq_len, stride=config["STRIDE"]
    )
    X_test_seq, Y_test_seq = create_sequences(
        X_norm[split_idx_raw:], Y_target[split_idx_raw:], seq_len, stride=1
    )
    print(
        f"[Device {device_id}] Train sequences: {X_train_seq.shape[0]} (stride={config['STRIDE']}), "
        f"Test sequences: {X_test_seq.shape[0]} (stride=1)"
    )

    # ── Power targets for the regression stage ────────────────────────────────
    # These are kept at the original (unbalanced) sample count so that regression
    # targets align exactly with the shared feature sequences stored in `shared`.
    Y_power = Y_device.astype(np.float32)
    Y_train_power_t = torch.tensor(
        Y_power[:split_idx_raw][seq_len - 1 :: config["STRIDE"]].copy(), dtype=torch.float32
    )
    Y_test_power_t = torch.tensor(
        Y_power[split_idx_raw:][seq_len - 1 :].copy(), dtype=torch.float32
    )

    # ── Class balancing ───────────────────────────────────────────────────────
    # Undersampling only affects the classifier training set; the regression
    # power targets remain unbalanced (full timeline).
    if config["BALANCE_DATA"]:
        X_train_seq, Y_train_seq = balance_sequences(X_train_seq, Y_train_seq)
        classes, counts = np.unique(Y_train_seq, return_counts=True)
        print(
            f"[Device {device_id}] After balancing: {dict(zip(classes.tolist(), counts.tolist()))}, "
            f"Total: {len(Y_train_seq)}"
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_seq, dtype=torch.long)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test_seq, dtype=torch.long)

    train_loader = build_dataloader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        config=config,
        device=device,
    )
    test_loader = build_dataloader(
        TensorDataset(X_test_t, Y_test_t),
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        config=config,
        device=device,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "input_size": shared["snn_input_size"] if config["MODEL_TYPE"] == "snn" else shared["input_size"],
        "output_size": 2,
        "class_names": class_names,
        "feature_names": shared["snn_feature_names"] if config["MODEL_TYPE"] == "snn" else shared["feature_names"],
        "raw_y": Y_device,
        "split_idx_raw": split_idx_raw,
        "Y_train_power_t": Y_train_power_t,
        "Y_test_power_t": Y_test_power_t,
    }



# =============================================================================
# HYPER-PARAMETERS & MODEL CONSTRUCTION
# =============================================================================

def build_hyperparams(config, trial=None):
    """Assemble the hyper-parameter dict used for model construction and training.

    When `trial` is None the values come directly from `config` (manual mode).
    When `trial` is an Optuna Trial object, values are sampled from the search
    space so that Optuna can optimise them automatically.
    """
    # Default: use values from config directly
    hyperparams = {
        "hidden_size": config["HIDDEN_SIZE"],
        "num_layers": config["NUM_LAYERS"],
        "beta": config["BETA"],
        "kernel_size": config["KERNEL_SIZE"],
        "dropout": config["DROPOUT"],
        # SNN typically benefits from a 5× higher learning rate than CNN/LSTM
        "lr": config["LR"] * 5 if config["MODEL_TYPE"] == "snn" else config["LR"],
    }

    if trial is None:
        return hyperparams

    # ── Optuna search space ───────────────────────────────────────────────────
    hyperparams["hidden_size"] = trial.suggest_int("hidden_size", 32, 256, step=32)
    hyperparams["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    hyperparams["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    # Model-type-specific hyper-parameters
    if config["MODEL_TYPE"] == "snn":
        hyperparams["beta"] = trial.suggest_float("beta", 0.7, 0.99)
    elif config["MODEL_TYPE"] == "cnn":
        hyperparams["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
    elif config["MODEL_TYPE"] == "lstm":
        hyperparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

    return hyperparams


def create_model_and_criterion(config, data_info, hyperparams, device):
    """Instantiate the classifier model, loss function, and Adam optimiser.

    Loss selection:
        SNN + membrane mode  → CrossEntropyLoss on mean membrane potential
        SNN + spike   mode   → MSELoss on spike-rate targets
        CNN / LSTM           → CrossEntropyLoss on logits
    """
    model = build_model(
        model_type=config["MODEL_TYPE"],
        input_size=data_info["input_size"],
        hidden_size=hyperparams["hidden_size"],
        output_size=data_info["output_size"],
        beta=hyperparams["beta"],
        num_layers=hyperparams["num_layers"],
        kernel_size=hyperparams["kernel_size"],
        dropout=hyperparams["dropout"],
    ).to(device)

    # Spike-rate targets require MSELoss; everything else uses cross-entropy
    if config["MODEL_TYPE"] == "snn" and config["SNN_LOSS_MODE"] == "spike":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    return model, criterion, optimizer



# =============================================================================
# TRAINING UTILITIES  (forward pass, prediction, full training loop, Optuna)
# =============================================================================

def forward_batch(model, criterion, X_batch, Y_batch, config, data_info, device):
    """Run one forward pass and compute the loss for a single mini-batch.

    Handles both SNN and standard (CNN/LSTM) models transparently.

    SNN-specific logic:
        'membrane' mode  – the mean membrane potential across all time steps is
                           used as class logits → CrossEntropyLoss.
        'spike' mode     – spike-rate targets are generated and MSELoss is applied
                           directly to the spike-count tensor.

    Returns:
        loss  : scalar loss tensor (with grad)
        preds : [batch] int64 tensor of predicted class indices
    """
    X_batch = move_tensor(X_batch, device)
    Y_batch = move_tensor(Y_batch, device)

    if config["MODEL_TYPE"] == "snn":
        # Convert raw features to the chosen spike encoding before feeding the SNN
        spike_input = encode_spikes(X_batch, config["CODING"], device)
        spk_rec, mem_rec = model(spike_input)  # spk_rec: [T, B, C],  mem_rec: [T, B, C]

        if config["SNN_LOSS_MODE"] == "membrane":
            # Average membrane potential over all time steps → class logits
            logits = mem_rec.mean(dim=0)   # [B, C]
            loss = criterion(logits, Y_batch)
        else:
            # Build deterministic spike-rate target patterns for each class
            target_spk = build_target_spikes(
                Y_batch,
                num_steps=spk_rec.size(0),
                num_classes=data_info["output_size"],
                on_rate=config["ON_RATE"],
                off_rate=config["OFF_RATE"],
            )
            loss = criterion(spk_rec, target_spk)

        preds = snn_predict(spk_rec, mem_rec, config["SNN_EVAL_MODE"])
    else:
        # Standard forward pass for CNN / LSTM
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        preds = logits.argmax(dim=1)

    return loss, preds


def predict_loader(
    model,
    loader,
    config,
    device,
    return_plot_data=False,
    return_debug_batch=False,
):
    """Evaluate the model on an entire DataLoader without computing gradients.

    Returns:
        accuracy  : float – fraction of correctly classified sequences
        all_preds : ndarray [n_samples] – predicted class indices
        all_true  : ndarray [n_samples] – ground-truth class indices
        plot_data : dict | None – optional representative sample data for plotting
        debug_batch : dict | None – optional first-batch debug tensors
    """
    model.eval()
    all_preds = []
    all_true = []
    plot_data = None
    debug_batch = None
    plot_input_strength = []
    plot_change_score = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = move_tensor(X_batch, device)
            sample_index = resolve_sample_index(config, X_batch.size(0))
            if config["MODEL_TYPE"] == "snn":
                spike_input = encode_spikes(X_batch, config["CODING"], device)
                spk_rec, mem_rec = model(spike_input)
                preds = snn_predict(spk_rec, mem_rec, config["SNN_EVAL_MODE"])
                output_summary = mem_rec.mean(dim=0) if config["SNN_EVAL_MODE"] == "membrane" else spk_rec.sum(dim=0)
                if return_plot_data:
                    plot_input_strength.append(X_batch.abs().mean(dim=(1, 2)).cpu().numpy())
                    plot_change_score.append(output_summary[:, 1].cpu().numpy())

                if return_plot_data and plot_data is None:
                    plot_data = {
                        "plot_input_sample": X_batch[sample_index].cpu(),
                        "plot_encoded_input_sample": spike_input[:, sample_index].cpu(),
                        "plot_output_spikes_sample": spk_rec[:, sample_index].cpu(),
                        "plot_output_mem_sample": mem_rec[:, sample_index].cpu(),
                        "plot_true_label": int(Y_batch[sample_index].item()),
                        "plot_pred_label": int(preds[sample_index].item()),
                        "plot_sample_index": sample_index,
                    }
                if return_debug_batch and debug_batch is None:
                    debug_batch = {
                        "input_batch": X_batch.cpu(),
                        "encoded_input_batch": spike_input.cpu(),
                        "output_spikes_batch": spk_rec.cpu(),
                        "output_mem_batch": mem_rec.cpu(),
                        "output_summary_batch": output_summary.cpu(),
                        "labels": Y_batch.cpu(),
                        "preds": preds.cpu(),
                    }
            else:
                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                if return_plot_data and plot_data is None:
                    plot_data = {
                        "plot_input_sample": X_batch[sample_index].cpu(),
                        "plot_output_logits_sample": logits[sample_index].cpu(),
                        "plot_true_label": int(Y_batch[sample_index].item()),
                        "plot_pred_label": int(preds[sample_index].item()),
                        "plot_sample_index": sample_index,
                    }
                if return_debug_batch and debug_batch is None:
                    debug_batch = {
                        "input_batch": X_batch.cpu(),
                        "output_logits_batch": logits.cpu(),
                        "labels": Y_batch.cpu(),
                        "preds": preds.cpu(),
                    }
            all_preds.append(preds.cpu().numpy())
            all_true.append(Y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    accuracy = (all_preds == all_true).mean()
    if plot_data is not None and plot_input_strength:
        plot_data["plot_input_strength"] = np.concatenate(plot_input_strength)
        plot_data["plot_change_score"] = np.concatenate(plot_change_score)
    return accuracy, all_preds, all_true, plot_data, debug_batch


def train_model(config, data_info, device, hyperparams, num_epochs, save_path=None):
    """Full classifier training loop with best-model checkpointing.

    If DO_TRAIN=False, the function skips training and loads the checkpoint
    directly from `save_path` instead.

    Checkpointing:
        The model state that achieves the highest test accuracy is saved to
        `save_path` after every evaluation epoch.  At the end of training
        the best state is loaded back.

    Returns:
        model    : the model loaded with the best weights
        best_acc : float – best test accuracy observed during training
    """
    model, criterion, optimizer = create_model_and_criterion(config, data_info, hyperparams, device)
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}, "
        f"LR: {hyperparams['lr']:.6f}, Hidden: {hyperparams['hidden_size']}, Layers: {hyperparams['num_layers']}"
    )

    best_acc = -1.0
    best_state = None

    if config["DO_TRAIN"]:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0

            # ── Mini-batch gradient descent ───────────────────────────────────
            for X_batch, Y_batch in data_info["train_loader"]:
                loss, preds = forward_batch(model, criterion, X_batch, Y_batch, config, data_info, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_correct += (preds.cpu() == Y_batch).sum().item()
                train_total += Y_batch.size(0)

            train_acc = train_correct / train_total
            avg_loss = epoch_loss / len(data_info["train_loader"])

            # Evaluate and checkpoint every 5 epochs, at epoch 1, and at the last epoch
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == num_epochs:
                test_acc, _, _, _, _ = predict_loader(model, data_info["test_loader"], config, device)
                print(
                    f"Epoch [{epoch + 1:>2}/{num_epochs}]  Loss: {avg_loss:.4f}  "
                    f"Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f}"
                )

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_state = deepcopy(model.state_dict())
                    if save_path is not None:
                        torch.save(best_state, save_path)
                        print(f"  -> Saved best model ({best_acc:.4f}) to {save_path}")

        print(f"\nBest test accuracy during training: {best_acc:.4f}")
    else:
        # Load-only mode: read the checkpoint and skip all training
        if save_path is None or not Path(save_path).exists():
            raise FileNotFoundError(f"Saved model not found: {save_path}")
        best_state = torch.load(save_path, map_location=device, weights_only=True)
        print(f"Skipping training, loading model from {save_path}")

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    model.load_state_dict(best_state)
    if save_path is not None and config["DO_TRAIN"]:
        print(f"Loaded model from {save_path}")
    return model, best_acc


def run_optuna(config, data_info, device):
    """Run an Optuna hyper-parameter search and return the best parameter dict.

    Each trial trains a classifier for OPTUNA_EPOCHS epochs (short runs)
    and evaluates test accuracy as the objective.  The best hyper-parameters
    are then used for the final full training run in main().
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Install it or set USE_OPTUNA=False.")

    def objective(trial):
        hyperparams = build_hyperparams(config, trial)
        model, _ = train_model(
            config={**config, "DO_TRAIN": True},
            data_info=data_info,
            device=device,
            hyperparams=hyperparams,
            num_epochs=config["OPTUNA_EPOCHS"],
            save_path=None,          # Don't save checkpoints during search
        )
        accuracy, _, _, _, _ = predict_loader(model, data_info["test_loader"], config, device)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["OPTUNA_TRIALS"])
    print(f"Optuna best value: {study.best_value:.4f}")
    print(f"Optuna best params: {study.best_params}")
    return study.best_params



# =============================================================================
# EVALUATION & SNN CLASSIFICATION PLOTS
# =============================================================================

def report_metrics(all_true, all_preds, class_names):
    """Print a comprehensive classification report with multiple metrics.

    Metrics reported:
        - Accuracy, Balanced Accuracy
        - Precision, Recall (Sensitivity), Specificity
        - F1 Score
        - Matthews Correlation Coefficient (MCC)
        - Cohen's Kappa
        - Confusion Matrix

    Returns:
        cm : ndarray [2, 2] – the confusion matrix
    """
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (Test Set)")
    print("=" * 50)
    print(classification_report(all_true, all_preds, target_names=class_names, zero_division=0))

    # Core metrics
    accuracy = (all_preds == all_true).mean()
    balanced_acc = balanced_accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, zero_division=0)
    precision = precision_score(all_true, all_preds, zero_division=0)
    recall = recall_score(all_true, all_preds, zero_division=0)  # = sensitivity
    mcc = matthews_corrcoef(all_true, all_preds)
    kappa = cohen_kappa_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    # Specificity = TN / (TN + FP)  — derived from the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Precision (PPV):   {precision:.4f}")
    print(f"Recall (Sens.):    {recall:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return cm


def plot_results(config, data_info, all_true, all_preds, cm, device_id, plot_data=None):
    """Generate and save three classification diagnostic plots for one device.

    Plot 1 – Input and output summaries plus true vs predicted class labels.
    Plot 2 – Raw power signal with ON-prediction regions shaded.
    Plot 3 – Confusion matrix heatmap.

    Files saved: pred_vs_true_dev{id}.png, power_vs_pred_dev{id}.png,
                 confusion_matrix_dev{id}.png
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ── Plot 1: representative model input/output with prediction timeline ───
    if plot_data is not None and config["MODEL_TYPE"] == "snn":
        fig, axes = plt.subplots(
            4, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.4]}
        )
        ax_delta, ax_in, ax_out, ax_pred = axes

        input_sample = plot_data["plot_input_sample"].cpu()
        encoded_input_sample = plot_data["plot_encoded_input_sample"].cpu()
        output_spikes_sample = plot_data["plot_output_spikes_sample"].cpu()
        output_mem_sample = plot_data["plot_output_mem_sample"].cpu().numpy()
        sample_index = plot_data["plot_sample_index"]
        true_label = data_info["class_names"][plot_data["plot_true_label"]]
        pred_label = data_info["class_names"][plot_data["plot_pred_label"]]

        delta_im = ax_delta.imshow(
            input_sample.numpy().T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
        ax_delta.set_ylabel("Feature")
        ax_delta.set_xlabel("Time step")
        ax_delta.set_title(f"Sample {sample_index} - SNN Change Signal ({describe_snn_input(config)})")
        fig.colorbar(delta_im, ax=ax_delta, pad=0.01, label="Normalized change")

        if config["CODING"] == "raw":
            enc_im = ax_in.imshow(
                encoded_input_sample.numpy().T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="Greys",
            )
            fig.colorbar(enc_im, ax=ax_in, pad=0.01, label="Encoded value")
        else:
            splt.raster(encoded_input_sample, ax_in, s=3, c="black")
        ax_in.set_ylabel("Feature")
        ax_in.set_xlabel("Time step")
        ax_in.set_title("Encoded input sent to the SNN")

        splt.raster(output_spikes_sample, ax_out, s=28, c="firebrick")
        ax_out.set_ylabel("Class")
        ax_out.set_xlabel("Time step")
        ax_out.set_yticks(range(len(data_info["class_names"])))
        ax_out.set_yticklabels(data_info["class_names"])
        ax_out.set_title(
            f"{config['MODEL_TYPE'].upper()} [{config['CODING']}] Device {device_id} - "
            f"Sample {sample_index} Output Spikes  |  True={true_label}, Pred={pred_label}"
        )
        mem_ax = ax_out.twinx()
        for class_index, class_name in enumerate(data_info["class_names"]):
            mem_ax.plot(output_mem_sample[:, class_index], linewidth=1.0, alpha=0.5, label=f"{class_name} mem")
        mem_ax.set_ylabel("Membrane")
        mem_ax.grid(False)

        ax_pred.plot(all_true, label=f"True ({'/'.join(data_info['class_names'])})", alpha=0.7, linewidth=0.8)
        ax_pred.plot(
            all_preds,
            label=f"Predicted ({'/'.join(data_info['class_names'])})",
            alpha=0.7,
            linewidth=0.8,
            linestyle="--",
        )
        ax_pred.set_xlabel("Test Sample Index")
        ax_pred.set_ylabel("Class")
        ax_pred.set_title(
            f"{config['MODEL_TYPE'].upper()} [{config['CODING']}] Device {device_id} - True vs Predicted "
            f"({'Changes' if config['USE_DERIVATIVE'] else 'State'}) (Test Set)"
        )
        ax_pred.legend()
    elif plot_data is not None:
        input_sample = plot_data["plot_input_sample"].cpu().numpy().T
        logits_sample = plot_data["plot_output_logits_sample"].cpu().numpy()
        sample_index = plot_data["plot_sample_index"]
        true_label = data_info["class_names"][plot_data["plot_true_label"]]
        pred_label = data_info["class_names"][plot_data["plot_pred_label"]]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.1, 0.9, 1.4]})
        ax_in, ax_logits, ax_pred = axes
        in_im = ax_in.imshow(input_sample, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax_in.set_ylabel("Feature")
        ax_in.set_xlabel("Time step")
        ax_in.set_title(f"Sample {sample_index} - Input fed to {config['MODEL_TYPE'].upper()}")
        fig.colorbar(in_im, ax=ax_in, pad=0.01, label="Input value")

        ax_logits.bar(range(len(logits_sample)), logits_sample, color=["steelblue", "tomato"][: len(logits_sample)])
        ax_logits.set_xticks(range(len(data_info["class_names"])))
        ax_logits.set_xticklabels(data_info["class_names"])
        ax_logits.set_ylabel("Logit")
        ax_logits.set_title(f"Sample {sample_index} Output  |  True={true_label}, Pred={pred_label}")

        ax_pred.plot(all_true, label=f"True ({'/'.join(data_info['class_names'])})", alpha=0.7, linewidth=0.8)
        ax_pred.plot(
            all_preds,
            label=f"Predicted ({'/'.join(data_info['class_names'])})",
            alpha=0.7,
            linewidth=0.8,
            linestyle="--",
        )
        ax_pred.set_xlabel("Test Sample Index")
        ax_pred.set_ylabel("Class")
        ax_pred.set_title(
            f"{config['MODEL_TYPE'].upper()} [{config['CODING']}] Device {device_id} - True vs Predicted "
            f"({'Changes' if config['USE_DERIVATIVE'] else 'State'}) (Test Set)"
        )
        ax_pred.legend()
    else:
        fig, ax_pred = plt.subplots(figsize=(14, 4))
        ax_pred.plot(all_true, label=f"True ({'/'.join(data_info['class_names'])})", alpha=0.7, linewidth=0.8)
        ax_pred.plot(
            all_preds,
            label=f"Predicted ({'/'.join(data_info['class_names'])})",
            alpha=0.7,
            linewidth=0.8,
            linestyle="--",
        )
        ax_pred.set_xlabel("Test Sample Index")
        ax_pred.set_ylabel("Class")
        ax_pred.set_title(
            f"{config['MODEL_TYPE'].upper()} [{config['CODING']}] Device {device_id} - True vs Predicted "
            f"({'Changes' if config['USE_DERIVATIVE'] else 'State'}) (Test Set)"
        )
        ax_pred.legend()

    plt.tight_layout()
    plt.savefig(results_dir / f"pred_vs_true_dev{device_id}.png", dpi=150)
    plt.show()

    # ── Plot 2: raw power with predicted ON regions shaded ────────────────────
    # Align the raw power signal with the test sequences (first seq_len-1 samples
    # are consumed by the first window and have no prediction)
    y_test_raw = data_info["raw_y"][
        data_info["split_idx_raw"] + config["SEQ_LEN"] - 1:
        data_info["split_idx_raw"] + config["SEQ_LEN"] - 1 + len(all_preds)
    ]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test_raw, label="True Power (W)", color="steelblue", linewidth=0.8)
    ax.axhline(config["THRESHOLD"], color="gray", linestyle=":", label=f"Threshold ({config['THRESHOLD']} W)")
    if config["USE_DERIVATIVE"]:
        true_change_idx = np.flatnonzero(all_true == 1)
        pred_change_idx = np.flatnonzero(all_preds == 1)
        if len(true_change_idx) > 0:
            ax.vlines(true_change_idx, 0, y_test_raw.max(), colors="green", linewidth=0.8, alpha=0.45, label="True change")
        if len(pred_change_idx) > 0:
            ax.vlines(pred_change_idx, 0, y_test_raw.max(), colors="tomato", linewidth=0.8, alpha=0.55, linestyles="--", label="Predicted change")
        if plot_data is not None and "plot_input_strength" in plot_data:
            ax_change = ax.twinx()
            ax_change.plot(
                plot_data["plot_input_strength"],
                color="darkorange",
                linewidth=1.0,
                alpha=0.8,
                label="Mean input change",
            )
            ax_change.plot(
                plot_data["plot_change_score"],
                color="firebrick",
                linewidth=1.0,
                alpha=0.5,
                label="Output change score",
            )
            ax_change.set_ylabel("Change activity")
            ax_change.grid(False)
            lines_left, labels_left = ax.get_legend_handles_labels()
            lines_right, labels_right = ax_change.get_legend_handles_labels()
            ax.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")
        else:
            ax.legend()
    else:
        ax.fill_between(
            range(len(all_preds)),
            0,
            y_test_raw.max(),
            where=(all_preds == 1),   # Shade wherever the model predicts ON
            alpha=0.15,
            color="red",
            label="Predicted ON",
        )
        ax.legend()
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Power (W)")
    ax.set_title(
        f"{config['MODEL_TYPE'].upper()} [{config['CODING']}] Device {device_id} - "
        f"{'Change events and change activity' if config['USE_DERIVATIVE'] else 'Predicted ON Regions vs True Power'}"
    )
    plt.tight_layout()
    plt.savefig(results_dir / f"power_vs_pred_dev{device_id}.png", dpi=150)
    plt.show()

    # ── Plot 3: confusion matrix ───────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(data_info["class_names"])
    ax.set_yticklabels(data_info["class_names"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix Device {device_id} [{config['MODEL_TYPE'].upper()} / {config['CODING']}]")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_dev{device_id}.png", dpi=150)
    plt.show()


def plot_debug_batch(config, data_info, debug_batch, device_id):
    """Plot the first evaluation batch entering and leaving the classifier."""
    if debug_batch is None:
        return

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    max_samples = min(int(config["DEBUG_BATCH_PLOT_SAMPLES"]), debug_batch["labels"].shape[0])
    sample_index = resolve_sample_index(config, debug_batch["labels"].shape[0])
    true_label = data_info["class_names"][int(debug_batch["labels"][sample_index].item())]
    pred_label = data_info["class_names"][int(debug_batch["preds"][sample_index].item())]

    if config["MODEL_TYPE"] == "snn":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_raw, ax_in, ax_out, ax_summary = axes.ravel()

        raw_sample = debug_batch["input_batch"][sample_index].numpy().T
        raw_im = ax_raw.imshow(raw_sample, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax_raw.set_title(f"Sample {sample_index} SNN Input Features ({describe_snn_input(config)})")
        ax_raw.set_xlabel("Time step")
        ax_raw.set_ylabel("Feature")
        fig.colorbar(raw_im, ax=ax_raw, pad=0.01, label="Input value")

        encoded_sample = debug_batch["encoded_input_batch"][:, sample_index]
        if config["CODING"] == "raw":
            enc_im = ax_in.imshow(
                encoded_sample.numpy().T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="Greys",
            )
            fig.colorbar(enc_im, ax=ax_in, pad=0.01, label="Encoded value")
        else:
            splt.raster(encoded_sample, ax_in, s=3, c="black")
        ax_in.set_title(f"Sample {sample_index} Encoded Change Input ({config['CODING']})")
        ax_in.set_xlabel("Time step")
        ax_in.set_ylabel("Feature")

        output_spikes = debug_batch["output_spikes_batch"][:, sample_index]
        splt.raster(output_spikes, ax_out, s=28, c="firebrick")
        ax_out.set_title(f"Sample {sample_index} Output Spikes  |  True={true_label}, Pred={pred_label}")
        ax_out.set_xlabel("Time step")
        ax_out.set_ylabel("Class")
        ax_out.set_yticks(range(len(data_info["class_names"])))
        ax_out.set_yticklabels(data_info["class_names"])

        summary_batch = debug_batch["output_summary_batch"][:max_samples].numpy().T
        summary_im = ax_summary.imshow(
            summary_batch,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="magma",
        )
        summary_label = "Membrane Avg." if config["SNN_EVAL_MODE"] == "membrane" else "Spike Count"
        ax_summary.set_title(f"First Test Batch Change Summary ({max_samples} samples)")
        ax_summary.set_xlabel("Batch sample index")
        ax_summary.set_ylabel("Class")
        ax_summary.set_yticks(range(len(data_info["class_names"])))
        ax_summary.set_yticklabels(data_info["class_names"])
        fig.colorbar(summary_im, ax=ax_summary, pad=0.01, label=summary_label)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax_sample, ax_batch, ax_out = axes

        sample_input = debug_batch["input_batch"][sample_index].numpy().T
        sample_im = ax_sample.imshow(sample_input, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax_sample.set_title(f"Sample {sample_index} Input  |  True={true_label}, Pred={pred_label}")
        ax_sample.set_xlabel("Time step")
        ax_sample.set_ylabel("Feature")
        fig.colorbar(sample_im, ax=ax_sample, pad=0.01, label="Input value")

        batch_summary = debug_batch["input_batch"][:max_samples].mean(dim=1).numpy().T
        batch_im = ax_batch.imshow(batch_summary, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax_batch.set_title(f"First Test Batch Mean Input ({max_samples} samples)")
        ax_batch.set_xlabel("Batch sample index")
        ax_batch.set_ylabel("Feature")
        fig.colorbar(batch_im, ax=ax_batch, pad=0.01, label="Mean input")

        logits_batch = debug_batch["output_logits_batch"][:max_samples].numpy().T
        logit_im = ax_out.imshow(logits_batch, aspect="auto", origin="lower", interpolation="nearest", cmap="coolwarm")
        ax_out.set_title(f"First Test Batch Output Logits ({max_samples} samples)")
        ax_out.set_xlabel("Batch sample index")
        ax_out.set_ylabel("Class")
        ax_out.set_yticks(range(len(data_info["class_names"])))
        ax_out.set_yticklabels(data_info["class_names"])
        fig.colorbar(logit_im, ax=ax_out, pad=0.01, label="Logit")

    plt.tight_layout()
    plt.savefig(results_dir / f"debug_batch_dev{device_id}.png", dpi=150)
    plt.show()



# =============================================================================
# STAGE 3 – MULTI-DEVICE POWER REGRESSION
# =============================================================================

def prepare_regression_data(config, snn_models, shared, device_infos, device):
    """Build DataLoaders for the multi-output power regression model.

    Depending on REG_USE_SNN_INPUT:
        True  → Input is the shared feature sequence concatenated with the
                spike train of every per-device SNN.
                Shape: [batch, seq_len, feature_size + n_devices * 2]
        False → Input is the shared feature sequence only.
                Shape: [batch, seq_len, feature_size]

    Target: [batch, n_devices] — continuous power in Watts for each device
            (the power reading at the last AC cycle of the window).

    Returns a dict with:
        train_loader    : DataLoader (shuffled)
        test_loader     : DataLoader (sequential)
        reg_input_size  : int – feature dimension of the regressor input
        n_devices       : int – number of appliance outputs
        plot_raw_test   : torch.Tensor [n_test, seq_len, raw_feature_dim]
        plot_spike_test : torch.Tensor | None [n_test, seq_len, spike_dim]
    """
    use_snn = config["REG_USE_SNN_INPUT"] and snn_models is not None and len(snn_models) > 0
    n_devices = len(device_infos)

    X_train_t_reg = shared["X_train_t_reg"]
    X_test_t_reg = shared["X_test_t_reg"]

    # Stack per-device power values → [n_samples, n_devices]
    Y_train_power = torch.stack([di["Y_train_power_t"] for di in device_infos], dim=1)
    Y_test_power = torch.stack([di["Y_test_power_t"] for di in device_infos], dim=1)

    target_normalizer = fit_normalizer(Y_train_power.numpy(), config["OUTPUT_NORM"])
    Y_train_power = torch.from_numpy(apply_normalizer(Y_train_power.numpy(), target_normalizer))
    Y_test_power = torch.from_numpy(apply_normalizer(Y_test_power.numpy(), target_normalizer))

    def extract_all_spikes(X_t):
        """Run every frozen SNN and concatenate their spike outputs.

        Returns: [n_samples, seq_len, n_devices * 2]
        Each SNN produces a [seq_len, 2] spike tensor per sample (ON/OFF neurons).
        """
        spike_parts = [[] for _ in snn_models]
        loader = build_dataloader(
            TensorDataset(X_t),
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            config=config,
            device=device,
        )
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = move_tensor(X_batch, device)
                spike_input = encode_spikes(X_batch, config["CODING"], device)
                for i, snn in enumerate(snn_models):
                    spk_rec, _ = snn(spike_input)
                    # [num_steps, batch, 2] → [batch, num_steps, 2]
                    spike_parts[i].append(spk_rec.permute(1, 0, 2).cpu())
        return torch.cat([torch.cat(parts, dim=0) for parts in spike_parts], dim=-1)

    if use_snn:
        print(f"Extracting spike trains from {len(snn_models)} SNN(s) for regression data...")
        for snn in snn_models:
            snn.eval()
        spk_train = extract_all_spikes(X_train_t_reg)
        spk_test = extract_all_spikes(X_test_t_reg)
        X_train_combined = torch.cat([X_train_t_reg, spk_train], dim=-1)
        X_test_combined = torch.cat([X_test_t_reg, spk_test], dim=-1)
    else:
        print("Regression using features only (REG_USE_SNN_INPUT=False)...")
        spk_test = None
        X_train_combined = X_train_t_reg
        X_test_combined = X_test_t_reg

    reg_input_size = X_train_combined.shape[-1]
    print(
        f"Regression input size: {reg_input_size}, output: {n_devices} device(s), "
        f"train samples: {X_train_combined.shape[0]}, test samples: {X_test_combined.shape[0]}"
    )

    train_loader = build_dataloader(
        TensorDataset(X_train_combined, Y_train_power),
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        config=config,
        device=device,
    )
    test_loader = build_dataloader(
        TensorDataset(X_test_combined, Y_test_power),
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        config=config,
        device=device,
    )
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "reg_input_size": reg_input_size,
        "n_devices": n_devices,
        "target_normalizer": target_normalizer,
        "plot_raw_test": X_test_t_reg,
        "plot_spike_test": spk_test,
    }


def eval_regressor(regressor, loader, device, target_normalizer=None):
    """Evaluate the power regressor on a DataLoader.

    Returns:
        metrics   : dict with keys 'rmse', 'mse', 'mae', 'r2', 'mape', 'max_err'
        all_preds : ndarray [n_samples, n_devices] – predicted power values
        all_true  : ndarray [n_samples, n_devices] – ground-truth power values
    """
    regressor.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            preds = regressor(move_tensor(X_batch, device)).cpu()   # [batch, n_devices] or [batch, 1]
            all_preds.append(preds)
            all_true.append(Y_batch)
    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()

    if target_normalizer is not None:
        all_preds = denormalize_array(all_preds, target_normalizer)
        all_true = denormalize_array(all_true, target_normalizer)

    # Flatten for global (across-all-devices) metrics
    flat_pred = all_preds.ravel()
    flat_true = all_true.ravel()

    mse_val = float(mean_squared_error(flat_true, flat_pred))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(mean_absolute_error(flat_true, flat_pred))
    r2_val = float(r2_score(flat_true, flat_pred))
    max_err_val = float(max_error(flat_true, flat_pred))
    # MAPE: guard against division by zero when true power is 0
    mask = flat_true != 0
    if mask.any():
        mape_val = float(mean_absolute_percentage_error(flat_true[mask], flat_pred[mask]))
    else:
        mape_val = float('nan')

    metrics = {
        "rmse": rmse_val,
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val,
        "mape": mape_val,
        "max_err": max_err_val,
    }
    return metrics, all_preds, all_true


def train_regressor(config, reg_data, device):
    """Train a CNN or LSTM to predict continuous power (one output per device).

    Uses MSELoss.  Best-model checkpointing is based on test RMSE.

    Returns:
        regressor : nn.Module with the best weights loaded
    """
    n_devices = reg_data["n_devices"]
    regressor = build_model(
        model_type=config["REGRESSOR_TYPE"],
        input_size=reg_data["reg_input_size"],
        hidden_size=config["REG_HIDDEN_SIZE"],
        output_size=n_devices,
        num_layers=config["REG_NUM_LAYERS"],
        kernel_size=config["REG_KERNEL_SIZE"],
        dropout=config["REG_DROPOUT"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=config["REGRESSOR_LR"])
    best_rmse = float("inf")
    best_state = None
    mdl_dir = Path("mdl")
    mdl_dir.mkdir(parents=True, exist_ok=True)
    regressor_save_path = str(mdl_dir / Path(config["REGRESSOR_SAVE_PATH"]).name)

    if config["DO_TRAIN_REGRESSOR"]:
        print(
            f"Regressor ({config['REGRESSOR_TYPE'].upper()}) params: "
            f"{sum(p.numel() for p in regressor.parameters()):,},  LR: {config['REGRESSOR_LR']}"
        )
        for epoch in range(config["REGRESSOR_EPOCHS"]):
            regressor.train()
            epoch_loss = 0.0
            for X_batch, Y_batch in reg_data["train_loader"]:
                X_batch = move_tensor(X_batch, device)
                Y_batch = move_tensor(Y_batch, device)
                preds = regressor(X_batch)   # [batch, n_devices]
                loss = criterion(preds, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == config["REGRESSOR_EPOCHS"]:
                reg_metrics, _, _ = eval_regressor(
                    regressor,
                    reg_data["test_loader"],
                    device,
                    target_normalizer=reg_data["target_normalizer"],
                )
                rmse = reg_metrics["rmse"]
                avg_loss = epoch_loss / len(reg_data["train_loader"])
                print(
                    f"Regressor Epoch [{epoch + 1:>2}/{config['REGRESSOR_EPOCHS']}]  "
                    f"Loss: {avg_loss:.4f}  Test RMSE: {rmse:.2f} W"
                )
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_state = deepcopy(regressor.state_dict())
                    torch.save(best_state, regressor_save_path)
                    print(f"  -> Saved best regressor ({best_rmse:.2f} W) to {regressor_save_path}")

        print(f"\nBest regressor RMSE: {best_rmse:.2f} W")
    else:
        if not Path(regressor_save_path).exists():
            raise FileNotFoundError(f"Regressor model not found: {regressor_save_path}")
        best_state = torch.load(regressor_save_path, map_location=device, weights_only=True)
        print(f"Loaded regressor from {regressor_save_path}")

    regressor.load_state_dict(best_state)
    return regressor


# =============================================================================
# REGRESSION PLOTS
# =============================================================================

def plot_regression(config, all_pred_power, all_true_power, device_ids):
    """Generate and save per-device regression diagnostic plots.

    For each device:
        Plot 1 – True vs Predicted power time series.
        Plot 2 – Scatter plot (true vs predicted) with ideal-fit line.

    Files saved: power_regression_dev{id}.png, power_regression_scatter_dev{id}.png
    """
    # Ensure 2-D shape [n_samples, n_devices] even for a single device
    if all_pred_power.ndim == 1:
        all_pred_power = all_pred_power[:, None]
        all_true_power = all_true_power[:, None]

    n_devices = all_pred_power.shape[1]
    input_label = "SNN + features" if config["REG_USE_SNN_INPUT"] else "features only"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for i, dev_id in enumerate(device_ids):
        pred = all_pred_power[:, i]
        true = all_true_power[:, i]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mae = np.mean(np.abs(pred - true))

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(true, label="True Power (W)", color="steelblue", linewidth=0.8)
        ax.plot(pred, label="Predicted Power (W)", color="tomato", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Power (W)")
        ax.set_title(
            f"Device {dev_id} — {config['REGRESSOR_TYPE'].upper()} Regression ({input_label})  "
            f"RMSE={rmse:.2f} W  MAE={mae:.2f} W"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"power_regression_dev{dev_id}.png", dpi=150)
        plt.show()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(true, pred, alpha=0.3, s=5, rasterized=True)
        lim_min = min(float(true.min()), float(pred.min()))
        lim_max = max(float(true.max()), float(pred.max()))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="Ideal")
        ax.set_xlabel("True Power (W)")
        ax.set_ylabel("Predicted Power (W)")
        ax.set_title(f"Device {dev_id} — Regression Scatter")
        ax.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"power_regression_scatter_dev{dev_id}.png", dpi=150)
        plt.show()


def plot_regression_with_inputs(config, reg_data, all_pred_power, all_true_power, device_ids):
    """Generate regression plots with separate spike/raw input panels above the output."""
    if all_pred_power.ndim == 1:
        all_pred_power = all_pred_power[:, None]
        all_true_power = all_true_power[:, None]

    input_label = "SNN + features" if config["REG_USE_SNN_INPUT"] else "features only"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    raw_summary = reg_data["plot_raw_test"].mean(dim=1).cpu().numpy().T
    spike_test = reg_data["plot_spike_test"]
    spike_summary = None if spike_test is None else spike_test.mean(dim=1).cpu().numpy().T
    raw_label = "Feature Ch." if config["USE_FEATURES"] else "Raw Input Ch."

    for i, dev_id in enumerate(device_ids):
        pred = all_pred_power[:, i]
        true = all_true_power[:, i]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mae = np.mean(np.abs(pred - true))

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.6]}
        )
        ax_spike, ax_raw, ax_out = axes

        if spike_summary is None:
            ax_spike.text(0.5, 0.5, "No spike inputs used", ha="center", va="center", transform=ax_spike.transAxes)
            ax_spike.set_yticks([])
        else:
            ax_spike.imshow(spike_summary, aspect="auto", origin="lower", interpolation="nearest", cmap="Greys")
        ax_spike.set_ylabel("Spike Ch.")
        ax_spike.set_title(
            f"Device {dev_id} - {config['REGRESSOR_TYPE'].upper()} Regression ({input_label})  "
            f"RMSE={rmse:.2f} W  MAE={mae:.2f} W"
        )

        ax_raw.imshow(raw_summary, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax_raw.set_ylabel(raw_label)

        ax_out.plot(true, label="True Power (W)", color="steelblue", linewidth=0.8)
        ax_out.plot(pred, label="Predicted Power (W)", color="tomato", linewidth=0.8, linestyle="--", alpha=0.8)
        ax_out.set_xlabel("Test Sample Index")
        ax_out.set_ylabel("Power (W)")
        ax_out.legend()

        plt.tight_layout()
        plt.savefig(results_dir / f"power_regression_dev{dev_id}.png", dpi=150)
        plt.show()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(true, pred, alpha=0.3, s=5, rasterized=True)
        lim_min = min(float(true.min()), float(pred.min()))
        lim_max = max(float(true.max()), float(pred.max()))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="Ideal")
        ax.set_xlabel("True Power (W)")
        ax.set_ylabel("Predicted Power (W)")
        ax.set_title(f"Device {dev_id} - Regression Scatter")
        ax.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"power_regression_scatter_dev{dev_id}.png", dpi=150)
        plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Orchestrate the full NILM pipeline.

    Execution flow:
        1. Print configuration summary.
        2. Load the shared waveform matrix and extract features  (Stage 1).
        3. For each device ID:
            a. Prepare per-device classification data  (binary ON/OFF labels).
            b. Optionally run Optuna hyper-parameter search.
            c. Train (or load) the SNN classifier  (Stage 2).
            d. Report metrics; optionally plot SNN results.
        4. If DO_REGRESSION=True:
            a. Build combined regression input  (features ± SNN spikes)  (Stage 3).
            b. Train (or load) the power regressor.
            c. Report RMSE/MAE; optionally plot regression results.
    """
    config = build_config()
    device_ids = config["DEVICE_IDS"]

    # ── Hardware & configuration summary ──────────────────────────────────────
    device = resolve_device(config)
    configure_runtime(device)
    print(f"Using device: {device}")
    print(f"PyTorch: {torch.__version__}  |  CUDA build: {torch.version.cuda or 'CPU-only'}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"CUDA device: {gpu_name}  |  VRAM: {total_vram_gb:.1f} GB")
    elif torch.version.cuda is None:
        print("CUDA status: current PyTorch build is CPU-only, so GPU acceleration is unavailable.")
    else:
        print("CUDA status: this PyTorch build includes CUDA, but no CUDA device is available to the process.")
    print(
        f"Model: {config['MODEL_TYPE'].upper()}, Coding: {config['CODING']}, "
        f"Features: {'extracted' if config['USE_FEATURES'] else 'raw'}"
    )
    print(f"Input selector: {describe_feature_selection(config)}")
    print(f"Normalisation - Input: {config['INPUT_NORM']}, Output: {config['OUTPUT_NORM']}")
    print(f"Device IDs: {device_ids}")
    print(f"Target: {'state changes (derivative)' if config['USE_DERIVATIVE'] else 'ON/OFF state'}")
    if config["MODEL_TYPE"] == "snn":
        print(f"SNN train: {config['SNN_LOSS_MODE']}, SNN eval: {config['SNN_EVAL_MODE']}")
        print(f"SNN classifier input: {describe_snn_input(config)}")
        if config["SNN_INPUT_TRANSFORM"] == "delta" and not config["USE_DERIVATIVE"]:
            print("WARNING: delta-style SNN input is usually paired with USE_DERIVATIVE=True.")
        if config["CODING"] in {"rate", "latency"} and config["INPUT_NORM"] != "0-1":
            print("WARNING: rate/latency coding works best when INPUT_NORM='0-1'.")
        if config["CODING"] == "rate" and config["SNN_INPUT_TRANSFORM"] == "delta" and config["SNN_DELTA_MODE"] == "signed":
            print("WARNING: signed delta inputs can suppress rate coding because negative values do not map naturally to spike rates.")
    print(f"Balance: {config['BALANCE_DATA']},  Optuna: {config['USE_OPTUNA']}")
    print(f"Plots — SNN: {config['PLOT_SNN']},  Regression: {config['PLOT_REGRESSION']}")
    print(f"Debug batch plot: {config['PLOT_DEBUG_BATCH']} (sample index {config['DEBUG_SAMPLE_INDEX']})")

    # ── Stage 1: shared feature extraction ────────────────────────────────────
    print("\n=== Loading & extracting shared features ===")
    _, Y_devices = load_data_multi(f"data/{config['NAME']}.mat", device_ids, maxLen=config["MAX_LEN"])
    shared = prepare_shared_features(config)

    # ── Stage 2: per-device SNN training ──────────────────────────────────────
    snn_models = []      # Stores one trained SNN per device (for regression input)
    device_infos = []    # Stores one data_info dict per device
    mdl_dir = Path("mdl")
    mdl_dir.mkdir(parents=True, exist_ok=True)

    for dev_id in device_ids:
        print(f"\n{'='*55}")
        print(f"  SNN Classifier — Device {dev_id}")
        print(f"{'='*55}")

        # Prepare per-device classification sequences + power targets
        data_info = prepare_device_classification(config, shared, dev_id, Y_devices[dev_id], device)
        device_infos.append(data_info)

        hyperparams = build_hyperparams(config)
        save_path = str(mdl_dir / config["SNN_SAVE_PATH_TEMPLATE"].format(device_id=dev_id))

        # Optional Optuna hyper-parameter search (only when training is enabled)
        if config["USE_OPTUNA"] and config["DO_TRAIN"]:
            best_params = run_optuna(config, data_info, device)
            hyperparams.update(best_params)

        # Train or load the SNN classifier
        snn_model, _ = train_model(
            config=config,
            data_info=data_info,
            device=device,
            hyperparams=hyperparams,
            num_epochs=config["EPOCHS"],
            save_path=save_path,
        )
        snn_models.append(snn_model)

        # Evaluate and report classification metrics
        _, all_preds, all_true, plot_data, debug_batch = predict_loader(
            snn_model,
            data_info["test_loader"],
            config,
            device,
            return_plot_data=config["PLOT_SNN"],
            return_debug_batch=config["PLOT_DEBUG_BATCH"],
        )
        cm = report_metrics(all_true, all_preds, data_info["class_names"])

        # Optionally generate SNN classification plots
        if config["PLOT_SNN"]:
            plot_results(config, data_info, all_true, all_preds, cm, device_id=dev_id, plot_data=plot_data)
        if config["PLOT_DEBUG_BATCH"]:
            plot_debug_batch(config, data_info, debug_batch, device_id=dev_id)

    # ── Stage 3: multi-device power regression ────────────────────────────────
    if config["DO_REGRESSION"]:
        if config["REG_USE_SNN_INPUT"] and config["MODEL_TYPE"] != "snn":
            print("WARNING: REG_USE_SNN_INPUT=True requires MODEL_TYPE='snn'. Skipping regression.")
        else:
            snn_for_reg = snn_models if config["REG_USE_SNN_INPUT"] else None
            input_label = "SNN spikes + features" if config["REG_USE_SNN_INPUT"] else "features only"
            print(
                f"\n{'='*55}\n"
                f"  Regression: {config['REGRESSOR_TYPE'].upper()} "
                f"({input_label} → power, {len(device_ids)} device(s))\n"
                f"{'='*55}"
            )

            # Build regressor DataLoaders (shared features ± SNN spike trains)
            reg_data = prepare_regression_data(config, snn_for_reg, shared, device_infos, device)

            # Train or load the power regressor
            regressor = train_regressor(config, reg_data, device)

            # Evaluate and report regression metrics
            reg_metrics, all_pred_power, all_true_power = eval_regressor(
                regressor,
                reg_data["test_loader"],
                device,
                target_normalizer=reg_data["target_normalizer"],
            )
            print(f"\n{'─'*50}")
            print(f"REGRESSION REPORT  ({len(device_ids)} device(s))")
            print(f"{'─'*50}")
            print(f"  RMSE:      {reg_metrics['rmse']:.2f} W")
            print(f"  MSE:       {reg_metrics['mse']:.2f} W²")
            print(f"  MAE:       {reg_metrics['mae']:.2f} W")
            print(f"  R²:        {reg_metrics['r2']:.4f}")
            print(f"  MAPE:      {reg_metrics['mape']:.2%}" if not np.isnan(reg_metrics['mape']) else "  MAPE:      N/A (true=0)")
            print(f"  Max Error: {reg_metrics['max_err']:.2f} W")

            # Optionally generate regression plots
            if config["PLOT_REGRESSION"]:
                plot_regression_with_inputs(config, reg_data, all_pred_power, all_true_power, device_ids)


if __name__ == "__main__":
    main()
