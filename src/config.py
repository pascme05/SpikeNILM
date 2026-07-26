def build_config_default():
    """Return the central configuration dictionary.

    All hyper-parameters, file paths, and pipeline switches are defined here
    so that no magic numbers are scattered throughout the code.
    Changing a single value in this dict is sufficient to alter behaviour
    anywhere in the pipeline.
    """
    return {
        # ── Dataset ──────────────────────────────────────────────────────────
        "NAME": "redd3HF",                                                      # Dataset name
        "T_SAMPLING": 3,                                                        # Sequence-frame sampling period (s)
        "DEVICE_IDS": [5],                                                      # Appliance device IDs to model (one SNN per ID)
        "THRESHOLD": 50,                                                        # Power threshold (W) separating ON from OFF state
        "SPLIT_TRAIN": 0.80,                                                    # Fraction of samples used for training
        "SPLIT_VAL": 0.10,                                                      # Fraction of samples used for validation
        "MAX_LEN": 50000,                                                          # Max AC cycles to load  (-1 = full dataset)
        "N_HARMONICS": 9,                                                       # FFT harmonics extracted per voltage/current channel
        "USE_FEATURES": True,                                                   # True = FFT features;  False = flattened raw waveform
        "REG_FEATURE_SELECTOR": {
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
            "current_stats": False,
            "power_stats": False,
        },
        "REG_INPUT_CHANNELS": ["voltage", "current"],                           # Raw mode selector: any subset of ['voltage', 'current']
        "SNN_RAW_INPUT_CHANNELS": ["current"],                                  # Raw SNN selector when USE_FEATURES=False
        "INPUT_NORM": "0-1",                                                    # Shared input normalisation: 'none' | '0-1' | 'mean/std'
        "OUTPUT_NORM": "none",                                                  # Regression target normalisation: 'none' | '0-1' | 'mean/std'
        "USE_DERIVATIVE": False,                                                 # True = predict state changes;  False = predict ON/OFF
        "BALANCE_DATA": True,                                                   # Undersample majority class for balanced training
        "DEVICE": "auto",                                                       # 'auto' = prefer CUDA, otherwise CPU; can also force 'cuda' or 'cpu'
        "GPU_INDEX": 0,                                                         # CUDA device index when DEVICE resolves to GPU
        "NUM_WORKERS": 0,                                                       # DataLoader workers (keep 0 on Windows unless profiling suggests otherwise)

        # ── General Model Para  ──────────────────────────────────────────────
        "WINDOW": 60,  # Number of AC cycles per input window
        "STRIDE": 1,  # Sliding-window stride used during training


        # ── SNN / Classifier ─────────────────────────────────────────────────
        "SNN_MODE": "s2p",                                                    # Classifier type: 's2s' sequence to sequence | 's2p' = sequence to point
        "SNN_HIDDEN_SIZE": 32,                                                      # Hidden layer width (neurons / channels)
        "SNN_NUM_LAYERS": 2,                                                        # Number of stacked layers
        "SNN_BETA": 0.95,                                                           # Initial LIF membrane decay factor  (SNN only)
        "SNN_KERNEL_SIZE": 5,                                                       # Convolutional kernel size           (CNN only)
        "SNN_DROPOUT": 0.2,                                                         # Dropout probability                 (LSTM only)
        "SNN_CODING": "raw",                                                        # Spike encoding: 'raw'|'rate'|'latency'|'delta'
        "SNN_INPUT_TRANSFORM": "absolute",                                         # 'delta' = consecutive-frame change signal, 'absolute' = original feature levels
        "SNN_DELTA_MODE": "absolute",                                           # 'absolute' = magnitude of change, 'signed' = signed change
        "SNN_LOSS_MODE": "membrane",                                            # Loss target: 'membrane' | 'spike'
        "SNN_EVAL_MODE": "spike_count",                                         # Prediction strategy: 'spike_count'|'membrane'|'spike_any'
        "SNN_ON_RATE": 0.8,                                                         # Target spike rate for ON class   (spike loss mode)
        "SNN_OFF_RATE": 0.0,                                                        # Target spike rate for OFF class  (spike loss mode)
        "SNN_BATCH_SIZE": 1024,                                                     # Mini-batch size for training
        "SNN_LR": 1e-3,                                                             # Learning rate for Adam optimiser
        "SNN_EPOCHS": 50,                                                           # Number of training epochs
        "SNN_PATIENCE": 5,                                                          # Number of training epochs without improvement
        "SNN_DO_TRAIN": True,                                                       # True = train;  False = load from checkpoint
        "SNN_SAVE_PATH": "mdl/best_snn_dev{device_id}.pt",                 # One checkpoint per device

        # ── Optuna hyper-parameter search ─────────────────────────────────────
        "USE_OPTUNA": False,                                                    # Run Optuna search before the final training run
        "OPTUNA_TRIALS": 15,                                                    # Number of Optuna trials
        "OPTUNA_EPOCHS": 10,                                                    # Epochs per trial  (kept short for speed)

        # ── Regression stage ──────────────────────────────────────────────────
        "REG_TYPE": "lstm",                                               # Regressor architecture: 'cnn' | 'lstm'
        "REG_USE_SNN_INPUT": True,                                              # True = features + SNN spikes;  False = features only
        "REG_DO_TRAIN": False,                                             # True = train;  False = load from checkpoint
        "REG_EPOCHS": 100,                                                # Number of training epochs for the regressor
        "REG_BATCH_SIZE": 256,                                                    # Mini-batch size for training the regressor
        "REG_LR": 1e-3,                                                   # Learning rate for the regressor
        "REG_HIDDEN_SIZE": 64,                                                  # Hidden layer width for the regressor
        "REG_NUM_LAYERS": 2,                                                    # Number of stacked layers for the regressor
        "REG_DROPOUT": 0.2,                                                     # Dropout probability for the regressor (LSTM only)
        "REG_KERNEL_SIZE": 5,                                                   # Convolutional kernel size for the regressor (CNN only)
        "REG_SAVE_PATH": "mdl/best_reg_dev{device_id}.pt",                             # Checkpoint path for the regressor

        # ── Plotting ──────────────────────────────────────────────────────────
        # Disable plots during automated runs (e.g. Optuna sweeps) to save time.
        "PLOT_SNN": True,                                                       # Generate per-device SNN classification plots
        "PLOT_REG": True,                                                # Generate per-device regression plots
        "PLOT_DEBUG_BATCH": True,                                              # Plot the first test batch input/output for model debugging
        "DEBUG_SAMPLE_INDEX": 0,                                                # Sample index inside the debug batch used for detailed views
        "DEBUG_BATCH_PLOT_SAMPLES": 24,                                         # Max number of batch samples shown in debug heatmaps
    }
