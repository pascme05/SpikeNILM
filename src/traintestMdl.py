#######################################################################################################################
#######################################################################################################################
# Title:        Spike NILM
# Topic:        Non-intrusive load monitoring
# File:         traintestMdl.py
# Date:         02.08.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Training and testing models
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
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import snntorch as snn
import copy

try:
    from snntorch import spikegen
    from snntorch import surrogate
except ImportError:
    spikegen = None


#######################################################################################################################
# DNN Functions
#######################################################################################################################
# ==============================================================================
# FNC: DNN Classifier
# ==============================================================================
class modelDNN(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_sizes, output_size, dropout=0.2, output_mode="s2p"):

        super().__init__()
        self.output_mode = output_mode.lower()
        self.sequence_length = sequence_length
        self.input_size = input_size

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        layers = []
        in_features = input_size * sequence_length

        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout)])
            in_features = hidden

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, output_size)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, sequence, features).")

        batch_size = x.size(0)

        # Flatten sequence
        x = x.reshape(batch_size, -1)
        x = self.feature_extractor(x)
        logits = self.output_layer(x)

        return logits


# ==============================================================================
# FNC: LSTM Model
# ==============================================================================
class modelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2, output_mode="s2p"):
        super().__init__()

        self.output_mode = output_mode.lower()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, sequence, features).")

        # outputs: (batch, seq, hidden)
        outputs, _ = self.lstm(x)

        # logits for every timestep
        logits = self.output_layer(outputs)

        if self.output_mode == "s2p":
            return logits[:, -1, :]

        elif self.output_mode == "s2s":
            return logits

        else:
            raise ValueError("output_mode must be 's2p' or 's2s'.")


# ==============================================================================
# FNC: DNN Training Loop
# ==============================================================================
def trainDnn(mdl, X_train, y_train, X_val, y_val, opt, loss_fnc, cfg, device, PATH, EPOCH=50, BATCH=64, PATIENCE=10):
    # ------------------------------------------
    # Description
    # ------------------------------------------

    # ------------------------------------------
    # Init
    # ------------------------------------------
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # ------------------------------------------
    # Data Prep
    # ------------------------------------------
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32))
    loader_kwargs = {"batch_size": BATCH, "num_workers": cfg.get("NUM_WORKERS", 0), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # ------------------------------------------
    # Learning Rate Scheduler
    # ------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.get("SNN_LR_FACTOR", 0.5),
                                                           patience=cfg.get("SNN_LR_PATIENCE", 5),
                                                           min_lr=cfg.get("SNN_MIN_LR", 1e-6))

    # ------------------------------------------
    # Train
    # ------------------------------------------
    for epoch in range(EPOCH):
        # Init
        mdl.train()
        epoch_train_loss = 0.0

        # Loop over Batches
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            opt.zero_grad()
            y_hat = mdl(xb)
            loss = loss_fnc(y_hat, yb)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item() * xb.size(0)

        # Eval Validation data
        mdl.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                y_hat = mdl(xb)
                epoch_val_loss += loss_fnc(y_hat, yb).item() * xb.size(0)

        # Average losses
        epoch_train_loss /= max(len(train_dataset), 1)
        epoch_val_loss /= max(len(val_dataset), 1)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Update learning rate
        scheduler.step(epoch_val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(
            f"SNN {epoch + 1:3d}/{EPOCH} | "
            f"Train: {epoch_train_loss:.5f} | "
            f"Val: {epoch_val_loss:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        # --------------------------
        # Save best model
        # --------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(mdl.state_dict())
            no_improve = 0

        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # ------------------------------------------
    # Restore Best Model
    # ------------------------------------------
    if best_state is not None:
        mdl.load_state_dict(best_state)

    # ------------------------------------------
    # Save Checkpoint
    # ------------------------------------------
    torch.save(
        {
            "model_state_dict": mdl.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rate": opt.param_groups[0]["lr"],
        },
        PATH,
    )

    print(f"\nSaved DNN checkpoint to {PATH}")

    return mdl


# ==============================================================================
# FNC: DNN Testing Loop
# ==============================================================================
def testDnn(mdl, X_test, cfg, device, PATH, load_checkpoint=True):
    # ------------------------------------------
    # Load Checkpoint
    # ------------------------------------------
    if load_checkpoint:
        if not os.path.exists(PATH):
            raise FileNotFoundError(f"Missing checkpoint: {PATH}")

        checkpoint = torch.load(PATH, map_location=device, weights_only=False)
        mdl.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {PATH}")

    # ------------------------------------------
    # Evaluation
    # ------------------------------------------
    mdl.eval()

    with torch.no_grad():
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_hat = mdl(X_test)
        # threshold = float(np.asarray(cfg["THRESHOLD"]).squeeze())
        probabilities = torch.sigmoid(y_hat)
        predictions = (probabilities >= 0.5).to(torch.int64)

    return {"logits": y_hat.cpu().numpy(), "probabilities": probabilities.cpu().numpy(), "predictions": predictions.cpu().numpy()}


#######################################################################################################################
# SNN Functions
#######################################################################################################################
# ==============================================================================
# SNN s2s and s2p
# ==============================================================================
def get_logits(mem_rec, mode):
    mode = mode.lower()

    if mode == "s2p":
        return mem_rec.mean(dim=1)

    if mode == "s2s":
        return mem_rec

    raise ValueError("mode must be 's2p' or 's2s'")


# ==============================================================================
# SNN Model
# ==============================================================================
class modelSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, beta=0.95, dropout=0.2):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_layers = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_size

        for _ in range(num_layers):
            self.hidden_layers.append(nn.Sequential(nn.Linear(in_features, hidden_size), nn.LayerNorm(hidden_size)))
            self.hidden_lifs.append(snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid()))
            self.dropouts.append(nn.Dropout(dropout))
            in_features = hidden_size

        # Residual classifier
        self.output_layer = nn.Sequential(nn.Linear(hidden_size + input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Input must be [batch, sequence, features]")

        hidden_memories = [lif.init_leaky() for lif in self.hidden_lifs]
        outputs = []
        T = x.shape[1]

        for t in range(T):
            # Current FFT
            current_input = x[:, t]
            activations = current_input

            for i, (layer, lif, dropout) in enumerate(zip(self.hidden_layers, self.hidden_lifs, self.dropouts)):
                current = layer(activations)
                spikes, hidden_memories[i] = lif(current, hidden_memories[i])
                activations = dropout(hidden_memories[i])

            # Residual connection
            classifier_input = torch.cat([activations, current_input], dim=1,)
            logits = self.output_layer(classifier_input)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)


# ==============================================================================
# Spike Encoding
# ==============================================================================
def encode(X, coding):
    """Convert normalized sequence features to the configured SNN input coding."""
    coding = coding.lower()
    if coding in {"raw", "current"}:
        return X

    if coding == "rate":
        # return spikegen.rate(X.clamp(0.0, 1.0), num_steps=100)
        return spikegen.rate(X.clamp(0.0, 1.0), time_var_input=True)

    if coding == "latency":
        return spikegen.latency(X.clamp(0.0, 1.0), num_steps=100, normalize=True, linear=True)

    if coding == "delta":
        return torch.diff(X, dim=1, prepend=X[:, :1]).abs()

    raise ValueError(f"Unknown SNN_CODING: {coding}. Use 'raw', 'rate', or 'delta'.")


# ==============================================================================
# FNC: SNN Training Loop
# ==============================================================================
def trainSNN(mdl, X_train, y_train, X_val, y_val, opt, loss_fnc, cfg, device):
    # ------------------------------------------
    # Init
    # ------------------------------------------
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # ------------------------------------------
    # Data
    # ------------------------------------------
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32))
    loader_kwargs = {"batch_size": cfg["SNN_BATCH_SIZE"], "num_workers": cfg.get("NUM_WORKERS", 0), "pin_memory": device.type == "cuda"}
    snn_train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    snn_val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # ------------------------------------------
    # Learning Rate Scheduler
    # ------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.get("SNN_LR_FACTOR", 0.5),
                                                           patience=cfg.get("SNN_LR_PATIENCE", 5),
                                                           min_lr=cfg.get("SNN_MIN_LR", 1e-6))

    # ------------------------------------------
    # Training Loop
    # ------------------------------------------
    for epoch in range(cfg["SNN_EPOCHS"]):
        # --------------------------
        # Training
        # --------------------------
        mdl.train()
        epoch_train_loss = 0.0

        for xb, yb in snn_train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            opt.zero_grad()
            logits = mdl(encode(xb, cfg["SNN_CODING"]))
            loss = loss_fnc(get_logits(logits, cfg["SNN_MODE"]), yb)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=cfg.get("SNN_GRAD_CLIP", 1.0))
            opt.step()
            epoch_train_loss += loss.item() * xb.size(0)

        # --------------------------
        # Validation
        # --------------------------
        mdl.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for xb, yb in snn_val_loader:
                xb = xb.to(device, non_blocking=device.type == "cuda")
                yb = yb.to(device, non_blocking=device.type == "cuda")
                logits = mdl(encode(xb, cfg["SNN_CODING"]))
                loss = loss_fnc(get_logits(logits, cfg["SNN_MODE"]), yb)
                epoch_val_loss += loss.item() * xb.size(0)

        # --------------------------
        # Average losses
        # --------------------------
        epoch_train_loss /= max(len(train_dataset), 1)
        epoch_val_loss /= max(len(val_dataset), 1)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # --------------------------
        # Update learning rate
        # --------------------------
        scheduler.step(epoch_val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(
            f"SNN {epoch+1:3d}/{cfg['SNN_EPOCHS']} | "
            f"Train: {epoch_train_loss:.5f} | "
            f"Val: {epoch_val_loss:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        # --------------------------
        # Save best model
        # --------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(mdl.state_dict())
            no_improve = 0

        else:
            no_improve += 1
            if no_improve >= cfg.get("SNN_PATIENCE", cfg["SNN_EPOCHS"]):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # ------------------------------------------
    # Restore Best Model
    # ------------------------------------------
    if best_state is not None:
        mdl.load_state_dict(best_state)

    # ------------------------------------------
    # Save Checkpoint
    # ------------------------------------------
    torch.save(
        {
            "model_state_dict": mdl.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rate": opt.param_groups[0]["lr"],
        },
        cfg["SNN_SAVE_PATH"],
    )

    print(f"\nSaved SNN checkpoint to {cfg['SNN_SAVE_PATH']}")

    return mdl


# ==============================================================================
# FNC: SNN Testing Loop
# ==============================================================================
def testSNN(mdl, X_test, cfg, device, load_checkpoint=True):
    # ------------------------------------------
    # Load Checkpoint
    # ------------------------------------------
    if load_checkpoint:
        if not os.path.exists(cfg["SNN_SAVE_PATH"]):
            raise FileNotFoundError(f"Missing SNN checkpoint: {cfg['SNN_SAVE_PATH']}")

        checkpoint = torch.load(cfg["SNN_SAVE_PATH"], map_location=device, weights_only=False)
        mdl.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded SNN checkpoint from {cfg['SNN_SAVE_PATH']}")

    # ------------------------------------------
    # Evaluation
    # ------------------------------------------
    mdl.eval()

    with torch.no_grad():
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        logits = mdl(encode(X_test, cfg["SNN_CODING"]))
        logits = get_logits(logits, cfg["SNN_MODE"])
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(torch.int64)

    return {"logits": logits.cpu().numpy(), "probabilities": probabilities.cpu().numpy(), "predictions": predictions.cpu().numpy()}
