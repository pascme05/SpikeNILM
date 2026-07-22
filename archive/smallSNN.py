#######################################################################################################################
#######################################################################################################################
# Title:        Scripting for testing a small SNN for grid simulation
# Topic:        ML & DL Smart Grid
# File:         smallSNN
# Date:         08.10.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
#######################################################################################################################
#######################################################################################################################

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

#######################################################################################################################
# Helper Functions
#######################################################################################################################
def get_config_default():
    """Default experiment setup for the full SNN -> optional LSTM pipeline."""
    return {
        # Data selection.
        "mat_file": "data/redd3HF.mat",
        "dev_ids": -1,  # Use -1 to select all devices; otherwise specify a list of device indices.
        "nt_max": 100000,
        "threshold": 50.0,

        # Execution mode.
        # "train": fit models and save checkpoints only.
        # "test": load checkpoints and evaluate only.
        # "both": fit, save, and evaluate in the same run.
        "run_mode": "both",
        "snn_checkpoint_path": "mdl/smallSNN_snn.pt",
        "lstm_checkpoint_path": "mdl/smallSNN_lstm.pt",

        # SNN input representation.
        # `n_harmonics` keeps only the first low-frequency FFT bins for the SNN.
        # `num_steps` is the spike sequence length used for rate encoding.
        "n_harmonics": 15,
        "encoding": "rate",
        "num_steps": 100,

        # Pipeline mode.
        # "SNN" trains only the classifier.
        # "SNN+LSTM" adds the regression stage on top of predicted device states.
        # "LSTM" trains only the regressor and is valid with harmonics-only features.
        "train_mode": "SNN+LSTM",

        # LSTM feature construction.
        # "harmonics": reduced FFT only; compatible with train_mode="LSTM".
        # "harmonics+state": reduced FFT + predicted binary device states.
        # "encoded+full_fft": full FFT + predicted states; this also forces
        # `num_steps` to the full FFT length inside main().
        "lstm_feature_mode": "harmonics+state",
        # Number of consecutive cycles/samples given to the LSTM as one sequence.
        "lstm_n_cycles": 5,

        # Shared training setup.
        "split": 0.8,
        "batch_size": 256,

        # SNN hyper-parameters.
        "snn_hidden": 64,
        "snn_beta": 0.9,
        "snn_epochs": 50,
        "snn_lr": 1e-3,
        "snn_patience": 10,

        # LSTM hyper-parameters.
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "lstm_epochs": 30,
        "lstm_lr": 1e-3,
        "lstm_patience": 10,

        # Set to 0 for fast runs without figures.
        "plotting": 1,
    }

def get_config_fast():
    """Small debug preset to check code paths quickly without long training."""
    cfg = get_config_default()
    cfg["nt_max"] = 2000
    cfg["snn_epochs"] = 3
    cfg["lstm_epochs"] = 3
    cfg["batch_size"] = 128
    cfg["plotting"] = 0
    return cfg

def load_data(mat_file, id_selector=-1, max_len=10000):
    data = loadmat(mat_file)
    x_data = data["input"]
    y_data = data["output"]

    if max_len is not None and max_len > 0:
        x_data = x_data[:max_len]
        y_data = y_data[:max_len]

    x_data = x_data[:, 2:277, :]

    if id_selector == -1:
        y_data = y_data[:, 1:]
    else:
        y_data = y_data[:, 1 + id_selector]

    return x_data, y_data

def prepare_checkpoint_paths(cfg):
    os.makedirs("mdl", exist_ok=True)
    for key in ["snn_checkpoint_path", "lstm_checkpoint_path"]:
        cfg[key] = os.path.join("mdl", os.path.basename(cfg[key]))
    return cfg

def compute_scalar_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true

    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mean_err = float(np.mean(err))
    max_err = float(np.max(np.abs(err)))

    true_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - true_mean) ** 2))
    ss_res = float(np.sum(err ** 2))
    if ss_tot > 1e-12:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = 1.0 if ss_res <= 1e-12 else float("nan")

    mask = np.abs(y_true) > 1e-8
    if np.any(mask):
        mape = float(np.mean(np.abs(err[mask] / y_true[mask])))
    else:
        mape = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_err": mean_err,
        "r2": r2,
        "mape": mape,
        "max_err": max_err,
    }

def compute_regression_metrics(reg_true, reg_pred, dev_ids):
    metrics = compute_scalar_regression_metrics(reg_true, reg_pred)
    total_metrics = compute_scalar_regression_metrics(reg_true.sum(axis=1), reg_pred.sum(axis=1))

    per_appliance_metrics = []
    for idx, dev_id in enumerate(dev_ids):
        dev_metrics = compute_scalar_regression_metrics(reg_true[:, idx], reg_pred[:, idx])
        dev_metrics["device_id"] = dev_id
        per_appliance_metrics.append(dev_metrics)

    return metrics, total_metrics, per_appliance_metrics

def print_metric_block(title, metrics, indent=""):
    print(f"{indent}{title}")
    print(f"{indent}  RMSE:      {metrics['rmse']:.4f}")
    print(f"{indent}  MSE:       {metrics['mse']:.4f}")
    print(f"{indent}  MAE:       {metrics['mae']:.4f}")
    print(f"{indent}  Mean Err:  {metrics['mean_err']:.4f}")
    print(f"{indent}  R2:        {metrics['r2']:.4f}" if not np.isnan(metrics["r2"]) else f"{indent}  R2:        N/A")
    print(f"{indent}  MAPE:      {metrics['mape']:.2%}" if not np.isnan(metrics["mape"]) else f"{indent}  MAPE:      N/A (true=0)")
    print(f"{indent}  Max Error: {metrics['max_err']:.4f}")

def build_lstm_feature_matrix(feature_mode, v_fft_reduced_norm, i_fft_reduced_norm, v_fft_full_norm, i_fft_full_norm, pred_states):
    # Build one feature vector per cycle/sample before temporal windowing.
    if feature_mode == "harmonics":
        feat_per_sample = np.concatenate((v_fft_reduced_norm, i_fft_reduced_norm), axis=1)
    elif feature_mode == "harmonics+state":
        feat_per_sample = np.concatenate((v_fft_reduced_norm, i_fft_reduced_norm, pred_states), axis=1)
    elif feature_mode == "encoded+full_fft":
        feat_per_sample = np.concatenate((v_fft_full_norm, i_fft_full_norm, pred_states), axis=1)
    else:
        raise ValueError(f"Unknown LSTM feature mode: {feature_mode}")

    return feat_per_sample.astype(np.float32)

def build_sliding_window(feature_matrix, window_size):
    # Convert per-sample features into short sequences for the LSTM.
    n_samples, feat_dim = feature_matrix.shape

    if window_size <= 1:
        return feature_matrix[:, None, :].astype(np.float32)

    pad = np.zeros((window_size - 1, feat_dim), dtype=np.float32)
    feat_padded = np.concatenate((pad, feature_matrix), axis=0)
    windows = np.stack([feat_padded[idx:idx + window_size] for idx in range(n_samples)], axis=0)
    return windows.astype(np.float32)

def train_lstm(lstm_inputs, y_data, split_idx, cfg, device):
    lstm_model = nn.LSTM(
        input_size=lstm_inputs.shape[-1],
        hidden_size=cfg["lstm_hidden"],
        num_layers=cfg["lstm_layers"],
        batch_first=True,
    ).to(device)
    reg_head = nn.Linear(cfg["lstm_hidden"], y_data.shape[-1]).to(device)
    reg_loss_fn = nn.MSELoss()
    reg_optimizer = torch.optim.Adam(list(lstm_model.parameters()) + list(reg_head.parameters()), lr=cfg["lstm_lr"])

    train_losses = []
    val_losses = []
    run_mode = cfg["run_mode"]

    if run_mode in ["train", "both"]:
        y_power_max = np.max(np.abs(y_data), axis=0, keepdims=True)
        y_power_max[y_power_max == 0] = 1.0
        y_data_norm = y_data / y_power_max

        reg_targets = torch.from_numpy(y_data_norm).float()
        reg_train_dataset = TensorDataset(lstm_inputs[:split_idx], reg_targets[:split_idx])
        reg_val_dataset = TensorDataset(lstm_inputs[split_idx:], reg_targets[split_idx:])
        reg_train_loader = DataLoader(reg_train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        reg_val_loader = DataLoader(reg_val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        best_val_loss = float("inf")
        best_lstm_state = None
        best_reg_head_state = None
        no_improve = 0

        for epoch in range(cfg["lstm_epochs"]):
            lstm_model.train()
            reg_head.train()
            epoch_train_loss = 0.0

            for xb, yb in reg_train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                reg_optimizer.zero_grad()
                lstm_out, _ = lstm_model(xb)
                reg_out = reg_head(lstm_out[:, -1, :])
                reg_loss = reg_loss_fn(reg_out, yb)
                reg_loss.backward()
                reg_optimizer.step()
                epoch_train_loss += reg_loss.item() * xb.size(0)

            lstm_model.eval()
            reg_head.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in reg_val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    lstm_out, _ = lstm_model(xb)
                    reg_out = reg_head(lstm_out[:, -1, :])
                    epoch_val_loss += reg_loss_fn(reg_out, yb).item() * xb.size(0)

            epoch_train_loss /= max(len(reg_train_dataset), 1)
            epoch_val_loss /= max(len(reg_val_dataset), 1)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            print(f"LSTM {epoch + 1}/{cfg['lstm_epochs']} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_lstm_state = copy.deepcopy(lstm_model.state_dict())
                best_reg_head_state = copy.deepcopy(reg_head.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["lstm_patience"]:
                    print(f"LSTM early stopping at epoch {epoch + 1}")
                    break

        if best_lstm_state is not None:
            lstm_model.load_state_dict(best_lstm_state)
            reg_head.load_state_dict(best_reg_head_state)

        torch.save(
            {
                "lstm_state_dict": lstm_model.state_dict(),
                "reg_head_state_dict": reg_head.state_dict(),
                "y_power_max": y_power_max,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            cfg["lstm_checkpoint_path"],
        )
        print(f"Saved LSTM checkpoint to {cfg['lstm_checkpoint_path']}")

    elif run_mode == "test":
        if not os.path.exists(cfg["lstm_checkpoint_path"]):
            raise FileNotFoundError(f"Missing LSTM checkpoint: {cfg['lstm_checkpoint_path']}")

        checkpoint = torch.load(cfg["lstm_checkpoint_path"], map_location=device, weights_only=False)
        lstm_model.load_state_dict(checkpoint["lstm_state_dict"])
        reg_head.load_state_dict(checkpoint["reg_head_state_dict"])
        y_power_max = checkpoint["y_power_max"]
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        print(f"Loaded LSTM checkpoint from {cfg['lstm_checkpoint_path']}")

    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")

    lstm_model.eval()
    reg_head.eval()
    with torch.no_grad():
        lstm_out, _ = lstm_model(lstm_inputs[split_idx:].to(device))
        reg_pred_norm = reg_head(lstm_out[:, -1, :]).cpu().numpy()

    reg_pred = reg_pred_norm * y_power_max
    reg_true = y_data[split_idx:]
    reg_metrics, total_metrics, per_appliance_metrics = compute_regression_metrics(reg_true, reg_pred, cfg["dev_ids"])

    return {
        "reg_pred": reg_pred,
        "reg_true": reg_true,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metrics": reg_metrics,
        "total_metrics": total_metrics,
        "per_appliance_metrics": per_appliance_metrics,
        "mae": reg_metrics["mae"],
        "rmse": reg_metrics["rmse"],
    }

def train_snn(snn_inputs, snn_targets, cfg, device):
    split_idx = int(cfg["split"] * snn_inputs.shape[0])

    snn_model = SmallSNN(
        input_dim=snn_inputs.shape[-1],
        output_dim=snn_targets.shape[-1],
        hidden_dim=cfg["snn_hidden"],
        beta=cfg["snn_beta"],
    ).to(device)
    cls_loss_fn = nn.BCEWithLogitsLoss()
    snn_optimizer = torch.optim.Adam(snn_model.parameters(), lr=cfg["snn_lr"])

    train_losses = []
    val_losses = []
    run_mode = cfg["run_mode"]

    if run_mode in ["train", "both"]:
        snn_train_dataset = TensorDataset(snn_inputs[:split_idx], snn_targets[:split_idx])
        snn_val_dataset = TensorDataset(snn_inputs[split_idx:], snn_targets[split_idx:])
        snn_train_loader = DataLoader(snn_train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        snn_val_loader = DataLoader(snn_val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(cfg["snn_epochs"]):
            snn_model.train()
            epoch_train_loss = 0.0

            for xb, yb in snn_train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                snn_optimizer.zero_grad()
                _, mem_rec = snn_model(xb)
                logits = mem_rec.mean(dim=1)
                loss = cls_loss_fn(logits, yb)
                loss.backward()
                snn_optimizer.step()
                epoch_train_loss += loss.item() * xb.size(0)

            snn_model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in snn_val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    _, mem_rec = snn_model(xb)
                    logits = mem_rec.mean(dim=1)
                    epoch_val_loss += cls_loss_fn(logits, yb).item() * xb.size(0)

            epoch_train_loss /= max(len(snn_train_dataset), 1)
            epoch_val_loss /= max(len(snn_val_dataset), 1)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            print(f"SNN {epoch + 1}/{cfg['snn_epochs']} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = copy.deepcopy(snn_model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["snn_patience"]:
                    print(f"SNN early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            snn_model.load_state_dict(best_state)

        torch.save(
            {
                "model_state_dict": snn_model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            cfg["snn_checkpoint_path"],
        )
        print(f"Saved SNN checkpoint to {cfg['snn_checkpoint_path']}")

    elif run_mode == "test":
        if not os.path.exists(cfg["snn_checkpoint_path"]):
            raise FileNotFoundError(f"Missing SNN checkpoint: {cfg['snn_checkpoint_path']}")

        checkpoint = torch.load(cfg["snn_checkpoint_path"], map_location=device, weights_only=False)
        snn_model.load_state_dict(checkpoint["model_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        print(f"Loaded SNN checkpoint from {cfg['snn_checkpoint_path']}")

    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")

    snn_model.eval()
    with torch.no_grad():
        _, mem_all = snn_model(snn_inputs.to(device))
        logits_all = mem_all.mean(dim=1).cpu()
        probs_all = torch.sigmoid(logits_all)
        preds_all = (probs_all >= 0.5).float()

        _, mem_sample = snn_model(snn_inputs[split_idx:split_idx + 1].to(device))
        mem_sample = mem_sample[0].cpu().numpy()

    val_targets = snn_targets[split_idx:]
    val_preds = preds_all[split_idx:]
    y_true = val_targets.numpy().astype(int).ravel()
    y_pred = val_preds.numpy().astype(int).ravel()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    per_node_acc = (val_preds == val_targets).float().mean(dim=0).numpy()

    return {
        "split_idx": split_idx,
        "preds_all": preds_all,
        "val_targets": val_targets,
        "val_preds": val_preds,
        "mem_sample": mem_sample,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_node_acc": per_node_acc,
    }

def plot_results(
    cfg,
    split_idx,
    y_data,
    p_agg,
    i_fft_reduced_log,
    mem_sample,
    val_targets,
    val_preds,
    snn_train_losses,
    snn_val_losses,
    lstm_train_losses,
    lstm_val_losses,
    reg_pred,
    reg_true,
):
    show_snn = len(snn_train_losses) > 0 or len(snn_val_losses) > 0
    show_lstm = len(lstm_train_losses) > 0 or len(lstm_val_losses) > 0

    if show_snn or show_lstm:
        n_rows = int(show_snn) + int(show_lstm)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=False)
        if n_rows == 1:
            axes = [axes]

        axis_idx = 0
        if show_snn:
            axes[axis_idx].plot(snn_train_losses, label="Train")
            axes[axis_idx].plot(snn_val_losses, label="Val")
            axes[axis_idx].set_title("SNN Convergence")
            axes[axis_idx].set_xlabel("Epoch")
            axes[axis_idx].set_ylabel("BCE Loss")
            axes[axis_idx].legend()
            axes[axis_idx].grid(True)
            axis_idx += 1

        if show_lstm:
            axes[axis_idx].plot(lstm_train_losses, label="Train")
            axes[axis_idx].plot(lstm_val_losses, label="Val")
            axes[axis_idx].set_title("LSTM Convergence")
            axes[axis_idx].set_xlabel("Epoch")
            axes[axis_idx].set_ylabel("MSE Loss (norm)")
            axes[axis_idx].legend()
            axes[axis_idx].grid(True)

        plt.tight_layout()
        plt.show()

    if val_targets is not None and val_preds is not None and mem_sample is not None:
        val_len = val_targets.shape[0]
        time_axis = np.arange(val_len)
        p_act_val = p_agg[split_idx:split_idx + val_len]
        p_sum_val = (val_preds.numpy() * y_data[split_idx:split_idx + val_len]).sum(axis=1)
        i_fft_val = i_fft_reduced_log[split_idx:split_idx + val_len, :]
        lif_time = np.arange(mem_sample.shape[0])
        s_pred_val = val_preds.numpy().sum(axis=1)
        s_true_val = val_targets.numpy().sum(axis=1)

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
        axes[0].plot(time_axis, p_act_val, label=r"$P_{agg}$")
        axes[0].plot(time_axis, p_sum_val, label=r"$P_{app}$")
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title("Aggregated Power and Summed Node Power")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xticklabels([])

        axes[1].imshow(i_fft_val.T, aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_ylabel("Harmonic number (f/fel)")
        axes[1].set_title("Input Current Harmonics")
        axes[1].set_xticklabels([])

        axes[2].plot(lif_time, mem_sample)
        axes[2].plot(lif_time, np.ones_like(lif_time), linestyle="--")
        axes[2].set_ylabel("Voltage (V)")
        axes[2].set_title("LIF Membrane Potential")
        axes[2].grid(True)
        axes[2].set_xticklabels([])

        axes[3].plot(time_axis, s_pred_val, label=r"$S_{pred}$")
        axes[3].plot(time_axis, s_true_val, label=r"$S_{true}$")
        axes[3].set_ylabel("State (-)")
        axes[3].set_xlabel("Time (sample)")
        axes[3].set_title("LIF Output vs. Ground Truth")
        axes[3].legend()
        axes[3].grid(True)
        plt.tight_layout()
        plt.show()

    if reg_pred is not None:
        time_axis = np.arange(reg_true.shape[0])
        reg_error = reg_pred - reg_true
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(time_axis, reg_true.sum(axis=1), label="True total power")
        axes[0].plot(time_axis, reg_pred.sum(axis=1), label="Pred total power")
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title("Summed Regression Power")
        axes[0].legend()
        axes[0].grid(True)

        for idx, dev_id in enumerate(cfg["dev_ids"]):
            axes[1].plot(time_axis, reg_error[:, idx], alpha=0.8, label=f"Dev {dev_id}" if idx < 8 else None)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[1].set_ylabel("Error (W)")
        axes[1].set_xlabel("Time (sample)")
        axes[1].set_title("Per-Appliance Regression Error")
        axes[1].grid(True)
        if len(cfg["dev_ids"]) <= 8:
            axes[1].legend()
        plt.tight_layout()
        plt.show()

#######################################################################################################################
# Models
#######################################################################################################################
class SmallSNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x_data):
        batch_size, num_steps, _ = x_data.shape
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x_data.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x_data.device)
        spk_rec = []
        mem_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x_data[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=1), torch.stack(mem_rec, dim=1)

#######################################################################################################################
# Main Code
#######################################################################################################################
def main(cfg=None):
    if cfg is None:
        cfg = get_config_default()
    cfg = prepare_checkpoint_paths(cfg)

    if cfg["run_mode"] not in ["train", "test", "both"]:
        raise ValueError("cfg['run_mode'] must be 'train', 'test', or 'both'.")
    if cfg["train_mode"] not in ["SNN", "SNN+LSTM", "LSTM"]:
        raise ValueError("cfg['train_mode'] must be 'SNN', 'SNN+LSTM', or 'LSTM'.")
    if cfg["train_mode"] == "LSTM" and cfg["lstm_feature_mode"] != "harmonics":
        raise ValueError("train_mode='LSTM' requires lstm_feature_mode='harmonics' because no SNN state inputs are available.")

    x_data, y_all = load_data(cfg["mat_file"], id_selector=-1, max_len=cfg["nt_max"])
    if cfg["dev_ids"] == -1:
        y_data = y_all.astype(np.float32)
        cfg["dev_ids"] = list(range(y_data.shape[1]))
    else:
        y_data = y_all[:, cfg["dev_ids"]].astype(np.float32)
    print("X shape:", x_data.shape)
    print("Y shape:", y_data.shape)
    print(f"Selected devices: {cfg['dev_ids']}")

    i_ac = np.squeeze(x_data[:, :, 1])
    v_ac = np.squeeze(x_data[:, :, 0])

    i_fft_reduced = np.abs(np.fft.rfft(i_ac, axis=1))[:, : cfg["n_harmonics"] + 1] / i_ac.shape[1] * 2
    v_fft_reduced = np.abs(np.fft.rfft(v_ac, axis=1))[:, : cfg["n_harmonics"] + 1] / v_ac.shape[1] * 2
    i_fft_full = np.abs(np.fft.rfft(i_ac, axis=1)) / i_ac.shape[1] * 2
    v_fft_full = np.abs(np.fft.rfft(v_ac, axis=1)) / v_ac.shape[1] * 2

    num_steps = cfg["num_steps"]
    if cfg["lstm_feature_mode"] == "encoded+full_fft":
        num_steps = i_fft_full.shape[1]

    binary_targets = y_data.copy()
    binary_targets[binary_targets < cfg["threshold"]] = 0.0
    binary_targets[binary_targets >= cfg["threshold"]] = 1.0

    p_agg = i_fft_full[:, 1] * v_fft_full[:, 1] / 2

    i_fft_reduced_log = np.log1p(i_fft_reduced)
    v_fft_reduced_log = np.log1p(v_fft_reduced)
    i_fft_full_log = np.log1p(i_fft_full)
    v_fft_full_log = np.log1p(v_fft_full)

    i_fft_reduced_norm = i_fft_reduced_log / (np.max(np.abs(i_fft_reduced_log)) + 1e-8)
    v_fft_reduced_norm = v_fft_reduced_log / (np.max(np.abs(v_fft_reduced_log)) + 1e-8)
    i_fft_full_norm = i_fft_full_log / (np.max(np.abs(i_fft_full_log)) + 1e-8)
    v_fft_full_norm = v_fft_full_log / (np.max(np.abs(v_fft_full_log)) + 1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_idx = int(cfg["split"] * y_data.shape[0])
    snn_result = None
    preds_all = None
    val_targets = None
    val_preds = None
    mem_sample = None
    snn_train_losses = []
    snn_val_losses = []

    if cfg["train_mode"] in ["SNN", "SNN+LSTM"]:
        if cfg["encoding"] != "rate":
            raise ValueError("Only rate encoding is implemented in this prototype.")

        i_spikes = spikegen.rate(torch.from_numpy(i_fft_reduced_norm.astype(np.float32)), num_steps=num_steps)
        v_spikes = spikegen.rate(torch.from_numpy(v_fft_reduced_norm.astype(np.float32)), num_steps=num_steps)
        snn_inputs = torch.cat((v_spikes, i_spikes), dim=2).permute(1, 0, 2).float()
        snn_targets = torch.from_numpy(binary_targets).float()

        snn_result = train_snn(snn_inputs, snn_targets, cfg, device)
        split_idx = snn_result["split_idx"]
        preds_all = snn_result["preds_all"]
        val_targets = snn_result["val_targets"]
        val_preds = snn_result["val_preds"]
        mem_sample = snn_result["mem_sample"]
        snn_train_losses = snn_result["train_losses"]
        snn_val_losses = snn_result["val_losses"]

        if cfg["run_mode"] in ["test", "both"]:
            print("\nClassification Metrics:")
            print(f"Accuracy:  {snn_result['accuracy']:.4f}")
            print(f"Precision: {snn_result['precision']:.4f}")
            print(f"Recall:    {snn_result['recall']:.4f}")
            print(f"F1:        {snn_result['f1']:.4f}")

            print("\nPer-Node Accuracy:")
            for idx, acc in enumerate(snn_result["per_node_acc"]):
                print(f"  Node {cfg['dev_ids'][idx]:2d}: {acc:.4f}")
    else:
        print("Skipping SNN stage because train_mode='LSTM' uses harmonics-only LSTM features.")

    reg_pred = None
    reg_true = None
    lstm_train_losses = []
    lstm_val_losses = []
    lstm_result = None

    if cfg["train_mode"] in ["SNN+LSTM", "LSTM"]:
        feat_per_sample = build_lstm_feature_matrix(
            cfg["lstm_feature_mode"],
            v_fft_reduced_norm,
            i_fft_reduced_norm,
            v_fft_full_norm,
            i_fft_full_norm,
            None if preds_all is None else preds_all.numpy().astype(np.float32),
        )
        lstm_seq = build_sliding_window(feat_per_sample, cfg["lstm_n_cycles"])
        lstm_inputs = torch.from_numpy(lstm_seq)
        lstm_result = train_lstm(lstm_inputs, y_data, split_idx, cfg, device)

        reg_pred = lstm_result["reg_pred"]
        reg_true = lstm_result["reg_true"]
        lstm_train_losses = lstm_result["train_losses"]
        lstm_val_losses = lstm_result["val_losses"]

        if cfg["run_mode"] in ["test", "both"]:
            print("\nRegression Metrics:")
            print_metric_block("Global (all appliance outputs flattened)", lstm_result["metrics"])
            print_metric_block("Total power (sum across appliances)", lstm_result["total_metrics"])

            print("Per-Appliance Metrics:")
            for dev_metrics in lstm_result["per_appliance_metrics"]:
                print_metric_block(f"Device {dev_metrics['device_id']}", dev_metrics, indent="  ")

    if cfg["plotting"] and cfg["run_mode"] in ["test", "both"]:
        plot_results(
            cfg,
            split_idx,
            y_data,
            p_agg,
            i_fft_reduced_log,
            mem_sample,
            val_targets,
            val_preds,
            snn_train_losses,
            snn_val_losses,
            lstm_train_losses,
            lstm_val_losses,
            reg_pred,
            reg_true,
        )

    if cfg["run_mode"] == "train":
        print("Training completed and checkpoints saved. Skipping evaluation/plots in train-only mode.")

    return {
        "snn_result": snn_result,
        "lstm_result": lstm_result if cfg["train_mode"] in ["SNN+LSTM", "LSTM"] else None,
    }

#######################################################################################################################
# Run Code
#######################################################################################################################
if __name__ == "__main__":
    main(get_config_default())
