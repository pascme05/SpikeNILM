from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper import load_data_multi

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    import snntorch as snn
except ImportError:
    snn = None


@dataclass
class DemoConfig:
    mat_file: str = "data/redd3HF.mat"
    device_ids: tuple = tuple(range(20))
    max_len: int = 1500
    fundamental_freq: float = 60.0
    num_harmonics: int = 15
    relative_floor_ratio: float = 0.10
    max_feature_spikes: int = 18
    fixed_num_steps: int = 64
    device_change_threshold_w: float = 1.0
    target_mode: str = "binary_event"
    input_mode: str = "signed"
    event_threshold_w: float = 80.0
    event_threshold_mode: str = "quantile"
    event_threshold_quantile: float = 0.95
    direction_threshold_w: float = 40.0
    direction_threshold_mode: str = "fixed"
    direction_threshold_quantile: float = 0.95
    regression_target: str = "abs_sum"
    train_split: float = 0.8
    split_mode: str = "stratified"
    hidden_size: int = 48
    beta: float = 0.95
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 12
    seed: int = 42
    plot_examples: int = 200
    show_plot: bool = False
    save_plot: bool = True


def infer_sampling_rate(samples_per_cycle, fundamental_freq):
    return int(round(samples_per_cycle * fundamental_freq))


def compute_harmonic_magnitudes(signal, sampling_rate, base_freq=60.0, num_harmonics=15):
    signal = np.asarray(signal, dtype=np.float32)
    n_samples = signal.shape[0]
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    magnitudes = []

    for harmonic_order in range(1, num_harmonics + 1):
        target_freq = harmonic_order * base_freq
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        amplitude = 2.0 * np.abs(fft[idx]) / n_samples
        magnitudes.append(float(amplitude))

    return np.asarray(magnitudes, dtype=np.float32)


def extract_cycle_features(cycle, sampling_rate, config):
    voltage_cycle = cycle[:, 0]
    current_cycle = cycle[:, 1]

    voltage_h = compute_harmonic_magnitudes(
        voltage_cycle,
        sampling_rate=sampling_rate,
        base_freq=config.fundamental_freq,
        num_harmonics=config.num_harmonics,
    )
    current_h = compute_harmonic_magnitudes(
        current_cycle,
        sampling_rate=sampling_rate,
        base_freq=config.fundamental_freq,
        num_harmonics=config.num_harmonics,
    )

    feature_names = [f"V_h{idx}" for idx in range(1, config.num_harmonics + 1)]
    feature_names += [f"I_h{idx}" for idx in range(1, config.num_harmonics + 1)]
    return np.concatenate([voltage_h, current_h]).astype(np.float32), feature_names


def compute_feature_deltas(previous_features, current_features):
    signed_delta = current_features - previous_features
    absolute_delta = np.abs(signed_delta)
    return signed_delta.astype(np.float32), absolute_delta.astype(np.float32)


def build_feature_denominators(previous_features, num_harmonics, relative_floor_ratio=0.10):
    previous_features = np.asarray(previous_features, dtype=np.float32)
    denominators = np.abs(previous_features).copy()
    voltage_floor = relative_floor_ratio * max(float(np.abs(previous_features[0])), 1e-6)
    current_floor = relative_floor_ratio * max(float(np.abs(previous_features[num_harmonics])), 1e-6)

    denominators[:num_harmonics] = np.maximum(denominators[:num_harmonics], voltage_floor)
    denominators[num_harmonics:] = np.maximum(denominators[num_harmonics:], current_floor)
    return denominators


def build_unsigned_change_strength(previous_features, absolute_delta, num_harmonics, relative_floor_ratio=0.10):
    denominators = build_feature_denominators(previous_features, num_harmonics, relative_floor_ratio)
    relative_delta = absolute_delta / (denominators + 1e-6)
    return (1.0 - np.exp(-relative_delta)).astype(np.float32)


def build_signed_change_strength(previous_features, signed_delta, num_harmonics, relative_floor_ratio=0.10):
    denominators = build_feature_denominators(previous_features, num_harmonics, relative_floor_ratio)
    relative_signed_delta = signed_delta / (denominators + 1e-6)
    positive = np.maximum(relative_signed_delta, 0.0)
    negative = np.maximum(-relative_signed_delta, 0.0)
    positive = 1.0 - np.exp(-positive)
    negative = 1.0 - np.exp(-negative)
    return np.concatenate([positive, negative]).astype(np.float32)


def encode_strength_to_fixed_spikes(change_strength, num_steps=64, max_feature_spikes=18):
    change_strength = np.clip(np.asarray(change_strength, dtype=np.float32), 0.0, 1.0)
    n_features = change_strength.shape[0]
    spikes = np.zeros((num_steps, n_features), dtype=np.float32)

    for feature_idx, feature_strength in enumerate(change_strength):
        feature_spike_count = int(round(feature_strength * max_feature_spikes))
        feature_spike_count = int(np.clip(feature_spike_count, 0, num_steps))
        if feature_spike_count == 0:
            continue
        spike_times = np.linspace(0, num_steps - 1, num=feature_spike_count, dtype=int)
        spikes[spike_times, feature_idx] = 1.0

    return spikes


def compute_aggregate_device_targets(Y_devices, device_change_threshold_w=1.0):
    device_ids = sorted(Y_devices.keys())
    device_matrix = np.stack([np.asarray(Y_devices[device_id], dtype=np.float32) for device_id in device_ids], axis=1)
    signed_device_deltas = np.diff(device_matrix, axis=0)
    absolute_device_deltas = np.abs(signed_device_deltas)
    summed_absolute_device_delta = absolute_device_deltas.sum(axis=1)
    summed_signed_device_delta = signed_device_deltas.sum(axis=1)
    summed_signed_device_delta_abs = np.abs(summed_signed_device_delta)
    changed_device_count = (absolute_device_deltas > float(device_change_threshold_w)).sum(axis=1).astype(np.float32)

    return {
        "device_ids": device_ids,
        "device_matrix": device_matrix,
        "signed_device_deltas": signed_device_deltas.astype(np.float32),
        "absolute_device_deltas": absolute_device_deltas.astype(np.float32),
        "summed_absolute_device_delta": summed_absolute_device_delta.astype(np.float32),
        "summed_signed_device_delta": summed_signed_device_delta.astype(np.float32),
        "summed_signed_device_delta_abs": summed_signed_device_delta_abs.astype(np.float32),
        "changed_device_count": changed_device_count,
    }


def resolve_target_thresholds(aggregate_targets, config):
    thresholds = {}

    if config.event_threshold_mode == "quantile":
        thresholds["event_threshold_w"] = float(
            np.quantile(aggregate_targets["summed_absolute_device_delta"], config.event_threshold_quantile)
        )
    else:
        thresholds["event_threshold_w"] = float(config.event_threshold_w)

    if config.direction_threshold_mode == "quantile":
        thresholds["direction_threshold_w"] = float(
            np.quantile(np.abs(aggregate_targets["summed_signed_device_delta"]), config.direction_threshold_quantile)
        )
    else:
        thresholds["direction_threshold_w"] = float(config.direction_threshold_w)

    return thresholds


def make_supervised_target(aggregate_targets, index, config, thresholds):
    abs_sum = float(aggregate_targets["summed_absolute_device_delta"][index])
    signed_sum = float(aggregate_targets["summed_signed_device_delta"][index])

    if config.target_mode == "binary_event":
        target = 1 if abs_sum >= thresholds["event_threshold_w"] else 0
        return target, ["quiet", "event"]

    if config.target_mode == "direction_3class":
        if signed_sum > thresholds["direction_threshold_w"]:
            target = 2
        elif signed_sum < -thresholds["direction_threshold_w"]:
            target = 0
        else:
            target = 1
        return target, ["decrease", "steady", "increase"]

    if config.target_mode == "regression":
        if config.regression_target == "signed_sum":
            return signed_sum, ["signed_sum"]
        return abs_sum, ["abs_sum"]

    raise ValueError(
        f"Unknown target_mode: {config.target_mode}. Use 'binary_event', 'direction_3class', or 'regression'."
    )


def build_supervised_dataset(X, aggregate_targets, sampling_rate, config):
    spike_sequences = []
    targets = []
    raw_strengths = []
    metadata = []
    class_names = None
    feature_names = None
    thresholds = resolve_target_thresholds(aggregate_targets, config)

    for cycle_idx in range(1, len(X)):
        previous_features, feature_names = extract_cycle_features(X[cycle_idx - 1], sampling_rate, config)
        current_features, _ = extract_cycle_features(X[cycle_idx], sampling_rate, config)
        signed_delta, absolute_delta = compute_feature_deltas(previous_features, current_features)

        unsigned_strength = build_unsigned_change_strength(
            previous_features,
            absolute_delta,
            num_harmonics=config.num_harmonics,
            relative_floor_ratio=config.relative_floor_ratio,
        )
        signed_strength = build_signed_change_strength(
            previous_features,
            signed_delta,
            num_harmonics=config.num_harmonics,
            relative_floor_ratio=config.relative_floor_ratio,
        )

        if config.input_mode == "signed":
            change_strength = signed_strength
        elif config.input_mode == "unsigned":
            change_strength = unsigned_strength
        else:
            raise ValueError(f"Unknown input_mode: {config.input_mode}. Use 'signed' or 'unsigned'.")

        spike_sequence = encode_strength_to_fixed_spikes(
            change_strength,
            num_steps=config.fixed_num_steps,
            max_feature_spikes=config.max_feature_spikes,
        )
        target, class_names = make_supervised_target(aggregate_targets, cycle_idx - 1, config, thresholds)

        spike_sequences.append(spike_sequence)
        targets.append(target)
        raw_strengths.append(change_strength)
        metadata.append(
            {
                "transition_index": cycle_idx - 1,
                "sum_abs_delta": float(aggregate_targets["summed_absolute_device_delta"][cycle_idx - 1]),
                "net_delta": float(aggregate_targets["summed_signed_device_delta"][cycle_idx - 1]),
                "changed_device_count": float(aggregate_targets["changed_device_count"][cycle_idx - 1]),
            }
        )

    dataset = {
        "spike_sequences": np.asarray(spike_sequences, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32 if config.target_mode == "regression" else np.int64),
        "raw_strengths": np.asarray(raw_strengths, dtype=np.float32),
        "metadata": metadata,
        "feature_names": feature_names,
        "class_names": class_names,
        "input_size": spike_sequences[0].shape[1],
        "thresholds": thresholds,
    }
    return dataset


def split_dataset(dataset, train_split=0.8, split_mode="chronological", seed=42, is_regression=False):
    n_samples = dataset["spike_sequences"].shape[0]
    if n_samples < 2:
        raise ValueError("Need at least two transitions to split the dataset.")

    if split_mode == "stratified" and not is_regression:
        rng = np.random.default_rng(seed)
        train_indices = []
        test_indices = []
        targets = dataset["targets"]
        for class_id in np.unique(targets):
            class_indices = np.where(targets == class_id)[0]
            shuffled = rng.permutation(class_indices)
            class_train_size = max(1, int(round(len(shuffled) * train_split)))
            class_train_size = min(class_train_size, len(shuffled) - 1) if len(shuffled) > 1 else 1
            train_indices.append(shuffled[:class_train_size])
            test_indices.append(shuffled[class_train_size:])

        train_indices = np.sort(np.concatenate(train_indices))
        test_indices = np.sort(np.concatenate(test_indices))
    else:
        train_size = max(1, int(round(n_samples * train_split)))
        train_size = min(train_size, n_samples - 1)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, n_samples)

    split = {}
    for key in ("spike_sequences", "targets", "raw_strengths"):
        split[f"{key}_train"] = dataset[key][train_indices]
        split[f"{key}_test"] = dataset[key][test_indices]
    split["metadata_train"] = [dataset["metadata"][idx] for idx in train_indices]
    split["metadata_test"] = [dataset["metadata"][idx] for idx in test_indices]
    split["class_names"] = dataset["class_names"]
    split["feature_names"] = dataset["feature_names"]
    split["input_size"] = dataset["input_size"]
    split["thresholds"] = dataset["thresholds"]
    return split


def summarize_dataset(split, config):
    targets_train = split["targets_train"]
    targets_test = split["targets_test"]

    print(f"Prepared supervised dataset with input size {split['input_size']} and {config.fixed_num_steps} time steps.")
    print(f"Train samples: {len(targets_train)}, Test samples: {len(targets_test)}")
    if "event_threshold_w" in split["thresholds"]:
        print(f"Event threshold used: {split['thresholds']['event_threshold_w']:.2f}")
    if "direction_threshold_w" in split["thresholds"]:
        print(f"Direction threshold used: {split['thresholds']['direction_threshold_w']:.2f}")

    if config.target_mode == "regression":
        print(
            f"Train target range: [{float(targets_train.min()):.2f}, {float(targets_train.max()):.2f}], "
            f"Test target range: [{float(targets_test.min()):.2f}, {float(targets_test.max()):.2f}]"
        )
        return

    def describe(labels):
        values, counts = np.unique(labels, return_counts=True)
        return ", ".join(f"{split['class_names'][int(value)]}={int(count)}" for value, count in zip(values, counts))

    print(f"Train class distribution: {describe(targets_train)}")
    print(f"Test class distribution: {describe(targets_test)}")


if torch is not None and snn is not None:
    class TinySupervisedSNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, beta=0.95):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.lif1 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)

        def forward(self, x):
            num_steps = x.size(0)
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk_rec = []
            mem_rec = []

            for step in range(num_steps):
                cur1 = self.fc1(x[step])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk_rec.append(spk2)
                mem_rec.append(mem2)

            return torch.stack(spk_rec), torch.stack(mem_rec)


def build_dataloaders(split, config):
    if torch is None or TensorDataset is None or DataLoader is None:
        raise ImportError("torch is required to build DataLoaders.")

    X_train = torch.tensor(split["spike_sequences_train"], dtype=torch.float32)
    X_test = torch.tensor(split["spike_sequences_test"], dtype=torch.float32)

    if config.target_mode == "regression":
        y_train = torch.tensor(split["targets_train"], dtype=torch.float32)
        y_test = torch.tensor(split["targets_test"], dtype=torch.float32)
    else:
        y_train = torch.tensor(split["targets_train"], dtype=torch.long)
        y_test = torch.tensor(split["targets_test"], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def decode_model_output(mem_rec, config):
    logits = mem_rec.mean(dim=0)
    if config.target_mode == "regression":
        return logits.squeeze(-1)
    return logits


def evaluate_supervised_model(model, loader, config, device):
    model.eval()
    predictions = []
    targets = []
    losses = []

    if config.target_mode == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device).permute(1, 0, 2)
            y_batch = y_batch.to(device)
            _, mem_rec = model(X_batch)
            decoded = decode_model_output(mem_rec, config)
            loss = criterion(decoded, y_batch)
            losses.append(float(loss.item()))

            if config.target_mode == "regression":
                predictions.append(decoded.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
            else:
                predictions.append(decoded.argmax(dim=1).cpu().numpy())
                targets.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    metrics = {"loss": float(np.mean(losses))}

    if config.target_mode == "regression":
        metrics["mae"] = float(np.mean(np.abs(y_pred - y_true)))
        metrics["rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        if y_pred.std() > 0 and y_true.std() > 0:
            metrics["corr"] = float(np.corrcoef(y_pred, y_true)[0, 1])
        else:
            metrics["corr"] = float("nan")
    else:
        metrics["accuracy"] = float(np.mean(y_pred == y_true))
        num_classes = len(loader.dataset.tensors[1].unique())
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_label, pred_label in zip(y_true, y_pred):
            confusion[int(true_label), int(pred_label)] += 1
        metrics["confusion"] = confusion

    return metrics, y_pred, y_true


def train_supervised_snn(split, config):
    if torch is None or snn is None or nn is None:
        raise ImportError("torch and snntorch are required to train the supervised SNN.")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_loader, test_loader = build_dataloaders(split, config)
    output_size = 1 if config.target_mode == "regression" else len(split["class_names"])
    model = TinySupervisedSNN(
        input_size=split["input_size"],
        hidden_size=config.hidden_size,
        output_size=output_size,
        beta=config.beta,
    )
    device = torch.device("cpu")
    model = model.to(device)

    if config.target_mode == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"train_loss": [], "test_loss": []}
    best_state = None
    best_score = float("inf") if config.target_mode == "regression" else -float("inf")

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).permute(1, 0, 2)
            y_batch = y_batch.to(device)

            _, mem_rec = model(X_batch)
            decoded = decode_model_output(mem_rec, config)
            loss = criterion(decoded, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        test_metrics, _, _ = evaluate_supervised_model(model, test_loader, config, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_metrics["loss"])

        if config.target_mode == "regression":
            selection_score = -test_metrics["rmse"]
        else:
            selection_score = test_metrics["accuracy"]

        if selection_score > best_score:
            best_score = selection_score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        if config.target_mode == "regression":
            print(
                f"Epoch {epoch + 1:>2}/{config.epochs}: "
                f"train_loss={train_loss:.4f}, test_rmse={test_metrics['rmse']:.2f}, test_corr={test_metrics['corr']:.3f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:>2}/{config.epochs}: "
                f"train_loss={train_loss:.4f}, test_acc={test_metrics['accuracy']:.3f}"
            )

    model.load_state_dict(best_state)
    final_metrics, y_pred, y_true = evaluate_supervised_model(model, test_loader, config, device)
    return model, history, final_metrics, y_pred, y_true


def plot_supervised_results(split, history, final_metrics, y_pred, y_true, config):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax_loss, ax_pred = axes

    ax_loss.plot(history["train_loss"], label="Train loss")
    ax_loss.plot(history["test_loss"], label="Test loss")
    ax_loss.set_title(f"Supervised SNN training curves ({config.target_mode})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    n_plot = min(config.plot_examples, len(y_true))
    x_axis = np.arange(n_plot)

    if config.target_mode == "regression":
        ax_pred.plot(x_axis, y_true[:n_plot], label="True target", linewidth=1.1)
        ax_pred.plot(x_axis, y_pred[:n_plot], label="Predicted target", linewidth=1.1, alpha=0.8)
        ax_pred.set_title(
            f"Regression test preview, RMSE={final_metrics['rmse']:.2f}, MAE={final_metrics['mae']:.2f}, corr={final_metrics['corr']:.3f}"
        )
    else:
        ax_pred.step(x_axis, y_true[:n_plot], where="mid", label="True class", linewidth=1.1)
        ax_pred.step(x_axis, y_pred[:n_plot], where="mid", label="Predicted class", linewidth=1.1, alpha=0.8)
        ax_pred.set_title(f"Classification test preview, accuracy={final_metrics['accuracy']:.3f}")

    ax_pred.set_xlabel("Test sample index")
    ax_pred.set_ylabel("Target")
    ax_pred.legend()
    ax_pred.grid(True, alpha=0.3)

    plt.tight_layout()

    if config.save_plot:
        plot_path = results_dir / f"snn_small_test_supervised_{config.target_mode}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to: {plot_path}")

    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)


def print_supervised_summary(split, final_metrics, config):
    print(f"\nSupervised SNN summary for target_mode='{config.target_mode}' and input_mode='{config.input_mode}':")

    if config.target_mode == "regression":
        print(
            f"  Test RMSE={final_metrics['rmse']:.2f}, "
            f"MAE={final_metrics['mae']:.2f}, corr={final_metrics['corr']:.3f}"
        )
        return

    print(f"  Test accuracy={final_metrics['accuracy']:.3f}")
    print("  Confusion matrix:")
    for row_idx, row in enumerate(final_metrics["confusion"]):
        label = split["class_names"][row_idx]
        print(f"    {label:>8}: {row.tolist()}")


def main():
    config = DemoConfig()
    X, Y_devices = load_data_multi(config.mat_file, device_ids=config.device_ids, maxLen=config.max_len)
    sampling_rate = infer_sampling_rate(X.shape[1], config.fundamental_freq)
    aggregate_targets = compute_aggregate_device_targets(
        Y_devices,
        device_change_threshold_w=config.device_change_threshold_w,
    )
    dataset = build_supervised_dataset(X, aggregate_targets, sampling_rate, config)
    split = split_dataset(
        dataset,
        train_split=config.train_split,
        split_mode=config.split_mode,
        seed=config.seed,
        is_regression=config.target_mode == "regression",
    )

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded {len(Y_devices)} device target channels")
    print(f"Using sampling rate: {sampling_rate} Hz")
    print(f"Using device IDs: {aggregate_targets['device_ids']}")
    print(f"Target mode: {config.target_mode}")
    print(f"Input mode: {config.input_mode}")
    summarize_dataset(split, config)

    if torch is None or snn is None:
        missing = []
        if torch is None:
            missing.append("torch")
        if snn is None:
            missing.append("snntorch")
        print(
            "\nSupervised SNN code is ready, but this shell cannot train it because "
            f"{', '.join(missing)} is not installed."
        )
        print("Install those packages in your project environment, then run this script again.")
        return

    model, history, final_metrics, y_pred, y_true = train_supervised_snn(split, config)
    _ = model
    print_supervised_summary(split, final_metrics, config)
    plot_supervised_results(split, history, final_metrics, y_pred, y_true, config)


if __name__ == "__main__":
    main()
