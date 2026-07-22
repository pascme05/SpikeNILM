import torch
import torch.nn as nn
import snntorch as snn


class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.95, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        layers = []

        for layer_index in range(num_layers):
            in_features = input_size if layer_index == 0 else hidden_size
            out_features = output_size if layer_index == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        num_steps = x.size(0)
        membranes = []
        for layer_index in range(self.num_layers):
            lif = self.layers[2 * layer_index + 1]
            membranes.append(lif.init_leaky())

        spike_record = []
        membrane_record = []

        for step in range(num_steps):
            current = x[step]
            for layer_index in range(self.num_layers):
                linear = self.layers[2 * layer_index]
                lif = self.layers[2 * layer_index + 1]
                current = linear(current)
                spikes, membranes[layer_index] = lif(current, membranes[layer_index])
                current = spikes
            spike_record.append(spikes)
            membrane_record.append(membranes[-1])

        return torch.stack(spike_record), torch.stack(membrane_record)


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, kernel_size=5):
        super().__init__()
        blocks = []

        for layer_index in range(num_layers):
            in_channels = input_size if layer_index == 0 else hidden_size
            blocks.append(
                nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            blocks.append(nn.BatchNorm1d(hidden_size))
            blocks.append(nn.ReLU())

        self.convs = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def build_model(
    model_type,
    input_size,
    hidden_size,
    output_size,
    beta=0.95,
    num_layers=2,
    kernel_size=5,
    dropout=0.2,
):
    if model_type == "snn":
        return SpikingNet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            beta=beta,
            num_layers=num_layers,
        )
    if model_type == "cnn":
        return ConvNet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
    if model_type == "lstm":
        return LSTMNet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Use 'snn', 'cnn', or 'lstm'.")
