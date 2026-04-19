import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# -----------------------------
# Network Class
# -----------------------------
class NILM_SNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        # Input to hidden
        self.fc_in = nn.Linear(input_size, hidden_size)

        # Recurrent layer for temporal processing
        self.lif_rec = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.recurrent = nn.Linear(hidden_size, hidden_size)

        # Hidden to output
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        """
        x: [batch, time, input_size]
        returns: [batch, time, output_size]
        """
        batch_size, time_steps, _ = x.shape

        # Initialize states
        mem_rec = self.lif_rec.init_leaky()
        mem_out = self.lif_out.init_leaky()

        outputs = []

        for t in range(time_steps):
            # Input at time t
            x_t = x[:, t, :]  # [batch, input_size]

            # Input to hidden
            hidden_in = self.fc_in(x_t)

            # Recurrent input
            rec_in = self.recurrent(spk) if t > 0 else torch.zeros_like(hidden_in)

            # Total input
            total_in = hidden_in + rec_in

            # Recurrent layer
            spk, mem_rec = self.lif_rec(total_in, mem_rec)

            # Output layer
            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            outputs.append(mem_out)

        return torch.stack(outputs, dim=1)  # [batch, time, output_size]
    

# -----------------------------
# Training function
# -----------------------------
def train_model(model, inputs, targets, epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # For multi-channel binary classification

    losses = []

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(inputs)  # inputs is [batch, seq_len, F]

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return losses

# -----------------------------
# Testing function
# -----------------------------
def test_model(model, test_inputs, test_targets):
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)  # test_inputs is [batch, seq_len, F]
        predictions = outputs  # [batch, seq_len, C]

        # Calculate accuracy (spike prediction accuracy)
        pred_binary = (predictions > 0.5).float()
        accuracy = (pred_binary == test_targets).float().mean().item()

        print(f"Test Accuracy: {accuracy:.4f}")

        return predictions, pred_binary
    