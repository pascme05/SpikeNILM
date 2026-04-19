import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# Generate training data
# -----------------------------
def generate_sequence_data(num_samples=1000, seq_length=20, F=1, C=1):
    """
    Generate sequences of spikes for training.
    Input: [seq_length, F] random binary sequences
    Target: [seq_length, C] random binary sequences (for general sequence learning)
    """
    inputs = []
    targets = []

    for _ in range(num_samples):
        # Random binary input sequence [seq_length, F]
        inp_seq = torch.randint(0, 2, (seq_length, F)).float()
        inputs.append(inp_seq)

        # Random binary target sequence [seq_length, C]
        tgt_seq = torch.randint(0, 2, (seq_length, C)).float()
        targets.append(tgt_seq)

    return torch.stack(inputs), torch.stack(targets)

# -----------------------------
# Spiking Neural Network Model
# -----------------------------
class SequenceSNN(nn.Module):
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

# -----------------------------
# Visualization
# -----------------------------
def plot_sequences(inputs, targets, predictions, num_examples=3):
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 8))

    for i in range(num_examples):
        ax = axes[i]
        time_steps = len(inputs[i])

        ax.plot(range(time_steps), inputs[i, :, 0].numpy(), label='Input (feature 0)', marker='o', linestyle='--')
        ax.plot(range(time_steps), targets[i, :, 0].numpy(), label='Target (channel 0)', marker='s', linestyle='-')
        ax.plot(range(time_steps), predictions[i, :, 0].numpy(), label='Prediction (channel 0)', marker='^', linestyle='-.')

        ax.set_title(f'Example {i+1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Spike')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # plt.show()  # Commented out for headless execution

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Config
    F = 2  # Number of input features
    C = 1  # Number of output channels
    HIDDEN_SIZE = 32
    SEQ_LEN = 20
    NUM_SAMPLES_TRAIN = 500
    NUM_SAMPLES_TEST = 100
    EPOCHS = 50
    LR = 1e-3

    # Generate data
    print("Generating training data...")
    train_inputs, train_targets = generate_sequence_data(
        num_samples=NUM_SAMPLES_TRAIN, seq_length=SEQ_LEN, F=F, C=C)
    test_inputs, test_targets = generate_sequence_data(
        num_samples=NUM_SAMPLES_TEST, seq_length=SEQ_LEN, F=F, C=C)

    print(f"Train inputs shape: {train_inputs.shape}")
    print(f"Train targets shape: {train_targets.shape}")

    # Create model
    model = SequenceSNN(input_size=F, hidden_size=HIDDEN_SIZE, output_size=C)

    # Train model
    print("Training model...")
    losses = train_model(model, train_inputs, train_targets, epochs=EPOCHS, lr=LR)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    # plt.show()  # Commented out for headless execution

    # Test model
    print("Testing model...")
    predictions, pred_binary = test_model(model, test_inputs, test_targets)

    # Visualize results
    print("Plotting results...")
    plot_sequences(test_inputs, test_targets, predictions, num_examples=3)

    plt.show()  # Commented out for headless execution

    print("Done!")