import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt


from helper import plot_sequences, load_data, prepare_dataset, train_test_split
from snnNet import NILM_SNN, train_model, test_model


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Config
    DEVICE_ID = 5          
    POWER_THRESHOLD = 50.0 
    TRAIN_SPLIT = 0.8
    HIDDEN_SIZE = 32
    EPOCHS = 500
    LR = 1e-3

    # Load data
    X, Y = load_data("data/redd3HF.mat", maxLen=50000)

    # Prepare dataset
    x, y = prepare_dataset(X, Y, thres=POWER_THRESHOLD, device_id=DEVICE_ID)
    W, F = x.shape[1], x.shape[2]
    C = 1

    # Split
    x_train, y_train, x_test, y_test = train_test_split(x, y, split=TRAIN_SPLIT)
    print(f"Train inputs shape: {x_train.shape}")
    print(f"Train targets shape: {y_train.shape}")

    # Create model
    model = NILM_SNN(input_size=F, hidden_size=HIDDEN_SIZE, output_size=C)

    # Train model
    print("Training model...")
    losses = train_model(model, x_train, y_train, epochs=EPOCHS, lr=LR)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)

    # Test model
    print("Testing model...")
    predictions, pred_binary = test_model(model, x_test, y_test)

    # Visualize results
    print("Plotting results...")
    plot_sequences(x_test, y_test, predictions)
    plt.show()  

    print("Done!")
