
import numpy as np
import pandas as pd

from data_preprocessing import (
    X_train_scaled, X_val_scaled,
    y_train_encoded, y_val_encoded
)
from neural_network import (
    initialize_parameters,
    forward_propagation,
    backward_propagation,
    update_parameters,
    cross_entropy_loss,
    accuracy_score
)

LEARNING_RATES = [0.001, 0.01, 0.1]
ACTIVATIONS = ["sigmoid", "relu", "tanh"]
ARCHITECTURES = [(8, 6), (16, 8)]

EPOCHS = 500
BATCH_SIZE = 16
results = []


def run_experiment(lr, activation, architecture):
    np.random.seed(42)

    hidden1, hidden2 = architecture
    params = initialize_parameters(hidden1, hidden2)

    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(X_train_scaled))

        for start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            X_batch = X_train_scaled[batch_idx]
            y_batch = y_train_encoded[batch_idx]

            A1, A2, A3, z1, z2 = forward_propagation(
                X_batch, *params,
                activation=activation,
                return_cache=True
            )

            gradients = backward_propagation(
                X_batch, y_batch,
                A1, A2, A3,
                params[2], params[4],
                z1=z1, z2=z2,
                activation=activation
            )

            params = update_parameters(
                *params, *gradients, lr
            )

    _, _, train_pred = forward_propagation(
        X_train_scaled, *params, activation=activation
    )
    _, _, val_pred = forward_propagation(
        X_val_scaled, *params, activation=activation
    )

    return {
        "learning_rate": lr,
        "activation": activation,
        "architecture": f"{hidden1}-{hidden2}",
        "train_loss": cross_entropy_loss(y_train_encoded, train_pred),
        "validation_loss": cross_entropy_loss(y_val_encoded, val_pred),
        "validation_accuracy": accuracy_score(y_val_encoded, val_pred)
    }


for lr in LEARNING_RATES:
    for activation in ACTIVATIONS:
        for architecture in ARCHITECTURES:
            print(
                f"Running: lr={lr}, activation={activation}, "
                f"architecture={architecture}"
            )
            results.append(
                run_experiment(lr, activation, architecture)
            )

df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)

print("\nExperiment results:")
print(df.to_string(index=False))
print("\nSaved to experiment_results.csv")
