import numpy as np 
from data_preprocessing import (
    X_train_scaled, X_val_scaled,
    y_train_encoded, y_val_encoded
)

from neural_network import(
    initialize_parameters, forward_propagation, backward_propagation, cross_entropy_loss, update_parameters, accuracy_score
)

epochs = 1000
learning_rate = 0.01
batch_size = 16

parameters = initialize_parameters()

train_losses = []
validation_losses = []
validation_accuracies = []


best_val_loss = float("inf")
patience = 30
wait = 0

for epoch in range(epochs):
    indices = np.random.permutation(X_train_scaled.shape[0])

    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train_encoded[indices]

    for start in range(0, len(X_shuffled), batch_size):
        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        A1, A2, A3, z1, z2 = forward_propagation(
            X_batch, *parameters, return_cache=True
        )

        gradients = backward_propagation(
            X_batch, y_batch, A1, A2, A3,
            parameters[2], parameters[4],
            z1=z1, z2=z2, activation="sigmoid"
        )

        parameters = update_parameters(*parameters, *gradients, learning_rate)



       
    _, _, train_pred = forward_propagation(X_train_scaled, *parameters)
    train_loss = cross_entropy_loss(y_train_encoded, train_pred)

    _, _, val_pred = forward_propagation(X_val_scaled, *parameters)
    val_loss = cross_entropy_loss(y_val_encoded, val_pred)
    val_acc = accuracy_score(y_val_encoded, val_pred)

    train_losses.append(train_loss)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_acc)

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_parameters = [p.copy() for p in parameters]
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            parameters = best_parameters
            break


parameters = best_parameters

np.savez(
    "best_model.npz",
    w1=parameters[0],
    b1=parameters[1],
    w2=parameters[2],
    b2=parameters[3],
    w3=parameters[4],
    b3=parameters[5],

)

print("Best model saved to best_model.npz")



import matplotlib.pyplot as plt
import pandas as pd


history = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "validation_loss": validation_losses,
    "validation_accuracy": validation_accuracies
})


history.to_csv("training_history.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
plt.plot(history["epoch"], history["validation_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/training_loss.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history["epoch"], history["validation_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/validation_accuracy.png")
plt.close()

print("Training history saved to training_history.csv")
print("Training graphs saved to visualizations/")