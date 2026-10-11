import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import X_test_scaled, y_test, y_test_encoded
from neural_network import forward_propagation, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns



model = np.load("best_model.npz")

w1 = model["w1"]
b1 = model["b1"]
w2 = model["w2"]
b2 = model["b2"]
w3 = model["w3"]
b3 = model["b3"]

A1, A2, predictions = forward_propagation(
    X_test_scaled, w1, b1, w2, b2, w3, b3

)

test_accuracy = accuracy_score(y_test_encoded, predictions)
predicted_classes = np.argmax(predictions, axis=1)

print("\nTest Accuracy:", round(test_accuracy, 4))
cm = confusion_matrix(y_test, predicted_classes)

print("\nConfusion Matrix:")
print(cm)

class_names = ["Setosa", "Versicolor", "Virginica"]

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("visualizations/confusion_matrix.png", dpi=300)
plt.close()

print("Confusion matrix heatmap saved.")
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=["Setosa", "Versicolor", "Virginica"],
                            zero_division=0
                     )
              )