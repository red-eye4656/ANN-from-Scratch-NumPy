import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import X_train_scaled, X_test_scaled, y_train, y_test



model = MLPClassifier(
    hidden_layer_sizes=(8, 6),
    activation="logistic",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42
)


model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print("Scikit-learn MLP Test Accuracy:", round(accuracy,4))