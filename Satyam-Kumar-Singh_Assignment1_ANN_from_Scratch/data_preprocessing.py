import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_temp, y_train,  y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Training daata:", X_train.shape)
print("validation data:", X_val.shape)
print("Testing data:", X_test.shape)

mean=np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

print("Training mean:", mean)
print("Training standard deviation:", std)

X_train_scaled = (X_train - mean) / std
X_val_scaled = (X_val - mean) / std
X_test_scaled = (X_test - mean) / std

print("scaled training data:", X_train_scaled.shape)
print("scaled validation data:", X_val_scaled.shape)
print("scaled testing data:", X_test_scaled.shape)

y_train_encoded = np.eye(3)[y_train]
y_val_encoded = np.eye(3)[y_val]
y_test_encoded = np.eye(3)[y_test]

print("Encoded traning labels:", y_train_encoded.shape)
print("Encoded validation labels:", y_val_encoded.shape)
print("Encoded testing labels:", y_test_encoded.shape)
