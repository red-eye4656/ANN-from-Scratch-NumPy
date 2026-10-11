import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_preprocessing import X_train_scaled, y_train
from neural_network import forward_propagation

model = np.load("best_model.npz")

w1 = model["w1"]
b1 = model["b1"]
w2 = model["w2"]
b2 = model["b2"]
w3 = model["w3"]
b3 = model["b3"]

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

print("Original training shape:", X_train_scaled.shape)
print("PCA training shape:", X_train_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)




x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)


grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_4d = pca.inverse_transform(grid_2d)


_, _, grid_predictions = forward_propagation(
    grid_4d, w1, b1, w2, b2, w3, b3
)

Z = np.argmax(grid_predictions, axis=1).reshape(xx.shape)


plt.figure(figsize=(9, 6))

plt.contourf(
    xx, yy, Z,
    levels=[-0.5, 0.5, 1.5, 2.5],
    alpha=0.3,
    cmap=plt.cm.Set1
)


scatter = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap=plt.cm.Set1,
    edgecolors="black",
    s=45
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("ANN Decision Boundary using PCA")

plt.legend(
    handles=scatter.legend_elements()[0],
    labels=["Setosa", "Versicolor", "Virginica"],
    title="Iris Classes"
)

plt.tight_layout()
plt.savefig("visualizations/decision_boundary.png", dpi=300)
plt.show()

print("Decision boundary saved to visualizations/decision_boundary.png")
