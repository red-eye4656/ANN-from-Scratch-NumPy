import numpy as np
from sklearn.datasets import load_iris


iris = load_iris()


X = iris.data
y = iris.target

print("Features shape:", X.shape)
print("Target shape:", y.shape)


print("Feature names:", iris.feature_names)
print("class names:", iris.target_names)


class_labels, class_counts = np.unique(y, return_counts=True)
print("class labels:", class_labels)
print("Samples per class:", class_counts)
   

import matplotlib.pyplot as plt

import os 

os.makedirs("visualizations", exist_ok=True)

for i, feature_name in enumerate(iris.feature_names):
    plt.figure(figsize=(6, 4))
    plt.hist(X[: ,1], bins=15, edgecolor="black")
    plt.title(f"Distribution of  {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.tight_layout()


    filename = feature_name.split()[0] + "_" + str(i) + ".png"
    plt.savefig("visualizations/" + filename)
    plt.close()

    



print("histrograms saved in the visualization foler . ")


import seaborn as sns
import pandas as pd

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = iris.target_names[y]

sns.pairplot(df, hue="species", diag_kind="hist")
plt.savefig("visualizations/iris_pairplot.png")
plt.close()



print("Pair plot saved successfully.")
correlation = df[iris.feature_names].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Iris Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/iris_correlation_heatmap.png")
plt.close()

print("Correlation heatmap saved successfully.")
