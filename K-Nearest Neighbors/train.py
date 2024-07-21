from matplotlib.colors import ListedColormap  # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # type: ignore
from sklearn import datasets
from KNN import KNN
import numpy as np

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
iris = datasets.load_iris()
X, y = iris["data"], iris["target"]  # type: ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor="k", s=20)
plt.show()

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(f"{acc * 100:.2f}%")
