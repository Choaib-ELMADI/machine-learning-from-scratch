from sklearn.model_selection import train_test_split
from DecisionTrees import DecisionTree
from sklearn import datasets
import numpy as np


def accuracy(y_pred, y_test):
    return (np.sum(y_pred == y_test) / len(y_test)) * 100


data = datasets.load_breast_cancer()
X, y = data["data"], data["target"]  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: { accuracy(y_pred, y_test):.2f}%")
