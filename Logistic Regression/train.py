from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt  # type: ignore
from sklearn import datasets
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)


def accuracy(y_pred, y_test):
    return (np.sum(y_pred == y_test) / len(y_test)) * 100


bc = datasets.load_breast_cancer()
X, y = bc["data"], bc["target"]  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = LogisticRegression(lr=0.00001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: { accuracy(y_pred, y_test):.2f}%")
