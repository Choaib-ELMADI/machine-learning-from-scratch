from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # type: ignore
from LinearRegression import LinearRegression
from sklearn import datasets
import numpy as np


def mean_squared_error(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="black", marker=".", s=30)  # type: ignore

reg = LinearRegression(n_iters=2000)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.3f}")

y_pred_line = reg.predict(X)
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color="red", s=30)  # type: ignore
m2 = plt.scatter(X_test, y_test, color="blue", s=30)  # type: ignore
plt.plot(X, y_pred_line, color="green", linewidth=2, label="Prediction")
plt.show()
