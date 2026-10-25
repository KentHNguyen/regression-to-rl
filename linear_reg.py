"""
Part 1: Supervised learning for linear regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


def rmse(y, y_hat):
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2))


# ---------- (1) Load the training dataset ----------
X = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7])
Y = np.array([15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

# ---------- (2) Scatter plot ----------
plt.figure(figsize=(6, 4))
plt.scatter(X, Y, color="blue", label="training data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training dataset")
plt.legend()
plt.show()

# ---------- (3) Train with LinearRegression() ----------
# The data is U-shaped, so a straight line fits poorly.
# We add polynomial features (x, x^2) and still use LinearRegression,
# since the model is linear in its weights w0, w1, w2.
DEGREE = 2
poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))

model = LinearRegression()
model.fit(X_poly, Y)
Y_hat = model.predict(X_poly)

# ---------- (4) Coefficients and equation ----------
w = np.concatenate(([model.intercept_], model.coef_))
print("Coefficient vector [w0, w1, ..., wn]:", np.round(w, 4))
terms = [f"{w[0]:.4f}"] + [f"{w[i]:+.4f}x^{i}" if i > 1 else f"{w[i]:+.4f}x"
                           for i in range(1, len(w))]
print("Equation: y = " + " ".join(terms))

# ---------- (5) Training RMSE ----------
print(f"Training RMSE (degree {DEGREE}): {rmse(Y, Y_hat):.4f}")

# For comparison: a plain straight line (degree 1)
line = LinearRegression().fit(X.reshape(-1, 1), Y)
print(f"Training RMSE (straight line): {rmse(Y, line.predict(X.reshape(-1, 1))):.4f}")

# Plot the fitted curve over the data
xs = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.figure(figsize=(6, 4))
plt.scatter(X, Y, color="blue", label="training data")
plt.plot(xs, model.predict(poly.transform(xs)), color="red", label=f"degree {DEGREE} fit")
plt.plot(xs, line.predict(xs), color="gray", linestyle="--", label="straight line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression fit")
plt.legend()
plt.show()

# ---------- (6) GasProperties.csv with least squares ----------
print("\n--- GasProperties.csv ---")
df = pd.read_csv("GasProperties.csv")
print("Columns:", list(df.columns))

# Last column is the target y, all other columns are features
X_gas = df.iloc[:, :-1].to_numpy(dtype=float)
y_gas = df.iloc[:, -1].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X_gas, y_gas, test_size=0.2, random_state=42)

# Add a column of 1s so w0 acts as the intercept
X_train_b = np.c_[np.ones(len(X_train)), X_train]
X_test_b = np.c_[np.ones(len(X_test)), X_test]

# Least squares: w = (X^T X)^-1 X^T y
w_gas = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
print("Coefficients [w0, w1, ..., wn]:", np.round(w_gas, 4))

train_rmse = rmse(y_train, X_train_b @ w_gas)
test_rmse = rmse(y_test, X_test_b @ w_gas)
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE:  {test_rmse:.4f}")
