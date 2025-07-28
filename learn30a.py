# ------------------ IMPORTS ------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

# ------------------ LOAD DATA ------------------
data = fetch_california_housing()
X = data.data
y = data.target

# Convert to DataFrame (for inspection)
df = pd.DataFrame(X, columns=data.feature_names)
df['Target'] = y

# ------------------ MODEL INIT ------------------
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

# ------------------ FIT MODELS ------------------
linear_model.fit(X, y)
ridge_model.fit(X, y)
lasso_model.fit(X, y)

# ------------------ PREDICTIONS ------------------
y_pred_linear = linear_model.predict(X)
y_pred_ridge = ridge_model.predict(X)
y_pred_lasso = lasso_model.predict(X)

# ------------------ CROSS-VAL MSE ------------------
mse_linear = -np.mean(cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_squared_error'))
mse_ridge = -np.mean(cross_val_score(ridge_model, X, y, cv=5, scoring='neg_mean_squared_error'))
mse_lasso = -np.mean(cross_val_score(lasso_model, X, y, cv=5, scoring='neg_mean_squared_error'))

# ------------------ PLOT ------------------
plt.figure(figsize=(12, 7))
plt.scatter(y_pred_linear, y - y_pred_linear, alpha=0.4, label=f'Linear (MSE={mse_linear:.3f})', color='blue', marker='o')
plt.scatter(y_pred_ridge, y - y_pred_ridge, alpha=0.4, label=f'Ridge (MSE={mse_ridge:.3f})', color='green', marker='s')
plt.scatter(y_pred_lasso, y - y_pred_lasso, alpha=0.4, label=f'Lasso (MSE={mse_lasso:.3f})', color='orange', marker='^')

# Horizontal line at y=0 for reference
plt.axhline(0, color='red', linestyle='--', linewidth=1)

# Labels & Legend
plt.title("Residuals vs Predicted - Linear vs Ridge vs Lasso", fontsize=14)
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals (y - y_pred)", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True)

# Save the plot as PNG image
plt.savefig("ridge_lasso_linear_comparison.png", dpi=300)
plt.show()
