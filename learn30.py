from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pandas as pd

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Convert to DataFrame to view
df = pd.DataFrame(X, columns=data.feature_names)
df['Target'] = y
print(df.head())  # ✅ Now it will work

# Ridge regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
y_pred = ridge_model.predict(X)

# Residuals
residuals = y - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (California Housing)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)
y_pred_lasso = lasso_model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lasso, y - y_pred_lasso, alpha=0.5, color='orange', label='Lasso Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted (Lasso)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.grid()
plt.show()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Linear regression for comparison
linear_model = LinearRegression()
cv_scores = cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_squared_error')
mse_linear = -np.mean(cv_scores)
print(f'Linear Regression MSE: {mse_linear}')

mse_ridge = -np.mean(cross_val_score(ridge_model, X, y, cv=5, scoring='neg_mean_squared_error'))
print(f'Ridge Regression MSE: {mse_ridge}')

mse_lasso = -np.mean(cross_val_score(lasso_model, X, y, cv=5, scoring='neg_mean_squared_error'))
print(f'Lasso Regression MSE: {mse_lasso}')
