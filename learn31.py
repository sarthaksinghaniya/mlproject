# -*- coding: utf-8 -*-
"""
Regression Analysis with Residual Plots
Linear vs Ridge vs Lasso Comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# =====================
# 1. DATA PREPARATION
# =====================
# Generate synthetic regression dataset (better for demonstration)
X, y = make_regression(
    n_samples=500,  # Total samples
    n_features=20,  # Features (some will be redundant)
    n_informative=15,  # Actually useful features
    noise=20,        # Add realistic noise
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =====================
# 2. MODEL TRAINING
# =====================
# Initialize models with comparable complexity
linear = LinearRegression()
ridge = Ridge(alpha=10.0)     # Regularization strength
lasso = Lasso(alpha=0.1)      # Regularization strength

# Train all models
linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# =====================
# 3. PREDICTIONS & METRICS
# =====================
# Generate predictions
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# Calculate residuals (actual - predicted)
res_linear = y_test - y_pred_linear
res_ridge = y_test - y_pred_ridge
res_lasso = y_test - y_pred_lasso

# Calculate Mean Squared Errors
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# =====================
# 4. RESIDUAL PLOTS
# =====================
# Setup plot style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Plot configurations
models = [
    (y_pred_linear, res_linear, mse_linear, 'Linear', 'o', 'royalblue'),
    (y_pred_ridge, res_ridge, mse_ridge, 'Ridge (L2)', '^', 'forestgreen'),
    (y_pred_lasso, res_lasso, mse_lasso, 'Lasso (L1)', 's', 'crimson')
]

# Plot each model's residuals
for pred, res, mse, label, marker, color in models:
    ax.scatter(
        pred, res,
        marker=marker,
        edgecolor='w',
        alpha=0.7,
        s=80,
        color=color,
        label=f"{label} (MSE: {mse:.1f})"
    )

# Add critical reference lines
ax.axhline(y=0, color='k', linestyle='-', linewidth=2)  # Zero error line
ax.axhline(y=np.mean(res_linear), color='royalblue', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(res_ridge), color='forestgreen', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(res_lasso), color='crimson', linestyle='--', alpha=0.5)

# Annotations and labels
ax.set_title('Regression Residual Analysis', fontsize=16, pad=20)
ax.set_xlabel('Predicted Values', fontsize=12)
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
ax.legend(title='Regression Models', title_fontsize=13, fontsize=11)

# ... (previous code remains the same until the theory_text section)

# =====================
# 5. PDF REPORT GENERATION
# =====================
with PdfPages('Regression_Analysis_Report.pdf') as pdf:
    # ... (title page code remains the same)
    
    # Fixed Theory Explanation Page
    fig_theory = plt.figure(figsize=(11, 8))
    plt.axis('off')
    theory_text = [
        ("REGRESSION THEORY SUMMARY", 20, 'bold'),
        ("\n\n", 12, 'normal'),  # FIXED: Added missing third value
        ("Linear Regression (Ordinary Least Squares)", 16, 'bold'),
        ("\n- Models relationship between features and target\n"
         "- Minimizes sum of squared residuals\n"
         "- Prone to overfitting with many features\n", 14, 'normal'),
        
        ("Ridge Regression (L2 Regularization)", 16, 'bold'),
        ("\n- Adds L2 penalty term (squared coefficients)\n"
         "- Shrinks coefficients but never to exactly zero\n"
         "- Reduces model variance, helps prevent overfitting\n", 14, 'normal'),
        
        ("Lasso Regression (L1 Regularization)", 16, 'bold'),
        ("\n- Adds L1 penalty term (absolute coefficients)\n"
         "- Can force coefficients to exactly zero\n"
         "- Performs automatic feature selection\n", 14, 'normal'),
        
        ("\nMSE INTERPRETATION", 16, 'bold'),
        ("\n- Mean Squared Error measures prediction accuracy\n"
         "- Lower values indicate better performance\n"
         "- Residual plots show error distribution patterns", 14, 'normal'),
        
        ("\nRESIDUAL PLOT DIAGNOSTICS", 16, 'bold'),
        ("\n- Ideal: Random scatter around zero line\n"
         "- Patterns indicate model deficiencies\n"
         "- Clustering suggests unmodeled relationships", 14, 'normal')
    ]
    
    y_position = 0.95
    for text, size, weight in theory_text:
        plt.figtext(
            0.1, y_position, 
            text, 
            ha='left', 
            fontsize=size, 
            weight=weight
        )
        y_position -= 0.07 if size < 16 else 0.09
    
    pdf.savefig(fig_theory)
    plt.close()
    
    # ... (rest of the code remains the same)
    
    # Save residual plot
    pdf.savefig(fig)
    
    # Individual Model Plots
    for pred, res, mse, label, marker, color in models:
        fig_ind = plt.figure(figsize=(10, 6))
        plt.scatter(pred, res, c=color, marker=marker, s=70, alpha=0.8, edgecolor='w')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=2)
        plt.title(f'{label} Regression Residuals\n(MSE: {mse:.1f})', fontsize=14)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        pdf.savefig(fig_ind)
        plt.close()

# Save main plot to image
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300)
plt.show()