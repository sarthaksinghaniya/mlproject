# -*- coding: utf-8 -*-
"""
COMPLETE REGRESSION ANALYSIS SCRIPT
Linear vs Ridge vs Lasso Comparison with Enhanced Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# =====================
# 1. SETUP AND CONFIGURATION 
# =====================
plt.style.use('seaborn')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
sns.set_style("whitegrid")

# =====================
# 2. DATA PREPARATION
# =====================
# Generate synthetic data with some correlated features
X, y = make_regression(
    n_samples=500,
    n_features=15,
    n_informative=8,
    noise=30,
    random_state=42
)

# Add some multicollinearity
X[:, 3] = X[:, 0] * 0.7 + np.random.normal(0, 0.5, X.shape[0])
X[:, 7] = X[:, 1] * 0.4 - X[:, 2] * 0.3 + np.random.normal(0, 0.3, X.shape[0])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =====================
# 3. MODEL TRAINING
# =====================
models = {
    'Linear': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=25.0),
    'Lasso (L1)': Lasso(alpha=0.5)
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)

# Generate predictions
predictions = {
    name: model.predict(X_test)
    for name, model in models.items()
}

# Calculate metrics
metrics = {
    name: {
        'MSE': mean_squared_error(y_test, pred),
        'R2': r2_score(y_test, pred)
    }
    for name, pred in predictions.items()
}

# =====================
# 4. VISUALIZATIONS
# =====================
def create_all_plots():
    """Generate all visualization plots"""
    
    # Color scheme
    colors = {
        'Linear': '#1f77b4',
        'Ridge (L2)': '#2ca02c', 
        'Lasso (L1)': '#d62728'
    }
    
    # 1. Residual Plot
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for name, pred in predictions.items():
        residuals = y_test - pred
        ax1.scatter(
            pred, residuals,
            label=f"{name} (MSE: {metrics[name]['MSE']:.1f}, R²: {metrics[name]['R2']:.2f})",
            color=colors[name],
            alpha=0.7,
            s=100
        )
    ax1.axhline(0, color='k', linestyle='--')
    ax1.set_title('Residual Analysis', fontsize=18)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.legend()
    
    # 2. Actual vs Predicted
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for name, pred in predictions.items():
        ax2.scatter(
            y_test, pred,
            label=name,
            color=colors[name],
            alpha=0.5
        )
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax2.set_title('Actual vs Predicted Values', fontsize=18)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.legend()
    
    # 3. Coefficient Comparison
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    x = np.arange(X.shape[1])
    width = 0.25
    
    for i, (name, model) in enumerate(models.items()):
        ax3.bar(
            x + i*width, 
            model.coef_, 
            width, 
            label=name, 
            color=colors[name]
        )
    
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'F{i+1}' for i in range(X.shape[1])])
    ax3.set_title('Feature Coefficients Comparison', fontsize=18)
    ax3.axhline(0, color='k', linewidth=0.8)
    ax3.legend()
    
    # 4. Pair Plot Comparison
    results_df = pd.DataFrame({
        'Actual': y_test,
        **{f'{name} Predicted': pred for name, pred in predictions.items()}
    })
    
    fig4 = sns.pairplot(
        results_df,
        x_vars=['Actual'],
        y_vars=[f'{name} Predicted' for name in predictions.keys()],
        kind='reg',
        height=4,
        aspect=1.2
    )
    fig4.fig.suptitle('Pairwise Comparison with Actual Values', y=1.02)
    
    return fig1, fig2, fig3, fig4

# =====================
# 5. GENERATE PDF REPORT
# =====================
with PdfPages('Regression_Comparison_Report.pdf') as pdf:
    # Title Page
    fig_title = plt.figure(figsize=(11, 8))
    plt.suptitle('Regression Model Comparison Report\nLinear vs Ridge vs Lasso', 
                fontsize=20, fontweight='bold')
    plt.figtext(0.5, 0.6, 
               "Analysis includes:\n"
               "- Residual plots\n"
               "- Actual vs Predicted comparisons\n"
               "- Coefficient analysis\n"
               "- Pair plots", 
               ha='center', fontsize=14)
    plt.axis('off')
    pdf.savefig(fig_title, bbox_inches='tight')
    plt.close()
    
    # Save all plots
    for fig in create_all_plots():
        if isinstance(fig, sns.PairGrid):
            pdf.savefig(fig.fig, bbox_inches='tight')
        else:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print("Report successfully generated: Regression_Comparison_Report.pdf")