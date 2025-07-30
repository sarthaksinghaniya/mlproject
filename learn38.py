import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Sample DataFrame
data = {
    'R&D Spend': [165349, 162597, 153441, 144372, 142107, 131876, 134615, 130298, 120542, 123335, 
                  101913, 100672, 93863.0, 91992.4],
    'Administration': [136897, 151377, 101145, 118671, 91391, 99814, 147198, 145530, 148719, 108679, 
                      118924, 91290.6, 127230, 110404],
    'Marketing Spend': [471784, 443898, 407934, 383199, 366168, 362861, 127716, 323876, 311613, 304032, 
                       223761, 249765, 248229, 250366],
    'State': ['New York', 'California', 'Florida', 'New York', 'Florida', 'New York', 'California', 
              'Florida', 'New York', 'California', 'Florida', 'California', 'Florida', 'California'],
    'Profit': [192261, 191792, 191050, 182902, 166187, 156991, 156122, 155752, 152212, 149760, 
               146122, 144259, 143266, 114809]
}
df = pd.DataFrame(data)

# Prepare features and target
x = df.drop('Profit', axis=1)
y = df['Profit']

# Convert categorical 'State' to dummy variables
states = pd.get_dummies(x['State'], drop_first=True)
x = x.drop('State', axis=1)
x = pd.concat([x, states], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

predictions = {}
metrics = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions[name] = model.predict(X_test)
    
    # Calculate metrics
    metrics[name] = {
        'MSE': mean_squared_error(y_test, predictions[name]),
        'R2': model.score(X_test, y_test)  # Equivalent to r2_score
    }

# Get the Linear Regression model specifically
linear_regressor = models['Linear']

# Print comprehensive results
print("="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

# Print metrics for all models
for model_name, score_dict in metrics.items():
    print(f"\n{model_name} Regression:")
    print(f"  R² Score: {score_dict['R2']:.4f}")
    print(f"  MSE: {score_dict['MSE']:.2f}")
    
    # Print coefficients for linear models
    if model_name == 'Linear':
        print("\n  Coefficients:")
        for feature, coef in zip(x.columns, linear_regressor.coef_):
            print(f"  {feature}: {coef:.2f}")
        print(f"  Intercept: {linear_regressor.intercept_:.2f}")

# Print prediction samples
print("\n" + "="*50)
print("PREDICTION SAMPLES")
print("="*50)
print("True Values vs Linear Regression Predictions:")
for i in range(3):
    print(f"  Sample {i+1}: True = {y_test.iloc[i]:.0f}, Predicted = {predictions['Linear'][i]:.0f}")

print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print("\nFeatures used:", list(x.columns))# 📚 Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 🧮 Creating a sample DataFrame with startup-related features
data = {
    'R&D Spend': [165349, 162597, 153441, 144372, 142107, 131876, 134615, 130298, 120542, 123335, 
                  101913, 100672, 93863.0, 91992.4],
    'Administration': [136897, 151377, 101145, 118671, 91391, 99814, 147198, 145530, 148719, 108679, 
                       118924, 91290.6, 127230, 110404],
    'Marketing Spend': [471784, 443898, 407934, 383199, 366168, 362861, 127716, 323876, 311613, 304032, 
                        223761, 249765, 248229, 250366],
    'State': ['New York', 'California', 'Florida', 'New York', 'Florida', 'New York', 'California', 
              'Florida', 'New York', 'California', 'Florida', 'California', 'Florida', 'California'],
    'Profit': [192261, 191792, 191050, 182902, 166187, 156991, 156122, 155752, 152212, 149760, 
               146122, 144259, 143266, 114809]
}
df = pd.DataFrame(data)

# 🎯 Separating features (X) and target (y)
x = df.drop('Profit', axis=1)
y = df['Profit']

# 🔄 Converting the categorical 'State' column into numerical dummy variables
states = pd.get_dummies(x['State'], drop_first=True)  # drop_first=True to avoid dummy variable trap
x = x.drop('State', axis=1)
x = pd.concat([x, states], axis=1)

# ✂️ Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ⚙️ Initializing different regression models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

# 📦 Dictionaries to store predictions and metrics for each model
predictions = {}
metrics = {}

# 🚀 Training and evaluating each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    predictions[name] = model.predict(X_test)  # Predict on test set
    metrics[name] = {
        'MSE': mean_squared_error(y_test, predictions[name]),
        'R2': model.score(X_test, y_test)  # R² Score = Explained Variance
    }

# 🧾 Fetch the linear regression model for coefficient analysis
linear_regressor = models['Linear']

# 📊 MODEL PERFORMANCE SUMMARY
print("="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

for model_name, score_dict in metrics.items():
    print(f"\n{model_name} Regression:")
    print(f"  R² Score: {score_dict['R2']:.4f}")
    print(f"  MSE: {score_dict['MSE']:.2f}")

    # 🧮 Print coefficients for Linear Regression
    if model_name == 'Linear':
        print("\n  Coefficients:")
        for feature, coef in zip(x.columns, linear_regressor.coef_):
            print(f"  {feature}: {coef:.2f}")
        print(f"  Intercept: {linear_regressor.intercept_:.2f}")

# 🎯 Prediction Samples
print("\n" + "="*50)
print("PREDICTION SAMPLES")
print("="*50)
print("True Values vs Linear Regression Predictions:")
for i in range(min(3, len(y_test))):
    print(f"  Sample {i+1}: True = {y_test.iloc[i]:.0f}, Predicted = {predictions['Linear'][i]:.0f}")

# 📌 Final summary of model and data used
print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print("\nFeatures used:", list(x.columns))

# ✅ FINAL NOTES:
"""
🧠 KEY CONCEPTS COVERED:
------------------------
- Dummy Encoding: Converts categorical values like 'State' into binary columns.
- Train-Test Split: Helps evaluate model performance on unseen data.
- Model Types:
  🔹 Linear Regression: Basic straight-line fit
  🔹 Ridge Regression: Linear + L2 regularization (to reduce overfitting)
  🔹 Lasso Regression: Linear + L1 regularization (also performs feature selection)
- R² Score: Measures how well the model explains the variation in the target
- MSE: Penalizes large errors; lower is better.

📈 The coefficient output tells you the impact of each feature on predicted Profit.
For example, a higher R&D Spend coefficient means it's strongly influencing profit.
"""
