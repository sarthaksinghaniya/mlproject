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

# Initialize models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

# Train models and make predictions
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

# Calculate R2 score for Linear Regression (the 'score' variable you wanted)
score = r2_score(y_test, predictions['Linear'])

# Evaluate all models
metrics = {}
for name in models.keys():
    metrics[name] = {
        'MSE': mean_squared_error(y_test, predictions[name]),
        'R2': r2_score(y_test, predictions[name])
    }

# Print results
print("="*50)
print("Model Evaluation Metrics:")
print(f"\nLinear Regression R2 Score (score variable): {score:.4f}")

for model_name, scores in metrics.items():
    print(f"\n{model_name} Regression:")
    print(f"  MSE: {scores['MSE']:.2f}")
    print(f"  R2 Score: {scores['R2']:.4f}")

# Additional summary
print("\n" + "="*50)
print("Model Summary:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features used: {list(x.columns)}")
print("\nSample predictions (Linear Regression):")
for i, pred in enumerate(predictions['Linear'][:3]):
    print(f"  Test sample {i+1}: True = {y_test.iloc[i]:.0f}, Predicted = {pred:.0f}")
print("="*50)

