import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions (CORRECTED variable name - using underscore instead of hyphen)
y_pred = regressor.predict(X_test)  # Changed from y-pred to y_pred

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("\nModel Summary:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Sample predictions:", y_pred[:5])

