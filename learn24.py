# ------------------ IMPORTS ------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ------------------ LOAD DATA ------------------
df = sns.load_dataset('titanic')
print("Original Data:")
print(df.head())
print("Shape:", df.shape)

# ------------------ DROP USELESS OR HIGH-NULL COLUMNS ------------------
df.drop(columns=['deck'], inplace=True)  # too many NaNs
df.dropna(inplace=True)  # drop remaining rows with missing data

# ------------------ ENCODE CATEGORICAL VARIABLES ------------------
df = pd.get_dummies(df, drop_first=True)  # convert string cols to numbers

# ------------------ HEATMAP OF NUMERIC FEATURES ------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap (After Encoding)")
plt.show()

# ------------------ MODEL TRAINING ------------------
# Separate input features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(x_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


