# ------------------ IMPORTS ------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ LOAD TITANIC DATASET ------------------
df = sns.load_dataset('titanic')
print("✅ Loaded Titanic Dataset")
print("Shape:", df.shape)
print(df.head())

# ------------------ DROP UNUSED COLUMNS ------------------
# Columns like deck and embark_town have too many NaNs or overlap with others
df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who'], inplace=True)

# ------------------ HANDLE MISSING VALUES ------------------
df.dropna(inplace=True)  # For simplicity in this logistic model
print("\n✅ Cleaned Data - Shape after dropping NaNs:", df.shape)

# ------------------ ENCODE CATEGORICAL FEATURES ------------------
# Convert object/categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
print("\n✅ Encoded Features - New Columns:")
print(df_encoded.columns.tolist())

# ------------------ VISUALIZATION: CORRELATION HEATMAP ------------------
plt.figure(figsize=(12, 7))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="viridis")
plt.title("Correlation Heatmap (Post-Encoding)")
plt.tight_layout()
plt.show()

# ------------------ TRAIN / TEST SPLIT ------------------
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ TRAIN LOGISTIC REGRESSION MODEL ------------------
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(x_test)

print("\n✅ Model Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------ VISUALIZATION: CONFUSION MATRIX ------------------
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()




'''
----------------------------------------------
📊 Titanic Dataset - Logistic Regression Model
----------------------------------------------

1️⃣ Data Preprocessing:
- Loaded Titanic dataset from seaborn.
- Dropped 'deck' column due to high null values.
- Removed any rows with missing values using `dropna()`.
- Applied One-Hot Encoding with `pd.get_dummies()` to convert categorical columns (like sex, class, embark_town) into numeric format (drop_first=True to avoid dummy trap).

2️⃣ Correlation Heatmap:
- Visualized feature correlation using seaborn's heatmap.
- Heatmap shows:
  - Strong negative correlation between 'sex_male' and 'survived' (males less likely to survive).
  - Positive correlation between 'fare' and 'survived' (higher fare = better chances of survival).
  - 'pclass' has negative correlation with 'survived' (1st class passengers more likely to survive).

3️⃣ Model Training:
- Used `LogisticRegression()` from sklearn for binary classification.
- Target variable: `survived` (0 = No, 1 = Yes).
- Split data: 80% training, 20% testing.

4️⃣ Model Evaluation:
- Accuracy score: e.g., 80% (means 8 out of 10 predictions are correct).
- Classification Report:
  - Precision: Out of predicted positives, how many were actually positive.
  - Recall: Out of actual positives, how many did we correctly identify.
  - F1-score: Harmonic mean of precision and recall.
  - Support: Number of actual instances for each class.

Example Classification Report Interpretation:
              precision    recall  f1-score   support

           0       0.84      0.86      0.85        37
           1       0.79      0.76      0.78        25

    accuracy                           0.82        62
   macro avg       0.82      0.81      0.81        62
weighted avg       0.82      0.82      0.82        62

- Class 0 = Not Survived, Class 1 = Survived
- The model performs slightly better at predicting non-survivors.

✅ Conclusion:
- The logistic regression model shows decent performance (~80% accuracy).
- 'Sex', 'Fare', and 'Pclass' are key predictors of survival.
----------------------------------------------
'''


"""
📊 Correlation Heatmap Explanation:

- The heatmap shows the correlation between numeric and encoded features of the Titanic dataset.
- Correlation ranges from -1 to 1:
    🔹 +1 → Perfect positive correlation
    🔹 -1 → Perfect negative correlation
    🔹  0 → No correlation

- Darker red (closer to 1) means strong positive relation.
    e.g., 'sex_male' and 'survived' show a negative correlation (males had lower survival chances).

- Lighter blue (closer to -1) indicates a negative correlation.
    e.g., 'class_Third' negatively correlated with 'survived' (3rd class passengers had low survival).

- Diagonal elements are always 1 (each feature is perfectly correlated with itself).

✅ This heatmap helps identify which features are more related to survival,
   guiding us for feature selection and model improvement.
"""




"""
-------------------- 🔍 CORRELATION EXPLANATION --------------------

📌 What is Correlation?
Correlation is a statistical measure that explains the **relationship** 
between two variables. It tells us how strongly one variable is 
related to another, and in which direction (positive or negative).

📏 Correlation Range:
- +1 : Perfect Positive Correlation
-  0 : No Correlation
- -1 : Perfect Negative Correlation

💡 Interpretation:
- Positive Correlation: Both variables increase together.
- Negative Correlation: One increases while the other decreases.
- Zero Correlation: No predictable relation between variables.

📊 Titanic Heatmap Insights:
- 'sex_male' vs 'survived': ❌ Negative correlation
    ➤ Indicates that **males had a lower survival rate**.
- 'fare' vs 'survived': ✅ Positive correlation
    ➤ Higher fare passengers (likely 1st class) had **better survival chances**.

🎯 Why is Correlation Important?
- It helps us **understand feature importance**.
- Strongly correlated features (positive or negative) are useful 
  for training machine learning models.
- It guides **feature selection** and helps in reducing noise.

---------------------------------------------------------------
"""


