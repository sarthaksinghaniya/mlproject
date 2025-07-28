# 📌 Importing essential libraries for data handling and visualization
import pandas as pd                     # For data manipulation
import seaborn as sns                   # For data visualization (built on matplotlib)
import matplotlib.pyplot as plt         # For additional plotting features

# 📊 Load Titanic dataset (comes built-in with seaborn)
df = sns.load_dataset('titanic')

# 🔍 Display first few rows to get a sense of the data
print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------ 📈 PLOT 1: Boxplot of Age by Passenger Class ------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=df, palette='winter')  # Shows age distribution per class
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.show()


# ------------------ 🧼 STEP: Handle Missing Age Values ------------------
# Define a function to fill missing 'age' values based on passenger class (domain assumption)
def impute_age(column):
    age = column[0]
    pclass = column[1]
    # Fill missing age using average age per class
    if pd.isnull(age):
        if pclass == 1:
            return 37  # Mean age for 1st class
        elif pclass == 2:
            return 29  # Mean age for 2nd class
        else:
            return 24  # Mean age for 3rd class
    else:
        return age  # Return existing age if not null

# Apply the function to update the 'age' column
df['age'] = df[['age', 'pclass']].apply(impute_age, axis=1)

# ✅ Display updated data to confirm changes
print(df.head())

# ------------------ 📊 PLOT 2: Survival Count Plot ------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='survived')  # Count of people who survived (1) vs not (0)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# ------------------ 🔥 PLOT 3: Heatmap of Missing Values ------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')  # Visualize nulls
plt.title("Missing Data Heatmap After Age Imputation")
plt.show()

# ------------------ 🧹 STEP: Drop Irrelevant Column ------------------
# 'deck' column has too many missing values, so we drop it
df.drop(columns=['deck'], inplace=True)
print("'deck' column dropped. Final columns:", df.columns.tolist())

# 🔁 Re-check for any missing values again
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap After Dropping 'deck'")
plt.show()

# ------------------ 🧬 STEP: Convert Categorical Variables (Embarked) ------------------
# Convert 'embarked' column to dummy variables (One-Hot Encoding), skip first to avoid dummy trap
pd.get_dummies(df['embarked'], drop_first=True).head()

# 📊 PLOT 4: Countplot for Embarked Port
plt.figure(figsize=(6, 4))  
sns.countplot(data=df, x='embarked')
plt.title("Count of Passengers by Embarked Port")
plt.xlabel("Embarked Port")
plt.ylabel("Number of Passengers")
plt.show()

# ------------------ 🔁 STEP: Convert Other Categorical Variables ------------------
# Convert 'sex' and 'embarked' columns to dummy variables
sex = pd.get_dummies(df['sex'], drop_first=True)
embark = pd.get_dummies(df['embarked'], drop_first=True)

# 🧪 [Incorrect Plot - Just to Show It Runs] Optional test plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df)
plt.show()

# 🔗 Merge dummy variables into main DataFrame
df = pd.concat([df, sex, embark], axis=1)

# 👀 Show updated data with new binary columns (like 'male', 'Q', 'S')
print(df.head())

# ------------------ 📊 PLOT 6: Pairplot for Feature Relationships ------------------
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='survived', vars=['age', 'fare', 'pclass'], palette='rainbow')  # Visualize pair-wise relation by survival
plt.suptitle("Pairplot of Numerical Features by Survival", y=1.02)
plt.show()

# ------------------ 🧠 STEP: Build Logistic Regression Model (Concept) ------------------
"""
Logistic Regression:
- 🔍 A supervised learning algorithm used for binary classification (like survive or not).
- 🧠 It predicts probability using the logistic (sigmoid) function.
- 🧮 If output > 0.5, class = 1 (survived), else class = 0 (not survived)
- ✅ Commonly used for medical results, spam detection, and this Titanic dataset.
"""

# Drop 'survived' temporarily if we want to train separately
df.drop('survived', axis=1).head()

# ------------------ 🔍 PLOT 7: Correlation Heatmap ------------------ 
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)  # Correlation matrix
plt.title("Correlation Heatmap of Features")
plt.show()
# ------------------ 🧪 STEP: Train-Test Split and Model Training -----------------
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LogisticRegression  # For logistic regression model
# Separate features and target variable
X = df.drop('survived', axis=1)  # Features
y = df['survived']  # Target variable
# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ------------------ 🧪 STEP: Train Logistic Regression Model ------------------
model = LogisticRegression(max_iter=1000)  # Initialize model with higher iterations for convergence
model.fit(x_train, y_train)  # Train the model on training data 
# ------------------ 🔍 PLOT 8: Model Evaluation ------------------
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation metrics
# Predict on test set
y_pred = model.predict(x_test)  # Make predictions on the test set
# Calculate accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score
# Print detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Detailed report of precision, recall, f1-score
# ------------------ END OF SCRIPT ------------------
# ------------------ END OF SCRIPT ------------------
# ------------------ END OF SCRIPT ------------------
# ------------------ END OF SCRIPT ------------------           