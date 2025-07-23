# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Display first few rows of the dataset
print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------ PLOT 1: Boxplot of Age by Passenger Class ------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=df, palette='winter')
plt.title("Age Distribution by Passenger Class")  # ✅ Corrected title
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.show()

# ------------------ STEP: Impute Missing Age Values ------------------
# Define function to fill missing 'age' based on 'pclass'
def impute_age(column):
    age = column[0]
    pclass = column[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37  # Average age for 1st class
        elif pclass == 2:
            return 29  # Average age for 2nd class
        else:
            return 24  # Average age for 3rd class
    else:
        return age

# Apply the imputation function to update 'age'
df['age'] = df[['age', 'pclass']].apply(impute_age, axis=1)

# Display data after imputation
print(df.head())

# ------------------ PLOT 2: Survival Count ------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='survived')
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# ------------------ PLOT 3: Heatmap of Missing Data ------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap After Age Imputation")
plt.show()

# ------------------ STEP: Drop Irrelevant or High-Null Column ------------------
# Drop the 'deck' column due to high number of missing values
df.drop(columns=['deck'], inplace=True)

# Print confirmation
print("'deck' column dropped. Final columns:", df.columns.tolist())

# ------------------ PLOT 3: Heatmap of Missing Data ------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap After Age Imputation and Dropping 'deck'")
plt.show()


pd.get_dummies(df['embarked'], drop_first=True).head()
# ------------------ PLOT 4: Countplot of Embarked ------------------
plt.figure(figsize=(6, 4))  
sns.countplot(data=df, x='embarked')
plt.title("Count of Passengers by Embarked Port")
plt.xlabel("Embarked Port")
plt.ylabel("Number of Passengers")
plt.show()

sex = pd.get_dummies(df['sex'], drop_first=True)
embark = pd.get_dummies(df['embarked'], drop_first=True)

# ------------------ PLOT 5: Countplot  
plt.figure(figsize=(6, 4))
sns.countplot(data = df )
plt.show()


df = pd.concat([df , sex , embark], axis=1)
# Display the updated DataFrame with new columns
print(df.head())
# ------------------ PLOT 6: Pairplot of Numerical Features ------------------
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='survived', vars=['age', 'fare', 'pclass'], palette='rainbow')
plt.title("Pairplot of Numerical Features by Survival")
plt.show()

#building logistic regression model

df.drop('survived', axis=1).head()
# ------------------ PLOT 7: Correlation Heatmap ------------------ 
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Features")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Features and label
x_df = df.drop('survived', axis=1)
y_df = df['survived']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

# Train model
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = logmodel.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
