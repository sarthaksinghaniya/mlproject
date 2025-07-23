import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load built-in Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Show the first few rows
print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Plot 1: Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='survived')
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# Plot 2: Survival Count by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sex', hue='survived')
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Plot 3: Age distribution by survival
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='survived', multiple='stack', kde=True)
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.show()

# Plot 4: Survival Rate by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='pclass', hue='survived')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Plot 5: Heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()



sns.displot(df['age'], kde=False, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()  

df['fare'].hist(color = 'green',  bins=40 , figsize=(8,4))
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=df , palette='winter')
plt.title("Fare Distribution by Class")
plt.show()

