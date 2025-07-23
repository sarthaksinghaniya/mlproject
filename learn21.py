import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Display first 5 rows
print(df.head())

# Print shape
print("Dataset shape:", df.shape)
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
