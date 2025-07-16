#CATEGORICAL PLOT
# seaborn
'''import seaborn as sns
import matplotlib.pyplot as plt  # ← needed for plt.show()

# Load dataset
df = sns.load_dataset("tips")

# Print the full DataFrame (optional)
print(df)

# Compute correlation only on numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
sns.countplot('sex',data=df)
'''
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("tips")

# Optional: Print the dataset
print(df)

# Compute correlation on numeric columns only
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation)

# Plot a countplot for 'sex'
sns.countplot(x='day', data=df)
plt.show()

sns.countplot(y='sex', data=df)

# Show the plot
plt.show()


