#seaborn 
"""import seaborn as sns
df = sns.load_dataset("tips")
df.head
print(df)
# Compute correlation only on numeric columns
correlation = df.select_dtypes(include=['number']).corr()
print(correlation)
#sns.heatmap(df.corr())
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

'''
"""
# seaborn
import seaborn as sns
import matplotlib.pyplot as plt  # ← needed for plt.show()

# Load dataset
df = sns.load_dataset("tips")

# Print the full DataFrame (optional)
print(df)

# Compute correlation only on numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
print(correlation)

# Draw heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
