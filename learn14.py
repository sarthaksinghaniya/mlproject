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
sns.jointplot(x='tip',y='total_bill',data=df,kind='hex')
plt.savefig("join.png")
plt.show()
