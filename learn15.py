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
sns.pairplot(df)
plt.savefig("pair.png")
sns.pairplot(df,hue="sex")
plt.savefig("hue.png")
sns.displot(df['tip'])
sns.displot(df['tip'],kde=False,bins=10)
plt.savefig("false.png")
sns.displot(df['tip'],kde=True,bins=10)
plt.savefig("dis.png")
plt.show()
