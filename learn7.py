#seaborn 
import seaborn as sns
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

