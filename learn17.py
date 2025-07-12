import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("tips")

# Optional: Print the dataset
print(df)

# Compute correlation on numeric columns only
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation)

sns.barplot(x='sex',y ='total_bill' ,data=df)
plt.show()

sns.boxplot(x='sex',y='total_bill' ,data=df, palette='rainbow')
plt.show()
sns.boxplot(x='sex',y='total_bill' ,data=df, orient='v')
plt.show()
sns.violinplot(x='day',y='total_bill' ,data=df, palette='rainbow')

plt.savefig("voilin.png")

plt.show()