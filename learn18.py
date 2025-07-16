import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset("tips")

# Optional: Print the dataset
print(df)

# Compute correlation on numeric columns only
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation)
