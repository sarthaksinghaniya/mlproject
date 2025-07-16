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
'''A Box Plot (or Whisker plot) display the summary of a data set, including minimum, first quartile, median, third quartile and maximum.
 it consists of a box from the first quartile to the third quartile, with a vertical line at the median.
 the x-axis denotes the data to be plotted while the y-axis shows the frequency distribution.
  he matplotlib.pyplot module of matplotlib library provides boxplot() function with the help of which we can create box plots.
 '''
# matplotlib.pyplot.boxplot(data)




sns.boxplot(x='sex',y='total_bill' ,data=df, palette='rainbow')
plt.show()
sns.boxplot(x='sex',y='total_bill' ,data=df, orient='v')
plt.show()
sns.violinplot(x='day',y='total_bill' ,data=df, palette='rainbow')

plt.savefig("voilin.png")

plt.show()