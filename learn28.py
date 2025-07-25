import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

# Read dataset
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

# Filter data for the year 2007
df_2007 = df[df['year'] == 2007]        
print(df_2007.head())

# Print shape
print("Dataset shape:", df_2007.shape)

# Location of United States in 2007
df_loc = df_2007.loc[df_2007['country'] == 'United States'] 
print("Location of United States in 2007:", df_loc.values[0])

# Location of Afghanistan in 2007
df_locc = df_2007.loc[df_2007['country'] == 'Afghanistan']
print("Location of Afghanistan in 2007:", df_locc.values[0])

# Plot United States (2007)
plt.plot(df_loc['year'], np.zeros_like(df_loc['year']), 'o')
plt.title("Yearly Data for United States in 2007")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

# Plot Afghanistan (2007)
plt.plot(df_locc['year'], np.zeros_like(df_locc['year']),'o')
plt.title("Yearly Data for Afghanistan in 2007")
plt.xlabel("Year")          
plt.ylabel("Value")
plt.show()

#bivariate plot

sns.FacetGrid(df_2007, hue='continent', height=5).map(plt.scatter, 'gdpPercap', 'lifeExp').add_legend()

plt.title("GDP per Capita vs Life Expectancy (2007)")
plt.xlabel("GDP per Capita")
plt.ylabel("Life Expectancy")
#plt.xscale('log')  # Log scale for better visibility
#plt.grid(True)

plt.show()
plt.savefig('bivariate_plot.png')  # Save the plot as an image file

##multivariate plot

sns.pairplot(df_2007, hue='continent' , size = 2.5)
plt.title("Pairplot of GDP, Life Expectancy, and Population (2007)")
plt.savefig('multivariate_plot.png')  # Save the plot as an image file
plt.show()


sns.histplot(df['lifeExp'], bins=10, kde=True, color='skyblue')
plt.title("Life Expectancy Distribution")
plt.xlabel("Life Expectancy")
plt.ylabel("Count")
plt.show()