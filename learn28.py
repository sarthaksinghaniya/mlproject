import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

# ---------------------- Load and Filter Data ---------------------- #

# Read dataset from URL
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

# Filter data for the year 2007
df_2007 = df[df['year'] == 2007]        
print(df_2007.head())

# Print shape of the 2007 data
print("Dataset shape:", df_2007.shape)

# Location details of United States in 2007
df_loc = df_2007.loc[df_2007['country'] == 'United States'] 
print("Location of United States in 2007:", df_loc.values[0])

# Location details of Afghanistan in 2007
df_locc = df_2007.loc[df_2007['country'] == 'Afghanistan']
print("Location of Afghanistan in 2007:", df_locc.values[0])

# ---------------------- Simple Point Plot ---------------------- #

# Plotting a single point for US in 2007 (x = year, y = 0 just to mark)
plt.plot(df_loc['year'], np.zeros_like(df_loc['year']), 'o')
plt.title("Yearly Data for United States in 2007")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

# Plotting a single point for Afghanistan in 2007
plt.plot(df_locc['year'], np.zeros_like(df_locc['year']), 'o')
plt.title("Yearly Data for Afghanistan in 2007")
plt.xlabel("Year")          
plt.ylabel("Value")
plt.show()

# ---------------------- Bivariate Plot ---------------------- #

# Scatter plot of GDP vs Life Expectancy with continent hue
sns.FacetGrid(df_2007, hue='continent', height=5).map(plt.scatter, 'gdpPercap', 'lifeExp').add_legend()

plt.title("GDP per Capita vs Life Expectancy (2007)")
plt.xlabel("GDP per Capita")
plt.ylabel("Life Expectancy")
#plt.xscale('log')  # You can use log scale if GDP values are very large
#plt.grid(True)
plt.show()

# Save the bivariate plot as image
plt.savefig('bivariate_plot.png')

# ---------------------- Multivariate Plot ---------------------- #

# Pairplot shows pairwise relationship between features
sns.pairplot(df_2007, hue='continent' , height = 2.5 )
plt.title("Pairplot of GDP, Life Expectancy, and Population (2007)")
plt.savefig('multivariate_plot.png')  # Save the plot as image
plt.show()

# ---------------------- Univariate Histogram ---------------------- #

# Histogram of Life Expectancy
sns.histplot(df['lifeExp'], bins=10, kde=True, color='skyblue')  # kde=True gives smooth curve
plt.title("Life Expectancy Distribution")
plt.xlabel("Life Expectancy")
plt.ylabel("Count")
plt.show()

# ---------------------- Output Summary Comment ---------------------- #

'''
📊 OUTPUT GRAPH SUMMARY:

1. 🔵 US & Afghanistan Point Plots:
   - Just single dots on x-axis (at year 2007), helps verify filtered data visually.
   - Y=0 used just for marking location of that year.

2. 🌍 Bivariate Plot (GDP vs LifeExp):
   - Each point = 1 country.
   - Shows rich countries (high GDP) tend to live longer (high Life Expectancy).
   - Color = continent → helpful to see regional patterns.
   - Can use log scale for better spacing.

3. 🔁 Multivariate Plot (Pairplot):
   - Compares all 3 variables: GDP, LifeExp, and Population.
   - Good for finding correlations & cluster groups continent-wise.

4. 📈 Histogram (Life Expectancy):
   - Shows how life expectancy is distributed across countries.
   - KDE line (smooth blue curve) helps visualize overall shape (normal or skewed).
   - Useful for univariate EDA → identify average, spread & outliers.
'''
