import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt     
import numpy as np

# ---------------------- Load and Filter Data ---------------------- #  
# Read dataset from URL
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')      

# Filter data for the year 2000 - 2025
df_filtered = df[(df['year'] >= 2000) & (df['year'] <= 2025)]
print(df_filtered.head())
# Print shape of the filtered data
print("Dataset shape:", df_filtered.shape)
# Location details of United States in the filtered years   
df_us = df_filtered[df_filtered['country'] == 'United States']
print("Location of United States in the filtered years:", df_us[['year', 'gdpPercap', 'lifeExp']].values)
# Location details of Afghanistan in the filtered years
df_afghanistan = df_filtered[df_filtered['country'] == 'Afghanistan']
print("Location of Afghanistan in the filtered years:", df_afghanistan[['year', 'gdpPercap', 'lifeExp']].values)

# ---------------------- Simple Point Plot ---------------------- #
# Plotting a single point for US in the filtered years (x = year, y = 0 just to mark)
plt.figure(figsize=(10, 5)) 
plt.plot(df_us['year'], np.zeros_like(df_us['year']), 'o', label='United States')
plt.title("Yearly Data for United States (2000-2025)")
plt.xlabel("Year")      
plt.ylabel("Value")
plt.legend()
plt.show()  

# Plotting a single point for Afghanistan in the filtered years 
plt.figure(figsize=(10, 5))
plt.plot(df_afghanistan['year'], np.zeros_like(df_afghanistan['year']), 'o', label='Afghanistan')
plt.title("Yearly Data for Afghanistan (2000-2025)")
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.show()
# ---------------------- Bivariate Plot ---------------------- #
# Scatter plot of GDP vs Life Expectancy with continent hue
sns.FacetGrid(df_filtered, hue='continent', height=5).map(plt.scatter, 'gdpPercap', 'lifeExp').add_legend()
plt.title("GDP per Capita vs Life Expectancy (2000-2025)")
plt.xlabel("GDP per Capita")    
plt.ylabel("Life Expectancy")
plt.xscale('log')  # Using log scale for better visibility
plt.grid(True)
plt.show()
 
                                                                 
