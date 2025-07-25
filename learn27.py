import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Plot bar graph of survival count
plt.figure(figsize=(6, 4))  # Set the size of the plot (width, height)

# Countplot makes a bar for each category in 'survived' column (0 = Died, 1 = Survived)
sns.countplot(data=df, x='survived', palette='Set2')

# Title of the plot
plt.title("Survival Count")  # Title above the plot

# X-axis label: tells what 0 and 1 mean
plt.xlabel("Survived (0 = No, 1 = Yes)")  # Label below the x-axis

# Y-axis label: how many people in each category
plt.ylabel("Number of Passengers")  # Label on the y-axis

# Add count numbers on top of bars
for p in plt.gca().patches:
    # Get height of each bar (i.e., the count)
    height = p.get_height()
    # Place the text slightly above each bar
    plt.text(x=p.get_x() + p.get_width()/2, y=height + 2, s=int(height), 
             ha='center', fontsize=10, fontweight='bold')

# Show the plot
plt.show()
