import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in "tips" dataset from Seaborn
# This dataset contains information about restaurant bills and tips
# including gender, smoking status, day, time, and more.
tips = sns.load_dataset("tips")

# Set a clean and readable white grid theme for all plots
sns.set_theme(style="whitegrid")

# 1. Scatterplot: Total bill vs Tip, color-coded by gender
plt.figure(figsize=(6,4))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.title("Total Bill vs Tip by Gender")  # Add plot title
plt.show()

# 2. Barplot: Average tip by day, separated by gender
plt.figure(figsize=(6,4))
sns.barplot(data=tips, x="day", y="tip", hue="sex")
plt.title("Average Tip by Day and Gender")  # Add plot title
plt.show()

# 3. Create a new column: tip percentage relative to total bill
tips["tip_percent"] = tips["tip"] / tips["total_bill"] * 100

# Plot a histogram of tip percentages with a KDE curve
plt.figure(figsize=(6,4))
sns.histplot(tips["tip_percent"], kde=True, bins=20)
plt.title("Distribution of Tip Percentage")  # Add plot title
plt.xlabel("Tip %")  # Label x-axis
plt.show()

# 4. Boxplot: Compare tip amounts between smokers and non-smokers by gender
plt.figure(figsize=(6,4))
sns.boxplot(data=tips, x="smoker", y="tip", hue="sex")
plt.title("Tip by Smoker Status and Gender")  # Add plot title
plt.show()

# 5. Heatmap: Show correlation between numerical features in the dataset
plt.figure(figsize=(6,4))
sns.heatmap(tips.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")  # Add plot title
plt.show()