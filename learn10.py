import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
tips = sns.load_dataset("tips")

# Set theme
sns.set_theme(style="whitegrid")

# 1. Total bill vs Tip
plt.figure(figsize=(6,4))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.title("Total Bill vs Tip by Gender")
plt.show()

# 2. Average tip by day
plt.figure(figsize=(6,4))
sns.barplot(data=tips, x="day", y="tip", hue="sex")
plt.title("Average Tip by Day and Gender")
plt.show()

# 3. Tip % distribution (tip / total_bill * 100)
tips["tip_percent"] = tips["tip"] / tips["total_bill"] * 100

plt.figure(figsize=(6,4))
sns.histplot(tips["tip_percent"], kde=True, bins=20)
plt.title("Distribution of Tip Percentage")
plt.xlabel("Tip %")
plt.show()

# 4. Boxplot - Tips by Smoking Status
plt.figure(figsize=(6,4))
sns.boxplot(data=tips, x="smoker", y="tip", hue="sex")
plt.title("Tip by Smoker Status and Gender")
plt.show()

# 5. Heatmap - Correlation Matrix
plt.figure(figsize=(6,4))
sns.heatmap(tips.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
