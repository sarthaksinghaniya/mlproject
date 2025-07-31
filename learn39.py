# Import required libraries
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt     
import numpy as np
import statsmodels.api as sm

# ------------------------ PART 1: MULTIPLE LINEAR REGRESSION ON ADVERTISING DATA ------------------------

# Load advertising dataset from online source (TV, radio, newspaper ad budgets)
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df_adv = pd.read_csv(url)

# Display the first few rows of the dataset
print("Advertising Dataset Preview:")
print(df_adv.head())

# Separate independent features (TV, radio, newspaper) and dependent variable (sales)
x = df_adv[['TV', 'radio', 'newspaper']]  # Features
y = df_adv['sales']                       # Target (what we want to predict)

# Add a constant term to include the intercept in the regression equation
X = sm.add_constant(x)

# Build the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Show detailed summary of the regression results
print("\n--- Advertising Regression Summary ---")
print(model.summary())

# Calculate and display correlation matrix between features
corr_matrix = x.corr()
print("\n--- Feature Correlation Matrix (Advertising Data) ---")
print(corr_matrix)

# Optional: Show correlation heatmap for visual understanding
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix - Advertising")
plt.show()

# ------------------------ PART 2: SIMPLE LINEAR REGRESSION ON SALARY DATA ------------------------

# Load Salary vs Years of Experience dataset
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/Salary_Data.csv"
df_salary = pd.read_csv(url)

# Show column names and top rows
print("\nSalary Dataset Columns:")
print(df_salary.columns)

print("\nSalary Dataset Preview:")
print(df_salary.head())

# Define feature and target (only one feature: YearsExperience)
x = df_salary[['YearsExperience']]
y = df_salary['Salary']

# Add constant for intercept
X = sm.add_constant(x)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Display model summary
print("\n--- Salary Regression Summary ---")
print(model.summary())

# Show correlation between YearsExperience and Salary
print("\n--- Correlation Matrix (Salary Data) ---")
print(df_salary.corr())


# --------------------------------------------
# 📤 OUTPUT EXPLANATION (FOR BEGINNERS / "Ghada")
# --------------------------------------------

# ✅ PART 1: Advertising Dataset Regression
# - We built a multiple linear regression model to predict 'sales' based on ad spending on TV, radio, and newspaper.
# - 'TV' has a strong positive impact on sales.
# - 'radio' has moderate impact.
# - 'newspaper' has almost no significant impact (p-value is high, so it's not useful).
# - R-squared value (around 0.89) shows that the model explains 89% of the variation in sales — that's very good.
# - p-values:
#   --> p < 0.05 = feature is statistically significant.
#   --> p > 0.05 = not significant (you can ignore it).
# - The regression summary also gives intercept and coefficients, which we can use to make predictions like:
#   Sales = a + b1*TV + b2*radio + b3*newspaper

# ✅ PART 2: Salary Dataset Regression
# - This is a simple linear regression with only one feature: YearsExperience.
# - Very high R² (~0.95), means experience is a very strong predictor of salary.
# - Positive slope: more experience = higher salary.
# - Regression formula from output can be used to predict salary for any experience.

# ✅ Correlation Matrix Explanation
# - Shows how strongly two columns are related (range: -1 to +1).
#   --> +1 = strong positive relation
#   --> 0 = no relation
#   --> -1 = strong negative relation
# - Helps identify multicollinearity (if two features are highly correlated, one may be removed).

# 📌 Summary:
# - Use regression when you want to predict something numeric.
# - Use correlation to understand which features move together.
# - Always check R² and p-values to judge model quality.

