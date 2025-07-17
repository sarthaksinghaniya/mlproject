import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [7, 8, 5, 6, 9, 10, 15, 14, 12, 11]

# Create box plot
sns.boxplot(data=data)
plt.title("Box Plot Example")
plt.show()


'''
 0%        25%         50%         75%       100%
 |---------|===========|===========|----------|
Min        Q1         Q2          Q3         Max



   Whisker   Box (Middle 50%)     Whisker
     |       |==========|            |
    Min      Q1    Q2   Q3          Max
           25%    50%   75%

'''

'''
Lower Whisker = Q1 - 1.5 × IQR

Upper Whisker = Q3 + 1.5 × IQR
'''


"""
Mid-line inside box	Median (Q2)
Bottom of box	Q1 (25th percentile)
Top of box	Q3 (75th percentile)
Box height	IQR = Q3 − Q1
Whiskers	Range within Q1−1.5×IQR to Q3+1.5×IQR
Dots outside	Outliers
X-axis	Categories (if any)
Y-axis	Data values (in vertical plot)
"""
