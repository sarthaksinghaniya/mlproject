import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

x = [2,8,20]
y = [11,14,10]

x2 = [12,8,20]
y2 = [11,4,10]

plt.bar(x,y,align = "center")
plt.bar(x2 ,y2 , color = 'g')
plt.title("bar graph")
plt.ylabel("company stock")
plt.xlabel("revenue")

plt.show()