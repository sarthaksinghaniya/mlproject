import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import random
data = [np.random.normal(0,std,100) for std in range (1,4)]
plt.boxplot(data,vert=True,patch_artist=True)
plt.show()
print(data)