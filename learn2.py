import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

x = np.arange(0, 10)
y = np.arange(11, 21)

a = np.arange(40, 50)
b = np.arange(50, 60)

plt.scatter(x, y, c="b")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("graph in 2D")
plt.savefig("text.png")

plt.plot(x,y,"r--")
plt.savefig("text.png")
plt.subplot(2,2,1)
plt.plot(x,y,'g')
plt.show()
