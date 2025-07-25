import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)
d_1 = np.random.normal(100, 10, 200)
d_2 = np.random.normal(90, 20, 200)
d_3 = np.random.normal(80, 30, 200)
d_4 = np.random.normal(70, 40, 200)
d = [d_1, d_2, d_3, d_4]

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)

bp = ax.boxplot(d, patch_artist = True,
                notch ='True', vert = 0)

colors = ['#0000FF', '#00FF00', 
          '#FFFF00', '#FF00FF']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")

# changing color and linewidth of
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)

for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)

# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)

ax.set_yticklabels(['d_1', 'd_2', 
                    'd_3', 'd_4'])
 
plt.title("Customized box plot")

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.savefig("bp_customized.png")
plt.show()

'''Box Plot Overview
A box plot, also called a whisker plot, visually summarizes data values, showing the minimum, first quartile, median, third quartile, and maximum.

Key Pointers:
Box plot includes a box from the first to the third quartile with a line for the median.
x-axis represents the data, while the y-axis shows frequency distribution.
Use matplotlib.pyplot.boxplot() to create box plots in Matplotlib.
Parameters include data, notch, vert, bootstrap, usermedians, positions, widths, patch_artist, labels, and meanline.
Data can be provided as a NumPy array, Python list, or tuple.
Example code generates random data using numpy.random.normal() and plots it.

Customization
Customizing the box plot involves setting specific attributes:

notch=True for a notched box plot.
patch_artist=True to fill the boxplot with color.
vert=False for a horizontal box plot.
Labels can be set based on data dimensions.
'''