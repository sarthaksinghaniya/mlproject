import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#list is iterable
lst =  [1, 2, 3, 4, 5]

for i in lst:
    print (i)


lst1 = iter(lst)  # Convert list to an iterator 

print(lst1)  # Output: <list_iterator object at ...>

print(next(lst1))  # Get the second element
# Output: 1


for i in lst1:  # Iterate through the iterator
    print(i)  

