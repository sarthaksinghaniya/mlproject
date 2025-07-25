import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define a function that checks if a number is even or odd
def even_or_odd(num):
    if num % 2 == 0:
        return "the {} is even".format(num)  # If divisible by 2, it's even
    else:
        return "the {} is odd".format(num)   # Otherwise, it's odd

# Call the function with a single number and print the result
print(even_or_odd(10))  # Output: "the 10 is even"

# Create a list of numbers
lst = [1, 2, 3, 4, 5]

# Use map to apply the even_or_odd function to each number in the list
# map(function, iterable) returns a map object (like a generator)
result = list(map(even_or_odd, lst))  # Convert map object to list for printing

# Print the result list
print(result)
# Output: ['the 1 is odd', 'the 2 is even', 'the 3 is odd', 'the 4 is even', 'the 5 is odd']


